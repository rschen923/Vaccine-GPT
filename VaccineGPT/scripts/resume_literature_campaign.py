from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from run_literature_campaign import API, HEADERS, ROUNDS, extract_sections, topic_score


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fetch_xml(pmcid: str, attempts: int = 3) -> dict[str, Any]:
    url = f"{API}/{pmcid}/fullTextXML"
    last_error = "not_attempted"
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=HEADERS, timeout=90)
            if response.status_code == 200 and response.content.lstrip().startswith(
                (b"<?xml", b"<!DOCTYPE", b"<article")
            ):
                return {
                    "status": "full_text_xml",
                    "url": response.url,
                    "body": response.content,
                }
            last_error = f"http_{response.status_code}"
        except requests.RequestException as error:
            last_error = f"request_error:{type(error).__name__}:{error}"
        time.sleep(0.5 * (2**attempt))
    return {"status": last_error, "url": url, "body": None}


def retrieve_candidates(
    candidates: list[dict[str, Any]],
    existing_ids: set[str],
    cached: dict[str, dict[str, Any]],
    fulltext_dir: Path,
    target: int,
    workers: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    needed = [row for row in candidates if row.get("pmcid") and row["pmcid"] not in existing_ids]
    selected_ids: set[str] = set(existing_ids)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []

    for row in candidates:
        pmcid = row.get("pmcid")
        if not pmcid or pmcid in selected_ids:
            continue
        if pmcid in cached:
            selected.append((row, cached[pmcid]))
            selected_ids.add(pmcid)
            if len(selected) >= target:
                return selected, failures

    remaining = [row for row in needed if row["pmcid"] not in selected_ids]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cursor = 0
        while cursor < len(remaining) and len(selected) < target:
            chunk_size = min(
                len(remaining) - cursor,
                max(workers * 2, min(target - len(selected) + 10, 32)),
            )
            batch = remaining[cursor : cursor + chunk_size]
            cursor += chunk_size
            futures = {pool.submit(fetch_xml, row["pmcid"]): row for row in batch}
            fetched: dict[str, dict[str, Any]] = {}
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                pmcid = row["pmcid"]
                if result["status"] != "full_text_xml":
                    failures.append(
                        {"pmcid": pmcid, "status": result["status"], "url": result["url"]}
                    )
                    continue
                body = result["body"]
                path = fulltext_dir / f"{pmcid}.xml.gz"
                if not path.exists():
                    with gzip.open(path, "wb", compresslevel=6) as stream:
                        stream.write(body)
                fetched[pmcid] = {
                    "path": str(path),
                    "xml_sha256": hashlib.sha256(body).hexdigest(),
                    "compressed_bytes": path.stat().st_size,
                    "xml_bytes": len(body),
                    "url": result["url"],
                    "retrieved_at": utc_now(),
                    "sections": extract_sections(body),
                }
            for row in batch:
                pmcid = row["pmcid"]
                metadata = fetched.get(pmcid) or cached.get(pmcid)
                if metadata is None or pmcid in selected_ids:
                    continue
                selected.append((row, metadata))
                selected_ids.add(pmcid)
                cached[pmcid] = metadata
                if len(selected) >= target:
                    break
    return selected, failures


def resume(root: Path, target_per_round: int = 200, workers: int = 6) -> dict[str, Any]:
    raw_dir = root / "raw"
    fulltext_dir = root / "full_text_xml_gz"
    round_dir = root / "rounds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fulltext_dir.mkdir(parents=True, exist_ok=True)
    round_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "query_manifest.jsonl"
    screening_path = root / "screening_log.jsonl"
    ledger_path = root / "evidence_ledger.jsonl"
    for path in (manifest_path, screening_path, ledger_path):
        path.touch(exist_ok=True)
    manifest_rows = read_jsonl(manifest_path)
    completed = {
        row["round"]
        for row in manifest_rows
        if row.get("full_text_xml_verified_count", 0) >= target_per_round
    }
    ledger_rows = read_jsonl(ledger_path)
    cached: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        pmcid = row.get("pmcid")
        if not pmcid or pmcid in cached:
            continue
        cached[pmcid] = {
            "path": row["fulltext_path"],
            "xml_sha256": row["fulltext_xml_sha256"],
            "compressed_bytes": Path(row["fulltext_path"]).stat().st_size
            if Path(row["fulltext_path"]).exists()
            else 0,
            "xml_bytes": row["fulltext_xml_bytes"],
            "url": row["fulltext_xml_url"],
            "retrieved_at": row["retrieved_at_utc"],
            "sections": row.get("fulltext_section_titles", []),
        }
    for path in fulltext_dir.glob("PMC*.xml.gz"):
        if path.stem in cached:
            continue
        with gzip.open(path, "rb") as stream:
            body = stream.read()
        from xml.etree import ElementTree

        sections = []
        try:
            xml_root = ElementTree.fromstring(body)
            sections = [
                " ".join("".join(node.itertext()).split())
                for node in xml_root.findall(".//sec/title")
            ]
        except ElementTree.ParseError:
            pass
        cached[path.stem] = {
            "path": str(path),
            "xml_sha256": hashlib.sha256(body).hexdigest(),
            "compressed_bytes": path.stat().st_size,
            "xml_bytes": len(body),
            "url": f"{API}/{path.stem}/fullTextXML",
            "retrieved_at": utc_now(),
            "sections": sections,
        }

    known_screening = {
        (row.get("round"), row.get("pmcid"))
        for row in read_jsonl(screening_path)
    }
    completed_summaries = []
    for spec in ROUNDS:
        if spec["round"] in completed:
            current = next(row for row in manifest_rows if row["round"] == spec["round"])
            completed_summaries.append(current)
            continue
        raw_path = raw_dir / f"round_{spec['round']:02d}_search_response.json"
        if raw_path.exists():
            raw_body = raw_path.read_bytes()
            payload = json.loads(raw_body)
            query_url = None
        else:
            response = requests.get(
                f"{API}/search",
                params={
                    "query": spec["query"],
                    "format": "json",
                    "pageSize": 1000,
                    "resultType": "core",
                    "cursorMark": "*",
                    "synonym": "false",
                },
                headers=HEADERS,
                timeout=90,
            )
            response.raise_for_status()
            raw_body = response.content
            payload = response.json()
            query_url = response.url
            raw_path.write_bytes(raw_body)
        if query_url is None:
            params = {
                "query": spec["query"],
                "format": "json",
                "pageSize": 1000,
                "resultType": "core",
                "cursorMark": "*",
                "synonym": "false",
            }
            query_url = requests.Request(
                "GET", f"{API}/search", params=params
            ).prepare().url
        raw_sha = hashlib.sha256(raw_body).hexdigest()
        records = payload.get("resultList", {}).get("result", [])
        candidates = []
        for record in records:
            pmcid = record.get("pmcid")
            score = topic_score(record, spec)
            open_access = record.get("isOpenAccess") == "Y"
            eligible = bool(pmcid and open_access and score > 0)
            key = (spec["round"], pmcid)
            if key not in known_screening:
                screen = {
                    "round": spec["round"],
                    "pmcid": pmcid,
                    "doi": record.get("doi"),
                    "title": record.get("title"),
                    "open_access": open_access,
                    "topic_score_title_abstract": score,
                    "screening_decision": "fulltext_fetch_candidate" if eligible else "exclude",
                    "exclusion_reason": None if eligible else (
                        "no_pmcid" if not pmcid else "not_open_access" if not open_access else "topic_keywords_not_matched"
                    ),
                }
                with screening_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(screen, ensure_ascii=False) + "\n")
                known_screening.add(key)
            if eligible:
                candidates.append((score, record))
        candidates.sort(
            key=lambda item: (
                -item[0],
                -(
                    int(item[1].get("firstPublicationDate", "")[:4])
                    if item[1].get("firstPublicationDate", "")[:4].isdigit()
                    else 0
                ),
            )
        )
        current_round_rows = [
            row for row in ledger_rows if row.get("round") == spec["round"]
        ]
        existing_ids = {row["pmcid"] for row in current_round_rows}
        remaining = max(0, target_per_round - len(existing_ids))
        selected, failures = retrieve_candidates(
            [record for _, record in candidates],
            existing_ids,
            cached,
            fulltext_dir,
            remaining,
            workers,
        )
        references = list(current_round_rows)
        for score, record in candidates:
            if record.get("pmcid") in existing_ids:
                continue
            metadata = cached.get(record.get("pmcid"))
            if metadata is None:
                for candidate_record, candidate_metadata in selected:
                    if candidate_record.get("pmcid") == record.get("pmcid"):
                        metadata = candidate_metadata
                        break
            if metadata is None:
                continue
            row = {
                "round": spec["round"],
                "round_name": spec["name"],
                "pmcid": record["pmcid"],
                "pmid": record.get("pmid"),
                "doi": record.get("doi"),
                "title": record.get("title"),
                "authorString": record.get("authorString"),
                "journalTitle": record.get("journalTitle"),
                "pubYear": record.get("pubYear"),
                "firstPublicationDate": record.get("firstPublicationDate"),
                "abstractText": record.get("abstractText"),
                "document_type": record.get("pubTypeList", {}).get("pubType", []),
                "license": record.get("license"),
                "is_open_access": record.get("isOpenAccess"),
                "full_text_url_list": record.get("fullTextUrlList", {}),
                "is_preprint": record.get("isPreprint"),
                "access_level": "open_full_text",
                "query_url": query_url,
                "search_response_sha256": raw_sha,
                "fulltext_xml_url": metadata["url"],
                "fulltext_xml_sha256": metadata["xml_sha256"],
                "fulltext_path": metadata["path"],
                "fulltext_xml_bytes": metadata["xml_bytes"],
                "fulltext_section_titles": metadata["sections"],
                "retrieved_at_utc": metadata["retrieved_at"],
                "topic_score_title_abstract": score,
                "claim_extraction_status": "metadata_and_fulltext_section_index; substantive_manual_review_not_claimed",
                "manual_review_status": "pending",
            }
            references.append(row)
            ledger_rows.append(row)
            existing_ids.add(record["pmcid"])
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        reference_path = round_dir / f"round_{spec['round']:02d}_references.jsonl"
        with reference_path.open("w", encoding="utf-8") as stream:
            for row in references:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "round": spec["round"],
            "round_name": spec["name"],
            "question": spec["question"],
            "query": spec["query"],
            "request_url": query_url,
            "completed_at_utc": utc_now(),
            "raw_response_path": str(raw_path),
            "raw_response_sha256": raw_sha,
            "reported_search_hit_count": payload.get("hitCount"),
            "first_page_records": len(records),
            "open_access_topic_candidates": len(candidates),
            "full_text_xml_verified_count": len(references),
            "quota_target": target_per_round,
            "quota_met_by_retrieval": len(references) >= target_per_round,
            "fetch_failures": failures,
            "round_reference_file": str(reference_path),
            "manual_content_review_count": 0,
            "claim_extraction_status": "not_a_manual_200-paper-synthesis",
        }
        manifest_rows.append(summary)
        manifest_rows.sort(key=lambda row: row["round"])
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        completed_summaries.append(summary)
        print(
            json.dumps(
                {
                    "round": spec["round"],
                    "name": spec["name"],
                    "verified": len(references),
                    "quota_met": summary["quota_met_by_retrieval"],
                    "unique_downloaded": len(cached) + sum(
                        1 for pmcid in existing_ids if pmcid not in cached
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    campaign = {
        "schema_version": "vaccinegpt-literature-campaign-1.1",
        "completed_at_utc": utc_now(),
        "rounds": completed_summaries,
        "requested_round_references": target_per_round * len(ROUNDS),
        "retrieved_round_references": sum(
            row.get("full_text_xml_verified_count", 0) for row in completed_summaries
        ),
        "round_retrieval_quotas_met": all(
            row.get("quota_met_by_retrieval", False) for row in completed_summaries
        ),
        "manual_fulltext_claim_review_count": 0,
        "interpretation": "OA full-text retrieval was verified; retrieval alone is not manual review or substantive citation validation.",
    }
    (root / "campaign_summary.json").write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume the ten-round OA full-text campaign.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--per-round", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(resume(args.root, args.per_round, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
