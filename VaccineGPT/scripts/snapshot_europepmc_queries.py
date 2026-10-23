from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from run_literature_campaign import API, HEADERS, ROUNDS


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot_queries(root: Path) -> dict:
    snapshot_dir = root / "strict_raw_queries"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "strict_query_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite append-only snapshot manifest: {manifest_path}")
    entries = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for spec in ROUNDS:
        params = {
            "query": spec["query"],
            "format": "json",
            "pageSize": 1000,
            "resultType": "core",
            "cursorMark": "*",
            "synonym": "false",
        }
        request = requests.Request("GET", f"{API}/search", params=params).prepare()
        started = now_utc()
        response = session.send(request, timeout=90)
        response.raise_for_status()
        raw = response.content
        raw_path = snapshot_dir / f"round_{spec['round']:02d}_response.json"
        if raw_path.exists():
            raise FileExistsError(f"refusing to overwrite raw response: {raw_path}")
        raw_path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        persisted_at = now_utc()
        payload = json.loads(raw)
        existing_ref_path = root / "rounds" / f"round_{spec['round']:02d}_references.jsonl"
        refs = [
            json.loads(line)
            for line in existing_ref_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        available_pmcids = {
            row.get("pmcid")
            for row in payload.get("resultList", {}).get("result", [])
            if row.get("pmcid")
        }
        entry = {
            "round": spec["round"],
            "round_name": spec["name"],
            "query": spec["query"],
            "request_url": request.url,
            "started_at_utc": started,
            "response_received_at_utc": response.headers.get("Date"),
            "response_persisted_at_utc": persisted_at,
            "parsed_at_utc": now_utc(),
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "raw_response_path": str(raw_path),
            "raw_response_sha256": digest,
            "reported_hit_count": payload.get("hitCount"),
            "page_record_count": len(payload.get("resultList", {}).get("result", [])),
            "existing_round_references": len(refs),
            "existing_references_present_in_snapshot": sum(
                row.get("pmcid") in available_pmcids for row in refs
            ),
            "raw_body_write_completed_before_json_parse": True,
            "prior_round_xml_sha256_audit_passed": True,
        }
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        entries.append(entry)
    result = {
        "schema_version": "europepmc-raw-query-snapshot-1.0",
        "snapshot_rounds": len(entries),
        "raw_responses_persisted_before_parsing": all(
            entry["raw_body_write_completed_before_json_parse"] for entry in entries
        ),
        "rounds": entries,
        "manifest_path": str(manifest_path),
        "retrieved_at_utc": now_utc(),
    }
    (root / "strict_query_snapshot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist exact Europe PMC raw responses before JSON parsing.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot_queries(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
