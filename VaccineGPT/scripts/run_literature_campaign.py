from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from xml.etree import ElementTree


API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
HEADERS = {"User-Agent": "VaccineGPT-literature-audit/1.0 (academic research)"}

ROUNDS = [
    {
        "round": 1,
        "name": "protein_foundation_models",
        "question": "What do protein language models and protein embedding methods validate, and what do they not validate for vaccine targeting?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:"protein language model" OR TITLE_ABS:ESM-2 OR TITLE_ABS:ESM2 OR TITLE_ABS:ESM-3 OR TITLE_ABS:ESM3 OR TITLE_ABS:ProtTrans OR TITLE_ABS:protein-embedding)',
        "include": ["protein", "esm", "language", "embedding", "sequence", "fold"],
        "exclude": ["insulin receptor", "plant disease"],
    },
    {
        "round": 2,
        "name": "genome_dna_foundation_models",
        "question": "What is validated for DNA/genome-scale language models, genomic windows, and transfer to bacterial gene-level inference?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:"DNA language model" OR TITLE_ABS:"genome language model" OR TITLE_ABS:genome-modeling OR TITLE_ABS:genome-modelling OR TITLE_ABS:DNABERT OR TITLE_ABS:Evo2 OR TITLE_ABS:"nucleotide transformer" OR TITLE_ABS:genomic-foundation)',
        "include": ["dna", "genome", "nucleotide", "evo", "dnabert", "sequence"],
        "exclude": ["human language", "speech"],
    },
    {
        "round": 3,
        "name": "contrastive_multimodal_learning",
        "question": "Which contrastive and multimodal protein/biological learning designs have reproducible evidence and what are the task-transfer boundaries?",
        "query": 'OPEN_ACCESS:Y AND ((TITLE_ABS:contrastive AND TITLE_ABS:protein) OR (TITLE_ABS:multimodal AND TITLE_ABS:protein) OR TITLE_ABS:ProtST OR TITLE_ABS:CLEF OR (TITLE_ABS:multi-omics AND TITLE_ABS:contrastive))',
        "include": ["contrastive", "protein", "multimodal", "multi-modal", "clef", "protst"],
        "exclude": ["visual sentiment", "speech recognition"],
    },
    {
        "round": 4,
        "name": "bacterial_vaccine_reverse_vaccinology",
        "question": "Which computational vaccine-target prioritization signals have wet-lab protection evidence, and where do antigenicity and protection diverge?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:"reverse vaccinology" OR TITLE_ABS:"bacterial vaccine" OR TITLE_ABS:"protective antigen" OR TITLE_ABS:Vaxign OR TITLE_ABS:immunoinformatics)',
        "include": ["vaccine", "antigen", "reverse vaccinology", "protective", "immunogenic", "vaxign"],
        "exclude": ["plant vaccine", "veterinary economics"],
    },
    {
        "round": 5,
        "name": "bacterial_functional_genomics",
        "question": "How should CRISPRi, Tn-seq, fitness trajectories, and condition-specific essentiality be interpreted for attenuation rather than antigen accessibility?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:CRISPRi OR TITLE_ABS:Tn-seq OR TITLE_ABS:Transposon-seq OR TITLE_ABS:essentialome OR TITLE_ABS:essentiality) AND (TITLE_ABS:bacteria OR TITLE_ABS:bacterial OR TITLE_ABS:microbial OR TITLE_ABS:pathogen)',
        "include": ["crispri", "tn-seq", "transposon", "essential", "fitness", "bacteria"],
        "exclude": ["plant essentiality"],
    },
    {
        "round": 6,
        "name": "bacterial_genome_annotation",
        "question": "How do genome annotation pipelines handle ORF boundaries, pseudogenes, completeness, and versioned reference databases?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:"bacterial genome annotation" OR TITLE_ABS:Prokka OR TITLE_ABS:Bakta OR TITLE_ABS:PGAP OR TITLE_ABS:"genome annotation" OR TITLE_ABS:gene-calling)',
        "include": ["annotation", "bacterial", "prokka", "bakta", "pgap", "genome", "orf"],
        "exclude": ["clinical annotation only"],
    },
    {
        "round": 7,
        "name": "graph_learning_biology",
        "question": "What graph neural-network and attention methods are validated in biology, and what evidence is required before interpreting graph scores biologically?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:"graph neural network" OR TITLE_ABS:GATv2 OR TITLE_ABS:"graph attention" OR TITLE_ABS:graph-learning) AND (TITLE_ABS:protein OR TITLE_ABS:biological OR TITLE_ABS:genome OR TITLE_ABS:interaction OR TITLE_ABS:gene)',
        "include": ["graph", "protein", "biological", "genome", "gene", "interaction", "attention"],
        "exclude": ["social network"],
    },
    {
        "round": 8,
        "name": "uncertainty_calibration",
        "question": "What calibration/conformal guarantees apply to biological prediction under family-level dependence and domain shift?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:conformal OR TITLE_ABS:calibration OR TITLE_ABS:uncertainty-quantification OR TITLE_ABS:prediction-interval) AND (TITLE_ABS:protein OR TITLE_ABS:bioinformatics OR TITLE_ABS:biological OR TITLE_ABS:genomics OR TITLE_ABS:sequence)',
        "include": ["conformal", "calibration", "uncertainty", "protein", "bioinformatics", "genomic", "sequence"],
        "exclude": ["weather forecast", "financial market"],
    },
    {
        "round": 9,
        "name": "multiobjective_decision",
        "question": "How should Pareto/multiobjective decision methods expose trade-offs without substituting arbitrary weighted scores for biological judgment?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:Pareto OR TITLE_ABS:multiobjective OR TITLE_ABS:"multi-objective" OR TITLE_ABS:optimization) AND (TITLE_ABS:biomedical OR TITLE_ABS:biological OR TITLE_ABS:vaccine OR TITLE_ABS:protein OR TITLE_ABS:antigen)',
        "include": ["pareto", "multiobjective", "multi-objective", "optimization", "vaccine", "protein", "biological"],
        "exclude": ["portfolio only"],
    },
    {
        "round": 10,
        "name": "data_quality_homology_leakage",
        "question": "How do exact duplicates, homologs, identifier ambiguity, label conflicts, and dataset leakage bias protein/genome benchmarks?",
        "query": 'OPEN_ACCESS:Y AND (TITLE_ABS:homology OR TITLE_ABS:redundancy OR TITLE_ABS:deduplication OR TITLE_ABS:leakage OR TITLE_ABS:"data leakage" OR TITLE_ABS:benchmark) AND (TITLE_ABS:protein OR TITLE_ABS:genome OR TITLE_ABS:biological OR TITLE_ABS:sequence)',
        "include": ["homology", "redundancy", "duplicate", "leakage", "benchmark", "protein", "genome", "sequence"],
        "exclude": ["data leakage electrical"],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def topic_score(record: dict[str, Any], spec: dict[str, Any]) -> int:
    text = normalize_text(
        f"{record.get('title', '')} {record.get('abstractText', '')}"
    ).casefold()
    score = sum(text.count(token.casefold()) for token in spec["include"])
    score -= 3 * sum(text.count(token.casefold()) for token in spec["exclude"])
    return score


def request_search(
    session: requests.Session, query: str, cursor: str = "*"
) -> tuple[requests.Response, dict[str, Any]]:
    params = {
        "query": query,
        "format": "json",
        "pageSize": 1000,
        "resultType": "core",
        "cursorMark": cursor,
        "synonym": "false",
    }
    response = session.get(f"{API}/search", params=params, timeout=90)
    response.raise_for_status()
    return response, response.json()


def pmc_fulltext(
    session: requests.Session, pmcid: str
) -> tuple[bytes | None, str | None, str]:
    url = f"{API}/{pmcid}/fullTextXML"
    response = session.get(url, timeout=90)
    if response.status_code != 200:
        return None, response.url, f"http_{response.status_code}"
    content = response.content
    if not content.lstrip().startswith((b"<?xml", b"<!DOCTYPE", b"<article")):
        return None, response.url, "not_xml"
    return content, response.url, "full_text_xml"


def extract_sections(xml_bytes: bytes) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return []
    output = []
    for node in root.findall(".//sec/title"):
        title = normalize_text("".join(node.itertext()))
        if title:
            output.append(title)
    return output


def run_campaign(root: Path, target_per_round: int = 200, pause_seconds: float = 0.15) -> dict[str, Any]:
    raw_dir = root / "raw"
    fulltext_dir = root / "full_text_xml_gz"
    round_dir = root / "rounds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fulltext_dir.mkdir(parents=True, exist_ok=True)
    round_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    fulltext_cache: dict[str, dict[str, Any]] = {}
    round_summaries = []
    manifest_path = root / "query_manifest.jsonl"
    screening_path = root / "screening_log.jsonl"
    ledger_path = root / "evidence_ledger.jsonl"
    for path in (manifest_path, screening_path, ledger_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite append-only campaign file: {path}")
        path.touch()

    for spec in ROUNDS:
        started = utc_now()
        response, payload = request_search(session, spec["query"])
        raw_body = response.content
        raw_path = raw_dir / f"round_{spec['round']:02d}_search_response.json"
        raw_path.write_bytes(raw_body)
        raw_sha = hashlib.sha256(raw_body).hexdigest()
        records = payload.get("resultList", {}).get("result", [])
        screened = []
        for record in records:
            pmcid = record.get("pmcid")
            open_access = record.get("isOpenAccess") == "Y"
            score = topic_score(record, spec)
            eligible = bool(pmcid and open_access and score > 0)
            screen_row = {
                "round": spec["round"],
                "pmcid": pmcid,
                "doi": record.get("doi"),
                "title": record.get("title"),
                "open_access": open_access,
                "has_full_text_xml_identifier": bool(pmcid),
                "topic_score_title_abstract": score,
                "screening_decision": "fulltext_fetch_candidate" if eligible else "exclude",
                "exclusion_reason": None if eligible else (
                    "no_pmcid" if not pmcid else "not_marked_open_access" if not open_access else "topic_keywords_not_matched"
                ),
            }
            with screening_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(screen_row, ensure_ascii=False) + "\n")
            if eligible:
                screened.append((score, record))
        screened.sort(
            key=lambda item: (
                -item[0],
                -(int(item[1].get("firstPublicationDate")[:4]) if item[1].get("firstPublicationDate", "")[:4].isdigit() else 0),
            )
        )
        selected_rows = []
        failures = []
        for score, record in screened:
            if len(selected_rows) >= target_per_round:
                break
            pmcid = record["pmcid"]
            if pmcid not in fulltext_cache:
                body, fulltext_url, status = pmc_fulltext(session, pmcid)
                if body is None:
                    failures.append({"pmcid": pmcid, "status": status, "url": fulltext_url})
                    continue
                compressed_path = fulltext_dir / f"{pmcid}.xml.gz"
                with gzip.open(compressed_path, "wb", compresslevel=6) as stream:
                    stream.write(body)
                fulltext_cache[pmcid] = {
                    "path": str(compressed_path),
                    "xml_sha256": hashlib.sha256(body).hexdigest(),
                    "compressed_bytes": compressed_path.stat().st_size,
                    "xml_bytes": len(body),
                    "url": fulltext_url,
                    "retrieved_at": utc_now(),
                    "sections": extract_sections(body),
                }
                time.sleep(pause_seconds)
            cached = fulltext_cache[pmcid]
            row = {
                "round": spec["round"],
                "round_name": spec["name"],
                "pmcid": pmcid,
                "pmid": record.get("pmid"),
                "doi": record.get("doi"),
                "title": record.get("title"),
                "authorString": record.get("authorString"),
                "journalTitle": record.get("journalTitle"),
                "pubYear": record.get("pubYear"),
                "firstPublicationDate": record.get("firstPublicationDate"),
                "abstractText": record.get("abstractText"),
                "document_type": record.get("pubTypeList", {}).get("pubType", []),
                "access_level": "open_full_text",
                "query_url": response.url,
                "search_response_sha256": raw_sha,
                "fulltext_xml_url": cached["url"],
                "fulltext_xml_sha256": cached["xml_sha256"],
                "fulltext_path": cached["path"],
                "fulltext_xml_bytes": cached["xml_bytes"],
                "fulltext_section_titles": cached["sections"],
                "retrieved_at_utc": cached["retrieved_at"],
                "topic_score_title_abstract": score,
                "claim_extraction_status": "automated_metadata_and_section_index_only; manual_fulltext_review_not_claimed",
                "manual_review_status": "pending",
                "version_relationship": record.get("pubYear"),
            }
            selected_rows.append(row)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(pause_seconds)

        selected_path = round_dir / f"round_{spec['round']:02d}_references.jsonl"
        with selected_path.open("w", encoding="utf-8") as stream:
            for row in selected_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "round": spec["round"],
            "round_name": spec["name"],
            "question": spec["question"],
            "query": spec["query"],
            "request_url": response.url,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "http_status": response.status_code,
            "raw_response_path": str(raw_path),
            "raw_response_sha256": raw_sha,
            "reported_search_hit_count": payload.get("hitCount"),
            "first_page_records": len(records),
            "topic_and_open_access_candidates": len(screened),
            "full_text_xml_verified_count": len(selected_rows),
            "quota_target": target_per_round,
            "quota_met_by_retrieval": len(selected_rows) >= target_per_round,
            "full_text_fetch_failures": failures,
            "round_reference_file": str(selected_path),
            "unique_full_text_works_in_campaign_so_far": len(fulltext_cache),
            "claim_extraction_status": "not_a_manual_200-paper-synthesis",
        }
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        round_summaries.append(manifest)
        print(
            json.dumps(
                {
                    "round": spec["round"],
                    "name": spec["name"],
                    "fulltext_verified": len(selected_rows),
                    "quota_met": manifest["quota_met_by_retrieval"],
                    "unique_downloaded": len(fulltext_cache),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    campaign = {
        "schema_version": "vaccinegpt-literature-campaign-1.0",
        "started_at_utc": round_summaries[0]["started_at_utc"] if round_summaries else None,
        "completed_at_utc": utc_now(),
        "rounds": round_summaries,
        "requested_fulltext_references": target_per_round * len(ROUNDS),
        "retrieved_fulltext_references": sum(row["full_text_xml_verified_count"] for row in round_summaries),
        "unique_fulltext_works_downloaded": len(fulltext_cache),
        "all_round_retrieval_quotas_met": all(row["quota_met_by_retrieval"] for row in round_summaries),
        "manual_content_review_count": 0,
        "important_boundary": "OA full-text retrieval is not equivalent to manual verification of all substantive claims.",
    }
    (root / "campaign_summary.json").write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve auditable Europe PMC OA full-text sets for ten predefined research iterations."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-round", type=int, default=200)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()
    print(
        json.dumps(
            run_campaign(args.output_root, args.per_round, args.pause_seconds),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
