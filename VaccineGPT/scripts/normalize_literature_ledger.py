from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
import gzip
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import normalize_protein

ROUNDS = {
    1: ("protein_foundation_models", "Protein-language-model and protein-embedding evidence for protein representation."),
    2: ("genome_dna_foundation_models", "DNA/genome language models and genome-window inference."),
    3: ("contrastive_multimodal", "Contrastive/multimodal learning for protein and biological modalities."),
    4: ("bacterial_vaccine", "Bacterial vaccine target/reverse vaccinology and protective antigen evidence."),
    5: ("functional_genomics", "CRISPRi/TnSeq/fitness and conditional gene essentiality."),
    6: ("bacterial_annotation", "Bacterial genome annotation and ORF boundary/provenance."),
    7: ("graph_biology", "Biological graph learning, graph attention, interaction/operon context."),
    8: ("uncertainty_calibration", "Conformal prediction/calibration/uncertainty under biological domain shift."),
    9: ("multiobjective_decision", "Pareto/multiobjective decisions in biomedical and vaccine prioritization."),
    10: ("data_quality_leakage", "Sequence deduplication, homology leakage, and benchmark splitting."),
}


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def work_key(row: dict[str, Any]) -> str:
    doi = str(row.get("doi") or "").strip().casefold()
    pmid = str(row.get("pmid") or "").strip()
    pmcid = str(row.get("pmcid") or "").strip().upper()
    if doi:
        return f"doi:{doi}"
    if pmid:
        return f"pmid:{pmid}"
    if pmcid:
        return f"pmcid:{pmcid}"
    return f"title:{normalize_title(str(row.get('title') or ''))}"


def xml_sections(path: Path) -> list[str]:
    with gzip.open(path, "rb") as stream:
        content = stream.read()
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return []
    return [
        " ".join("".join(item.itertext()).split())
        for item in root.findall(".//sec/title")
        if " ".join("".join(item.itertext()).split())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize retrieved Europe PMC full-text records into an auditable deduplicated ledger."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.campaign_root / "evidence_ledger.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_by_round: dict[int, dict[str, Any]] = {}
    for round_number in ROUNDS:
        raw_path = args.campaign_root / "raw" / f"round_{round_number:02d}_search_response.json"
        if not raw_path.exists():
            continue
        raw_bytes = raw_path.read_bytes()
        raw_payload = json.loads(raw_bytes)
        raw_by_round[round_number] = {
            row.get("pmcid"): row
            for row in raw_payload.get("resultList", {}).get("result", [])
            if row.get("pmcid")
        }
    deduplicated: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for source in records:
        key = work_key(source)
        raw = raw_by_round.get(source["round"], {}).get(source.get("pmcid"), {})
        fulltext = Path(source["fulltext_path"])
        if not fulltext.is_absolute():
            fulltext = (Path.cwd() / fulltext).resolve()
        sections = xml_sections(fulltext) if fulltext.exists() else list(
            source.get("fulltext_section_titles", [])
        )
        round_name, question = ROUNDS[source["round"]]
        normalized = {
            "work_id": key,
            "title": source.get("title") or "",
            "authors": source.get("authorString") or "未核实",
            "year": source.get("pubYear"),
            "venue": source.get("journalTitle"),
            "doi": source.get("doi"),
            "pmid": source.get("pmid"),
            "pmcid": source.get("pmcid"),
            "url": source.get("fulltext_xml_url"),
            "document_type": source.get("document_type", []),
            "version": {
                "is_preprint": raw.get("isPreprint"),
                "pub_year": source.get("pubYear"),
            },
            "license": raw.get("license"),
            "access_level": "full_text" if fulltext.exists() else "metadata_only",
            "full_text_access_state": "open_full_text" if fulltext.exists() else "missing_cached_xml",
            "query_source": "Europe PMC REST search and fullTextXML",
            "retrieved_at": source.get("retrieved_at_utc"),
            "request_url": source.get("query_url"),
            "raw_response_sha256": source.get("search_response_sha256"),
            "fulltext_sha256": source.get("fulltext_xml_sha256"),
            "research_question": question,
            "rounds_referenced": [source["round"]],
            "methods": "未逐篇人工提取；请在全文 Methods section 核查。",
            "data": "摘要级概述：" + str(source.get("abstractText") or "摘要缺失"),
            "baselines": "未逐篇人工提取。",
            "quantitative_results": "未逐篇人工提取；不可从摘要省略项推断。",
            "limitations": "未逐篇人工提取；通用迁移限制需结合目标菌验证。",
            "code_url": None,
            "code_status": "未逐篇核验",
            "evidence_locations": sections,
            "importance_band": (
                "近2年" if source.get("pubYear") and int(source["pubYear"]) >= 2025
                else "2–5年" if source.get("pubYear") and int(source["pubYear"]) >= 2021
                else "5年以上"
            ),
            "reliability_components": {
                "venue": source.get("journalTitle") or "未核实",
                "impact_factor": {"value": None, "status": "未核实"},
                "citations": {"value": None, "status": "未核实", "source": None},
                "citing_paper_quality": "未核实",
                "internal_validity": "title/abstract retrieval plus full-text XML fetch; manual claim review pending",
                "license": raw.get("license"),
            },
            "reliability_band": "limited_pending_manual_review",
            "claims": [
                {
                    "value": source.get("abstractText") or "摘要不可用",
                    "evidence_level": "摘要级证据",
                    "source_url": source.get("url") or source.get("fulltext_xml_url"),
                    "confidence": "low_until_section_verification",
                }
            ],
            "uncertainties": [
                "本条已下载开放全文，但实质主张仍未逐篇人工全文核验。",
                "不得将模型任务相近视为疫苗保护效果已验证。",
            ],
            "screening_decision": "include_fulltext_retrieved_manual_review_pending",
            "screening_reason": f"Round {source['round']} topic score {source.get('topic_score_title_abstract')}; OA fullTextXML retrieved.",
            "duplicate_merge_reason": [],
        }
        if key not in deduplicated:
            deduplicated[key] = normalized
        else:
            existing = deduplicated[key]
            existing["rounds_referenced"] = sorted(
                set(existing["rounds_referenced"] + [source["round"]])
            )
            if normalized["title"] != existing["title"] or (
                normalized["doi"] and existing["doi"] and normalized["doi"] != existing["doi"]
            ):
                conflicts.append(
                    {
                        "work_id": key,
                        "first_title": existing["title"],
                        "new_title": normalized["title"],
                        "first_doi": existing["doi"],
                        "new_doi": normalized["doi"],
                    }
                )
            existing["duplicate_merge_reason"].append(
                f"reused in iteration {source['round']}; identifier priority DOI>PMID>PMCID>normalized-title"
            )
    output_rows = list(deduplicated.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "vaccinegpt-normalized-literature-ledger-1.0",
        "input_rows": len(records),
        "unique_works": len(output_rows),
        "duplicate_merge_count": len(records) - len(output_rows),
        "metadata_conflicts": conflicts,
        "round_counts": {
            str(round_number): sum(
                round_number in row["rounds_referenced"] for row in output_rows
            )
            for round_number in ROUNDS
        },
        "full_text_xml_present": sum(row["access_level"] == "full_text" for row in output_rows),
        "manual_fulltext_claim_review_count": 0,
        "gbt7714_2025_status": "not_claimed",
        "output": str(args.output),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = args.output.with_name("normalized_literature_ledger_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
