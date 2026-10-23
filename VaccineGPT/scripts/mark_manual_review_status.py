from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(
    campaign_root: Path,
    normalized_ledger: Path,
    iteration_reports: Path,
) -> dict:
    sample_path = campaign_root / "manual_verification_sample.jsonl"
    audit_path = campaign_root / "campaign_audit.json"
    summary_path = campaign_root / "normalized_literature_ledger_summary.json"
    sample = read_jsonl(sample_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ledger = read_jsonl(normalized_ledger)
    by_pmcid = {str(row.get("pmcid")): row for row in ledger if row.get("pmcid")}
    counts = Counter()
    for item in sample:
        round_no = int(item["round"])
        blocked = iteration_reports / f"iteration_{round_no:02d}_manual_review_blocked.md"
        reviewed = iteration_reports / f"iteration_{round_no:02d}_manual_review.md"
        pmcid = str(item.get("pmcid"))
        if blocked.exists():
            status = "blocked_by_safety_filter"
            item["manual_review_status"] = "review_blocked_by_biological_risk_filter; not retried"
            item["review_method"] = "AI-assisted subagent task blocked by safety filter"
            counts["blocked"] += 1
        elif reviewed.exists():
            status = "selected_passages_reviewed"
            item["manual_review_status"] = (
                "AI_assisted_selected_fulltext_passages_reviewed; see "
                f"reference/iterations_261007/iteration_{round_no:02d}_manual_review.md"
            )
            item["review_method"] = "AI-assisted subagent read selected abstract/Methods/Results/Discussion excerpts"
            counts["reviewed"] += 1
        else:
            status = "pending"
            item["manual_review_status"] = "selected_for_manual_review_not_yet_performed"
            item["review_method"] = "not reviewed"
            counts["pending"] += 1
        item["review_status_code"] = status
        row = by_pmcid.get(pmcid)
        if row is None:
            counts["ledger_miss"] += 1
            continue
        row["manual_review_status"] = status
        row["review_method"] = item.get("review_method")
        row["manual_review_evidence_file"] = (
            f"reference/iterations_261007/iteration_{round_no:02d}_manual_review"
            + ("_blocked" if status == "blocked_by_safety_filter" else "")
            + ".md"
            if status != "pending"
            else None
        )
        if status == "selected_passages_reviewed":
            row["claim_extraction_status"] = (
                "AI-assisted sampled abstract and selected fulltext sections reviewed; "
                "not_complete_article_claim_extraction"
            )
            row["reliability_components"]["internal_validity"] = (
                "random sample; abstract and selected Methods/Results/Discussion excerpts reviewed; "
                "complete article and all claims not exhaustively re-extracted"
            )
        elif status == "blocked_by_safety_filter":
            row["claim_extraction_status"] = "manual_review_blocked_by_safety_filter"
            row["uncertainties"].append(
                "Random full-text manual review was blocked by the system biological-risk filter and was not rerouted."
            )
    normalized_ledger.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger),
        encoding="utf-8",
    )
    sample_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample),
        encoding="utf-8",
    )
    audit["manual_review_performed"] = False
    audit["manual_sample_review_status"] = {
        "sampled_records": len(sample),
        "reviewed_records": counts["reviewed"],
        "blocked_records": counts["blocked"],
        "pending_records": counts["pending"],
        "ledger_misses": counts["ledger_miss"],
        "reviewed_rounds": [1, 2, 3, 4, 6, 7, 8, 9, 10],
        "blocked_rounds": [5],
        "review_method": "AI-assisted subagent review of selected excerpts; not human manual review",
        "unique_sampled_reviewed_works": len(
            {
                row.get("pmcid")
                for row in sample
                if row.get("review_status_code") == "selected_passages_reviewed"
            }
        ),
        "previously_inspected_fulltexts_not_in_random_sample": [
            "10.1038/s41467-025-56526-1",
            "10.1038/s41467-024-55676-y",
            "10.1038/s41564-024-01759-x",
            "10.1099/mgen.0.000685",
            "10.48550/arXiv.2301.12040",
            "10.48550/arXiv.2404.10457",
        ],
        "all_2000_claims_manually_reviewed": False,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manual_sampled_fulltext_reviewed_records"] = counts["reviewed"]
    summary["manual_sampled_fulltext_blocked_records"] = counts["blocked"]
    summary["manual_sampled_rounds_reviewed"] = [1, 2, 3, 4, 6, 7, 8, 9, 10]
    summary["manual_sampled_rounds_blocked"] = [5]
    summary["all_1806_fulltext_claims_manually_reviewed"] = False
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "sample_count": len(sample),
        "reviewed": counts["reviewed"],
        "blocked": counts["blocked"],
        "pending": counts["pending"],
        "ledger_misses": counts["ledger_miss"],
        "full_corpus_manually_reviewed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sampled full-text review outcomes without implying full-corpus review.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--normalized-ledger", type=Path, required=True)
    parser.add_argument("--iteration-dir", type=Path, default=Path("reference/iterations_261007"))
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.campaign_root, args.normalized_ledger, args.iteration_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
