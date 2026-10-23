from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import normalize_protein


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(root: Path, sample_per_round: int = 10, seed: int = 261007) -> dict[str, Any]:
    manifest_rows = read_jsonl(root / "query_manifest.jsonl")
    latest_manifest = {}
    for row in manifest_rows:
        latest_manifest[int(row["round"])] = row
    integrity_errors = []
    round_results = []
    manual_review_sample = []
    globally_seen: dict[str, dict[str, Any]] = {}
    canonical_works: set[str] = set()
    identity_conflicts = []
    rng = random.Random(seed)
    for round_no in range(1, 11):
        spec_path = root / "rounds" / f"round_{round_no:02d}_references.jsonl"
        rows = read_jsonl(spec_path)
        pmcids = [row.get("pmcid") for row in rows]
        duplicates = len(pmcids) - len(set(pmcids))
        verified = 0
        for row in rows:
            pmcid = str(row.get("pmcid") or "")
            path = Path(row.get("fulltext_path") or "")
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not pmcid or row.get("access_level") != "open_full_text":
                integrity_errors.append(
                    {"round": round_no, "pmcid": pmcid, "error": "access_state_not_open_full_text"}
                )
                continue
            if not path.is_file():
                integrity_errors.append(
                    {"round": round_no, "pmcid": pmcid, "error": "fulltext_file_missing", "path": str(path)}
                )
                continue
            try:
                with gzip.open(path, "rb") as stream:
                    xml = stream.read()
            except OSError as error:
                integrity_errors.append(
                    {"round": round_no, "pmcid": pmcid, "error": f"gzip_error:{error}"}
                )
                continue
            actual_sha = hashlib.sha256(xml).hexdigest()
            if actual_sha != row.get("fulltext_xml_sha256"):
                integrity_errors.append(
                    {"round": round_no, "pmcid": pmcid, "error": "fulltext_sha256_mismatch"}
                )
                continue
            if not xml.lstrip().startswith((b"<?xml", b"<!DOCTYPE", b"<article")):
                integrity_errors.append(
                    {"round": round_no, "pmcid": pmcid, "error": "fulltext_not_xml"}
                )
                continue
            verified += 1
            for identifier_name, identifier in (
                ("doi", row.get("doi")),
                ("pmid", row.get("pmid")),
                ("pmcid", pmcid),
            ):
                if not identifier:
                    continue
                key = f"{identifier_name}:{str(identifier).casefold()}"
                prior = globally_seen.get(key)
                if prior and prior.get("title") != row.get("title"):
                    identity_conflicts.append(
                        {
                            "identifier": key,
                            "title_1": prior.get("title"),
                            "title_2": row.get("title"),
                            "rounds": [prior.get("round"), round_no],
                        }
                    )
                else:
                    globally_seen[key] = {
                        "title": row.get("title"),
                        "round": round_no,
                        "doi": row.get("doi"),
                        "pmid": row.get("pmid"),
                        "pmcid": pmcid,
                    }
        if rows:
            for row in rows:
                doi = str(row.get("doi") or "").strip().casefold()
                pmid = str(row.get("pmid") or "").strip()
                pmcid = str(row.get("pmcid") or "").strip().upper()
                title_key = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    unicodedata.normalize("NFKD", str(row.get("title") or "")).casefold(),
                ).strip()
                canonical_works.add(
                    f"doi:{doi}" if doi else f"pmid:{pmid}" if pmid else f"pmcid:{pmcid}" if pmcid else f"title:{title_key}"
                )
            picks = rng.sample(rows, min(sample_per_round, len(rows)))
            for row in picks:
                manual_review_sample.append(
                    {
                        "round": round_no,
                        "pmcid": row.get("pmcid"),
                        "doi": row.get("doi"),
                        "title": row.get("title"),
                        "authors": row.get("authorString"),
                        "year": row.get("pubYear"),
                        "license": row.get("license"),
                        "fulltext_sha256": row.get("fulltext_xml_sha256"),
                        "evidence_locations_to_check": row.get("fulltext_section_titles", []),
                        "manual_review_status": "selected_for_manual_review_not_yet_performed",
                    }
                )
        round_results.append(
            {
                "round": round_no,
                "references_in_round_file": len(rows),
                "unique_pmcid_count": len(set(pmcids)),
                "duplicate_pmcid_rows": duplicates,
                "fulltext_xml_sha256_verified": verified,
                "quota_200_references_met": len(rows) >= 200 and verified >= 200,
                "manifest": latest_manifest.get(round_no),
            }
        )
    result = {
        "schema_version": "vaccinegpt-literature-audit-1.0",
        "rounds": round_results,
        "all_10_round_200_fulltext_retrieval_quotas_met": all(
            row["quota_200_references_met"] for row in round_results
        ),
        "unique_work_count_doi_pmid_pmcid_title_priority": len(canonical_works),
        "distinct_identifiers_observed": len(globally_seen),
        "identifier_conflicts": identity_conflicts,
        "fulltext_integrity_errors": integrity_errors,
        "random_manual_verification_sample": manual_review_sample,
        "manual_review_performed": False,
        "interpretation": (
            "This audit verifies cached open-access PMC fullTextXML bytes and round reference counts. "
            "It does not claim that a human manually reviewed every paper's claims."
        ),
    }
    (root / "campaign_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "manual_verification_sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manual_review_sample),
        encoding="utf-8",
    )
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Verify full-text campaign artifacts and round quotas.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-per-round", type=int, default=10)
    parser.add_argument("--seed", type=int, default=261007)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.sample_per_round, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
