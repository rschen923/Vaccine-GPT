from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROUND_FILES = {
    1: "iteration_01_code_review.md",
    2: "iteration_02_code_review.md",
    3: "iteration_03_code_review.md",
    4: "iteration_04_code_review.md",
    5: "iteration_05_code_review.md",
    6: "iteration_06_code_review.md",
    7: "iteration_07_code_review.md",
    8: "iteration_08_code_review.md",
    9: "iteration_09_code_review.md",
    10: "iteration_10_code_review.md",
}


def parse_row(line: str, round_no: int, source_path: Path) -> dict:
    match = re.search(r"\[([^\]]+)\]\((https://github\.com/[^)]+)\)", line)
    if not match:
        raise ValueError(f"no canonical GitHub URL found in counted repository row: {line}")
    name, url = match.group(1), match.group(2)
    repo = url.removeprefix("https://github.com/").rstrip("/")
    evidence_paths = re.findall(r"`([^`]+)`", line)
    license_status = line.rsplit("|", 1)[-1].strip().strip("|").strip()
    return {
        "round": round_no,
        "repo": repo,
        "url": url,
        "display_name": name,
        "license_file_or_status": license_status,
        "evidence_paths": evidence_paths,
        "evidence_row": line.strip(),
        "source_review_file": str(source_path),
        "retrieved_at": "2026-10-07",
        "verification_scope": "README/relevant source/license path inspection as stated by round reviewer; not a legal compatibility audit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize 10 per-round public code review catalogues.")
    parser.add_argument("--source-dir", type=Path, default=Path("reference/iterations_261007"))
    parser.add_argument("--output", type=Path, default=Path("reference/code_evidence_ledger_261007.jsonl"))
    args = parser.parse_args()
    records = []
    round_counts = {}
    errors = []
    for round_no, filename in ROUND_FILES.items():
        path = args.source_dir / filename
        if not path.is_file():
            errors.append({"round": round_no, "error": "review_file_missing", "path": str(path)})
            round_counts[round_no] = 0
            continue
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "https://github.com/" not in line or not line.lstrip().startswith("|"):
                continue
            try:
                row = parse_row(line, round_no, path)
            except ValueError as error:
                errors.append({"round": round_no, "line": line_no, "error": str(error)})
                continue
            rows.append(row)
        records.extend(rows)
        round_counts[round_no] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    global_repos = {row["repo"].casefold() for row in records}
    summary = {
        "schema_version": "vaccinegpt-code-evidence-ledger-1.0",
        "round_counts": {str(key): value for key, value in round_counts.items()},
        "round_reference_total": len(records),
        "globally_unique_repositories": len(global_repos),
        "duplicate_references_across_rounds": len(records) - len(global_repos),
        "quota_target_per_round": 20,
        "all_round_quotas_met": all(value >= 20 for value in round_counts.values()),
        "parse_errors": errors,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    summary_path = args.output.with_name("code_evidence_ledger_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
