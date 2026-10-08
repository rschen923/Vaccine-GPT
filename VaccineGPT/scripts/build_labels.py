from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import classify_label_evidence
from shared.dataset import merge_annotations, write_jsonl


def records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_label(
    row: dict[str, Any], source: str, task: str | None
) -> dict[str, Any] | None:
    gene_id = (
        row.get("gene_id")
        or row.get("locus_tag")
        or row.get("protein_id")
        or row.get("accession")
        or row.get("epitope_id")
    )
    if not gene_id:
        return None
    inferred_task = task or row.get("task")
    value = row.get("label", row.get("value", row.get("score")))
    normalized_source = source.casefold()
    if inferred_task is None and normalized_source.startswith("vfdb"):
        inferred_task, value = "T2", 1.0
    if inferred_task is None and normalized_source.startswith("iedb"):
        inferred_task = "T3b"
        measures = [str(item).casefold() for item in row.get("qualitative_measures", [])]
        positive = any("positive" in item for item in measures)
        negative = any("negative" in item for item in measures)
        if positive == negative:
            return None
        value = 1.0 if positive else 0.0
    if inferred_task is None or value in (None, ""):
        return None
    try:
        label = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= label <= 1:
        raise ValueError(f"label must lie in [0, 1], got {label}")
    if inferred_task == "T3":
        inferred_task = "T3b"
    item = {
        **row,
        "gene_id": str(gene_id),
        "task": str(inferred_task),
        "label": label,
        "source": str(row.get("source") or source),
        "source_record_id": row.get("source_record_id") or row.get("pmid") or row.get("doi"),
        "assay": row.get("assay") or row.get("assay_type") or row.get("assay_names"),
        "method": row.get("method") or row.get("evidence_type"),
        "organism": row.get("organism") or row.get("taxon") or row.get("target_organism"),
        "is_private": bool(row.get("is_private") or row.get("private_dataset")),
    }
    return {**item, **classify_label_evidence(item)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build row-level, evidence-graded labels without promoting source membership to gold truth."
    )
    parser.add_argument("--input", action="append", required=True, help="JSON/JSONL source records")
    parser.add_argument("--task", action="append", help="Task override aligned with --input")
    parser.add_argument(
        "--level",
        action="append",
        help="Deprecated legacy label level; retained only as source metadata, never used as E1-E4 evidence",
    )
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output", default="data/processed/labels.jsonl")
    parser.add_argument("--rejected-output", default="data/processed/rejected_labels.jsonl")
    args = parser.parse_args()
    tasks = args.task or []
    legacy_levels = args.level or []
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    lineage["batch_id"] = args.batch_id
    lineage["batch_id"] = args.batch_id
    accepted, rejected = [], []
    for index, filename in enumerate(args.input):
        source = Path(filename).stem
        task = tasks[index] if index < len(tasks) else None
        legacy_level = legacy_levels[index] if index < len(legacy_levels) else None
        for row in records(Path(filename)):
            label = make_label(row, source, task)
            if label is None:
                rejected.append(
                    {
                        "source": source,
                        "source_record": row,
                        "reason": "missing_identifier_task_label_or_unambiguous_outcome",
                    }
                )
                continue
            label["legacy_level"] = legacy_level or row.get("level")
            label["lineage"] = lineage
            accepted.append(label)
    merged = []
    for task in sorted({row["task"] for row in accepted}):
        merged.extend(
            merge_annotations(
                [row for row in accepted if row["task"] == task],
                task,
                lineage,
            )
        )
    write_jsonl(merged, args.output)
    write_jsonl(rejected, args.rejected_output)
    print(
        json.dumps(
            {
                "accepted_evidence_rows": len(merged),
                "rejected_rows": len(rejected),
                "gold_eligible_rows": sum(row["training_eligible_as_gold"] for row in merged),
                "conflict_rows": sum(row["label_conflict"] for row in merged),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
