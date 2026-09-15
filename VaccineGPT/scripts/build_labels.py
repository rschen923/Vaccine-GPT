from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.dataset import merge_annotations, write_jsonl


TIER_WEIGHT = {"L1": 1.0, "L2": 0.7, "L3": 0.5, "L4": 0.3}


def records(path: Path) -> list[dict]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_label(row: dict, source: str, task: str | None, default_level: str) -> dict | None:
    gene_id = row.get("gene_id") or row.get("locus_tag") or row.get("protein_id") or row.get("accession")
    if not gene_id:
        return None
    inferred_task = task or row.get("task")
    value = row.get("label", row.get("value", row.get("score")))
    if inferred_task is None and source.lower().startswith("vfdb"):
        inferred_task, value = "T2", 1
    if inferred_task is None and source.lower().startswith("iedb"):
        inferred_task = "T3"
        value = row.get("immunogenicity", row.get("response", value))
    if inferred_task is None or value in (None, ""):
        return None
    try:
        label = float(value)
    except (TypeError, ValueError):
        return None
    level = str(row.get("level") or default_level)
    if level not in TIER_WEIGHT:
        raise ValueError(f"unsupported evidence level: {level}")
    if inferred_task == "T3":
        inferred_task = "T3b"
    return {
        "gene_id": str(gene_id),
        "task": str(inferred_task),
        "level": level,
        "label": label,
        "weight": TIER_WEIGHT[level],
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Map source records into evidence-aware UDC-03 labels")
    parser.add_argument("--input", action="append", required=True, help="JSON/JSONL source records")
    parser.add_argument("--task", action="append", help="Task override aligned with --input")
    parser.add_argument("--level", action="append", help="Evidence level aligned with --input")
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output", default="data/processed/labels.jsonl")
    parser.add_argument("--rejected-output", default="data/processed/rejected_labels.jsonl")
    args = parser.parse_args()
    tasks = args.task or []
    levels = args.level or []
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    accepted, rejected = [], []
    for index, filename in enumerate(args.input):
        source = Path(filename).stem
        task = tasks[index] if index < len(tasks) else None
        level = levels[index] if index < len(levels) else "L2"
        for row in records(Path(filename)):
            label = make_label(row, source, task, level)
            (accepted if label else rejected).append(row if label is None else label)
    merged = []
    for task in sorted({row["task"] for row in accepted}):
        merged.extend(merge_annotations(
            [dict(row, lineage=lineage) for row in accepted if row["task"] == task],
            task,
            lineage,
        ))
    write_jsonl(merged, args.output)
    write_jsonl(rejected, args.rejected_output)
    print(json.dumps({"accepted": len(merged), "rejected": len(rejected), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
