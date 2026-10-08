from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import group_labels_by_gene_task


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evidence-gated manifest; weak labels are not gold supervision."
    )
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", default="data/processed/training_manifest.json")
    args = parser.parse_args()
    sequences = read_jsonl(Path(args.sequences))
    labels = read_jsonl(Path(args.labels))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    split_by_gene = {gene: split for split, genes in splits.items() for gene in genes}
    grouped = group_labels_by_gene_task(labels)
    by_gene: dict[str, list[dict]] = defaultdict(list)
    for (gene_id, task), group in grouped.items():
        by_gene[gene_id].append({"task": task, **group})
    manifest = []
    counts = defaultdict(int)
    for sequence in sequences:
        gene_id = str(sequence.get("gene_id") or "")
        task_groups = by_gene.get(gene_id, [])
        if gene_id not in split_by_gene or not task_groups:
            continue
        direct_tasks = []
        weak_tasks = []
        conflicts = []
        for group in task_groups:
            task = group["task"]
            if group["label_conflict"]:
                conflicts.append(task)
                counts["conflict_groups"] += 1
                continue
            evidence = group["records"]
            if any(row.get("evidence_level") in {"E1", "E2"} for row in evidence):
                direct_tasks.append(task)
                counts[f"direct_{task}"] += 1
            else:
                weak_tasks.append(task)
                counts[f"weak_{task}"] += 1
        status = (
            "quarantine_conflict"
            if conflicts
            else "eligible_for_supervised_training"
            if direct_tasks
            else "weak_supervision_only"
        )
        manifest.append(
            {
                "gene_id": gene_id,
                "split": split_by_gene[gene_id],
                "group_id": sequence.get("operon_id")
                or sequence.get("contig_id")
                or sequence.get("genome_id")
                or "unknown",
                "sequence_hash": sequence.get("sequence_hash"),
                "sequence_role": sequence.get("sequence_role", "unknown"),
                "direct_supervision_tasks": sorted(set(direct_tasks)),
                "weak_supervision_tasks": sorted(set(weak_tasks)),
                "conflict_tasks": sorted(set(conflicts)),
                "status": status,
            }
        )
    eligible = sum(row["status"] == "eligible_for_supervised_training" for row in manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "vaccinegpt-training-manifest-1.0",
                "records": manifest,
                "record_count": len(manifest),
                "direct_supervision_record_count": eligible,
                "weak_supervision_only_record_count": sum(
                    row["status"] == "weak_supervision_only" for row in manifest
                ),
                "conflict_quarantine_record_count": sum(
                    row["status"] == "quarantine_conflict" for row in manifest
                ),
                "task_counts": dict(counts),
                "training_ready": eligible > 0
                and all(row["group_id"] != "unknown" for row in manifest if row["status"] == "eligible_for_supervised_training"),
                "policy": {
                    "gold_levels": ["E1", "E2"],
                    "E3_E4": "weak supervision only",
                    "conflicts": "quarantine; never average or overwrite",
                    "unlabeled_records": "not converted to negative labels",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(manifest),
                "direct_supervision": eligible,
                "training_ready": eligible > 0
                and all(row["group_id"] != "unknown" for row in manifest if row["status"] == "eligible_for_supervised_training"),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
