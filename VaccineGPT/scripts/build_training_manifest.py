from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.dataset import balanced_sample_weights


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training-ready manifest from UDC records")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", default="data/processed/training_manifest.json")
    args = parser.parse_args()
    sequences = read_jsonl(Path(args.sequences))
    labels = read_jsonl(Path(args.labels))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    split_by_gene = {gene: split for split, genes in splits.items() for gene in genes}
    by_gene = {}
    for record in labels:
        by_gene.setdefault(str(record["gene_id"]), {})[str(record["task"])] = record
    training_labels = [
        int(float(by_gene[gene_id]["T1"].get("label", 0)))
        for gene_id, split in split_by_gene.items()
        if split == "train" and gene_id in by_gene and "T1" in by_gene[gene_id]
    ]
    training_tiers = [
        str(by_gene[gene_id]["T1"].get("level", "L4"))
        for gene_id, split in split_by_gene.items()
        if split == "train" and gene_id in by_gene and "T1" in by_gene[gene_id]
    ]
    class_weights = {}
    if training_labels:
        weighted = balanced_sample_weights(training_labels, training_tiers)
        for label, tier, weight in zip(training_labels, training_tiers, weighted):
            class_weights[(label, tier)] = float(weight)
    manifest = []
    for sequence in sequences:
        gene_id = str(sequence["gene_id"])
        task_labels = by_gene.get(gene_id, {})
        if gene_id not in split_by_gene or not task_labels:
            continue
        t1 = task_labels.get("T1")
        manifest.append({
            "gene_id": gene_id,
            "split": split_by_gene[gene_id],
            "group_id": sequence.get("operon_id") or sequence.get("contig_id") or sequence.get("genome_id") or gene_id,
            "sequence_hash": sequence.get("sequence_hash"),
            "tasks": sorted(task_labels),
            "label_levels": {task: item.get("level", "L4") for task, item in task_labels.items()},
            "sample_weight": class_weights.get(
                (
                    int(float(t1.get("label", 0))) if t1 else 0,
                    str(t1.get("level", "L4")) if t1 else "L4",
                ),
                0.3,
            ),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "records": manifest,
        "split_sizes": {
            split: sum(item["split"] == split for item in manifest)
            for split in ("train", "validation", "test")
        },
        "policy": {
            "duplicates": "none",
            "weights": "effective-number times evidence-tier reliability",
            "evaluation": "grouped fixed split; use repeated grouped CV for small datasets",
        },
    }, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(manifest), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
