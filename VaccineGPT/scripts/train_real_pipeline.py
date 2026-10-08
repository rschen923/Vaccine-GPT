from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_m2.training import fit_supervised_heads


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


def train(
    features_path: Path,
    labels_path: Path,
    groups_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_id: str = "UNSPECIFIED",
) -> dict[str, object]:
    feature_records = read_jsonl(features_path)
    label_records = read_jsonl(labels_path)
    group_records = read_jsonl(groups_path)
    feature_map = {}
    for record in feature_records:
        gene_id = str(record.get("gene_id") or "")
        if not gene_id or gene_id in feature_map:
            raise ValueError(f"missing or duplicate feature gene_id: {gene_id!r}")
        vector = record.get("z_gctx")
        if not isinstance(vector, list) or len(vector) != 128:
            raise ValueError(f"{gene_id}: z_gctx must contain 128 values")
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(f"{gene_id}: z_gctx contains non-finite values")
        feature_map[gene_id] = [float(value) for value in vector]
    gene_ids = sorted(feature_map)
    if not gene_ids:
        raise ValueError("feature file contains no genes")
    group_map = {
        str(row.get("gene_id") or ""): str(row.get("group_id") or "")
        for row in group_records
    }
    split_methods = {str(row.get("split_group_method") or "") for row in group_records}
    if split_methods != {"mmseqs2_90"}:
        raise ValueError("M2 training requires explicit MMseqs2 90%-identity split groups")
    missing_groups = [gene_id for gene_id in gene_ids if not group_map.get(gene_id) or group_map[gene_id] == "unknown"]
    if missing_groups:
        raise ValueError(
            f"{len(missing_groups)} genes lack trustworthy group_id values; "
            "leakage-safe training is blocked"
        )
    index = {gene_id: i for i, gene_id in enumerate(gene_ids)}
    heads = ("H1", "H2", "H3", "H4")
    labels = {
        head: torch.full((len(gene_ids),), float("nan"), dtype=torch.float32)
        for head in heads
    }
    levels = {head: ["L4"] * len(gene_ids) for head in heads}
    evidence_by_gene_task: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for record in label_records:
        gene_id = str(record.get("gene_id") or "")
        task = str(record.get("task") or "")
        if gene_id not in index or task not in heads:
            continue
        level = str(record.get("evidence_level") or "")
        value = float(record["label"])
        if not math.isfinite(value) or value not in {0.0, 1.0}:
            raise ValueError(f"{gene_id}/{task}: binary target must be 0 or 1")
        key = (gene_id, task)
        evidence_by_gene_task.setdefault(key, []).append((value, level))
    for (gene_id, task), evidence in evidence_by_gene_task.items():
        values = {value for value, _ in evidence}
        if len(values) > 1:
            raise ValueError(f"conflicting labels require quarantine: {gene_id}/{task}")
        labels[task][index[gene_id]] = next(iter(values))
        evidence_levels = {level for _, level in evidence}
        levels[task][index[gene_id]] = (
            "E1" if "E1" in evidence_levels else "E2" if "E2" in evidence_levels else (
                "E3" if "E3" in evidence_levels else "E4"
            )
        )
    return fit_supervised_heads(
        torch.tensor([feature_map[gene_id] for gene_id in gene_ids], dtype=torch.float32),
        labels,
        levels,
        [group_map[gene_id] for gene_id in gene_ids],
        output_dir,
        epochs=epochs,
        batch_id=batch_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit M2 heads only from leakage-grouped, row-provenanced E1/E2 labels."
    )
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--groups-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.features_jsonl,
                args.labels_jsonl,
                args.groups_jsonl,
                args.output_dir,
                args.epochs,
                args.batch_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
