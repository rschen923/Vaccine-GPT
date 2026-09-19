from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import torch

from shared.contracts import validate_label


def load_task_labels(
    path: str | Path,
    gene_ids: Sequence[str],
    task: str,
) -> torch.Tensor:
    wanted = set(gene_ids)
    labels: Dict[str, object] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            validate_label(record)
            if record["task"] == task and record["gene_id"] in wanted:
                labels[record["gene_id"]] = record["label"]
    missing = wanted - set(labels)
    if missing:
        raise KeyError(f"missing {task} labels for {len(missing)} genes")
    dtype = torch.long if task == "T1" else torch.float32
    return torch.tensor([labels[gene_id] for gene_id in gene_ids], dtype=dtype)


def load_gene_groups(path: str | Path, gene_ids: Sequence[str]) -> list[str]:
    groups = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            groups[str(record["gene_id"])] = str(record["group_id"])
    missing = set(gene_ids) - set(groups)
    if missing:
        raise KeyError(f"missing group IDs for {len(missing)} genes")
    return [groups[gene_id] for gene_id in gene_ids]


def load_label_tiers(path: str | Path, gene_ids: Sequence[str], task: str) -> list[str]:
    wanted = set(gene_ids)
    tiers: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            validate_label(record)
            if record["task"] == task and record["gene_id"] in wanted:
                tiers[record["gene_id"]] = str(record["level"])
    missing = wanted - set(tiers)
    if missing:
        raise KeyError(f"missing {task} tiers for {len(missing)} genes")
    return [tiers[gene_id] for gene_id in gene_ids]
