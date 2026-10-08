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
    require_gold: bool = True,
) -> torch.Tensor:
    wanted = set(gene_ids)
    labels: Dict[str, object] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            validate_label(record)
            if record["task"] == task and record["gene_id"] in wanted:
                level = record.get("evidence_level")
                if require_gold and level not in {"E1", "E2"}:
                    continue
                gene_id = record["gene_id"]
                if gene_id in labels and float(labels[gene_id]) != float(record["label"]):
                    raise ValueError(f"conflicting labels require quarantine: {gene_id}/{task}")
                labels[gene_id] = record["label"]
    missing = wanted - set(labels)
    if missing:
        scope = "E1/E2 " if require_gold else ""
        raise KeyError(f"missing {scope}{task} labels for {len(missing)} genes")
    dtype = torch.float32
    return torch.tensor([labels[gene_id] for gene_id in gene_ids], dtype=dtype)


def load_gene_groups(path: str | Path, gene_ids: Sequence[str]) -> list[str]:
    groups = {}
    methods = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            gene_id = str(record["gene_id"])
            group_id = str(record.get("split_group_id") or record.get("group_id") or "unknown")
            method = str(record.get("split_group_method") or "unknown")
            if not group_id or group_id == "unknown":
                raise ValueError(f"missing leakage-safe split group for {gene_id}")
            if gene_id in groups and groups[gene_id] != group_id:
                raise ValueError(f"conflicting split group assignments for {gene_id}")
            groups[gene_id] = group_id
            methods.add(method)
    missing = set(gene_ids) - set(groups)
    if missing:
        raise KeyError(f"missing group IDs for {len(missing)} genes")
    if methods != {"mmseqs2_90"}:
        raise ValueError("group manifest must document MMseqs2 90%-identity clusters")
    return [groups[gene_id] for gene_id in gene_ids]


def load_label_tiers(path: str | Path, gene_ids: Sequence[str], task: str) -> list[str]:
    wanted = set(gene_ids)
    tiers: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            validate_label(record)
            if record["task"] == task and record["gene_id"] in wanted:
                tiers[record["gene_id"]] = str(
                    record.get("evidence_level") or record.get("level") or "E4"
                )
    missing = wanted - set(tiers)
    if missing:
        raise KeyError(f"missing {task} tiers for {len(missing)} genes")
    return [tiers[gene_id] for gene_id in gene_ids]
