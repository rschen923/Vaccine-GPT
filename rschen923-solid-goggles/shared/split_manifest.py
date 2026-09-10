from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


def stratified_group_split(
    records: Sequence[Dict],
    group_key: str = "group_id",
    label_key: str = "stratum",
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> Dict[str, list[str]]:
    """Assign whole groups while approximately preserving label strata."""
    if not records:
        raise ValueError("cannot split an empty dataset")
    groups = defaultdict(list)
    for record in records:
        group = str(record.get(group_key) or record["gene_id"])
        groups[group].append(record)
    by_stratum = defaultdict(list)
    for group, members in groups.items():
        counts = defaultdict(int)
        for member in members:
            counts[str(member.get(label_key, "unknown"))] += 1
        stratum = max(counts, key=counts.get)
        by_stratum[stratum].append(group)
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "test": []}
    for stratum, stratum_groups in by_stratum.items():
        shuffled = list(stratum_groups)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_fraction))
        n_valid = max(1, int(len(shuffled) * validation_fraction)) if len(shuffled) > 3 else 0
        if n_train + n_valid >= len(shuffled) and len(shuffled) > 1:
            n_train = max(1, len(shuffled) - 1 - n_valid)
        result["train"].extend(shuffled[:n_train])
        result["validation"].extend(shuffled[n_train:n_train + n_valid])
        result["test"].extend(shuffled[n_train + n_valid:])
    if not result["test"]:
        candidates = result["validation"] or result["train"]
        if len(candidates) < 2:
            raise ValueError("split requires at least two groups")
        moved = candidates.pop()
        result["test"].append(moved)
    return result


def write_split_manifest(
    records: Sequence[Dict], output: str | Path, **kwargs
) -> Dict[str, list[str]]:
    manifest = stratified_group_split(records, **kwargs)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
