"""Leakage-safe group splitting without scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def validate_split(split: Split, group_ids: Sequence[str]) -> None:
    arrays = [np.asarray(split.train), np.asarray(split.validation), np.asarray(split.test)]
    if any(array.ndim != 1 or len(array) == 0 for array in arrays):
        raise ValueError("train, validation, and test must all be non-empty 1-D arrays")
    n = len(group_ids)
    flat = np.concatenate(arrays)
    if np.any(flat < 0) or np.any(flat >= n) or len(np.unique(flat)) != n:
        raise ValueError("split indices must cover each row exactly once")
    groups = np.asarray(group_ids)
    sets = [set(groups[array].tolist()) for array in arrays]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("group leakage detected between splits")


def grouped_split(
    group_ids: Sequence[str],
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> Split:
    if not group_ids:
        raise ValueError("group_ids cannot be empty")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive and leave a test split")
    unique = np.asarray(list(dict.fromkeys(str(value) for value in group_ids)), dtype=object)
    if len(unique) < 3:
        raise ValueError("at least three groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = max(1, int(round(len(unique) * train_fraction)))
    n_valid = max(1, int(round(len(unique) * validation_fraction)))
    if n_train + n_valid >= len(unique):
        n_train, n_valid = len(unique) - 2, 1
    train_groups = set(unique[:n_train])
    validation_groups = set(unique[n_train:n_train + n_valid])
    groups = np.asarray(group_ids, dtype=object)
    split = Split(
        np.flatnonzero(np.isin(groups, list(train_groups))),
        np.flatnonzero(np.isin(groups, list(validation_groups))),
        np.flatnonzero(~np.isin(groups, list(train_groups | validation_groups))),
    )
    validate_split(split, group_ids)
    return split

