from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def grouped_split(
    group_ids: Sequence[str],
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by group before fitting preprocessing, preventing group leakage."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must leave a test split")
    groups = np.asarray(list(dict.fromkeys(group_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = max(1, int(len(groups) * train_fraction))
    n_valid = max(1, int(len(groups) * validation_fraction))
    train_groups = set(groups[:n_train])
    valid_groups = set(groups[n_train:n_train + n_valid])
    train = np.asarray([i for i, group in enumerate(group_ids) if group in train_groups])
    valid = np.asarray([i for i, group in enumerate(group_ids) if group in valid_groups])
    test = np.asarray([i for i, group in enumerate(group_ids) if group not in train_groups | valid_groups])
    if not len(test):
        raise ValueError("grouped split produced an empty test set")
    return train, valid, test


def assert_disjoint(*indices: Iterable[int]) -> None:
    sets = [set(index) for index in indices]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            if sets[left] & sets[right]:
                raise ValueError("data split leakage detected")
