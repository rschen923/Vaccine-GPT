from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def normalized_entropy(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=float)
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 0 or len(values) <= 1:
        return 0.0
    probabilities = values / total
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return float(entropy / np.log(len(values)))


def protein_immunogenicity(
    epitope_scores: Iterable[float], diversity_power: float = 1.0
) -> float:
    scores = np.asarray(list(epitope_scores), dtype=float)
    if scores.size == 0:
        raise ValueError("at least one epitope score is required")
    density = float(np.clip(scores, 0.0, 1.0).mean())
    return density * normalized_entropy(scores) ** diversity_power
