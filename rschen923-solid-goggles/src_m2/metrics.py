from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def ndcg_at_k(scores: Sequence[float], relevance: Sequence[float], k: int = 10) -> float:
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    if scores.shape != relevance.shape:
        raise ValueError("scores and relevance must have the same shape")
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    predicted = np.argsort(-scores)[:k]
    ideal = np.sort(relevance)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2.0 ** relevance[predicted] - 1.0) * discounts)
    idcg = np.sum((2.0 ** ideal - 1.0) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_group_ndcg(
    scores: Sequence[float], relevance: Sequence[float], groups: Iterable[str], k: int = 10
) -> float:
    grouped = {}
    for score, label, group in zip(scores, relevance, groups):
        grouped.setdefault(group, ([], []))
        grouped[group][0].append(score)
        grouped[group][1].append(label)
    if not grouped:
        return 0.0
    return float(np.mean([ndcg_at_k(s, r, k) for s, r in grouped.values()]))


def mean_group_mrr(
    scores: Sequence[float], relevance: Sequence[float], groups: Iterable[str]
) -> float:
    grouped = {}
    for score, label, group in zip(scores, relevance, groups):
        grouped.setdefault(group, ([], []))
        grouped[group][0].append(score)
        grouped[group][1].append(label)
    reciprocal_ranks = []
    for group_scores, group_labels in grouped.values():
        order = np.argsort(-np.asarray(group_scores))
        positives = np.asarray(group_labels)[order] > 0
        reciprocal_ranks.append(1.0 / (np.argmax(positives) + 1) if positives.any() else 0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def pairwise_ranking_loss(scores, relevance):
    import torch
    import torch.nn.functional as functional

    losses = [
        functional.softplus(-(scores[i] - scores[j]))
        for i in range(len(scores))
        for j in range(len(scores))
        if relevance[i] > relevance[j]
    ]
    return torch.stack(losses).mean() if losses else scores.square().mean() * 0
