"""Masked, tier-aware losses for the five VaccineGPT tasks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def _weighted_mean(values: torch.Tensor, mask: torch.Tensor | None = None, weights: torch.Tensor | None = None) -> torch.Tensor:
    active = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.bool()
    factor = torch.ones_like(values) if weights is None else weights.to(values)
    factor = factor * active.to(values)
    return (values * factor).sum() / factor.sum().clamp_min(1.0)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long(), reduction="none")
    return _weighted_mean(((1 - torch.exp(-ce)) ** gamma) * ce, weights=weights)


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(query.shape[0], device=query.device))


def domain_adversarial_loss(logits: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, domain.long())


def vib_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()


def listmle_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(relevance, descending=True)
    ordered = scores[order]
    return (-ordered + torch.logcumsumexp(ordered, dim=0)).sum()


def lambdarank_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    diff = scores[:, None] - scores[None, :]
    target_diff = relevance[:, None] - relevance[None, :]
    mask = target_diff != 0
    if not torch.any(mask):
        return scores.sum() * 0.0
    weights = target_diff.abs().detach()
    return (F.softplus(-torch.sign(target_diff) * diff) * weights)[mask].mean()


def non_negative_pu_loss(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    positive_prior: float = 0.5,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0 < positive_prior < 1:
        raise ValueError("positive_prior must be in (0, 1)")
    positive = F.softplus(-positive_logits)
    negative_unlabeled = F.softplus(unlabeled_logits)
    positive_as_negative = F.softplus(positive_logits)
    risk = negative_unlabeled.mean() - positive_prior * positive_as_negative.mean()
    return positive_prior * _weighted_mean(positive, weights=weights) + torch.relu(risk)


def pairwise_rank_loss(scores: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    if scores.numel() < 2:
        return scores.sum() * 0.0
    diff = scores[:, None] - scores[None, :]
    target_diff = target[:, None] - target[None, :]
    mask = target_diff != 0
    desired = torch.sign(target_diff)
    return F.relu(margin - desired * diff)[mask].mean() if torch.any(mask) else scores.sum() * 0.0


def multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    task_weights: Mapping[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute available task losses while masking missing labels as zeros."""

    weights = {"T1": 1.0, "T2": 1.0, "T3a": 1.0, "T3b": 1.0, "T4": 1.0}
    weights.update(task_weights or {})
    losses: dict[str, torch.Tensor] = {}
    for task in ("T1", "T2", "T3a", "T3b", "T4"):
        if task not in outputs or task not in targets:
            continue
        target = targets[task]
        mask = target >= 0
        if not torch.any(mask):
            continue
        if task == "T1":
            value = F.cross_entropy(outputs[task][mask], target[mask].long(), reduction="none")
            losses[task] = _weighted_mean(value, weights=sample_weights[mask] if sample_weights is not None else None)
        elif task == "T4":
            value = F.mse_loss(outputs[task][mask], target[mask].float(), reduction="none")
            losses[task] = _weighted_mean(value, weights=sample_weights[mask] if sample_weights is not None else None)
        else:
            value = F.binary_cross_entropy_with_logits(outputs[task][mask], target[mask].float(), reduction="none")
            losses[task] = _weighted_mean(value, weights=sample_weights[mask] if sample_weights is not None else None)
    if not losses:
        raise ValueError("no supervised labels are available")
    total = sum(losses[task] * float(weights.get(task, 1.0)) for task in losses)
    return total, losses
