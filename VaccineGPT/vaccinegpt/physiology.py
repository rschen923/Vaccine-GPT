"""Differentiable analytic consistency penalties, not fitted biological claims."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def consistency_loss(predicted: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, expected)


def ode_consistency_loss(trajectory: torch.Tensor, rhs: torch.Tensor, dt: float) -> torch.Tensor:
    if trajectory.shape[0] < 2:
        return trajectory.sum() * 0
    return F.mse_loss((trajectory[1:] - trajectory[:-1]) / dt, rhs[:-1])


def softmax_consistency_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities.sum(dim=-1) - 1.0).square().mean()


def markov_consistency_loss(transition: torch.Tensor) -> torch.Tensor:
    return F.relu(-transition).mean() + (transition.sum(dim=-1) - 1.0).square().mean()


def topology_loss(scores: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    return ((scores[..., None] - scores[..., None, :]) * adjacency).square().mean()


def powerlaw_loss(x: torch.Tensor, y: torch.Tensor, exponent: float) -> torch.Tensor:
    return F.mse_loss(torch.log(x.clamp_min(1e-6)), torch.log(y.clamp_min(1e-6)) * exponent)


def lv_consistency_loss(trajectory: torch.Tensor) -> torch.Tensor:
    return F.relu(-trajectory).mean()


def hill_consistency_loss(response: torch.Tensor) -> torch.Tensor:
    return F.relu(-response).mean() + F.relu(response - 1.0).mean()


def sir_consistency_loss(state: torch.Tensor) -> torch.Tensor:
    return F.relu(-state).mean() + F.relu(state.sum(dim=-1) - 1.0).mean()


def hla_consistency_loss(weights: torch.Tensor) -> torch.Tensor:
    return F.relu(-weights).mean() + (weights.sum(dim=-1) - 1.0).square().mean()


def evolution_consistency_loss(attenuation: torch.Tensor) -> torch.Tensor:
    return F.relu(-attenuation).mean() + F.relu(attenuation - 1.0).mean()


def mhc_peptide_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    if mask is not None:
        values = values[mask.bool()]
    return values.mean() if values.numel() else logits.sum() * 0.0
