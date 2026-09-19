"""Detached cross-layer coupling signals and bounded prior transforms."""

from __future__ import annotations

import torch

from .config import ModelConfig


def detach_signal(value: torch.Tensor) -> torch.Tensor:
    """Make the cross-layer boundary explicit and testable."""
    return value.detach()


def tau_to_v0(tau_ti: torch.Tensor, config: ModelConfig = ModelConfig()) -> torch.Tensor:
    """Primary tau_TI -> v0 coupling; upstream gradients never cross it."""
    tau = detach_signal(tau_ti)
    return (config.v0_base + config.eta_v * (tau - config.tau_ref) / config.tau_ref).clamp(0.05, 0.99)


def bounded_alpha(
    raw: torch.Tensor,
    config: ModelConfig = ModelConfig(),
    tau_ti: torch.Tensor | None = None,
) -> torch.Tensor:
    alpha = config.alpha_min + (config.alpha_max - config.alpha_min) * torch.sigmoid(raw)
    if config.enable_tau_alpha_coupling and tau_ti is not None:
        alpha = alpha + 0.02 * torch.tanh(detach_signal(tau_ti) / config.tau_ref)
    return alpha.clamp(config.alpha_min, config.alpha_max)
