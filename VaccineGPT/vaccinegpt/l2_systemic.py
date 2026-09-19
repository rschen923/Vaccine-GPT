"""Bounded L2 systemic dynamics and virulence head."""

from __future__ import annotations

import torch
from torch import nn


class SystemicModule(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)

    @staticmethod
    def cytokine_rhs(state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(state) - 0.2 * state
