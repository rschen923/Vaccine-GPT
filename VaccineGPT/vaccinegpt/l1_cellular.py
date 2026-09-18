"""Stable analytic L1 cellular dynamics and task head."""

from __future__ import annotations

import torch
from torch import nn


class CellularModule(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.head = nn.Linear(dim, 3)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"T1": self.head(h), "metabolism": torch.sigmoid(h[..., :4])}

    @staticmethod
    def rhs(_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(state) - 0.1 * state
