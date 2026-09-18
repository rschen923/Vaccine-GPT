"""L3 organism ranker with bounded ecological consistency outputs."""

from __future__ import annotations

import torch
from torch import nn


class OrganismModule(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.ranker = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.ranker(h).squeeze(-1)
