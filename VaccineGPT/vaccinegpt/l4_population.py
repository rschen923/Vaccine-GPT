"""L4 population target and MHC-peptide binding predictor."""

from __future__ import annotations

import torch
from torch import nn


class PopulationModule(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.target = nn.Linear(dim, 1)
        self.mhc = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"T4": self.target(h).squeeze(-1), "mhc_peptide": self.mhc(h).squeeze(-1)}
