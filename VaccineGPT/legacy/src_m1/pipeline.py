from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from src_m1.encoders.foundation import FoundationFeatureStore
from src_m1.models import M1Encoder


class FoundationM1Encoder(nn.Module):
    """Real-sequence M1 path: foundation encoders -> trainable M1 projections."""

    def __init__(
        self,
        feature_store: FoundationFeatureStore,
        view_dims: Dict[str, int],
        track: str = "SOM",
        latent_dim: int = 128,
    ):
        super().__init__()
        self.feature_store = feature_store
        self.track = track
        self.m1 = M1Encoder(view_dims, latent_dim=latent_dim, track=track)

    def forward(
        self,
        sequences: Dict[str, Sequence[str]],
        omics: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.m1.parameters()).device
        features = {
            view: self.feature_store.encode(view, values)
            .to(device=device, dtype=torch.float32)
            for view, values in sequences.items()
            if view != "omics"
        }
        if self.track == "INT":
            if omics is None:
                raise ValueError("INT track requires an omics tensor and observation mask")
            features["omics"] = omics.to(device=device, dtype=torch.float32)
        return self.m1(features, edge_index=edge_index, edge_type=edge_type)

    def trainable_m1_parameters(self):
        return self.m1.parameters()
