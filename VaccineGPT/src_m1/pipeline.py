from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src_m1.encoders.foundation import FoundationFeatureStore
from src_m1.models import M1Encoder


class FoundationM1Encoder(nn.Module):
    """Real protein-sequence M1 path: frozen foundation vectors -> sequence encoder/GCE."""

    def __init__(
        self,
        feature_store: FoundationFeatureStore,
        latent_dim: int = 128,
        residue_dim: int | None = None,
    ):
        super().__init__()
        self.feature_store = feature_store
        adapter = feature_store.adapters.get("protein")
        if adapter is None:
            raise ValueError("the feature store must contain the primary protein adapter")
        configured_dimension = getattr(adapter, "config", None)
        configured_dimension = (
            configured_dimension.extra.get("embedding_dimension")
            if configured_dimension is not None
            else None
        )
        if residue_dim is not None and residue_dim < 1:
            raise ValueError("residue_dim must be positive")
        self.residue_dim = int(
            residue_dim if residue_dim is not None else (
                configured_dimension if configured_dimension is not None else 1536
            )
        )
        self.m1 = M1Encoder(residue_dim=self.residue_dim, latent_dim=latent_dim)

    def forward(
        self,
        sequences: Sequence[str],
        private_features: torch.Tensor | None = None,
        private_feature_mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.m1.parameters()).device
        adapter = self.feature_store.adapters.get("protein")
        if adapter is None or not hasattr(adapter, "encode_tokens"):
            raise ValueError("the protein adapter must expose frozen residue-level encode_tokens")
        residues, lengths = adapter.encode_tokens(sequences)
        if residues.shape[-1] != self.residue_dim:
            raise ValueError(
                f"primary protein adapter returned dimension {residues.shape[-1]}, "
                f"but M1 is configured for {self.residue_dim}"
            )
        residues = residues.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device=device)
        positions = torch.arange(residues.shape[1], device=device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return self.m1(
            residues,
            mask,
            private_features=private_features.to(device) if private_features is not None else None,
            private_feature_mask=private_feature_mask.to(device) if private_feature_mask is not None else None,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
        )

    def trainable_m1_parameters(self):
        return self.m1.parameters()
