"""End-to-end coupled multi-view VaccineGPT model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .layers import MultiViewFusion, RelationMessagePassing, ResidualMLP


class VaccineGPTModel(nn.Module):
    """M1 representation learning plus coupled M2 task heads.

    ``detach_couplings=True`` prevents task heads from changing the shared
    representation through auxiliary-task probabilities.  The main latent
    path remains trainable; this makes the detach boundary explicit and
    testable rather than relying on an accidental ``no_grad`` block.
    """

    def __init__(
        self,
        view_dims: Mapping[str, int],
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_relations: int = 7,
        num_classes: int = 3,
    ):
        super().__init__()
        required = ("protein", "dna", "rna", "omics")
        missing = [name for name in required if name not in view_dims]
        if missing:
            raise ValueError(f"view_dims missing {missing}")
        self.view_names = required
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleDict(
            {name: ResidualMLP(int(view_dims[name]), latent_dim, hidden_dim) for name in required}
        )
        self.fusion = MultiViewFusion(latent_dim, required)
        self.graph = RelationMessagePassing(latent_dim, num_relations)
        self.representation = ResidualMLP(latent_dim, latent_dim, hidden_dim)
        self.t1_head = nn.Linear(latent_dim, num_classes)
        self.t2_head = nn.Sequential(nn.Linear(latent_dim + num_classes, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.t3a_head = nn.Sequential(nn.Linear(latent_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.t3b_head = nn.Sequential(nn.Linear(latent_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.t4_head = nn.Sequential(nn.Linear(latent_dim + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def encode(
        self,
        views: Mapping[str, torch.Tensor],
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [name for name in self.view_names if name not in views]
        if missing:
            raise KeyError(f"missing views: {missing}")
        encoded = {name: self.encoders[name](views[name]) for name in self.view_names}
        fused, weights = self.fusion(encoded)
        return self.representation(self.graph(fused, edge_index, edge_type)), weights

    def forward(
        self,
        views: Mapping[str, torch.Tensor],
        *,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        detach_couplings: bool = True,
    ) -> dict[str, Any]:
        latent, view_weights = self.encode(views, edge_index, edge_type)
        t1_logits = self.t1_head(latent)
        t1_prob = torch.softmax(t1_logits, dim=-1)
        coupling_t1 = t1_prob.detach() if detach_couplings else t1_prob
        t2_logit = self.t2_head(torch.cat([latent, coupling_t1], dim=-1)).squeeze(-1)
        t2_prob = torch.sigmoid(t2_logit)
        coupling_t2 = t2_prob.detach() if detach_couplings else t2_prob
        t3a_logit = self.t3a_head(torch.cat([latent, coupling_t2.unsqueeze(-1)], dim=-1)).squeeze(-1)
        t3a_prob = torch.sigmoid(t3a_logit)
        t3a_coupling = t3a_prob.detach() if detach_couplings else t3a_prob
        t3b_logit = self.t3b_head(torch.cat([latent, t3a_coupling.unsqueeze(-1)], dim=-1)).squeeze(-1)
        rank_features = torch.cat(
            [
                latent,
                coupling_t1,
                (t3a_prob.detach() if detach_couplings else t3a_prob).unsqueeze(-1),
            ],
            dim=-1,
        )
        t4_score = self.t4_head(rank_features).squeeze(-1)
        return {
            "latent": latent,
            "view_weights": view_weights,
            "T1": t1_logits,
            "T2": t2_logit,
            "T3a": t3a_logit,
            "T3b": t3b_logit,
            "T4": t4_score,
            "couplings_detached": detach_couplings,
        }
