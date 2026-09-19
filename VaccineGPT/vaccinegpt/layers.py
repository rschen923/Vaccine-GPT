"""Small torch layers used by VaccineGPT."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_dim or max(in_dim, out_dim)
        self.projection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x) + self.net(x))


class MultiViewFusion(nn.Module):
    """Attention-like gated fusion that tolerates different input dimensions."""

    def __init__(self, latent_dim: int, view_names: tuple[str, ...]):
        super().__init__()
        self.view_names = view_names
        self.gates = nn.ModuleDict({name: nn.Linear(latent_dim, 1) for name in view_names})
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, views: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [views[name] for name in self.view_names]
        stack = torch.stack(encoded, dim=1)
        logits = torch.cat([self.gates[name](views[name]) for name in self.view_names], dim=1)
        weights = torch.softmax(logits, dim=1)
        return self.norm((stack * weights.unsqueeze(-1)).sum(dim=1)), weights


class RelationMessagePassing(nn.Module):
    """A dependency-free relation layer with optional sparse graph messages."""

    def __init__(self, dim: int, num_relations: int = 7):
        super().__init__()
        self.num_relations = num_relations
        self.relation = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_relations)])
        self.self_loop = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None or edge_type is None:
            message = torch.stack([layer(x) for layer in self.relation], dim=0).mean(dim=0)
        else:
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape [2, edges]")
            if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
                raise ValueError("edge_type must have one value per edge")
            message = torch.zeros_like(x)
            for relation, layer in enumerate(self.relation):
                mask = edge_type == relation
                if not torch.any(mask):
                    continue
                src, dst = edge_index[:, mask].long()
                if src.max() >= x.shape[0] or dst.max() >= x.shape[0]:
                    raise ValueError("edge index is outside the batch")
                message.index_add_(0, dst, layer(x[src]))
            degree = torch.bincount(edge_index[1].long(), minlength=x.shape[0]).clamp_min(1)
            message = message / degree.to(message).unsqueeze(-1)
        return self.norm(torch.relu(self.self_loop(x) + message))

