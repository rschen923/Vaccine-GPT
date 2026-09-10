from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ViewEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        hidden_dim = max(in_dim // 2, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiViewAttentionFusion(nn.Module):
    def __init__(self, view_dim: int = 128, num_views: int = 4, view_names=None):
        super().__init__()
        self.num_views = num_views
        self.view_names = tuple(view_names or ("protein", "dna", "rna", "omics"))
        self.score_layer = nn.Linear(num_views, num_views)

    def forward(self, views: Dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([views[name] for name in self.view_names], dim=1)
        logits = self.score_layer(stacked.mean(dim=-1))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        fused = (weights * stacked).sum(dim=1)
        return fused


class RelationGraphEncoder(nn.Module):
    def __init__(self, dim: int = 128, num_relations: int = 7, num_bases: int = 4):
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.bases = nn.Parameter(torch.randn(num_bases, dim, dim) * 0.02)
        self.coefficients = nn.Parameter(torch.randn(num_relations, num_bases) * 0.02)
        self.self_loop = nn.Linear(dim, dim)
        self.relation_attention = nn.Parameter(torch.zeros(num_relations))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None,
                edge_type: torch.Tensor | None = None) -> torch.Tensor:
        relation_weights = torch.softmax(self.relation_attention, dim=0)
        if edge_index is None or edge_type is None:
            messages = []
            for relation in range(self.num_relations):
                weight = torch.einsum("b,bxy->xy", self.coefficients[relation], self.bases)
                messages.append(x @ weight.T * relation_weights[relation])
            aggregate = torch.stack(messages, dim=0).mean(dim=0)
        else:
            aggregate = torch.zeros_like(x)
            for relation in range(self.num_relations):
                mask = edge_type == relation
                if not torch.any(mask):
                    continue
                weight = torch.einsum("b,bxy->xy", self.coefficients[relation], self.bases)
                src, dst = edge_index[:, mask]
                aggregate.index_add_(0, dst, (x[src] @ weight.T) * relation_weights[relation])
        return torch.relu(self.self_loop(x) + aggregate)


class M1Encoder(nn.Module):
    def __init__(self, view_dims: Dict[str, int], latent_dim: int = 128, track: str = "INT"):
        super().__init__()
        if track not in ("SOM", "INT"):
            raise ValueError("track must be SOM or INT")
        self.track = track
        self.view_names = ("protein", "dna", "rna") if track == "SOM" else ("protein", "dna", "rna", "omics")
        missing_dims = set(self.view_names) - set(view_dims)
        if missing_dims:
            raise ValueError(f"missing dimensions for track {track}: {sorted(missing_dims)}")
        self.encoders = nn.ModuleDict(
            {
                name: ViewEncoder(in_dim=in_dim, out_dim=latent_dim)
                for name, in_dim in view_dims.items()
            }
        )
        self.fusion = MultiViewAttentionFusion(
            view_dim=latent_dim, num_views=len(self.view_names), view_names=self.view_names
        )
        self.rgcn = RelationGraphEncoder(dim=latent_dim, num_relations=7)
        self.output = nn.Linear(latent_dim, latent_dim)

    def forward(self, features: Dict[str, torch.Tensor],
                edge_index: torch.Tensor | None = None,
                edge_type: torch.Tensor | None = None) -> torch.Tensor:
        # NOTE: if external pretrained encoders are absent, this branch can consume cached view embeddings
        # from a feature store without changing the public interface. This preserves a stable contract for M2.
        encoded = {
            name: self.encoders[name](features[name])
            for name in self.view_names
            if name in features
        }
        if len(encoded) < len(self.view_names):
            missing = set(self.view_names) - set(encoded)
            raise KeyError(f"Missing required M1 views: {sorted(missing)}")
        fused = self.fusion(encoded)
        final = self.rgcn(fused, edge_index=edge_index, edge_type=edge_type)
        return self.output(final)
