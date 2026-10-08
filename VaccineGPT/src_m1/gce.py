from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GATv2Layer(nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        heads: int = 4,
        num_relations: int = 6,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = dimension
        self.heads = heads
        self.head_dim = dimension // heads
        self.num_relations = num_relations
        self.source_projection = nn.Linear(dimension, dimension, bias=False)
        self.target_projection = nn.Linear(dimension, dimension, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        self.edge_bias = nn.Parameter(torch.zeros(num_relations, heads, self.head_dim))
        self.output_projection = nn.Linear(dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope
        self.norm = nn.LayerNorm(dimension)
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"values must have shape [nodes, {self.dimension}]")
        node_count = values.shape[0]
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=values.device)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        edge_index = edge_index.to(device=values.device, dtype=torch.long)
        src, dst = edge_index
        if src.numel() and (src.min() < 0 or src.max() >= node_count or dst.min() < 0 or dst.max() >= node_count):
            raise ValueError("edge_index contains a node outside the graph")
        supplied_count = src.numel()
        loops = torch.arange(node_count, device=values.device)
        src = torch.cat((src, loops))
        dst = torch.cat((dst, loops))
        if edge_type is None:
            edge_type = torch.zeros(supplied_count, dtype=torch.long, device=values.device)
        else:
            edge_type = edge_type.to(device=values.device, dtype=torch.long)
            if edge_type.numel() != supplied_count:
                raise ValueError("edge_type must contain one type per supplied edge")
        edge_type = torch.cat(
            (edge_type, torch.zeros(node_count, dtype=torch.long, device=values.device))
        )
        if edge_type.numel() and (
            edge_type.min() < 0 or edge_type.max() >= self.num_relations
        ):
            raise ValueError("edge_type is outside the configured relation vocabulary")
        if edge_weight is None:
            edge_weight = torch.ones(supplied_count, dtype=values.dtype, device=values.device)
        else:
            edge_weight = edge_weight.to(device=values.device, dtype=values.dtype)
            if edge_weight.numel() != supplied_count:
                raise ValueError("edge_weight must contain one weight per supplied edge")
            if not torch.isfinite(edge_weight).all() or torch.any(edge_weight < 0):
                raise ValueError("edge_weight must be finite and nonnegative")
        edge_weight = torch.cat((edge_weight, torch.ones(node_count, dtype=values.dtype, device=values.device)))

        source = self.source_projection(values).view(node_count, self.heads, self.head_dim)
        target = self.target_projection(values).view(node_count, self.heads, self.head_dim)
        neighbor_values = source[src] + edge_weight[:, None, None] * self.edge_bias[edge_type]
        target_values = target[dst]
        pair = torch.cat((target_values, neighbor_values), dim=-1)
        logits = F.leaky_relu(
            (pair * self.attention.unsqueeze(0)).sum(dim=-1),
            negative_slope=self.negative_slope,
        )
        coefficients = torch.empty_like(logits)
        for node in range(node_count):
            incoming = dst == node
            coefficients[incoming] = torch.softmax(logits[incoming], dim=0)
        coefficients = self.dropout(coefficients)
        messages = neighbor_values * coefficients.unsqueeze(-1)
        aggregate = torch.zeros(
            (node_count, self.heads, self.head_dim),
            dtype=values.dtype,
            device=values.device,
        )
        aggregate.index_add_(0, dst, messages)
        aggregate = self.output_projection(aggregate.reshape(node_count, self.dimension))
        return self.norm(values + self.dropout(aggregate))


class GCE(nn.Module):
    """Two-layer graph-context encoder with self loops and evidence-typed edges."""

    def __init__(
        self,
        dimension: int = 128,
        num_relations: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.layers = nn.ModuleList(
            GATv2Layer(dimension, heads, num_relations, dropout) for _ in range(2)
        )

    def forward(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values, edge_index, edge_type, edge_weight)
        return values
