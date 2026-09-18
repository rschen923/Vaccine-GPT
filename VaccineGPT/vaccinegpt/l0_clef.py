"""L0 CLEF dual encoder for sequence-architecture and tabular modalities."""

from __future__ import annotations

import torch
from torch import nn


class DualEncoder(nn.Module):
    """CPU-friendly two-layer sequence Transformer plus tabular MLP.

    Inputs are ``x_a (B,L,64)`` and ``x_b (B,32)``; both are projected into a
    shared 128-dimensional representation and separate VIB latents.
    """

    def __init__(self, shared_dim: int = 128, latent_dim: int = 32, heads: int = 8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(64, heads, batch_first=True, dim_feedforward=128)
        self.sequence = nn.TransformerEncoder(layer, num_layers=2)
        self.seq_projection = nn.Linear(64, shared_dim)
        self.tabular = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, shared_dim))
        self.mu = nn.Linear(shared_dim, latent_dim)
        self.logvar = nn.Linear(shared_dim, latent_dim)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_a.ndim != 3 or x_a.shape[-1] != 64:
            raise ValueError("x_a must have shape (B,L,64)")
        if x_b.ndim != 2 or x_b.shape[-1] != 32:
            raise ValueError("x_b must have shape (B,32)")
        sequence = self.seq_projection(self.sequence(x_a).mean(dim=1))
        tabular = self.tabular(x_b)
        shared = 0.5 * (sequence + tabular)
        mu, logvar = self.mu(shared), self.logvar(shared).clamp(-10, 10)
        noise = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        return {
            "h_shared": shared, "sequence": sequence, "tabular": tabular,
            "z": mu + noise * torch.exp(0.5 * logvar), "mu": mu, "logvar": logvar,
        }
