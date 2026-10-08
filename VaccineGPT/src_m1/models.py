from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from src_m1.gce import GCE


class CLEFEncoderLayer(nn.Module):
    def __init__(self, dimension: int = 128, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm_attention(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=False,
        )
        values = values + self.dropout(attended)
        values = values + self.dropout(self.feedforward(self.norm_feedforward(values)))
        return values, weights


class CLEFEncoderA(nn.Module):
    """Trainable sequence encoder over frozen foundation-model residue vectors."""

    def __init__(
        self,
        input_dim: int = 1536,
        latent_dim: int = 128,
        heads: int = 8,
        layers: int = 2,
        max_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList(
            CLEFEncoderLayer(latent_dim, heads, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, residue_embeddings: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residue_embeddings.ndim != 3:
            raise ValueError("residue_embeddings must have shape [batch, length, dimension]")
        if sequence_mask.ndim != 2 or sequence_mask.shape != residue_embeddings.shape[:2]:
            raise ValueError("sequence_mask must have shape [batch, length]")
        if residue_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected residue dimension {self.input_dim}, got {residue_embeddings.shape[-1]}"
            )
        values = residue_embeddings[:, : self.max_length]
        mask = sequence_mask[:, : self.max_length].to(dtype=torch.bool)
        lengths = mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("every protein must contain at least one unmasked residue")
        values = self.input_projection(values)
        last_attention = None
        for layer in self.layers:
            values, last_attention = layer(values, mask)
        values = self.output_norm(values)
        pooled = (values * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(-1).to(values.dtype)
        if last_attention is None:
            raise RuntimeError("CLEF encoder must have at least one transformer layer")
        valid_queries = mask[:, None, :, None].to(last_attention.dtype)
        scores = (last_attention * valid_queries).sum(dim=1).sum(dim=1)
        scores = scores / lengths.unsqueeze(-1).to(scores.dtype)
        scores = scores * mask.to(scores.dtype)
        minimum = scores.masked_fill(~mask, torch.inf).amin(dim=1, keepdim=True)
        maximum = scores.masked_fill(~mask, -torch.inf).amax(dim=1, keepdim=True)
        span = maximum - minimum
        position_scores = torch.where(
            span > 0,
            (scores - minimum) / span.clamp_min(torch.finfo(scores.dtype).eps),
            torch.zeros_like(scores),
        )
        return pooled, position_scores


class PrivateSequenceAdapter(nn.Module):
    """Small private-track adapter; it never updates the frozen foundation model."""

    def __init__(self, sequence_dim: int = 1536, private_dim: int = 14, latent_dim: int = 128):
        super().__init__()
        self.delta_scale = nn.Parameter(torch.zeros(sequence_dim))
        self.sequence_projection = nn.Linear(sequence_dim, latent_dim)
        self.private_projection = nn.Sequential(
            nn.LayerNorm(private_dim),
            nn.Linear(private_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.private_dim = private_dim

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        sequence_mask: torch.Tensor,
        private_features: torch.Tensor | None,
        private_feature_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = sequence_mask[:, : residue_embeddings.shape[1]].to(torch.bool)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (residue_embeddings[:, : mask.shape[1]] * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / lengths.to(pooled.dtype)
        adapted = self.sequence_projection(pooled + pooled * self.delta_scale)
        available = torch.zeros(pooled.shape[0], dtype=torch.bool, device=pooled.device)
        private = torch.zeros(
            (pooled.shape[0], self.private_dim), dtype=pooled.dtype, device=pooled.device
        )
        if private_features is not None:
            if private_features.shape != private.shape:
                raise ValueError(
                    f"private_features must have shape {tuple(private.shape)}, "
                    f"got {tuple(private_features.shape)}"
                )
            if private_feature_mask is None:
                raise ValueError("private_feature_mask is required with private_features")
            if private_feature_mask.shape != private_features.shape:
                raise ValueError("private_feature_mask must match private_features")
            valid = private_feature_mask.to(torch.bool) & torch.isfinite(private_features)
            private = torch.where(valid, private_features, torch.zeros_like(private_features))
            available = valid.any(dim=1)
        private_embedding = self.private_projection(private)
        fused = self.fusion(torch.cat((adapted, private_embedding), dim=-1))
        fused = F.normalize(fused, p=2, dim=-1)
        return fused, available, F.normalize(adapted, p=2, dim=-1), F.normalize(
            private_embedding, p=2, dim=-1
        )


class M1Encoder(nn.Module):
    """v1 M1: protein sequence, isolated public/private tracks, then one GCE."""

    def __init__(
        self,
        residue_dim: int = 1536,
        latent_dim: int = 128,
        private_dim: int = 14,
        max_length: int = 256,
        gce_relations: int = 6,
        public_modality_dims: Dict[str, int] | None = None,
    ):
        super().__init__()
        self.encoder_a = CLEFEncoderA(
            input_dim=residue_dim, latent_dim=latent_dim, max_length=max_length
        )
        self.public_projection = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        default_public_dims = {
            "annotation": 768,
            "structure_3di": 1024,
            "msa": 768,
            "pssm": 400,
        }
        configured_public_dims = public_modality_dims or default_public_dims
        self.public_modalities = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dimension),
                    nn.Linear(dimension, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim),
                )
                for name, dimension in configured_public_dims.items()
            }
        )
        self.private_adapter = PrivateSequenceAdapter(residue_dim, private_dim, latent_dim)
        self.context_input = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.gce = GCE(latent_dim, num_relations=gce_relations)

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        sequence_mask: torch.Tensor,
        private_features: torch.Tensor | None = None,
        private_feature_mask: torch.Tensor | None = None,
        public_features: Dict[str, torch.Tensor] | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        public_raw, position_scores = self.encoder_a(residue_embeddings, sequence_mask)
        z_pub = F.normalize(self.public_projection(public_raw), p=2, dim=-1)
        private_residues = residue_embeddings[:, : self.encoder_a.max_length]
        private_sequence_mask = sequence_mask[:, : self.encoder_a.max_length]
        z_exp, private_available, z_priv_sequence, z_priv_features = self.private_adapter(
            private_residues,
            private_sequence_mask,
            private_features,
            private_feature_mask,
        )
        z_exp_for_context = z_exp * private_available.unsqueeze(-1).to(z_exp.dtype)
        context_input = self.context_input(
            torch.cat(
                (
                    z_pub,
                    z_exp_for_context,
                    private_available.unsqueeze(-1).to(z_pub.dtype),
                ),
                dim=-1,
            )
        )
        z_gctx = self.gce(context_input, edge_index, edge_type, edge_weight)
        z_public_modalities: dict[str, torch.Tensor] = {}
        for name, features in (public_features or {}).items():
            if name not in self.public_modalities:
                raise ValueError(f"unsupported public modality: {name}")
            expected = self.public_modalities[name][0].normalized_shape[0]
            if features.ndim != 2 or features.shape != (z_pub.shape[0], expected):
                raise ValueError(
                    f"public modality {name} must have shape [{z_pub.shape[0]}, {expected}]"
                )
            z_public_modalities[name] = F.normalize(
                self.public_modalities[name](features), p=2, dim=-1
            )
        return {
            "z_pub": z_pub,
            "z_exp": z_exp,
            "z_gctx": z_gctx,
            "position_scores": position_scores,
            "private_available": private_available,
            "z_priv_sequence": z_priv_sequence,
            "z_priv_features": z_priv_features,
            "z_public_modalities": z_public_modalities,
        }


def freeze_for_species_adaptation(model: M1Encoder) -> int:
    """Freeze shared M1 and leave only the sequence-width private scale trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.private_adapter.delta_scale.requires_grad = True
    return model.private_adapter.delta_scale.numel()


def info_nce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.1,
    group_ids: list[str] | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with optional within-genome-only private negatives."""
    if anchors.shape != positives.shape or anchors.ndim != 2:
        raise ValueError("anchors and positives must have equal [batch, dimension] shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = anchors.shape[0]
    if batch < 2:
        raise ValueError("InfoNCE requires at least two paired examples")
    similarities = F.normalize(anchors, dim=-1) @ F.normalize(positives, dim=-1).T
    similarities = similarities / temperature
    if group_ids is not None:
        if len(group_ids) != batch:
            raise ValueError("group_ids must contain one value per paired example")
        allowed = torch.tensor(
            [[left == right for right in group_ids] for left in group_ids],
            device=similarities.device,
            dtype=torch.bool,
        )
        allowed.fill_diagonal_(True)
        similarities = similarities.masked_fill(~allowed, -torch.inf)
    targets = torch.arange(batch, device=similarities.device)
    return 0.5 * (
        F.cross_entropy(similarities, targets)
        + F.cross_entropy(similarities.T, targets)
    )
