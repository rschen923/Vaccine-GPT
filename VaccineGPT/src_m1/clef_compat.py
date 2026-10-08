from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


CLEF_REPOSITORY = "https://github.com/PengYueECUST1997/CLEF"
CLEF_COMMIT = "6ad84260db7a7ecdf491b08e54a89b8428bac52d"
CLEF_LICENSE = "MIT"
CLEF_UNUSED_MULTIMODAL_ENCODER_PREFIXES = ("feat_encoder.", "ln_f.")


class _CLEFAttention(nn.Module):
    def __init__(self, dimension: int, heads: int = 8, dropout: float = 0.05):
        super().__init__()
        if dimension % heads:
            raise ValueError("CLEF embedding dimension must be divisible by attention heads")
        head_dimension = dimension // heads
        self.scale = head_dimension**-0.5
        self.heads = heads
        self.to_q = nn.Linear(dimension, head_dimension * heads, bias=False)
        self.to_k = nn.Linear(dimension, head_dimension * heads, bias=False)
        self.to_v = nn.Linear(dimension, head_dimension * heads, bias=False)
        self.to_out = nn.Linear(head_dimension * heads, dimension)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = values.shape
        head_dimension = self.to_q.out_features // self.heads

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.reshape(
                batch_size, sequence_length, self.heads, head_dimension
            ).transpose(1, 2)

        query = split_heads(self.to_q(values)) * self.scale
        key = split_heads(self.to_k(values))
        value = split_heads(self.to_v(values))
        logits = query @ key.transpose(-2, -1)
        if mask is not None:
            # Match upstream CLEF: its non-in-place masked_fill call leaves logits unchanged.
            logits.masked_fill(mask, -1e9)
        attention = self.attn_dropout(torch.softmax(logits, dim=-1))
        attended = attention @ value
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.heads * head_dimension
        )
        return self.to_out(attended), attention


class _CLEFTransformerLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int = 8,
        dropout_rate: float = 0.45,
        attention_dropout: float = 0.05,
    ):
        super().__init__()
        self.attn = _CLEFAttention(dimension, heads, attention_dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension),
            nn.Dropout(dropout_rate),
        )
        self.layernorm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attn(self.layernorm(values), mask)
        values = values + self.dropout(attended)
        return values + self.ffn(values), attention


class CLEFSequenceEncoder(nn.Module):
    """Sequence-only inference encoder compatible with public CLEF `clef_enc` weights.

    Source implementation: PengYueECUST1997/CLEF, commit 6ad8426.
    The official inference path uses only ESM2 sequence embeddings; auxiliary
    features are needed during CLEF contrastive pretraining, not inference.
    """

    def __init__(
        self,
        num_embeds: int = 1280,
        num_hiddens: int = 128,
        max_length: int = 256,
        final_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_embeds = num_embeds
        self.num_hiddens = num_hiddens
        self.max_length = max_length
        self.layers = nn.ModuleList(
            [_CLEFTransformerLayer(num_embeds, 8, 0.45, 0.05) for _ in range(2)]
        )
        self.Dropout = nn.Dropout(final_dropout)
        self.ln = nn.LayerNorm(num_embeds)
        self.mlp = nn.Sequential(
            nn.Linear(num_embeds, 2 * num_embeds),
            nn.ReLU(),
            nn.Linear(2 * num_embeds, num_hiddens),
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_residue_representations: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "esm_feature" not in batch or "valid_lens" not in batch:
            raise ValueError("CLEF input requires esm_feature and valid_lens")
        values = batch["esm_feature"]
        valid_lens = batch["valid_lens"].to(device=values.device, dtype=torch.long)
        if values.ndim != 3 or values.shape[-1] != self.num_embeds:
            raise ValueError(
                f"esm_feature must have shape [batch, length, {self.num_embeds}]"
            )
        if valid_lens.shape != (values.shape[0],):
            raise ValueError("valid_lens must contain one length per sequence")
        if torch.any(valid_lens < 2) or torch.any(valid_lens > values.shape[1]):
            raise ValueError("valid_lens must include ESM2 special tokens and fit input length")

        values = values[:, : self.max_length]
        token_lens = valid_lens.clamp(max=values.shape[1])
        positions = torch.arange(values.shape[1], device=values.device)
        padding_mask = positions.unsqueeze(0) >= token_lens.unsqueeze(1)
        attention_mask = padding_mask[:, None, None, :]
        for layer in self.layers:
            values, _ = layer(values, attention_mask)

        if return_residue_representations:
            pooled = torch.stack(
                [
                    values[index, : int(length)].mean(dim=0)
                    for index, length in enumerate(valid_lens)
                ]
            )
            output_values = values
        else:
            pooled = torch.stack(
                [
                    values[index, : min(int(length) + 2, values.shape[1])].mean(dim=0)
                    for index, length in enumerate(valid_lens)
                ]
            )
            output_values = pooled
        return output_values, self.mlp(self.Dropout(pooled))


class CLEFProteinClassifier(nn.Module):
    """Public CLEF binary downstream classifier (`test_dnn`) for one effector task."""

    def __init__(self, num_embeds: int = 1280, final_dropout: float = 0.5):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(num_embeds, 2 * num_embeds),
            nn.ReLU(),
            nn.Linear(2 * num_embeds, num_embeds),
        )
        self.out = nn.Sequential(
            nn.Linear(num_embeds, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.binaryclass = True
        self.Dropout = nn.Dropout(final_dropout)
        self.ln = nn.LayerNorm(num_embeds)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.ln(self.dnn(values))
        return torch.sigmoid(self.out(self.Dropout(values))).squeeze(-1)


def load_clef_weights(module: nn.Module, checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CLEF checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping):
        for wrapper_key in ("state_dict", "model"):
            if wrapper_key in state and isinstance(state[wrapper_key], Mapping):
                state = state[wrapper_key]
                break
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(f"CLEF checkpoint is not a plain tensor state dict: {path}")
    incompatible = module.load_state_dict(state, strict=False)
    allowed_unexpected = (
        CLEF_UNUSED_MULTIMODAL_ENCODER_PREFIXES
        if isinstance(module, CLEFSequenceEncoder)
        else ()
    )
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected)
    ]
    if incompatible.missing_keys or unexpected:
        raise ValueError(
            f"incompatible CLEF checkpoint {path}; missing keys="
            f"{incompatible.missing_keys}, unexpected keys={unexpected}"
        )
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
