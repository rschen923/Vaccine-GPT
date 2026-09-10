"""Optional adapters for the original pretrained model weights.

The adapters intentionally import external packages lazily. This keeps cached
feature mode usable on CPU while making the production path explicit and
reproducible when the upstream environments are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PretrainedSpec:
    family: str
    model_id: str
    source: str
    revision: str | None = None
    layer: int | str | None = None
    frozen: bool = True


def _freeze(module: nn.Module, frozen: bool) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad = not frozen
    if frozen:
        module.eval()
    return module


class ESM2Adapter(nn.Module):
    """ESM-2 per-sequence embedding adapter using the original fair-esm model."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        layer: int = 33,
        frozen: bool = True,
    ):
        super().__init__()
        try:
            import esm
        except ImportError as exc:
            raise ImportError("Install fair-esm to use ESM2Adapter") from exc
        loader = getattr(esm.pretrained, model_name, None)
        if loader is None:
            raise ValueError(f"unknown fair-esm model: {model_name}")
        self.model, self.alphabet = loader()
        self.layer = layer
        _freeze(self.model, frozen)
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        data = [(str(index), sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
            result = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
        representations = result["representations"][self.layer]
        lengths = (tokens != self.alphabet.padding_idx).sum(dim=1)
        pooled = [
            representations[i, 1 : int(length) - 1].mean(dim=0)
            for i, length in enumerate(lengths)
        ]
        return torch.stack(pooled)


class TransformersSequenceAdapter(nn.Module):
    """Adapter for DNABERT-2 or ProteomeLM checkpoints on Hugging Face."""

    def __init__(
        self,
        model_id: str,
        tokenizer_id: str | None = None,
        revision: str | None = None,
        frozen: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use this adapter") from exc
        kwargs = {"trust_remote_code": trust_remote_code}
        if revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id, **kwargs)
        self.model = AutoModel.from_pretrained(model_id, **kwargs)
        _freeze(self.model, frozen)

    def forward(self, sequences: Sequence[str], max_length: int = 1024) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            list(sequences), padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
            outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded.get("attention_mask", torch.ones(hidden.shape[:2], device=device))
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


class Evo2Adapter(nn.Module):
    """Evo2 DNA embedding adapter.

    Evo2 is an inference package with a CUDA/Linux-oriented runtime. The layer
    is configurable because released checkpoints expose different layer names.
    """

    def __init__(
        self,
        model_name: str = "evo2_7b_base",
        layer_name: str = "blocks.28.mlp.l3",
        frozen: bool = True,
    ):
        super().__init__()
        try:
            from evo2 import Evo2
        except ImportError as exc:
            raise ImportError("Install evo2 in its supported Linux/WSL2 environment") from exc
        self.model = Evo2(model_name)
        self.layer_name = layer_name
        _freeze(self.model, frozen)

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        pooled = []
        for sequence in sequences:
            input_ids = torch.tensor(
                self.model.tokenizer.tokenize(sequence), dtype=torch.int,
            ).unsqueeze(0).to(next(self.model.parameters()).device)
            with torch.no_grad():
                _, embeddings = self.model(
                    input_ids, return_embeddings=True, layer_names=[self.layer_name],
                )
            pooled.append(embeddings[self.layer_name].mean(dim=1).squeeze(0))
        return torch.stack(pooled)


class ProteinDualAdapter(nn.Module):
    """Concatenate frozen ESM-2 and ProteomeLM sequence representations."""

    def __init__(self, esm: ESM2Adapter, proteomelm: TransformersSequenceAdapter):
        super().__init__()
        self.esm = esm
        self.proteomelm = proteomelm

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        return torch.cat([self.esm(sequences), self.proteomelm(sequences)], dim=-1)


def build_pretrained_adapters(config: Dict[str, Dict]) -> Dict[str, nn.Module]:
    """Build configured view adapters without changing their original weights."""
    adapters: Dict[str, nn.Module] = {}
    for view, spec in config.items():
        family = spec["family"].lower()
        common = {"frozen": bool(spec.get("frozen", True))}
        if family == "esm2":
            adapters[view] = ESM2Adapter(
                model_name=spec.get("model_id", "esm2_t33_650M_UR50D"),
                layer=int(spec.get("layer", 33)), **common,
            )
        elif family in {"dnabert2", "proteomelm"}:
            adapters[view] = TransformersSequenceAdapter(
                model_id=spec["model_id"],
                tokenizer_id=spec.get("tokenizer_id"),
                revision=spec.get("revision"), **common,
            )
        elif family == "evo2":
            adapters[view] = Evo2Adapter(
                model_name=spec.get("model_id", "evo2_7b_base"),
                layer_name=spec.get("layer_name", "blocks.28.mlp.l3"), **common,
            )
        elif family == "protein_dual":
            esm_spec = spec["esm2"]
            plm_spec = spec["proteomelm"]
            adapters[view] = ProteinDualAdapter(
                ESM2Adapter(
                    model_name=esm_spec.get("model_id", "esm2_t33_650M_UR50D"),
                    layer=int(esm_spec.get("layer", 33)), **common,
                ),
                TransformersSequenceAdapter(
                    model_id=plm_spec["model_id"],
                    tokenizer_id=plm_spec.get("tokenizer_id"),
                    revision=plm_spec.get("revision"), **common,
                ),
            )
        else:
            raise ValueError(f"unsupported pretrained family: {spec['family']}")
    return adapters
