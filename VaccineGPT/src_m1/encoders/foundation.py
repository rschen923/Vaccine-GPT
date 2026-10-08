from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn


@dataclass
class FoundationModelConfig:
    """Runtime configuration for pretrained feature extraction.

    The checkpoint/model identifier is an upstream pretrained artifact. Only
    the projection and M1 layers are trained by default.
    """

    model_id: str
    local_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "float32"
    freeze: bool = True
    unfreeze_last_n_layers: int = 0
    trust_remote_code: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def _dtype(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


class FrozenAdapter(nn.Module):
    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config
        self.model: Optional[nn.Module] = None

    def _freeze_model(self) -> None:
        if self.model is None:
            raise RuntimeError("adapter model has not been loaded")
        parameters = list(self.model.named_parameters())
        if not self.config.freeze:
            for _, parameter in parameters:
                parameter.requires_grad = True
        else:
            for _, parameter in parameters:
                parameter.requires_grad = False
            if self.config.unfreeze_last_n_layers:
                for _, parameter in parameters[-self.config.unfreeze_last_n_layers:]:
                    parameter.requires_grad = True
        if self.config.freeze and not self.config.unfreeze_last_n_layers:
            self.model.eval()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        if self.model is None:
            raise RuntimeError("load the adapter before requesting parameters")
        return (p for p in self.model.parameters() if p.requires_grad)


class ESM2Adapter(FrozenAdapter):
    """ESM-2 adapter using the official `esm.pretrained` loaders."""

    def load(self) -> "ESM2Adapter":
        try:
            import esm
        except ImportError as exc:
            raise ImportError("install fair-esm to use ESM2Adapter") from exc
        if self.config.local_path:
            checkpoint = Path(self.config.local_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"ESM-2 checkpoint does not exist: {checkpoint}")
            safe_globals = getattr(torch.serialization, "safe_globals", None)
            if safe_globals is None:
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
                    str(checkpoint)
                )
            else:
                with safe_globals([argparse.Namespace]):
                    self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
                        str(checkpoint)
                    )
        else:
            loader = getattr(esm.pretrained, self.config.model_id)
            self.model, self.alphabet = loader()
        self.model.to(self.config.resolved_device())
        self._freeze_model()
        return self

    @torch.no_grad()
    def _encode_tokens_with_special_tokens(
        self, sequences: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            self.load()
        converter = self.alphabet.get_batch_converter()
        max_length = int(self.config.extra.get("max_length", 256))
        batch = [(str(i), str(seq)[:max_length]) for i, seq in enumerate(sequences)]
        _, _, tokens = converter(batch)
        tokens = tokens.to(self.config.resolved_device())
        outputs = self.model(tokens, repr_layers=[self.model.num_layers])
        reps = outputs["representations"][self.model.num_layers]
        lengths = (tokens != self.alphabet.padding_idx).sum(dim=1)
        return reps, lengths

    @torch.no_grad()
    def encode_tokens(self, sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        reps, lengths = self._encode_tokens_with_special_tokens(sequences)
        return reps[:, 1:-1], (lengths - 2).clamp_min(0)

    @torch.no_grad()
    def encode_tokens_with_special_tokens(
        self, sequences: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return BOS/EOS embeddings and token lengths, as required by CLEF."""
        return self._encode_tokens_with_special_tokens(sequences)

    @torch.no_grad()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        tokens, lengths = self.encode_tokens(sequences)
        mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        return (tokens * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)


class ESM3Adapter(FrozenAdapter):
    """Optional local ESM3 sequence-track embeddings; not a protective-antigen classifier."""

    def load(self) -> "ESM3Adapter":
        try:
            from esm.models.esm3 import ESM3
        except ImportError as exc:
            raise ImportError("install the official Biohub `esm` package to use ESM3Adapter") from exc
        source = self.config.local_path or self.config.model_id or "esm3-sm-open-v1"
        self.model = ESM3.from_pretrained(
            source, device=self.config.resolved_device()
        )
        self._freeze_model()
        return self

    @torch.inference_mode()
    def encode_tokens(self, sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            self.load()
        try:
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise ImportError("the installed Biohub `esm` package lacks the ESM3 inference API") from exc
        outputs: list[torch.Tensor] = []
        lengths: list[int] = []
        max_length = int(self.config.extra.get("max_length", 256))
        for sequence in sequences:
            trimmed = str(sequence)[:max_length]
            protein_tensor = self.model.encode(ESMProtein(sequence=trimmed))
            result = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
            embedding = result.embeddings
            if embedding is None:
                raise RuntimeError("ESM3 did not return embeddings for the sequence track")
            if embedding.ndim == 3 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            if embedding.ndim != 2:
                raise ValueError(f"unexpected ESM3 embedding shape: {tuple(embedding.shape)}")
            expected_dimension = self.config.extra.get("embedding_dimension")
            if expected_dimension is not None and embedding.shape[-1] != int(expected_dimension):
                raise ValueError(
                    f"ESM3 embedding dimension {embedding.shape[-1]} does not match "
                    f"configured dimension {expected_dimension}"
                )
            if embedding.shape[0] == len(trimmed) + 2:
                embedding = embedding[1:-1]
            elif embedding.shape[0] != len(trimmed):
                raise ValueError(
                    f"ESM3 residue count {embedding.shape[0]} does not match sequence length "
                    f"{len(trimmed)}"
                )
            outputs.append(embedding.detach())
            lengths.append(len(trimmed))
        if not outputs:
            return torch.empty((0, 0, 0)), torch.empty((0,), dtype=torch.long)
        dimension = outputs[0].shape[-1]
        if any(item.shape[-1] != dimension for item in outputs):
            raise ValueError("ESM3 returned inconsistent embedding dimensions")
        padded = torch.zeros(
            (len(outputs), max(item.shape[0] for item in outputs), dimension),
            device=outputs[0].device,
            dtype=outputs[0].dtype,
        )
        for index, item in enumerate(outputs):
            padded[index, : item.shape[0]] = item
        return padded, torch.tensor(lengths, dtype=torch.long, device=padded.device)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        tokens, lengths = self.encode_tokens(sequences)
        mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        return (tokens * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)


class TransformersSequenceAdapter(FrozenAdapter):
    """Adapter for DNABERT-2, ProteomeLM, and compatible HF checkpoints."""

    def load(self) -> "TransformersSequenceAdapter":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("install transformers to use this adapter") from exc
        source = self.config.local_path or self.config.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=self.config.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            source, trust_remote_code=self.config.trust_remote_code
        )
        self.model.to(self.config.resolved_device(), dtype=_dtype(self.config.dtype))
        self._freeze_model()
        return self

    @torch.no_grad()
    def encode(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        if self.model is None:
            self.load()
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if max_length is not None:
            kwargs["max_length"] = max_length
        tokens = self.tokenizer(list(sequences), **kwargs)
        tokens = {key: value.to(self.config.resolved_device()) for key, value in tokens.items()}
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        mask = tokens.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device))
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


class Evo2Adapter(FrozenAdapter):
    """Evo2 DNA adapter using an intermediate embedding layer."""

    def load(self) -> "Evo2Adapter":
        try:
            from evo2 import Evo2
        except ImportError as exc:
            raise ImportError("install evo2 in its supported CUDA/WSL environment") from exc
        self.model = Evo2(self.config.model_id)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return self

    @torch.no_grad()
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        if self.model is None:
            self.load()
        layer_name = self.config.extra.get("layer_name", "blocks.28.mlp.l3")
        outputs = []
        for sequence in sequences:
            device = self.config.resolved_device()
            token_ids = torch.tensor(
                self.model.tokenizer.tokenize(sequence), dtype=torch.int, device=device
            ).unsqueeze(0)
            _, embeddings = self.model(
                token_ids, return_embeddings=True, layer_names=[layer_name]
            )
            outputs.append(embeddings[layer_name].mean(dim=1).squeeze(0).cpu())
        return torch.stack(outputs)


class RNAFMAdapter(TransformersSequenceAdapter):
    """RNA-FM adapter using the upstream `fm.pretrained` API."""

    def load(self) -> "RNAFMAdapter":
        try:
            import fm
        except ImportError as exc:
            raise ImportError("install rna-fm or the RNA-FM repository package") from exc
        loader = getattr(fm.pretrained, self.config.model_id or "rna_fm_t12")
        self.model, self.alphabet = loader()
        self.model.to(self.config.resolved_device())
        self._freeze_model()
        return self

    @torch.no_grad()
    def encode(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        if self.model is None:
            self.load()
        converter = self.alphabet.get_batch_converter()
        batch = [(str(i), seq.upper().replace("T", "U")) for i, seq in enumerate(sequences)]
        _, _, tokens = converter(batch)
        tokens = tokens.to(self.config.resolved_device())
        result = self.model(tokens, repr_layers=[12])
        reps = result["representations"][12]
        padding_idx = getattr(self.alphabet, "padding_idx", 1)
        lengths = (tokens != padding_idx).sum(dim=1) - 2
        body = reps[:, 1:-1]
        mask = torch.arange(body.shape[1], device=body.device).unsqueeze(0) < lengths.unsqueeze(1)
        return (body * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)


class CompositeProteinAdapter(FrozenAdapter):
    """ESM-2 -> CLEF feature path, optionally concatenated with ProteomeLM."""

    def __init__(
        self,
        config: FoundationModelConfig,
        esm: ESM2Adapter,
        clef: Optional[CLEFCheckpointAdapter] = None,
        proteome: Optional[TransformersSequenceAdapter] = None,
    ):
        super().__init__(config)
        self.esm = esm
        self.clef = clef
        self.proteome = proteome

    def load(self) -> "CompositeProteinAdapter":
        self.esm.load()
        if self.clef is not None:
            self.clef.load()
        if self.proteome is not None:
            self.proteome.load()
        return self

    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        tokens, lengths = self.esm.encode_tokens(sequences)
        mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        pooled_esm = (tokens * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        outputs = [pooled_esm]
        if self.clef is not None:
            clef_output = self.clef.encode({"esm_feature": tokens, "valid_lens": lengths})
            outputs.append(clef_output)
        if self.proteome is not None:
            outputs.append(self.proteome.encode(sequences).to(pooled_esm.device))
        return torch.cat(outputs, dim=-1)


class CLEFCheckpointAdapter(FrozenAdapter):
    """Load an existing CLEF module and checkpoint without retraining CLEF."""

    def __init__(self, config: FoundationModelConfig, module_name: str, class_name: str):
        super().__init__(config)
        self.module_name = module_name
        self.class_name = class_name

    def load(self) -> "CLEFCheckpointAdapter":
        import importlib

        module = importlib.import_module(self.module_name)
        model_class = getattr(module, self.class_name)
        self.model = model_class(**self.config.extra.get("init_kwargs", {}))
        checkpoint = self.config.local_path
        if not checkpoint:
            raise ValueError("CLEFCheckpointAdapter requires a local pretrained checkpoint")
        from src_m1.clef_compat import load_clef_weights

        load_clef_weights(self.model, checkpoint)
        self.model.to(self.config.resolved_device())
        self._freeze_model()
        return self

    def encode(self, batch: object) -> torch.Tensor:
        if self.model is None:
            self.load()
        output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        return output


class FoundationFeatureStore:
    """Selects real adapters or cached UDC-02 embeddings by view."""

    def __init__(self, adapters: Mapping[str, FrozenAdapter]):
        self.adapters = dict(adapters)

    def encode(self, view: str, sequences: Sequence[str]) -> torch.Tensor:
        if view not in self.adapters:
            raise KeyError(f"no foundation adapter configured for view {view}")
        return self.adapters[view].encode(sequences)

    def load_all(self) -> None:
        for adapter in self.adapters.values():
            adapter.load()
