from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class SpeciesAdapter(nn.Module):
    """Identity-initialized, low-capacity adapter for later species-specific tuning."""

    def __init__(self, input_dim: int = 128, mode: str = "scale", rank: int = 8):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if mode == "scale":
            self.scale = nn.Parameter(torch.zeros(input_dim))
            self.down = None
            self.up = None
        elif mode == "lora":
            if rank < 1 or rank >= input_dim:
                raise ValueError("LoRA rank must be positive and smaller than input_dim")
            self.register_parameter("scale", None)
            self.down = nn.Linear(input_dim, rank, bias=False)
            self.up = nn.Linear(rank, input_dim, bias=False)
            nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
            nn.init.zeros_(self.up.weight)
        else:
            raise ValueError("adapter mode must be 'scale' or 'lora'")
        self.input_dim = input_dim
        self.mode = mode

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"adapter input must have shape [batch, {self.input_dim}]")
        if self.mode == "scale":
            return values * (1.0 + self.scale)
        return values + self.up(self.down(values))


def freeze_for_species_adaptation(model: nn.Module) -> None:
    """Freeze the shared model and leave only an explicitly attached adapter trainable."""
    adapter = getattr(model, "species_adapter", None)
    if adapter is None:
        raise ValueError("model has no species_adapter; attach one before adaptation")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in adapter.parameters():
        parameter.requires_grad = True


class TaskHead(nn.Module):
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(-1)


class M2Predictor(nn.Module):
    """Shared-space H1-H4 heads; H5 is enabled only for the v2 DNA track."""

    def __init__(
        self,
        input_dim: int = 128,
        enable_h5: bool = False,
        enabled_heads: list[str] | None = None,
        species_adapter_mode: str | None = None,
        species_adapter_rank: int = 8,
    ):
        super().__init__()
        requested_heads = ("H1", "H2", "H3", "H4") if enabled_heads is None else tuple(enabled_heads)
        if not requested_heads or len(set(requested_heads)) != len(requested_heads):
            raise ValueError("enabled_heads must contain unique task names")
        unsupported = set(requested_heads) - {"H1", "H2", "H3", "H4"}
        if unsupported:
            raise ValueError(f"unsupported M2 heads: {sorted(unsupported)}")
        self.species_adapter = (
            SpeciesAdapter(input_dim, species_adapter_mode, species_adapter_rank)
            if species_adapter_mode is not None
            else None
        )
        self.task_heads = nn.ModuleDict(
            {name: TaskHead(input_dim) for name in requested_heads}
        )
        self.h5 = TaskHead(input_dim) if enable_h5 else None

    def forward(self, h_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        if h_final.ndim != 2:
            raise ValueError("h_final must have shape [genes, latent_dimension]")
        if self.species_adapter is not None:
            h_final = self.species_adapter(h_final)
        outputs = {name: head(h_final) for name, head in self.task_heads.items()}
        if self.h5 is not None:
            outputs["H5"] = self.h5(h_final)
        return outputs

    def immune_profile(self, h_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(h_final)
        return {"H2_antigen_score": torch.sigmoid(outputs["H2"])}
