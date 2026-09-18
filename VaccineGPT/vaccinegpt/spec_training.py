"""Training entry points for the authoritative five-layer tensor contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .contracts import REQUIRED_FIELDS, validate_batch_contract
from .five_layer import FiveLayerVaccineGPT


def load_contract_jsonl(path: str | Path) -> dict[str, torch.Tensor]:
    """Load exactly one JSONL batch object, preserving biological tensor shapes."""
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError("--data must contain exactly one batch object per file; shard data before invoking training")
    record = records[0]
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise KeyError(f"training batch lacks required fields: {', '.join(missing)}")
    batch = {field: torch.as_tensor(record[field]) for field in REQUIRED_FIELDS}
    validate_batch_contract(batch)
    return batch


def make_synthetic_contract_batch(batch_size: int = 8, sequence_length: int = 12, seed: int = 7) -> dict[str, torch.Tensor]:
    """Make synthetic tensors only for software validation, never biological evaluation."""
    generator = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, generator=generator)
    x_a = rand(batch_size, sequence_length, 64)
    return {
        "x_a": x_a, "x_b": rand(batch_size, 32), "S_adj": rand(batch_size),
        "pcd_A": rand(batch_size, 4), "ch_covariates": rand(batch_size, 16),
        "fba": rand(batch_size), "y3": rand(batch_size), "y_vivo": torch.randint(0, 2, (batch_size,), generator=generator),
        "y_vitro": torch.randint(0, 2, (batch_size,), generator=generator), "PPI_adj": torch.eye(batch_size),
        "node_x": rand(batch_size, 8), "topo_feat": rand(batch_size, 4), "y_sec": rand(batch_size),
        "domain": torch.randint(0, 2, (batch_size,), generator=generator), "pu_pos_idx": torch.arange(batch_size // 2),
        "pu_unl_idx": torch.arange(batch_size // 2, batch_size), "N_p": rand(batch_size) + 0.1,
        "N_star": rand(3) + 0.1, "hla_freq": torch.softmax(rand(6), 0), "pep": rand(batch_size, 64),
        "pseudo": rand(batch_size, 34), "pep_mask": torch.ones(batch_size), "ic50": rand(batch_size),
        "el": torch.randint(0, 2, (batch_size,), generator=generator), "t4_feat": rand(batch_size, 4),
        "t4_rel": rand(batch_size), "R0": rand(batch_size), "v0": rand(batch_size), "t_obs": rand(batch_size),
    }


def train_contract_batch(batch: dict[str, torch.Tensor], epochs: int = 8, stage_epochs: int = 1) -> tuple[FiveLayerVaccineGPT, list[dict[str, Any]]]:
    """Train the real contract model; stage weights prevent premature task coupling."""
    validate_batch_contract(batch)
    model = FiveLayerVaccineGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    stages = ("l0", "l1", "l2", "l3", "l4", "joint")
    history = []
    layer_keys = {
        "l0": ("info_nce", "dann", "vib_l1"), "l1": ("ode_l1", "softmax_l1", "markov_l1", "t1"),
        "l2": ("ode_l2", "topology_l2", "iron_l2", "t2"), "l3": ("powerlaw_l3", "lv_l3", "hill_l3", "t3"),
        "l4": ("sir_l4", "hla_l4", "evol_l4", "t4", "l4_ai", "alpha_prior"),
    }
    for epoch in range(epochs):
        stage = stages[min(epoch // max(stage_epochs, 1), len(stages) - 1)]
        output = model(batch)
        losses = model.losses(batch, output)
        selected = tuple(losses) if stage == "joint" else layer_keys[stage]
        total = sum(losses[key] for key in selected)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "stage": stage, "loss": float(total.detach())})
    return model, history
