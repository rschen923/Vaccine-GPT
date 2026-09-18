"""Tensor-level contracts for the authoritative five-layer training schema."""

from __future__ import annotations

from collections.abc import Mapping

import torch


REQUIRED_FIELDS = (
    "x_a", "x_b", "S_adj", "pcd_A", "ch_covariates", "fba", "y3",
    "y_vivo", "y_vitro", "PPI_adj", "node_x", "topo_feat", "y_sec",
    "domain", "pu_pos_idx", "pu_unl_idx", "N_p", "N_star", "hla_freq",
    "pep", "pseudo", "pep_mask", "ic50", "el", "t4_feat", "t4_rel",
    "R0", "v0", "t_obs",
)


def validate_batch_contract(batch: Mapping[str, torch.Tensor]) -> None:
    """Validate required tensors and their invariant leading dimensions.

    This function intentionally reports every missing field rather than
    silently inventing measurements. Synthetic generation is separate.
    """
    missing = [name for name in REQUIRED_FIELDS if name not in batch]
    if missing:
        raise KeyError(f"missing required batch fields: {', '.join(missing)}")
    for name in REQUIRED_FIELDS:
        if not isinstance(batch[name], torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.isfinite(batch[name].float()).all():
            raise ValueError(f"{name} contains non-finite values")
    batch_size = batch["x_a"].shape[0] if batch["x_a"].ndim else None
    if batch_size is None:
        raise ValueError("x_a must have a batch dimension")
    batch_fields = (
        "x_b", "S_adj", "pcd_A", "ch_covariates", "fba", "y3", "y_vivo",
        "y_vitro", "topo_feat", "y_sec", "domain", "N_p", "pep", "pseudo",
        "pep_mask", "ic50", "el", "t4_feat", "t4_rel", "R0", "v0", "t_obs",
    )
    for name in batch_fields:
        if batch[name].shape[0] != batch_size:
            raise ValueError(f"{name} must share batch dimension B={batch_size}")
    if batch["PPI_adj"].ndim != 2 or batch["PPI_adj"].shape != (batch_size, batch_size):
        raise ValueError("PPI_adj must have shape (B, B)")
    if batch["node_x"].ndim != 2 or batch["node_x"].shape[0] != batch_size:
        raise ValueError("node_x must have shape (B, node_features)")
    if batch["x_a"].ndim != 3 or batch["x_a"].shape[-1] != 64:
        raise ValueError("x_a must have shape (B, L, 64)")
    if batch["x_b"].ndim != 2 or batch["x_b"].shape[-1] != 32:
        raise ValueError("x_b must have shape (B, 32)")
    if batch["S_adj"].ndim != 1 or batch["pcd_A"].shape[-1] != 4:
        raise ValueError("S_adj must be (B,) and pcd_A must end in 4")
    if batch["ch_covariates"].shape[-1] != 16:
        raise ValueError("ch_covariates must end in 16")
    if batch["topo_feat"].shape[-1] != 4 or batch["t4_feat"].shape[-1] != 4:
        raise ValueError("topo_feat and t4_feat must end in 4")
    if "task_mask" in batch:
        if batch["task_mask"].shape != (batch_size, 4):
            raise ValueError("task_mask must have shape (B, 4) for T1, T2, T3, and T4")
        if not torch.all((batch["task_mask"] >= 0) & (batch["task_mask"] <= 1)):
            raise ValueError("task_mask values must be in [0, 1]")


def derive_required_fields(record: Mapping[str, object]) -> dict[str, object]:
    """Return explicitly supplied canonical fields, rejecting missing inputs.

    Dataset adapters may call this after deriving fields from documented raw
    measurements. It never fabricates a missing biological measurement.
    """
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise KeyError(f"record is missing required fields: {', '.join(missing)}")
    return {name: record[name] for name in REQUIRED_FIELDS}
