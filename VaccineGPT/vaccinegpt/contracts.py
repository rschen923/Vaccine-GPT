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
    if batch["x_a"].ndim != 3 or batch["x_a"].shape[-1] != 64:
        raise ValueError("x_a must have shape (B, L, 64)")
    if batch["x_b"].ndim != 2 or batch["x_b"].shape[-1] != 32:
        raise ValueError("x_b must have shape (B, 32)")
    if batch["S_adj"].ndim != 1 or batch["pcd_A"].shape[-1] != 4:
        raise ValueError("S_adj must be (B,) and pcd_A must end in 4")
    if batch["ch_covariates"].shape[-1] != 16:
        raise ValueError("ch_covariates must end in 16")


def derive_required_fields(record: Mapping[str, object]) -> dict[str, object]:
    """Return explicitly supplied canonical fields, rejecting missing inputs.

    Dataset adapters may call this after deriving fields from documented raw
    measurements. It never fabricates a missing biological measurement.
    """
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise KeyError(f"record is missing required fields: {', '.join(missing)}")
    return {name: record[name] for name in REQUIRED_FIELDS}
