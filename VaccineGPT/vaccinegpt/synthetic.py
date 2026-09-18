"""Deterministic synthetic records used by smoke and verification commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_DIMS, write_jsonl


@dataclass(frozen=True)
class SyntheticConfig:
    records: int = 24
    seed: int = 7
    dims: dict[str, int] | None = None


def make_synthetic_records(config: SyntheticConfig = SyntheticConfig()) -> list[dict]:
    dims = dict(config.dims or DEFAULT_DIMS)
    rng = np.random.default_rng(config.seed)
    records = []
    for index in range(config.records):
        latent = rng.normal(size=8)
        views = {
            name: rng.normal(0, 1, size=dim).round(6).tolist()
            for name, dim in dims.items()
        }
        signal = float(latent.mean())
        records.append(
            {
                "sample_id": f"synthetic-{index:04d}",
                "group_id": f"strain-{index % max(3, config.records // 4):02d}",
                "views": views,
                "labels": {
                    "T1": int(np.digitize(signal, (-0.25, 0.25))),
                    "T2": float(signal > 0),
                    "T3a": float(signal > -0.15),
                    "T3b": float(np.clip(0.5 + signal / 3, 0, 1)),
                    "T4": float(np.clip(0.5 + signal / 3, 0, 1)),
                },
                "weight": 0.25 if index % 7 == 0 else 1.0,
                "metadata": {"tier": "L4" if index % 7 == 0 else "L1"},
            }
        )
    return records


def generate_synthetic_jsonl(path: str | Path, config: SyntheticConfig = SyntheticConfig()) -> int:
    return write_jsonl(make_synthetic_records(config), path)


def make_synthetic_batch(
    batch_size: int = 8,
    *,
    dims: dict[str, int] | None = None,
    seed: int = 7,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    records = make_synthetic_records(SyntheticConfig(batch_size, seed, dims))
    from .data import records_to_batch

    return records_to_batch(records)

