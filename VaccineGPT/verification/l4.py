"""Verify that low-confidence L4 records remain present but are down-weighted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from vaccinegpt.losses import multitask_loss
    from vaccinegpt.synthetic import make_synthetic_records
    from vaccinegpt.data import records_to_batch
    from vaccinegpt.models import VaccineGPTModel
except ModuleNotFoundError:
    from VaccineGPT.vaccinegpt.losses import multitask_loss
    from VaccineGPT.vaccinegpt.synthetic import make_synthetic_records
    from VaccineGPT.vaccinegpt.data import records_to_batch
    from VaccineGPT.vaccinegpt.models import VaccineGPTModel


def run() -> dict:
    records = make_synthetic_records()
    assert any(record["metadata"]["tier"] == "L4" for record in records)
    assert any(record["weight"] < 1 for record in records)
    views, labels, weights = records_to_batch(records)
    model = VaccineGPTModel({name: len(values[0]) for name, values in views.items()}, latent_dim=16, hidden_dim=16)
    outputs = model(views)
    total, losses = multitask_loss(outputs, labels, sample_weights=weights)
    assert torch.isfinite(total)
    assert set(losses) == {"T1", "T2", "T3a", "T3b", "T4"}
    return {"ok": True, "records": len(records), "l4_records": sum(record["weight"] < 1 for record in records)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
