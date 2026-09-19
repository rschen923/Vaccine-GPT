"""Contract and signal-usage checks for the authoritative L0-L4 path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaccinegpt.five_layer import FiveLayerVaccineGPT
from vaccinegpt.spec_training import make_synthetic_contract_batch


def run() -> dict[str, object]:
    batch = make_synthetic_contract_batch(batch_size=6)
    model = FiveLayerVaccineGPT()
    output = model(batch)
    losses = model.losses(batch, output)
    assert output["t4"].shape == (6,)
    assert all(torch.isfinite(value).all() for value in losses.values())

    # Graph and physical inputs must affect the forward path.
    altered = {key: value.clone() for key, value in batch.items()}
    altered["node_x"] = altered["node_x"] + 1.0
    altered_output = model(altered)
    assert not torch.allclose(output["h_shared"], altered_output["h_shared"])
    return {
        "ok": True,
        "required_fields": len(batch),
        "loss_terms": len(losses),
        "graph_signal_used": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
