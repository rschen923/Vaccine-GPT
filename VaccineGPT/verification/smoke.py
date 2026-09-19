"""Smoke verification for JSONL -> model -> PCGrad training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from vaccinegpt.data import records_to_batch, validate_jsonl
    from vaccinegpt.models import VaccineGPTModel
    from vaccinegpt.synthetic import SyntheticConfig, generate_synthetic_jsonl, make_synthetic_records
    from vaccinegpt.training import TrainingConfig, train_records
except ModuleNotFoundError:
    from VaccineGPT.vaccinegpt.data import records_to_batch, validate_jsonl
    from VaccineGPT.vaccinegpt.models import VaccineGPTModel
    from VaccineGPT.vaccinegpt.synthetic import SyntheticConfig, generate_synthetic_jsonl, make_synthetic_records
    from VaccineGPT.vaccinegpt.training import TrainingConfig, train_records


def run() -> dict:
    records = make_synthetic_records(SyntheticConfig(records=12, seed=11))
    model, history = train_records(records, TrainingConfig(epochs=2, latent_dim=16, hidden_dim=16))
    views, _, _ = records_to_batch(records[:4])
    with torch.no_grad():
        outputs = model(views)
    assert outputs["T1"].shape == (4, 3)
    assert outputs["T4"].shape == (4,)
    assert all(torch.isfinite(value).all() for value in outputs.values() if isinstance(value, torch.Tensor))
    return {"ok": True, "records": len(records), "history": history}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        generate_synthetic_jsonl(args.output)
        result["jsonl"] = validate_jsonl(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
