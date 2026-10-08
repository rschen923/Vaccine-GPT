from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_m1.models import M1Encoder
from src_m1.training import make_synthetic_batch
from src_m2.models import M2Predictor


def main(output_dir: str = "artifacts") -> dict:
    torch.manual_seed(42)
    residues, mask = make_synthetic_batch(batch_size=8, length=64)
    m1 = M1Encoder(residue_dim=20)
    m2 = M2Predictor()
    m1.eval()
    m2.eval()
    with torch.no_grad():
        representation = m1(residues, mask)
        predictions = m2(representation["z_gctx"])
    result = {
        "mode": "architecture_smoke_only_synthetic",
        "input_contract": "protein residue embeddings only; v1 does not accept DNA sequence",
        "m1_shapes": {
            name: list(representation[name].shape)
            for name in ("z_pub", "z_exp", "z_gctx", "position_scores")
        },
        "m2_shapes": {name: list(value.shape) for name, value in predictions.items()},
        "private_features_available": int(representation["private_available"].sum()),
        "trained_checkpoint_loaded": False,
        "biological_metrics": None,
        "interpretation": "This is a tensor-contract smoke test, not evidence of biological performance.",
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "end_to_end_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
