from __future__ import annotations

from typing import Dict

import torch

from src_m1.models import M1Encoder
from src_m1.training import make_synthetic_batch


def evaluate_m1() -> Dict[str, object]:
    model = M1Encoder(residue_dim=20)
    model.eval()
    embeddings, mask = make_synthetic_batch(batch_size=8)
    with torch.no_grad():
        outputs = model(embeddings, mask)
    return {
        "mode": "architecture_smoke_only_synthetic",
        "output_shapes": {
            name: list(outputs[name].shape)
            for name in ("z_pub", "z_exp", "z_gctx", "position_scores")
        },
        "private_available_count": int(outputs["private_available"].sum()),
        "biological_metrics": None,
    }
