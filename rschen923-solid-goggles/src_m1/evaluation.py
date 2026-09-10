from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from src_m1.models import M1Encoder
from src_m1.training import make_synthetic_batch


def evaluate_m1() -> Dict[str, float]:
    model = M1Encoder(
        view_dims={"protein": 256, "dna": 192, "rna": 160, "omics": 128},
        latent_dim=128,
    )
    model.eval()

    features, target = make_synthetic_batch(batch_size=128)
    with torch.no_grad():
        representation = model(features)
        prediction = representation[:, 0]
        mse = float(((prediction - target) ** 2).mean().item())
        mean_abs_error = float((prediction - target).abs().mean().item())
        mean_target = float(target.mean().item())
        std_target = float(target.std(unbiased=False).item())

    return {
        "mse": mse,
        "mae": mean_abs_error,
        "target_mean": mean_target,
        "target_std": std_target,
        "embedding_shape": list(representation.shape),
    }


if __name__ == "__main__":
    print(np.array2string(np.asarray(list(evaluate_m1().items())), separator=", "))
