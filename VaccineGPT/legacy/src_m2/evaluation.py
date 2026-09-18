from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from src_m2.models import M2Predictor
from src_m2.training import make_synthetic_task_data


def evaluate_m2() -> Dict[str, float]:
    model = M2Predictor(input_dim=128, hidden_dim=64)
    model.eval()

    h_final, labels = make_synthetic_task_data(batch_size=128)
    with torch.no_grad():
        predictions = model(h_final)
        metrics = {
            "T1_cross_entropy": float(torch.nn.functional.cross_entropy(predictions["T1"], labels["T1"]).item()),
            "T2_bce": float(torch.nn.functional.binary_cross_entropy_with_logits(predictions["T2"], labels["T2"]).item()),
            "T3_bce": float(torch.nn.functional.binary_cross_entropy_with_logits(predictions["T3"], labels["T3"]).item()),
            "T4_mse": float(torch.nn.functional.mse_loss(predictions["T4"], labels["T4"]).item()),
            "output_shapes": {name: list(value.shape) for name, value in predictions.items()},
        }
    return metrics


if __name__ == "__main__":
    print(np.array2string(np.asarray(list(evaluate_m2().items())), separator=", "))
