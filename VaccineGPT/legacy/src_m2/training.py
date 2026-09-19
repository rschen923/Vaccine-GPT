from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim

from shared.lineage import lineage_tag
from src_m2.models import M2Predictor


def make_synthetic_task_data(batch_size: int = 64) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    rng = np.random.default_rng(31)
    base = rng.normal(0.0, 1.0, size=(batch_size, 128)).astype(np.float32)
    target_t1 = np.digitize(base[:, :10].mean(axis=1), bins=(-0.25, 0.25)).astype(np.float32)
    target_t2 = (base[:, 10:20].mean(axis=1) > 0.1).astype(np.float32)
    target_t3 = (base[:, 20:30].mean(axis=1) > -0.1).astype(np.float32)
    target_t4 = (base[:, 30:40].mean(axis=1) + 0.5).clip(0.0, 1.0).astype(np.float32)

    labels = {
        "T1": torch.tensor(target_t1, dtype=torch.long),
        "T2": torch.tensor(target_t2, dtype=torch.float32),
        "T3": torch.tensor(target_t3, dtype=torch.float32),
        "T4": torch.tensor(target_t4, dtype=torch.float32),
    }
    return torch.tensor(base, dtype=torch.float32), labels


def train_m2(num_epochs: int = 12, batch_size: int = 64, learning_rate: float = 1e-3) -> Dict[str, object]:
    model = M2Predictor(input_dim=128, hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(num_epochs):
        model.train()
        h_final, labels = make_synthetic_task_data(batch_size=batch_size)
        optimizer.zero_grad()

        predictions = model(h_final)
        t1_loss = nn.functional.cross_entropy(
            predictions["T1"], labels["T1"]
        )
        t2_loss = nn.functional.binary_cross_entropy_with_logits(predictions["T2"], labels["T2"])
        t3_loss = nn.functional.binary_cross_entropy_with_logits(predictions["T3"], labels["T3"])
        t4_loss = nn.functional.mse_loss(predictions["T4"], labels["T4"])
        loss = t1_loss + t2_loss + t3_loss + t4_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().numpy()))

    final_metrics = {
        "mode": "m2_smoke",
        "epochs": num_epochs,
        "final_loss": float(history[-1]),
        "history": history,
        "lineage": lineage_tag(
            model_version="M1M2-GIC-SDE-0.1.0",
            feat_version="FEAT-0.1.0",
            label_version="LABEL-0.1.0",
            batch_id="B2026-09",
        ),
    }
    return final_metrics


if __name__ == "__main__":
    print(json.dumps(train_m2(), indent=2))
