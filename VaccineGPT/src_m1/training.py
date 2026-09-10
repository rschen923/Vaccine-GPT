from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim

from shared.lineage import lineage_tag
from src_m1.models import M1Encoder


def make_synthetic_batch(batch_size: int = 64) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    rng = np.random.default_rng(17)
    protein = rng.normal(0.0, 1.0, size=(batch_size, 256)).astype(np.float32)
    dna = rng.normal(0.0, 1.0, size=(batch_size, 192)).astype(np.float32)
    rna = rng.normal(0.0, 1.0, size=(batch_size, 160)).astype(np.float32)
    omics = rng.normal(0.0, 1.0, size=(batch_size, 128)).astype(np.float32)

    target_raw = (
        protein[:, :12].mean(axis=1)
        + dna[:, :12].mean(axis=1)
        + rna[:, :12].mean(axis=1)
        + omics[:, :12].mean(axis=1)
    ) / 4.0
    target = torch.tensor(target_raw, dtype=torch.float32)

    features = {
        "protein": torch.tensor(protein, dtype=torch.float32),
        "dna": torch.tensor(dna, dtype=torch.float32),
        "rna": torch.tensor(rna, dtype=torch.float32),
        "omics": torch.tensor(omics, dtype=torch.float32),
    }
    return features, target


def train_m1(num_epochs: int = 12, batch_size: int = 64, learning_rate: float = 1e-3) -> Dict[str, object]:
    model = M1Encoder(
        view_dims={"protein": 256, "dna": 192, "rna": 160, "omics": 128},
        latent_dim=128,
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(num_epochs):
        model.train()
        features, target = make_synthetic_batch(batch_size=batch_size)

        optimizer.zero_grad()
        repr_vector = model(features)
        prediction = repr_vector[:, 0]
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().numpy()))

    final_metrics = {
        "mode": "m1_smoke",
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
    print(json.dumps(train_m1(), indent=2))
