from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts import validate_feature
from shared.lineage import lineage_tag
from shared.splits import assert_disjoint, grouped_split
from src_m1.models import M1Encoder
from src_m2.metrics import mean_group_mrr, mean_group_ndcg, pairwise_ranking_loss
from src_m2.models import M2Predictor


def main(output_dir: str = "artifacts") -> dict:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    n = 96
    groups = [f"operon-{i // 3}" for i in range(n)]
    train_idx, valid_idx, test_idx = grouped_split(groups, seed=42)
    assert_disjoint(train_idx, valid_idx, test_idx)
    lineage = lineage_tag("M1M2-GIC-SDE-0.1.0", "FEAT-0.1.0", "LABEL-0.1.0", "B2026-09")

    features = {
        "protein": torch.tensor(rng.normal(size=(n, 256)), dtype=torch.float32),
        "dna": torch.tensor(rng.normal(size=(n, 192)), dtype=torch.float32),
        "rna": torch.tensor(rng.normal(size=(n, 160)), dtype=torch.float32),
        "omics": torch.tensor(rng.normal(size=(n, 128)), dtype=torch.float32),
    }
    validate_feature({
        "gene_id": "synthetic-0", "view": "protein", "track": "INT",
        "embedding": [0.0] * 256, "lineage": lineage,
    })
    m1 = M1Encoder(
        {"protein": 256, "dna": 192, "rna": 160, "omics": 128}, latent_dim=128
    )
    m2 = M2Predictor(input_dim=128, hidden_dim=64)
    optimizer = torch.optim.AdamW(list(m1.parameters()) + list(m2.parameters()), lr=1e-3)
    target_t1 = torch.tensor(rng.integers(0, 3, n), dtype=torch.long)
    target_t2 = torch.tensor(rng.integers(0, 2, n), dtype=torch.float32)
    target_t3 = torch.tensor(rng.random(n), dtype=torch.float32)
    target_t4 = target_t1.float() + target_t2 + target_t3
    history = []
    for _ in range(8):
        m1.train(); m2.train(); optimizer.zero_grad()
        h_final = m1(features)
        prediction = m2(h_final)
        loss = (
            torch.nn.functional.cross_entropy(prediction["T1"][train_idx], target_t1[train_idx])
            + torch.nn.functional.binary_cross_entropy_with_logits(prediction["T2"][train_idx], target_t2[train_idx])
            + torch.nn.functional.mse_loss(torch.sigmoid(prediction["T3"][train_idx]), target_t3[train_idx])
            + pairwise_ranking_loss(prediction["T4"][train_idx], target_t4[train_idx])
        )
        loss.backward(); torch.nn.utils.clip_grad_norm_(m1.parameters(), 1.0); optimizer.step()
        history.append(float(loss.detach()))

    m1.eval(); m2.eval()
    with torch.no_grad():
        outputs = m2(m1(features))
        valid_t1 = torch.argmax(outputs["T1"][valid_idx], dim=-1)
        test_t1 = torch.argmax(outputs["T1"][test_idx], dim=-1)
        valid_acc = float((valid_t1 == target_t1[valid_idx]).float().mean())
        test_acc = float((test_t1 == target_t1[test_idx]).float().mean())
        test_rank = float(torch.corrcoef(torch.stack((outputs["T4"][test_idx], target_t4[test_idx])))[0, 1])
    result = {
        "lineage": lineage, "split_sizes": [len(train_idx), len(valid_idx), len(test_idx)],
        "history": history, "valid_t1_accuracy": valid_acc,
        "test_t1_accuracy": test_acc,         "test_t4_pearson_proxy": test_rank,
        "test_t4_ndcg_at_10": mean_group_ndcg(
            outputs["T4"][test_idx].numpy(), target_t4[test_idx].numpy(),
            [groups[i] for i in test_idx], k=10,
        ),
        "test_t4_mrr": mean_group_mrr(
            outputs["T4"][test_idx].numpy(), target_t4[test_idx].numpy(),
            [groups[i] for i in test_idx],
        ),
        "h_final_shape": list(m1(features).shape),
    }
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    (path / "end_to_end_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"m1": m1.state_dict(), "m2": m2.state_dict(), "lineage": lineage}, path / "m1_m2_checkpoint.pt")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
