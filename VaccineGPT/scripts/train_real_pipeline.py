from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shared.lineage import lineage_tag
from shared.splits import assert_disjoint, grouped_split
from src_m1.models import M1Encoder
from src_m1.real_data import load_feature_views
from src_m2.metrics import pairwise_ranking_loss
from src_m2.models import M2Predictor
from src_m2.real_data import load_gene_groups, load_label_tiers, load_task_labels
from shared.dataset import balanced_sample_weights


def main() -> dict:
    parser = argparse.ArgumentParser(description="Train M1/M2 from UDC-02 and UDC-03 JSONL artifacts")
    parser.add_argument("--features-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--groups-jsonl", required=True)
    parser.add_argument("--track", choices=("SOM", "INT"), default="SOM")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="artifacts/real_run")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()

    label_records = [
        json.loads(line) for line in Path(args.labels_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gene_ids = sorted({record["gene_id"] for record in label_records})
    groups = load_gene_groups(args.groups_jsonl, gene_ids)
    train_idx, valid_idx, test_idx = grouped_split(groups, seed=42)
    assert_disjoint(train_idx, valid_idx, test_idx)
    features = load_feature_views(args.features_jsonl, gene_ids, args.track)
    labels = {
        "T1": load_task_labels(args.labels_jsonl, gene_ids, "T1"),
        "T2": load_task_labels(args.labels_jsonl, gene_ids, "T2"),
        "T3": load_task_labels(args.labels_jsonl, gene_ids, "T3b"),
        "T4": load_task_labels(args.labels_jsonl, gene_ids, "T4"),
    }
    sample_weights = torch.tensor(
        balanced_sample_weights(
            labels["T1"].tolist(),
            load_label_tiers(args.labels_jsonl, gene_ids, "T1"),
        ),
        dtype=torch.float32,
    )
    model = M1Encoder({view: tensor.shape[1] for view, tensor in features.items()}, track=args.track)
    predictor = M2Predictor()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()), lr=args.learning_rate,
    )
    history = []
    for _ in range(args.epochs):
        model.train(); predictor.train(); optimizer.zero_grad()
        outputs = predictor(model(features))
        loss = (
            torch.nn.functional.cross_entropy(
                outputs["T1"][train_idx], labels["T1"][train_idx],
                reduction="none",
            ).mul(sample_weights[train_idx]).mean()
            + torch.nn.functional.binary_cross_entropy_with_logits(outputs["T2"][train_idx], labels["T2"][train_idx])
            + torch.nn.functional.mse_loss(torch.sigmoid(outputs["T3"][train_idx]), labels["T3"][train_idx])
            + pairwise_ranking_loss(outputs["T4"][train_idx], labels["T4"][train_idx])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lineage = lineage_tag("M1M2-GIC-SDE-0.2.0", "FEAT-EXT-0.1.0", "LABEL-REAL-0.1.0", args.batch_id)
    torch.save({"m1": model.state_dict(), "m2": predictor.state_dict(), "lineage": lineage}, output / "checkpoint.pt")
    report = {
        "lineage": lineage,
        "track": args.track,
        "genes": len(gene_ids),
        "split_sizes": [len(train_idx), len(valid_idx), len(test_idx)],
        "history": history,
        "note": "Run on supplied UDC records; this report is not generated from synthetic data.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
