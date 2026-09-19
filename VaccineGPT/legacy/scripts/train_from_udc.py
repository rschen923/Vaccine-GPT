from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.splits import assert_disjoint, grouped_split
from shared.udc_io import read_jsonl
from src_m1.models import M1Encoder
from src_m1.real_data import load_feature_views
from src_m2.metrics import mean_group_mrr, mean_group_ndcg, pairwise_ranking_loss
from src_m2.models import M2Predictor


def load_task_labels(path: str, genes: list[str]):
    wanted = set(genes)
    values, groups = {}, {}
    for record in read_jsonl(path, kind="label"):
        gene = record["gene_id"]
        if gene not in wanted:
            continue
        values.setdefault(gene, {})[record["task"]] = float(record["label"])
        groups[gene] = record.get("group_id", gene)
    tasks = {"T1", "T2", "T3b", "T4"}
    missing = [gene for gene in genes if not tasks.issubset(values.get(gene, {}))]
    if missing:
        raise ValueError(f"missing T1/T2/T3b/T4 labels for {len(missing)} genes")
    labels = {
        task: torch.tensor([values[gene][task] for gene in genes])
        for task in ("T2", "T3b", "T4")
    }
    labels["T1"] = torch.tensor([values[gene]["T1"] for gene in genes], dtype=torch.long)
    return labels, [groups.get(gene, gene) for gene in genes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M1 and M2 from validated UDC JSONL")
    parser.add_argument("--features", required=True, help="UDC-02 pooled feature JSONL")
    parser.add_argument("--labels", required=True, help="UDC-03 task label JSONL")
    parser.add_argument("--genes", required=True, help="JSON file containing ordered gene IDs")
    parser.add_argument("--track", choices=("SOM", "INT"), default="SOM")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", default="checkpoints/real_m1_m2.pt")
    parser.add_argument("--lineage", required=True, help="JSON lineage tuple")
    args = parser.parse_args()

    genes = json.loads(Path(args.genes).read_text(encoding="utf-8"))
    features = load_feature_views(args.features, genes, args.track)
    labels, groups = load_task_labels(args.labels, genes)
    train_idx, valid_idx, test_idx = grouped_split(groups, seed=42)
    assert_disjoint(train_idx, valid_idx, test_idx)
    view_dims = {view: int(value.shape[1]) for view, value in features.items()}
    m1 = M1Encoder(view_dims, latent_dim=128, track=args.track)
    m2 = M2Predictor(input_dim=128, hidden_dim=64)
    optimizer = torch.optim.AdamW(
        list(m1.parameters()) + list(m2.parameters()), lr=1e-3, weight_decay=1e-4
    )
    history = []
    for _ in range(args.epochs):
        m1.train(); m2.train(); optimizer.zero_grad()
        outputs = m2(m1(features))
        loss = (
            torch.nn.functional.cross_entropy(outputs["T1"][train_idx], labels["T1"][train_idx])
            + torch.nn.functional.binary_cross_entropy_with_logits(outputs["T2"][train_idx], labels["T2"][train_idx])
            + torch.nn.functional.mse_loss(torch.sigmoid(outputs["T3"][train_idx]), labels["T3b"][train_idx])
            + pairwise_ranking_loss(outputs["T4"][train_idx], labels["T4"][train_idx])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(m1.parameters()) + list(m2.parameters()), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    m1.eval(); m2.eval()
    with torch.no_grad():
        outputs = m2(m1(features))
        predicted = outputs["T1"].argmax(dim=-1)
        accuracy = float((predicted[test_idx] == labels["T1"][test_idx]).float().mean())
        ranking_scores = outputs["T4"][test_idx].cpu().numpy()
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    report = {
        "track": args.track,
        "genes": len(genes),
        "split_sizes": [len(train_idx), len(valid_idx), len(test_idx)],
        "test_t1_accuracy": accuracy,
        "test_t4_ndcg_at_10": mean_group_ndcg(
            ranking_scores, labels["T4"][test_idx].numpy(),
            [groups[i] for i in test_idx], k=10,
        ),
        "test_t4_mrr": mean_group_mrr(
            ranking_scores, labels["T4"][test_idx].numpy(),
            [groups[i] for i in test_idx],
        ),
        "history": history,
        "lineage": lineage,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"m1": m1.state_dict(), "m2": m2.state_dict(), "lineage": lineage}, output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
