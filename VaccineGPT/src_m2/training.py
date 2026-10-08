from __future__ import annotations

import json
import math
from typing import Dict

import torch
from torch import nn, optim

from shared.splits import assert_disjoint, grouped_split
from shared.lineage import lineage_tag
from src_m2.models import M2Predictor


def train_m2(num_epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5) -> Dict[str, object]:
    if num_epochs < 1 or batch_size < 2:
        raise ValueError("num_epochs must be positive and batch_size at least two")
    generator = torch.Generator().manual_seed(31)
    features = torch.randn(batch_size, 128, generator=generator)
    labels = {
        name: torch.randint(0, 2, (batch_size,), generator=generator).float()
        for name in ("H1", "H2", "H3", "H4")
    }
    model = M2Predictor()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = sum(
            nn.functional.binary_cross_entropy_with_logits(outputs[name], labels[name])
            for name in labels
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return {
        "mode": "architecture_smoke_only_synthetic",
        "epochs": num_epochs,
        "loss_history": history,
        "lineage": lineage_tag(
            model_version="VaccineGPT-v1-dev",
            feature_version="synthetic-latent-test",
            label_version="synthetic-binary-test",
            batch_id="SMOKE",
        ),
    }


def fit_supervised_heads(
    embeddings: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    evidence_levels: Dict[str, list[str]],
    group_ids: list[str],
    output_dir: str | Path,
    epochs: int = 20,
    learning_rate: float = 2e-5,
    batch_id: str = "UNSPECIFIED",
) -> dict[str, object]:
    """Fit only explicitly E1/E2 labels after a genome/group-disjoint split."""
    from pathlib import Path

    if embeddings.ndim != 2 or embeddings.shape[1] != 128:
        raise ValueError("embeddings must have shape [genes, 128]")
    count = embeddings.shape[0]
    if len(group_ids) != count:
        raise ValueError("group_ids must have one value per embedding")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not batch_id.strip() or batch_id.casefold() in {
        "unspecified",
        "unknown",
        "replace_with_batch_id",
        "template",
    }:
        raise ValueError("a real, immutable batch_id is required for M2 training")
    train_indices, valid_indices, test_indices = grouped_split(group_ids, seed=42)
    assert_disjoint(train_indices, valid_indices, test_indices)
    usable: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    dropped_weak_labels = 0
    head_data_gates: dict[str, dict[str, object]] = {}
    for head in ("H1", "H2", "H3", "H4"):
        if head not in labels or head not in evidence_levels:
            continue
        values = labels[head].to(dtype=torch.float32)
        levels = evidence_levels[head]
        if values.shape != (count,) or len(levels) != count:
            raise ValueError(f"{head} labels and evidence levels must align to embeddings")
        invalid_values = torch.isfinite(values) & (values != 0) & (values != 1)
        if invalid_values.any():
            raise ValueError(f"{head} labels must be binary values 0 or 1")
        eligible = torch.tensor(
            [level in {"E1", "E2"} for level in levels], dtype=torch.bool
        ) & torch.isfinite(values)
        dropped_weak_labels += int((torch.isfinite(values) & ~eligible).sum())
        if not eligible.any():
            continue
        train_values = values[train_indices][eligible[train_indices]]
        train_groups = [
            group_ids[index]
            for index in train_indices
            if bool(eligible[index])
        ]
        validation_values = values[valid_indices][eligible[valid_indices]]
        validation_groups = [
            group_ids[index]
            for index in valid_indices
            if bool(eligible[index])
        ]
        test_values = values[test_indices][eligible[test_indices]]
        test_groups = [
            group_ids[index]
            for index in test_indices
            if bool(eligible[index])
        ]

        def class_group_counts(split_values: torch.Tensor, split_groups: list[str]) -> dict[str, int]:
            return {
                str(label): len(
                    {
                        group
                        for value, group in zip(split_values.tolist(), split_groups)
                        if value == label
                    }
                )
                for label in (0.0, 1.0)
            }

        train_group_counts = class_group_counts(train_values, train_groups)
        validation_group_counts = class_group_counts(validation_values, validation_groups)
        test_group_counts = class_group_counts(test_values, test_groups)
        train_class_counts = {
            str(label): int((train_values == label).sum()) for label in (0.0, 1.0)
        }
        status = "trainable"
        if min(train_group_counts.values()) < 2:
            status = "insufficient_training_class_groups"
        elif min(validation_group_counts.values()) < 1:
            status = "insufficient_validation_classes"
        elif min(test_group_counts.values()) < 1:
            status = "insufficient_test_classes"
        head_data_gates[head] = {
            "status": status,
            "train_class_counts": train_class_counts,
            "train_independent_groups_per_class": train_group_counts,
            "validation_independent_groups_per_class": validation_group_counts,
            "test_independent_groups_per_class": test_group_counts,
            "calibration_gate_status": "blocked_no_independent_calibration_set",
            "independent_calibration_groups": 0,
            "probability_status": "uncalibrated_not_for_biological_probability_claims",
        }
        if status == "trainable":
            usable[head] = (values, eligible)
    if not usable:
        positive_only = [
            head
            for head, gate in head_data_gates.items()
            if gate["train_class_counts"]["0.0"] == 0
            and gate["train_class_counts"]["1.0"] > 0
        ]
        if positive_only:
            raise ValueError(
                "positive-only labels cannot train a binary probability head; "
                f"use candidate ranking/similarity for {positive_only}"
            )
        raise ValueError(
            "no E1/E2 labels or head passed the binary-class and independent-group gates; "
            "E3/E4 weak annotations are not accepted as gold supervision"
        )
    model = M2Predictor(enabled_heads=sorted(usable))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_idx = torch.as_tensor(train_indices, dtype=torch.long)
    valid_idx = torch.as_tensor(valid_indices, dtype=torch.long)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(embeddings)
        losses = []
        for head, (values, eligible) in usable.items():
            selected = eligible[train_idx]
            if selected.any():
                losses.append(
                    nn.functional.binary_cross_entropy_with_logits(
                        predictions[head][train_idx][selected], values[train_idx][selected]
                    )
                )
        if not losses:
            raise ValueError("the grouped training split contains no E1/E2 labels")
        loss = torch.stack(losses).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    metrics: dict[str, object] = {}
    model.eval()
    with torch.no_grad():
        predictions = model(embeddings)
        for head, (values, eligible) in usable.items():
            selected = eligible[valid_idx]
            metrics[head] = {
                "validation_n": int(selected.sum()),
                "validation_bce": (
                    float(
                        nn.functional.binary_cross_entropy_with_logits(
                            predictions[head][valid_idx][selected], values[valid_idx][selected]
                        )
                    )
                    if selected.any()
                    else None
                ),
            }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"m2": model.state_dict(), "lineage": lineage_tag(
            "VaccineGPT-v1-dev", "FEAT-E1E2-0.1.0", "LABEL-E1E2-0.1.0", batch_id
        )},
        output / "m2_e1e2_checkpoint.pt",
    )
    report = {
        "lineage": lineage_tag(
            "VaccineGPT-v1-dev", "FEAT-E1E2-0.1.0", "LABEL-E1E2-0.1.0", batch_id
        ),
        "n_genes": count,
        "n_groups": len(set(group_ids)),
        "split_sizes": [len(train_indices), len(valid_indices), len(test_indices)],
        "head_validation_metrics": metrics,
        "head_data_gates": head_data_gates,
        "trained_heads": sorted(usable),
        "probability_calibration": "not_performed; outputs are uncalibrated logits",
        "weak_labels_excluded_from_gold_training": dropped_weak_labels,
        "history": history,
        "test_indices_reserved": True,
        "calibration_status": "not_calibrated_by_this_training_function",
    }
    (output / "m2_e1e2_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(train_m2(), indent=2))
