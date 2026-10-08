from __future__ import annotations

import math
from typing import Dict

import torch

from src_m2.models import M2Predictor


def evaluate_m2() -> Dict[str, object]:
    torch.manual_seed(37)
    features = torch.randn(32, 128)
    model = M2Predictor().eval()
    with torch.no_grad():
        predictions = model(features)
    return {
        "mode": "architecture_smoke_only_untrained",
        "output_shapes": {name: list(value.shape) for name, value in predictions.items()},
        "biological_metrics": None,
    }


def evaluate_supervised_heads(
    logits: Dict[str, list[float]],
    labels: Dict[str, list[float]],
    evidence_levels: Dict[str, list[str]],
    group_ids: list[str],
    threshold: float = 0.5,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate a held-out split using only E1/E2 labels and group bootstrap CIs."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not group_ids or any(not str(group) or str(group) == "unknown" for group in group_ids):
        raise ValueError("held-out evaluation requires known group IDs")
    rng = np.random.default_rng(seed)
    result: dict[str, object] = {
        "evidence_policy": "E1/E2 only; E3/E4 excluded from gold evaluation",
        "threshold": threshold,
        "bootstrap_unit": "group",
        "heads": {},
    }
    for head, raw_logits in logits.items():
        if head not in labels or head not in evidence_levels:
            continue
        if not (len(raw_logits) == len(labels[head]) == len(evidence_levels[head]) == len(group_ids)):
            raise ValueError(f"{head}: logits/labels/evidence/group lengths do not match")
        raw = np.asarray(raw_logits, dtype=np.float64)
        observed = np.asarray(labels[head], dtype=np.float64)
        eligible = np.isin(np.asarray(evidence_levels[head]), ["E1", "E2"])
        eligible &= np.isfinite(raw) & np.isfinite(observed)
        raw, observed = raw[eligible], observed[eligible].astype(int)
        groups = np.asarray(group_ids, dtype=object)[eligible]
        if not len(observed):
            result["heads"][head] = {"status": "no_E1_E2_heldout_labels", "n": 0}
            continue
        if not set(np.unique(observed)).issubset({0, 1}):
            raise ValueError(f"{head}: observed labels must be binary")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
        predictions = probabilities >= threshold
        matrix = confusion_matrix(observed, predictions, labels=[0, 1])
        base = {
            "n": int(len(observed)),
            "n_groups": int(len(set(groups))),
            "roc_auc": float(roc_auc_score(observed, probabilities)) if len(set(observed)) == 2 else None,
            "pr_auc": float(average_precision_score(observed, probabilities)) if len(set(observed)) == 2 else None,
            "balanced_accuracy": float(balanced_accuracy_score(observed, predictions)),
            "brier": float(brier_score_loss(observed, probabilities)),
            "confusion_matrix_labels_0_1": matrix.tolist(),
        }
        group_values = sorted(set(groups))
        boot = {name: [] for name in ("roc_auc", "pr_auc", "balanced_accuracy", "brier")}
        for _ in range(bootstrap_samples):
            draw = rng.choice(group_values, size=len(group_values), replace=True)
            indices = np.concatenate([np.flatnonzero(groups == group) for group in draw])
            target = observed[indices]
            score = probabilities[indices]
            if len(set(target)) == 2:
                boot["roc_auc"].append(float(roc_auc_score(target, score)))
                boot["pr_auc"].append(float(average_precision_score(target, score)))
            boot["balanced_accuracy"].append(
                float(balanced_accuracy_score(target, score >= threshold))
            )
            boot["brier"].append(float(brier_score_loss(target, score)))
        base["group_bootstrap_95ci"] = {
            name: (
                [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                if values
                else None
            )
            for name, values in boot.items()
        }
        result["heads"][head] = base
    return result
