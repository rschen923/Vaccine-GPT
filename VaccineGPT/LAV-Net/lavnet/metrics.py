from __future__ import annotations

import numpy as np
import pandas as pd


def hard_gate(scores: pd.DataFrame, tau_a: float, tau_i: float, tau_e: float) -> pd.Series:
    return (
        (scores["attenuation_score"] >= tau_a)
        & (scores["immunogenicity_score"] < tau_i)
        & (scores["essentiality_score"] < tau_e)
    )


def gate_metrics(scores: pd.DataFrame, selected: pd.Series) -> dict[str, float | int]:
    n = int(selected.sum())
    known = scores.get("known_target", pd.Series(False, index=scores.index))
    antigens = scores.get("known_antigen", pd.Series(False, index=scores.index))
    essential = scores.get("in_vitro_essential", pd.Series(False, index=scores.index))
    return {
        "candidate_count": n,
        "known_target_recall": float((selected & known).sum() / max(int(known.sum()), 1)),
        "immunogenicity_deletion_rate": float((selected & antigens).sum() / max(n, 1)),
        "in_vitro_essential_deletion_rate": float((selected & essential).sum() / max(n, 1)),
    }


def bootstrap_metric(values: np.ndarray, statistic, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.array([statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)])
    return float(statistic(values)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))

