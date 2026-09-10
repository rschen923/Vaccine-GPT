from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from shared.contracts import validate_label


LEVEL_WEIGHT = {"L1": 1.0, "L2": 0.7, "L3": 0.5, "L4": 0.3}


def build_four_tier_labels(
    consensus: Iterable[Dict],
    single_source: Iterable[Dict],
    homolog_positive: Iterable[Dict],
    zero_shot: Dict[str, float],
    lineage: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict] = []
    for source_rows, level, source in (
        (consensus, "L1", "Tnseq+CRISPRi"),
        (single_source, "L2", "single_source"),
        (homolog_positive, "L3", "homology"),
    ):
        for row in source_rows:
            record = {
                "gene_id": str(row["gene_id"]), "task": row.get("task", "T1"),
                "level": level, "label": row["label"],
                "weight": LEVEL_WEIGHT[level], "source": source, "lineage": lineage,
            }
            validate_label(record)
            rows.append(record)
    for gene_id, score in zero_shot.items():
        record = {
            "gene_id": str(gene_id), "task": "T3b", "level": "L4",
            "label": float(score), "weight": LEVEL_WEIGHT["L4"],
            "source": "zeroshot", "lineage": lineage,
        }
        validate_label(record)
        rows.append(record)
    return pd.DataFrame(rows)


def check_label_consistency(left: pd.DataFrame, right: pd.DataFrame) -> float:
    from sklearn.metrics import cohen_kappa_score

    merged = left.merge(right, on="gene_id", suffixes=("_left", "_right"))
    if merged.empty:
        raise ValueError("cannot calculate label consistency on an empty overlap")
    return float(cohen_kappa_score(merged["label_left"], merged["label_right"]))
