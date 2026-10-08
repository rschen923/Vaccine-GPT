from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from shared.contracts import validate_label
from shared.data_quality import classify_label_evidence


LEGACY_LEVELS = {
    "consensus": "L1",
    "single_source": "L2",
    "homology": "L3",
    "zero_shot": "L4",
}


def build_four_tier_labels(
    consensus: Iterable[Dict],
    single_source: Iterable[Dict],
    homolog_positive: Iterable[Dict],
    zero_shot: Dict[str, float],
    lineage: Dict[str, str],
) -> pd.DataFrame:
    """Compatibility adapter: retain old groups as metadata, never as E1-E4 evidence."""
    rows: List[Dict] = []
    for source_rows, legacy_group, source in (
        (consensus, "consensus", "Tnseq+CRISPRi"),
        (single_source, "single_source", "single_source"),
        (homolog_positive, "homology", "homology"),
    ):
        for row in source_rows:
            task = row.get("task")
            if task is None:
                raise ValueError(
                    f"{legacy_group} record for {row.get('gene_id')} lacks an explicit target task"
                )
            evidence = classify_label_evidence(
                {
                    **row,
                    "source": row.get("source") or source,
                    "assay": row.get("assay") or row.get("assay_type"),
                    "method": row.get("method") or (
                        "homology prediction" if legacy_group == "homology" else ""
                    ),
                    "is_private": bool(row.get("is_private") or row.get("private_dataset")),
                }
            )
            record = {
                "gene_id": str(row["gene_id"]),
                "task": str(task),
                "label": row["label"],
                "source": row.get("source") or source,
                "legacy_group": legacy_group,
                "legacy_level": LEGACY_LEVELS[legacy_group],
                "lineage": lineage,
                **evidence,
            }
            validate_label(record)
            rows.append(record)
    for gene_id, score in zero_shot.items():
        evidence = classify_label_evidence(
            {"source": "zero-shot prediction", "method": "zero-shot prediction"}
        )
        record = {
            "gene_id": str(gene_id),
            "task": "T3b",
            "label": float(score),
            "source": "zeroshot",
            "legacy_group": "zero_shot",
            "legacy_level": LEGACY_LEVELS["zero_shot"],
            "lineage": lineage,
            **evidence,
        }
        validate_label(record)
        rows.append(record)
    return pd.DataFrame(rows)


def check_label_consistency(left: pd.DataFrame, right: pd.DataFrame) -> float:
    from sklearn.metrics import cohen_kappa_score

    join_keys = ["gene_id"]
    if "task" in left.columns and "task" in right.columns:
        join_keys.append("task")
    merged = left.merge(right, on=join_keys, suffixes=("_left", "_right"))
    if merged.empty:
        raise ValueError("cannot calculate label consistency on an empty gene/task overlap")
    return float(cohen_kappa_score(merged["label_left"], merged["label_right"]))
