from __future__ import annotations

from typing import Any, Dict, Iterable

LINEAGE_KEYS = ("model_version", "feat_version", "label_version", "batch_id")
VIEWS = ("protein", "dna", "rna", "omics")
TRACKS = ("SOM", "INT")
TASKS = ("T1", "T2", "T3a", "T3b", "T4")
LABEL_LEVELS = ("L1", "L2", "L3", "L4")


def validate_lineage(value: Dict[str, Any]) -> None:
    missing = [key for key in LINEAGE_KEYS if not value.get(key)]
    if missing:
        raise ValueError(f"lineage missing required fields: {missing}")


def validate_feature(record: Dict[str, Any]) -> None:
    required = ("gene_id", "view", "track", "embedding", "lineage")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"UDC-02 missing required fields: {missing}")
    if record["view"] not in VIEWS or record["track"] not in TRACKS:
        raise ValueError("UDC-02 has an unsupported view or track")
    validate_lineage(record["lineage"])


def validate_label(record: Dict[str, Any]) -> None:
    required = ("gene_id", "task", "level", "label", "weight", "lineage")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"UDC-03 missing required fields: {missing}")
    if record["task"] not in TASKS or record["level"] not in LABEL_LEVELS:
        raise ValueError("UDC-03 has an unsupported task or label level")
    if not 0 <= float(record["weight"]) <= 1:
        raise ValueError("UDC-03 weight must be in [0, 1]")
    validate_lineage(record["lineage"])


def validate_graph(record: Dict[str, Any]) -> None:
    required = ("node_ids", "edge_index", "edge_type", "num_rels", "lineage")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"UDC-04 missing required fields: {missing}")
    if int(record["num_rels"]) != 7:
        raise ValueError("UDC-04 requires seven relation types")
    validate_lineage(record["lineage"])


def validate_records(records: Iterable[Dict[str, Any]], kind: str) -> None:
    validator = {"feature": validate_feature, "label": validate_label, "graph": validate_graph}.get(kind)
    if validator is None:
        raise ValueError(f"unknown contract kind: {kind}")
    for record in records:
        validator(record)
