from __future__ import annotations

import math
from typing import Any, Dict, Iterable

LINEAGE_KEYS = ("model_version", "feature_version", "label_version", "batch_id")
VIEWS = (
    "protein",
    "protein_residue",
    "annotation",
    "structure_3di",
    "msa",
    "pssm",
    "omics",
    "dna",
    "dna_window",
    "rna",
)
TRACKS = ("v1_protein", "v2_dna", "private_omics", "SOM", "INT")
TASKS = ("H1", "H2", "H3", "H4", "H5", "T1", "T2", "T3a", "T3b", "T4")
EVIDENCE_LEVELS = ("E1", "E2", "E3", "E4")
LEGACY_LEVELS = ("L1", "L2", "L3", "L4")


def validate_lineage(value: Dict[str, Any]) -> None:
    missing = [
        key
        for key in ("model_version", "label_version", "batch_id")
        if not value.get(key)
    ]
    if not value.get("feature_version") and not value.get("feat_version"):
        missing.append("feature_version")
    if missing:
        raise ValueError(f"lineage missing required fields: {missing}")
    for key in ("model_version", "feature_version", "label_version", "batch_id"):
        item = value.get(key, value.get("feat_version") if key == "feature_version" else None)
        if str(item).strip().casefold() in {
            "unspecified",
            "unknown",
            "replace_with_batch_id",
            "template",
        }:
            raise ValueError(f"lineage {key} is a placeholder and must be replaced")


def validate_feature(record: Dict[str, Any]) -> None:
    required = ("gene_id", "view", "track", "lineage")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"UDC-02 missing required fields: {missing}")
    if not any(key in record for key in ("embedding", "residue_embeddings", "embedding_path")):
        raise ValueError("UDC-02 requires embedding, residue_embeddings, or embedding_path")
    if record["view"] not in VIEWS or record["track"] not in TRACKS:
        raise ValueError("UDC-02 has an unsupported view or track")
    if record["view"] in {"dna", "dna_window"} and record["track"] != "v2_dna":
        raise ValueError("DNA sequence features are not accepted on the v1 protein track")
    if record["track"] == "v2_dna" and record["view"] not in {"dna", "dna_window"}:
        raise ValueError("the v2 DNA track accepts genomic DNA or DNA-window features")
    if "embedding" in record:
        try:
            if len(record["embedding"]) == 0:
                raise ValueError("UDC-02 embedding cannot be empty")
        except TypeError as error:
            raise ValueError("UDC-02 embedding must be a non-empty sequence") from error
    validate_lineage(record["lineage"])


def validate_label(record: Dict[str, Any]) -> None:
    required = ("gene_id", "task", "label", "lineage")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"UDC-03 missing required fields: {missing}")
    if record["task"] not in TASKS:
        raise ValueError("UDC-03 has an unsupported task")
    if "evidence_level" in record:
        if record["evidence_level"] not in EVIDENCE_LEVELS:
            raise ValueError("UDC-03 evidence_level must be E1, E2, E3, or E4")
        if record.get("training_eligible_as_gold") and record["evidence_level"] not in {"E1", "E2"}:
            raise ValueError("only E1/E2 evidence may be marked as gold-training eligible")
    elif record.get("level") not in LEGACY_LEVELS:
        raise ValueError("UDC-03 requires evidence_level E1-E4 or an explicitly legacy L1-L4 level")
    if "weight" in record and not 0 <= float(record["weight"]) <= 1:
        raise ValueError("UDC-03 weight must be in [0, 1]")
    try:
        label = float(record["label"])
    except (TypeError, ValueError) as error:
        raise ValueError("UDC-03 label must be numeric") from error
    if not math.isfinite(label):
        raise ValueError("UDC-03 label must be finite")
    validate_lineage(record["lineage"])


def validate_graph(record: Dict[str, Any]) -> None:
    required = ("node_ids", "edge_index", "edge_type", "num_rels", "lineage")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"UDC-04 missing required fields: {missing}")
    if int(record["num_rels"]) != 6:
        raise ValueError("UDC-04 requires six v1/v2 relation slots (four v1 plus two reserved)")
    node_count = len(record["node_ids"])
    edge_index = record["edge_index"]
    if not isinstance(edge_index, (list, tuple)) or len(edge_index) != 2:
        raise ValueError("UDC-04 edge_index must contain source and target rows")
    edge_count = len(edge_index[0])
    if len(edge_index[1]) != edge_count or len(record["edge_type"]) != edge_count:
        raise ValueError("UDC-04 edge arrays must have equal lengths")
    if "edge_weight" in record and len(record["edge_weight"]) != edge_count:
        raise ValueError("UDC-04 edge_weight length must match edge count")
    if len(set(map(str, record["node_ids"]))) != node_count:
        raise ValueError("UDC-04 node_ids must be unique")
    if edge_count:
        if min(min(edge_index[0]), min(edge_index[1])) < 0 or max(max(edge_index[0]), max(edge_index[1])) >= node_count:
            raise ValueError("UDC-04 edge_index contains an invalid node index")
        if min(record["edge_type"]) < 0 or max(record["edge_type"]) >= 6:
            raise ValueError("UDC-04 edge_type is outside the six-slot relation vocabulary")
    validate_lineage(record["lineage"])


def validate_records(records: Iterable[Dict[str, Any]], kind: str) -> None:
    validator = {"feature": validate_feature, "label": validate_label, "graph": validate_graph}.get(kind)
    if validator is None:
        raise ValueError(f"unknown contract kind: {kind}")
    for record in records:
        validator(record)
