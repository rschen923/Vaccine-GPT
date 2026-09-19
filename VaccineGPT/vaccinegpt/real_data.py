"""Adapter from canonical public JSONL records to the L0-L4 tensor contract.

Missing biological labels are represented by task masks, never by synthetic
negative labels. Feature fallback values are recorded in the returned
provenance so a pilot run cannot be mistaken for a complete assay dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _vector(sequence: str, length: int = 64) -> list[float]:
    values = [0.0] * length
    sequence = str(sequence or "").upper()
    for index, residue in enumerate(sequence[:length]):
        values[index] = (AMINO_ACIDS.find(residue) + 1) / len(AMINO_ACIDS) if residue in AMINO_ACIDS else 0.0
    return values


def _embedding(sequence: str) -> list[float]:
    vector = _vector(sequence, 64)
    buckets = [0.0] * 32
    for index, value in enumerate(vector):
        buckets[index % 32] += value
    return [value / 2.0 for value in buckets]


def load_real_batch(
    sequences_path: str | Path,
    labels_path: str | Path,
    *,
    ppi_path: str | Path | None = None,
    max_records: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    sequences = [
        json.loads(line) for line in Path(sequences_path).read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    if max_records:
        sequences = sequences[:max_records]
    ids = [str(row["gene_id"]) for row in sequences]
    index = {gene_id: offset for offset, gene_id in enumerate(ids)}
    labels: dict[str, dict[str, float]] = defaultdict(dict)
    for line in Path(labels_path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gene_id = str(row["gene_id"])
        if gene_id in index:
            labels[gene_id][str(row["task"])] = float(row["label"])
    n = len(sequences)
    x_a = torch.tensor([[_vector(row.get("protein_sequence") or row.get("sequence", "")) for _ in range(1)] for row in sequences])
    x_b = torch.tensor([_embedding(row.get("protein_sequence") or row.get("sequence", "")) for row in sequences])
    y_t1 = torch.zeros(n, dtype=torch.long)
    y_t2 = torch.zeros(n)
    y_t3 = torch.zeros(n)
    y_t4 = torch.zeros(n)
    mask = torch.zeros(n, 4)
    for offset, gene_id in enumerate(ids):
        values = labels.get(gene_id, {})
        if "T1" in values:
            y_t1[offset] = int(round(values["T1"]))
            mask[offset, 0] = 1
        if "T2" in values:
            y_t2[offset] = values["T2"]
            mask[offset, 1] = 1
        if "T3b" in values or "T3" in values:
            y_t3[offset] = values.get("T3b", values.get("T3", 0.0))
            mask[offset, 2] = 1
        if "T4" in values:
            y_t4[offset] = values["T4"]
            mask[offset, 3] = 1
    adjacency = torch.eye(n)
    ppi_edges = 0
    if ppi_path:
        for line in Path(ppi_path).read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[0] in index and parts[1] in index:
                left, right = index[parts[0]], index[parts[1]]
                adjacency[left, right] = adjacency[right, left] = 1.0
                ppi_edges += 1
    batch = {
        "x_a": x_a, "x_b": x_b, "S_adj": torch.zeros(n), "pcd_A": torch.zeros(n, 4),
        "ch_covariates": torch.zeros(n, 16), "fba": torch.zeros(n), "y3": y_t3,
        "y_vivo": y_t1, "y_vitro": y_t1, "PPI_adj": adjacency, "node_x": x_b,
        "topo_feat": torch.zeros(n, 4), "y_sec": torch.zeros(n), "domain": torch.zeros(n, dtype=torch.long),
        "pu_pos_idx": torch.where((mask[:, 1] > 0) & (y_t2 > 0))[0],
        "pu_unl_idx": torch.where(mask[:, 1] == 0)[0], "N_p": torch.ones(n),
        "N_star": torch.ones(3), "hla_freq": torch.ones(6) / 6.0,
        "pep": torch.zeros(n, 64), "pseudo": torch.zeros(n, 34), "pep_mask": torch.zeros(n),
        "ic50": torch.zeros(n), "el": torch.zeros(n), "t4_feat": torch.zeros(n, 4),
        "t4_rel": y_t4, "R0": torch.ones(n), "v0": torch.full((n,), 0.8), "t_obs": torch.ones(n),
        "task_mask": mask,
    }
    provenance = {
        "sequences": str(sequences_path), "labels": str(labels_path), "ppi": str(ppi_path) if ppi_path else None,
        "records": n, "ppi_edges": ppi_edges, "sequence_encoder": "deterministic amino-acid position encoding",
        "label_coverage": {name: float(mask[:, column].mean()) for column, name in enumerate(("T1", "T2", "T3", "T4"))},
        "fallbacks": ["missing T1/T4 remain masked", "missing auxiliary assays use explicit zero covariates"],
        "dataset_fingerprint": hashlib.sha256("".join(ids).encode()).hexdigest(),
    }
    return batch, provenance
