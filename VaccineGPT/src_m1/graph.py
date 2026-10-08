from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


EDGE_TYPES = {
    "operon": 0,
    "coexpression": 1,
    "coadaptation": 2,
    "interaction": 3,
    "dna_regulation": 4,
    "dna_element_proximity": 5,
}


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else None


def build_gene_graph(
    gene_records: Sequence[Mapping[str, Any]],
    expression_vectors: Mapping[str, Sequence[float]] | None = None,
    pace_vectors: Mapping[str, Sequence[float]] | None = None,
    interaction_scores: Mapping[tuple[str, str], tuple[float, str]] | None = None,
    adjacency_limit_nt: int = 60,
) -> dict[str, Any]:
    if adjacency_limit_nt < 0:
        raise ValueError("adjacency_limit_nt must be nonnegative")
    genes = [dict(record) for record in gene_records]
    ids = [str(row.get("gene_id") or "") for row in genes]
    if any(not gene_id for gene_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("gene_records require unique non-empty gene_id values")
    indices = {gene_id: index for index, gene_id in enumerate(ids)}
    edges: dict[tuple[str, str, int], dict[str, Any]] = {}

    def add(left: str, right: str, relation: str, weight: float, evidence: str, source: str) -> None:
        if left == right or left not in indices or right not in indices:
            return
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("edge weights must be finite and nonnegative")
        key = (left, right, EDGE_TYPES[relation])
        edges[key] = {
            "source_gene": left,
            "target_gene": right,
            "relation": relation,
            "relation_id": EDGE_TYPES[relation],
            "weight": weight,
            "evidence_level": evidence,
            "source": source,
        }

    by_contig: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in genes:
        contig = str(row.get("contig_id") or "")
        genome = str(row.get("genome_id") or "")
        if row.get("start") is not None and row.get("end") is not None:
            by_contig.setdefault((genome, contig), []).append(row)
    for rows in by_contig.values():
        rows.sort(key=lambda row: min(int(row["start"]), int(row["end"])))
        for left, right in zip(rows, rows[1:]):
            left_strand, right_strand = str(left.get("strand")), str(right.get("strand"))
            left_end = max(int(left["start"]), int(left["end"]))
            right_start = min(int(right["start"]), int(right["end"]))
            gap = max(0, right_start - left_end - 1)
            same_strand = left_strand == right_strand and left_strand in {"+", "-"}
            same_operon = bool(left.get("operon_id")) and left.get("operon_id") == right.get("operon_id")
            if same_operon or (same_strand and gap <= adjacency_limit_nt):
                weight = 1.0 if same_operon else 0.8
                source = "shared_operon_annotation" if same_operon else "same_strand_gap_le_60nt"
                for a, b in (
                    (str(left["gene_id"]), str(right["gene_id"])),
                    (str(right["gene_id"]), str(left["gene_id"])),
                ):
                    add(a, b, "operon", weight, "E3", source)

    if expression_vectors:
        expressed_ids = [gene_id for gene_id in ids if gene_id in expression_vectors]
        for i, left in enumerate(expressed_ids):
            for right in expressed_ids[i + 1 :]:
                correlation = _pearson(expression_vectors[left], expression_vectors[right])
                if correlation is not None and abs(correlation) >= 0.8:
                    weight = max(0.0, correlation**2)
                    add(left, right, "coexpression", weight, "E2", "private_rnaseq")
                    add(right, left, "coexpression", weight, "E2", "private_rnaseq")

    if pace_vectors:
        fitted_ids = [gene_id for gene_id in ids if gene_id in pace_vectors]
        for i, left in enumerate(fitted_ids):
            for right in fitted_ids[i + 1 :]:
                similarity = _cosine(pace_vectors[left], pace_vectors[right])
                if similarity is not None and similarity >= 0.9:
                    add(left, right, "coadaptation", similarity, "E2", "private_pace")
                    add(right, left, "coadaptation", similarity, "E2", "private_pace")

    for pair, value in (interaction_scores or {}).items():
        if len(pair) != 2:
            raise ValueError("interaction keys must contain two gene IDs")
        score, source = float(value[0]), str(value[1])
        threshold = 0.5 if source.casefold() in {"clef_eei", "eei"} else 0.7
        if score >= threshold:
            for left, right in (pair, (pair[1], pair[0])):
                add(str(left), str(right), "interaction", score, "E3", source)

    rows = list(edges.values())
    return {
        "node_ids": ids,
        "edge_index": torch.tensor(
            [[indices[row["source_gene"]] for row in rows], [indices[row["target_gene"]] for row in rows]],
            dtype=torch.long,
        ).reshape(2, len(rows)),
        "edge_type": torch.tensor([row["relation_id"] for row in rows], dtype=torch.long),
        "edge_weight": torch.tensor([row["weight"] for row in rows], dtype=torch.float32),
        "evidence_level": [row["evidence_level"] for row in rows],
        "edge_records": rows,
        "num_relations": len(EDGE_TYPES),
    }
