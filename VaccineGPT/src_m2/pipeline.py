from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from src_m2.aep import external_parameter_declaration, validate_aep_row
from src_m2.decision import (
    DEFAULTS,
    acquisition_scores,
    antigen_retention,
    attenuation_score,
    gradient_contributions,
    normalized_betweenness,
    pareto_tiers,
    safe_score,
    three_reader_consensus,
)


def _sigmoid(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("head logits must be finite")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _rank(scores: Mapping[str, float]) -> list[str]:
    return [
        gene_id
        for gene_id, _ in sorted(
            scores.items(), key=lambda item: (-float(item[1]), str(item[0]))
        )
    ]


def decide_genome(
    gene_records: Sequence[Mapping[str, Any]],
    head_logits: Mapping[str, Mapping[str, float]],
    pace_b: Mapping[str, float],
    graph_edges: Sequence[Mapping[str, Any]],
    operon_neighbors: Mapping[str, Sequence[str]],
    interaction_neighbors: Mapping[str, Sequence[str]],
    safety_flags: Mapping[str, Mapping[str, bool]] | None,
    calibration_predictions: Sequence[float],
    calibration_targets: Sequence[float],
    attention_reader: Mapping[str, float] | None = None,
    prototype_reader: Mapping[str, float] | None = None,
    validated_model: bool = False,
    checkpoint_sha256: str | None = None,
    passed_gates: Sequence[str] = (),
) -> dict[str, Any]:
    """Wire M2 logits and measured inputs into auditable per-gene AEP rows."""
    gene_ids = [str(row.get("gene_id") or "") for row in gene_records]
    if any(not gene_id for gene_id in gene_ids) or len(gene_ids) != len(set(gene_ids)):
        raise ValueError("gene_records require unique non-empty gene_id values")
    required_gates = {"G-A", "G-B", "G-C", "G-leak", "G-safe", "G-grade"}
    if not validated_model:
        rows = []
        for record in gene_records:
            aep_row = {
                "gene_or_product": str(record["gene_id"]),
                "att_interval": {"lower": 0.0, "upper": 1.0, "status": "unavailable_unvalidated_model"},
                "ret": None,
                "safe": None,
                "evidence_level": record.get("evidence_level", "E4"),
                "three_reader_consistency": {"status": "not_assessed_unvalidated_model"},
                "mechanism_annotation": record.get("mechanism_annotation"),
                "warning": "Unvalidated or untrained checkpoint; no biological ranking is emitted.",
                "implementation_path": "",
                "tier": "unranked",
                "utility": None,
                "degrade_reasons": ["model_validation_gates_not_passed"],
                "decision_status": "unranked",
            }
            validate_aep_row(aep_row, version="v1")
            rows.append(aep_row)
        return {
            "aep_rows": rows,
            "virulence_gene_list": [],
            "retained_antigen_list": [],
            "unresolved_genes": gene_ids,
            "acquisition_scores": {},
            "external_parameter_declaration": external_parameter_declaration(),
            "reader_rankings": [[], [], []],
            "systematic_reader_disagreement": None,
            "missing_pace_gene_ids": sorted(set(gene_ids) - set(pace_b)),
            "biological_use_status": "blocked_model_not_validated",
        }
    if checkpoint_sha256 is None or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha256):
        raise ValueError("validated decision requires the checkpoint SHA-256")
    if not required_gates.issubset(set(passed_gates)):
        raise ValueError(
            f"validated decision requires gates {sorted(required_gates)}; "
            f"missing {sorted(required_gates - set(passed_gates))}"
        )
    if len(calibration_predictions) < 100 or len(calibration_predictions) != len(calibration_targets):
        raise ValueError("validated v1 decision requires at least 100 matched calibration observations")
    if "H1" not in head_logits or "H2" not in head_logits or "H4" not in head_logits:
        raise ValueError("H1, H2, and H4 logits are required for M2 decision processing")
    if any(gene_id not in pace_b for gene_id in gene_ids):
        missing_pace = [gene_id for gene_id in gene_ids if gene_id not in pace_b]
    else:
        missing_pace = []
    for head in ("H1", "H2", "H4"):
        missing = set(gene_ids) - set(head_logits[head])
        if missing:
            raise ValueError(f"{head} logits missing for {len(missing)} genes")
    h1 = {gene_id: _sigmoid(head_logits["H1"][gene_id]) for gene_id in gene_ids}
    h2 = {gene_id: _sigmoid(head_logits["H2"][gene_id]) for gene_id in gene_ids}
    h3 = {
        gene_id: _sigmoid(head_logits["H3"][gene_id])
        for gene_id in gene_ids
        if "H3" in head_logits and gene_id in head_logits["H3"]
    }
    h4 = {gene_id: _sigmoid(head_logits["H4"][gene_id]) for gene_id in gene_ids}
    normalized_edges = []
    for edge in graph_edges:
        source, target = str(edge.get("source_gene") or ""), str(edge.get("target_gene") or "")
        if source in gene_ids and target in gene_ids:
            relation = edge.get("relation")
            normalized_edges.append((source, target, relation))
    centrality = normalized_betweenness(
        gene_ids,
        [
            (source, target)
            for source, target, relation in normalized_edges
            if relation in {"operon", "interaction"}
        ],
    )
    antigen = h2
    ret_scores = {
        gene_id: antigen_retention(
            gene_id,
            antigen,
            operon_neighbors,
            interaction_neighbors,
            centrality,
        )
        for gene_id in gene_ids
    }
    genome_pace = {gene_id: float(pace_b[gene_id]) for gene_id in gene_ids if gene_id in pace_b}
    pace_population_complete = not missing_pace
    att_results = {
        gene_id: attenuation_score(
            h1.get(gene_id),
            pace_b.get(gene_id),
            genome_pace_b=genome_pace if pace_population_complete else None,
            gene_id=gene_id,
            calibration_predictions=calibration_predictions,
            calibration_targets=calibration_targets,
        )
        for gene_id in gene_ids
    }
    safe_results = {
        gene_id: safe_score((safety_flags or {}).get(gene_id))
        for gene_id in gene_ids
    }
    gradient_reader: dict[str, float] = {}
    for gene_id in gene_ids:
        att_value = att_results[gene_id]["estimate"]
        ret_value = ret_scores[gene_id]
        safe_value = safe_results[gene_id]["score"]
        if att_value is not None and ret_value is not None and safe_value is not None:
            contributions = gradient_contributions(att_value, ret_value, safe_value)
            gradient_reader[gene_id] = sum(contributions.values())
    rankings = [
        _rank(attention_reader or {}),
        _rank(prototype_reader or {}),
        _rank(gradient_reader),
    ]
    consensus = three_reader_consensus(rankings)
    rows = []
    for record in gene_records:
        gene_id = str(record["gene_id"])
        att = att_results[gene_id]
        safety = safe_results[gene_id]
        rows.append(
            {
                "gene_id": gene_id,
                "att_estimate": att["estimate"],
                "att_interval": att["interval"],
                "att_status": att["status"],
                "ret": ret_scores[gene_id],
                "safe": safety["score"],
                "safety_status": safety["status"],
                "h4_probability": h4.get(gene_id),
                "h3_probability": h3.get(gene_id),
                "h2_probability": h2.get(gene_id),
                "evidence_level": record.get("evidence_level", "E4"),
                "consensus_count": consensus["counts"].get(gene_id, 0),
                "degrade_reasons": list(record.get("degrade_reasons", [])),
                "mechanism_annotation": record.get("mechanism_annotation"),
                "warning": record.get("warning"),
            }
        )
    ranked = pareto_tiers(rows)
    aep_rows = []
    for row in ranked:
        status = "ranked" if row["tier"] in {"A", "B", "C"} else "unranked"
        warning = row.get("warning") or ""
        if row["tier"] == "excluded_h4_core_essential":
            warning = "; ".join(filter(None, (warning, "H4 hard-veto: predicted core essential")))
        if row["tier"] == "unranked":
            warning = "; ".join(filter(None, (warning, "missing required calibrated decision components")))
        aep_row = {
            "gene_or_product": row["gene_id"],
            "att_interval": row["att_interval"],
            "ret": row["ret"],
            "safe": row["safe"],
            "evidence_level": row["evidence_level"],
            "three_reader_consistency": {
                "count": row["consensus_count"],
                "threshold": 2,
                "systematic_disagreement": consensus["systematic_disagreement"],
                "reader_status": [
                    "available" if ranking else "unavailable" for ranking in rankings
                ],
            },
            "mechanism_annotation": row.get("mechanism_annotation"),
            "warning": warning,
            "implementation_path": "",
            "tier": row["tier"],
            "utility": row["utility"],
            "degrade_reasons": row["degrade_reasons"],
            "decision_status": status,
        }
        validate_aep_row(aep_row, version="v1")
        aep_rows.append(aep_row)
    virulence = [
        gene_id
        for gene_id in gene_ids
        if h1.get(gene_id, 0.0) >= 0.5 and h3.get(gene_id, 0.0) >= 0.5
    ]
    retained_antigens = [
        gene_id
        for gene_id in gene_ids
        if h2.get(gene_id, 0.0) >= 0.5 and ret_scores.get(gene_id) is not None
    ]
    pace_missing = sorted(set(missing_pace))
    acq = {}
    acq_inputs = [
        gene_id for gene_id in gene_ids
        if att_results[gene_id]["estimate"] is not None
        and ret_scores[gene_id] is not None
    ]
    if acq_inputs and all(safe_results[gene_id]["score"] is not None for gene_id in acq_inputs):
        acq = acquisition_scores(
            {
                gene_id: (
                    float(att_results[gene_id]["interval"]["lower"]),
                    float(att_results[gene_id]["interval"]["upper"]),
                )
                for gene_id in acq_inputs
            },
            {gene_id: int(consensus["counts"].get(gene_id, 0)) for gene_id in acq_inputs},
            {gene_id: float(att_results[gene_id]["estimate"]) for gene_id in acq_inputs},
            {gene_id: float(ret_scores[gene_id]) for gene_id in acq_inputs},
        )
    return {
        "aep_rows": aep_rows,
        "virulence_gene_list": virulence,
        "retained_antigen_list": retained_antigens,
        "unresolved_genes": [
            row["gene_id"] for row in ranked if row["tier"] == "unranked"
        ],
        "acquisition_scores": acq,
        "external_parameter_declaration": external_parameter_declaration(),
        "reader_rankings": rankings,
        "systematic_reader_disagreement": consensus["systematic_disagreement"],
        "missing_pace_gene_ids": pace_missing,
        "validated_checkpoint_sha256": checkpoint_sha256,
        "passed_gates": sorted(set(passed_gates)),
        "biological_use_status": "validated_model_outputs_not_experimental_evidence",
    }
