from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULTS = {
    "pace_threshold": -0.15,
    "att_h1_weight": 0.5,
    "att_pace_weight": 0.5,
    "conformal_alpha": 0.10,
    "ret_centrality_lambda": 2.0,
    "safe_flag_penalty": 0.2,
    "h4_veto_threshold": 0.5,
    "tier_a_att": 0.6,
    "tier_a_ret": 0.6,
    "utility_att_power": 1.0,
    "utility_ret_power": 1.0,
    "utility_safe_power": 1.0,
}


def fit_pace(timepoints: Sequence[float], fitness: Sequence[float]) -> dict[str, Any]:
    if len(timepoints) != len(fitness):
        raise ValueError("timepoints and fitness must have equal lengths")
    if len(timepoints) < 3:
        return {"status": "not_fitted", "reason": "fewer_than_three_timepoints"}
    times = np.asarray(timepoints, dtype=np.float64)
    values = np.asarray(fitness, dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        return {"status": "not_fitted", "reason": "non_finite_measurements"}
    if np.any(times < 0):
        return {"status": "not_fitted", "reason": "negative_timepoint"}
    maximum = float(np.max(times))
    if maximum <= 0:
        return {"status": "not_fitted", "reason": "non_positive_max_time"}
    normalized = times / maximum
    design = np.column_stack((normalized**2, normalized, np.ones_like(normalized)))
    if np.linalg.matrix_rank(design) < 3:
        return {"status": "not_fitted", "reason": "singular_design_matrix"}
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return {
        "status": "fitted",
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "c": float(coefficients[2]),
        "essentiality_proxy": bool(coefficients[1] < DEFAULTS["pace_threshold"]),
        "threshold": DEFAULTS["pace_threshold"],
        "time_normalization": "t / max(t)",
    }


def minmax_normalize(values: Mapping[str, float]) -> dict[str, float]:
    finite = {key: float(value) for key, value in values.items() if math.isfinite(float(value))}
    if not finite:
        return {}
    low, high = min(finite.values()), max(finite.values())
    if high == low:
        return {key: 0.5 for key in finite}
    return {key: (value - low) / (high - low) for key, value in finite.items()}


def conformal_interval(
    point: float | None,
    calibration_predictions: Sequence[float],
    calibration_targets: Sequence[float],
    alpha: float = DEFAULTS["conformal_alpha"],
    minimum_calibration_size: int = 100,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if point is None or not math.isfinite(float(point)):
        return {"lower": 0.0, "upper": 1.0, "status": "unavailable", "n_cal": 0}
    if len(calibration_predictions) != len(calibration_targets):
        raise ValueError("calibration predictions and targets must have equal lengths")
    if not calibration_predictions:
        return {"lower": 0.0, "upper": 1.0, "status": "no_calibration_data", "n_cal": 0}
    residuals = np.abs(
        np.asarray(calibration_predictions, dtype=np.float64)
        - np.asarray(calibration_targets, dtype=np.float64)
    )
    if not np.isfinite(residuals).all():
        raise ValueError("calibration residuals must be finite")
    n_cal = int(residuals.size)
    if n_cal < minimum_calibration_size:
        quantile = float(np.max(residuals) * (1.0 + 1.0 / n_cal))
        status = "small_sample_widened"
    else:
        rank = int(math.ceil((1.0 - alpha) * (n_cal + 1)))
        quantile = (
            float(np.max(residuals) * (1.0 + 1.0 / n_cal))
            if rank > n_cal
            else float(np.partition(residuals, rank - 1)[rank - 1])
        )
        status = "split_conformal"
    estimate = float(point)
    return {
        "lower": max(0.0, estimate - quantile),
        "upper": min(1.0, estimate + quantile),
        "quantile": quantile,
        "alpha": alpha,
        "n_cal": n_cal,
        "status": status,
    }


def attenuation_score(
    h1_probability: float | None,
    pace_b: float | None,
    genome_pace_b: Mapping[str, float] | None = None,
    gene_id: str | None = None,
    calibration_predictions: Sequence[float] = (),
    calibration_targets: Sequence[float] = (),
) -> dict[str, Any]:
    if h1_probability is None or pace_b is None or genome_pace_b is None or gene_id is None:
        return {
            "estimate": None,
            "interval": {"lower": 0.0, "upper": 1.0, "status": "incomplete_components"},
            "status": "unavailable",
            "missing_components": [
                name
                for name, value in (
                    ("h1_probability", h1_probability),
                    ("pace_b", pace_b),
                    ("genome_pace_b", genome_pace_b),
                    ("gene_id", gene_id),
                )
                if value is None
            ],
        }
    if not 0 <= h1_probability <= 1 or not math.isfinite(float(pace_b)):
        raise ValueError("H1 probability must be in [0, 1] and PACE b must be finite")
    normalized_population = minmax_normalize(
        {key: -float(value) for key, value in genome_pace_b.items()}
    )
    if gene_id not in normalized_population:
        return {
            "estimate": None,
            "interval": {"lower": 0.0, "upper": 1.0, "status": "gene_missing_from_pace_population"},
            "status": "unavailable",
            "missing_components": ["gene_id_in_genome_pace_b"],
        }
    normalized_pace = normalized_population[gene_id]
    estimate = (
        DEFAULTS["att_h1_weight"] * float(h1_probability)
        + DEFAULTS["att_pace_weight"] * normalized_pace
    )
    interval = conformal_interval(
        estimate, calibration_predictions, calibration_targets
    )
    return {"estimate": estimate, "interval": interval, "status": interval["status"]}


def normalized_betweenness(
    node_ids: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, float]:
    nodes = list(dict.fromkeys(map(str, node_ids)))
    count = len(nodes)
    if count < 3:
        return {node: 0.0 for node in nodes}
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        left, right = str(left), str(right)
        if left in adjacency and right in adjacency and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    centrality = dict.fromkeys(nodes, 0.0)
    for source in nodes:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {node: [] for node in nodes}
        paths = dict.fromkeys(nodes, 0.0)
        paths[source] = 1.0
        distance = dict.fromkeys(nodes, -1)
        distance[source] = 0
        queue = [source]
        while queue:
            vertex = queue.pop(0)
            stack.append(vertex)
            for neighbor in adjacency[vertex]:
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1
                if distance[neighbor] == distance[vertex] + 1:
                    paths[neighbor] += paths[vertex]
                    predecessors[neighbor].append(vertex)
        dependency = dict.fromkeys(nodes, 0.0)
        while stack:
            vertex = stack.pop()
            for previous in predecessors[vertex]:
                if paths[vertex]:
                    dependency[previous] += (
                        paths[previous] / paths[vertex]
                    ) * (1.0 + dependency[vertex])
            if vertex != source:
                centrality[vertex] += dependency[vertex]
    for node in nodes:
        centrality[node] /= 2.0
    normalization = ((count - 1) * (count - 2)) / 2.0
    return {node: min(1.0, value / normalization) for node, value in centrality.items()}


def antigen_retention(
    gene_id: str,
    antigen_scores: Mapping[str, float],
    operon_neighbors: Mapping[str, Iterable[str]],
    interaction_neighbors: Mapping[str, Iterable[str]],
    normalized_centrality: Mapping[str, float],
) -> float | None:
    if gene_id not in antigen_scores:
        return None
    if any(not math.isfinite(float(score)) or not 0 <= score <= 1 for score in antigen_scores.values()):
        raise ValueError("antigen scores must be finite probabilities")
    own = float(antigen_scores[gene_id])
    operon = set(map(str, operon_neighbors.get(gene_id, ()))) & antigen_scores.keys()
    interactions = set(map(str, interaction_neighbors.get(gene_id, ()))) & antigen_scores.keys()
    centrality = float(normalized_centrality.get(gene_id, 0.0))
    operon_mean = (own + sum(float(antigen_scores[node]) for node in operon)) / (1 + len(operon))
    interaction_mean = (
        own + sum(float(antigen_scores[node]) for node in interactions)
    ) / (1 + len(interactions))
    return math.exp(-DEFAULTS["ret_centrality_lambda"] * centrality) * operon_mean * interaction_mean


def safe_score(flags: Mapping[str, bool] | None) -> dict[str, Any]:
    if flags is None:
        return {"score": None, "status": "not_assessed", "flag_count": None, "flags": {}}
    allowed = {"mge", "amr", "phase_variation", "toxicity"}
    unknown = set(flags) - allowed
    if unknown:
        raise ValueError(f"unsupported safety flags: {sorted(unknown)}")
    normalized = {name: bool(value) for name, value in flags.items()}
    if set(normalized) != allowed:
        return {
            "score": None,
            "status": "partial_assessment",
            "flag_count": sum(normalized.values()),
            "unassessed_flags": sorted(allowed - set(normalized)),
            "flags": normalized,
        }
    normalized = {name: normalized[name] for name in sorted(allowed)}
    count = sum(normalized.values())
    score = 1.0 - min(1.0, DEFAULTS["safe_flag_penalty"] * count)
    return {"score": score, "status": "assessed", "flag_count": count, "flags": normalized}


def three_reader_consensus(
    rankings: Sequence[Sequence[str]], top_k: int = 50
) -> dict[str, Any]:
    if len(rankings) != 3:
        raise ValueError("exactly three reader rankings are required")
    top_sets = [set(map(str, ranking[:top_k])) for ranking in rankings]
    counts: dict[str, int] = defaultdict(int)
    for group in top_sets:
        for gene in group:
            counts[gene] += 1
    overlaps = []
    for left, right in itertools.combinations(top_sets, 2):
        denominator = max(1, top_k)
        overlaps.append(len(left & right) / denominator)
    return {
        "counts": dict(counts),
        "agreement_threshold": 2,
        "pairwise_top_k_overlap": overlaps,
        "systematic_disagreement": bool(overlaps) and float(np.mean(overlaps)) < 0.3,
    }


def prototype_scores(
    query_vectors: np.ndarray, support_vectors: np.ndarray
) -> np.ndarray:
    query = np.asarray(query_vectors, dtype=np.float64)
    support = np.asarray(support_vectors, dtype=np.float64)
    if query.ndim != 2 or support.ndim != 2 or query.shape[1] != support.shape[1]:
        raise ValueError("query and support vectors must be compatible 2D matrices")
    if not len(support):
        raise ValueError("prototype scoring requires at least one support vector")
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    if np.any(query_norm == 0) or np.any(support_norm == 0):
        raise ValueError("prototype scoring does not accept zero vectors")
    similarities = (query / query_norm) @ (support / support_norm).T
    return similarities.mean(axis=1)


def attention_position_scores(
    attention_weights: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    attention = np.asarray(attention_weights, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if attention.ndim != 4:
        raise ValueError("attention_weights must have shape [batch, heads, query, key]")
    if mask.shape != (attention.shape[0], attention.shape[2]):
        raise ValueError("valid_mask must match the batch and query dimensions")
    if attention.shape[2] != attention.shape[3]:
        raise ValueError("attention query/key dimensions must be equal")
    scores = (attention * mask[:, None, :, None]).sum(axis=1).sum(axis=1)
    lengths = mask.sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("each sequence must contain at least one valid residue")
    scores = scores / lengths[:, None]
    scores *= mask
    for index, valid in enumerate(mask):
        values = scores[index, valid]
        span = float(values.max() - values.min())
        scores[index, valid] = (values - values.min()) / span if span > 0 else 0.0
    return scores


def gradient_contributions(
    attenuation: float, retention: float, safety: float
) -> dict[str, float]:
    values = {"Att": attenuation, "Ret": retention, "Safe": safety}
    output = {}
    for name, value in values.items():
        if not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite probability")
        output[name] = math.log(value) if value > 0 else 0.0
    return output


def acquisition_scores(
    intervals: Mapping[str, tuple[float, float]],
    consensus_counts: Mapping[str, int],
    attenuation: Mapping[str, float],
    retention: Mapping[str, float],
) -> dict[str, float]:
    gene_ids = sorted(set(intervals) & set(consensus_counts) & set(attenuation) & set(retention))
    if not gene_ids:
        return {}
    widths = {}
    disagreement = {}
    threshold_gap = {}
    for gene_id in gene_ids:
        low, high = map(float, intervals[gene_id])
        if not 0 <= low <= high <= 1:
            raise ValueError(f"{gene_id}: invalid attenuation interval")
        count = int(consensus_counts[gene_id])
        if not 0 <= count <= 3:
            raise ValueError(f"{gene_id}: consensus count must be in [0, 3]")
        att, ret = float(attenuation[gene_id]), float(retention[gene_id])
        if not 0 <= att <= 1 or not 0 <= ret <= 1:
            raise ValueError(f"{gene_id}: attenuation and retention must be in [0, 1]")
        widths[gene_id] = (high - low) / 2.0
        disagreement[gene_id] = float(3 - count)
        threshold_gap[gene_id] = max(0.6 - att, 0.0) + max(0.6 - ret, 0.0)
    width_norm = minmax_normalize(widths)
    disagreement_norm = minmax_normalize(disagreement)
    gap_norm = minmax_normalize(threshold_gap)
    return {
        gene_id: width_norm[gene_id]
        * disagreement_norm[gene_id]
        * (1.0 - gap_norm[gene_id])
        for gene_id in gene_ids
    }


def in_silico_knockout(
    gce,
    node_features,
    edge_index,
    edge_type,
    edge_weight,
    node_index: int,
) -> dict[str, Any]:
    """Compare GCE outputs before/after removing one node; report neighbor displacement."""
    import torch

    node_count = node_features.shape[0]
    if not 0 <= node_index < node_count:
        raise IndexError("node_index lies outside the graph")
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=node_features.device)
    src, dst = edge_index.to(device=node_features.device, dtype=torch.long)
    neighbor_indices = torch.unique(
        torch.cat((src[dst == node_index], dst[src == node_index]))
    )
    neighbor_indices = neighbor_indices[neighbor_indices != node_index]
    was_training = gce.training
    gce.eval()
    try:
        with torch.no_grad():
            baseline = gce(node_features, edge_index, edge_type, edge_weight)
            keep = torch.arange(node_count, device=node_features.device) != node_index
            remap = torch.full((node_count,), -1, dtype=torch.long, device=node_features.device)
            remap[keep] = torch.arange(int(keep.sum()), device=node_features.device)
            edge_keep = keep[src] & keep[dst]
            changed_edges = torch.stack((remap[src[edge_keep]], remap[dst[edge_keep]]))
            changed_types = edge_type[edge_keep] if edge_type is not None else None
            changed_weights = edge_weight[edge_keep] if edge_weight is not None else None
            perturbed = gce(
                node_features[keep], changed_edges, changed_types, changed_weights
            )
            if neighbor_indices.numel():
                displacement = torch.linalg.vector_norm(
                    perturbed[remap[neighbor_indices]] - baseline[neighbor_indices], dim=-1
                )
                mean_displacement = float(displacement.mean())
            else:
                displacement = torch.empty((0,), device=node_features.device)
                mean_displacement = 0.0
    finally:
        gce.train(was_training)
    return {
        "node_index": node_index,
        "neighbor_indices": neighbor_indices.cpu().tolist(),
        "neighbor_displacement": displacement.cpu().tolist(),
        "isk": mean_displacement,
    }


def pareto_tiers(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for row in output:
        for field in ("att_estimate", "ret", "safe", "h4_probability"):
            value = row.get(field)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"{field} must be null or a finite probability")
    usable = [
        i
        for i, row in enumerate(output)
        if row.get("att_estimate") is not None and row.get("ret") is not None
        and not (
            row.get("h4_probability") is not None
            and float(row["h4_probability"]) > DEFAULTS["h4_veto_threshold"]
        )
    ]
    dominated: set[int] = set()
    for i in usable:
        for j in usable:
            if i == j:
                continue
            left, right = output[j], output[i]
            if (
                float(left["att_estimate"]) >= float(right["att_estimate"])
                and float(left["ret"]) >= float(right["ret"])
                and (
                    float(left["att_estimate"]) > float(right["att_estimate"])
                    or float(left["ret"]) > float(right["ret"])
                )
            ):
                dominated.add(i)
                break
    for index, row in enumerate(output):
        reasons: list[str] = list(row.get("degrade_reasons", []))
        if index not in usable:
            row["tier"] = "unranked"
            reasons.append("attenuation_or_antigen_retention_unavailable")
        elif index in dominated:
            row["tier"] = "C"
            reasons.append("pareto_dominated")
        else:
            agreement = row.get("consensus_count")
            above_thresholds = (
                float(row["att_estimate"]) >= DEFAULTS["tier_a_att"]
                and float(row["ret"]) >= DEFAULTS["tier_a_ret"]
            )
            if above_thresholds and agreement is not None and int(agreement) >= 2:
                row["tier"] = "A"
            else:
                row["tier"] = "B"
                if not above_thresholds:
                    reasons.append("below_tier_a_component_threshold")
                if agreement is None or int(agreement) < 2:
                    reasons.append("three_reader_consensus_not_met")
        safe = row.get("safe")
        if safe is not None and row.get("att_estimate") is not None and row.get("ret") is not None:
            row["utility"] = (
                float(row["att_estimate"]) ** DEFAULTS["utility_att_power"]
                * float(row["ret"]) ** DEFAULTS["utility_ret_power"]
                * float(safe) ** DEFAULTS["utility_safe_power"]
            )
        else:
            row["utility"] = None
        h4 = row.get("h4_probability")
        row["hard_veto"] = h4 is not None and float(h4) > DEFAULTS["h4_veto_threshold"]
        if row["hard_veto"]:
            row["tier"] = "excluded_h4_core_essential"
            reasons.append("h4_probability_above_hard_veto_threshold")
        row["degrade_reasons"] = sorted(set(map(str, reasons)))
        row.setdefault("implementation_path", "")
        row.setdefault("biological_use_status", "research_only_pending_target_validation")
    return output


def combination_scores(
    candidate_ids: Sequence[str],
    att: Mapping[str, float],
    ret: Mapping[str, float],
    safe: Mapping[str, float],
    antigen_scores: Mapping[str, float],
    operon_neighbors: Mapping[str, Iterable[str]],
    interaction_neighbors: Mapping[str, Iterable[str]],
    h4_probabilities: Mapping[str, float],
    all_gene_ids: Sequence[str],
    max_size: int = 3,
    shortlist_size: int = 60,
    att_intervals: Mapping[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    if max_size < 1:
        raise ValueError("max_size must be positive")
    shortlist = sorted(
        set(candidate_ids) & att.keys() & ret.keys() & safe.keys(),
        key=lambda gene: att[gene] * ret[gene] * safe[gene],
        reverse=True,
    )[:shortlist_size]
    combinations = []
    for size in range(1, min(max_size, len(shortlist)) + 1):
        for members_tuple in itertools.combinations(shortlist, size):
            members = set(members_tuple)
            if any(h4_probabilities.get(gene, -math.inf) > DEFAULTS["h4_veto_threshold"] for gene in members):
                continue
            excluded = set(members)
            for gene in members:
                excluded.update(operon_neighbors.get(gene, ()))
                excluded.update(interaction_neighbors.get(gene, ()))
            total_antigen = sum(float(antigen_scores.get(gene, 0.0)) for gene in all_gene_ids)
            retained = sum(
                float(antigen_scores.get(gene, 0.0))
                for gene in set(all_gene_ids) - excluded
            )
            result = {
                "gene_ids": sorted(members),
                "att_comb": 1.0 - math.prod(1.0 - float(att[gene]) for gene in members),
                "ret_comb": retained / total_antigen if total_antigen > 0 else None,
                "safe_comb": min(float(safe[gene]) for gene in members),
                "interval_method": "member_endpoint_propagation_not_independently_calibrated",
            }
            if att_intervals is not None and members.issubset(att_intervals):
                intervals = [att_intervals[gene] for gene in members]
                if any(not 0 <= low <= high <= 1 for low, high in intervals):
                    raise ValueError("member attenuation intervals must lie in [0, 1]")
                result["att_comb_interval"] = {
                    "lower": 1.0 - math.prod(1.0 - low for low, _ in intervals),
                    "upper": 1.0 - math.prod(1.0 - high for _, high in intervals),
                    "calibration": "endpoint_propagation_not_independent",
                }
            else:
                result["att_comb_interval"] = None
            combinations.append(result)
    return combinations
