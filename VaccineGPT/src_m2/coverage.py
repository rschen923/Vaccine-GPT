from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def calculate_multivalent_coverage(
    antigen_ids: Sequence[str],
    serotype_weights: Mapping[str, float],
    presence: Mapping[str, Mapping[str, float]],
    expression: Mapping[str, Mapping[str, float]],
    p_on_intervals: Mapping[str, Mapping[str, tuple[float, float]]],
) -> dict[str, Any]:
    """Calculate the declared coverage expression only where all inputs exist."""
    if not serotype_weights:
        return {"coverage": None, "status": "no_serotype_weights", "by_serotype": {}}
    if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in serotype_weights.values()):
        raise ValueError("serotype weights must be finite and nonnegative")
    total_weight = sum(float(weight) for weight in serotype_weights.values())
    if total_weight <= 0:
        raise ValueError("at least one serotype weight must be positive")
    by_serotype: dict[str, dict[str, float]] = {}
    for serotype in serotype_weights:
        lower_noncoverage = 1.0
        upper_noncoverage = 1.0
        missing: list[str] = []
        for antigen in antigen_ids:
            if antigen not in presence or serotype not in presence[antigen]:
                missing.append(f"presence:{antigen}:{serotype}")
                continue
            if antigen not in expression or serotype not in expression[antigen]:
                missing.append(f"expression:{antigen}:{serotype}")
                continue
            if antigen not in p_on_intervals or serotype not in p_on_intervals[antigen]:
                missing.append(f"p_on:{antigen}:{serotype}")
                continue
            present = float(presence[antigen][serotype])
            expressed = float(expression[antigen][serotype])
            on_low, on_high = map(float, p_on_intervals[antigen][serotype])
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (present, expressed, on_low, on_high)):
                raise ValueError("presence, expression, and P(ON) values must be probabilities")
            if on_low > on_high:
                raise ValueError("P(ON) interval lower bound exceeds upper bound")
            lower_noncoverage *= 1.0 - present * expressed * on_low
            upper_noncoverage *= 1.0 - present * expressed * on_high
        if missing:
            return {
                "coverage": None,
                "status": "incomplete_inputs",
                "missing_inputs": missing,
                "by_serotype": by_serotype,
            }
        by_serotype[serotype] = {
            "lower": 1.0 - lower_noncoverage,
            "upper": 1.0 - upper_noncoverage,
        }
    lower = sum(
        float(serotype_weights[key]) * item["lower"]
        for key, item in by_serotype.items()
    ) / total_weight
    upper = sum(
        float(serotype_weights[key]) * item["upper"]
        for key, item in by_serotype.items()
    ) / total_weight
    return {
        "coverage": {"lower": lower, "upper": upper},
        "status": "estimated_from_input_probabilities",
        "by_serotype": by_serotype,
        "non_model_inputs": ["serotype_weights", "presence", "expression", "P(ON)"],
    }
