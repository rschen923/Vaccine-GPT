from __future__ import annotations

import math
from typing import Any, Mapping


AEP_FIELDS = (
    "gene_or_product",
    "att_interval",
    "ret",
    "safe",
    "evidence_level",
    "three_reader_consistency",
    "mechanism_annotation",
    "warning",
    "implementation_path",
)
EXTERNAL_PARAMETERS = (
    "dose_and_immunization_schedule",
    "delivery_platform_and_route",
    "target_population_and_equity",
    "clinical_trial_design",
)


def unavailable_aep_row(
    gene_or_product: str,
    warning: str,
    evidence_level: str = "E4",
    version: str = "v1",
) -> dict[str, Any]:
    if evidence_level not in {"E1", "E2", "E3", "E4"}:
        raise ValueError("evidence_level must be E1, E2, E3, or E4")
    row = {
        "gene_or_product": gene_or_product,
        "att_interval": {"lower": 0.0, "upper": 1.0, "status": "unavailable"},
        "ret": None,
        "safe": None,
        "evidence_level": evidence_level,
        "three_reader_consistency": {"status": "not_assessed"},
        "mechanism_annotation": None,
        "warning": warning,
        "implementation_path": "" if version == "v1" else None,
    }
    validate_aep_row(row, version)
    return row


def validate_aep_row(row: Mapping[str, Any], version: str = "v1") -> None:
    missing = [field for field in AEP_FIELDS if field not in row]
    if missing:
        raise ValueError(f"AEP row is missing required fields: {missing}")
    if not str(row["gene_or_product"]).strip():
        raise ValueError("AEP gene_or_product must be non-empty")
    if row["evidence_level"] not in {"E1", "E2", "E3", "E4"}:
        raise ValueError("AEP evidence_level must be E1-E4")
    interval = row["att_interval"]
    if not isinstance(interval, Mapping) or not {"lower", "upper"} <= set(interval):
        raise ValueError("AEP att_interval must contain lower and upper")
    lower, upper = float(interval["lower"]), float(interval["upper"])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1):
        raise ValueError("AEP attenuation interval must satisfy 0 <= lower <= upper <= 1")
    for field in ("ret", "safe"):
        value = row[field]
        if value is not None and (
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"AEP {field} must be null or a probability")
    if version == "v1" and row["implementation_path"] != "":
        raise ValueError("v1 AEP implementation_path must be empty")


def external_parameter_declaration() -> dict[str, Any]:
    return {
        name: {
            "status": "external_not_computed",
            "model_output": False,
            "value": None,
        }
        for name in EXTERNAL_PARAMETERS
    }
