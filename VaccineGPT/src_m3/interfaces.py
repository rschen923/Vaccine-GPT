from __future__ import annotations

from typing import Any, Mapping, Sequence

from src_m2.coverage import calculate_multivalent_coverage


def subunit_interface(
    retained_antigen_ids: Sequence[str],
    presentation: Mapping[str, Any] | None = None,
    immunogenicity: Mapping[str, Any] | None = None,
    protection: Mapping[str, Any] | None = None,
    coverage_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """M3-S schema only: presentation, immunogenicity, and protection stay distinct."""
    presentation = dict(presentation or {})
    immunogenicity = dict(immunogenicity or {})
    protection = dict(protection or {})
    coverage = (
        calculate_multivalent_coverage(
            retained_antigen_ids,
            coverage_inputs.get("serotype_weights", {}),
            coverage_inputs.get("presence", {}),
            coverage_inputs.get("expression", {}),
            coverage_inputs.get("p_on_intervals", {}),
        )
        if coverage_inputs is not None
        else {"coverage": None, "status": "interface_only_missing_external_inputs"}
    )
    return {
        "interface": "M3-S",
        "direction": "read_only_from_M1_M2",
        "retained_antigen_ids": list(retained_antigen_ids),
        "presentation": presentation or {"status": "not_assessed"},
        "immunogenicity": immunogenicity or {"status": "not_assessed"},
        "protection": protection or {"status": "not_assessed"},
        "coverage": coverage,
        "semantic_warning": "presentation is not immunogenicity; immunogenicity is not protection",
        "writes_back_to_core": False,
    }


def nucleic_acid_interface(
    antigen_id: str,
    coding_sequence: str | None = None,
    codon_usage_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """M3-N contract only; no codon redesign is claimed without a validated host table."""
    if not antigen_id:
        raise ValueError("antigen_id is required")
    if coding_sequence is not None:
        coding_sequence = "".join(coding_sequence.upper().split())
        if len(coding_sequence) % 3:
            raise ValueError("coding_sequence length must be divisible by three")
        if set(coding_sequence) - set("ACGT"):
            raise ValueError("coding_sequence must contain only A/C/G/T")
    if codon_usage_weights is not None and not codon_usage_weights:
        raise ValueError("codon_usage_weights cannot be an empty mapping")
    return {
        "interface": "M3-N",
        "direction": "read_only_from_M1_M2",
        "antigen_id": antigen_id,
        "coding_sequence_available": coding_sequence is not None,
        "codon_usage_table_available": codon_usage_weights is not None,
        "status": "interface_only_not_optimized",
        "writes_back_to_core": False,
        "external_parameters": [
            "expression_system",
            "delivery_platform",
            "dose",
            "immunization_schedule",
        ],
    }
