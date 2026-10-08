from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AMINO_ACIDS = set("XBZJUO")
EVIDENCE_WEIGHTS = {"E1": 1.0, "E2": 0.8, "E3": 0.4, "E4": 0.2}


def normalize_protein(raw_sequence: Any, sequence_role: str = "protein") -> dict[str, Any]:
    if sequence_role not in {"protein", "epitope_peptide", "unknown"}:
        raise ValueError(f"unsupported sequence role: {sequence_role}")
    raw = str(raw_sequence or "")
    sequence = "".join(raw.split()).upper()
    terminal_stop = sequence.endswith("*")
    if terminal_stop:
        sequence = sequence[:-1]
    gaps = sequence.count("-")
    sequence = sequence.replace("-", "")
    invalid = sorted(set(sequence) - PROTEIN_ALPHABET - AMBIGUOUS_AMINO_ACIDS)
    ambiguous = sorted(set(sequence) & AMBIGUOUS_AMINO_ACIDS)
    canonical = "".join(letter for letter in sequence if letter in PROTEIN_ALPHABET | AMBIGUOUS_AMINO_ACIDS)
    reasons: list[str] = []
    if not canonical:
        reasons.append("empty_after_normalization")
    if invalid:
        reasons.append("invalid_characters")
    if sequence_role == "epitope_peptide":
        if not 8 <= len(canonical) <= 30:
            reasons.append("peptide_length_outside_common_epitope_review_range")
    elif len(canonical) < 20:
        reasons.append("short_protein_review")
    if len(canonical) > 100000:
        reasons.append("implausibly_long_protein")
    if ambiguous:
        reasons.append("ambiguous_residues")
    if gaps:
        reasons.append("sequence_gaps_removed")
    return {
        "sequence": canonical,
        "sequence_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest() if canonical else None,
        "length": len(canonical),
        "terminal_stop_removed": terminal_stop,
        "gap_count_removed": gaps,
        "ambiguous_residues": ambiguous,
        "invalid_characters": invalid,
        "quality_status": "invalid" if invalid or not canonical or len(canonical) > 100000 else (
            "review" if reasons else "pass"
        ),
        "quality_reasons": reasons,
    }


def classify_label_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Conservatively map explicit row provenance to E1-E4 without source-name promotion."""
    source = str(record.get("source") or record.get("database") or "unknown").strip()
    normalized_source = source.casefold()
    private = bool(record.get("is_private") or record.get("private_dataset"))
    assay = str(record.get("assay") or record.get("assay_type") or "").strip()
    method = str(record.get("method") or record.get("evidence_type") or "").strip().casefold()
    reference = next(
        (
            str(record.get(field)).strip()
            for field in ("doi", "pmid", "pmc", "pmcid", "primary_reference", "publication")
            if record.get(field)
        ),
        "",
    )
    organism = str(
        record.get("organism") or record.get("taxon") or record.get("target_organism") or ""
    ).casefold()
    target_confirmed = any(token in organism for token in ("piscicida", "eib202"))
    computational = any(
        token in f"{normalized_source} {method}"
        for token in ("prediction", "computational", "in silico", "homology", "blast", "model score")
    )
    direct_experiment = bool(assay or method) and any(
        token in f"{assay} {method}".casefold()
        for token in (
            "mutant",
            "knockout",
            "knockdown",
            "challenge",
            "infection",
            "tn-seq",
            "crispri",
            "rna-seq",
            "proteomics",
            "growth assay",
            "phenotype",
        )
    )
    if private and direct_experiment:
        level, rationale = "E2", "private_measured_assay_explicitly_recorded"
    elif direct_experiment and reference and target_confirmed:
        level, rationale = "E1", "target_organism_experiment_with_row_level_reference"
    elif computational:
        level, rationale = "E3", "public_computational_or_predicted_evidence"
    else:
        level, rationale = "E4", "primary_assay_or_target_organism_provenance_unverified"
    eligible = level in {"E1", "E2"}
    return {
        "evidence_level": level,
        "evidence_weight": EVIDENCE_WEIGHTS[level],
        "evidence_rationale": rationale,
        "training_role": "direct_supervision" if eligible else "weak_supervision_only",
        "training_eligible_as_gold": eligible,
        "requires_manual_provenance_review": not eligible,
        "provenance": {
            "source": source,
            "assay": assay or None,
            "reference": reference or None,
            "target_organism_confirmed": target_confirmed,
        },
    }


def detect_legacy_identifier_problem(gene_id: Any) -> list[str]:
    value = str(gene_id or "").strip()
    flags: list[str] = []
    if not value:
        flags.append("missing_gene_id")
    if re.search(r"\((?:gb|ref|emb|dbj)$", value, re.IGNORECASE):
        flags.append("possible_truncated_accession_header")
    if re.fullmatch(r"VFG\d+\(gb\|[A-Z0-9_.]+\)", value):
        return flags
    if "|" in value or any(character.isspace() for character in value):
        flags.append("compound_or_whitespace_identifier")
    return flags


def resolve_label_group(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep every source row and quarantine opposite labels rather than averaging them."""
    evidence = [dict(record) for record in records]
    numeric = []
    for record in evidence:
        try:
            value = float(record.get("label"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            numeric.append(value)
    distinct = sorted(set(numeric))
    conflict = len(distinct) > 1
    levels = [record.get("evidence_level") for record in evidence]
    if conflict:
        role = "quarantine_conflict"
    elif any(level in {"E1", "E2"} for level in levels):
        role = "direct_supervision"
    elif evidence:
        role = "weak_supervision_only"
    else:
        role = "unlabelled"
    return {
        "records": evidence,
        "label_conflict": conflict,
        "observed_labels": distinct,
        "training_role": role,
    }


def group_labels_by_gene_task(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record.get("gene_id") or ""), str(record.get("task") or ""))
        groups[key].append(dict(record))
    return {key: resolve_label_group(items) for key, items in groups.items()}
