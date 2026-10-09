from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STUDY_DEFAULTS = {
    "PRJNA480066": {
        "assay": "Tn-seq",
        "condition_patterns": [(r"16.?c", "seawater_16C"), (r"28.?c", "seawater_28C")],
    },
    "PRJNA1082880": {
        "assay": "Tn-seq",
        "condition_patterns": [(r"low.?glucose", "low_glucose"), (r"\bDMEM\b", "DMEM")],
    },
    "PRJNA1113837": {
        "assay": "CRISPRi-seq",
        "condition_patterns": [
            (r"LB group|control", "LB_control"),
            (r"in vivo essential", "in_vivo_essential"),
            (r"DMEM", "DMEM"),
            (r"seawater", "seawater"),
            (r"iron.?limiting", "iron_limiting"),
            (r"polymyxin", "polymyxin_B"),
            (r"ROS tolerance", "ROS"),
            (r"acid tolerance", "acid"),
            (r"esrB", "esrB"),
            (r"bottleneck", "turbot_bottleneck"),
        ],
    },
}


def study_accession(path: Path) -> str | None:
    match = re.search(r"(PRJNA\d+)", str(path), re.IGNORECASE)
    return match.group(1).upper() if match else None


def normalize_condition(study: str, text: str) -> str:
    for pattern, condition in STUDY_DEFAULTS.get(study, {}).get("condition_patterns", []):
        if re.search(pattern, text, flags=re.IGNORECASE):
            return condition
    return "unresolved"


def read_ena_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def fastq_runs(dataset_root: Path) -> set[str]:
    runs = set()
    for path in dataset_root.glob("raw/public_training_sources/studies/PRJNA*/fastq/*"):
        match = re.search(r"(SRR\d+)", path.name, re.IGNORECASE)
        if match:
            runs.add(match.group(1).upper())
    return runs


def read_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    response_root = dataset_root / "raw" / "public_training_sources" / "source_responses"
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(response_root.glob("ena_PRJNA*.tsv")):
        study = study_accession(path)
        if study not in STUDY_DEFAULTS:
            continue
        for row in read_ena_rows(path):
            run = str(row.get("run_accession") or "").upper()
            if not run:
                continue
            key = (study or "", run)
            if key in seen:
                continue
            seen.add(key)
            title = " ".join(
                str(row.get(field) or "")
                for field in ("experiment_title", "study_title", "description")
            ).strip()
            condition = normalize_condition(study or "", title)
            rows.append(
                {
                    "study_accession": study,
                    "run_accession": run,
                    "sample_accession": row.get("sample_accession") or None,
                    "library_strategy": row.get("library_strategy") or None,
                    "experiment_title": row.get("experiment_title") or None,
                    "study_title": row.get("study_title") or None,
                    "condition": condition,
                    "fastq_present": run in fastq_runs(dataset_root),
                    "metadata_source": str(path.relative_to(dataset_root)),
                    "metadata_status": (
                        "condition_inferred_from_title"
                        if condition != "unresolved"
                        else "condition_unresolved"
                    ),
                }
            )
    if not rows:
        raise ValueError(f"no supported ENA metadata found under {response_root}")
    return rows


def load_gene_effects(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


def gate_gene_effects(
    records: Iterable[dict[str, Any]], sample_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    sample_counts = Counter(
        (row["study_accession"], row["condition"]) for row in sample_rows
    )
    output: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        required = ("gene_id", "study_accession", "condition", "effect_size")
        missing = [field for field in required if record.get(field) in (None, "")]
        if missing:
            reasons["missing_gene_effect_fields"] += 1
            continue
        key = (str(record["study_accession"]), str(record["condition"]))
        replicates = int(record.get("independent_replicates", sample_counts[key]))
        qc_pass = bool(record.get("qc_pass", False))
        mapped = bool(record.get("reference_mapped", False))
        has_control = bool(record.get("control_defined", False))
        status = "eligible_E2" if qc_pass and mapped and has_control and replicates >= 2 else "audit_only"
        if status != "eligible_E2":
            reasons["insufficient_gene_level_evidence"] += 1
        output.append(
            {
                **record,
                "task": "H1",
                "assay_scope": "conditional_gene_fitness",
                "evidence_level": "E2" if status == "eligible_E2" else "E4",
                "training_eligible_as_gold": status == "eligible_E2",
                "label_status": status,
                "label_semantics": (
                    "Condition-specific gene fitness effect from mapped, QC-passed "
                    "target-organism sequencing with an explicit control and independent "
                    "replication."
                    if status == "eligible_E2"
                    else "Raw or incompletely audited evidence; not a binary H1 gold label."
                ),
            }
        )
    return output, reasons


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def process(dataset_root: Path, output_dir: Path, gene_effects: Path | None = None) -> dict[str, Any]:
    samples = read_metadata(dataset_root)
    effects, gate_reasons = gate_gene_effects(load_gene_effects(gene_effects), samples)
    by_study = defaultdict(list)
    for row in samples:
        by_study[row["study_accession"]].append(row)
    study_summary = {}
    for study, rows in sorted(by_study.items()):
        conditions = Counter(row["condition"] for row in rows)
        study_summary[study] = {
            "run_count": len(rows),
            "fastq_run_count": sum(row["fastq_present"] for row in rows),
            "conditions": dict(conditions),
            "library_strategies": dict(Counter(row["library_strategy"] for row in rows)),
            "requires_manual_design_review": True,
        }
    sample_count = write_jsonl(output_dir / "h1_sample_manifest.jsonl", samples)
    effect_count = write_jsonl(output_dir / "h1_gene_evidence.jsonl", effects)
    training_ready = any(row["training_eligible_as_gold"] for row in effects)
    report = {
        "schema_version": "vaccinegpt-eib202-h1-audit-1.0",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "studies": study_summary,
        "sample_record_count": sample_count,
        "gene_effect_record_count": effect_count,
        "training_ready": training_ready,
        "eligible_E2_gene_effect_count": sum(
            row["training_eligible_as_gold"] for row in effects
        ),
        "gate_reasons": dict(gate_reasons),
        "policy": {
            "raw_reads_are_not_gene_labels": True,
            "unresolved_conditions_are_not_negative_labels": True,
            "single_run_conditions_are_audit_only": True,
            "E2_requires_qc_mapping_control_and_two_independent_replicates": True,
            "PRJNA1113837_metadata_is_included_only_when_archived": True,
            "PRJNA1113837_reads_require_explicit_download_and_checksum_verification": True,
        },
        "sha256": hashlib.sha256(
            json.dumps(samples, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h1_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit EIB202 CRISPRi/Tn-seq inputs and gate H1 gene effects."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gene-effects",
        type=Path,
        help="Optional mapped gene-effect JSONL; raw reads alone never become H1 labels.",
    )
    args = parser.parse_args()
    print(json.dumps(process(args.dataset_root, args.output_dir, args.gene_effects), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
