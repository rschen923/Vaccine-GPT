from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
TARGET_TAXON_PATTERN = re.compile(r"\b(edwardsiella|piscicida|tarda)\b", re.IGNORECASE)
HUMAN_PATTERN = re.compile(r"\b(homo sapiens|human|taxid\s*[:=]?\s*9606)\b", re.IGNORECASE)
MODEL_VERSION = "VaccineGPT-GEN-H3-T3-0.1.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
            records.append(record)
    return records


def _sequence(record: dict[str, Any], source: Path) -> str:
    sequence = "".join(str(record.get("protein_sequence") or record.get("sequence") or "").split()).upper()
    invalid = sorted(set(sequence) - AA_ALPHABET)
    if not sequence or invalid:
        raise ValueError(
            f"{source}: invalid or empty protein sequence for "
            f"{record.get('record_id') or record.get('gene_id') or 'unknown record'}: {invalid}"
        )
    return sequence


def _is_human_derived_record(record: dict[str, Any]) -> bool:
    organism = " ".join(
        str(record.get(field) or "")
        for field in ("organism", "taxon", "organism_strain", "taxid")
    )
    return bool(HUMAN_PATTERN.search(organism))


def _has_explicit_non_t3ss_pathway(annotation: dict[str, Any]) -> bool:
    pathway = re.sub(
        r"[\s_-]+",
        "",
        str(annotation.get("secretion_or_translocation_pathway") or "").casefold(),
    )
    return bool(
        pathway
        and pathway not in {"t3ss", "typeiiisecretionsystem", "typeiiisecretion"}
        and re.search(r"(?:t[1-6]ss|type(?:i|ii|iv|v|vi)secretion|omv|vesicle)", pathway)
    )


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "sequence_features",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    lowercase=False,
                    min_df=2,
                    max_features=150_000,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predicted = scores >= 0.0
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "class_counts": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
        },
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy_at_zero_logit": float(balanced_accuracy_score(labels, predicted)),
        "confusion_matrix_at_zero_logit": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "score_type": "uncalibrated_logit",
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_rank(
    positive_path: Path,
    negative_path: Path,
    independent_test_path: Path,
    proteome_path: Path,
    target_positive_path: Path,
    output_dir: Path,
    folds: int = 5,
    top_k: int = 50,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_train: list[dict[str, Any]] = []
    excluded_target_taxon = 0
    excluded_human = 0
    for label, path in ((1, positive_path), (0, negative_path)):
        for record in read_jsonl(path):
            if _is_human_derived_record(record):
                excluded_human += 1
                continue
            organism = str(record.get("organism") or record.get("source_header") or "")
            if TARGET_TAXON_PATTERN.search(organism):
                excluded_target_taxon += 1
                continue
            if int(record.get("label", -1)) != label:
                raise ValueError(f"{path}: source file contains a row with an unexpected label")
            record = dict(record)
            record["_sequence"] = _sequence(record, path)
            if not organism.strip():
                raise ValueError(f"{path}: organism is required for organism-grouped validation")
            record["_organism_group"] = organism.strip().casefold()
            source_train.append(record)

    train_sequences = [record["_sequence"] for record in source_train]
    train_labels = np.asarray([int(record["label"]) for record in source_train], dtype=np.int64)
    train_groups = np.asarray([record["_organism_group"] for record in source_train])
    if set(train_labels.tolist()) != {0, 1}:
        raise ValueError("public T3SEpp training data must contain both benchmark classes")
    if len(set(train_groups.tolist())) < folds:
        raise ValueError("too few independent organism groups for requested cross-validation")

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    cross_validated_scores = np.full(len(train_labels), np.nan, dtype=np.float64)
    fold_reports = []
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(train_sequences, train_labels, train_groups), start=1
    ):
        if set(train_labels[train_indices].tolist()) != {0, 1}:
            raise ValueError(f"fold {fold} training partition lacks a benchmark class")
        if set(train_labels[valid_indices].tolist()) != {0, 1}:
            raise ValueError(f"fold {fold} validation partition lacks a benchmark class")
        pipeline = _pipeline()
        pipeline.fit(
            [train_sequences[index] for index in train_indices],
            train_labels[train_indices],
        )
        scores = pipeline.decision_function(
            [train_sequences[index] for index in valid_indices]
        )
        cross_validated_scores[valid_indices] = scores
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train_indices)),
                "validation_rows": int(len(valid_indices)),
                "train_organism_groups": int(len(set(train_groups[train_indices].tolist()))),
                "validation_organism_groups": int(len(set(train_groups[valid_indices].tolist()))),
                "metrics": _metrics(train_labels[valid_indices], scores),
            }
        )
    if not np.isfinite(cross_validated_scores).all():
        raise RuntimeError("grouped cross-validation failed to score every training record")

    independent_records = []
    for record in read_jsonl(independent_test_path):
        if _is_human_derived_record(record):
            excluded_human += 1
            continue
        organism = str(record.get("organism") or "")
        if TARGET_TAXON_PATTERN.search(organism):
            excluded_target_taxon += 1
            continue
        sequence = _sequence(record, independent_test_path)
        if sequence in set(train_sequences):
            continue
        independent_records.append((record, sequence))
    if not independent_records:
        raise ValueError("independent test set is empty after species and exact-sequence exclusions")

    target_positives = read_jsonl(target_positive_path) if target_positive_path.is_file() else []
    target_positive_map: dict[str, dict[str, Any]] = {}
    for record in target_positives:
        if str(record.get("task") or "") != "H3":
            continue
        if int(record.get("label", -1)) != 1:
            raise ValueError(f"{target_positive_path}: only curated H3 positives are accepted")
        gene_id = str(record.get("gene_id") or "")
        if not gene_id:
            raise ValueError(f"{target_positive_path}: curated target H3 record lacks gene_id")
        target_positive_map[gene_id] = record

    model = _pipeline()
    model.fit(train_sequences, train_labels)
    external_sequences = [sequence for _, sequence in independent_records]
    external_labels = np.asarray(
        [int(record["label"]) for record, _ in independent_records], dtype=np.int64
    )
    if set(external_labels.tolist()) != {0, 1}:
        raise ValueError("independent test set must contain both benchmark classes")
    external_scores = model.decision_function(external_sequences)

    target_records = read_jsonl(proteome_path)
    target_sequences = [_sequence(record, proteome_path) for record in target_records]
    target_scores = model.decision_function(target_sequences)
    in_scope_indices = [
        index
        for index, record in enumerate(target_records)
        if not _has_explicit_non_t3ss_pathway(
            target_positive_map.get(str(record.get("gene_id") or ""), {})
        )
    ]
    score_order = sorted(
        in_scope_indices,
        key=lambda index: (-float(target_scores[index]), str(target_records[index].get("gene_id") or "")),
    )
    ranks = {index: rank for rank, index in enumerate(score_order, start=1)}
    predictions = []
    for index, (record, sequence, score) in enumerate(
        zip(target_records, target_sequences, target_scores)
    ):
        gene_id = str(record.get("gene_id") or "")
        known_positive = target_positive_map.get(gene_id)
        in_scope = index in ranks
        experimental_pathway = (
            str(known_positive.get("secretion_or_translocation_pathway") or "")
            if known_positive
            else None
        )
        candidate_rank = ranks.get(index)
        predictions.append(
            {
                "gene_id": gene_id,
                "gene_symbol": record.get("gene_symbol"),
                "product": record.get("product"),
                "protein_id": record.get("protein_id"),
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "task": "H3_T3SS_effector_sequence_benchmark",
                "model_version": MODEL_VERSION,
                "candidate_rank": candidate_rank,
                "uncalibrated_logit": float(score) if in_scope else None,
                "ranking_score_percentile": (
                    float((len(score_order) - candidate_rank) / max(1, len(score_order) - 1))
                    if candidate_rank is not None
                    else None
                ),
                "probability": None,
                "candidate_status": (
                    "published_experimental_positive_other_pathway_not_scored_by_T3_model"
                    if known_positive and not in_scope
                    else (
                        "published_experimental_positive_and_ranked_candidate"
                        if known_positive
                        else "sequence_ranked_candidate_not_experimental_positive"
                    )
                ),
                "experimental_positive_evidence": known_positive,
                "experimental_secretion_or_translocation_pathway": experimental_pathway,
                "label_scope": (
                    "T3SS effector benchmark transfer only; not vaccine protection or "
                    "EIB202 phenotype"
                ),
                "warning": (
                    "Exploratory cross-species ranking only. The score is not a calibrated "
                    "probability or an experimental result."
                ),
            }
        )
    predictions.sort(
        key=lambda row: (
            row["candidate_rank"] is None,
            row["candidate_rank"] or 0,
            row["gene_id"],
        )
    )

    model_path = output_dir / "h3_t3ss_sequence_ranker.joblib"
    joblib.dump(model, model_path)
    ranking_path = output_dir / "eib202_h3_t3ss_candidate_rankings.jsonl"
    prediction_count = _write_jsonl(ranking_path, predictions)
    scored_predictions = [row for row in predictions if row["candidate_rank"] is not None]
    shortlist = scored_predictions[: min(top_k, len(scored_predictions))]
    shortlist_path = output_dir / f"eib202_h3_t3ss_top_{len(shortlist)}.jsonl"
    _write_jsonl(shortlist_path, shortlist)
    known_positive_records = [
        row for row in predictions if row["experimental_positive_evidence"] is not None
    ]
    known_positive_path = output_dir / "eib202_h3_published_experimental_positives.jsonl"
    _write_jsonl(known_positive_path, known_positive_records)
    report = {
        "schema_version": "vaccinegpt-h3-public-ranker-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "task": "T3SS effector sequence benchmark transfer; not T6SS or vaccine protection",
        "input_contract": "full-length bacterial protein sequences; epitope peptides excluded",
        "representation": "TF-IDF amino-acid character n-grams (2-4)",
        "classifier": "class-weighted logistic regression; ranking logits are uncalibrated",
        "training_sources": [
            {
                "path": str(positive_path),
                "sha256": _sha256_file(positive_path),
                "label": 1,
                "source_dataset": "T3SEpp training benchmark",
            },
            {
                "path": str(negative_path),
                "sha256": _sha256_file(negative_path),
                "label": 0,
                "source_dataset": "T3SEpp training benchmark",
            },
        ],
        "excluded_target_taxon_rows": excluded_target_taxon,
        "excluded_human_organism_rows": excluded_human,
        "training_rows": int(len(train_labels)),
        "training_class_counts": {
            "negative": int((train_labels == 0).sum()),
            "positive": int((train_labels == 1).sum()),
        },
        "organism_grouped_cross_validation": {
            "grouping": "normalized organism field; whole organism held out per fold",
            "folds": fold_reports,
            "pooled_metrics": _metrics(train_labels, cross_validated_scores),
        },
        "official_independent_test": {
            "source": str(independent_test_path),
            "sha256": _sha256_file(independent_test_path),
            "exact_sequence_overlap_with_training_excluded": True,
            "metrics": _metrics(external_labels, external_scores),
        },
        "target_proteome": {
            "source": str(proteome_path),
            "sha256": _sha256_file(proteome_path),
            "genome_accession": "CP001135.1",
            "proteins_in_proteome": prediction_count,
            "proteins_scored_for_T3SS": len(scored_predictions),
            "known_non_T3SS_positives_excluded_from_T3_ranking": sum(
                str(record.get("gene_id") or "") in target_positive_map
                and index not in ranks
                for index, record in enumerate(target_records)
            ),
            "curated_EIB202_H3_positive_annotations": len(target_positive_map),
            "rankings_file": str(ranking_path),
            "candidate_shortlist_size": len(shortlist),
            "candidate_shortlist_file": str(shortlist_path),
            "published_experimental_positives_file": str(known_positive_path),
            "published_positive_count": len(known_positive_records),
        },
        "homology_leakage_status": (
            "MMseqs2 90%-identity clusters were not available in this environment. "
            "Validation holds out whole organism groups and excludes exact sequence overlap, "
            "but residual cross-species homology is not ruled out."
        ),
        "human_data_policy": (
            "No human participant/clinical records are loaded. Human-organism protein "
            "records are excluded; bacterial experiments using commercial cell lines, if "
            "documented in source annotations, are assay provenance only."
        ),
        "species_adaptation": {
            "status": "interface_reserved_not_fitted",
            "default": "128-parameter diagonal scale adapter on 128-dimensional shared features",
            "larger_option": "low-rank residual adapter; requires target-specific training and validation gates",
        },
        "artifacts": {
            "model": str(model_path),
            "model_sha256": _sha256_file(model_path),
            "candidate_rankings": str(ranking_path),
            "candidate_shortlist": str(shortlist_path),
            "published_experimental_positives": str(known_positive_path),
        },
        "limitations": [
            "This compact sequence baseline is not a recovered or fine-tuned CLEF checkpoint.",
            "Cross-species transfer is not target-strain experimental validation.",
            "The H3 classifier is T3SS benchmark-specific; it must not be applied as a T6SS label.",
            "Ranking percentiles are relative to this EIB202 proteome and are not probabilities.",
            "Explicitly documented non-T3SS experimental positives are retained as evidence but excluded from this T3SS ranking.",
            "Candidate selection requires independent biological review.",
        ],
    }
    (output_dir / "h3_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    default_data_root = Path(__file__).resolve().parents[2] / ".." / "data"
    parser = argparse.ArgumentParser(
        description=(
            "Train a task-isolated, organism-held-out public T3SS effector ranker "
            "and score the EIB202 proteome."
        )
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--proteome", type=Path)
    parser.add_argument("--target-positives", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    source_dir = (
        args.data_root
        / "processed"
        / "public_training_sources_20261007_055028"
    )
    proteome_path = args.proteome or (
        args.data_root
        / "processed"
        / "vaccinegpt_eib202_proteome_training_261007"
        / "eib202_proteome.jsonl"
    )
    target_positive_path = args.target_positives or (
        args.data_root
        / "raw"
        / "public_training_sources"
        / "m2_public_candidates"
        / "PMC5820615"
        / "curated_h3_eib202_positive_candidates.jsonl"
    )
    result = train_and_rank(
        source_dir / "t3sepp_training_positive.jsonl",
        source_dir / "t3sepp_training_negative.jsonl",
        source_dir / "t3sepp_independent_test.jsonl",
        proteome_path,
        target_positive_path,
        args.output_dir.resolve(),
        folds=args.folds,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
