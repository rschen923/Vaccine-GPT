from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_m1.clef_compat import (  # noqa: E402
    CLEF_COMMIT,
    CLEF_REPOSITORY,
    CLEFProteinClassifier,
    CLEFSequenceEncoder,
    load_clef_weights,
)
from src_m1.encoders.foundation import ESM2Adapter, FoundationModelConfig  # noqa: E402


CLEF_TASKS = {
    "T3SS": {
        "encoder": "CLEF-DP+AT.pt",
        "classifier": "T3classifier-CLEF-DP+AT-0.5cutoff.pt",
        "cutoff": 0.5,
        "pathway": "T3SS",
    },
    "T4SS": {
        "encoder": "CLEF-DP+MSA+3Di+AT.pt",
        "classifier": "T4classifier-CLEF-DP+MSA+3Di+AT-0.83cutoff.pt",
        "cutoff": 0.83,
        "pathway": "T4SS",
    },
    "T6SS": {
        "encoder": "CLEF-DP+MSA+3Di+AT.pt",
        "classifier": "T6classifier-CLEF-DP+MSA+3Di+AT-0.7cutoff.pt",
        "cutoff": 0.7,
        "pathway": "T6SS",
    },
}
ESM2_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
ESM2_RESIDUE_LIMIT = 254
CLEF_TOKEN_LIMIT = 256


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
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_sequence(sequence: Any, gene_id: str) -> tuple[str, int]:
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise ValueError(f"{gene_id}: empty protein sequence")
    replaced = sum(residue not in ESM2_ALPHABET for residue in normalized)
    cleaned = "".join(residue if residue in ESM2_ALPHABET else "X" for residue in normalized)
    return cleaned[:ESM2_RESIDUE_LIMIT], replaced


def _load_positive_evidence(
    path: Path, proteome_by_gene: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    evidence_by_gene: dict[str, list[dict[str, Any]]] = {}
    out_of_scope = []
    for row in read_jsonl(path):
        if str(row.get("task") or "") != "H3" or int(row.get("label", -1)) != 1:
            raise ValueError(f"{path}: expected curated positive H3 evidence rows only")
        gene_id = str(row.get("gene_id") or "")
        if gene_id not in proteome_by_gene:
            raise ValueError(f"{path}: positive gene {gene_id!r} is absent from the proteome")
        sequence = str(
            proteome_by_gene[gene_id].get("protein_sequence")
            or proteome_by_gene[gene_id].get("sequence")
            or ""
        )
        actual_hash = hashlib.sha256("".join(sequence.split()).upper().encode("ascii")).hexdigest()
        expected_hash = str(row.get("sequence_sha256") or "")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"{path}: sequence hash mismatch for positive gene {gene_id}")
        pathway = str(row.get("secretion_or_translocation_pathway") or "").upper()
        if pathway not in {"T3SS", "T4SS", "T6SS"}:
            row = dict(row)
            row["model_status"] = "retained_as_experimental_evidence_not_scored_by_CLEF_task"
            out_of_scope.append(row)
            continue
        evidence_by_gene.setdefault(gene_id, []).append(row)
    return evidence_by_gene, out_of_scope


def predict(
    proteome_path: Path,
    positive_path: Path,
    esm2_weights: Path,
    clef_weights_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = 2,
    top_k: int = 50,
    device_name: str = "auto",
) -> dict[str, Any]:
    if batch_size < 1 or top_k < 1:
        raise ValueError("batch_size and top_k must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    if not esm2_weights.is_file():
        raise FileNotFoundError(f"ESM2 weights not found: {esm2_weights}")
    if not clef_weights_dir.is_dir():
        raise FileNotFoundError(f"CLEF weights directory not found: {clef_weights_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    proteome = read_jsonl(proteome_path)
    proteome_by_gene: dict[str, dict[str, Any]] = {}
    sequences: list[str] = []
    replaced_counts: list[int] = []
    for row in proteome:
        gene_id = str(row.get("gene_id") or "")
        if not gene_id or gene_id in proteome_by_gene:
            raise ValueError(f"{proteome_path}: missing or duplicate gene_id {gene_id!r}")
        sequence = row.get("protein_sequence") or row.get("sequence")
        normalized, replaced = _clean_sequence(sequence, gene_id)
        proteome_by_gene[gene_id] = row
        sequences.append(normalized)
        replaced_counts.append(replaced)

    evidence_by_gene, out_of_scope_positives = _load_positive_evidence(
        positive_path, proteome_by_gene
    )
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )

    esm = ESM2Adapter(
        FoundationModelConfig(
            model_id="esm2_t33_650M_UR50D",
            local_path=str(esm2_weights.resolve()),
            device=str(device),
            freeze=True,
            extra={"max_length": ESM2_RESIDUE_LIMIT},
        )
    )
    esm.load()
    encoders: dict[str, CLEFSequenceEncoder] = {}
    encoder_hashes: dict[str, str] = {}
    classifiers: dict[str, CLEFProteinClassifier] = {}
    classifier_hashes: dict[str, str] = {}
    for task, spec in CLEF_TASKS.items():
        encoder_name = str(spec["encoder"])
        if encoder_name not in encoders:
            encoder = CLEFSequenceEncoder(max_length=CLEF_TOKEN_LIMIT).to(device).eval()
            encoder_hashes[encoder_name] = load_clef_weights(
                encoder, clef_weights_dir / encoder_name
            )
            encoders[encoder_name] = encoder
        classifier = CLEFProteinClassifier().to(device).eval()
        classifier_hashes[task] = load_clef_weights(
            classifier, clef_weights_dir / str(spec["classifier"])
        )
        classifiers[task] = classifier

    task_scores: dict[str, list[float]] = {task: [] for task in CLEF_TASKS}
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(sequences), batch_size), start=1):
        batch_sequences = sequences[start : start + batch_size]
        esm_representations, valid_lens = esm.encode_tokens_with_special_tokens(
            batch_sequences
        )
        if esm_representations.shape[-1] != 1280:
            raise ValueError(
                f"expected ESM2 1280-dimensional embeddings, got "
                f"{esm_representations.shape[-1]}"
            )
        padded = esm_representations.new_zeros(
            (len(batch_sequences), CLEF_TOKEN_LIMIT, 1280)
        )
        copied_length = min(esm_representations.shape[1], CLEF_TOKEN_LIMIT)
        padded[:, :copied_length] = esm_representations[:, :copied_length]
        valid_lens = valid_lens.clamp(max=CLEF_TOKEN_LIMIT)
        for task, spec in CLEF_TASKS.items():
            pooled_features = encoders[str(spec["encoder"])](
                {"esm_feature": padded, "valid_lens": valid_lens}
            )[0]
            scores = classifiers[task](pooled_features)
            task_scores[task].extend(float(score) for score in scores.cpu().tolist())
        if batch_number == total_batches or batch_number % 25 == 0:
            print(
                f"CLEF inference: {min(start + batch_size, len(sequences))}/"
                f"{len(sequences)} proteins",
                file=sys.stderr,
                flush=True,
            )

    if any(len(scores) != len(proteome) for scores in task_scores.values()):
        raise RuntimeError("CLEF inference did not produce one score per protein and task")

    rank_by_task: dict[str, dict[str, int]] = {}
    for task, scores in task_scores.items():
        order = sorted(
            range(len(proteome)),
            key=lambda index: (
                -scores[index],
                str(proteome[index].get("gene_id") or ""),
            ),
        )
        rank_by_task[task] = {
            str(proteome[index]["gene_id"]): rank
            for rank, index in enumerate(order, start=1)
        }

    output_rows = []
    for index, row in enumerate(proteome):
        gene_id = str(row["gene_id"])
        predictions = {}
        for task, spec in CLEF_TASKS.items():
            score = task_scores[task][index]
            cutoff = float(spec["cutoff"])
            above_cutoff = score >= cutoff
            has_matching_positive = any(
                str(item.get("secretion_or_translocation_pathway") or "").upper()
                == str(spec["pathway"])
                for item in evidence_by_gene.get(gene_id, [])
            )
            if has_matching_positive:
                candidate_status = (
                    "source_reported_experimental_positive_above_published_cutoff"
                    if above_cutoff
                    else "source_reported_experimental_positive_below_published_cutoff"
                )
            else:
                candidate_status = (
                    "CLEF_cutoff_candidate_without_curated_positive_evidence"
                    if above_cutoff
                    else "CLEF_ranked_below_cutoff_without_curated_positive_evidence"
                )
            predictions[task] = {
                "clef_classifier_score": score,
                "score_type": "uncalibrated_sigmoid_score_not_probability",
                "published_cutoff": cutoff,
                "cutoff_call": above_cutoff,
                "candidate_rank": rank_by_task[task][gene_id],
                "candidate_status": candidate_status,
                "encoder_checkpoint": spec["encoder"],
                "classifier_checkpoint": spec["classifier"],
            }
        output_rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": row.get("gene_symbol"),
                "product": row.get("product"),
                "protein_id": row.get("protein_id"),
                "sequence_sha256": hashlib.sha256(
                    sequences[index].encode("ascii")
                ).hexdigest(),
                "sequence_length_used": len(sequences[index]),
                "noncanonical_residues_replaced_with_X": replaced_counts[index],
                "task": "H3_CLEF_secretion_system_effector_prediction",
                "model_version": f"CLEF@{CLEF_COMMIT}",
                "task_predictions": predictions,
                "experimental_positive_evidence": evidence_by_gene.get(gene_id, []),
                "label_scope": (
                    "T3SS/T4SS/T6SS effector prediction only; not OMV, vaccine "
                    "protection, immunogenicity, or EIB202 phenotype"
                ),
            }
        )

    output_by_gene = {str(row["gene_id"]): row for row in output_rows}
    full_path = output_dir / "eib202_h3_clef_effector_predictions.jsonl"
    with full_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    task_reports = {}
    for task, spec in CLEF_TASKS.items():
        ranked = sorted(
            (
                {
                    **row,
                    "task_prediction": row["task_predictions"][task],
                }
                for row in output_rows
            ),
            key=lambda item: (
                item["task_prediction"]["candidate_rank"],
                item["gene_id"],
            ),
        )
        task_shortlist = ranked[: min(top_k, len(ranked))]
        shortlist_path = output_dir / f"eib202_h3_clef_{task.lower()}_top_{len(task_shortlist)}.jsonl"
        with shortlist_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in task_shortlist:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        pathway_positives = [
            item
            for rows in evidence_by_gene.values()
            for item in rows
            if str(item.get("secretion_or_translocation_pathway") or "").upper()
            == str(spec["pathway"])
        ]
        positive_gene_ids = sorted(
            {str(item["gene_id"]) for item in pathway_positives}
        )
        matched_calls = [
            output_by_gene[gene_id]["task_predictions"][task]["cutoff_call"]
            for gene_id in positive_gene_ids
        ]
        task_reports[task] = {
            "encoder_checkpoint": str((clef_weights_dir / str(spec["encoder"])).resolve()),
            "encoder_checkpoint_sha256": encoder_hashes[str(spec["encoder"])],
            "classifier_checkpoint": str(
                (clef_weights_dir / str(spec["classifier"])).resolve()
            ),
            "classifier_checkpoint_sha256": classifier_hashes[task],
            "published_cutoff": spec["cutoff"],
            "proteins_ranked": len(proteome),
            "candidate_shortlist_file": str(shortlist_path),
            "matching_curated_positive_count": len(positive_gene_ids),
            "matching_curated_positive_cutoff_recall": (
                sum(matched_calls) / len(matched_calls) if matched_calls else None
            ),
            "positive_validation_scope": (
                "positive-only recall/rank check; no specificity, calibration, or "
                "independent-validation claim"
                if positive_gene_ids
                else "no matching curated positives; no target validation metric"
            ),
            "matching_positive_gene_ids": positive_gene_ids,
        }

    out_of_scope_path = output_dir / "eib202_h3_clef_out_of_scope_positives.jsonl"
    with out_of_scope_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in out_of_scope_positives:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": "vaccinegpt-clef-effector-inference-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clef_repository": CLEF_REPOSITORY,
        "clef_commit": CLEF_COMMIT,
        "clef_code_license": "MIT",
        "clef_weight_license_status": (
            "checkpoint-specific reuse terms not independently verified; "
            "weights are not redistributed"
        ),
        "input_proteome": str(proteome_path.resolve()),
        "input_proteome_sha256": sha256_file(proteome_path),
        "curated_positive_evidence": str(positive_path.resolve()),
        "curated_positive_evidence_sha256": sha256_file(positive_path),
        "esm2_checkpoint": str(esm2_weights.resolve()),
        "esm2_checkpoint_sha256": sha256_file(esm2_weights),
        "esm2_model": "esm2_t33_650M_UR50D",
        "esm2_residue_limit": ESM2_RESIDUE_LIMIT,
        "clef_fixed_token_length": CLEF_TOKEN_LIMIT,
        "proteins_scored": len(proteome),
        "task_reports": task_reports,
        "omv_positives_retained_unscored": len(out_of_scope_positives),
        "validation_caveat": (
            "Curated EIB202 positives were not used for fitting in this script. "
            "Their overlap with the upstream CLEF training/benchmark corpora has "
            "not been fully audited; reported recall is not an independent test."
        ),
        "model_scope": (
            "Sequence-only inference from official CLEF encoders and task-specific "
            "classifiers. CLEF contrastive pretraining used complementary biological "
            "modalities; no modality feature is re-created or imputed at inference."
        ),
        "outputs": {
            "full_predictions": str(full_path),
            "out_of_scope_positive_evidence": str(out_of_scope_path),
        },
        "limitations": [
            "Classifier sigmoid values are uncalibrated scores, not probabilities.",
            "CLEF predicts T3SS/T4SS/T6SS effectors, not OMV-mediated translocation.",
            "An effector prediction is not evidence of vaccine protection or immunogenicity.",
            "No target-specific adapter or fine-tuning was performed.",
        ],
    }
    (output_dir / "clef_effector_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    default_data = Path(__file__).resolve().parents[2] / ".." / "data"
    parser = argparse.ArgumentParser(
        description="Run official task-specific CLEF sequence-only effector inference."
    )
    parser.add_argument(
        "--proteome",
        type=Path,
        default=default_data
        / "processed"
        / "vaccinegpt_eib202_proteome_training_261007"
        / "eib202_proteome.jsonl",
    )
    parser.add_argument(
        "--target-positives",
        type=Path,
        default=default_data
        / "raw"
        / "public_training_sources"
        / "m2_public_candidates"
        / "PMC5820615"
        / "curated_h3_eib202_positive_candidates.jsonl",
    )
    parser.add_argument("--esm2-weights", type=Path, required=True)
    parser.add_argument("--clef-weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = predict(
        args.proteome,
        args.target_positives,
        args.esm2_weights,
        args.clef_weights_dir,
        args.output_dir.resolve(),
        batch_size=args.batch_size,
        top_k=args.top_k,
        device_name=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
