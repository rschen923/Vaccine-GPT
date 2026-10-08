from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_and_clean_data import audit_and_clean_public_final, read_jsonl
from src_m2.aep import external_parameter_declaration, unavailable_aep_row
from src_m1.encoders.foundation import ESM3Adapter, FoundationModelConfig
from src_m1.models import M1Encoder
from src_m2.models import M2Predictor
from shared.lineage import lineage_tag


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}


def one_hot_batch(records: list[dict[str, Any]], max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.zeros((len(records), max_length, len(AMINO_ACIDS)), dtype=torch.float32)
    mask = torch.zeros((len(records), max_length), dtype=torch.bool)
    for row_index, record in enumerate(records):
        sequence = str(record.get("sequence") or "")[:max_length]
        for position, residue in enumerate(sequence):
            amino_acid_index = AMINO_ACID_INDEX.get(residue)
            if amino_acid_index is not None:
                embeddings[row_index, position, amino_acid_index] = 1.0
            mask[row_index, position] = True
    return embeddings, mask


def run_sample(
    data_root: Path,
    output_dir: Path,
    sample_size: int = 1000,
    seed: int = 20261007,
    batch_size: int = 8,
    backend: str = "onehot-smoke",
    sample_file: Path | None = None,
) -> dict[str, Any]:
    if sample_size < 1 or batch_size < 1:
        raise ValueError("sample_size and batch_size must be positive")
    if backend not in {"onehot-smoke", "esm3"}:
        raise ValueError("backend must be onehot-smoke or esm3")
    torch.manual_seed(seed)
    random.seed(seed)
    if sample_file is None:
        audit = audit_and_clean_public_final(data_root, output_dir, seed, sample_size)
        sample_path = output_dir / audit["outputs"]["sample_sequences"]
    else:
        sample_path = sample_file
        audit_path = sample_path.parent / "audit_summary.json"
        if not audit_path.is_file():
            raise FileNotFoundError(f"sample audit summary not found: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sample = list(read_jsonl(sample_path))
    if len(sample) != min(sample_size, audit["sample_size"]):
        raise ValueError("sample file size does not match the requested audited sample size")
    residue_dim = len(AMINO_ACIDS) if backend == "onehot-smoke" else 1536
    model = M1Encoder(residue_dim=residue_dim, max_length=256).eval()
    predictor = M2Predictor().eval()
    esm_adapter = None
    if backend == "esm3":
        esm_adapter = ESM3Adapter(
            FoundationModelConfig(
                model_id="esm3-sm-open-v1",
                device="auto",
                freeze=True,
                extra={"max_length": 256, "embedding_dimension": 1536},
            )
        ).load()
    m1_shapes: dict[str, list[int]] = {}
    m2_shapes: dict[str, list[int]] = {}
    processed = 0
    with torch.no_grad():
        for start in range(0, len(sample), batch_size):
            batch = sample[start : start + batch_size]
            if esm_adapter is None:
                residues, mask = one_hot_batch(batch)
            else:
                sequences = [str(row["sequence"])[:256] for row in batch]
                residues, lengths = esm_adapter.encode_tokens(sequences)
                residues = residues.cpu().float()
                positions = torch.arange(residues.shape[1]).unsqueeze(0)
                mask = positions < lengths.cpu().unsqueeze(1)
            representation = model(residues, mask)
            predictions = predictor(representation["z_gctx"])
            processed += len(batch)
            m1_shapes = {
                name: list(representation[name].shape)
                for name in ("z_pub", "z_exp", "z_gctx", "position_scores")
            }
            m2_shapes = {name: list(value.shape) for name, value in predictions.items()}

    sample_hash = hashlib.sha256(
        "\n".join(sorted(str(row["sequence_sha256"]) for row in sample)).encode("utf-8")
    ).hexdigest()
    feature_version = (
        "FEAT-v1-ESM3-1.0" if backend == "esm3" else "FEAT-onehot-smoke-1.0"
    )
    aep_rows = []
    unresolved = []
    for row in sample:
        aep_rows.append(
            {
                **unavailable_aep_row(
                    row["gene_id"],
                    "No trained checkpoint, EIB202-specific experimental anchor, "
                    "or target-specific evidence was used; this row is not a vaccine recommendation.",
                ),
                "three_reader_consistency": {"status": "not_assessed_untrained_heads"},
                "decision_status": "unranked_no_validated_biological_model",
            }
        )
        unresolved.append(
            {
                "gene_id": row["gene_id"],
                "source": row.get("source"),
                "sequence_quality_status": row["sequence_quality_status"],
                "identifier_quality_flags": row["identifier_quality_flags"],
                "status": "requires_target_specific_annotation_and_evidence_review",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "aep": output_dir / "sample_aep_unranked.jsonl",
        "unresolved": output_dir / "sample_unresolved_appendix.jsonl",
    }
    for key, rows in (("aep", aep_rows), ("unresolved", unresolved)):
        with paths[key].open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "vaccinegpt-real-data-smoke-1.0",
        "lineage": lineage_tag(
            "VaccineGPT-v1-integration-smoke",
            feature_version,
            "LABEL-E4-audit-1.0",
            f"REAL-{seed}-{len(sample)}",
        ),
        "sample_seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_processed": processed,
        "sample_sequence_sha256": sample_hash,
        "input_dataset": str(data_root / "processed" / "public_final"),
        "sample_is_real_public_sequence_data": True,
        "feature_backend": (
            "amino_acid_one_hot_smoke_only_not_ESM3"
            if backend == "onehot-smoke"
            else "frozen_ESM3_open_small_sequence_embeddings"
        ),
        "protein_backbone": (
            "none_onehot_smoke" if backend == "onehot-smoke" else "esm3-sm-open-v1"
        ),
        "residue_dimension": residue_dim,
        "m1_checkpoint_loaded": False,
        "esm3_model_loaded": esm_adapter is not None,
        "m2_checkpoint_loaded": False,
        "private_evidence_loaded": False,
        "edges_loaded": False,
        "m1_output_shapes_last_batch": m1_shapes,
        "m2_output_shapes_last_batch": m2_shapes,
        "rows_through_m1_and_m2_forward": processed,
        "biological_prediction_status": (
            "not_performed_random_initialization_and_no_target_specific_gold_labels"
        ),
        "aep_contract_rows": processed,
        "virulence_gene_list": [],
        "retained_antigen_list": [],
        "external_parameters_not_computed": [
            "dose_and_immunization_schedule",
            "delivery_platform_and_route",
            "target_population_and_equity",
            "clinical_trial_design",
        ],
        "external_parameter_declaration": external_parameter_declaration(),
        "audit_summary": audit,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    (output_dir / "real_sample_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an integration smoke on real public protein records without claiming model predictions."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backend", choices=("onehot-smoke", "esm3"), default="onehot-smoke")
    parser.add_argument("--sample-file", type=Path)
    args = parser.parse_args()
    result = run_sample(
        args.data_root,
        args.output_dir,
        args.sample_size,
        args.seed,
        args.batch_size,
        args.backend,
        args.sample_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
