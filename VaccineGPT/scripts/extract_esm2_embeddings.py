from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_m1.encoders.foundation import ESM3Adapter, FoundationModelConfig
from shared.lineage import lineage_tag


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def extract(
    input_path: Path,
    output_dir: Path,
    model_id: str = "esm3-sm-open-v1",
    batch_size: int = 1,
    max_length: int = 256,
    limit: int | None = None,
    device: str = "auto",
    batch_id: str = "UNSPECIFIED",
) -> dict[str, Any]:
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")
    if model_id != "esm3-sm-open-v1":
        raise ValueError("VaccineGPT M1 feature extraction requires esm3-sm-open-v1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty embedding cache: {output_dir}")
    if batch_id.casefold() in {"unspecified", "unknown", "replace_with_batch_id", "template"}:
        raise ValueError("a real, immutable batch_id is required for an embedding cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage = lineage_tag(
        "VaccineGPT-v1-dev", "FEAT-v1-ESM3-1.0", "LABEL-evidence-1.0", batch_id
    )
    expected_dimension = 1536
    adapter = ESM3Adapter(
        FoundationModelConfig(
            model_id=model_id,
            device=device,
            freeze=True,
            extra={"max_length": max_length, "embedding_dimension": expected_dimension},
        )
    ).load()
    weights_sha256 = None
    runtime_package_version = version("esm")
    checkpoint_status = (
        "ESM3 loader-managed checkpoint; package and model ID recorded, "
        "local checkpoint SHA-256 unavailable"
    )
    records = []
    total = 0
    shard = 0
    batch_manifest = []

    def flush(batch: list[dict[str, Any]]) -> None:
        nonlocal shard, total
        sequences = [str(row.get("protein_sequence") or row.get("sequence") or "") for row in batch]
        if any(not sequence for sequence in sequences):
            raise ValueError("input contains an empty protein sequence")
        embeddings, lengths = adapter.encode_tokens(sequences)
        mask = torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0) < lengths.unsqueeze(1)
        features = {
            "gene_ids": [str(row.get("gene_id") or "") for row in batch],
            "group_ids": [
                str(row.get("genome_id") or row.get("group_id") or "unknown")
                for row in batch
            ],
            "split_group_ids": [
                str(row.get("split_group_id") or "unknown") for row in batch
            ],
            "split_group_methods": [
                str(row.get("split_group_method") or "unknown") for row in batch
            ],
            "sequence_hashes": [row.get("sequence_hash") for row in batch],
            "metadata_rows": [
                {
                    "split": row.get("split"),
                    "public_features": row.get("public_features", {}),
                    "public_feature_masks": row.get("public_feature_masks", {}),
                    "private_features": row.get("private_features"),
                    "private_feature_mask": row.get("private_feature_mask"),
                    "edges": row.get("edges", []),
                }
                for row in batch
            ],
            "residue_embeddings": embeddings.detach().cpu().to(torch.float32),
            "sequence_mask": mask.detach().cpu(),
            "embedding_dimension": int(embeddings.shape[-1]),
            "backend": "esm3",
            "model_id": model_id,
            "max_length": max_length,
            "lineage": lineage,
        }
        if any(not gene_id for gene_id in features["gene_ids"]):
            raise ValueError("every input sequence must have a non-empty gene_id")
        path = output_dir / f"esm3_embeddings_{shard:06d}.pt"
        torch.save(features, path)
        batch_manifest.append(
            {
                "path": path.name,
                "records": len(batch),
                "bytes": path.stat().st_size,
            }
        )
        total += len(batch)
        shard += 1

    for row in iter_records(input_path):
        if limit is not None and total >= limit:
            break
        records.append(row)
        if len(records) >= batch_size or (
            limit is not None and total + len(records) >= limit
        ):
            remaining = max(0, limit - total) if limit is not None else len(records)
            flush(records[:remaining])
            records = []
            if limit is not None and total >= limit:
                break
    if records:
        flush(records)
    manifest = {
        "schema_version": "vaccinegpt-esm3-cache-1.0",
        "input": str(input_path),
        "lineage": lineage,
        "backend": "esm3",
        "model_id": model_id,
        "embedding_dimension": expected_dimension,
        "runtime_package": "Biohub esm",
        "runtime_package_version": runtime_package_version,
        "weights_sha256": weights_sha256,
        "checkpoint_path_or_status": checkpoint_status,
        "device": str(adapter.config.resolved_device()),
        "max_length": max_length,
        "record_count": total,
        "batches": batch_manifest,
        "embedding_role": "frozen_residue_features_not_finetuned_model_outputs",
    }
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen Biohub ESM-3 residue embeddings with input provenance."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="esm3-sm-open-v1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            extract(
                args.input_jsonl,
                args.output_dir,
                args.model_id,
                args.batch_size,
                args.max_length,
                args.limit,
                args.device,
                args.batch_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
