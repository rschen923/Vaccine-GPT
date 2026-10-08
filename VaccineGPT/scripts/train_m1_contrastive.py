from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_m1.models import M1Encoder
from src_m1.training import train_precomputed_batches
from shared.lineage import lineage_tag


def _materialize_metadata(batch: dict, path: Path) -> dict:
    metadata = batch.get("metadata_rows")
    if metadata is None:
        return batch
    if len(metadata) != len(batch["gene_ids"]):
        raise ValueError(f"{path}: metadata_rows do not align to gene_ids")
    batch_size = len(metadata)
    batch["group_ids"] = [
        str(value or "unknown") for value in batch.get("group_ids", [])
    ]
    public_names = sorted(
        {
            name
            for row in metadata
            for name in (row.get("public_features") or {})
        }
    )
    public_features = {}
    public_masks = {}
    for name in public_names:
        values = [((row.get("public_features") or {}).get(name)) for row in metadata]
        present = [value is not None for value in values]
        dimension = len(next(value for value in values if value is not None))
        if any(value is not None and len(value) != dimension for value in values):
            raise ValueError(f"{path}: inconsistent dimension for public modality {name}")
        public_features[name] = torch.tensor(
            [value if value is not None else [0.0] * dimension for value in values],
            dtype=torch.float32,
            device=batch["residue_embeddings"].device,
        )
        public_masks[name] = torch.tensor(
            [
                present[index]
                and bool((metadata[index].get("public_feature_masks") or {}).get(name, True))
                for index in range(batch_size)
            ],
            dtype=torch.bool,
            device=batch["residue_embeddings"].device,
        )
    batch["public_features"] = public_features
    batch["public_feature_masks"] = public_masks
    private_values = [row.get("private_features") for row in metadata]
    private_dim = len(next((value for value in private_values if value is not None), [0.0] * 14))
    if private_dim != 14:
        raise ValueError(f"{path}: private feature vectors must have 14 dimensions")
    batch["private_features"] = torch.tensor(
        [value if value is not None else [0.0] * private_dim for value in private_values],
        dtype=torch.float32,
        device=batch["residue_embeddings"].device,
    )
    batch["private_feature_mask"] = torch.tensor(
        [
            row.get("private_feature_mask")
            if row.get("private_feature_mask") is not None
            else ([True] * private_dim if row.get("private_features") is not None else [False] * private_dim)
            for row in metadata
        ],
        dtype=torch.bool,
        device=batch["residue_embeddings"].device,
    )
    gene_index = {str(gene_id): index for index, gene_id in enumerate(batch["gene_ids"])}
    relation_types = {
        "operon": 0,
        "coexpression": 1,
        "coadaptation": 2,
        "interaction": 3,
        "dna_regulation": 4,
        "dna_element_proximity": 5,
    }
    edge_src, edge_dst, edge_type, edge_weight = [], [], [], []
    for source_index, row in enumerate(metadata):
        for edge in row.get("edges", []):
            target = str(edge.get("target_gene") or "")
            relation = str(edge.get("relation") or "")
            if target not in gene_index:
                continue
            if relation not in relation_types:
                raise ValueError(f"{path}: unsupported GCE relation {relation}")
            edge_src.append(source_index)
            edge_dst.append(gene_index[target])
            edge_type.append(relation_types[relation])
            edge_weight.append(float(edge.get("weight", 1.0)))
    device = batch["residue_embeddings"].device
    batch["edge_index"] = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    batch["edge_type"] = torch.tensor(edge_type, dtype=torch.long, device=device)
    batch["edge_weight"] = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    return batch


def _load_batch(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    batch = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(batch, dict):
        raise ValueError(f"{path} must contain a dictionary batch")
    required = {"residue_embeddings", "sequence_mask", "group_ids"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"{path} missing batch fields: {sorted(missing)}")
    if batch["residue_embeddings"].shape[0] != len(batch["group_ids"]):
        raise ValueError(f"{path}: group_ids do not align to batch rows")
    if any(str(group) == "unknown" for group in batch["group_ids"]):
        raise ValueError(f"{path}: unknown genome/group IDs prevent leakage-safe training")
    if "split_group_ids" not in batch or "split_group_methods" not in batch:
        raise ValueError(f"{path}: provide MMseqs2 90%-identity split-group IDs and method")
    if any(str(group) == "unknown" for group in batch["split_group_ids"]):
        raise ValueError(f"{path}: missing 90%-identity split group")
    if set(batch["split_group_methods"]) != {"mmseqs2_90"}:
        raise ValueError(f"{path}: only explicit mmseqs2_90 split provenance is accepted")
    return _materialize_metadata(batch, path)


class BatchFiles:
    def __init__(self, paths: list[Path], device: torch.device):
        self.paths = paths
        self.device = device

    def __iter__(self) -> Iterator[dict]:
        for path in self.paths:
            yield _load_batch(path, self.device)


def split_groups(paths: list[Path]) -> set[str]:
    groups = set()
    for path in paths:
        batch = torch.load(path, map_location="cpu", weights_only=True)
        if "split_group_ids" not in batch or "split_group_methods" not in batch:
            raise ValueError(f"{path}: provide MMseqs2 90%-identity split-group IDs and method")
        if set(batch["split_group_methods"]) != {"mmseqs2_90"}:
            raise ValueError(f"{path}: only explicit mmseqs2_90 split provenance is accepted")
        values = set(map(str, batch["split_group_ids"]))
        if "unknown" in values:
            raise ValueError(f"{path}: missing 90%-identity split group")
        groups.update(values)
    return groups


def validate_esm3_cache_contract(paths: list[Path]) -> None:
    if not paths:
        raise ValueError("at least one ESM3 feature batch is required")
    for path in paths:
        batch = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(batch, dict):
            raise ValueError(f"{path}: embedding cache batch must be a dictionary")
        embeddings = batch.get("residue_embeddings")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            raise ValueError(f"{path}: residue_embeddings must have shape [batch, length, 1536]")
        if (
            batch.get("backend") != "esm3"
            or batch.get("model_id") != "esm3-sm-open-v1"
            or batch.get("embedding_dimension") != 1536
            or embeddings.shape[-1] != 1536
        ):
            raise ValueError(
                f"{path}: expected esm3-sm-open-v1 cache with 1536-dimensional embeddings"
            )


def train(
    train_files: list[Path],
    validation_files: list[Path],
    output_dir: Path,
    epochs: int = 20,
    device_name: str = "auto",
    batch_id: str = "UNSPECIFIED",
) -> dict:
    if batch_id.casefold() in {"unspecified", "unknown", "replace_with_batch_id", "template"}:
        raise ValueError("a real, immutable batch_id is required for M1 training")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else (
            "cpu" if device_name == "auto" else device_name
        )
    )
    validate_esm3_cache_contract(train_files + validation_files)
    train_batches = BatchFiles(train_files, device)
    validation_batches = BatchFiles(validation_files, device)
    train_groups = split_groups(train_files)
    validation_groups = split_groups(validation_files)
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(
            f"genome/group leakage between M1 train and validation batches: {sorted(overlap)[:10]}"
        )
    model = M1Encoder().to(device)
    result = train_precomputed_batches(
        model, train_batches, validation_batches, epochs=epochs
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "v1_CLEF_ESM3_GATv2",
            "protein_backbone": "esm3-sm-open-v1",
            "residue_dimension": 1536,
            "input_track": "protein",
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "lineage": lineage_tag(
                "VaccineGPT-v1-dev",
                "FEAT-v1-ESM3-1.0",
                "LABEL-evidence-1.0",
                batch_id,
            ),
        },
        output_dir / "m1_contrastive_checkpoint.pt",
    )
    report = {
        "architecture": "v1_CLEF_ESM3_GATv2",
        "protein_backbone": "esm3-sm-open-v1",
        "residue_dimension": 1536,
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "lineage": lineage_tag(
            "VaccineGPT-v1-dev",
            "FEAT-v1-ESM3-1.0",
            "LABEL-evidence-1.0",
            batch_id,
        ),
        "test_data_used": False,
        "training": {key: value for key, value in result.items() if key != "model_state_dict"},
        "biological_validation": "not_claimed_by_this_training_run",
    }
    (output_dir / "m1_contrastive_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train only trainable M1 contrastive/GCE layers on precomputed paired features."
    )
    parser.add_argument("--train-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.train_batches,
                args.validation_batches,
                args.output_dir,
                args.epochs,
                args.device,
                args.batch_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
