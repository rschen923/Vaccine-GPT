from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_and_clean_data import read_jsonl
from shared.lineage import lineage_tag
from src_m1.models import M1Encoder, info_nce_loss


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TOKEN_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}


def split_by_group(
    records: list[dict[str, Any]], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    parent: dict[str, str] = {}
    sequence_owner: dict[str, str] = {}

    def find(group_id: str) -> str:
        parent.setdefault(group_id, group_id)
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in records:
        sequence = str(row.get("sequence") or row.get("protein_sequence") or "")
        if not sequence:
            raise ValueError(f"{row.get('gene_id')}: missing normalized sequence")
        digest = str(row.get("sequence_sha256") or hashlib.sha256(sequence.encode("ascii")).hexdigest())
        group_id = str(
            row.get("split_group_id")
            or row.get("genome_id")
            or row.get("group_id")
            or digest
        )
        find(group_id)
        if digest in sequence_owner:
            union(group_id, sequence_owner[digest])
        else:
            sequence_owner[digest] = group_id
        groups.setdefault(group_id, []).append(
            {**row, "sequence": sequence, "sequence_sha256": digest}
        )
    merged_groups: dict[str, list[dict[str, Any]]] = {}
    for group_id, members in groups.items():
        merged_groups.setdefault(find(group_id), []).extend(members)
    ordered = sorted(
        merged_groups.values(),
        key=lambda row: hashlib.sha256(
            f"{seed}:{row[0]['sequence_sha256']}".encode("ascii")
        ).hexdigest(),
    )
    n = len(ordered)
    if n < 3:
        raise ValueError("at least 3 independent groups are required for train/validation/test")
    train_end = max(1, min(int(n * 0.70), n - 2))
    valid_end = max(train_end + 1, min(int(n * 0.85), n - 1))
    method = (
        "provided split_group_id"
        if all(row.get("split_group_id") for row in records)
        else "genome_id/group_id fallback; MMseqs2 homology leakage not controlled"
    )
    return (
        {
            "train": [row for group in ordered[:train_end] for row in group],
            "validation": [row for group in ordered[train_end:valid_end] for row in group],
            "test": [row for group in ordered[valid_end:] for row in group],
        },
        method,
    )


def crop_pair(
    sequence: str,
    rng: random.Random,
    max_length: int = 256,
) -> tuple[list[int | None], list[int | None]]:
    sequence = "".join(sequence.upper().split())
    known_count = sum(residue in TOKEN_INDEX for residue in sequence)
    if len(sequence) < 2 or known_count < 2:
        raise ValueError("sequence must have at least two canonical amino acids")
    upper = min(len(sequence), max_length)
    lower = min(upper, max(2, int(upper * 0.8)))
    shared_length = rng.randint(lower, upper)
    max_start = max(0, len(sequence) - shared_length)
    anchor = rng.randint(0, max_start) if max_start else 0
    jitter = max(1, int(shared_length * 0.05))
    starts = [
        max(0, min(max_start, anchor + rng.randint(-jitter, jitter))),
        max(0, min(max_start, anchor + rng.randint(-jitter, jitter))),
    ]
    views = []
    for start in starts:
        crop = sequence[start : start + shared_length]
        views.append(
            [TOKEN_INDEX.get(residue) for residue in crop[:max_length]]
        )
    return views[0], views[1]


def collate(
    batch: list[dict[str, Any]],
    rng: random.Random,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [crop_pair(row["sequence"], rng, max_length) for row in batch]
    width = max(max(len(left), len(right)) for left, right in pairs)
    first = torch.zeros((len(batch), width, len(AMINO_ACIDS)), dtype=torch.float32)
    second = torch.zeros_like(first)
    for row_index, (left, right) in enumerate(pairs):
        left_positions = [index for index, token in enumerate(left) if token is not None]
        right_positions = [index for index, token in enumerate(right) if token is not None]
        if len(left_positions) < 2 or len(right_positions) < 2:
            raise ValueError("cropped views must contain at least two canonical amino acids")
        first[row_index, left_positions, [left[index] for index in left_positions]] = 1.0
        second[row_index, right_positions, [right[index] for index in right_positions]] = 1.0
    return first, second


class SelfSupervisedProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = M1Encoder(
            residue_dim=len(AMINO_ACIDS),
            latent_dim=128,
            private_dim=14,
            max_length=256,
            gce_relations=6,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = features.sum(dim=-1) > 0
        return self.encoder(features, mask)


def paired_retrieval_accuracy(
    model: SelfSupervisedProteinEncoder,
    records: list[dict[str, Any]],
    seed: int,
    batch_size: int,
) -> float:
    rng = random.Random(seed)
    first_embeddings, second_embeddings = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            left, right = collate(records[start : start + batch_size], rng)
            first_embeddings.append(model(left)["z_gctx"])
            second_embeddings.append(model(right)["z_gctx"])
    left_all = torch.cat(first_embeddings)
    right_all = torch.cat(second_embeddings)
    similarity = left_all @ right_all.T
    return float((similarity.argmax(dim=1) == torch.arange(len(records))).float().mean())


def _train_epoch(
    model: SelfSupervisedProteinEncoder,
    records: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    batch_size: int,
) -> float:
    model.train()
    order = list(range(len(records)))
    rng.shuffle(order)
    losses = []
    for start in range(0, len(order), batch_size):
        ids = order[start : start + batch_size]
        if len(ids) < 2:
            continue
        left, right = collate([records[index] for index in ids], rng)
        left_embedding, right_embedding = model(left), model(right)
        public_loss = info_nce_loss(
            left_embedding["z_pub"], right_embedding["z_pub"], temperature=0.1
        )
        context_loss = info_nce_loss(
            left_embedding["z_gctx"], right_embedding["z_gctx"], temperature=0.1
        )
        loss = public_loss + context_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("training split yielded no batches with at least two records")
    return float(np.mean(losses))


def train(
    sample_path: Path,
    output_dir: Path,
    seed: int = 20261007,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 2 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty training output: {output_dir}")
    records = list(read_jsonl(sample_path))
    splits, split_method = split_by_group(records, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = SelfSupervisedProteinEncoder()
    baseline_model = copy.deepcopy(model)
    baseline_accuracy = paired_retrieval_accuracy(
        baseline_model,
        splits["test"],
        seed + 20000,
        batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history = []
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    patience = 2
    start_time = time.perf_counter()
    for epoch in range(epochs):
        train_loss = _train_epoch(
            model,
            splits["train"],
            optimizer,
            random.Random(seed + epoch),
            batch_size,
        )
        model.eval()
        validation_losses = []
        validation_rng = random.Random(seed + 10000 + epoch)
        with torch.no_grad():
            for start in range(0, len(splits["validation"]), batch_size):
                batch = splits["validation"][start : start + batch_size]
                if len(batch) < 2:
                    continue
                left, right = collate(batch, validation_rng)
                left_output, right_output = model(left), model(right)
                validation_losses.append(
                    float(
                        info_nce_loss(
                            left_output["z_pub"], right_output["z_pub"], temperature=0.1
                        )
                        + info_nce_loss(
                            left_output["z_gctx"],
                            right_output["z_gctx"],
                            temperature=0.1,
                        )
                    )
                )
        if not validation_losses:
            raise ValueError("validation split has no valid contrastive batch")
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_infonce": train_loss,
                "validation_infonce": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_key = lambda row: str(
        row.get("split_group_id")
        or row.get("genome_id")
        or row.get("group_id")
        or row["sequence_sha256"]
    )
    train_ids = {group_key(row) for row in splits["train"]}
    valid_ids = {group_key(row) for row in splits["validation"]}
    test_ids = {group_key(row) for row in splits["test"]}
    if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
        raise RuntimeError("exact sequence leakage detected across splits")
    checkpoint_path = output_dir / "m1_real_sample_self_supervised.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "full-M1-CLEF-style-encoder-plus-GCE-over-real-protein-onehot",
            "training_objective": "two-stochastic-crops-of-same-sequence-InfoNCE",
            "pretrained_weights_used": False,
            "split_method": split_method,
            "train_group_ids": sorted(train_ids),
            "validation_group_ids": sorted(valid_ids),
            "test_group_ids": sorted(test_ids),
        },
        checkpoint_path,
    )
    model_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    final_accuracy = paired_retrieval_accuracy(
        model,
        splits["test"],
        seed + 20000,
        batch_size,
    )
    lineage = lineage_tag(
        "VaccineGPT-v1-selfsup-smoke",
        "FEAT-real-protein-crop-onehot-1.0",
        "LABEL-none-selfsup-1.0",
        f"REAL-{seed}-{len(records)}",
    )
    report = {
        "schema_version": "vaccinegpt-real-self-supervised-training-1.0",
        "lineage": lineage,
        "sample_input": str(sample_path),
        "sample_input_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "sample_count": len(records),
        "unique_exact_sequences": len(train_ids | valid_ids | test_ids),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "split_method": split_method,
        "training_target": "self-supervised sequence-view agreement only",
        "not_used_as_targets": ["VFDB membership", "IEDB peptide assay labels", "EIB202 H1-H4 labels"],
        "feature_backend": "amino-acid one-hot; no pretrained ESM-3/CLEF weights",
        "trained_components": [
            "CLEF-style Encoder A",
            "public projection",
            "private adapter parameters remained without paired private observations",
            "GCE self-loop path; no genome graph edges available",
        ],
        "pretrained_weights_used": False,
        "loss_history": history,
        "best_validation_infonce": best_validation_loss,
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "test_paired_retrieval_top1_before_training": baseline_accuracy,
        "test_paired_retrieval_top1_after_training": final_accuracy,
        "test_metric_scope": "self-supervised crop retrieval, not biological classification",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": model_hash,
        "training_seconds": time.perf_counter() - start_time,
        "supervised_m2_training": "not run; this sequence-only training run has no E1/E2 gold labels",
        "biological_prediction_status": "not_evaluated",
        "model_adjustment_assessment": (
            "No biological architecture change is justified by this self-supervised smoke. "
            "Validation loss and paired-view retrieval test the same artificial crop task; "
            "test retrieval at ceiling is non-discriminative. A homolog-cluster split and "
            "target-specific E1/E2 validation set are required before adjusting H1-H4."
        ),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "vaccinegpt-self-supervised-split-1.0",
                "seed": seed,
                "method": split_method,
                "groups": {
                    "train": sorted(train_ids),
                    "validation": sorted(valid_ids),
                    "test": sorted(test_ids),
                },
                "record_counts": {key: len(value) for key, value in splits.items()},
                "leakage_check": "passed",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-controlled self-supervised training on real protein sequences."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sample-jsonl", type=Path)
    input_group.add_argument(
        "--dataset-root",
        type=Path,
        help="uploaded dataset root; discovers processed/refseq_training/sequences.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    sample_jsonl = args.sample_jsonl
    if args.dataset_root is not None:
        sample_jsonl = (
            args.dataset_root / "processed" / "refseq_training" / "sequences.jsonl"
        )
    if sample_jsonl is None or not sample_jsonl.is_file():
        raise FileNotFoundError(f"protein sequence JSONL not found: {sample_jsonl}")
    print(
        json.dumps(
            train(
                sample_jsonl,
                args.output_dir,
                args.seed,
                args.epochs,
                args.batch_size,
                args.learning_rate,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
