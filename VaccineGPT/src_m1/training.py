from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import optim

from shared.lineage import lineage_tag
from src_m1.models import M1Encoder, info_nce_loss


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_synthetic_batch(batch_size: int = 8, length: int = 48) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    residue_ids = torch.randint(len(AMINO_ACIDS), (batch_size, length), generator=generator)
    embeddings = torch.nn.functional.one_hot(residue_ids, num_classes=len(AMINO_ACIDS)).float()
    mask = torch.ones((batch_size, length), dtype=torch.bool)
    return embeddings, mask


def m1_contrastive_objective(model: M1Encoder, batch: Dict[str, object]) -> torch.Tensor:
    residues = batch["residue_embeddings"]
    sequence_mask = batch["sequence_mask"]
    if not isinstance(residues, torch.Tensor) or not isinstance(sequence_mask, torch.Tensor):
        raise TypeError("residue_embeddings and sequence_mask must be tensors")
    private_features = batch.get("private_features")
    private_mask = batch.get("private_feature_mask")
    if private_features is not None and not isinstance(private_features, torch.Tensor):
        raise TypeError("private_features must be a tensor")
    if private_mask is not None and not isinstance(private_mask, torch.Tensor):
        raise TypeError("private_feature_mask must be a tensor")
    public_features = batch.get("public_features", {})
    public_masks = batch.get("public_feature_masks", {})
    if not isinstance(public_features, dict) or not isinstance(public_masks, dict):
        raise TypeError("public_features and public_feature_masks must be mappings")
    group_ids = batch.get("group_ids")
    if not isinstance(group_ids, list) or len(group_ids) != residues.shape[0]:
        raise ValueError("group_ids must contain one genome ID per protein")
    if iter(group_ids) is group_ids:
        raise ValueError("group_ids must be a reusable sequence, not a one-shot iterator")
    output = model(
        residues,
        sequence_mask,
        private_features=private_features,
        private_feature_mask=private_mask,
        public_features=public_features,
        edge_index=batch.get("edge_index"),
        edge_type=batch.get("edge_type"),
        edge_weight=batch.get("edge_weight"),
    )
    losses = []
    for name, modality_embedding in output["z_public_modalities"].items():
        valid = public_masks.get(name)
        if not isinstance(valid, torch.Tensor) or valid.shape != (residues.shape[0],):
            raise ValueError(f"public_feature_masks[{name}] must be a batch-length boolean tensor")
        valid = valid.to(device=residues.device, dtype=torch.bool)
        if int(valid.sum()) >= 2:
            losses.append(
                info_nce_loss(
                    output["z_pub"][valid],
                    modality_embedding[valid],
                    temperature=0.1,
                )
            )
    private_valid = output["private_available"]
    private_mask_tensor = private_valid.to(torch.bool)
    private_counts = Counter(
        group for group, keep in zip(group_ids, private_mask_tensor.tolist()) if keep
    )
    private_mask_tensor = torch.tensor(
        [
            bool(keep) and private_counts[group] >= 2
            for group, keep in zip(group_ids, private_mask_tensor.tolist())
        ],
        dtype=torch.bool,
        device=residues.device,
    )
    private_groups = [group for group, keep in zip(group_ids, private_mask_tensor.tolist()) if keep]
    if len(private_groups) >= 2:
        losses.append(
            1.0
            * info_nce_loss(
                output["z_priv_sequence"][private_mask_tensor],
                output["z_priv_features"][private_mask_tensor],
                temperature=0.1,
                group_ids=private_groups,
            )
        )
    edge_index = batch.get("edge_index")
    edge_type = batch.get("edge_type")
    if (
        isinstance(edge_index, torch.Tensor)
        and edge_index.numel()
        and isinstance(edge_type, torch.Tensor)
    ):
        edge_index = edge_index.to(device=residues.device, dtype=torch.long)
        edge_type = edge_type.to(device=residues.device, dtype=torch.long)
        context_edges = (edge_type == 0) | (edge_type == 2)
        src, dst = edge_index[:, context_edges]
        if src.numel() >= 2:
            edge_groups = [group_ids[int(index)] for index in src]
            if max(Counter(edge_groups).values()) >= 2:
                losses.append(
                    info_nce_loss(
                        output["z_gctx"][dst],
                        output["z_pub"][src],
                        temperature=0.1,
                        group_ids=edge_groups,
                    )
                )
    if not losses:
        raise ValueError(
            "batch has no valid paired public modality, private omics, or graph-context positives"
        )
    return torch.stack(losses).sum()


def train_precomputed_batches(
    model: M1Encoder,
    training_batches: Iterable[Dict[str, object]],
    validation_batches: Iterable[Dict[str, object]],
    epochs: int = 20,
    learning_rate: float = 2e-5,
    patience: int = 3,
) -> dict[str, object]:
    """Train trainable CLEF/GCE layers from cached residue and paired-modality batches."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    if iter(training_batches) is training_batches or iter(validation_batches) is validation_batches:
        raise ValueError("training_batches and validation_batches must be re-iterable")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in training_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = m1_contrastive_objective(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        if not train_losses:
            raise ValueError("training_batches yielded no batches")
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in validation_batches:
                valid_losses.append(float(m1_contrastive_objective(model, batch)))
        if not valid_losses:
            raise ValueError("validation_batches yielded no batches")
        validation_loss = float(np.mean(valid_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("M1 training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return {
        "model_state_dict": best_state,
        "history": history,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "early_stopping_patience": patience,
    }


def train_m1(
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> Dict[str, object]:
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    residues, mask = make_synthetic_batch(batch_size)
    private_features = torch.randn(batch_size, 14, generator=torch.Generator().manual_seed(31))
    private_mask = torch.ones_like(private_features, dtype=torch.bool)
    groups = [f"genome-{index // 2}" for index in range(batch_size)]
    model = M1Encoder(residue_dim=len(AMINO_ACIDS), max_length=256)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(residues, mask, private_features, private_mask)
        loss = info_nce_loss(
            output["z_priv_sequence"],
            output["z_priv_features"],
            temperature=0.1,
            group_ids=groups,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return {
        "mode": "architecture_smoke_only_synthetic",
        "epochs": num_epochs,
        "loss_history": history,
        "lineage": lineage_tag(
            model_version="VaccineGPT-v1-dev",
            feature_version="synthetic-residue-test",
            label_version="none",
            batch_id="SMOKE",
        ),
    }


if __name__ == "__main__":
    print(json.dumps(train_m1(), indent=2))
