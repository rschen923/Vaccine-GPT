"""CPU-friendly end-to-end training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .curriculum import Curriculum
from .data import DEFAULT_DIMS, load_jsonl, records_to_batch
from .losses import multitask_loss
from .models import VaccineGPTModel
from .pcgrad import PCGrad
from .splits import grouped_split


@dataclass
class TrainingConfig:
    epochs: int = 6
    learning_rate: float = 1e-3
    latent_dim: int = 32
    hidden_dim: int = 32
    seed: int = 7
    pcgrad: bool = True
    pcgrad_threshold: float = -0.1
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    stage_epochs: int = 1


def train_records(records: list[dict], config: TrainingConfig = TrainingConfig()) -> tuple[VaccineGPTModel, list[dict]]:
    if len(records) < 3:
        raise ValueError("at least three records are required")
    torch.manual_seed(config.seed)
    dimensions = {name: len(records[0]["views"][name]) for name in DEFAULT_DIMS}
    model = VaccineGPTModel(dimensions, config.latent_dim, config.hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pc_optimizer = (
        PCGrad(optimizer, model.parameters(), threshold=config.pcgrad_threshold)
        if config.pcgrad else None
    )
    curriculum = Curriculum()
    history: list[dict] = []
    # Splitting is performed before any model fitting.  The one-batch smoke
    # trainer keeps validation/test records available for callers.
    split = grouped_split([record["group_id"] for record in records], seed=config.seed)
    train_records_only = [records[int(index)] for index in split.train]
    views, targets, sample_weights = records_to_batch(train_records_only)
    for epoch in range(config.epochs):
        model.train()
        stage = curriculum.stage(epoch)
        task_weights = {
            task: float(stage.task_weights.get(task, 0.0))
            for task in ("T1", "T2", "T3a", "T3b", "T4")
        }
        outputs = model(views, detach_couplings=stage.detach_couplings)
        _, task_losses = multitask_loss(
            outputs, targets, task_weights=task_weights, sample_weights=sample_weights
        )
        losses = [task_losses[name] * task_weights[name] for name in task_losses if task_weights[name] > 0]
        if pc_optimizer is not None and len(losses) > 1:
            pc_optimizer.backward(losses)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            pc_optimizer.step()
            total = sum(losses)
        else:
            total = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        history.append({"epoch": epoch, "stage": stage.name, "loss": float(total.detach())})
    return model, history


def train_jsonl(path: str, config: TrainingConfig = TrainingConfig()) -> tuple[VaccineGPTModel, list[dict]]:
    return train_records(load_jsonl(path), config)
