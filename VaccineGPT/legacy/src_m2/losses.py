from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss for the imbalanced T1 labels."""

    def __init__(self, gamma: float = 2.0, class_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits, target.long(), weight=self.class_weight, reduction="none"
        )
        probability = torch.exp(-cross_entropy)
        return (((1.0 - probability) ** self.gamma) * cross_entropy).mean()


class NonNegativePULoss(nn.Module):
    """Non-negative PU risk for T2 positive and unlabeled subsets."""

    def __init__(self, positive_prior: float):
        super().__init__()
        if not 0 < positive_prior < 1:
            raise ValueError("positive_prior must be in (0, 1)")
        self.positive_prior = positive_prior

    def forward(self, positive_logits: torch.Tensor, unlabeled_logits: torch.Tensor) -> torch.Tensor:
        positive_risk = F.softplus(-positive_logits).mean()
        unlabeled_negative_risk = F.softplus(unlabeled_logits).mean()
        positive_negative_risk = F.softplus(positive_logits).mean()
        negative_risk = unlabeled_negative_risk - self.positive_prior * positive_negative_risk
        return self.positive_prior * positive_risk + torch.relu(negative_risk)
