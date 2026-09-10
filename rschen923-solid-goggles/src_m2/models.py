from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class M2Predictor(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.task_heads = nn.ModuleDict(
            {
                "T1": nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.3), nn.Linear(hidden_dim, 3)),
                "T2": TaskHead(input_dim=input_dim, hidden_dim=hidden_dim),
                "T3": TaskHead(input_dim=input_dim, hidden_dim=hidden_dim),
            }
        )
        self.rank_head = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {name: head(h_final) for name, head in self.task_heads.items()}
        t1_prob = torch.softmax(outputs["T1"], dim=-1)
        rank_input = torch.cat([h_final, t1_prob], dim=-1)
        outputs["T4"] = self.rank_head(rank_input).squeeze(-1)
        return outputs

    def immune_profile(self, h_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(h_final)
        return {
            "T3a_epitope_logit": outputs["T3"],
            "T3b_immunogenicity": torch.sigmoid(outputs["T3"]),
        }
