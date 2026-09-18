"""A minimal PCGrad implementation for shared multi-task parameters."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class PCGrad:
    """Wrap an optimizer and project pairwise-conflicting task gradients."""

    def __init__(self, optimizer: torch.optim.Optimizer, parameters: Iterable[nn.Parameter] | None = None, threshold: float = -0.1):
        self.optimizer = optimizer
        if parameters is None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
        else:
            params = list(parameters)
        self.parameters = [p for p in params if p.requires_grad]
        self.threshold = threshold

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, losses: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        losses = list(losses)
        if not losses:
            raise ValueError("PCGrad.backward requires at least one loss")
        self.zero_grad()
        gradients: list[list[torch.Tensor | None]] = []
        for index, loss in enumerate(losses):
            gradients.append(
                list(torch.autograd.grad(loss, self.parameters, retain_graph=index < len(losses) - 1, allow_unused=True))
            )
        projected: list[torch.Tensor] = []
        for parameter_index, parameter in enumerate(self.parameters):
            task_grads = [
                gradient[parameter_index].detach().clone()
                for gradient in gradients
                if gradient[parameter_index] is not None
            ]
            if not task_grads:
                projected.append(torch.zeros_like(parameter))
                continue
            for i in range(len(task_grads)):
                for j in range(len(task_grads)):
                    if i == j:
                        continue
                    dot = torch.sum(task_grads[i] * task_grads[j])
                    if dot < self.threshold:
                        denom = torch.sum(task_grads[j] * task_grads[j]).clamp_min(1e-12)
                        task_grads[i] = task_grads[i] - dot / denom * task_grads[j]
            projected.append(torch.stack(task_grads).mean(dim=0))
        for parameter, gradient in zip(self.parameters, projected):
            parameter.grad = gradient
        return projected

    def step(self) -> None:
        self.optimizer.step()
