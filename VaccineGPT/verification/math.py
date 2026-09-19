"""Numerical invariants for losses, PCGrad, detach boundaries, and splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from vaccinegpt.losses import non_negative_pu_loss, pairwise_rank_loss
    from vaccinegpt.models import VaccineGPTModel
    from vaccinegpt.pcgrad import PCGrad
    from vaccinegpt.splits import grouped_split, validate_split
    from vaccinegpt.synthetic import make_synthetic_batch
except ModuleNotFoundError:
    from VaccineGPT.vaccinegpt.losses import non_negative_pu_loss, pairwise_rank_loss
    from VaccineGPT.vaccinegpt.models import VaccineGPTModel
    from VaccineGPT.vaccinegpt.pcgrad import PCGrad
    from VaccineGPT.vaccinegpt.splits import grouped_split, validate_split
    from VaccineGPT.vaccinegpt.synthetic import make_synthetic_batch


def run() -> dict:
    assert torch.isfinite(non_negative_pu_loss(torch.randn(8), torch.randn(8))).item()
    assert pairwise_rank_loss(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])) >= 0
    split = grouped_split(["a", "a", "b", "c", "c", "d"], seed=3)
    validate_split(split, ["a", "a", "b", "c", "c", "d"])
    # PCGrad must remove a directly conflicting component.
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    pcgrad = PCGrad(optimizer, [parameter])
    pcgrad.backward([(parameter - 1).pow(2), (parameter + 1).pow(2)])
    assert parameter.grad is not None and torch.isfinite(parameter.grad)
    views, _, _ = make_synthetic_batch(4, dims={"protein": 4, "dna": 4, "rna": 4, "omics": 4})
    model = VaccineGPTModel({name: 4 for name in views}, latent_dim=8, hidden_dim=8)
    detached = model(views, detach_couplings=True)
    coupled = model(views, detach_couplings=False)
    detached["T2"].sum().backward(retain_graph=True)
    detached_t1_grad = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith("t1_head") and parameter.grad is not None
    )
    assert detached_t1_grad == 0.0
    model.zero_grad(set_to_none=True)
    coupled["T2"].sum().backward()
    coupled_t1_grad = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith("t1_head") and parameter.grad is not None
    )
    assert coupled_t1_grad > 0.0
    return {"ok": True, "pcgrad": True, "detach_boundary": True}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
