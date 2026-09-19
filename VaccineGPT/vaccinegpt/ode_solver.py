"""Small differentiable ODE integrator with an actionable optional dependency."""

from __future__ import annotations

from collections.abc import Callable

import torch


def integrate(
    rhs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor,
    *,
    method: str = "euler",
    steps: int = 1000,
    dt: float = 0.02,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> torch.Tensor:
    """Integrate ``dy/dt = rhs(t, y)`` and return the trajectory.

    Euler is the standard-library path and is deliberately stable for the
    bounded RHS functions in this package. ``dopri5`` delegates to torchdiffeq
    when installed and raises a precise installation error otherwise.
    """
    if method == "dopri5":
        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise RuntimeError(
                "method='dopri5' requires optional dependency torchdiffeq; "
                "install it with `pip install -r requirements-external.txt`"
            ) from exc
        return odeint(rhs, y0, t, rtol=rtol, atol=atol, method="dopri5")
    if method != "euler":
        raise ValueError("method must be 'euler' or 'dopri5'")
    if steps < 1 or dt <= 0:
        raise ValueError("steps must be positive and dt must be positive")
    current = y0
    trajectory = [current]
    for index in range(steps):
        time = t[index] if index < t.numel() else t[-1]
        current = current + dt * rhs(time, current)
        trajectory.append(current)
    return torch.stack(trajectory)
