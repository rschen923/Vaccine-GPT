"""Configuration defaults for the dependency-light VaccineGPT architecture."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    shared_dim: int = 128
    latent_dim: int = 32
    hidden_dim: int = 64
    alpha_prior: float = 0.25
    alpha_sd: float = 0.08
    alpha_min: float = 0.05
    alpha_max: float = 0.95
    eta_B: float = 40.0
    v0_base: float = 0.80
    eta_v: float = 0.25
    tau_ref: float = 180.0
    delta_H: float = 0.10
    enable_tau_alpha_coupling: bool = False


@dataclass(frozen=True)
class SolverConfig:
    method: str = "euler"
    steps: int = 1000
    dt: float = 0.02
    rtol: float = 1e-5
    atol: float = 1e-7


@dataclass(frozen=True)
class TrainingDefaults:
    pcgrad_threshold: float = -0.1
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    coupling_grids: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "k_N": (0.5, 1.0, 2.0),
            "eta_pyro": (0.5, 1.0, 2.0),
            "eta_IL22": (0.1, 0.5, 1.0),
            "eta_iron": (0.5, 1.0, 2.0),
            "eta_vir": (0.5, 1.0, 2.0),
        }
    )
