"""Authoritative L0--L4 VaccineGPT model and loss wiring.

The modules model constrained computational hypotheses.  Their consistency
losses are numerical regularizers, not assertions of biological validation.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import validate_batch_contract
from .l0_clef import DualEncoder
from .losses import (
    focal_loss, info_nce_loss, lambdarank_loss, listmle_loss, non_negative_pu_loss,
    vib_kl_loss,
)
from .physiology import (
    hla_consistency_loss, markov_consistency_loss, mhc_peptide_loss,
    ode_consistency_loss, powerlaw_loss, sir_consistency_loss,
    softmax_consistency_loss, topology_loss,
)


class FiveLayerVaccineGPT(nn.Module):
    """Coupled vaccine-design model with explicit detached layer boundaries."""

    def __init__(self, hidden_dim: int = 128, vib_dim: int = 32, domains: int = 8) -> None:
        super().__init__()
        self.l0 = DualEncoder(shared_dim=hidden_dim, latent_dim=vib_dim)
        self.node_encoder = nn.LazyLinear(hidden_dim)
        self.graph_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.vib_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, vib_dim), nn.GELU(), nn.Linear(vib_dim, vib_dim))
            for _ in range(4)
        )
        self.domain = nn.Linear(hidden_dim, domains)
        self.l1_head = nn.Linear(hidden_dim + 1 + 4 + 16, 2)
        self.l1_pcd = nn.Linear(hidden_dim, 4)
        self.l2_head = nn.Linear(hidden_dim + 4, 1)
        self.l2_cytokines = nn.Linear(hidden_dim, 3)  # IL-1beta, IL-22, TNF
        self.l2_iron = nn.Linear(hidden_dim, 1)
        self.l3_head = nn.Linear(hidden_dim + 2, 1)
        self.l3_ecology = nn.Linear(hidden_dim, 3)
        self.l4_head = nn.Linear(hidden_dim + 1 + 2 + 4 + 1, 1)
        self.l4_mhc = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha_ab(self) -> Tensor:
        return 0.05 + 0.90 * torch.sigmoid(self.alpha_raw)

    @staticmethod
    def _as_column(value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], -1)[:, :1]

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        validate_batch_contract(batch)
        l0 = self.l0(batch["x_a"], batch["x_b"])
        h = l0["h_shared"]
        node_state = self.node_encoder(batch["node_x"].float())
        adjacency = batch["PPI_adj"].float()
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        graph_context = (adjacency @ node_state) / degree
        h = h + torch.tanh(self.graph_gate(torch.cat((h, graph_context), dim=-1)))
        l1_input = torch.cat((h, batch["S_adj"].unsqueeze(-1), batch["pcd_A"], batch["ch_covariates"]), dim=-1)
        t1 = self.l1_head(l1_input)
        pcd = torch.sigmoid(self.l1_pcd(h))

        # L1 -> L2 is a one-way state injection: no cross-layer gradient.
        l2_input = torch.cat((h, pcd.detach()), dim=-1)
        t2 = self.l2_head(l2_input).squeeze(-1)
        cytokines = F.softplus(self.l2_cytokines(h))
        iron = F.softplus(self.l2_iron(h)).squeeze(-1)

        # L2 -> L3 one-way detached physiology.
        l3_input = torch.cat((h, cytokines[:, 1:2].detach(), iron.detach().unsqueeze(-1)), dim=-1)
        t3 = self.l3_head(l3_input).squeeze(-1)
        ecology = F.softplus(self.l3_ecology(h))
        tau_ti = 180.0 / (1.0 + ecology[:, 0])

        # Primary L3 -> L4 route is tau_TI -> v0, never alpha by default.
        v0 = (0.80 + 0.25 * (tau_ti.detach() / 180.0 - 1.0)).clamp(0.05, 0.99)
        l4_input = torch.cat(
            (h, v0.unsqueeze(-1), ecology[:, :2].detach(), batch["t4_feat"].float(),
             batch["v0"].reshape(-1, 1).float()),
            dim=-1,
        )
        t4 = self.l4_head(l4_input).squeeze(-1)
        mhc_input = torch.cat((batch["pep"].float(), batch["pseudo"].float()), dim=-1)
        mhc = self.l4_mhc(mhc_input).squeeze(-1)
        return {
            "h_shared": h, "sequence_embedding": l0["sequence"], "tabular_embedding": l0["tabular"],
            "mu": l0["mu"], "logvar": l0["logvar"], "domain_logits": self.domain(h),
            "z_l1": self.vib_layers[0](h), "z_l2": self.vib_layers[1](h),
            "z_l3": self.vib_layers[2](h), "z_l4": self.vib_layers[3](h),
            "t1": t1, "t2": t2, "t3": t3, "t4": t4, "pcd": pcd,
            "cytokines": cytokines, "iron": iron, "ecology": ecology, "tau_ti": tau_ti,
            "v0_coupled": v0, "alpha_ab": self.alpha_ab, "mhc": mhc,
        }

    def losses(self, batch: Mapping[str, Tensor], output: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return every differentiable objective with zero-safe loss terms."""
        zeros = output["t1"].sum() * 0.0
        task_mask = batch.get("task_mask", torch.ones(output["t1"].shape[0], 4, device=output["t1"].device))
        def masked(value: Tensor, column: int) -> Tensor:
            mask = task_mask[:, column].to(value)
            return (value * mask).sum() / mask.sum().clamp_min(1.0)
        transition = torch.softmax(output["pcd"], dim=-1)
        state = torch.softmax(torch.stack((batch["R0"], output["v0_coupled"], 1 - output["v0_coupled"]), -1), -1)
        rhs = -0.1 * output["cytokines"]
        trajectory = torch.stack((output["cytokines"], output["cytokines"] + 0.02 * rhs), 0)
        p_idx, u_idx = batch["pu_pos_idx"].long(), batch["pu_unl_idx"].long()
        pu = zeros if p_idx.numel() == 0 or u_idx.numel() == 0 else non_negative_pu_loss(
            output["t2"][p_idx], output["t2"][u_idx]
        )
        return {
            "info_nce": info_nce_loss(output["sequence_embedding"], output["tabular_embedding"]),
            "dann": F.cross_entropy(output["domain_logits"], batch["domain"].long()),
            "vib_l1": vib_kl_loss(output["mu"], output["logvar"]),
            "vib_l2": output["z_l2"].square().mean() * 1e-4,
            "vib_l3": output["z_l3"].square().mean() * 1e-4,
            "vib_l4": output["z_l4"].square().mean() * 1e-4,
            "ode_l1": F.mse_loss(output["pcd"].mean(-1), batch["fba"]),
            "softmax_l1": softmax_consistency_loss(output["t1"]),
            "markov_l1": markov_consistency_loss(transition),
            "t1": masked(
                torch.nn.functional.cross_entropy(output["t1"], batch["y_vivo"].long(), reduction="none"),
                0,
            ),
            "ode_l2": ode_consistency_loss(trajectory, torch.stack((rhs, rhs), 0), 0.02),
            "topology_l2": topology_loss(output["t2"], batch["PPI_adj"]),
            "iron_l2": F.mse_loss(output["iron"], batch["y_sec"]) + 0.1 * F.mse_loss(
                output["iron"], batch["N_star"].float().mean().expand_as(output["iron"])
            ),
            "t2": pu * task_mask[:, 1].mean(),
            "powerlaw_l3": powerlaw_loss(output["ecology"][:, 0] + 1e-6, batch["N_p"] + 1e-6, 0.25),
            "lv_l3": F.relu(-output["ecology"]).mean(),
            "hill_l3": F.mse_loss(torch.sigmoid(output["t3"]), batch["y3"]),
            "t3": listmle_loss(output["t3"], batch["y3"]) * task_mask[:, 2].mean() + 0.1 * F.mse_loss(
                torch.tanh(output["tau_ti"] / 180.0),
                torch.tanh(batch["t_obs"].float() / 180.0),
            ),
            "sir_l4": sir_consistency_loss(state),
            "hla_l4": hla_consistency_loss(batch["hla_freq"].unsqueeze(0)),
            "evol_l4": F.relu(output["t4"] - 1.0).mean() + F.relu(-output["t4"]).mean(),
            "t4": lambdarank_loss(output["t4"], batch["t4_rel"]) * task_mask[:, 3].mean(),
            "l4_ai": mhc_peptide_loss(output["mhc"], batch["el"], batch["pep_mask"]) + 0.1 * F.mse_loss(
                torch.sigmoid(output["mhc"]), batch["ic50"].float().clamp(0.0, 1.0)
            ),
            "alpha_prior": ((output["alpha_ab"] - 0.25) / 0.08).square(),
        }
