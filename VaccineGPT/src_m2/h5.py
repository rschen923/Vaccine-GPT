from __future__ import annotations

from typing import Any, Mapping

from torch import nn


class H5ImplementationHead(nn.Module):
    """v2-only implementation-route score head; untrained outputs are never operational advice."""

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.route_score = nn.Linear(input_dim, 1)

    def forward(self, context_embedding):
        return self.route_score(context_embedding).squeeze(-1)


def implementation_route(
    score: float | None,
    replacement_length_bp: int | None,
    prior_failures: int | None,
    model_trained: bool,
) -> dict[str, Any]:
    if not model_trained or score is None:
        return {
            "route": None,
            "status": "unavailable_untrained_h5",
            "non_model_rule": None,
            "warning": "No experimental construction and passage labels are available.",
        }
    if replacement_length_bp is None or prior_failures is None:
        return {
            "route": None,
            "status": "incomplete_inputs",
            "non_model_rule": None,
            "warning": "Replacement length and prior construction history are required.",
        }
    if replacement_length_bp < 0 or prior_failures < 0:
        raise ValueError("replacement length and prior failures must be nonnegative")
    if replacement_length_bp > 3000 or prior_failures >= 2:
        return {
            "route": "sacB_two_step_exchange",
            "status": "rule_based_fallback",
            "non_model_rule": "replacement_length_bp > 3000 or prior_failures >= 2",
            "h5_score": score,
        }
    return {
        "route": "lambda_red_candidate",
        "status": "model_supported_rule_fallback",
        "non_model_rule": "otherwise default to lambda-Red",
        "h5_score": score,
    }
