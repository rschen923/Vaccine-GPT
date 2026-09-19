"""The dependency-light VaccineGPT training architecture.

The package deliberately has no dependency on the historical ``src_m1`` and
``src_m2`` implementations.  It provides one stable data contract, a coupled
multi-task model, and small training utilities that also work on CPU.
"""

from .data import (
    JsonlValidationError,
    load_jsonl,
    records_to_batch,
    validate_jsonl,
    validate_record,
    write_jsonl,
)
from .models import VaccineGPTModel
from .five_layer import FiveLayerVaccineGPT
from .spec_training import load_contract_jsonl, make_synthetic_contract_batch, train_contract_batch
from .config import ModelConfig, SolverConfig, TrainingDefaults
from .coupling import bounded_alpha, detach_signal, tau_to_v0
from .synthetic import SyntheticConfig, generate_synthetic_jsonl, make_synthetic_batch
from .real_data import load_real_batch

__all__ = [
    "JsonlValidationError",
    "SyntheticConfig",
    "VaccineGPTModel",
    "FiveLayerVaccineGPT",
    "ModelConfig",
    "SolverConfig",
    "TrainingDefaults",
    "bounded_alpha",
    "detach_signal",
    "tau_to_v0",
    "generate_synthetic_jsonl",
    "load_jsonl",
    "load_contract_jsonl",
    "make_synthetic_batch",
    "make_synthetic_contract_batch",
    "records_to_batch",
    "validate_jsonl",
    "validate_record",
    "write_jsonl",
    "train_contract_batch",
    "load_real_batch",
]
