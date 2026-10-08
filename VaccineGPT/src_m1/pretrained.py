from __future__ import annotations

from typing import Mapping

from src_m1.encoders.factory import build_feature_store
from src_m1.encoders.foundation import (
    CLEFCheckpointAdapter,
    CompositeProteinAdapter,
    ESM2Adapter,
    ESM3Adapter,
    Evo2Adapter,
    FoundationFeatureStore,
    FoundationModelConfig,
    RNAFMAdapter,
    TransformersSequenceAdapter,
)


def build_pretrained_adapters(config: Mapping[str, object]):
    """Build lazy adapters from the v1/v2 model manifest without loading weights."""
    if "adapters" in config:
        specifications = dict(config["adapters"])
    else:
        specifications = {}
        for version in ("v1", "v2"):
            section = config.get(version)
            if isinstance(section, Mapping):
                specifications.update(dict(section.get("adapters", {})))
        if not specifications:
            specifications = {
                name: value
                for name, value in config.items()
                if isinstance(value, Mapping) and "backend" in value and "config" in value
            }
    if not specifications:
        raise ValueError("model manifest contains no adapter specifications")
    v1_config = config.get("v1")
    primary_protein_adapter = (
        v1_config.get("primary_protein_adapter")
        if isinstance(v1_config, Mapping)
        else None
    )
    if (
        "protein" not in specifications
        and isinstance(primary_protein_adapter, str)
        and primary_protein_adapter in specifications
    ):
        specifications["protein"] = specifications[primary_protein_adapter]
    if "protein" not in specifications and "protein_esm2" in specifications:
        specifications["protein"] = specifications["protein_esm2"]
    if "protein" not in specifications and "protein_esm3" in specifications:
        specifications["protein"] = specifications["protein_esm3"]
    normalized_specifications = {}
    for name, raw_specification in specifications.items():
        specification = dict(raw_specification)
        if "config" not in specification:
            model_id = specification.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(f"adapter {name} must define a model_id")
            extra = dict(specification.get("extra", {}))
            if "max_length" in specification:
                extra["max_length"] = specification["max_length"]
            if "output_dimension" in specification:
                extra["embedding_dimension"] = specification["output_dimension"]
            specification["config"] = {
                "model_id": model_id,
                "local_path": specification.get("local_path"),
                "device": specification.get("device", "auto"),
                "dtype": specification.get("dtype", "float32"),
                "freeze": specification.get("frozen", True),
                "unfreeze_last_n_layers": specification.get(
                    "unfreeze_last_n_layers", 0
                ),
                "trust_remote_code": specification.get("trust_remote_code", False),
                "extra": extra,
            }
        normalized_specifications[name] = specification
    return build_feature_store(normalized_specifications).adapters


__all__ = [
    "CLEFCheckpointAdapter",
    "CompositeProteinAdapter",
    "ESM2Adapter",
    "ESM3Adapter",
    "Evo2Adapter",
    "FoundationFeatureStore",
    "FoundationModelConfig",
    "RNAFMAdapter",
    "TransformersSequenceAdapter",
    "build_pretrained_adapters",
]
