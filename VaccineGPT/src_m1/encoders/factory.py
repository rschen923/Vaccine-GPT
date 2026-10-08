from __future__ import annotations

from typing import Mapping

from .foundation import (
    CLEFCheckpointAdapter,
    ESM2Adapter,
    ESM3Adapter,
    Evo2Adapter,
    FoundationFeatureStore,
    FoundationModelConfig,
    RNAFMAdapter,
    TransformersSequenceAdapter,
)


def build_feature_store(config: Mapping[str, Mapping[str, object]]) -> FoundationFeatureStore:
    adapters = {}
    for view, raw in config.items():
        item = dict(raw)
        model_config = FoundationModelConfig(**item.pop("config"))
        backend = item.pop("backend")
        if backend == "esm2":
            adapter = ESM2Adapter(model_config)
        elif backend == "esm3":
            adapter = ESM3Adapter(model_config)
        elif backend in ("dnabert2", "proteomelm"):
            adapter = TransformersSequenceAdapter(model_config)
        elif backend == "evo2":
            adapter = Evo2Adapter(model_config)
        elif backend == "rna_fm":
            adapter = RNAFMAdapter(model_config)
        elif backend == "clef":
            adapter = CLEFCheckpointAdapter(
                model_config, item["module_name"], item["class_name"]
            )
        else:
            raise ValueError(f"unsupported foundation backend: {backend}")
        adapters[view] = adapter
    return FoundationFeatureStore(adapters)
