from __future__ import annotations

from typing import Mapping

from .foundation import (
    CLEFCheckpointAdapter,
    CompositeProteinAdapter,
    ESM2Adapter,
    Evo2Adapter,
    FoundationFeatureStore,
    FoundationModelConfig,
    RNAFMAdapter,
    TransformersSequenceAdapter,
)


def build_feature_store(config: Mapping[str, Mapping[str, object]]) -> FoundationFeatureStore:
    adapters = {}
    for view, raw in config.items():
        if view in ("clef", "proteome"):
            continue
        item = dict(raw)
        model_config = FoundationModelConfig(**item.pop("config"))
        backend = item.pop("backend")
        if backend == "esm2":
            adapter = ESM2Adapter(model_config)
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
    if "protein" in config and config["protein"]["backend"] == "esm2":
        protein = config["protein"]
        esm = adapters["protein"]
        clef = None
        proteome = None
        if "clef" in config:
            raw = dict(config["clef"])
            clef_cfg = FoundationModelConfig(**raw.pop("config"))
            clef = CLEFCheckpointAdapter(clef_cfg, raw["module_name"], raw["class_name"])
        if "proteome" in config:
            proteome_cfg = FoundationModelConfig(**dict(config["proteome"])["config"])
            proteome = TransformersSequenceAdapter(proteome_cfg)
        adapters["protein"] = CompositeProteinAdapter(
            FoundationModelConfig(**dict(protein["config"])),
            esm,
            clef=clef,
            proteome=proteome,
        )
    return FoundationFeatureStore(adapters)
