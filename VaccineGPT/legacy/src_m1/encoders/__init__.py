"""Optional pretrained foundation-model adapters for M1.

Adapters are imported lazily so cached-feature and CPU smoke workflows do not
require fair-esm, transformers, or the Evo2 runtime.
"""

from .foundation import FoundationFeatureStore, FoundationModelConfig

__all__ = ["FoundationFeatureStore", "FoundationModelConfig"]
