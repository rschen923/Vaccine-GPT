from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class Lineage:
    model_version: str
    feature_version: str
    label_version: str
    batch_id: str

    def __post_init__(self) -> None:
        placeholders = {"", "unspecified", "unknown", "replace_with_batch_id", "template"}
        values = {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "batch_id": self.batch_id,
        }
        invalid = [
            key for key, value in values.items()
            if not str(value).strip() or str(value).strip().casefold() in placeholders
        ]
        if invalid:
            raise ValueError(f"lineage fields must be concrete values: {invalid}")

    def as_dict(self) -> Dict[str, str]:
        return {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "batch_id": self.batch_id,
        }


def lineage_tag(
    model_version: str,
    feature_version: str,
    label_version: str,
    batch_id: str,
) -> Dict[str, str]:
    return Lineage(model_version, feature_version, label_version, batch_id).as_dict()


def ensure_compatibility(current: Dict[str, str], required: Dict[str, str]) -> bool:
    for key in ("model_version", "label_version", "batch_id"):
        if current.get(key) != required.get(key):
            return False
    current_feature = current.get("feature_version", current.get("feat_version"))
    required_feature = required.get("feature_version", required.get("feat_version"))
    return current_feature == required_feature
