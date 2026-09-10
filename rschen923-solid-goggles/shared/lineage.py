from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class Lineage:
    model_version: str
    feat_version: str
    label_version: str
    batch_id: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "model_version": self.model_version,
            "feat_version": self.feat_version,
            "label_version": self.label_version,
            "batch_id": self.batch_id,
        }


def lineage_tag(model_version: str, feat_version: str, label_version: str, batch_id: str) -> Dict[str, str]:
    return Lineage(model_version, feat_version, label_version, batch_id).as_dict()


def ensure_compatibility(current: Dict[str, str], required: Dict[str, str]) -> bool:
    for key in ("model_version", "feat_version", "label_version", "batch_id"):
        if current.get(key) != required.get(key):
            return False
    return True
