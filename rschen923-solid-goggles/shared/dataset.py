from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def canonical_sequence(sequence: str, alphabet: str) -> str:
    value = "".join(str(sequence).upper().split()).replace("-", "")
    allowed = set(alphabet)
    return "".join(base for base in value if base in allowed)


def sequence_key(sequence: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> str:
    canonical = canonical_sequence(sequence, alphabet)
    if not canonical:
        raise ValueError("empty sequence after canonicalization")
    return hashlib.sha256(canonical.encode()).hexdigest()


def merge_annotations(records: Iterable[Mapping], task: str, lineage: Dict[str, str]) -> List[Dict]:
    """Merge duplicate evidence without allowing a weak source to overwrite strong evidence."""
    grouped = defaultdict(list)
    for record in records:
        item = dict(record)
        item["gene_id"] = str(item["gene_id"])
        item["task"] = task
        grouped[item["gene_id"]].append(item)
    priority = {"L1": 4, "L2": 3, "L3": 2, "L4": 1}
    merged = []
    for gene_id, evidence in grouped.items():
        evidence.sort(key=lambda item: priority.get(str(item.get("level", "L4")), 0), reverse=True)
        best = evidence[0]
        levels = {str(item.get("level", "L4")) for item in evidence}
        best["level"] = min(levels, key=lambda level: -priority.get(level, 0))
        best["weight"] = {"L1": 1.0, "L2": 0.7, "L3": 0.5, "L4": 0.3}[best["level"]]
        best["evidence_sources"] = sorted({str(item.get("source", "unknown")) for item in evidence})
        best["lineage"] = lineage
        merged.append(best)
    return merged


def effective_class_weights(labels: Sequence[int], beta: float = 0.999) -> Dict[int, float]:
    counts = Counter(int(label) for label in labels)
    raw = {label: (1.0 - beta) / (1.0 - beta ** count) for label, count in counts.items()}
    mean = float(np.mean(list(raw.values()))) if raw else 1.0
    return {label: value / mean for label, value in raw.items()}


def balanced_sample_weights(
    labels: Sequence[int], tiers: Sequence[str], beta: float = 0.999
) -> np.ndarray:
    """Combine effective-number class weighting with tier reliability."""
    class_weights = effective_class_weights(labels, beta=beta)
    tier_weights = {"L1": 1.0, "L2": 0.7, "L3": 0.5, "L4": 0.3}
    return np.asarray([
        class_weights[int(label)] * tier_weights.get(str(tier), 0.3)
        for label, tier in zip(labels, tiers)
    ], dtype=np.float32)


def write_jsonl(records: Iterable[Mapping], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")
