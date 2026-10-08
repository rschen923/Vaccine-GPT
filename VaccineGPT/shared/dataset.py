from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from shared.data_quality import EVIDENCE_WEIGHTS, classify_label_evidence


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
    """Preserve row-level evidence; quarantine conflicts instead of choosing a winner."""
    grouped = defaultdict(list)
    for record in records:
        item = dict(record)
        item["gene_id"] = str(item["gene_id"])
        item["task"] = task
        item.update(classify_label_evidence(item))
        item["legacy_level"] = item.get("level")
        item["weight"] = EVIDENCE_WEIGHTS[item["evidence_level"]]
        item["lineage"] = lineage
        grouped[item["gene_id"]].append(item)
    merged = []
    for gene_id, evidence in grouped.items():
        label_values = set()
        for item in evidence:
            try:
                label_values.add(float(item["label"]))
            except (KeyError, TypeError, ValueError):
                continue
        conflict = len(label_values) > 1
        source_names = sorted(
            {str(item.get("source") or item.get("database") or "unknown") for item in evidence}
        )
        for item in evidence:
            item["evidence_sources"] = source_names
            item["label_conflict"] = conflict
            if conflict:
                item["training_role"] = "quarantine_conflict"
                item["training_eligible_as_gold"] = False
            merged.append(item)
    return merged


def effective_class_weights(labels: Sequence[int], beta: float = 0.999) -> Dict[int, float]:
    counts = Counter(int(label) for label in labels)
    raw = {label: (1.0 - beta) / (1.0 - beta ** count) for label, count in counts.items()}
    mean = float(np.mean(list(raw.values()))) if raw else 1.0
    return {label: value / mean for label, value in raw.items()}


def balanced_sample_weights(
    labels: Sequence[int], tiers: Sequence[str], beta: float = 0.999
) -> np.ndarray:
    """Combine effective-number weights with conservative evidence-level weights."""
    class_weights = effective_class_weights(labels, beta=beta)
    tier_weights = {"E1": 1.0, "E2": 0.8, "E3": 0.4, "E4": 0.2}
    return np.asarray([
        class_weights[int(label)] * tier_weights.get(str(tier), 0.2)
        for label, tier in zip(labels, tiers)
    ], dtype=np.float32)


def write_jsonl(records: Iterable[Mapping], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")
