from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def inspect_dataset(
    sequences_path: str | Path,
    labels_path: str | Path,
    split_path: str | Path | None = None,
) -> dict[str, Any]:
    sequences = _read_jsonl(Path(sequences_path))
    labels = _read_jsonl(Path(labels_path))
    sequence_ids = {str(row.get("gene_id")) for row in sequences}
    label_ids = {str(row.get("gene_id")) for row in labels}
    issues: list[dict[str, Any]] = []
    if not sequences:
        issues.append({"code": "EMPTY_SEQUENCES", "severity": "blocking"})
    if not labels:
        issues.append({"code": "EMPTY_LABELS", "severity": "blocking"})
    missing_labels = sorted(sequence_ids - label_ids)
    missing_sequences = sorted(label_ids - sequence_ids)
    if missing_labels:
        issues.append({"code": "UNLABELED_SEQUENCES", "severity": "warning", "count": len(missing_labels)})
    if missing_sequences:
        issues.append({"code": "LABELS_WITHOUT_SEQUENCE", "severity": "warning", "count": len(missing_sequences)})
    duplicate_ids = len(sequences) - len(sequence_ids)
    if duplicate_ids:
        issues.append({"code": "DUPLICATE_SEQUENCE_IDS", "severity": "error", "count": duplicate_ids})
    task_counts = Counter(str(row.get("task", "unknown")) for row in labels)
    level_counts = Counter(str(row.get("level", "unknown")) for row in labels)
    for task, count in task_counts.items():
        if count < 5:
            issues.append({"code": "SMALL_TASK", "severity": "warning", "task": task, "count": count})
    split_report = None
    if split_path:
        splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
        split_sets = {name: set(map(str, values)) for name, values in splits.items()}
        overlap = {}
        names = list(split_sets)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                common = sorted(split_sets[left] & split_sets[right])
                if common:
                    overlap[f"{left}:{right}"] = common
        if overlap:
            issues.append({"code": "SPLIT_OVERLAP", "severity": "blocking", "pairs": overlap})
        split_report = {
            "sizes": {name: len(values) for name, values in split_sets.items()},
            "overlap_pairs": {key: len(value) for key, value in overlap.items()},
        }
    blocking = sum(issue["severity"] == "blocking" for issue in issues)
    recommendations = []
    if missing_labels:
        recommendations.append("retain unlabeled sequences for unsupervised pretraining, but exclude them from supervised loss")
    if any(issue["code"] == "SMALL_TASK" for issue in issues):
        recommendations.append("use repeated grouped cross-validation and tier-weighted loss for sparse tasks")
    if blocking:
        recommendations.append("stop training until blocking issues are corrected")
    else:
        recommendations.append("continue with current split and record this audit in the checkpoint lineage")
    return {
        "status": "fail" if blocking else ("warn" if issues else "pass"),
        "sequence_count": len(sequences),
        "label_count": len(labels),
        "task_counts": dict(task_counts),
        "level_counts": dict(level_counts),
        "issues": issues,
        "split": split_report,
        "recommendations": recommendations,
    }


def optimize_training_policy(audit: dict[str, Any]) -> dict[str, Any]:
    """Turn audit findings into a conservative, reproducible training policy."""
    policy = {
        "drop_duplicates": True,
        "use_effective_number_weights": True,
        "use_evidence_tier_weights": True,
        "augment_sequences": False,
        "repeated_grouped_cv": False,
        "require_manual_review": audit.get("status") == "fail",
    }
    if any(issue["code"] == "SMALL_TASK" for issue in audit.get("issues", [])):
        policy["repeated_grouped_cv"] = True
    return policy
