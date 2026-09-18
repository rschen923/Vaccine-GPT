"""JSONL contract and tensor conversion for the new architecture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

VIEWS = ("protein", "dna", "rna", "omics")
TASKS = ("T1", "T2", "T3a", "T3b", "T4")
DEFAULT_DIMS = {"protein": 16, "dna": 12, "rna": 10, "omics": 8}


class JsonlValidationError(ValueError):
    """A schema error with a useful source line and field path."""

    def __init__(self, message: str, line_number: int | None = None):
        self.line_number = line_number
        prefix = f"line {line_number}: " if line_number is not None else ""
        super().__init__(prefix + message)


def _fail(message: str, line_number: int | None) -> None:
    raise JsonlValidationError(message, line_number)


def _finite_vector(value: Any, field: str, line_number: int | None) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        _fail(f"{field} must be a non-empty array", line_number)
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _fail(f"{field}[{index}] must be numeric", line_number)
        number = float(item)
        if not math.isfinite(number):
            _fail(f"{field}[{index}] must be finite", line_number)
        result.append(number)
    return result


def validate_record(
    record: Mapping[str, Any],
    *,
    line_number: int | None = None,
    dimensions: Mapping[str, int] | None = None,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Validate and normalize one canonical sample.

    Schema:
    ``sample_id`` and ``group_id`` are non-empty strings, ``views`` contains
    all four numeric vectors, and labels are optional only for inference.
    Missing labels are represented by ``null`` and are masked during training.
    """

    if not isinstance(record, Mapping):
        _fail("record must be a JSON object", line_number)
    for key in ("sample_id", "group_id", "views"):
        if key not in record:
            _fail(f"missing required field '{key}'", line_number)
    for key in ("sample_id", "group_id"):
        if not isinstance(record[key], str) or not record[key].strip():
            _fail(f"{key} must be a non-empty string", line_number)
    views = record["views"]
    if not isinstance(views, Mapping):
        _fail("views must be an object", line_number)
    dims = dict(dimensions or {})
    normalized_views: dict[str, list[float]] = {}
    for view in VIEWS:
        if view not in views:
            _fail(f"views is missing '{view}'", line_number)
        vector = _finite_vector(views[view], f"views.{view}", line_number)
        if view in dims and len(vector) != int(dims[view]):
            _fail(
                f"views.{view} has dimension {len(vector)}, expected {int(dims[view])}",
                line_number,
            )
        normalized_views[view] = vector

    labels = record.get("labels")
    if require_labels and labels is None:
        _fail("missing required field 'labels'", line_number)
    if labels is not None and not isinstance(labels, Mapping):
        _fail("labels must be an object or null", line_number)
    normalized_labels: dict[str, float | int | None] = {}
    if labels is not None:
        unknown_tasks = sorted(set(labels) - set(TASKS))
        if unknown_tasks:
            _fail(f"labels contains unknown task(s): {unknown_tasks}", line_number)
        for task in TASKS:
            value = labels.get(task)
            if value is None:
                normalized_labels[task] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail(f"labels.{task} must be numeric or null", line_number)
            number = float(value)
            if not math.isfinite(number):
                _fail(f"labels.{task} must be finite", line_number)
            if task == "T1":
                if number != int(number) or not 0 <= int(number) <= 2:
                    _fail("labels.T1 must be an integer in [0, 2]", line_number)
                normalized_labels[task] = int(number)
            else:
                if not 0 <= number <= 1:
                    _fail(f"labels.{task} must be in [0, 1]", line_number)
                normalized_labels[task] = number

    if "metadata" in record and not isinstance(record["metadata"], Mapping):
        _fail("metadata must be an object", line_number)
    weight = record.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        _fail("weight must be numeric", line_number)
    if not math.isfinite(float(weight)) or not 0 <= float(weight) <= 1:
        _fail("weight must be finite and in [0, 1]", line_number)
    return {
        "sample_id": record["sample_id"].strip(),
        "group_id": record["group_id"].strip(),
        "views": normalized_views,
        "labels": normalized_labels if labels is not None else None,
        "weight": float(weight),
        **({"metadata": dict(record["metadata"])} if isinstance(record.get("metadata"), Mapping) else {}),
    }


def load_jsonl(
    path: str | Path,
    *,
    dimensions: Mapping[str, int] | None = None,
    require_labels: bool = True,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    """Read, validate, and normalize JSONL with duplicate-ID detection."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_dimensions: dict[str, int] | None = (
        {name: int(value) for name, value in dimensions.items()} if dimensions is not None else None
    )
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise JsonlValidationError(f"invalid JSON ({exc.msg})", line_number) from exc
            record = validate_record(
                value,
                line_number=line_number,
                dimensions=expected_dimensions,
                require_labels=require_labels,
            )
            if expected_dimensions is None:
                expected_dimensions = {name: len(record["views"][name]) for name in VIEWS}
            if record["sample_id"] in seen:
                raise JsonlValidationError(
                    f"duplicate sample_id '{record['sample_id']}'", line_number
                )
            seen.add(record["sample_id"])
            records.append(record)
    if not records and not allow_empty:
        raise JsonlValidationError(f"{source} contains no records")
    return records


def validate_jsonl(
    path: str | Path,
    *,
    dimensions: Mapping[str, int] | None = None,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Return a machine-readable validation report, raising on bad input."""

    records = load_jsonl(path, dimensions=dimensions, require_labels=require_labels)
    groups = {record["group_id"] for record in records}
    return {"valid": True, "records": len(records), "groups": len(groups), "path": str(path)}


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Validate records and write deterministic, UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    dimensions: dict[str, int] | None = None
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for count, value in enumerate(records, 1):
            record = validate_record(
                value, line_number=count, dimensions=dimensions, require_labels=False
            )
            if dimensions is None:
                dimensions = {name: len(record["views"][name]) for name in VIEWS}
            if record["sample_id"] in seen:
                raise JsonlValidationError(f"duplicate sample_id '{record['sample_id']}'", count)
            seen.add(record["sample_id"])
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    if count == 0:
        raise JsonlValidationError("cannot write an empty JSONL dataset")
    return count


def records_to_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Convert validated records into views, labels, and per-row weights."""

    if not records:
        raise ValueError("records_to_batch requires at least one record")
    normalized = [validate_record(record, require_labels=False) for record in records]
    views = {
        view: torch.tensor(np.asarray([row["views"][view] for row in normalized], dtype=np.float32), device=device)
        for view in VIEWS
    }
    labels: dict[str, torch.Tensor] = {}
    for task in TASKS:
        values = [
            (row["labels"] or {}).get(task) if row["labels"] is not None else None
            for row in normalized
        ]
        labels[task] = torch.tensor(
            [-1.0 if value is None else float(value) for value in values],
            dtype=torch.long if task == "T1" else torch.float32,
            device=device,
        )
    weights = torch.tensor([row["weight"] for row in normalized], dtype=torch.float32, device=device)
    return views, labels, weights
