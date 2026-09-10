from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

from shared.contracts import validate_feature, validate_graph, validate_label


VALIDATORS = {"feature": validate_feature, "label": validate_label, "graph": validate_graph}


def read_jsonl(path: str | Path, kind: str | None = None) -> Iterator[Dict]:
    validator = VALIDATORS.get(kind) if kind else None
    if kind and validator is None:
        raise ValueError(f"unsupported UDC kind: {kind}")
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if validator:
                try:
                    validator(record)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield record


def write_jsonl(records: Iterable[Mapping], path: str | Path, kind: str | None = None) -> None:
    validator = VALIDATORS.get(kind) if kind else None
    if kind and validator is None:
        raise ValueError(f"unsupported UDC kind: {kind}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            item = dict(record)
            if validator:
                validator(item)
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
