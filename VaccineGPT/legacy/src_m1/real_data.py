from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import torch

from shared.contracts import validate_feature


def load_feature_views(
    path: str | Path,
    gene_ids: Sequence[str],
    track: str,
) -> Dict[str, torch.Tensor]:
    """Load UDC-02 cached vectors and infer each view's original dimension."""
    wanted = set(gene_ids)
    values: Dict[str, Dict[str, list[float]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            validate_feature(record)
            if record["track"] == track and record["gene_id"] in wanted:
                values.setdefault(record["view"], {})[record["gene_id"]] = record["embedding"]
    views = {}
    for view in ("protein", "dna", "rna") + (("omics",) if track == "INT" else ()):
        missing = wanted - set(values.get(view, {}))
        if missing:
            raise KeyError(f"missing {view} features for {len(missing)} genes")
        views[view] = torch.tensor([values[view][gene_id] for gene_id in gene_ids], dtype=torch.float32)
    return views
