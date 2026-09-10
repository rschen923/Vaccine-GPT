from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from shared.contracts import validate_feature
from shared.lineage import lineage_tag


def save_feature_records(
    records: Iterable[Dict],
    output_path: str | Path,
    lineage: Dict[str, str],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for record in records:
        item = dict(record)
        item["lineage"] = lineage
        validate_feature(item)
        serializable.append(item)
    output.write_text("\n".join(json.dumps(item) for item in serializable) + "\n", encoding="utf-8")


def load_feature_matrix(
    records_path: str | Path,
    gene_ids: Sequence[str],
    view: str,
    track: str,
) -> torch.Tensor:
    wanted = set(gene_ids)
    loaded = {}
    with Path(records_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["gene_id"] in wanted and record["view"] == view and record["track"] == track:
                loaded[record["gene_id"]] = np.asarray(record["embedding"], dtype=np.float32)
    missing = wanted - set(loaded)
    if missing:
        raise KeyError(f"missing cached features for {view}/{track}: {sorted(missing)}")
    return torch.from_numpy(np.stack([loaded[gene_id] for gene_id in gene_ids]))


def extract_cached_embeddings(
    sequences: Sequence[Dict[str, str]],
    adapter: torch.nn.Module,
    view: str,
    track: str,
    lineage: Dict[str, str],
    batch_size: int = 4,
) -> list[Dict]:
    records = []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            embeddings = adapter([item["sequence"] for item in batch]).detach().cpu().numpy()
            for item, embedding in zip(batch, embeddings):
                records.append({
                    "gene_id": item["gene_id"], "view": view, "track": track,
                    "embedding": embedding.astype(np.float32).tolist(),
                    "lineage": lineage,
                })
    return records
