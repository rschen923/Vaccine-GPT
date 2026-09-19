from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_m1.encoders.factory import build_feature_store
from shared.lineage import lineage_tag
from shared.udc_io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UDC-02 features from pretrained models")
    parser.add_argument("--config", default="configs/foundation_models.json")
    parser.add_argument("--input", required=True, help="JSONL records with gene_id and sequences")
    parser.add_argument("--output", required=True, help="UDC-02 JSONL output")
    parser.add_argument("--lineage", required=True, help="JSON lineage object")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    store = build_feature_store(config)
    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for view in ("protein", "dna", "rna"):
        if view not in store.adapters:
            continue
        sequence_key = "sequence" if view != "rna" else "rna_sequence"
        selected = [record for record in records if record.get(sequence_key)]
        if not selected:
            continue
        embeddings = store.encode(view, [record[sequence_key] for record in selected]).cpu()
        for record, embedding in zip(selected, embeddings):
            output.append({
                "gene_id": record["gene_id"],
                "view": view,
                "track": record.get("track", "SOM"),
                "embedding": embedding.tolist(),
                "lineage": lineage,
            })
    write_jsonl(output, args.output, kind="feature")


if __name__ == "__main__":
    main()
