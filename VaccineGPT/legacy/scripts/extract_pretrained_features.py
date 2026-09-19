from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shared.lineage import lineage_tag
from src_m1.feature_store import extract_cached_embeddings, save_feature_records
from src_m1.pretrained import build_pretrained_adapters


def read_sequences(path: str) -> list[dict[str, str]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if not record.get("gene_id") or not record.get("sequence"):
                raise ValueError(f"sequence record {line_number} needs gene_id and sequence")
            records.append({"gene_id": str(record["gene_id"]), "sequence": str(record["sequence"])})
    if not records:
        raise ValueError("sequence input is empty")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen upstream embeddings into UDC-02 JSONL")
    parser.add_argument("--config", default="configs/pretrained_models.json")
    parser.add_argument("--adapter", required=True, help="adapter key from the config")
    parser.add_argument("--view", required=True, choices=("protein", "dna", "rna"))
    parser.add_argument("--track", default="SOM", choices=("SOM", "INT"))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-version", default="M1M2-GIC-SDE-0.2.0")
    parser.add_argument("--feature-version", default="FEAT-EXT-0.1.0")
    parser.add_argument("--label-version", default="LABEL-0.1.0")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = config["adapters"].get(args.adapter)
    if spec is None:
        raise KeyError(f"adapter not found: {args.adapter}")
    if spec.get("status") == "reference_only_not_loaded":
        raise ValueError("selected RNA-FM entry is reference metadata only; configure a real local RNA-FM adapter")
    adapter = build_pretrained_adapters({args.adapter: spec})[args.adapter]
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        adapter.cuda()
    sequences = read_sequences(args.input_jsonl)
    lineage = lineage_tag(args.model_version, args.feature_version, args.label_version, args.batch_id)
    records = extract_cached_embeddings(
        sequences, adapter, args.view, args.track, lineage, batch_size=args.batch_size,
    )
    save_feature_records(records, args.output_jsonl, lineage)
    print(json.dumps({"records": len(records), "output": args.output_jsonl, "lineage": lineage}, indent=2))


if __name__ == "__main__":
    main()
