from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def audit_file(path: Path) -> dict:
    records = read_records(path)
    ids = [str(row.get("gene_id", row.get("accession", ""))) for row in records]
    sequences = [str(row.get("protein_sequence") or row.get("dna_sequence") or row.get("sequence") or "") for row in records]
    sequence_hashes = [
        hashlib.sha256("".join(sequence.upper().split()).replace("-", "").encode()).hexdigest()
        for sequence in sequences if sequence
    ]
    has_sequence_field = any(
        row.get("protein_sequence") or row.get("dna_sequence") or row.get("sequence")
        for row in records
    )
    return {
        "path": str(path),
        "records": len(records),
        "missing_gene_id": sum(not item for item in ids),
        "duplicate_gene_id": len(ids) - len(set(item for item in ids if item)),
        "duplicate_sequence": len(sequence_hashes) - len(set(sequence_hashes)),
        "empty_sequence": sum(not item for item in sequences) if has_sequence_field else None,
        "tasks": dict(Counter(str(row.get("task", "unlabelled")) for row in records)),
        "levels": dict(Counter(str(row.get("level", "unknown")) for row in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit downloaded and normalized VaccineGPT data")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--download-manifest")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output", default="data/metadata/dataset_audit.json")
    args = parser.parse_args()
    report = {"files": [audit_file(Path(path)) for path in args.inputs]}
    if args.download_manifest:
        manifest = json.loads(Path(args.download_manifest).read_text(encoding="utf-8"))
        report["downloads"] = manifest.get("results", [])
        report["download_failures"] = [
            item for item in report["downloads"] if item.get("status") == "failed"
        ]
    if args.split_manifest:
        splits = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        flattened = [gene for genes in splits.values() for gene in genes]
        report["splits"] = {
            "sizes": {name: len(genes) for name, genes in splits.items()},
            "duplicate_gene_ids": len(flattened) - len(set(flattened)),
            "overlap": sorted(
                set(splits.get("train", [])) & set(splits.get("validation", []))
                | set(splits.get("train", [])) & set(splits.get("test", []))
                | set(splits.get("validation", [])) & set(splits.get("test", []))
            ),
        }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
