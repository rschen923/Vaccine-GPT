from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts import validate_label
from shared.dataset import merge_annotations, sequence_key, write_jsonl
from shared.split_manifest import write_split_manifest


def read_table(path: str, delimiter: str | None = None) -> list[dict]:
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=delimiter or ",\t")
        return list(csv.DictReader(handle, dialect=dialect))


def value(row: dict, *names: str, required: bool = True) -> str | None:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    if required:
        raise ValueError(f"missing one of columns {names}")
    return None


def normalize_sequences(input_path: str, output_path: str) -> list[dict]:
    records, seen = [], set()
    for row in read_table(input_path):
        protein = value(row, "protein_sequence", "protein", "aa_sequence", required=False)
        dna = value(row, "dna_sequence", "cds", "sequence", required=False)
        if not protein and not dna:
            continue
        sequence = protein or dna
        alphabet = "ACDEFGHIKLMNPQRSTVWY" if protein else "ACGTN"
        key = sequence_key(sequence, alphabet)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "gene_id": value(row, "gene_id", "locus_tag", "protein_id", "accession"),
            "genome_id": value(row, "genome_id", "assembly", "organism", required=False) or "unknown",
            "operon_id": value(row, "operon_id", "operon", required=False),
            "contig_id": value(row, "contig_id", "contig", required=False),
            "protein_sequence": protein,
            "dna_sequence": dna,
            "sequence_hash": key,
            "source": value(row, "source", "database", required=False) or "unknown",
        })
    write_jsonl(records, output_path)
    return records


def normalize_labels(
    input_paths: list[str], output_path: str, lineage_path: str
) -> list[dict]:
    lineage = json.loads(Path(lineage_path).read_text(encoding="utf-8"))
    all_records = []
    for path in input_paths:
        for row in read_table(path):
            task = value(row, "task")
            level = value(row, "level", required=False) or "L2"
            record = {
                "gene_id": value(row, "gene_id", "locus_tag", "protein_id", "accession"),
                "task": task,
                "level": level,
                "label": float(value(row, "label", "value", "score")),
                "weight": {"L1": 1.0, "L2": 0.7, "L3": 0.5, "L4": 0.3}.get(level, 0.3),
                "source": value(row, "source", "database", required=False) or Path(path).stem,
                "lineage": lineage,
            }
            validate_label(record)
            all_records.append(record)
    merged = []
    for task in sorted({record["task"] for record in all_records}):
        merged.extend(merge_annotations(
            [record for record in all_records if record["task"] == task], task, lineage
        ))
    write_jsonl(merged, output_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize public tables into VaccineGPT manifests")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    output = Path(args.output_dir)
    sequence_records = normalize_sequences(args.sequences, output / "sequences.jsonl")
    label_records = normalize_labels(args.labels, output / "labels.jsonl", args.lineage)
    records_for_split = [
        {"gene_id": record["gene_id"], "group_id": record.get("operon_id") or record.get("contig_id") or record["gene_id"],
         "stratum": next((label["level"] for label in label_records if label["gene_id"] == record["gene_id"]), "unknown")}
        for record in sequence_records
    ]
    write_split_manifest(records_for_split, output / "split_manifest.json")
    print(json.dumps({
        "sequences": len(sequence_records), "labels": len(label_records),
        "split_manifest": str(output / "split_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
