from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts import validate_label
from shared.data_quality import classify_label_evidence, normalize_protein
from shared.dataset import merge_annotations, sequence_key, write_jsonl
from shared.split_manifest import write_split_manifest


def read_table(path: str, delimiter: str | None = None) -> list[dict]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    logical_suffix = source.with_suffix("").suffix if source.suffix == ".gz" else source.suffix
    if logical_suffix.lower() in {".json", ".jsonl"}:
        with opener(source, "rt", encoding="utf-8") as handle:
            if logical_suffix.lower() == ".jsonl":
                return [json.loads(line) for line in handle if line.strip()]
            payload = json.load(handle)
            return payload if isinstance(payload, list) else [payload]
    if logical_suffix.lower() in {".fa", ".fna", ".fas", ".fasta"}:
        records, header, sequence = [], None, []
        with opener(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        records.append({"accession": header.split()[0], "sequence": "".join(sequence)})
                    header, sequence = line[1:], []
                else:
                    sequence.append(line)
        if header is not None:
            records.append({"accession": header.split()[0], "sequence": "".join(sequence)})
        return records
    with opener(source, "rt", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=delimiter or ",\t;")
        except csv.Error:
            dialect = csv.excel_tab if "\t" in sample else csv.excel
        return list(csv.DictReader(handle, dialect=dialect))


def value(row: dict, *names: str, required: bool = True) -> str | None:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    if required:
        raise ValueError(f"missing one of columns {names}")
    return None


def normalize_sequences(input_path: str, output_path: str) -> list[dict]:
    records = []
    for row in read_table(input_path):
        protein = value(row, "protein_sequence", "protein", "aa_sequence", required=False)
        dna = value(row, "dna_sequence", "cds", "sequence", required=False)
        if not protein and not dna:
            continue
        sequence_role = "protein" if protein else "dna"
        if protein:
            normalized = normalize_protein(protein, "protein")
            sequence = normalized["sequence"]
            key = normalized["sequence_sha256"]
            quality_status = normalized["quality_status"]
            quality_reasons = normalized["quality_reasons"]
        else:
            raw = "".join(str(dna).upper().split()).replace("-", "")
            invalid = sorted(set(raw) - set("ACGTN"))
            sequence = raw
            key = sequence_key(raw, alphabet="ACGTN") if raw and not invalid else None
            quality_reasons = ["invalid_dna_characters"] if invalid else []
            quality_status = "invalid" if invalid or not raw else "pass"
        records.append({
            "gene_id": value(row, "gene_id", "locus_tag", "protein_id", "accession"),
            "genome_id": value(row, "genome_id", "assembly", "organism", required=False) or "unknown",
            "operon_id": value(row, "operon_id", "operon", required=False),
            "contig_id": value(row, "contig_id", "contig", required=False),
            "protein_sequence": sequence if sequence_role == "protein" else None,
            "dna_sequence": sequence if sequence_role == "dna" else None,
            "sequence_role": sequence_role,
            "sequence_hash": key,
            "sequence_quality_status": quality_status,
            "sequence_quality_reasons": quality_reasons,
            "source": value(row, "source", "database", required=False) or "unknown",
        })
    write_jsonl(records, output_path)
    return records


def normalize_labels(
    input_paths: list[str], output_path: str, lineage: dict
) -> list[dict]:
    all_records = []
    for path in input_paths:
        for row in read_table(path):
            task = value(row, "task")
            if task == "T3":
                task = "T3b"
            source = value(row, "source", "database", required=False) or Path(path).stem
            record = {
                "gene_id": value(row, "gene_id", "locus_tag", "protein_id", "accession"),
                "task": task,
                "label": float(value(row, "label", "value", "score")),
                "source": source,
                "assay": row.get("assay") or row.get("assay_type") or row.get("assay_names"),
                "method": row.get("method") or row.get("evidence_type"),
                "primary_reference": (
                    row.get("primary_reference") or row.get("doi") or row.get("pmid")
                    or row.get("pmc") or row.get("publication")
                ),
                "organism": row.get("organism") or row.get("taxon") or row.get("target_organism"),
                "is_private": bool(row.get("is_private") or row.get("private_dataset")),
                "legacy_level": value(row, "level", required=False),
                "lineage": lineage,
            }
            evidence = classify_label_evidence(record)
            record.update(evidence)
            record["weight"] = evidence["evidence_weight"]
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
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    lineage["batch_id"] = args.batch_id
    output = Path(args.output_dir)
    sequence_records = normalize_sequences(args.sequences, output / "sequences.jsonl")
    label_records = normalize_labels(args.labels, output / "labels.jsonl", lineage)
    label_levels = {}
    for label in label_records:
        label_levels.setdefault(label["gene_id"], label.get("level", "unknown"))
    records_for_split = [
        {"gene_id": record["gene_id"], "group_id": record.get("operon_id") or record.get("contig_id") or record["gene_id"],
         "stratum": label_levels.get(record["gene_id"], "unknown")}
        for record in sequence_records
    ]
    write_split_manifest(records_for_split, output / "split_manifest.json")
    print(json.dumps({
        "sequences": len(sequence_records), "labels": len(label_records),
        "split_manifest": str(output / "split_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
