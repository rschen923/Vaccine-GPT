from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.dataset import sequence_key, write_jsonl
from shared.lineage import lineage_tag


def fasta(path: Path, source: str) -> list[dict]:
    records, header, sequence = [], None, []
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append(_fasta_record(header, sequence, source))
                header, sequence = line[1:], []
            elif line:
                sequence.append(line)
    if header is not None:
        records.append(_fasta_record(header, sequence, source))
    return records


def _fasta_record(header: str, sequence: list[str], source: str) -> dict:
    value = "".join(sequence).upper().replace("-", "")
    gene_id = re.split(r"\s+|\|", header, maxsplit=1)[0]
    return {
        "gene_id": gene_id,
        "protein_sequence": value,
        "sequence_hash": sequence_key(value),
        "source": source,
        "genome_id": "unknown",
        "evidence_level": "L2",
    }


def iedb(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences, labels = [], []
    for row in payload:
        peptide = row.get("linear_sequence")
        structure_id = row.get("structure_id")
        if not peptide or not structure_id:
            continue
        gene_id = f"IEDB_EPITOPE:{structure_id}"
        source_organisms = row.get("source_organism_iris") or ["unknown"]
        sequences.append({
            "gene_id": gene_id,
            "protein_sequence": peptide,
            "sequence_hash": sequence_key(peptide),
            "source": "IEDB",
            "genome_id": str(source_organisms[0]),
            "evidence_level": "L2",
        })
        qualitative = [str(item).lower() for item in row.get("qualitative_measures", [])]
        label = 1.0 if any("positive" in item for item in qualitative) else 0.0
        labels.append({
            "gene_id": gene_id,
            "task": "T3",
            "level": "L2",
            "label": label,
            "weight": 0.7,
            "source": "IEDB",
            "mhc_alleles": row.get("mhc_allele_names", []),
            "assays": row.get("assay_names", []),
        })
    return sequences, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse downloaded VFDB and IEDB files into auditable UDC records")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output-dir", default="data/processed/public")
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    output = Path(args.output_dir)
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    sequences, labels = [], []
    for filename in ("VFDB_setA_pro.fas.gz", "VFDB_setB_pro.fas.gz"):
        path = raw / filename
        if path.exists():
            records = fasta(path, "VFDB")
            sequences.extend(records)
            labels.extend({
                "gene_id": row["gene_id"], "task": "T2", "level": "L1",
                "label": 1.0, "weight": 1.0, "source": "VFDB",
            } for row in records)
    iedb_path = raw / "iedb_epitope_search.json"
    if iedb_path.exists():
        iedb_sequences, iedb_labels = iedb(iedb_path)
        sequences.extend(iedb_sequences)
        labels.extend(iedb_labels)
    for row in sequences + labels:
        row["lineage"] = lineage
    unique = {}
    gene_to_hash = {row["gene_id"]: row["sequence_hash"] for row in sequences}
    for row in sequences:
        unique.setdefault(row["sequence_hash"], row)
    hash_to_gene = {sequence_hash: row["gene_id"] for sequence_hash, row in unique.items()}
    for row in labels:
        sequence_hash = gene_to_hash.get(row["gene_id"])
        if sequence_hash in hash_to_gene:
            row["gene_id"] = hash_to_gene[sequence_hash]
    write_jsonl(unique.values(), output / "sequences.jsonl")
    write_jsonl(labels, output / "labels.jsonl")
    rejected = [
        {"reason": "duplicate_sequence", "gene_id": row["gene_id"]}
        for row in sequences if unique.get(row["sequence_hash"], {}).get("gene_id") != row["gene_id"]
    ]
    write_jsonl(rejected, output / "rejected.jsonl")
    print(json.dumps({
        "sequences": len(unique), "labels": len(labels), "rejected": len(rejected),
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
