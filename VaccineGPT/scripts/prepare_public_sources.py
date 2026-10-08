from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import classify_label_evidence, normalize_protein
from shared.contracts import validate_lineage
from shared.dataset import sequence_key, write_jsonl


def fasta(path: Path, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    header = None
    sequence: list[str] = []
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


def _fasta_record(header: str, sequence_parts: list[str], source: str) -> dict[str, Any]:
    normalized = normalize_protein("".join(sequence_parts), "protein")
    full_identifier = header.split()[0] if header.split() else ""
    if not full_identifier:
        raise ValueError("VFDB FASTA header has no identifier")
    return {
        "gene_id": full_identifier,
        "original_accession": full_identifier,
        "source_header": header,
        "protein_sequence": normalized["sequence"],
        "sequence_hash": sequence_key(normalized["sequence"]),
        "sequence_role": "protein",
        "source": source,
        "genome_id": "unknown",
        "sequence_quality_status": normalized["quality_status"],
        "sequence_quality_reasons": normalized["quality_reasons"],
    }


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Read a top-level JSON array incrementally without materializing the 500MB payload."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        started = False
        finished = False
        while not finished:
            if position >= len(buffer):
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                buffer, position = chunk, 0
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"{path}: expected top-level JSON array")
                position += 1
                started = True
                continue
            while position < len(buffer) and buffer[position] in " ,\r\n\t":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                finished = True
                position += 1
                break
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError(f"{path}: incomplete JSON array")
                buffer, position = buffer[position:] + chunk, 0
                continue
            if not isinstance(item, dict):
                raise ValueError(f"{path}: expected array elements to be JSON objects")
            yield item
            position = end
        if not started or not finished:
            raise ValueError(f"{path}: missing closing JSON array")


def iedb(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sequences: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row_number, row in enumerate(_iter_json_array(path), 1):
        peptide = row.get("linear_sequence")
        if not peptide:
            rejected.append({"source": "IEDB", "row_number": row_number, "reason": "missing_linear_sequence"})
            continue
        normalized = normalize_protein(peptide, "epitope_peptide")
        if normalized["quality_status"] == "invalid":
            rejected.append(
                {
                    "source": "IEDB",
                    "row_number": row_number,
                    "reason": "invalid_epitope_sequence",
                    "sequence_quality_reasons": normalized["quality_reasons"],
                }
            )
            continue
        organism_iris = row.get("source_organism_iris") or []
        if not isinstance(organism_iris, list):
            organism_iris = [organism_iris]
        organism = str(organism_iris[0]) if organism_iris else "unknown"
        explicit_id = row.get("epitope_id") or row.get("record_id")
        if explicit_id:
            source_record_id = str(explicit_id)
        else:
            source_record_id = hashlib.sha256(
                (
                    f"{organism}\0{row.get('structure_id') or ''}\0"
                    f"{normalized['sequence_sha256']}"
                ).encode("utf-8")
            ).hexdigest()
        gene_id = f"IEDB_EPITOPE:{source_record_id}"
        sequences.append(
            {
                "gene_id": gene_id,
                "original_accession": source_record_id,
                "protein_sequence": normalized["sequence"],
                "sequence_hash": normalized["sequence_sha256"],
                "sequence_role": "epitope_peptide",
                "source": "IEDB",
                "genome_id": organism,
                "source_record_id": source_record_id,
                "sequence_quality_status": normalized["quality_status"],
                "sequence_quality_reasons": normalized["quality_reasons"],
            }
        )
        raw_measures = row.get("qualitative_measures") or []
        if not isinstance(raw_measures, list):
            raw_measures = [raw_measures]
        measures = [str(item).strip().casefold() for item in raw_measures]
        positive = any("positive" in item for item in measures)
        negative = any("negative" in item for item in measures)
        if positive == negative:
            rejected.append(
                {
                    "gene_id": gene_id,
                    "source": "IEDB",
                    "row_number": row_number,
                    "reason": "qualitative_outcome_missing_or_ambiguous",
                    "qualitative_measures": measures,
                }
            )
            continue
        assay_names = row.get("assay_names") or []
        if not isinstance(assay_names, list):
            assay_names = [assay_names]
        label = {
            "gene_id": gene_id,
            "task": "T3b",
            "label": 1.0 if positive else 0.0,
            "source": "IEDB",
            "source_record_id": source_record_id,
            "source_organism_iris": organism_iris,
            "primary_reference": row.get("reference") or row.get("pmid") or row.get("pubmed_id"),
            "assays": assay_names,
            "assay": " ".join(map(str, assay_names)),
            "qualitative_measures": measures,
            "label_semantics": "IEDB qualitative assay result; not a protection endpoint",
        }
        labels.append({**label, **classify_label_evidence(label)})
    return sequences, labels, rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse public protein and epitope sources while preserving role and evidence provenance."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", default="data/processed/public")
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    output = Path(args.output_dir)
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    lineage["batch_id"] = args.batch_id
    validate_lineage(lineage)
    sequences: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for filename in ("VFDB_setA_pro.fas.gz", "VFDB_setB_pro.fas.gz"):
        path = raw / filename
        if path.exists():
            records = fasta(path, "VFDB")
            sequences.extend(records)
            for row in records:
                label = {
                    "gene_id": row["gene_id"],
                    "task": "T2",
                    "label": 1.0,
                    "source": "VFDB",
                    "source_header": row["source_header"],
                    "label_semantics": "curated VFDB membership; not direct target-organism validation",
                }
                labels.append({**label, **classify_label_evidence(label)})
    iedb_path = raw / "iedb_epitope_search.json"
    if iedb_path.exists():
        iedb_sequences, iedb_labels, iedb_rejected = iedb(iedb_path)
        sequences.extend(iedb_sequences)
        labels.extend(iedb_labels)
        rejected.extend(iedb_rejected)
    for row in sequences + labels:
        row["lineage"] = lineage
    for row in rejected:
        row["lineage"] = lineage
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(sequences, output / "sequences.jsonl")
    write_jsonl(labels, output / "labels.jsonl")
    write_jsonl(rejected, output / "rejected.jsonl")
    summary = {
        "sequences": len(sequences),
        "label_evidence_rows": len(labels),
        "rejected_rows": len(rejected),
        "sequence_roles": {
            role: sum(row["sequence_role"] == role for row in sequences)
            for role in ("protein", "epitope_peptide")
        },
        "evidence_levels": {
            level: sum(row["evidence_level"] == level for row in labels)
            for level in ("E1", "E2", "E3", "E4")
        },
        "raw_inputs_modified": False,
        "output_dir": str(output),
    }
    (output / "source_preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
