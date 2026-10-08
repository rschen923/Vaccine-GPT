from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any


GENOME_ACCESSION = "CP001135.1"
SOURCE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id=CP001135.1&rettype=gbwithparts&retmode=text"
)
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_start(line: str) -> tuple[str, str] | None:
    if not line.startswith("     ") or len(line) <= 5:
        return None
    key = line[5:21].strip()
    if not key:
        return None
    return key, line[21:].strip()


def _parse_feature(lines: list[str]) -> dict[str, str]:
    qualifiers: dict[str, str] = {}
    active_key: str | None = None
    active_value = ""
    quoted = False

    def flush() -> None:
        nonlocal active_key, active_value, quoted
        if active_key is not None:
            qualifiers[active_key] = active_value
        active_key = None
        active_value = ""
        quoted = False

    for line in lines:
        field = line[21:].strip()
        if field.startswith("/"):
            flush()
            match = re.match(r"/([A-Za-z_][A-Za-z0-9_]*)=(.*)$", field)
            if not match:
                continue
            active_key, raw_value = match.groups()
            if raw_value.startswith('"'):
                raw_value = raw_value[1:]
                if raw_value.endswith('"'):
                    active_value = raw_value[:-1]
                else:
                    active_value = raw_value
                    quoted = True
            else:
                active_value = raw_value
        elif active_key is not None and quoted:
            continuation = field
            if continuation.endswith('"'):
                active_value += continuation[:-1]
                quoted = False
            else:
                active_value += continuation
        if active_key is not None and not quoted:
            flush()
    flush()
    return qualifiers


def iter_cds(path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    feature_key: str | None = None
    feature_location = ""
    feature_lines: list[str] = []
    in_features = False
    with path.open("r", encoding="ascii", errors="replace") as stream:
        for line in stream:
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN"):
                break
            if not in_features:
                continue
            start = _feature_start(line)
            if start is not None:
                if feature_key == "CDS":
                    qualifiers = _parse_feature(feature_lines)
                    qualifiers["location"] = feature_location
                    yield "CDS", qualifiers
                feature_key, feature_location = start
                feature_lines = []
            elif feature_key is not None:
                feature_lines.append(line.rstrip("\r\n"))
    if feature_key == "CDS":
        qualifiers = _parse_feature(feature_lines)
        qualifiers["location"] = feature_location
        yield "CDS", qualifiers


def extract_proteome(genbank_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing proteome file: {output_path}")
    report_path = output_path.with_suffix(".report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite extraction report: {report_path}")
    version_line = next(
        (line for line in genbank_path.read_text(encoding="ascii", errors="replace").splitlines()
         if line.startswith("VERSION")),
        "",
    )
    if GENOME_ACCESSION not in version_line:
        raise ValueError(f"expected GenBank VERSION {GENOME_ACCESSION}, got {version_line!r}")

    source_hash = sha256_file(genbank_path)
    records: list[dict[str, Any]] = []
    missing_locus_tag = 0
    missing_translation = 0
    invalid_translation = 0
    locus_tags: set[str] = set()
    duplicate_locus_tags: list[str] = []
    for _, qualifiers in iter_cds(genbank_path):
        locus_tag = qualifiers.get("locus_tag", "").strip()
        if not locus_tag:
            missing_locus_tag += 1
            continue
        translation = "".join(qualifiers.get("translation", "").split()).upper()
        if not translation:
            missing_translation += 1
            continue
        if len(translation) < 2 or set(translation) - AMINO_ACIDS:
            invalid_translation += 1
            continue
        if locus_tag in locus_tags:
            duplicate_locus_tags.append(locus_tag)
            continue
        locus_tags.add(locus_tag)
        records.append(
            {
                "gene_id": locus_tag,
                "sequence": translation,
                "sequence_sha256": hashlib.sha256(translation.encode("ascii")).hexdigest(),
                "gene_symbol": qualifiers.get("gene"),
                "protein_id": qualifiers.get("protein_id"),
                "product": qualifiers.get("product"),
                "location": qualifiers.get("location"),
                "organism": "Edwardsiella piscicida",
                "strain": "EIB202",
                "genome_accession": GENOME_ACCESSION,
                "source_url": SOURCE_URL,
                "source_genbank_sha256": source_hash,
                "label_status": "unlabelled",
                "label_note": "Protein sequence only; no essentiality or vaccine label inferred.",
            }
        )
    if duplicate_locus_tags:
        raise ValueError(f"duplicate locus tags in GenBank CDS records: {duplicate_locus_tags[:10]}")
    if not records:
        raise ValueError("no translated CDS records with unique locus_tag values were found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    if temporary_path.exists():
        raise FileExistsError(f"refusing to overwrite partial output: {temporary_path}")
    with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_path.replace(output_path)

    lengths = [len(record["sequence"]) for record in records]
    report = {
        "schema_version": "vaccinegpt-target-proteome-1.0",
        "genome_accession": GENOME_ACCESSION,
        "source_genbank": str(genbank_path),
        "source_genbank_sha256": source_hash,
        "source_url": SOURCE_URL,
        "output_jsonl": str(output_path),
        "output_sha256": sha256_file(output_path),
        "records": len(records),
        "missing_locus_tag_cds_excluded": missing_locus_tag,
        "missing_translation_cds_excluded": missing_translation,
        "invalid_translation_cds_excluded": invalid_translation,
        "duplicate_locus_tags": duplicate_locus_tags,
        "exact_duplicate_sequences": len(records) - len({record["sequence_sha256"] for record in records}),
        "protein_length_min": min(lengths),
        "protein_length_median": sorted(lengths)[len(lengths) // 2],
        "protein_length_max": max(lengths),
        "label_status": "all records unlabelled; no Tn-seq essentiality inference",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract annotated EIB202 protein translations with source provenance."
    )
    parser.add_argument("--genbank", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract_proteome(args.genbank, args.output_jsonl), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
