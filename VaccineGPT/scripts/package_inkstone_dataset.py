from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator


SOURCE_PATHS = (
    "metadata",
    "raw",
    "refseq_bacteria",
    "processed/refseq_training",
    "processed/uniprot",
    "processed/sra",
    "processed/public_training_sources_20261007_055028",
    "processed/eib202_h1_audit",
    "processed/vaccinegpt_eib202_proteome_training_261007",
    "processed/vaccinegpt_h3_clef_261008",
    "processed/vaccinegpt_h3_public_ranker_261007_candidate_v4",
)
EXCLUDED_PATH_MARKERS = ("iedb", "antigendb")
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".tmp"}
HUMAN_HOST_TAXON_IDS = {"9606"}
HUMAN_HOST_PATTERN = re.compile(r"\b(human|homo sapiens|h\. sapiens)\b", re.IGNORECASE)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _excluded_reason(relative_path: Path) -> str | None:
    parts = [part.casefold() for part in relative_path.parts]
    if any(marker in part for marker in EXCLUDED_PATH_MARKERS for part in parts):
        if "processed" not in parts:
            return "raw archive retained out of the processed, human-filtered bundle"
    if "bacfitbase" in relative_path.name.casefold() and relative_path.name.casefold() != (
        "bacfitbase_h1_observations.jsonl"
    ):
        return "raw BacFITBase archive requires row-level human-host filtering"
    if "m2_public_candidates" in parts and relative_path.suffix.casefold() not in {
        ".json",
        ".jsonl",
    }:
        return "literature document or supplementary artifact, not a curated data record"
    if relative_path.name.casefold().endswith(".challenge.html"):
        return "access-denial response page, not a source data artifact"
    if relative_path.suffix.casefold() in EXCLUDED_SUFFIXES:
        return "model checkpoint, fitted estimator, temporary file, or training artifact"
    return None


def _is_human_host(row: dict[str, Any]) -> bool:
    taxon_id = str(row.get("host_taxon_id") or "").strip()
    host_name = str(row.get("host_name") or "")
    return taxon_id in HUMAN_HOST_TAXON_IDS or bool(HUMAN_HOST_PATTERN.search(host_name))


def _is_human_record(row: dict[str, Any]) -> bool:
    if _is_human_host(row):
        return True
    relevant_keys = ("host", "organism", "species", "subject", "patient", "source")
    for key, value in row.items():
        if any(token in key.casefold() for token in relevant_keys):
            text = str(value)
            if re.search(r"\b(9606|human|homo sapiens|h\. sapiens)\b", text, re.IGNORECASE):
                return True
    return False


def _iter_files(root: Path, source_relative: Path) -> Iterator[tuple[Path, Path, str | None]]:
    source = root / source_relative
    if not source.is_dir():
        raise FileNotFoundError(f"required dataset source directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root)
            yield path, relative, _excluded_reason(relative)


def _copy_bacfitbase_without_human_hosts(source: Path, destination: Path) -> tuple[int, int]:
    included = 0
    excluded = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as input_stream, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_stream:
        for line_number, line in enumerate(input_stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSONL: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            if _is_human_record(row):
                excluded += 1
                continue
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            included += 1
    return included, excluded


def _copy_jsonl_without_human_records(source: Path, destination: Path) -> tuple[int, int]:
    included = 0
    excluded = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as input_stream, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_stream:
        for line_number, line in enumerate(input_stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSONL: {error}") from error
            if isinstance(row, dict) and _is_human_record(row):
                excluded += 1
                continue
            output_stream.write(line)
            included += 1
    return included, excluded


def package_dataset(
    data_root: Path,
    output_dir: Path,
    repo_root: Path,
    mode: str = "hardlink",
) -> dict[str, Any]:
    if mode not in {"hardlink", "copy"}:
        raise ValueError("mode must be hardlink or copy")
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    if _is_within(output_dir, repo_root):
        raise ValueError("dataset output must be outside the Git repository")
    if output_dir == data_root:
        raise ValueError("dataset output cannot replace the retained source data root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset output: {output_dir}")
    for source_relative in SOURCE_PATHS:
        if not (data_root / source_relative).is_dir():
            raise FileNotFoundError(
                f"required dataset source directory is missing: {data_root / source_relative}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    excluded_files: list[dict[str, str]] = []
    totals: dict[str, dict[str, int]] = {}
    for source_relative in SOURCE_PATHS:
        for source, relative, excluded_reason in _iter_files(data_root, Path(source_relative)):
            if excluded_reason is not None:
                excluded_files.append(
                    {"path": relative.as_posix(), "reason": excluded_reason}
                )
                continue
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative.suffix.casefold() == ".jsonl" and (
                "iedb" in relative.name.casefold()
                or "antigendb" in relative.name.casefold()
                or "bacfitbase" in relative.name.casefold()
            ):
                rows_included, human_rows_excluded = _copy_jsonl_without_human_records(
                    source, destination
                )
                copy_method = "filtered_copy"
            else:
                if mode == "hardlink":
                    os.link(source, destination)
                    copy_method = "hardlink"
                else:
                    shutil.copy2(source, destination)
                    copy_method = "copy"
                rows_included = None
                human_rows_excluded = 0
            dataset_name = Path(source_relative).parts[-1]
            group = totals.setdefault(dataset_name, {"files": 0, "bytes": 0})
            size = destination.stat().st_size
            group["files"] += 1
            group["bytes"] += size
            entries.append(
                {
                    "path": relative.as_posix(),
                    "bytes": size,
                    "copy_method": copy_method,
                    "jsonl_rows_included": rows_included,
                    "human_host_rows_excluded": human_rows_excluded,
                }
            )

    manifest = {
        "schema_version": "vaccinegpt-inkstone-m1-dataset-1.0",
        "created_for": "Intern Inkstone M1 training upload",
        "source_root": str(data_root),
        "package_root": str(output_dir),
        "storage_mode": mode,
        "protein_backbone_target": {
            "model_id": "esm3-sm-open-v1",
            "embedding_dimension": 1536,
            "runtime": "Biohub esm in a separate environment",
            "precomputed_esm2_embeddings_included": False,
        },
        "human_data_policy": {
            "IEDB_included_after_human_record_filter": True,
            "AntigenDB_included_after_human_record_filter": True,
            "BacFITBase_human_host_rows_filtered": True,
            "raw_archives_excluded_because_processed_records_are_included": True,
            "human_host_taxon_id": "9606",
        },
        "training_readiness": {
            "raw_bacterial_sequences_and_genomes_included": True,
            "public_labels_are_not_all_experimental_gold": True,
            "weak_or_predicted_labels_must_remain_task_specific": True,
            "mmseqs2_90_homology_split_available": False,
            "required_before_supervised_training": [
                "create and record MMseqs2 90%-identity groups before train/validation/test splitting",
                "keep unmeasured and unlabelled records separate from explicit negatives",
                "review per-task evidence and class/group counts",
                "generate new 1536-dimensional ESM-3 caches; do not reuse old ESM-2 caches or checkpoints",
            ],
        },
        "included_groups": totals,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "excluded_file_count": len(excluded_files),
        "excluded_files": excluded_files,
        "files": entries,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# VaccineGPT M1 upload dataset

This folder is a data-only upload bundle for Intern Inkstone. It is outside the
Git repository; the GitHub repository is kept code-only.

## Contents

- `refseq_bacteria/`: public bacterial assemblies and annotations for sequence
  crosswalks and gene-context features.
- `processed/refseq_training/`: normalized bacterial protein sequences and
  source-attributed VFDB matches. These labels are weak evidence, not blanket
  experimental negatives or gold labels.
- `raw/`: acquired public bacterial sequence, annotation, and CRISPRi/Tn-seq
  inputs. FASTQ reads are preserved as source data and still require QC,
  sample/condition mapping, guide or insertion-site mapping, and effect
  estimation before any supported H1/H4 labels can be emitted.
- `processed/public_training_sources_20261007_055028/`: task-specific public
  bacterial sources (including T3SEpp, SecReT6, BacFITBase, and DEG where
  eligible); predicted records remain predictions, not experimental positives.
- `processed/vaccinegpt_eib202_proteome_training_261007/` and the H3 result
  directories: EIB202 protein records and candidate rankings for experimental
  review. Ranking scores are not probabilities or vaccine-protection claims.
- `metadata/`: source acquisition and audit metadata.

Processed IEDB and AntigenDB records are included after filtering rows whose
host, organism, subject, or source fields explicitly identify human data.
BacFITBase rows with human host taxon 9606 or an explicit human host name are
filtered; its unfiltered raw archives are excluded. Article and
supplementary files, access-denial pages, model checkpoints, fitted estimators,
temporary files, literature-campaign exports, and the old mixed IEDB/VFDB
public-final snapshots are not included. No human organism or human participant
data are intentionally included. The manifest records filtered row counts and
excluded paths and reasons.

## M1 training contract

The intended M1 protein backbone is frozen Biohub ESM-3 `esm3-sm-open-v1`
(1,536 residue dimensions). The bundle contains source sequences, not
precomputed ESM-2 features or checkpoints. Generate ESM-3 features in its
separate runtime, record model/runtime provenance, and reject old ESM-2 caches.
Before supervised fitting, create an MMseqs2 90%-identity homology grouping and
split by group. The current public annotations are not uniformly E1/E2 gold
labels; preserve evidence level, task, source, measurement status, and
unmeasured/unknown states. Positive-only evidence supports candidate ranking or
similarity analysis, not a trained positive probability.

`dataset_manifest.json` lists every packaged file, its byte size, and whether it
is a hard link or a copied/filtered file. With the default hard-link mode, the
bundle appears as ordinary files to an uploader while avoiding a second
full-size local copy; keep the retained source tree intact until upload finishes.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root.parent / "data"
    parser = argparse.ArgumentParser(
        description="Create an upload-ready, code-external bacterial dataset bundle for M1."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_root / "inkstone_m1_dataset_20261008",
    )
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    result = package_dataset(args.data_root, args.output_dir, repo_root, args.mode)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema_version",
                    "package_root",
                    "storage_mode",
                    "file_count",
                    "total_bytes",
                    "included_groups",
                    "excluded_file_count",
                    "human_data_policy",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
