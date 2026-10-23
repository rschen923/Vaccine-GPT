from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import (
    classify_label_evidence,
    detect_legacy_identifier_problem,
    group_labels_by_gene_task,
    normalize_protein,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def _write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _vfdb_sequence_aliases(data_root: Path) -> dict[str, list[dict[str, str]]]:
    aliases: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_dir = data_root / "raw"
    for name in ("VFDB_setA_pro.fas.gz", "VFDB_setB_pro.fas.gz"):
        path = raw_dir / name
        if not path.is_file():
            continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            header = None
            sequence_parts: list[str] = []
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        normalized = normalize_protein("".join(sequence_parts))
                        if normalized["sequence_sha256"]:
                            aliases[normalized["sequence_sha256"]].append(
                                {"gene_id": header.split()[0], "source_file": name}
                            )
                    header, sequence_parts = line[1:], []
                else:
                    sequence_parts.append(line)
            if header is not None:
                normalized = normalize_protein("".join(sequence_parts))
                if normalized["sequence_sha256"]:
                    aliases[normalized["sequence_sha256"]].append(
                        {"gene_id": header.split()[0], "source_file": name}
                    )
    return aliases


def audit_and_clean_public_final(
    data_root: Path, output_dir: Path, seed: int = 20261007, sample_size: int = 1000
) -> dict[str, Any]:
    source_dir = data_root / "processed" / "public_final"
    sequence_path = source_dir / "sequences.jsonl"
    label_path = source_dir / "labels.jsonl"
    if not sequence_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"expected sequences.jsonl and labels.jsonl in {source_dir}")

    raw_label_rows = list(read_jsonl(label_path))
    label_context: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in raw_label_rows:
        label_context[str(row.get("gene_id") or "")].add(
            (
                str(row.get("source") or row.get("database") or "unknown"),
                str(row.get("task") or "unknown"),
            )
        )
    source_aliases = _vfdb_sequence_aliases(data_root)
    original_to_corrected: dict[str, str] = {}
    sequence_rows: list[dict[str, Any]] = []
    sequence_by_original_id: dict[str, dict[str, Any]] = {}
    sequence_by_id: dict[str, dict[str, Any]] = {}
    duplicate_gene_ids: list[str] = []
    sequence_digest_to_ids: dict[str, list[str]] = defaultdict(list)
    sequence_reasons: Counter[str] = Counter()
    identifier_flags: Counter[str] = Counter()
    sequence_role_counts: Counter[str] = Counter()
    recovered_identifiers = 0
    unmatched_vfdb_sequences = 0
    for raw in read_jsonl(sequence_path):
        original_gene_id = str(raw.get("gene_id") or "")
        if original_gene_id in sequence_by_original_id:
            duplicate_gene_ids.append(original_gene_id)
        contexts = label_context.get(original_gene_id, set())
        source_names = {source.casefold() for source, _ in contexts}
        tasks = {task for _, task in contexts}
        if "iedb" in source_names and "T3b" in tasks and "vfdb" not in source_names:
            sequence_role = "epitope_peptide"
        elif "vfdb" in source_names and "T2" in tasks:
            sequence_role = "protein"
        else:
            sequence_role = "unknown"
        sequence_role_counts[sequence_role] += 1
        normalized = normalize_protein(raw.get("protein_sequence") or "", sequence_role)
        matches = source_aliases.get(normalized["sequence_sha256"] or "", [])
        matched_ids = sorted({str(match["gene_id"]) for match in matches})
        recovered_id = matched_ids[0] if len(matched_ids) == 1 else original_gene_id
        if "vfdb" in source_names and not matched_ids:
            unmatched_vfdb_sequences += 1
        if recovered_id != original_gene_id:
            recovered_identifiers += 1
        original_to_corrected[original_gene_id] = recovered_id
        gene_id = recovered_id
        flags = detect_legacy_identifier_problem(gene_id)
        identifier_flags.update(flags)
        sequence_reasons.update(normalized["quality_reasons"])
        record = {
            **raw,
            "gene_id": gene_id,
            "original_gene_id": original_gene_id,
            "sequence": normalized["sequence"],
            "sequence_sha256": normalized["sequence_sha256"],
            "sequence_length": normalized["length"],
            "sequence_role": sequence_role,
            "sequence_quality_status": normalized["quality_status"],
            "sequence_quality_reasons": normalized["quality_reasons"],
            "identifier_quality_flags": flags,
            "source_identifier_recovery": {
                "status": "recovered_exact_sequence_match" if recovered_id != original_gene_id else (
                    "ambiguous_exact_sequence_matches" if len(matched_ids) > 1 else (
                        "no_exact_source_match" if "vfdb" in source_names else "not_applicable"
                    )
                ),
                "source_ids": matched_ids,
                "source_files": sorted({str(match["source_file"]) for match in matches}),
            },
            "data_role": "sequence_record",
        }
        sequence_rows.append(record)
        sequence_by_original_id[original_gene_id] = record
        sequence_by_id[gene_id] = record
        if normalized["sequence_sha256"]:
            sequence_digest_to_ids[normalized["sequence_sha256"]].append(gene_id)

    label_rows: list[dict[str, Any]] = []
    labels_by_gene_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_label_rows:
        evidence = classify_label_evidence(raw)
        original_gene_id = str(raw.get("gene_id") or "")
        corrected_gene_id = original_to_corrected.get(original_gene_id, original_gene_id)
        record = {
            **raw,
            "gene_id": corrected_gene_id,
            "original_gene_id": original_gene_id,
            **evidence,
            "legacy_level": raw.get("level"),
            "data_role": "label_evidence",
        }
        label_rows.append(record)
        key = (str(record.get("gene_id") or ""), str(record.get("task") or ""))
        labels_by_gene_task[key].append(record)

    groups = group_labels_by_gene_task(label_rows)
    label_source_counts = Counter(str(row.get("source") or "unknown") for row in label_rows)
    task_counts = Counter(str(row.get("task") or "unknown") for row in label_rows)
    evidence_counts = Counter(str(row["evidence_level"]) for row in label_rows)
    label_values = Counter()
    for row in label_rows:
        try:
            value = float(row["label"])
            label_values[str(value)] += 1
        except (TypeError, ValueError, KeyError):
            label_values["missing_or_non_numeric"] += 1
    weak_source_counts = Counter(
        str(row.get("source") or "unknown")
        for row in label_rows
        if row["training_role"] == "weak_supervision_only"
    )
    orphans = sorted(
        {gene_id for gene_id, _ in labels_by_gene_task} - set(sequence_by_id)
    )
    missing_labels = len(set(sequence_by_id) - {gene_id for gene_id, _ in labels_by_gene_task})
    sequence_duplicates = {
        digest: ids for digest, ids in sequence_digest_to_ids.items() if len(ids) > 1
    }
    duplicate_sequence_ids = sum(len(ids) - 1 for ids in sequence_duplicates.values())
    conflicting_groups = [
        {"gene_id": gene_id, "task": task, **group}
        for (gene_id, task), group in groups.items()
        if group["label_conflict"]
    ]
    ambiguous_ids = [
        row["gene_id"] for row in sequence_rows if row["identifier_quality_flags"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_sequences = output_dir / "sequences_training_candidate.jsonl"
    clean_labels = output_dir / "labels_training_candidate.jsonl"
    aliases_path = output_dir / "exact_sequence_aliases.jsonl"
    _write_jsonl(clean_sequences, sequence_rows)
    _write_jsonl(clean_labels, label_rows)
    aliases = (
        {"sequence_sha256": digest, "gene_ids": sorted(ids), "duplicate_count": len(ids) - 1}
        for digest, ids in sorted(sequence_duplicates.items())
    )
    _write_jsonl(aliases_path, aliases)

    eligible = [
        row
        for row in sequence_rows
        if row["sequence_quality_status"] != "invalid"
        and row["sequence_role"] == "protein"
        and not row["identifier_quality_flags"]
    ]
    if len(eligible) < sample_size:
        raise ValueError(
            f"Only {len(eligible)} unambiguous full-protein candidates remain; "
            f"cannot draw requested sample of {sample_size}."
        )
    rng = random.Random(seed)
    sample_count = min(sample_size, len(eligible))
    sampled = sorted(rng.sample(eligible, sample_count), key=lambda row: row["gene_id"])
    sampled_ids = {row["original_gene_id"] for row in sampled}
    sample_labels = [
        row for row in label_rows if str(row.get("gene_id") or "") in sampled_ids
    ]
    _write_jsonl(output_dir / f"sample_{sample_count}_sequences_seed_{seed}.jsonl", sampled)
    _write_jsonl(output_dir / f"sample_{sample_count}_labels_seed_{seed}.jsonl", sample_labels)

    summary = {
        "schema_version": "vaccinegpt-data-audit-1.0",
        "data_root": str(data_root),
        "source_files": {
            "sequences": str(sequence_path),
            "labels": str(label_path),
        },
        "seed": seed,
        "requested_sample_size": sample_size,
        "sample_size": sample_count,
        "sequence_count": len(sequence_rows),
        "label_evidence_rows": len(label_rows),
        "unique_gene_task_groups": len(groups),
        "duplicate_gene_ids": len(duplicate_gene_ids),
        "recovered_source_identifiers": recovered_identifiers,
        "unmatched_vfdb_sequences": unmatched_vfdb_sequences,
        "sequence_role_counts": dict(sequence_role_counts),
        "exact_duplicate_sequence_groups": len(sequence_duplicates),
        "exact_duplicate_sequence_excess_records": duplicate_sequence_ids,
        "label_conflict_groups": len(conflicting_groups),
        "orphan_label_gene_ids": len(orphans),
        "sequences_without_any_label": missing_labels,
        "invalid_sequence_records": sum(
            row["sequence_quality_status"] == "invalid" for row in sequence_rows
        ),
        "review_sequence_records": sum(
            row["sequence_quality_status"] == "review" for row in sequence_rows
        ),
        "sequence_quality_reasons": dict(sequence_reasons),
        "identifier_quality_flags": dict(identifier_flags),
        "ambiguous_identifier_records": len(ambiguous_ids),
        "ambiguous_source_identifier_matches": sum(
            row["source_identifier_recovery"]["status"] == "ambiguous_exact_sequence_matches"
            for row in sequence_rows
        ),
        "task_counts": dict(task_counts),
        "label_value_counts": dict(label_values),
        "legacy_source_level_counts": dict(Counter(str(row.get("legacy_level") or "missing") for row in label_rows)),
        "conservative_evidence_level_counts": dict(evidence_counts),
        "source_counts": dict(label_source_counts),
        "weak_supervision_source_counts": dict(weak_source_counts),
        "gold_eligible_rows": sum(row["training_eligible_as_gold"] for row in label_rows),
        "weak_supervision_only_rows": sum(
            row["training_role"] == "weak_supervision_only" for row in label_rows
        ),
        "conflict_examples": conflicting_groups[:100],
        "orphan_gene_id_examples": orphans[:100],
        "ambiguous_identifier_examples": ambiguous_ids[:100],
        "split_status": "not_created_no_genome_groups_or_homology_cluster_assignment",
        "training_readiness": {
            "normalized_copy_written": True,
            "raw_data_modified": False,
            "direct_experimental_gold_available": any(
                row["training_eligible_as_gold"] for row in label_rows
            ),
            "weak_labels_may_be_used_as_gold": False,
            "leakage_safe_split_available": False,
            "negative_label_rows": label_values["0.0"],
            "positive_label_rows": label_values["1.0"],
            "positive_only_label_distribution": label_values["1.0"] > 0 and label_values["0.0"] == 0,
            "non_binary_or_missing_label_rows": sum(
                label_values[value] for value in label_values if value not in {"0.0", "1.0"}
            ),
            "blocking_reasons": [
                "existing public_final rows lack row-level experiment/reference provenance",
                "gene identifiers and genome/group identifiers require independent validation",
                "no MMseqs2 90%-identity cluster assignment was recorded for this snapshot",
            ],
        },
        "outputs": {
            "sequences": str(clean_sequences),
            "labels": str(clean_labels),
            "exact_sequence_aliases": str(aliases_path),
            "sample_sequences": f"sample_{sample_count}_sequences_seed_{seed}.jsonl",
            "sample_labels": f"sample_{sample_count}_labels_seed_{seed}.jsonl",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def audit_all_processed_label_files(data_root: Path, output_dir: Path) -> dict[str, Any]:
    processed = data_root / "processed"
    files = sorted(processed.rglob("*labels.jsonl"))
    per_file: list[dict[str, Any]] = []
    totals = Counter()
    tiered_output = output_dir / "all_labels_evidence_tiered.jsonl"
    tiered_output.parent.mkdir(parents=True, exist_ok=True)
    tiered_output.write_text("", encoding="utf-8")
    conflict_examples: list[dict[str, Any]] = []
    for path in files:
        raw_records = list(read_jsonl(path))
        records = len(raw_records)
        invalid_json = 0
        sources: Counter[str] = Counter()
        levels: Counter[str] = Counter()
        tasks: Counter[str] = Counter()
        duplicates: set[tuple[str, str]] = set()
        conflicts: set[tuple[str, str]] = set()
        seen_labels: dict[tuple[str, str], set[float]] = defaultdict(set)
        evidence_rows = []
        grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for raw in raw_records:
            source = str(raw.get("source") or raw.get("database") or "unknown")
            sources[source] += 1
            levels[str(raw.get("level") or "missing")] += 1
            tasks[str(raw.get("task") or "missing")] += 1
            key = (str(raw.get("gene_id") or ""), str(raw.get("task") or ""))
            if key in seen_labels:
                duplicates.add(key)
            try:
                label = float(raw.get("label"))
                if math.isfinite(label):
                    seen_labels[key].add(label)
            except (TypeError, ValueError):
                invalid_json += 1
            if len(seen_labels[key]) > 1:
                conflicts.add(key)
            evidence = {
                **raw,
                **classify_label_evidence(raw),
                "source_file": str(path.relative_to(data_root)),
                "assembly_id": (
                    path.name.removesuffix(".labels.jsonl")
                    if path.name.endswith(".labels.jsonl")
                    else path.parent.name
                ),
                "legacy_level": raw.get("level"),
                "data_role": "label_evidence",
            }
            evidence_rows.append(evidence)
            grouped_rows[key].append(evidence)
        for key, group in grouped_rows.items():
            if len(group) > 1:
                duplicates.add(key)
            resolved = group_labels_by_gene_task(group)[key]
            if resolved["label_conflict"]:
                conflicts.add(key)
                if len(conflict_examples) < 100:
                    conflict_examples.append(
                        {
                            "source_file": str(path.relative_to(data_root)),
                            "gene_id": key[0],
                            "task": key[1],
                            "observed_labels": resolved["observed_labels"],
                        }
                    )
            for evidence in group:
                evidence["duplicate_gene_task_in_file"] = len(group) > 1
                evidence["label_conflict_in_file"] = resolved["label_conflict"]
                if resolved["label_conflict"]:
                    evidence["training_role"] = "quarantine_conflict"
        entry = {
            "path": str(path.relative_to(data_root)),
            "bytes": path.stat().st_size,
            "records": records,
            "duplicate_gene_task_groups": len(duplicates),
            "conflicting_gene_task_groups": len(conflicts),
            "invalid_or_non_numeric_labels": invalid_json,
            "source_counts": dict(sources),
            "legacy_level_counts": dict(levels),
            "task_counts": dict(tasks),
        }
        per_file.append(entry)
        totals.update(
            {
                "files": 1,
                "records": records,
                "duplicate_gene_task_groups": len(duplicates),
                "conflicting_gene_task_groups": len(conflicts),
                "invalid_or_non_numeric_labels": invalid_json,
                "bytes": path.stat().st_size,
            }
        )
        with tiered_output.open("a", encoding="utf-8", newline="\n") as stream:
            for record in evidence_rows:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "all_processed_label_files.jsonl", per_file)
    result = {
        "schema_version": "vaccinegpt-label-inventory-1.0",
        "processed_root": str(processed),
        "file_count": totals["files"],
        "label_row_count": totals["records"],
        "label_bytes": totals["bytes"],
        "duplicate_gene_task_groups_file_local_sum": totals["duplicate_gene_task_groups"],
        "conflicting_gene_task_groups_file_local_sum": totals["conflicting_gene_task_groups"],
        "invalid_or_non_numeric_labels": totals["invalid_or_non_numeric_labels"],
        "within_file_conflict_examples": conflict_examples,
        "tiered_label_copy": str(tiered_output),
        "training_status": (
            "weak_supervision_only_until_row_level_source_and_target_strain_provenance_are_verified"
        ),
        "scope_note": (
            "This pass inventories every processed *labels.jsonl file and checks "
            "row syntax/source/task/legacy level/within-file duplicate and conflict "
            "patterns. It does not validate all 100GB of sequence records against "
            "every label or prove each source's underlying experiment. The copied "
            "label ledger is evidence-tiered, preserves file/assembly provenance, "
            "and quarantines within-file conflicts; it is not yet a model-ready "
            "feature/label join."
        ),
    }
    (output_dir / "all_processed_labels_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def audit_all_processed_sequences(data_root: Path, output_dir: Path) -> dict[str, Any]:
    processed = data_root / "processed"
    paths = sorted(
        path
        for path in processed.rglob("*.jsonl")
        if path.name == "sequences.jsonl"
        or path.name.endswith(".sequences.jsonl")
        or (path.parent.name == "uniprot" and path.name == "validation_sample.jsonl")
    )
    if not paths:
        raise FileNotFoundError(f"no processed sequence JSONL files found under {processed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "sequence_hash_index.sqlite"
    if database_path.exists():
        raise FileExistsError(
            f"refusing to overwrite prior sequence audit index: {database_path}"
        )
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            """CREATE TABLE sequence_groups (
                sequence_role TEXT NOT NULL,
                sequence_sha256 TEXT NOT NULL,
                representative_gene_id TEXT NOT NULL,
                first_source_file TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                PRIMARY KEY (sequence_role, sequence_sha256)
            )"""
        )
        connection.execute(
            """CREATE TABLE sequence_file_occurrences (
                sequence_role TEXT NOT NULL,
                sequence_sha256 TEXT NOT NULL,
                source_file TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                PRIMARY KEY (sequence_role, sequence_sha256, source_file)
            )"""
        )
        file_audit_path = output_dir / "all_processed_sequence_files.jsonl"
        totals = Counter()
        with file_audit_path.open("w", encoding="utf-8", newline="\n") as audit_stream:
            for path in paths:
                line_count = 0
                missing_sequence = 0
                invalid_sequence = 0
                local_hashes: set[tuple[str, str]] = set()
                local_duplicates = 0
                relative_path = str(path.relative_to(data_root))
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"{path}:{line_number}: invalid JSON: {error}"
                            ) from error
                        line_count += 1
                        sequence = row.get("protein_sequence") or row.get("sequence")
                        if not sequence:
                            missing_sequence += 1
                            continue
                        source = str(row.get("source") or "").casefold()
                        sequence_role = str(
                            row.get("sequence_role")
                            or ("epitope_peptide" if source == "iedb" else "protein")
                        )
                        normalized = normalize_protein(sequence, sequence_role)
                        if normalized["quality_status"] == "invalid":
                            invalid_sequence += 1
                        digest = normalized["sequence_sha256"]
                        if not digest:
                            continue
                        key = (sequence_role, digest)
                        if key in local_hashes:
                            local_duplicates += 1
                        local_hashes.add(key)
                        gene_id = str(row.get("gene_id") or row.get("accession") or "")
                        connection.execute(
                            """INSERT INTO sequence_groups VALUES (?, ?, ?, ?, 1)
                               ON CONFLICT(sequence_role, sequence_sha256)
                               DO UPDATE SET record_count=record_count+1""",
                            (sequence_role, digest, gene_id, relative_path),
                        )
                        connection.execute(
                            """INSERT INTO sequence_file_occurrences VALUES (?, ?, ?, 1)
                               ON CONFLICT(sequence_role, sequence_sha256, source_file)
                               DO UPDATE SET record_count=record_count+1""",
                            (sequence_role, digest, relative_path),
                        )
                        if line_count % 50000 == 0:
                            connection.commit()
                entry = {
                    "path": relative_path,
                    "bytes": path.stat().st_size,
                    "records": line_count,
                    "missing_sequence": missing_sequence,
                    "invalid_sequence": invalid_sequence,
                    "exact_duplicate_excess_within_file": local_duplicates,
                }
                audit_stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                totals.update(
                    {
                        "files": 1,
                        "records": line_count,
                        "missing_sequence": missing_sequence,
                        "invalid_sequence": invalid_sequence,
                        "within_file_duplicate_excess": local_duplicates,
                        "bytes": path.stat().st_size,
                    }
                )
                connection.commit()
        duplicate_path = output_dir / "global_exact_sequence_duplicates.jsonl"
        duplicate_groups = 0
        duplicate_excess = 0
        cross_file_groups = 0
        with duplicate_path.open("w", encoding="utf-8", newline="\n") as duplicate_stream:
            query = """
                SELECT g.sequence_role, g.sequence_sha256, g.representative_gene_id,
                       g.first_source_file, g.record_count, COUNT(o.source_file)
                FROM sequence_groups g
                JOIN sequence_file_occurrences o
                  ON o.sequence_role=g.sequence_role
                 AND o.sequence_sha256=g.sequence_sha256
                GROUP BY g.sequence_role, g.sequence_sha256
                HAVING g.record_count > 1
            """
            for role, digest, representative, first_path, count, path_count in connection.execute(query):
                duplicate_groups += 1
                duplicate_excess += int(count) - 1
                cross_file_groups += int(path_count) > 1
                duplicate_stream.write(
                    json.dumps(
                        {
                            "sequence_role": role,
                            "sequence_sha256": digest,
                            "representative_gene_id": representative,
                            "first_source_file": first_path,
                            "record_count": int(count),
                            "source_file_count": int(path_count),
                            "exact_duplicate_excess": int(count) - 1,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"new sequence audit index failed integrity check: {integrity}")
        result = {
            "schema_version": "vaccinegpt-sequence-audit-1.0",
            "processed_root": str(processed),
            "file_count": totals["files"],
            "record_count": totals["records"],
            "sequence_file_bytes": totals["bytes"],
            "missing_sequence_records": totals["missing_sequence"],
            "invalid_sequence_records": totals["invalid_sequence"],
            "within_file_exact_duplicate_excess": totals["within_file_duplicate_excess"],
            "global_exact_duplicate_groups": duplicate_groups,
            "global_exact_duplicate_excess": duplicate_excess,
            "duplicate_groups_across_multiple_files": cross_file_groups,
            "index_path": str(database_path),
            "index_integrity": integrity,
            "sequence_file_audit": str(file_audit_path),
            "duplicate_groups_file": str(duplicate_path),
            "homology_audit_status": (
                "not_run_mmseqs2_unavailable; exact identity only, not 90-percent homology clustering"
            ),
            "scope_note": (
                "Scanned all processed sequence JSONL artifacts selected by canonical "
                "sequence filenames; raw downloaded archives were inventoried but not "
                "expanded or rescanned. Exact duplicates are grouped by sequence role; "
                "this is not a homology/leakage-safe split."
            ),
        }
        (output_dir / "all_processed_sequences_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively audit and clean processed VaccineGPT labels without modifying raw data."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--all-label-files",
        action="store_true",
        help="also inventory every processed *labels.jsonl shard (can take several minutes)",
    )
    parser.add_argument(
        "--all-sequence-files",
        action="store_true",
        help="scan every canonical processed sequence JSONL for exact duplicates (can take a long time)",
    )
    args = parser.parse_args()
    result = audit_and_clean_public_final(
        args.data_root, args.output_dir, args.seed, args.sample_size
    )
    if args.all_label_files:
        result["all_processed_labels"] = audit_all_processed_label_files(
            args.data_root, args.output_dir
        )
    if args.all_sequence_files:
        result["all_processed_sequences"] = audit_all_processed_sequences(
            args.data_root, args.output_dir
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
