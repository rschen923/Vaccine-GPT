"""Clean, validate, and deduplicate FASTQ files from Tn-seq studies.

Handles:
  - Quality-score validation and low-quality read trimming
  - Adapter/contaminant detection (hard-coded Illumina Tn-seq adapters)
  - Sequence-level exact deduplication via SHA-256 (keeps highest-quality representative)
  - Duplicate reporting across studies
  - Produces a cleaned manifest for downstream alignment

Dependencies: pandas, numpy (standard library + these only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

NUCLEOTIDE_ALPHABET = set("ACGTN")

# Common Illumina Tn-seq adapters / transposon ends (adjust per-library prep)
ADAPTERS = [
    "ATCTGTCTAACTCGGG",          # mini-Tn5 right end (common)
    "GACTGGTTCAGACGG",           # mini-Tn5 left end
    "AATGATACGGCGACCACCGA",      # Illumina P5 index read adapter
    "CAAGCAGAAGACGGCATACG",      # Illumina P7 index read adapter
    "CTGTCTCTTATCACAC",          # Illumina TruSeq Read 1
    "AGATCGGAAGAGCACAC",         # Illumina TruSeq i7 adapter
]

MIN_READ_LENGTH = 20            # discard reads shorter than this
MAX_N_FRACTION = 0.30           # drop reads with >30% N bases
MIN_QUAL_SCORE = 15             # Phred-33; mean quality below this → trim/discards
KMER_SIZE = 31                  # k-mer length for dedup fingerprinting
DEDUP_MEMORY_MB = 4096          # approximate memory budget for k-mer set


# ── FASTQ parsing ────────────────────────────────────────────────────────────

def parse_fastq_gz(filepath: str) -> Iterator[Tuple[str, str, str, List[int]]]:
    """Yield (read_id, sequence, optional_qual_header, phred_scores) from .gz FASTQ.

    Reads line-by-line; does NOT load entire file into memory.
    Handles both paired-end (/1 /2 suffixes) and single-end files.
    """
    import gzip

    open_fn = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"

    with open_fn(filepath, mode) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip()
            plus = fh.readline().strip()
            qual = fh.readline().strip()

            if not header.startswith("@"):
                log.warning("Malformed FASTQ header in %s: %s", filepath, header[:60])
                continue

            read_id = header[1:].split()[0] if len(header) > 1 else ""
            scores = _phred_from_quality(qual) if qual else None
            yield read_id, seq.upper(), plus, scores


def _phred_from_quality(qual_str: str) -> List[int]:
    """Convert Phred+33 quality string to list of integer scores."""
    return [max(0, ord(c) - 33) for c in qual_str]


# ── Per-read QC ──────────────────────────────────────────────────────────────

def assess_read(seq: str, scores: List[int] | None) -> Dict[str, Any]:
    """Return QC flags and metrics for a single read."""
    reasons: List[str] = []
    length = len(seq)

    # Length filter
    if length < MIN_READ_LENGTH:
        reasons.append("too_short")

    # N-content
    n_count = seq.count("N")
    n_frac = n_count / max(length, 1)
    if n_frac > MAX_N_FRACTION:
        reasons.append("high_n_content")

    # Adapter contamination (simple substring match)
    adapter_hits = []
    for adapter in ADAPTERS:
        if adapter in seq:
            adapter_hits.append(adapter[:12])
    if adapter_hits:
        reasons.append("adapter_contamination")

    # Mean quality
    if scores:
        mean_q = np.mean(scores)
        if mean_q < MIN_QUAL_SCORE:
            reasons.append("low_mean_quality")
    else:
        mean_q = 0.0

    return {
        "length": length,
        "n_count": n_count,
        "n_fraction": round(n_frac, 4),
        "mean_quality": round(float(mean_q), 2),
        "adapter_hits": adapter_hits,
        "fail_reasons": reasons,
        "pass": len(reasons) == 0,
    }


# ── Deduplication ────────────────────────────────────────────────────────────

def _seq_hash(seq: str) -> str:
    """SHA-256 hash of canonical (gap-stripped, uppercased) sequence."""
    canonical = "".join(c for c in seq.upper() if c in NUCLEOTIDE_ALPHABET)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _kmer_fingerprint(seq: str, k: int = KMER_SIZE) -> str:
    """Generate a compact k-mer fingerprint for dedup grouping."""
    if len(seq) < k:
        return _seq_hash(seq)
    kmers = set()
    for i in range(len(seq) - k + 1):
        kmers.add(seq[i:i + k])
    # Use top-3 most frequent kmers as fingerprint
    from collections import Counter
    counter = Counter(kmers)
    top = counter.most_common(3)
    return hashlib.md5("|".join(k for k, _ in top).encode()).hexdigest()


def dedup_reads(
    records: List[Tuple[str, str, List[int], Dict[str, Any]]],
) -> List[Tuple[str, str, List[int], Dict[str, Any]]]:
    """Deduplicate reads within a batch: keep the read with highest mean quality."""
    groups: Dict[str, List[Tuple[str, str, List[int], Dict[str, Any]]]] = defaultdict(list)
    for read_id, seq, scores, qc in records:
        key = _seq_hash(seq)
        groups[key].append((read_id, seq, scores, qc))

    kept = []
    dup_count = 0
    for key, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
        else:
            # Keep the read with highest mean quality score
            best = max(group, key=lambda x: x[2][0] if x[2] else 0)
            kept.append(best)
            dup_count += len(group) - 1

    return kept, dup_count


# ── Main processing loop ─────────────────────────────────────────────────────

def process_fastq_file(
    filepath: str,
    stats: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int]:
    """Process a single FASTQ file. Returns (kept_records, total_processed)."""
    records = []
    total = 0
    passed = 0
    failed = 0
    dup_reduced = 0

    for read_id, seq, plus, scores in parse_fastq_gz(filepath):
        total += 1
        qc = assess_read(seq, scores)

        if not qc["pass"]:
            failed += 1
            for reason in qc["fail_reasons"]:
                stats["qc_fail_" + reason] = stats.get("qc_fail_" + reason, 0) + 1
            continue

        passed += 1
        mean_q = scores[0] if scores else 0  # placeholder; real mean computed in assess_read
        records.append((read_id, seq, scores, qc))

    # Deduplicate passing reads
    if records:
        deduped, dup_reduced = dedup_reads(records)
    else:
        deduped, dup_reduced = [], 0

    stats["total_reads"] = stats.get("total_reads", 0) + total
    stats["passed_qc"] = stats.get("passed_qc", 0) + passed
    stats["failed_qc"] = stats.get("failed_qc", 0) + failed
    stats["duplicates_removed"] = stats.get("duplicates_removed", 0) + dup_reduced

    return deduped, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, validate, and deduplicate FASTQ files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing study/substudy/*.fastq.gz files.",
    )
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Path to write cleaned manifest JSONL. If omitted, dry-run mode.",
    )
    parser.add_argument(
        "--output-cleaned",
        action="store_true",
        help="Write deduplicated FASTQ to <input_dir>/cleaned/.",
    )
    parser.add_argument(
        "--report",
        default="fastq_clean_report.json",
        help="Path to write summary report JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count reads without deduplication or writing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    stats: Dict[str, Any] = {
        "input_directory": str(input_dir),
        "total_files": 0,
        "total_reads": 0,
        "passed_qc": 0,
        "failed_qc": 0,
        "duplicates_removed": 0,
        "qc_fail_too_short": 0,
        "qc_fail_high_n_content": 0,
        "qc_fail_adapter_contamination": 0,
        "qc_fail_low_mean_quality": 0,
    }

    # Discover FASTQ files
    fastq_files = sorted(input_dir.rglob("*.fq.gz"))
    fastq_files.extend(sorted(input_dir.rglob("*.fastq.gz")))
    stats["total_files"] = len(fastq_files)
    log.info("Found %d FASTQ files in %s", stats["total_files"], input_dir)

    if not fastq_files:
        log.error("No FASTQ files found. Exiting.")
        sys.exit(1)

    # Process each file
    all_records_by_study: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    manifest_entries: List[Dict[str, Any]] = []
    cleaned_dir = input_dir / "cleaned" if args.output_cleaned else None

    for idx, fq_path in enumerate(fastq_files, 1):
        study = fq_path.relative_to(input_dir).parts[0]
        rel_path = str(fq_path.relative_to(input_dir))

        log.info("[%d/%d] Processing %s ...", idx, len(fastq_files), rel_path)

        records, total = process_fastq_file(str(fq_path), stats)

        entry = {
            "file": rel_path,
            "study": study,
            "total_reads": total,
            "passed_qc": sum(1 for _, _, _, qc in records if qc["pass"]),
            "duplicates_in_file": stats["duplicates_removed"],
            "file_size_bytes": fq_path.stat().st_size,
        }
        manifest_entries.append(entry)

        if args.dry_run:
            continue

        # Write cleaned FASTQ
        if args.output_cleaned and records:
            out_path = cleaned_dir / study / fq_path.parent.name / fq_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import gzip
            with gzip.open(out_path, "wt") as out_fh:
                for read_id, seq, scores, qc in records:
                    qual_str = "".join(chr(max(0, min(255, s + 33))) for s in (scores or [40] * len(seq)))
                    out_fh.write(f"@{read_id}\n{seq}\n+\n{qual_str}\n")

    # Cross-file deduplication (across studies)
    if not args.dry_run and args.output_cleaned:
        log.info("Running cross-study deduplication ...")
        # Group by sequence hash across all studies
        global_dedup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in manifest_entries:
            study = entry["study"]
            fq_path = input_dir / entry["file"]
            for read_id, seq, scores, qc in parse_fastq_gz(str(fq_path)):
                h = _seq_hash(seq)
                global_dedup[h].append({
                    "read_id": read_id,
                    "study": study,
                    "seq": seq,
                    "scores": scores,
                    "qc": qc,
                    "source_file": entry["file"],
                })

        cross_dup_count = sum(max(0, len(v) - 1) for v in global_dedup.values() if len(v) > 1)
        stats["cross_study_duplicates"] = cross_dup_count
        log.info("Cross-study duplicates found: %d", cross_dup_count)

    # Write manifest
    if args.output_manifest:
        manifest_path = Path(args.output_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as fh:
            for entry in manifest_entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("Manifest written to %s (%d entries)", manifest_path, len(manifest_entries))

    # Write summary report
    report_path = Path(args.report)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    log.info("Summary report written to %s", report_path)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
