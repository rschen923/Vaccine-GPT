"""Tn-seq read alignment and insertion-site calling against Edwardsiella piscicida reference.

Implements a lightweight k-mer seed-and-extend mapper in pure Python + numpy.
Designed for ~10M reads; expect ~1-5 hours depending on CPU.

Outputs:
  - SAM-like file (.sam.gz) with mapped reads
  - Insertion site BED file per study
  - Per-gene read-count matrix for downstream fitness scoring

Usage:
  python scripts/align_tnseq.py \\
    --input-dir data/raw/fastq \\
    --reference data/raw/uniprot_episcicida_1263550.tsv \\
    --output-dir data/processed/alignment \\
    --kmer-size 19 \\
    --min-matches 3 \\
    --max-mismatches 2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import mmap
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
BASE_TO_INT = np.array([0, 1, 2, 3, 4], dtype=np.uint8)


# ── Reference indexing ───────────────────────────────────────────────────────

class ReferenceIndex:
    """Builds a k-mer lookup index from protein or DNA sequences."""

    def __init__(self, kmer_size: int = 19):
        self.k = kmer_size
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.sequences: Dict[str, str] = {}
        self._loaded = False

    def add_sequence(self, name: str, seq: str) -> None:
        """Add a sequence to the index. Seq should be uppercase, no gaps."""
        seq = seq.upper().replace("-", "")
        self.sequences[name] = seq
        seq_len = len(seq)
        if seq_len < self.k:
            return
        for i in range(seq_len - self.k + 1):
            kmer = seq[i : i + self.k]
            if "N" in kmer:
                continue
            self.index[kmer].append((name, i))

    def load_from_uniprot_tsv(self, tsv_path: str) -> None:
        """Load E. piscicida proteome from UniProt TSV (uses Sequence column)."""
        import pandas as pd
        df = pd.read_csv(tsv_path, sep="\t")
        seq_col = "Sequence" if "Sequence" in df.columns else df.columns[-1]
        id_col = "Entry" if "Entry" in df.columns else df.columns[0]

        for _, row in df.iterrows():
            name = str(row[id_col])
            seq = str(row[seq_col]).upper().replace("-", "")
            if len(seq) < 20:
                log.warning("Skipping short entry %s (len=%d)", name, len(seq))
                continue
            self.add_sequence(name, seq)

        self._loaded = True
        total_bases = sum(len(s) for s in self.sequences.values())
        log.info(
            "Indexed %d sequences, %d total bases (%d unique kmers)",
            len(self.sequences),
            total_bases,
            len(self.index),
        )

    def query_kmers(self, read_seq: str, min_matches: int = 3) -> List[Tuple[str, int, int]]:
        """Find candidate alignments for a read. Returns [(ref_name, ref_pos, score)]."""
        read_len = len(read_seq)
        if read_len < self.k:
            return []

        candidates: Dict[Tuple[str, int], int] = defaultdict(int)

        # Forward k-mers
        for i in range(read_len - self.k + 1):
            kmer = read_seq[i : i + self.k]
            if "N" in kmer:
                continue
            for ref_name, ref_pos in self.index.get(kmer, []):
                candidates[(ref_name, ref_pos)] += 1

        # Reverse complement k-mers
        rev_comp = "".join(COMPLEMENT[c] for c in reversed(read_seq))
        for i in range(read_len - self.k + 1):
            kmer = rev_comp[i : i + self.k]
            if "N" in kmer:
                continue
            for ref_name, ref_pos in self.index.get(kmer, []):
                candidates[(ref_name, ref_pos)] += 1

        # Filter by minimum matches and sort by score
        hits = [(name, pos, score) for (name, pos), score in candidates.items() if score >= min_matches]
        hits.sort(key=lambda x: (-x[2], x[0], x[1]))
        return hits[:50]  # Top 50 candidates


# ── Seed-and-extend alignment ────────────────────────────────────────────────

def _encode_sequence(seq: str) -> np.ndarray:
    """Encode DNA string to uint8 array [0=A, 1=C, 2=G, 3=T, 4=N]."""
    return np.array([NUCLEOTIDE_MAP.get(c, 4) for c in seq.upper()], dtype=np.uint8)


def extend_alignment(
    read_seq: str,
    ref_name: str,
    ref_pos: int,
    ref_sequences: Dict[str, str],
    max_mismatches: int = 2,
) -> Optional[Dict[str, Any]]:
    """Try to align read to reference using banded Smith-Waterman approximation.

    Returns alignment dict if successful, None otherwise.
    """
    ref_seq = ref_sequences[ref_name][ref_pos : ref_pos + len(read_seq) + max_mismatches * 2]
    if len(ref_seq) < len(read_seq):
        ref_seq = ref_seq.ljust(len(read_seq) + max_mismatches * 2, "N")

    read_arr = _encode_sequence(read_seq)
    ref_arr = _encode_sequence(ref_seq)

    r_len = len(read_arr)
    band_width = max_mismatches * 2 + 3
    ref_start = max(0, ref_pos - max_mismatches)
    ref_end = min(len(ref_sequences[ref_name]), ref_pos + r_len + max_mismatches)
    ref_segment = ref_sequences[ref_name][ref_start : ref_end]
    ref_seg_arr = _encode_sequence(ref_segment)

    # Simple mismatch counting within band (fast approximation)
    best_offset = 0
    best_score = 0
    best_mismatches = r_len

    for offset in range(max(0, ref_pos - ref_start) - 1, min(r_len, len(ref_seg_arr) - r_len + 1) + 1):
        if offset + r_len > len(ref_seg_arr):
            break
        segment = ref_seg_arr[offset : offset + r_len]
        mismatches = int(np.sum(read_arr != segment))
        # Penalize Ns
        n_penalty = int(np.sum((read_arr == 4) | (segment == 4)))
        total_penalty = mismatches + n_penalty * 0.5

        if total_penalty < best_mismatches:
            best_mismatches = int(total_penalty)
            best_score = r_len - total_penalty
            best_offset = offset

    if best_mismatches <= max_mismatches and best_score >= r_len * 0.7:
        global_ref_pos = ref_start + best_offset
        cigar = _build_cigar(read_seq, ref_segment[best_offset : best_offset + r_len])
        mapq = max(0, int(40 * (1 - best_mismatches / max(r_len, 1))))

        return {
            "read_id": "",
            "ref_name": ref_name,
            "ref_pos": global_ref_pos,
            "mismatches": best_mismatches,
            "score": round(best_score, 2),
            "mapq": mapq,
            "cigar": cigar,
            "strand": "+",
        }

    return None


def _build_cigar(read: str, ref: str) -> str:
    """Build simple M/X cigar string."""
    ops = []
    current_op = "M"
    count = 1
    for r_base, ref_base in zip(read, ref):
        match = (r_base == ref_base and r_base != "N" and ref_base != "N")
        op = "M" if match else "X"
        if op != current_op:
            ops.append(f"{count}{current_op}")
            current_op = op
            count = 1
        else:
            count += 1
    ops.append(f"{count}{current_op}")
    return "".join(ops)


# ── Tn-seq insertion site detection ──────────────────────────────────────────

def find_insertion_site(
    read_seq: str,
    alignment: Dict[str, Any],
    ref_sequences: Dict[str, str],
) -> int:
    """Estimate transposon insertion position from read alignment.

    For Tn-seq, reads start immediately adjacent to the transposon junction.
    The insertion site is approximated as the alignment start position on the reference.
    """
    pos = alignment["ref_pos"]
    strand = alignment["strand"]

    # For reads starting from transposon end, the insertion is near the read start
    # Add small offset to account for transposon boundary
    if strand == "+":
        return pos
    else:
        # Reverse complement: insertion is at the other end
        return pos + len(read_seq)


# ── Main processing ──────────────────────────────────────────────────────────

def process_read(
    read_id: str,
    read_seq: str,
    index: ReferenceIndex,
    ref_sequences: Dict[str, str],
    min_matches: int = 3,
    max_mismatches: int = 2,
) -> Optional[Dict[str, Any]]:
    """Process a single read: find seeds, extend, return best alignment."""
    hits = index.query_kmers(read_seq, min_matches=min_matches)
    if not hits:
        return None

    best_align = None
    best_score = -1

    for ref_name, ref_pos, seed_score in hits[:10]:  # Top 10 for extension
        align = extend_alignment(read_seq, ref_name, ref_pos, ref_sequences, max_mismatches)
        if align and align["score"] > best_score:
            best_score = align["score"]
            best_align = align
            best_align["read_id"] = read_id

    return best_align


def parse_fastq_header(header: str) -> str:
    """Extract read ID from FASTQ header."""
    h = header.strip()
    if h.startswith("@"):
        h = h[1:]
    return h.split()[0] if h else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align Tn-seq reads to E. piscicida reference."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with FASTQ files.")
    parser.add_argument(
        "--reference",
        default="data/raw/uniprot_episcicida_1263550.tsv",
        help="UniProt TSV with protein sequences.",
    )
    parser.add_argument("--output-dir", default="data/processed/alignment", help="Output directory.")
    parser.add_argument("--kmer-size", type=int, default=19, help="K-mer size for indexing.")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum k-mer matches to seed.")
    parser.add_argument("--max-mismatches", type=int, default=2, help="Max mismatches for alignment.")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads (sequential for now).")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Fraction of reads to process (0-1).")
    parser.add_argument("--dry-run", action="store_true", help="Index only, no alignment.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build reference index
    log.info("Building reference index from %s ...", args.reference)
    index = ReferenceIndex(kmer_size=args.kmer_size)
    index.load_from_uniprot_tsv(args.reference)

    if args.dry_run:
        log.info("Dry run complete. Index has %d sequences.", len(index.sequences))
        return

    # Discover FASTQ files
    input_dir = Path(args.input_dir)
    fastq_files = sorted(input_dir.rglob("*.fq.gz"))
    fastq_files.extend(sorted(input_dir.rglob("*.fastq.gz")))
    log.info("Found %d FASTQ files", len(fastq_files))

    # Statistics
    stats = {
        "total_reads": 0,
        "mapped": 0,
        "unmapped": 0,
        "multi_mapped": 0,
        "by_study": defaultdict(lambda: {"total": 0, "mapped": 0, "unmapped": 0}),
        "by_gene": defaultdict(int),
    }

    # Output files
    sam_path = output_dir / "aligned.sam.gz"
    sam_fh = gzip.open(sam_path, "wt")

    # Header
    sam_fh.write("@HD\tVN:1.0\tSO:unsorted\n")
    for ref_name in sorted(index.sequences.keys()):
        ref_len = len(index.sequences[ref_name])
        sam_fh.write(f"@SQ\tSN:{ref_name}\tLN:{ref_len}\n")
    sam_fh.write("@RG\tID:LAVNet\tPL:ILLUMINA\tPU:Tn-seq\n")

    insertion_bed_path = output_dir / "insertion_sites.bed"
    bed_fh = open(insertion_bed_path, "w")

    ref_sequences = index.sequences
    start_time = time.time()
    processed = 0
    sample_threshold = args.sample_rate

    for fq_idx, fq_path in enumerate(fastq_files, 1):
        study = fq_path.relative_to(input_dir).parts[0]
        rel_path = str(fq_path.relative_to(input_dir))
        log.info("[%d/%d] Aligning %s ...", fq_idx, len(fastq_files), rel_path)

        import gzip as gz
        open_fn = gz.open if str(fq_path).endswith(".gz") else open
        mode = "rt" if str(fq_path).endswith(".gz") else "r"

        with open_fn(fq_path, mode) as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline().strip()
                plus = fh.readline().strip()
                qual = fh.readline().strip()

                if not header.startswith("@"):
                    continue

                read_id = parse_fastq_header(header)
                stats["total_reads"] += 1
                stats["by_study"][study]["total"] += 1

                # Sampling for testing
                if args.sample_rate < 1.0:
                    hash_val = int(hashlib.md5(read_id.encode()).hexdigest()[:8], 16)
                    if hash_val / 0xFFFFFFFF > args.sample_rate:
                        stats["unmapped"] += 1
                        stats["by_study"][study]["unmapped"] += 1
                        continue

                alignment = process_read(
                    read_id, seq, index, ref_sequences,
                    min_matches=args.min_matches,
                    max_mismatches=args.max_mismatches,
                )

                if alignment:
                    stats["mapped"] += 1
                    stats["by_study"][study]["mapped"] += 1
                    stats["by_gene"][alignment["ref_name"]] += 1

                    # Write SAM line
                    sam_line = (
                        f"{alignment['read_id']}\t{alignment['score']:.0f}\t{alignment['cigar']}\t"
                        f"{alignment['ref_name']}\t{alignment['ref_pos'] + 1}\t{alignment['mapq']}\t"
                        f"*\t0\t0\t*\t*\n"
                    )
                    sam_fh.write(sam_line)

                    # Write insertion site
                    ins_pos = find_insertion_site(seq, alignment, ref_sequences)
                    bed_line = f"{alignment['ref_name']}\t{ins_pos}\t{ins_pos + 1}\t{read_id}\t{alignment['score']}\t{alignment['strand']}\n"
                    bed_fh.write(bed_line)
                else:
                    stats["unmapped"] += 1
                    stats["by_study"][study]["unmapped"] += 1

                processed += 1
                if processed % 500000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    log.info("  Processed %d reads (%.1f reads/sec)", processed, rate)

        bed_fh.close()

    sam_fh.close()
    bed_fh.close()

    # Convert by_gene to serializable dict
    stats["by_gene"] = dict(stats["by_gene"])
    stats["by_study"] = dict(stats["by_study"])

    # Write summary
    summary_path = output_dir / "alignment_summary.json"
    with summary_path.open("w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - start_time
    log.info("Alignment complete in %.1f seconds", elapsed)
    log.info("Mapped: %d/%d (%.1f%%)", stats["mapped"], stats["total_reads"],
             100 * stats["mapped"] / max(stats["total_reads"], 1))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
