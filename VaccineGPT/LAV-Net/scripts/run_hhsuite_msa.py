"""HH-suite MSA generation pipeline for LAV-Net evolutionary features.

Implements end-to-end HH-suite workflow:
  1. Download and index UniRef30/UniClust30 databases
  2. Run hhblits for iterative HMM-HMM search against UniRef30
  3. Generate A3M alignments with coverage-aware filtering
  4. Extract evolutionary features for CMCL input schema

Input: E. piscicida protein sequences (from UniProt proteome)
Output: Per-gene A3M alignments + evolutionary feature tensors

Dependencies: HH-suite tools (hhblits, hhsearch, hhmake), numpy, pandas
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_UNIREF_URL = "https://wwwuser.gwdg.de/~compbiol/uniclust/2022_02/UniRef30_2022_02_hhsuite.tar.gz"
DEFAULT_HHR_CACHE_URL = "https://wwwuser.gwdg.de/~compbiol/uniclust/2022_02/UniRef30_2022_02_hhsuite_dbinfo.tar.gz"
MSA_DEPTH_TARGET = 50           # minimum effective sequence count (Neff)
E_VALUE_CUTOFF = 1e-5          # HHblits e-value cutoff
NUM_ITERATIONS = 3             # HHblits iteration count
KMER_SIZE = 4                  # default for HH-suite (fixed)


# ── Database Management ─────────────────────────────────────────────────────

class HHDatabaseManager:
    """Manage HH-suite database downloads, indexing, and caching."""

    def __init__(self, db_root: str):
        self.db_root = Path(db_root)
        self.db_root.mkdir(parents=True, exist_ok=True)
        self.uniref_path = self.db_root / "UniRef30_2022_02_hhsuite"
        self.hhr_cache_path = self.db_root / "hhr_cache"
        self.hhr_cache_path.mkdir(exist_ok=True)

    def database_exists(self) -> bool:
        """Check if all required HH-suite database files exist."""
        required = [".ci", ".idx", ".pkl", ".pre"]
        for ext in required:
            if not (self.uniref_path.with_suffix(ext)).exists():
                return False
        return True

    def download_and_extract(self, url: str = DEFAULT_UNIREF_URL, timeout_hours: int = 24) -> bool:
        """Download and extract UniRef30 HH-suite database."""
        tarball = self.db_root / "UniRef30_2022_02_hhsuite.tar.gz"

        if self.database_exists():
            log.info("Database already exists at %s", self.uniref_path)
            return True

        if not tarball.exists():
            log.info("Downloading UniRef30 database from %s ...", url)
            log.info("Size: ~61 GB — this will take significant time")
            cmd = [
                "curl.exe", "-L", "--fail", "--retry", "10",
                "--retry-delay", "30", "--progress-bar",
                "-o", str(tarball), url,
            ]
            try:
                result = subprocess.run(
                    cmd, check=True, timeout=timeout_hours * 3600
                )
            except subprocess.TimeoutExpired:
                log.error("Download timed out after %d hours", timeout_hours)
                return False
            except Exception as exc:
                log.error("Download failed: %s", exc)
                return False

        log.info("Extracting %s ...", tarball)
        try:
            subprocess.run(
                ["tar", "-xzf", str(tarball), "-C", str(self.db_root)],
                check=True, timeout=7200,
            )
        except Exception as exc:
            log.error("Extraction failed: %s", exc)
            return False

        # Verify extraction
        if self.database_exists():
            log.info("Database extracted successfully")
            return True
        else:
            log.error("Database verification failed after extraction")
            return False

    def build_custom_database(self, fasta_path: str, db_name: str = "episcicida_custom") -> Path:
        """Build a custom HH-suite database from protein FASTA."""
        db_path = self.db_root / db_name
        a3m_path = self.db_root / f"{db_name}.a3m"

        # Convert FASTA to A3M (simple identity mapping)
        fasta_file = Path(fasta_path)
        if not fasta_file.exists():
            log.error("FASTA file not found: %s", fasta_path)
            return Path()

        # Use hhfilter/hhmake to convert
        log.info("Building custom database from %s ...", fasta_path)
        cmd = [
            "hhfilter", "-i", str(fasta_file),
            "-o", str(a3m_path), "-id", "100", "-cov", "50",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except FileNotFoundError:
            log.warning("hhfilter not found; copying FASTA as A3M fallback")
            shutil.copy2(fasta_file, a3m_path)
        except Exception as exc:
            log.error("Custom DB build failed: %s", exc)
            return Path()

        # Build HH-suite index
        cmd = ["hhmake", "-i", str(a3m_path), "-o", str(db_path)]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except FileNotFoundError:
            log.warning("hhmake not found; skipping index build")
        except Exception as exc:
            log.warning("Index build failed (non-fatal): %s", exc)

        return db_path

    def get_db_info(self) -> Dict[str, Any]:
        """Get database statistics."""
        info = {
            "path": str(self.uniref_path),
            "exists": self.database_exists(),
            "size_bytes": 0,
        }
        if self.uniref_path.exists():
            info["size_bytes"] = sum(
                f.stat().st_size for f in self.uniref_path.parent.glob(
                    self.uniref_path.name + "*"
                )
            )
        return info


# ── HHblits Execution ────────────────────────────────────────────────────────

class HHblitsRunner:
    """Execute HHblits for evolutionary profile search."""

    def __init__(
        self,
        uniref_db: str,
        hhr_cache_dir: str,
        num_iterations: int = NUM_ITERATIONS,
        e_value: float = E_VALUE_CUTOFF,
    ):
        self.uniref_db = uniref_db
        self.hhr_cache_dir = hhr_cache_dir
        self.num_iterations = num_iterations
        self.e_value = e_value

    def run(
        self,
        query_a3m: str,
        output_a3m: str,
        output_hhr: Optional[str] = None,
        output_hipr: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run HHblits and return statistics."""
        cmd = [
            "hhblits",
            "-i", query_a3m,
            "-d", self.uniref_db,
            "-oa3m", output_a3m,
            "-n", str(self.num_iterations),
            "-e", str(self.e_value),
            "-cpu", "8",
            "-all",
        ]

        if output_hhr:
            cmd.extend(["-ohhr", output_hhr])
        if output_hipr:
            cmd.extend(["-ohipr", output_hipr])

        # Add HHR cache for speedup
        if os.path.isdir(self.hhr_cache_dir):
            cmd.extend(["-HHR_CACHE", self.hhr_cache_dir])

        log.info("Running HHblits: %s", " ".join(cmd))
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=86400
            )
            elapsed = time.time() - start_time

            stats = {
                "success": result.returncode == 0,
                "elapsed_seconds": round(elapsed, 2),
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else "",
            }

            # Parse key metrics from stdout
            for line in result.stdout.split("\n"):
                if "number of aligned sequences with" in line:
                    stats["msa_depth"] = int(line.split(":")[-1].strip())
                elif "best total probability" in line:
                    try:
                        stats["best_prob"] = float(line.split(":")[-1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass

            log.info("HHblits completed in %.1fs: %s", elapsed,
                     "SUCCESS" if stats["success"] else "FAILED")
            return stats

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "timeout after 24h"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


# ── Evolutionary Feature Extraction ──────────────────────────────────────────

class EvolutionaryFeatureExtractor:
    """Extract evolutionary features from A3M alignments for CMCL input."""

    def __init__(self, conservation_bins: int = 32, dipeptide_dim: int = 32):
        self.conservation_bins = conservation_bins
        self.dipeptide_dim = dipeptide_dim

    def extract_from_a3m(
        self,
        a3m_path: str,
        query_length: int,
    ) -> Dict[str, Any]:
        """Extract full evolutionary feature set from A3M alignment.

        Returns dict with:
          - conservation_spectrum: np.array [32]
          - dipeptide_composition: np.array [32]
          - msa_depth: int
          - pairwise_identity_stats: dict
          - gap_distribution: np.array [query_length]
        """
        alignments = self._parse_a3m(a3m_path)

        if not alignments:
            return self._empty_features(query_length)

        msa_depth = len(alignments)
        consensus_seq = self._compute_consensus(alignments)

        features = {
            "conservation_spectrum": self._compute_conservation_spectrum(
                alignments, query_length
            ),
            "dipeptide_composition": self._compute_dipeptide_composition(
                consensus_seq[:query_length]
            ),
            "msa_depth": msa_depth,
            "effective_depth": self._estimate_neff(alignments),
            "pairwise_identity_mean": self._mean_pairwise_identity(alignments),
            "gap_distribution": self._compute_gap_distribution(
                alignments, query_length
            ),
            "coverage": self._alignment_coverage(alignments, query_length),
        }

        return features

    def _parse_a3m(self, a3m_path: str) -> List[str]:
        """Parse A3M file, returning aligned sequences (uppercase only)."""
        sequences = []
        open_fn = gzip.open if a3m_path.endswith(".gz") else open
        mode = "rt" if a3m_path.endswith(".gz") else "r"

        with open_fn(a3m_path, mode) as fh:
            current_seq = []
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                    current_seq = []
                elif line and not line.startswith("#"):
                    current_seq.append(line.upper())

            if current_seq:
                sequences.append("".join(current_seq))

        # Remove query sequence itself (first entry usually)
        return sequences[1:] if len(sequences) > 1 else sequences

    def _compute_conservation_spectrum(
        self,
        alignments: List[str],
        query_length: int,
    ) -> List[float]:
        """Compute 32-bin conservation spectrum per position.

        For each position, compute fraction of non-gap, non-N residues,
        then bin into 32 equal-width bins across the sequence.
        """
        if not alignments:
            return [0.0] * self.conservation_bins

        # Stack alignments
        positions = []
        for pos in range(query_length):
            residues = []
            for seq in alignments:
                if pos < len(seq):
                    r = seq[pos]
                    if r not in ("-.", "N"):
                        residues.append(r)
            if residues:
                conservation = len(residues) / len(alignments)
            else:
                conservation = 0.0
            positions.append(conservation)

        # Bin into 32 bins
        if not positions:
            return [0.0] * self.conservation_bins

        binned = [0.0] * self.conservation_bins
        bin_width = max(len(positions) / self.conservation_bins, 1)

        for i, val in enumerate(positions):
            bin_idx = min(int(i / bin_width), self.conservation_bins - 1)
            binned[bin_idx] += val

        # Normalize
        n_positions = len(positions)
        if n_positions > 0:
            binned = [v / n_positions for v in binned]

        return binned

    def _compute_dipeptide_composition(self, seq: str) -> List[float]:
        """Compute 32-dim dipeptide composition vector.

        Uses reduced amino acid alphabet (6 groups) → 6×6=36 dipeptides,
        truncated/pooled to 32 dimensions.
        """
        # Six-group AA reduction (based on properties)
        aa_groups = {
            'A': 0, 'V': 0, 'L': 0, 'I': 0,           # aliphatic
            'F': 1, 'Y': 1, 'W': 1,                      # aromatic
            'S': 2, 'T': 2, 'C': 2, 'N': 2, 'Q': 2, 'K': 2,  # polar
            'D': 3, 'E': 3,                               # acidic
            'R': 3, 'H': 3,                               # basic (merged with acidic for 32)
            'P': 4,                                       # special
            'G': 5,                                       # special
        }

        dipeptide_counts = [0] * 36  # 6×6
        seq_clean = "".join(c for c in seq if c in aa_groups)

        for i in range(len(seq_clean) - 1):
            g1 = aa_groups.get(seq_clean[i], 0)
            g2 = aa_groups.get(seq_clean[i + 1], 0)
            idx = g1 * 6 + g2
            if idx < 36:
                dipeptide_counts[idx] += 1

        # Normalize
        total = sum(dipeptide_counts)
        if total > 0:
            dipeptide_counts = [c / total for c in dipeptide_counts]

        # Pool last 4 into first 4 to get 32 dims
        pooled = dipeptide_counts[:32].copy()
        for i in range(4):
            pooled[i] += dipeptide_counts[32 + i] / 4

        return pooled

    def _estimate_neff(self, alignments: List[str]) -> float:
        """Estimate effective number of sequences (Neff)."""
        if not alignments:
            return 0.0

        # Simple heuristic: count sequences with >60% identity to consensus
        consensus = self._compute_consensus(alignments)
        neff = 0

        for seq in alignments:
            identity = self._sequence_identity(seq, consensus)
            if identity >= 0.6:
                neff += 1

        return neff

    def _compute_consensus(self, alignments: List[str]) -> str:
        """Compute majority-rule consensus from alignments."""
        if not alignments:
            return ""

        min_len = min(len(s) for s in alignments)
        consensus = []

        for pos in range(min_len):
            residues = [seq[pos] for seq in alignments if pos < len(seq) and seq[pos] not in "-."]
            if residues:
                from collections import Counter
                consensus.append(Counter(residues).most_common(1)[0][0])
            else:
                consensus.append("X")

        return "".join(consensus)

    def _mean_pairwise_identity(self, alignments: List[str]) -> float:
        """Compute mean pairwise sequence identity."""
        if len(alignments) < 2:
            return 1.0

        identities = []
        sample = alignments[:min(100, len(alignments))]  # subsample for speed

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                ident = self._sequence_identity(sample[i], sample[j])
                identities.append(ident)

        return sum(identities) / len(identities) if identities else 0.0

    def _sequence_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity between two aligned sequences."""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0

        matches = sum(
            1 for a, b in zip(seq1[:min_len], seq2[:min_len])
            if a == b and a not in "-."
        )
        return matches / min_len

    def _compute_gap_distribution(
        self,
        alignments: List[str],
        query_length: int,
    ) -> List[float]:
        """Compute gap fraction per position, normalized to query_length."""
        if not alignments:
            return [0.0] * query_length

        gaps = [0.0] * query_length
        for seq in alignments:
            for pos in range(min(len(seq), query_length)):
                if seq[pos] in "-.":
                    gaps[pos] += 1

        # Normalize by number of alignments
        n_seqs = len(alignments)
        return [g / n_seqs for g in gaps]

    def _alignment_coverage(
        self,
        alignments: List[str],
        query_length: int,
    ) -> float:
        """Compute fraction of query positions covered by ≥1 alignment."""
        if not alignments:
            return 0.0

        covered = set()
        for seq in alignments:
            for pos in range(min(len(seq), query_length)):
                if seq[pos] not in "-.":
                    covered.add(pos)

        return len(covered) / query_length

    def _empty_features(self, query_length: int) -> Dict[str, Any]:
        """Return zero-filled features when no MSA is available."""
        return {
            "conservation_spectrum": [0.0] * self.conservation_bins,
            "dipeptide_composition": [0.0] * self.dipeptide_dim,
            "msa_depth": 0,
            "effective_depth": 0,
            "pairwise_identity_mean": 0.0,
            "gap_distribution": [0.0] * query_length,
            "coverage": 0.0,
        }


# ── Main Pipeline ────────────────────────────────────────────────────────────

def load_protein_sequences(fasta_path: str) -> Dict[str, str]:
    """Load protein sequences from FASTA or UniProt TSV."""
    sequences = {}
    path = Path(fasta_path)

    if not path.exists():
        log.error("File not found: %s", fasta_path)
        return sequences

    if fasta_path.endswith(".tsv"):
        import pandas as pd
        df = pd.read_csv(fasta_path, sep="\t")
        seq_col = "Sequence" if "Sequence" in df.columns else df.columns[-1]
        id_col = "Entry" if "Entry" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            sequences[str(row[id_col])] = str(row[seq_col]).upper()
    else:
        # FASTA format
        current_id = ""
        current_seq = []
        open_fn = gzip.open if fasta_path.endswith(".gz") else open
        mode = "rt" if fasta_path.endswith(".gz") else "r"

        with open_fn(fasta_path, mode) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(line.upper())

            if current_id:
                sequences[current_id] = "".join(current_seq)

    return sequences


def run_msa_pipeline(
    proteome_path: str,
    output_dir: str,
    db_root: str,
    uniref_url: str = DEFAULT_UNIREF_URL,
    skip_download: bool = False,
    sample_genes: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the complete MSA + feature extraction pipeline.

    Args:
        proteome_path: Path to E. piscicida protein sequences (TSV or FASTA)
        output_dir: Directory for output alignments and features
        db_root: Directory for HH-suite databases
        uniref_url: URL for UniRef30 download
        skip_download: Skip download if database exists
        sample_genes: Process only N genes (for testing)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load sequences
    log.info("Loading protein sequences from %s ...", proteome_path)
    sequences = load_protein_sequences(proteome_path)
    log.info("Loaded %d protein sequences", len(sequences))

    if sample_genes:
        sequences = dict(list(sequences.items())[:sample_genes])
        log.info("Sampling %d genes for processing", len(sequences))

    if not sequences:
        log.error("No sequences loaded. Exiting.")
        return {"error": "no sequences"}

    # Step 2: Initialize database manager
    db_manager = HHDatabaseManager(db_root)

    # Step 3: Ensure UniRef30 database
    if not skip_download:
        if not db_manager.database_exists():
            success = db_manager.download_and_extract(uniref_url)
            if not success:
                log.error("Database download/extraction failed")
                return {"error": "database_setup_failed"}
        else:
            log.info("Using existing UniRef30 database")
    else:
        if not db_manager.database_exists():
            log.error("Database not found and download skipped")
            return {"error": "database_missing"}

    # Step 4: Build custom database
    custom_fasta = output_path / "episcicida_proteins.fasta"
    with custom_fasta.open("w") as fh:
        for gene_id, seq in sequences.items():
            fh.write(f">{gene_id}\n{seq}\n")

    custom_db = db_manager.build_custom_database(str(custom_fasta))
    if not custom_db.exists():
        log.error("Failed to build custom database")
        return {"error": "custom_db_build_failed"}

    # Step 5: Run HHblits for each gene
    runner = HHblitsRunner(
        uniref_db=str(db_manager.uniref_path),
        hhr_cache_dir=str(db_manager.hhr_cache_path),
        num_iterations=3,
        e_value=1e-5,
    )

    extractor = EvolutionaryFeatureExtractor()
    results = []
    gene_features_path = output_path / "evolutionary_features.jsonl"

    for idx, (gene_id, seq) in enumerate(sequences.items(), 1):
        log.info("[%d/%d] Processing %s (len=%d) ...",
                 idx, len(sequences), gene_id, len(seq))

        # Write per-gene query A3M
        query_a3m = output_path / f"{gene_id}.query.a3m"
        with query_a3m.open("w") as fh:
            fh.write(f">{gene_id}\n{seq}\n")

        # Run HHblits
        output_a3m = output_path / f"{gene_id}.msa.a3m"
        hhr_result = runner.run(
            query_a3m=str(query_a3m),
            output_a3m=str(output_a3m),
        )

        # Extract features
        features = extractor.extract_from_a3m(str(output_a3m), len(seq))
        features["gene_id"] = gene_id
        features["seq_length"] = len(seq)
        features["hhblits_success"] = hhr_result.get("success", False)
        features["msa_depth"] = features.get("msa_depth", 0)

        results.append(features)

        # Write to JSONL
        with gene_features_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                k: (v.tolist() if isinstance(v, list) and v and isinstance(v[0], float) else v)
                for k, v in features.items()
            }, ensure_ascii=False) + "\n")

        if idx % 50 == 0:
            log.info("Processed %d genes so far", idx)

    # Summary
    successful = sum(1 for r in results if r.get("hhblits_success"))
    avg_msa_depth = np.mean([r.get("msa_depth", 0) for r in results]) if results else 0

    summary = {
        "total_genes": len(sequences),
        "successful_msas": successful,
        "average_msa_depth": round(float(avg_msa_depth), 2),
        "output_dir": str(output_path),
        "features_file": str(gene_features_path),
    }

    log.info("Pipeline complete: %s", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HH-suite MSA pipeline for LAV-Net evolutionary features."
    )
    parser.add_argument(
        "--proteome",
        default="data/raw/uniprot_episcicida_1263550.tsv",
        help="E. piscicida protein sequences (TSV or FASTA).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/evolutionary_features",
        help="Output directory for alignments and features.",
    )
    parser.add_argument(
        "--db-root",
        default="data/raw/msa_db",
        help="Root directory for HH-suite databases.",
    )
    parser.add_argument(
        "--uniref-url",
        default=DEFAULT_UNIREF_URL,
        help="URL for UniRef30 HH-suite database.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if database exists.",
    )
    parser.add_argument(
        "--sample-genes",
        type=int,
        default=None,
        help="Process only N genes (for testing).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    proteome = root / args.proteome
    output_dir = root / args.output_dir
    db_root = root / args.db_root

    result = run_msa_pipeline(
        proteome_path=str(proteome),
        output_dir=str(output_dir),
        db_root=str(db_root),
        uniref_url=args.uniref_url,
        skip_download=args.skip_download,
        sample_genes=args.sample_genes,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import numpy as np
    main()
