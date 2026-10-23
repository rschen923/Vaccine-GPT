"""Compute gene-level Tn-seq fitness scores from aligned read counts.

Takes alignment output (SAM/BED) and produces per-gene fitness matrices
ready for downstream ML training.

Fitness score = log2(essential_condition_count / nonessential_condition_count + pseudocount)

Outputs:
  - Per-study gene count matrix (genes × conditions)
  - Composite fitness score per gene
  - Training-ready JSONL with gene features + labels
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

PSEUDOCOUNT = 1.0
MIN_READS_PER_GENE = 5          # minimum coverage to trust fitness estimate
LOG2_BASE = 2


# ── SAM parsing ──────────────────────────────────────────────────────────────

def parse_sam_gz(sam_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse SAM file and return per-gene read counts with condition metadata.

    Returns: {gene_id: {"total": int, "conditions": {condition_name: count}, ...}}
    """
    gene_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "conditions": defaultdict(int)})

    if sam_path.endswith(".gz"):
        fh = gzip.open(sam_path, "rt")
    else:
        fh = open(sam_path, "r")

    try:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("@"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            # Skip unmapped reads (flag & 4 != 0)
            try:
                flag = int(parts[1])
            except ValueError:
                continue
            if flag & 4:
                continue

            ref_name = parts[2]
            ref_pos = int(parts[3]) - 1  # 0-based

            # Extract condition from read ID pattern
            # Expected patterns: SRRxxxxx_*, or study_run_*
            read_id = parts[0]
            condition = _extract_condition(read_id)

            gene_data[ref_name]["total"] += 1
            gene_data[ref_name]["conditions"][condition] += 1

    finally:
        fh.close()

    return dict(gene_data)


def _extract_condition(read_id: str) -> str:
    """Extract experimental condition from read ID.

    Handles common naming patterns:
      - SRR12345678_study_PRJNA390582 → PRJNA390582
      - run_12345_in vivo → in_vivo
      - Default: 'unknown_condition'
    """
    # Try to find study accession
    match = re.search(r"(PRJN[A-Z]*\d+)", read_id, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Try to find condition keyword
    condition_keywords = [
        "in_vivo", "in_vitro", "control", "treatment",
        "knockout", "wildtype", "mutant", "stress",
        "challenge", "mock", "timepoint",
    ]
    read_lower = read_id.lower()
    for kw in condition_keywords:
        if kw in read_lower:
            return kw.replace("-", "_").replace(" ", "_")

    return "unknown_condition"


# ── BED parsing ──────────────────────────────────────────────────────────────

def parse_bed_gz(bed_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse insertion site BED file.

    Returns: {gene_id: {"insertions": int, "positions": set, "strands": Counter}}
    """
    gene_insertions: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"insertions": 0, "positions": set(), "strands": defaultdict(int)}
    )

    open_fn = gzip.open if bed_path.endswith(".gz") else open
    mode = "rt" if bed_path.endswith(".gz") else "r"

    with open_fn(bed_path, mode) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            gene_id = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            strand = parts[5]

            gene_insertions[gene_id]["insertions"] += 1
            gene_insertions[gene_id]["positions"].add(start)
            gene_insertions[gene_id]["strands"][strand] += 1

    return dict(gene_insertions)


# ── Fitness computation ──────────────────────────────────────────────────────

def compute_fitness_scores(
    gene_counts: Dict[str, Dict[str, Any]],
    gene_insertions: Dict[str, Dict[str, Any]],
    proteome_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute fitness scores for all genes across conditions.

    Args:
        gene_counts: Per-gene read counts by condition from SAM parsing.
        gene_insertions: Per-gene insertion site data from BED parsing.
        proteome_df: Optional proteome DataFrame for annotation merging.

    Returns:
        DataFrame with columns:
          gene_id, total_reads, conditions, fitness_score, essentiality,
          coverage_quality, insertion_density
    """
    records = []

    all_conditions = set()
    for gene_data in gene_counts.values():
        all_conditions.update(gene_data["conditions"].keys())
    all_conditions.discard("unknown_condition")
    all_conditions = sorted(all_conditions)

    for gene_id, data in gene_counts.items():
        total_reads = data["total"]
        conditions = data["conditions"]

        # Compute fitness score: log2 ratio of essential vs non-essential
        # Heuristic: conditions with fewer reads → essential (growth defect)
        condition_means = {k: max(v, PSEUDOCOUNT) for k, v in conditions.items()}
        if len(condition_means) >= 2:
            median_reads = np.median(list(condition_means.values()))
            essential_cond = [k for k, v in condition_means.items() if v <= median_reads]
            nonessential_cond = [k for k, v in condition_means.items() if v > median_reads]

            essential_sum = sum(condition_means.get(k, PSEUDOCOUNT) for k in essential_cond)
            nonessential_sum = sum(condition_means.get(k, PSEUDOCOUNT) for k in nonessential_cond)

            fitness = np.log2((essential_sum + PSEUDOCOUNT) / (nonessential_sum + PSEUDOCOUNT))
        else:
            fitness = 0.0  # single condition: neutral

        # Coverage quality
        has_enough_coverage = total_reads >= MIN_READS_PER_GENE
        coverage_quality = "high" if total_reads >= 50 else ("medium" if total_reads >= MIN_READS_PER_GENE else "low")

        # Insertion density (insertions per kb)
        insertion_info = gene_insertions.get(gene_id, {"insertions": 0, "positions": set()})
        n_insertions = insertion_info["insertions"]
        unique_positions = len(insertion_info["positions"])

        # Estimate gene length from proteome if available
        gene_length = 0
        if proteome_df is not None:
            row = proteome_df[proteome_df["Entry"] == gene_id]
            if not row.empty:
                gene_length = int(row.iloc[0].get("Length", 0))

        insertion_density = (n_insertions / max(gene_length / 1000, 1)) if gene_length > 0 else 0.0

        # Essentiality call
        if total_reads < MIN_READS_PER_GENE:
            essentiality = "untested"
        elif fitness < -1.0:
            essentiality = "essential"
        elif fitness > 1.0:
            essentiality = "dispensable"
        else:
            essentiality = "conditional"

        record = {
            "gene_id": gene_id,
            "total_reads": total_reads,
            "n_conditions": len([c for c in condition_means if condition_means[c] > 0]),
            "fitness_score": round(float(fitness), 4),
            "coverage_quality": coverage_quality,
            "has_enough_coverage": has_enough_coverage,
            "n_insertions": n_insertions,
            "unique_insertion_positions": unique_positions,
            "insertion_density_per_kb": round(insertion_density, 4),
            "estimated_gene_length": gene_length,
            "essentiality_call": essentiality,
            "condition_breakdown": {k: int(v) for k, v in sorted(condition_means.items())},
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


# ── Condition clustering ─────────────────────────────────────────────────────

def cluster_conditions(
    gene_counts: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Cluster conditions by correlation of read-count profiles.

    Groups conditions that show similar global coverage patterns,
    which may indicate biological replicates or shared stress responses.
    """
    all_conditions = set()
    for data in gene_counts.values():
        all_conditions.update(data["conditions"].keys())
    all_conditions.discard("unknown_condition")
    all_conditions = sorted(all_conditions)

    if len(all_conditions) < 2:
        return {"single_cluster": list(all_conditions)}

    # Build condition × gene matrix
    genes = list(gene_counts.keys())
    matrix = np.zeros((len(genes), len(all_conditions)))

    for i, gene in enumerate(genes):
        for j, cond in enumerate(all_conditions):
            matrix[i, j] = np.log2(gene_counts[gene]["conditions"].get(cond, PSEUDOCOUNT) + PSEUDOCOUNT)

    # Compute pairwise Pearson correlation
    corr_matrix = np.corrcoef(matrix.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)

    # Simple agglomerative clustering: group conditions with r > 0.8
    threshold = 0.8
    clusters: Dict[str, List[str]] = {}
    visited = set()
    cluster_idx = 0

    for i in range(len(all_conditions)):
        if i in visited:
            continue
        cluster_name = f"cluster_{cluster_idx}"
        cluster = [all_conditions[i]]
        visited.add(i)
        for j in range(i + 1, len(all_conditions)):
            if j not in visited and corr_matrix[i, j] >= threshold:
                cluster.append(all_conditions[j])
                visited.add(j)
        clusters[cluster_name] = cluster
        cluster_idx += 1

    return clusters


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gene-level Tn-seq fitness scores."
    )
    parser.add_argument(
        "--sam-file",
        required=True,
        help="Path to aligned.sam.gz from align_tnseq.py.",
    )
    parser.add_argument(
        "--bed-file",
        default=None,
        help="Path to insertion_sites.bed from align_tnseq.py.",
    )
    parser.add_argument(
        "--proteome",
        default="data/raw/uniprot_episcicida_1263550.tsv",
        help="UniProt TSV for gene annotations.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/alignment",
        help="Output directory.",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=MIN_READS_PER_GENE,
        help="Minimum reads per gene to include.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load proteome for annotations
    proteome_df = None
    proteome_path = Path(args.proteome)
    if proteome_path.exists():
        proteome_df = pd.read_csv(proteome_path, sep="\t")
        log.info("Loaded proteome: %d genes", len(proteome_df))

    # Parse SAM
    log.info("Parsing SAM file: %s", args.sam_file)
    gene_counts = parse_sam_gz(args.sam_file)
    log.info("Found %d genes with mapped reads", len(gene_counts))

    # Parse BED (optional)
    gene_insertions = {}
    if args.bed_file and Path(args.bed_file).exists():
        log.info("Parsing BED file: %s", args.bed_file)
        gene_insertions = parse_bed_gz(args.bed_file)
        log.info("Found %d genes with insertion sites", len(gene_insertions))

    # Compute fitness scores
    log.info("Computing fitness scores ...")
    fitness_df = compute_fitness_scores(gene_counts, gene_insertions, proteome_df)

    # Filter by minimum coverage
    filtered_df = fitness_df[fitness_df.total_reads >= args.min_reads].copy()
    log.info(
        "Filtered: %d/%d genes have >= %d reads",
        len(filtered_df), len(fitness_df), args.min_reads,
    )

    # Cluster conditions
    condition_clusters = cluster_conditions(gene_counts)
    log.info("Condition clusters: %d groups", len(condition_clusters))

    # Save outputs
    # 1. Full fitness table
    fitness_csv = output_dir / "gene_fitness_scores.csv"
    filtered_df.to_csv(fitness_csv, index=False)
    log.info("Fitness scores saved to %s (%d genes)", fitness_csv, len(filtered_df))

    # 2. Summary statistics
    summary = {
        "total_genes": len(fitness_df),
        "genes_with_coverage": len(filtered_df),
        "essential_genes": int((filtered_df.essentiality_call == "essential").sum()),
        "dispensable_genes": int((filtered_df.essentiality_call == "dispensable").sum()),
        "conditional_genes": int((filtered_df.essentiality_call == "conditional").sum()),
        "untested_genes": int((filtered_df.essentiality_call == "untested").sum()),
        "mean_fitness_score": round(float(filtered_df.fitness_score.mean()), 4) if len(filtered_df) > 0 else 0,
        "median_fitness_score": round(float(filtered_df.fitness_score.median()), 4) if len(filtered_df) > 0 else 0,
        "mean_coverage": round(float(filtered_df.total_reads.mean()), 1) if len(filtered_df) > 0 else 0,
        "condition_clusters": condition_clusters,
    }

    summary_path = output_dir / "fitness_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary saved to %s", summary_path)

    # 3. Gene count matrix (genes × conditions)
    if len(filtered_df) > 0:
        # Pivot condition breakdown into matrix
        condition_cols = []
        for _, row in filtered_df.iterrows():
            condition_cols.extend(row["condition_breakdown"].keys())
        all_conds = sorted(set(condition_cols))

        matrix_records = []
        for _, row in filtered_df.iterrows():
            record = {"gene_id": row["gene_id"]}
            for cond in all_conds:
                record[cond] = row["condition_breakdown"].get(cond, 0)
            record["fitness_score"] = row["fitness_score"]
            record["essentiality_call"] = row["essentiality_call"]
            matrix_records.append(record)

        count_matrix = pd.DataFrame(matrix_records)
        count_matrix_path = output_dir / "gene_condition_matrix.csv"
        count_matrix.to_csv(count_matrix_path, index=False)
        log.info("Count matrix saved to %s", count_matrix_path)

    # 4. Training-ready JSONL (merge with existing labels)
    label_path = Path("data/processed/labels/gene_labels.jsonl")
    training_records = []
    if label_path.exists():
        with label_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                gene_id = rec.get("gene_id", "")
                fitness_row = filtered_df[filtered_df.gene_id == gene_id]
                if not fitness_row.empty:
                    fr = fitness_row.iloc[0]
                    rec["tnseq_fitness_score"] = float(fr["fitness_score"])
                    rec["tnseq_coverage"] = int(fr["total_reads"])
                    rec["tnseq_essentiality"] = fr["essentiality_call"]
                    rec["tnseq_coverage_quality"] = fr["coverage_quality"]
                    training_records.append(rec)

    training_path = output_dir / "training_ready.jsonl"
    with training_path.open("w", encoding="utf-8") as f:
        for rec in training_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Training-ready records saved to %s (%d genes)", training_path, len(training_records))

    print(json.dumps({
        "genes_total": len(fitness_df),
        "genes_filtered": len(filtered_df),
        "essential": summary["essential_genes"],
        "dispensable": summary["dispensable_genes"],
        "conditional": summary["conditional_genes"],
        "mean_fitness": summary["mean_fitness_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
