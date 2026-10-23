"""Orchestrate the full Tn-seq processing pipeline: clean → align → fitness.

Usage:
  python scripts/run_tnseq_pipeline.py --input-dir data/raw/fastq
  python scripts/run_tnseq_pipeline.py --input-dir data/raw/fastq --sample-rate 0.1
  python scripts/run_tnseq_pipeline.py --input-dir data/raw/fastq --skip-alignment
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name: str, args: list[str], label: str) -> bool:
    """Run a pipeline script and return True on success."""
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)] + args
    log.info("=== %s ===", label)
    log.info("Command: %s", " ".join(cmd))
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=86400,  # 24h max per step
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            log.info("✓ %s completed in %.1fs", label, elapsed)
            # Print last few lines of output
            for line in result.stdout.strip().split("\n")[-5:]:
                log.info("  %s", line)
            return True
        else:
            log.error("✗ %s failed (exit %d) in %.1fs", label, result.returncode, elapsed)
            log.error("STDERR: %s", result.stderr[-500:] if result.stderr else "(none)")
            return False
    except subprocess.TimeoutExpired:
        log.error("✗ %s timed out after 24h", label)
        return False
    except Exception as exc:
        log.error("✗ %s error: %s", label, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Tn-seq processing pipeline.")
    parser.add_argument("--input-dir", required=True, help="Directory with FASTQ files.")
    parser.add_argument("--reference", default="data/raw/uniprot_episcicida_1263550.tsv", help="Proteome reference.")
    parser.add_argument("--output-dir", default="data/processed/alignment", help="Output directory.")
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning, use raw FASTQ.")
    parser.add_argument("--skip-alignment", action="store_true", help="Skip alignment step.")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Process fraction of reads (0-1).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only.")
    parser.add_argument("--resume-from", default=None, help="JSON file with previous step results.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pipeline root: %s", root)
    log.info("Input directory: %s", input_dir)
    log.info("Output directory: %s", output_dir)

    # Validate inputs
    fastq_files = list(input_dir.rglob("*.fq.gz")) + list(input_dir.rglob("*.fastq.gz"))
    if not fastq_files:
        log.error("No FASTQ files found in %s", input_dir)
        sys.exit(1)
    log.info("Found %d FASTQ files", len(fastq_files))

    ref_path = root / args.reference
    if not ref_path.exists():
        log.error("Reference not found: %s", ref_path)
        sys.exit(1)

    # Load resume state if available
    resume_state = {}
    if args.resume_from:
        resume_path = root / args.resume_from
        if resume_path.exists():
            resume_state = json.loads(resume_path.read_text())
            log.info("Resumed from %s", resume_path)

    pipeline_results = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dir": str(input_dir),
        "reference": str(ref_path),
        "steps": {},
    }

    # Step 1: Clean & deduplicate FASTQ
    if not args.skip_clean and "clean" not in resume_state.get("steps", {}):
        clean_start = time.time()
        success = run_script(
            "clean_fastq.py",
            [
                "--input-dir", str(input_dir),
                "--output-manifest", str(output_dir / "clean_manifest.jsonl"),
                "--output-cleaned",
                "--report", str(output_dir / "clean_report.json"),
                f"--sample-rate", str(args.sample_rate),
            ],
            "Step 1: Clean & deduplicate FASTQ",
        )
        pipeline_results["steps"]["clean"] = {"success": success, "elapsed": time.time() - clean_start}
        if not success:
            log.error("Cleaning failed. Use --skip-clean to proceed with raw data.")
            # Write partial results
            with open(output_dir / "pipeline_status.json", "w") as f:
                json.dump(pipeline_results, f, indent=2)
            sys.exit(1)

        # Update input to cleaned directory for next steps
        cleaned_input = input_dir / "cleaned"
        if cleaned_input.exists():
            log.info("Switching to cleaned input: %s", cleaned_input)
            input_dir = cleaned_input

    # Step 2: Align reads
    if not args.skip_alignment and "align" not in resume_state.get("steps", {}):
        align_start = time.time()
        align_args = [
            "--input-dir", str(input_dir),
            "--reference", str(ref_path),
            "--output-dir", str(output_dir),
            f"--sample-rate", str(args.sample_rate),
        ]
        if args.dry_run:
            align_args.append("--dry-run")

        success = run_script(
            "align_tnseq.py",
            align_args,
            "Step 2: Align reads to reference",
        )
        pipeline_results["steps"]["align"] = {"success": success, "elapsed": time.time() - align_start}
        if not success and not args.dry_run:
            log.error("Alignment failed.")
            with open(output_dir / "pipeline_status.json", "w") as f:
                json.dump(pipeline_results, f, indent=2)
            sys.exit(1)

    # Step 3: Compute fitness scores
    if "fitness" not in resume_state.get("steps", {}):
        sam_file = output_dir / "aligned.sam.gz"
        bed_file = output_dir / "insertion_sites.bed"

        if not sam_file.exists():
            log.warning("SAM file not found at %s — skipping fitness computation.", sam_file)
        else:
            fitness_start = time.time()
            fitness_args = [
                "--sam-file", str(sam_file),
                "--proteome", str(ref_path),
                "--output-dir", str(output_dir),
            ]
            if bed_file.exists():
                fitness_args.extend(["--bed-file", str(bed_file)])

            success = run_script(
                "compute_fitness_scores.py",
                fitness_args,
                "Step 3: Compute fitness scores",
            )
            pipeline_results["steps"]["fitness"] = {"success": success, "elapsed": time.time() - fitness_start}

    # Final status
    pipeline_results["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    status_path = output_dir / "pipeline_status.json"
    with status_path.open("w") as f:
        json.dump(pipeline_results, f, indent=2)

    log.info("=== Pipeline complete ===")
    log.info("Status written to %s", status_path)

    # List outputs
    log.info("Outputs:")
    for p in sorted(output_dir.glob("*")):
        size_str = ""
        if p.is_file():
            size_str = f" ({p.stat().st_size / 1e6:.1f} MB)"
        elif p.is_dir():
            size_str = f" ({len(list(p.glob('**/*')))} items)"
        log.info("  %s%s", p.name, size_str)


if __name__ == "__main__":
    main()
