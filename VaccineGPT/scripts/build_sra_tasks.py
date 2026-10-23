from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible SRA preprocessing tasks")
    parser.add_argument("--accessions", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default="data/metadata/sra_processing_tasks.json")
    args = parser.parse_args()
    accessions = [
        line.strip() for line in Path(args.accessions).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    tasks = []
    for accession in accessions:
        workdir = f"data/processed/sra/{accession}"
        tasks.append({
            "accession": accession,
            "steps": [
                {"tool": "fastp", "command": f"fastp -i {accession}_R1.fastq.gz -I {accession}_R2.fastq.gz -o {workdir}/clean_R1.fastq.gz -O {workdir}/clean_R2.fastq.gz"},
                {"tool": "bowtie2", "command": f"bowtie2 -x {args.reference} -1 {workdir}/clean_R1.fastq.gz -2 {workdir}/clean_R2.fastq.gz -S {workdir}/aligned.sam"},
                {"tool": "TRADE-seq/ESSENTIALS", "command": f"trade_seq --input {workdir}/aligned.sam --output {workdir}/gene_effects.tsv"},
            ],
            "expected_output": f"{workdir}/gene_effects.tsv",
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"tasks": tasks, "reference": args.reference}, indent=2), encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
