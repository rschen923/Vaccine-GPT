"""Convert a downloaded VFDB FASTA release into canonical pilot JSONL records."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    sequence_records = []
    labels = []
    current_id = None
    current_parts: list[str] = []
    with gzip.open(args.input, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequence_records.append({"gene_id": current_id, "sequence": "".join(current_parts)})
                header = line[1:].split()[0]
                current_id = re.sub(r"[^A-Za-z0-9_.:-]", "_", header)
                current_parts = []
            else:
                current_parts.append(line)
        if current_id is not None:
            sequence_records.append({"gene_id": current_id, "sequence": "".join(current_parts)})
    if args.max_records:
        sequence_records = sequence_records[: args.max_records]
    for record in sequence_records:
        labels.append({"gene_id": record["gene_id"], "task": "T2", "label": 1.0, "level": "L2",
                       "group_id": record["gene_id"]})
    for path, records in ((args.sequences, sequence_records), (args.labels, labels)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(sequence_records), "label_task": "T2", "source": args.input}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
