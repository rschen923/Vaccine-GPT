"""Run the authoritative five-layer model on canonical public JSONL data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaccinegpt.real_data import load_real_batch
from vaccinegpt.spec_training import train_contract_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--ppi")
    parser.add_argument("--max-records", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", default="checkpoints/real_five_layer")
    args = parser.parse_args()
    batch, provenance = load_real_batch(
        args.sequences, args.labels, ppi_path=args.ppi, max_records=args.max_records
    )
    model, history = train_contract_batch(batch, epochs=args.epochs, stage_epochs=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history, "provenance": provenance}, output.with_suffix(".pt"))
    report = {"history": history, "provenance": provenance, "note": "pilot run; masked task coverage is reported explicitly"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
