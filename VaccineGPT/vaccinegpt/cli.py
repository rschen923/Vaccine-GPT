"""Command line entry points for data validation, smoke tests, and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import validate_jsonl
from .synthetic import SyntheticConfig, generate_synthetic_jsonl
from .training import TrainingConfig, train_jsonl
from .spec_training import load_contract_jsonl, make_synthetic_contract_batch, train_contract_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vaccinegpt")
    parser.add_argument("--smoke", action="store_true", help="run the CPU synthetic smoke test")
    parser.add_argument("--data", default=None, help="validated JSONL training data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage-epochs", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    commands = parser.add_subparsers(dest="command", required=False)

    synthetic = commands.add_parser("synthetic", help="generate a deterministic JSONL dataset")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--records", type=int, default=24)
    synthetic.add_argument("--seed", type=int, default=7)

    validate = commands.add_parser("validate", help="validate a VaccineGPT JSONL dataset")
    validate.add_argument("input")
    validate.add_argument("--inference", action="store_true", help="allow labels to be omitted")

    smoke = commands.add_parser("smoke", help="run a synthetic one-step training smoke test")
    smoke.add_argument("--output", default=None, help="optional JSONL path to generate first")

    train = commands.add_parser("train", help="train on a validated JSONL dataset")
    train.add_argument("input")
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--no-pcgrad", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke or args.command is None:
        target = args.data or "<synthetic in-memory batch>"
        model, history = train_contract_batch(
            load_contract_jsonl(args.data) if args.data else make_synthetic_contract_batch(),
            epochs=args.epochs or 2, stage_epochs=args.stage_epochs,
        )
        if args.checkpoint:
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            import torch
            torch.save({"model": model.state_dict(), "history": history}, args.checkpoint)
        print(json.dumps({"output": target, "history": history}, indent=2))
        return 0
    if args.command == "synthetic":
        count = generate_synthetic_jsonl(args.output, SyntheticConfig(args.records, args.seed))
        print(json.dumps({"written": count, "output": str(args.output)}))
        return 0
    if args.command == "validate":
        print(json.dumps(validate_jsonl(args.input, require_labels=not args.inference), indent=2))
        return 0
    if args.command == "smoke":
        target = args.output or str(Path("synthetic_smoke.jsonl"))
        generate_synthetic_jsonl(target)
        _, history = train_jsonl(target, TrainingConfig(epochs=2, pcgrad=True))
        print(json.dumps({"output": target, "history": history}, indent=2))
        return 0
    model, history = train_jsonl(
        args.input,
        TrainingConfig(epochs=args.epochs, learning_rate=args.learning_rate, pcgrad=not args.no_pcgrad),
    )
    if args.checkpoint:
        import torch
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "history": history}, args.checkpoint)
    print(json.dumps({"history": history}, indent=2))
    return 0
