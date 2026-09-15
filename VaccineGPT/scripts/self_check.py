from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.self_correction import inspect_dataset, optimize_training_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VaccineGPT dataset self-check and optimization policy")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits")
    parser.add_argument("--output", default="data/metadata/self_check.json")
    args = parser.parse_args()
    audit = inspect_dataset(args.sequences, args.labels, args.splits)
    audit["training_policy"] = optimize_training_policy(audit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
