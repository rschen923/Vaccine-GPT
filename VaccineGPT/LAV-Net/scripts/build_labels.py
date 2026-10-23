from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lavnet.labels import build_candidate_table, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/label_sources.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    frame = build_candidate_table(
        root / config["proteome"],
        known_antigens=root / config["known_antigen_file"],
        attenuation_evidence=root / config["attenuation_evidence_file"],
        fitness=root / config["fitness_file"],
    )
    write_jsonl(frame, root / config["label_output"], config["label_schema_version"])
    print(json.dumps({"rows": len(frame), "output": config["label_output"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
