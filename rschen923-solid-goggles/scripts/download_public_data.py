from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.download import run_download_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download enabled public data sources reproducibly")
    parser.add_argument("--config", default="configs/data_sources.json")
    args = parser.parse_args()
    print(json.dumps(run_download_manifest(args.config), indent=2))


if __name__ == "__main__":
    main()
