"""Deterministic mathematical/software invariants for VaccineGPT."""

try:
    from .math import run
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verification.math import run


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
