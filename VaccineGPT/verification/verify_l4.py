"""Deterministic L4/MHC weighting and masking verification."""

try:
    from .l4 import run
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verification.l4 import run


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
