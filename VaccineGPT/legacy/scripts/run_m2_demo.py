from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src_m2.training import train_m2


if __name__ == "__main__":
    # Fallback note: if feature extraction requires external models, this script can switch to cache-backed inputs.
    # The M2 contract remains stable because h_final is expected to be a fixed 128-d vector regardless of source.
    metrics = train_m2(num_epochs=6, batch_size=32)
    print(json.dumps(metrics, indent=2))
