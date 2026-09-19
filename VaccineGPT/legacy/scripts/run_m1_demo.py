from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src_m1.training import train_m1


if __name__ == "__main__":
    # Fallback note: if pretrained encoders are unavailable, this stage consumes cached feature vectors.
    # The cache path can be provided later by the pipeline configuration, while keeping the training interface stable.
    metrics = train_m1(num_epochs=6, batch_size=32)
    print(json.dumps(metrics, indent=2))
