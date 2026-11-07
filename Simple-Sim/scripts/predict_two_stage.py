#!/usr/bin/env python3
"""Two-stage prediction: profile identification + defect classification.

Shim for ``python -m simple_sim predict-two-stage``; the command lives in :mod:`simple_sim.commands.predict_two_stage`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/predict_two_stage.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.predict_two_stage import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
