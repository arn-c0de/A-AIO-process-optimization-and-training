#!/usr/bin/env python3
"""Predict defect class for a single image using a trained checkpoint.

Shim for ``python -m simple_sim predict``; the command lives in :mod:`simple_sim.commands.predict`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/predict.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.predict import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
