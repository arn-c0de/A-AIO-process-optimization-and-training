#!/usr/bin/env python3
"""Batch predict on a dataset split and report accuracy/metrics + history compare.

Shim for ``python -m simple_sim batch-predict``; the command lives in :mod:`simple_sim.commands.batch_predict`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/batch_predict.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.batch_predict import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
