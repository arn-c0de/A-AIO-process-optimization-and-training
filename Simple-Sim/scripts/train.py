#!/usr/bin/env python3
"""Train ResNet18 classifier on synthetic dataset.

Shim for ``python -m simple_sim train``; the command lives in :mod:`simple_sim.commands.train`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/train.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
