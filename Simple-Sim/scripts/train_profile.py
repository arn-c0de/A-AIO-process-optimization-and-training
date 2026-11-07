#!/usr/bin/env python3
"""Train ResNet18 profile (component type) classifier on mixed-profile dataset.

Shim for ``python -m simple_sim train-profile``; the command lives in :mod:`simple_sim.commands.train_profile`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/train_profile.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.train_profile import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
