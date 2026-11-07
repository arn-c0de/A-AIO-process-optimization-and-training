#!/usr/bin/env python3
"""Evaluate trained profile classifier on test set.

Shim for ``python -m simple_sim eval-profile``; the command lives in :mod:`simple_sim.commands.eval_profile`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/eval_profile.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.eval_profile import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
