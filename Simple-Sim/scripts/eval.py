#!/usr/bin/env python3
"""Evaluate trained model on frozen test set.

Shim for ``python -m simple_sim eval``; the command lives in :mod:`simple_sim.commands.eval`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/eval.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
