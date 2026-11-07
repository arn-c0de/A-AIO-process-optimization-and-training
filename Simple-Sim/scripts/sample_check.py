#!/usr/bin/env python3
"""Sample a few images per class from a dataset split and verify predictions vs GT.

Shim for ``python -m simple_sim sample-check``; the command lives in :mod:`simple_sim.commands.sample_check`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/sample_check.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.sample_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
