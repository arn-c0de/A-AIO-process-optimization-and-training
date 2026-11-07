#!/usr/bin/env python3
"""Generate synthetic PCB defect dataset.

Shim for ``python -m simple_sim generate``; the command lives in :mod:`simple_sim.commands.generate`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/generate.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.generate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
