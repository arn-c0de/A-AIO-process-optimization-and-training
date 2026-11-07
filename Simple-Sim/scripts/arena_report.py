#!/usr/bin/env python3
"""Generate a markdown "Arena" ranking report for tracked models.

Shim for ``python -m simple_sim arena-report``; the command lives in :mod:`simple_sim.commands.arena_report`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/arena_report.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.arena_report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
