#!/usr/bin/env python3
"""Part catalog: list, look up and add real part geometry (datasheet heights), check a board3d against it.

Shim for ``python -m simple_sim parts``; the command lives in :mod:`simple_sim.commands.parts`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/parts.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.parts import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
