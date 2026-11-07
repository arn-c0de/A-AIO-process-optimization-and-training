#!/usr/bin/env python3
"""Render small per-profile preview images for debugging 2D/3D profiles.

Shim for ``python -m simple_sim render-debug-previews``; the command lives in :mod:`simple_sim.commands.render_debug_previews`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/render_debug_previews.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.render_debug_previews import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
