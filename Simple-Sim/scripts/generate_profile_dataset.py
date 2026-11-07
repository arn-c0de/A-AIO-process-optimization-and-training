#!/usr/bin/env python3
"""Generate a mixed-profile dataset for training a profile (component type) classifier.

Shim for ``python -m simple_sim generate-profile-dataset``; the command lives in :mod:`simple_sim.commands.generate_profile_dataset`.
"""

import sys
from pathlib import Path

# Lets ``python scripts/generate_profile_dataset.py`` run from a checkout without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.commands.generate_profile_dataset import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
