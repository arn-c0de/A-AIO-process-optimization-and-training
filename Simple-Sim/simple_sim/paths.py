"""Checkout paths (stdlib only): the Simple-Sim root that holds ``simple_sim/``, ``configs/`` and ``outputs/``."""

from __future__ import annotations

from pathlib import Path

#: The Simple-Sim directory (parent of the ``simple_sim`` package).
SIM_ROOT = Path(__file__).resolve().parents[1]
#: ``<SIM_ROOT>/outputs`` (run artefacts: models, datasets, board3d scenes, GUI settings).
OUTPUTS_DIR = SIM_ROOT / "outputs"
#: ``<SIM_ROOT>/configs`` (run configs, ``profiles/``, ``realism_profiles.yaml``).
CONFIGS_DIR = SIM_ROOT / "configs"

__all__ = ["CONFIGS_DIR", "OUTPUTS_DIR", "SIM_ROOT"]
