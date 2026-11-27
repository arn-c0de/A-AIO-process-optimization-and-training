"""Where the Blender executable is: one lookup for dataset generation, the GUI previews and the board3d renderer.

Stdlib-only, like the rest of :mod:`simple_sim.blender`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from simple_sim.blender.job_schema import BLENDER_ENV_VAR, DEFAULT_BLENDER_EXECUTABLE

__all__ = ["require_blender", "resolve_blender_executable"]


def resolve_blender_executable(configured: str | None = None) -> str:
    """Blender executable: *configured* (``render.blender.executable``) > ``$SIMPLE_SIM_BLENDER`` > ``blender``."""
    return str(configured or os.environ.get(BLENDER_ENV_VAR) or DEFAULT_BLENDER_EXECUTABLE)


def require_blender(configured: str | None = None) -> str:
    """The resolved executable (:func:`resolve_blender_executable`) as a runnable path: looked up on the PATH, or an
    existing file; ``FileNotFoundError`` when there is none."""
    exe = resolve_blender_executable(configured)
    found = shutil.which(exe) or (exe if Path(exe).exists() else None)
    if not found:
        raise FileNotFoundError(f"Blender not found: {exe} (set {BLENDER_ENV_VAR} or render.blender.executable)")
    return found
