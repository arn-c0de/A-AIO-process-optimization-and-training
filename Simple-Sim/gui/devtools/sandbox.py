"""Build a throw-away Simple-Sim root for GUI tests and screenshots.

Code (``scripts``, ``simple_sim``, ``tools``, ``gui``, launchers, ``.venv``) is
*symlinked* into the checkout; anything written through those links lands in
the real repo, so GUI actions on a sandbox must not write there. ``configs/``
(run configs and profiles, which the GUI and the profile editor do write) is a
*copy*, and ``outputs/`` is empty, so settings, datasets, models and configs of
the real checkout are never touched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_LINKED = ("scripts", "simple_sim", "tools", "gui", "predict.sh", "run_pipeline.sh", ".venv")
_COPIED = ("configs",)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_sandbox_root(dest: Path, *, src: Path | None = None) -> Path:
    """Create ``dest`` as a sandbox root linked to ``src`` (default: this checkout)."""
    src = Path(src or repo_root())
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in _LINKED:
        target = src / name
        link = dest / name
        if target.exists() and not link.exists():
            os.symlink(target, link, target_is_directory=target.is_dir())
    for name in _COPIED:
        target = src / name
        copy = dest / name
        if target.is_dir() and not copy.exists():
            shutil.copytree(target, copy, symlinks=True)
    for sub in ("outputs/sim_data/runs", "outputs/models", "outputs/gui"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    return dest
