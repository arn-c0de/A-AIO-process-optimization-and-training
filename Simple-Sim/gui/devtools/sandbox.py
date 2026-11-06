"""Build a throw-away Simple-Sim root for GUI tests and screenshots.

The sandbox links the read-only parts of the checkout (``configs``, ``scripts``,
``simple_sim``, ``tools``, launchers) and gets its own empty ``outputs/`` so a
GUI started on it never touches the real settings, datasets or models.
"""

from __future__ import annotations

import os
from pathlib import Path

_LINKED = ("configs", "scripts", "simple_sim", "tools", "gui", "predict.sh", "run_pipeline.sh", ".venv")


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
    for sub in ("outputs/sim_data/runs", "outputs/models", "outputs/gui"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    return dest
