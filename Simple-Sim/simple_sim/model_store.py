"""Model store layout: checkpoint snapshots under ``outputs/models/versions/<stem>/`` (plan L12, X6).

The Pipeline tab (automatic snapshots per run) and the Weights tab (manual "snapshot") used to
build snapshot names independently. :func:`snapshot_path` is the one naming scheme::

    versions/<stem>/<stem>_<YYYYmmdd_HHMMSS>[_run<NNN>][_<NNN>].pt

``_run<NNN>`` is the pipeline run iteration (absent for manual snapshots) and the trailing
``_<NNN>`` only disambiguates snapshots taken within the same second. Pure stdlib.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from simple_sim.io_utils import now_tag

__all__ = ["SNAPSHOT_RUN_RE", "snapshot_checkpoint", "snapshot_path", "versions_dir"]

#: Matches the ``_run<NNN>`` part of a snapshot name (group ``run``).
SNAPSHOT_RUN_RE = re.compile(r"_run(?P<run>\d+)(?:_|\.|$)")

_MAX_SAME_SECOND = 999


def versions_dir(models_root: Path, model_path: Path) -> Path:
    """``<models_root>/versions/<stem>``: where snapshots of *model_path* go."""
    return Path(models_root) / "versions" / Path(model_path).stem


def snapshot_path(models_root: Path, model_path: Path, *, run: int | None = None, tag: str | None = None) -> Path:
    """First free snapshot path for *model_path* (see the module docstring for the scheme).

    *tag* defaults to the current local time (``now_tag(micro=False)``); *run* adds ``_run<NNN>``.
    """
    model_path = Path(model_path)
    root = versions_dir(models_root, model_path)
    stem = f"{model_path.stem}_{tag or now_tag(micro=False)}" + (f"_run{int(run):03d}" if run is not None else "")
    base = root / f"{stem}.pt"
    if not base.exists():
        return base
    for i in range(1, _MAX_SAME_SECOND + 1):
        cand = root / f"{stem}_{i:03d}.pt"
        if not cand.exists():
            return cand
    return base


def snapshot_checkpoint(models_root: Path, model_path: Path, *, run: int | None = None) -> Path:
    """Copy *model_path* (with metadata) to :func:`snapshot_path` and return the snapshot path."""
    dst = snapshot_path(models_root, model_path, run=run)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, dst)
    return dst
