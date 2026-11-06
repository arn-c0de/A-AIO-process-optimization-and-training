"""Well-known Simple-Sim output locations shared by GUI tabs (tk-free)."""

from __future__ import annotations

from pathlib import Path


def model_root(sim_root: Path) -> Path:
    """Root directory for model checkpoints (``outputs/models``)."""
    return Path(sim_root) / "outputs" / "models"


def newest_file(directory: Path, pattern: str, *, min_mtime: float | None = None) -> Path | None:
    """Most recently modified file in *directory* matching *pattern* (optionally ``mtime >= min_mtime``)."""
    try:
        cand = [(p.stat().st_mtime, p) for p in Path(directory).glob(pattern)]
    except OSError:
        return None
    if min_mtime is not None:
        cand = [t for t in cand if t[0] >= min_mtime]
    return sorted(cand, key=lambda t: t[0])[-1][1] if cand else None
