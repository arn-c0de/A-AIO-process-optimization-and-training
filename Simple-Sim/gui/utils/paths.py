"""Well-known Simple-Sim output locations shared by GUI tabs (tk-free)."""

from __future__ import annotations

from pathlib import Path


def model_root(sim_root: Path) -> Path:
    """Root directory for model checkpoints (``outputs/models``)."""
    return Path(sim_root) / "outputs" / "models"
