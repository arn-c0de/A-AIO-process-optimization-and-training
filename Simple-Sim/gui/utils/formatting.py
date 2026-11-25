"""Small display-formatting helpers shared by GUI tabs."""

from __future__ import annotations

from simple_sim.io_utils import format_bytes


RULE = "=" * 60


def section_header(title: str) -> str:
    """``title`` between two rules (info dialogs), ending with a newline."""
    return f"{RULE}\n{title}\n{RULE}\n"


def fmt_bytes(n: int) -> str:
    """Human-readable byte count with one decimal (``1.5 MB``); negative values show as ``0 B``."""
    return format_bytes(n, decimals=1)
