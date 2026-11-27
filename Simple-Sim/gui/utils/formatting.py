"""Small display-formatting helpers shared by GUI tabs."""

from __future__ import annotations

from simple_sim.io_utils import format_bytes


RULE = "=" * 60


def section_header(title: str) -> str:
    """``title`` between two rules (info dialogs), ending with a newline."""
    return f"{RULE}\n{title}\n{RULE}\n"


def fmt_list(items: list[str], limit: int = 8) -> str:
    """Newline-joined *items*, truncated to *limit* entries with a ``... (+N more)`` line."""
    shown = "\n".join(items[:limit])
    return f"{shown}\n... (+{len(items) - limit} more)" if len(items) > limit else shown


def fmt_bytes(n: int) -> str:
    """Human-readable byte count with one decimal (``1.5 MB``); negative values show as ``0 B``."""
    return format_bytes(n, decimals=1)
