"""Small display-formatting helpers shared by GUI tabs."""

from __future__ import annotations


def fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (negative values show as 0 B)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(max(0, n))
    u = 0
    while v >= 1024.0 and u < len(units) - 1:
        v /= 1024.0
        u += 1
    if u == 0:
        return f"{int(v)} {units[u]}"
    return f"{v:.1f} {units[u]}"
