"""Lenient value coercion for user-edited settings (JSON/YAML/GUI fields). Stdlib only."""

from __future__ import annotations

from typing import Any, TypeVar

_D = TypeVar("_D")

TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})

_NUMBER_ERRORS = (TypeError, ValueError, OverflowError)


def as_bool(value: Any, default: bool = False) -> bool:
    """``True``/``False`` from bools, numbers and strings (``"1"``/``"true"``/``"yes"``/``"on"``,
    case-insensitive; any other string is ``False``); *default* for anything else (``None``, lists)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in TRUE_STRINGS
    return default


def as_float(value: Any, default: float | _D, lo: float | None = None, hi: float | None = None) -> float | _D:
    """``float(value)``, or *default* if that fails (as float unless it is ``None``); then clamped
    to ``[lo, hi]`` when both are given."""
    try:
        v = float(value)
    except _NUMBER_ERRORS:
        v = default if default is None else float(default)  # type: ignore[arg-type]
    if lo is not None and hi is not None and v is not None:
        v = max(lo, min(hi, v))
    return v


def as_int(value: Any, default: int | _D, lo: int | None = None, hi: int | None = None) -> int | _D:
    """``int(value)`` (so ``"2.5"`` fails, ``2.5`` truncates), or *default*; then clamped like :func:`as_float`."""
    try:
        v = int(value)
    except _NUMBER_ERRORS:
        v = default  # type: ignore[assignment]
    if lo is not None and hi is not None and v is not None:
        v = max(lo, min(hi, v))
    return v
