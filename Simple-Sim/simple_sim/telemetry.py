"""Lightweight JSONL telemetry for live monitoring.

If the environment variable SIMPLE_SIM_EVENT_LOG is set to a file path, scripts
can emit JSON events for a GUI/monitor to consume. If not set, emit() is a no-op.

The sink is resolved once per process (plan S2): on the first :func:`emit` from
``$SIMPLE_SIM_EVENT_LOG``, or explicitly with :func:`set_sink` (a CLI boundary or test).
The file is opened once in append mode and every event is flushed as one line, so a
monitor tailing it sees the same lines as before. Changing the environment variable
after the first event has no effect; call :func:`set_sink` (or :func:`reset_sink` to
re-read the variable) instead.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import IO, Any

EVENT_LOG_ENV = "SIMPLE_SIM_EVENT_LOG"

_UNRESOLVED = object()
_lock = threading.Lock()
_path: Any = _UNRESOLVED  # str (sink path), None (disabled) or _UNRESOLVED
_fh: IO[str] | None = None


def _close() -> None:
    global _fh
    if _fh is not None:
        try:
            _fh.close()
        except OSError:
            pass
        _fh = None


def set_sink(path: str | os.PathLike[str] | None) -> None:
    """Send events to *path* (``None`` or ``""``: disable) instead of ``$SIMPLE_SIM_EVENT_LOG``."""
    global _path
    with _lock:
        _close()
        _path = str(path) if path else None


def reset_sink() -> None:
    """Forget the resolved sink; the next :func:`emit` reads ``$SIMPLE_SIM_EVENT_LOG`` again."""
    global _path
    with _lock:
        _close()
        _path = _UNRESOLVED


def current_sink() -> str | None:
    """Path events go to (resolving ``$SIMPLE_SIM_EVENT_LOG`` if not done yet), or ``None``."""
    global _path
    if _path is _UNRESOLVED:
        with _lock:
            if _path is _UNRESOLVED:
                _path = os.environ.get(EVENT_LOG_ENV) or None
    return _path


def _handle() -> IO[str] | None:
    """Open sink file (lazily, once); caller holds ``_lock``."""
    global _fh
    if _fh is None and _path:
        os.makedirs(os.path.dirname(_path) or ".", exist_ok=True)
        _fh = Path(_path).open("a", encoding="utf-8")
    return _fh


def emit(event: str, **fields: Any) -> None:
    """Append a single JSON event line to the sink (no-op if none is configured)."""
    if not current_sink():
        return
    # Best-effort: never crash the pipeline due to telemetry.
    try:
        line = json.dumps({"ts": time.time(), "event": event, **fields}, ensure_ascii=True) + "\n"
        with _lock:
            fh = _handle()
            if fh is not None:
                fh.write(line)
                fh.flush()
    except (OSError, TypeError, ValueError):
        return


atexit.register(reset_sink)
