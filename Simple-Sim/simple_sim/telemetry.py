"""Lightweight JSONL telemetry for live monitoring.

If the environment variable SIMPLE_SIM_EVENT_LOG is set to a file path, scripts
can emit JSON events for a GUI/monitor to consume. If not set, emit() is a no-op.
"""

from __future__ import annotations

import os
import time
from typing import Any

from simple_sim.io_utils import append_jsonl

EVENT_LOG_ENV = "SIMPLE_SIM_EVENT_LOG"


def emit(event: str, **fields: Any) -> None:
    """Append a single JSON event line to ``$SIMPLE_SIM_EVENT_LOG`` (no-op if unset)."""
    path = os.environ.get(EVENT_LOG_ENV)
    if not path:
        return
    # Best-effort: never crash the pipeline due to telemetry.
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        append_jsonl(path, [{"ts": time.time(), "event": event, **fields}])
    except (OSError, TypeError, ValueError):
        return

