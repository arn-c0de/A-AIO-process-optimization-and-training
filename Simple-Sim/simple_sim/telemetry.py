"""Lightweight JSONL telemetry for live monitoring.

If the environment variable SIMPLE_SIM_EVENT_LOG is set to a file path, scripts
can emit JSON events for a GUI/monitor to consume. If not set, emit() is a no-op.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _event_path() -> Optional[str]:
    return os.environ.get("SIMPLE_SIM_EVENT_LOG") or None


def emit(event: str, **fields: Any) -> None:
    """Append a single JSON event line to SIMPLE_SIM_EVENT_LOG."""
    path = _event_path()
    if not path:
        return

    payload: Dict[str, Any] = {
        "ts": time.time(),
        "event": event,
        **fields,
    }

    # Best-effort: never crash the pipeline due to telemetry.
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
            f.flush()
    except Exception:
        return

