"""Machine-readable CLI output convention for agents and automation.

Every script that supports it accepts ``--json``. When set, the script prints exactly one
JSON object as the **last line of stdout**, prefixed with ``RESULT_JSON_PREFIX`` so it can be
found reliably even if libraries print other output::

    SIMPLE_SIM_RESULT {"ok": true, "command": "eval", ...}

The object always has ``ok`` (bool) and ``command`` (str). On failure it has ``error`` (str)
and the process exits non-zero. Use :func:`parse_result` to read it back.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

RESULT_JSON_PREFIX = "SIMPLE_SIM_RESULT "


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--json`` flag to *parser*."""
    parser.add_argument(
        "--json",
        action="store_true",
        help=f"Print a machine-readable result line ('{RESULT_JSON_PREFIX.strip()} {{...}}') as last stdout line",
    )


def emit_result(command: str, payload: dict[str, Any] | None = None, *, ok: bool = True, error: str | None = None) -> None:
    """Print the result line (call only when ``--json`` is set)."""
    obj: dict[str, Any] = {"ok": ok, "command": command}
    if error is not None:
        obj["error"] = error
    obj.update(payload or {})
    sys.stdout.flush()
    print(RESULT_JSON_PREFIX + json.dumps(obj, ensure_ascii=True, default=str), flush=True)


def parse_result(stdout: str) -> dict[str, Any] | None:
    """Return the last result object found in *stdout*, or ``None``."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_JSON_PREFIX):
            return json.loads(line[len(RESULT_JSON_PREFIX):])
    return None
