"""Machine-readable CLI output convention for agents and automation.

Every script that supports it accepts ``--json``. When set, the script prints exactly one
JSON object as the **last line of stdout**, prefixed with ``RESULT_JSON_PREFIX`` so it can be
found reliably even if libraries print other output::

    SIMPLE_SIM_RESULT {"ok": true, "command": "eval", ...}

The object always has ``ok`` (bool) and ``command`` (str, the ``python -m simple_sim``
subcommand name, e.g. ``batch-predict``). On failure it has ``error`` (str)
and the process exits non-zero. Use :func:`parse_result` to read it back.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
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
    obj: dict[str, Any] = {"ok": ok, "command": command.replace("_", "-")}
    if error is not None:
        obj["error"] = error
    obj.update(payload or {})
    sys.stdout.flush()
    print(RESULT_JSON_PREFIX + json.dumps(obj, ensure_ascii=True, default=str), flush=True)


class UsageError(ValueError):
    """Invalid combination of arguments; shown as a plain message (no traceback) by :func:`run_cli`."""


def run_cli(command: str, fn: Callable[[], dict[str, Any] | None], json_mode: bool) -> dict[str, Any] | None:
    """Run a script's core *fn* and handle ``--json`` output.

    Without *json_mode* exceptions propagate unchanged (a :class:`UsageError` becomes a plain
    ``SystemExit`` message). With *json_mode* the returned payload is emitted as the result line;
    any exception or failing ``SystemExit`` emits ``ok=false`` + ``error`` and exits with status 1.
    """
    if not json_mode:
        try:
            return fn()
        except UsageError as e:
            raise SystemExit(str(e)) from None
    try:
        payload = fn()
    except SystemExit as e:
        if e.code in (None, 0):
            raise
        emit_result(command, ok=False, error=str(e.code))
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - reported to the caller as the JSON error
        emit_result(command, ok=False, error=f"{type(e).__name__}: {e}")
        sys.exit(1)
    emit_result(command, payload or {})
    return payload


def parse_result(stdout: str) -> dict[str, Any] | None:
    """Return the last result object found in *stdout*, or ``None``."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_JSON_PREFIX):
            return json.loads(line[len(RESULT_JSON_PREFIX):])
    return None
