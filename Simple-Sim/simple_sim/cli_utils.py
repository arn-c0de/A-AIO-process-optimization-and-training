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
import logging
import sys
from collections.abc import Callable
from typing import Any

RESULT_JSON_PREFIX = "SIMPLE_SIM_RESULT "

#: Flag advice appended by :func:`with_hint` to the library's neutral mismatch errors.
PROFILE_MISMATCH_HINT = "To override (NOT recommended): pass --allow-profile-mismatch"
RENDERER_MISMATCH_HINT = "To mix them anyway: pass --allow-renderer-mismatch"


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--json`` flag to *parser*."""
    parser.add_argument(
        "--json",
        action="store_true",
        help=f"Print a machine-readable result line ('{RESULT_JSON_PREFIX.strip()} {{...}}') as last stdout line",
    )


# ---------------------------------------------------------------------------
# Shared argparse fragments (plan D2). The help text is passed by each script, so ``--help``
# output stays exactly as it was; values are plain strings (callers wrap them in ``Path``).
# ---------------------------------------------------------------------------

def add_data_arg(parser: argparse.ArgumentParser, help: str, *, required: bool = True) -> None:
    """Add ``--data`` (dataset directory)."""
    parser.add_argument("--data", type=str, required=required, help=help)


def add_model_arg(parser: argparse.ArgumentParser, help: str, *, required: bool = True) -> None:
    """Add ``--model`` (checkpoint ``.pt`` or model bundle directory)."""
    parser.add_argument("--model", type=str, required=required, help=help)


def add_out_arg(parser: argparse.ArgumentParser, help: str, *, required: bool = True, default: str | None = None) -> None:
    """Add ``--out`` (output path or directory; *default* only applies when not *required*)."""
    parser.add_argument("--out", type=str, required=required, default=default, help=help)


def add_amp_flags(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive ``--amp`` / ``--no-amp`` pair (``args.amp``, default True)."""
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', dest='amp', action='store_true', help='Enable mixed precision on CUDA (default).')
    amp_group.add_argument('--no-amp', dest='amp', action='store_false', help='Disable mixed precision.')
    parser.set_defaults(amp=True)


def add_profile_mismatch_flag(parser: argparse.ArgumentParser, help: str) -> None:
    """Add ``--allow-profile-mismatch`` (``args.allow_profile_mismatch``); see :data:`PROFILE_MISMATCH_HINT`."""
    parser.add_argument("--allow-profile-mismatch", action="store_true", help=help)


def emit_result(command: str, payload: dict[str, Any] | None = None, *, ok: bool = True, error: str | None = None) -> None:
    """Print the result line (call only when ``--json`` is set)."""
    obj: dict[str, Any] = {"ok": ok, "command": command.replace("_", "-")}
    if error is not None:
        obj["error"] = error
    obj.update(payload or {})
    sys.stdout.flush()
    print(RESULT_JSON_PREFIX + json.dumps(obj, ensure_ascii=True, default=str), flush=True)


class _CurrentStdoutHandler(logging.StreamHandler):
    """Stream handler that always writes to the *current* ``sys.stdout`` (pytest capture, redirects)."""

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:
        pass


class _CliFormatter(logging.Formatter):
    """Plain message for INFO and below; ``WARNING: ...`` / ``ERROR: ...`` prefix above."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.WARNING:
            lead = len(msg) - len(msg.lstrip("\n"))
            msg = msg[:lead] + f"{record.levelname}: " + msg[lead:]
        return msg


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Show ``simple_sim`` log records on stdout, as the scripts' console output.

    Idempotent. Records keep propagating, so an application or pytest's ``caplog`` still sees them.
    """
    logger = logging.getLogger("simple_sim")
    logger.setLevel(level)
    if not any(getattr(h, "_simple_sim_cli", False) for h in logger.handlers):
        handler = _CurrentStdoutHandler()
        handler.setFormatter(_CliFormatter("%(message)s"))
        handler._simple_sim_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


class UsageError(ValueError):
    """Invalid combination of arguments; shown as a plain message (no traceback) by :func:`run_cli`."""


def run_cli(command: str, fn: Callable[[], dict[str, Any] | None], json_mode: bool) -> dict[str, Any] | None:
    """Run a script's core *fn* and handle ``--json`` output.

    Without *json_mode* exceptions propagate unchanged (a :class:`UsageError` becomes a plain
    ``SystemExit`` message). With *json_mode* the returned payload is emitted as the result line;
    any exception or failing ``SystemExit`` emits ``ok=false`` + ``error`` and exits with status 1.
    """
    configure_cli_logging()
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


def with_hint(fn: Callable[[], Any], errors: type[BaseException] | tuple[type[BaseException], ...], hint: str) -> Callable[[], Any]:
    """Wrap *fn* so that *errors* become a :class:`UsageError` with the CLI *hint* appended.

    The library raises neutral errors (it knows no command-line flags); scripts add the flag advice
    here, e.g. ``with_hint(run, ProfileMismatchError, "To override: pass --allow-profile-mismatch")``.
    """
    def wrapped() -> Any:
        try:
            return fn()
        except errors as e:
            raise UsageError(f"{e}\n\n{hint}") from e
    return wrapped


def parse_result(stdout: str) -> dict[str, Any] | None:
    """Return the last result object found in *stdout*, or ``None``."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_JSON_PREFIX):
            return json.loads(line[len(RESULT_JSON_PREFIX):])
    return None
