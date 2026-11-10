"""Best-effort UI calls: errors are swallowed, but no longer without a trace.

``with quiet():`` replaces ``with contextlib.suppress(Exception):`` around cosmetic Tk calls (a widget
that is already destroyed, a style the theme lacks, ...). The call still never raises, but the error
is logged at DEBUG level with its traceback under the ``gui.quiet`` logger, so
``logging.getLogger("gui.quiet").setLevel(logging.DEBUG)`` shows what the GUI silently ignores.
"""

from __future__ import annotations

import logging
from types import TracebackType

__all__ = ["quiet"]

_LOG = logging.getLogger("gui.quiet")


class quiet:  # noqa: N801 - used like contextlib.suppress
    """Context manager swallowing any ``Exception`` (not ``KeyboardInterrupt`` / ``SystemExit``) and logging it at DEBUG."""

    __slots__ = ("what",)

    def __init__(self, what: str = "") -> None:
        self.what = what

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        if exc_type is None:
            return False
        if not issubclass(exc_type, Exception):
            return False
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug("ignored UI error%s: %r", f" ({self.what})" if self.what else "", exc, exc_info=(exc_type, exc, tb))
        return True
