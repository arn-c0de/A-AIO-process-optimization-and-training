"""Subprocess helpers shared by GUI tabs (tk-free).

``run_streaming`` replaces the per-tab "Popen, stream stdout into the log
queue, honour a stop event, wait" loops; ``terminate_process`` stops a child
and, when it leads its own process group (``start_new_session=True``), the
whole group so grandchildren (e.g. the python started by ``predict.sh``) do
not survive.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
from pathlib import Path
from collections.abc import Callable, Sequence


def _is_group_leader(proc: subprocess.Popen) -> bool:
    if not hasattr(os, "getpgid"):
        return False
    try:
        return os.getpgid(proc.pid) == proc.pid
    except Exception:
        return False


def _signal_process(proc: subprocess.Popen, sig: int) -> None:
    if _is_group_leader(proc) and hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, sig)
            return
        except Exception:
            pass
    try:
        if sig == getattr(signal, "SIGKILL", None):
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


def terminate_process(proc: subprocess.Popen | None, *, kill_after: float | None = 2.0) -> None:
    """Send SIGTERM to ``proc`` (its whole process group if it leads one).

    Non-blocking: when ``kill_after`` is set, a timer escalates to SIGKILL if
    the process is still alive after that many seconds. No-op for ``None`` or
    already-exited processes.
    """
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    _signal_process(proc, signal.SIGTERM)

    if kill_after is None:
        return
    kill_sig = getattr(signal, "SIGKILL", signal.SIGTERM)

    def _kill_later() -> None:
        try:
            if proc.poll() is None:
                _signal_process(proc, kill_sig)
        except Exception:
            pass

    timer = threading.Timer(kill_after, _kill_later)
    timer.daemon = True
    timer.start()


def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: str | Path,
    log_q: queue.Queue[str],
    stop_evt: threading.Event | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
    start_new_session: bool = True,
) -> int | None:
    """Run ``cmd`` and stream its combined stdout/stderr lines into ``log_q``.

    - ``on_start(proc)`` is called right after the process starts (callers use
      it to store ``self.proc`` so their stop methods can reach it).
    - Each line is put on ``log_q`` and then passed to ``on_line`` (for parsing
      e.g. "Report saved to:").
    - If ``stop_evt`` is set when streaming ends (set during the run, or the
      stream hit EOF because a stop method killed the child), the process
      (group) is terminated *before* waiting, our end of the pipe is closed so
      the child cannot block on a full pipe, and ``None`` is returned.
    - Otherwise the process exit code is returned.

    The child runs in its own session/process group by default so that a stop
    also reaches grandchildren (shell wrappers). Popen errors propagate.
    """
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=start_new_session,
    )
    if on_start is not None:
        on_start(proc)

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if stop_evt is not None and stop_evt.is_set():
                break
            log_q.put(line)
            if on_line is not None:
                on_line(line)
    except BaseException:
        _abort(proc)
        raise

    stopped = stop_evt is not None and stop_evt.is_set()
    if stopped:
        _abort(proc)
    rc = proc.wait()
    return None if stopped else rc


def _abort(proc: subprocess.Popen) -> None:
    """Terminate (group-aware) and close our pipe end so wait() cannot hang."""
    terminate_process(proc)
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:
        pass

__all__: list[str] = ["run_streaming", "terminate_process"]
