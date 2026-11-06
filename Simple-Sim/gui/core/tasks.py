"""Background jobs with UI-thread callbacks (tk-free core).

Replaces the per-tab ``threading.Thread`` + ``queue`` + ``after()`` polling::

    ctx.tasks.run(
        lambda task: heavy_work(progress=task.progress, cancelled=task.cancelled),
        name="Validate dataset",
        on_done=lambda result: show(result),
        on_error=lambda exc: show_error("Validation failed", exc),
    )

``UiDispatcher`` is the only Tk touch point: ``attach(widget)`` drains posted
callbacks on the Tk event loop. Without it (tests) call ``drain()`` manually.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class UiDispatcher:
    """Thread-safe queue of callables executed on the UI thread."""

    def __init__(self, interval_ms: int = 40) -> None:
        self._q: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self.interval_ms = int(interval_ms)
        self._widget: Any = None
        self._after_id: str | None = None

    def post(self, fn: Callable[[], None]) -> None:
        self._q.put(fn)

    def drain(self, max_items: int = 500) -> int:
        n = 0
        while n < max_items:
            try:
                fn = self._q.get_nowait()
            except queue.Empty:
                break
            n += 1
            try:
                fn()
            except Exception:
                import traceback

                traceback.print_exc()
        return n

    def attach(self, widget: Any) -> None:
        """Start draining on ``widget``'s Tk loop."""
        self._widget = widget
        self._tick()

    def detach(self) -> None:
        if self._widget is not None and self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
        self._widget = None
        self._after_id = None

    def _tick(self) -> None:
        if self._widget is None:
            return
        self.drain()
        try:
            self._after_id = self._widget.after(self.interval_ms, self._tick)
        except Exception:
            self._widget = None


class TaskCancelled(Exception):
    """Raise from a task function (or via ``TaskHandle.check_cancelled``) to stop early."""


_ids = itertools.count(1)


@dataclass
class TaskHandle:
    name: str
    id: int = field(default_factory=lambda: next(_ids))
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    fraction: float | None = None  # 0..1, None = indeterminate
    message: str = ""
    state: str = "running"  # running | done | error | cancelled
    error: BaseException | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _on_progress: Callable[[TaskHandle], None] | None = field(default=None, repr=False)

    # --- called from the worker thread -------------------------------------------------
    def progress(self, fraction: float | None = None, message: str | None = None) -> None:
        if fraction is not None:
            self.fraction = max(0.0, min(1.0, float(fraction)))
        if message is not None:
            self.message = str(message)
        if self._on_progress is not None:
            self._on_progress(self)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise TaskCancelled(self.name)

    # --- called from the UI thread -----------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


class TaskRunner:
    """Runs callables on daemon threads; results/errors come back via ``post``."""

    def __init__(self, post: Callable[[Callable[[], None]], None], bus: Any = None) -> None:
        self._post = post
        self._bus = bus
        self._lock = threading.Lock()
        self._active: dict[int, TaskHandle] = {}

    @property
    def active(self) -> list[TaskHandle]:
        with self._lock:
            return list(self._active.values())

    def run(
        self,
        fn: Callable[[TaskHandle], Any],
        *,
        name: str = "Task",
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finally: Callable[[], None] | None = None,
    ) -> TaskHandle:
        handle = TaskHandle(name=name)
        handle._on_progress = lambda h: self._publish("task.progress", h)
        with self._lock:
            self._active[handle.id] = handle
        self._publish("task.started", handle)

        def worker() -> None:
            result: Any = None
            error: BaseException | None = None
            try:
                result = fn(handle)
            except TaskCancelled:
                handle.state = "cancelled"
            except BaseException as exc:  # noqa: BLE001 - reported to the UI
                error = exc
            self._post(lambda: self._finish(handle, result, error, on_done, on_error, on_finally))

        threading.Thread(target=worker, name=f"task-{handle.id}-{name}", daemon=True).start()
        return handle

    def cancel_all(self) -> None:
        for h in self.active:
            h.cancel()

    def _finish(self, handle, result, error, on_done, on_error, on_finally) -> None:
        handle.finished_at = time.time()
        if handle.state == "running":
            handle.state = "error" if error is not None else ("cancelled" if handle.cancelled() else "done")
        handle.error = error
        with self._lock:
            self._active.pop(handle.id, None)
        try:
            if error is not None:
                if on_error is not None:
                    on_error(error)
                else:
                    import traceback

                    traceback.print_exception(type(error), error, error.__traceback__)
            elif handle.state == "done" and on_done is not None:
                on_done(result)
        finally:
            if on_finally is not None:
                try:
                    on_finally()
                except Exception:
                    pass
            self._publish("task.finished", handle)

    def _publish(self, topic: str, handle: TaskHandle) -> None:
        if self._bus is not None:
            self._bus.publish(topic, task=handle)
