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
        # Callbacks for the *next* drain: lets a consumer yield to Tk between batches.
        self._next: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self.interval_ms = int(interval_ms)
        self._widget: Any = None
        self._after_id: str | None = None

    def post(self, fn: Callable[[], None]) -> None:
        self._q.put(fn)

    def post_next_tick(self, fn: Callable[[], None]) -> None:
        """Like ``post`` but never runs inside the drain that is currently executing:
        ``fn`` runs at the start of the next ``drain()`` (next Tk tick)."""
        self._next.put(fn)

    def drain(self, max_items: int = 500) -> int:
        n = 0
        # Move deferred callbacks first (in order) so they run before newer posts.
        deferred: list[Callable[[], None]] = []
        while True:
            try:
                deferred.append(self._next.get_nowait())
            except queue.Empty:
                break
        for fn in deferred:
            n += 1
            try:
                fn()
            except Exception:
                import traceback

                traceback.print_exc()
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
        """Start draining on ``widget``'s Tk loop (idempotent: re-attaching
        cancels the previous polling chain first)."""
        self.detach()
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
        widget = self._widget
        if widget is None:
            return
        # Schedule the next tick *before* draining: a drained callback may open a
        # modal dialog (wait_window) and must not stall all other posted work.
        try:
            self._after_id = widget.after(self.interval_ms, self._tick)
        except Exception:
            self._widget = None
            self._after_id = None
            return
        self.drain()


class DispatchQueue(queue.Queue):
    """``queue.Queue`` that drains itself on the UI thread instead of being polled.

    Producers keep calling ``put`` / ``put_nowait`` (so logic classes that take a
    ``queue.Queue`` stay unchanged); the first ``put`` after a drain posts one drain
    callback through ``post``. The drain hands up to ``max_items`` items, in FIFO order,
    to ``handler(items)`` and, if more are left, reschedules itself via ``post_next``
    (default: ``post``) so a flood is processed in batches with Tk events in between,
    like the old ``after(120)`` tick that took at most N items per tick.

    Without ``bind`` (no handler yet) it is a plain queue; ``bind`` drains what was
    queued before. ``get_nowait`` still works (tests, legacy consumers).
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self._sched_lock = threading.Lock()
        self._scheduled = False
        self._post: Callable[[Callable[[], None]], None] | None = None
        self._post_next: Callable[[Callable[[], None]], None] | None = None
        self._handler: Callable[[list[Any]], None] | None = None
        self.max_items = 200

    def bind(
        self,
        post: Callable[[Callable[[], None]], None],
        handler: Callable[[list[Any]], None],
        *,
        max_items: int = 200,
        post_next: Callable[[Callable[[], None]], None] | None = None,
    ) -> DispatchQueue:
        self._post, self._post_next, self._handler = post, post_next, handler
        self.max_items = max(1, int(max_items))
        if not self.empty():
            self._schedule(post)
        return self

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        if self._post is not None:
            self._schedule(self._post)

    def _schedule(self, post: Callable[[Callable[[], None]], None]) -> None:
        with self._sched_lock:
            if self._scheduled:
                return
            self._scheduled = True
        post(self.drain_now)

    def drain_now(self) -> int:
        """Hand up to ``max_items`` queued items to the handler (UI thread). Returns the count."""
        with self._sched_lock:
            # Cleared before reading: a put() racing with this drain schedules a new one.
            self._scheduled = False
        handler = self._handler
        if handler is None:
            return 0
        items: list[Any] = []
        while len(items) < self.max_items:
            try:
                items.append(self.get_nowait())
            except queue.Empty:
                break
        try:
            if items:
                handler(items)
        finally:
            if not self.empty():
                self._schedule(self._post_next or self._post)  # type: ignore[arg-type]
        return len(items)


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
    _progress_pending: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    # --- called from the worker thread -------------------------------------------------
    def progress(self, fraction: float | None = None, message: str | None = None) -> None:
        with self._lock:
            if fraction is not None:
                self.fraction = max(0.0, min(1.0, float(fraction)))
            if message is not None:
                self.message = str(message)
            # Coalesce: at most one progress notification is in flight; the receiver reads the
            # latest fraction/message via ``snapshot()``. Test-and-set under the lock, so two
            # worker threads reporting at once cannot both post.
            if self._on_progress is None or self._progress_pending.is_set():
                return
            self._progress_pending.set()
        self._on_progress(self)

    def snapshot(self) -> tuple[float | None, str]:
        """Consistent (fraction, message) pair; clears the coalescing flag."""
        self._progress_pending.clear()
        with self._lock:
            return self.fraction, self.message

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise TaskCancelled(self.name)

    # --- called from the UI thread -----------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker thread; True if it has finished."""
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

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

    def post(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the UI thread (thread-safe)."""
        self._post(fn)

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
        handle._on_progress = self._progress_cb
        with self._lock:
            self._active[handle.id] = handle
        self._publish("task.started", handle)

        def worker() -> None:
            result: Any = None
            error: BaseException | None = None
            # The outcome is decided here, on the worker, when ``fn`` returns: a
            # task that finished normally stays "done" even if cancel() lands
            # before the UI thread drains ``_finish``.
            outcome = "done"
            try:
                result = fn(handle)
            except TaskCancelled:
                outcome = "cancelled"
            except BaseException as exc:  # noqa: BLE001 - reported to the UI
                error = exc
                outcome = "error"
            self._post(lambda: self._finish(handle, outcome, result, error, on_done, on_error, on_finally))

        t = threading.Thread(target=worker, name=f"task-{handle.id}-{name}", daemon=True)
        handle._thread = t
        t.start()
        return handle

    def cancel_all(self) -> None:
        for h in self.active:
            h.cancel()

    def shutdown(self, timeout: float = 3.0) -> list[TaskHandle]:
        """Cancel all tasks and wait up to ``timeout`` s in total for their
        workers, so a write in progress is not killed with the interpreter.
        Returns the tasks still running afterwards."""
        self.cancel_all()
        deadline = time.monotonic() + max(0.0, float(timeout))
        left: list[TaskHandle] = []
        for h in self.active:
            if not h.join(max(0.0, deadline - time.monotonic())):
                left.append(h)
        return left

    def _progress_cb(self, handle: TaskHandle) -> None:
        if self._bus is None:
            handle._progress_pending.clear()
            return

        def deliver() -> None:
            handle._progress_pending.clear()  # later progress() calls post again
            self._publish("task.progress", handle)

        self._post(deliver)

    def _finish(self, handle, outcome, result, error, on_done, on_error, on_finally) -> None:
        handle.finished_at = time.time()
        if handle.state == "running":
            handle.state = outcome
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


def as_task_fn(fn: Callable[..., Any]) -> Callable[[TaskHandle], Any]:
    """Adapt ``fn()`` or ``fn(task)`` to the ``TaskRunner.run`` signature."""
    import inspect

    try:
        wants_handle = len(inspect.signature(fn).parameters) >= 1
    except (TypeError, ValueError):
        wants_handle = False
    return fn if wants_handle else (lambda _h: fn())


def dispatcher_for(state: Any, widget: Any) -> UiDispatcher:
    """The app's ``ctx.dispatcher`` when available, else a private ``UiDispatcher``
    attached to ``widget`` (created once per widget). Thread-safe ``post`` target."""
    ctx = getattr(state, "ctx", None)
    disp = getattr(ctx, "dispatcher", None) if ctx is not None else None
    if disp is not None:
        return disp
    disp = getattr(widget, "_simple_sim_dispatcher", None)
    if disp is None:
        disp = UiDispatcher()
        disp.attach(widget)
        try:
            widget._simple_sim_dispatcher = disp
        except Exception:
            pass
    return disp


def runner_for(state: Any, widget: Any) -> TaskRunner:
    """The app's ``ctx.tasks`` when available, else a private runner whose
    callbacks are marshalled onto ``widget``'s Tk loop by a ``UiDispatcher``.

    Never call ``widget.after`` from a worker thread; use this instead.
    """
    ctx = getattr(state, "ctx", None)
    tasks = getattr(ctx, "tasks", None) if ctx is not None else None
    if tasks is not None:
        return tasks
    runner = getattr(widget, "_simple_sim_task_runner", None)
    if runner is None:
        runner = TaskRunner(dispatcher_for(state, widget).post)
        try:
            widget._simple_sim_task_runner = runner
        except Exception:
            pass
    return runner
