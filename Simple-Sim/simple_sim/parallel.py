"""Ordered, bounded parallel map for CPU-bound image rendering (threads or processes).

Every sample is fully determined by its own seeded ``rng``, so rendering samples in parallel gives
the same images as a sequential loop, whatever the executor.

* **Threads** (default of :func:`imap_ordered`): no pickling, usable with closures and from the Tk
  GUI. The renderer releases the GIL in numpy/OpenCV, but its Python parts do not, so threads stop
  scaling at about 4-8 workers.
* **Processes** (``processes=True``): scale with the cores; *fn* and the items must be picklable
  (module-level function). Workers are started with ``forkserver`` (``spawn`` where unavailable),
  never ``fork``, so a multi-threaded parent cannot deadlock them. Dataset generation uses them for
  large runs (:func:`render_executor`).

OpenCV's own thread pool is switched off in the workers so they do not oversubscribe the CPU.
``SIMPLE_SIM_RENDER_WORKERS`` overrides the worker count (``1`` = sequential),
``SIMPLE_SIM_RENDER_EXECUTOR`` (``thread`` | ``process``) the executor of dataset generation.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

MAX_DEFAULT_WORKERS = 8  # measured: 2D rendering stops scaling at ~8 threads (GIL-bound Python parts)
MAX_DEFAULT_PROCESSES = 16
EXECUTORS = ("thread", "process")
#: Below this many items per worker the start-up of worker processes (~1 s) costs more than it saves.
PROCESS_MIN_ITEMS_PER_WORKER = 8

_CV2_LOCK = threading.Lock()
_CV2_USERS = 0
_CV2_PREVIOUS: int | None = None


def render_workers(requested: int | None = None) -> int:
    """Worker count: *requested* if > 0, else ``$SIMPLE_SIM_RENDER_WORKERS``, else ``min(8, cpu count)``."""
    if requested is not None and int(requested) > 0:
        return int(requested)
    env = os.environ.get("SIMPLE_SIM_RENDER_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(MAX_DEFAULT_WORKERS, os.cpu_count() or 1))


def render_processes(requested: int | None = None) -> int:
    """Process count: *requested* if > 0, else ``$SIMPLE_SIM_RENDER_WORKERS``, else ``min(16, cpu count)``."""
    if requested is not None and int(requested) > 0:
        return int(requested)
    env = os.environ.get("SIMPLE_SIM_RENDER_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(MAX_DEFAULT_PROCESSES, os.cpu_count() or 1))


def render_executor(n_items: int, requested: str | None = None) -> str:
    """``thread`` or ``process`` for rendering *n_items* images.

    *requested* (or ``$SIMPLE_SIM_RENDER_EXECUTOR``) wins when set; otherwise processes are used when
    there are enough items to amortise their start-up (:data:`PROCESS_MIN_ITEMS_PER_WORKER`).
    """
    choice = (requested or os.environ.get("SIMPLE_SIM_RENDER_EXECUTOR", "")).strip().lower()
    if choice:
        if choice not in EXECUTORS:
            raise ValueError(f"render executor must be one of {EXECUTORS}, got {choice!r}")
        return choice
    n = render_processes()
    return "process" if n > 1 and n_items >= n * PROCESS_MIN_ITEMS_PER_WORKER else "thread"


def _mp_context() -> multiprocessing.context.BaseContext:
    """``forkserver`` if it can run here, else ``spawn`` (e.g. where Unix sockets are not allowed)."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        try:
            from multiprocessing import forkserver

            forkserver.ensure_running()
            return multiprocessing.get_context("forkserver")
        except OSError:
            pass
    return multiprocessing.get_context("spawn")


def _process_worker_init() -> None:
    import cv2

    cv2.setNumThreads(1)


@contextmanager
def _cv2_single_threaded(active: bool) -> Iterator[None]:
    """``cv2.setNumThreads(1)`` while at least one parallel map runs (reference counted)."""
    global _CV2_USERS, _CV2_PREVIOUS
    if not active:
        yield
        return
    import cv2

    with _CV2_LOCK:
        if _CV2_USERS == 0:
            _CV2_PREVIOUS = cv2.getNumThreads()
            cv2.setNumThreads(1)
        _CV2_USERS += 1
    try:
        yield
    finally:
        with _CV2_LOCK:
            _CV2_USERS -= 1
            if _CV2_USERS == 0 and _CV2_PREVIOUS is not None:
                cv2.setNumThreads(_CV2_PREVIOUS)
                _CV2_PREVIOUS = None


def imap_ordered(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int | None = None,
    prefetch: int = 2,
    should_stop: Callable[[], bool] | None = None,
    processes: bool = False,
) -> Iterator[R]:
    """Yield ``fn(item)`` for every item, in input order, computed by a thread (or process) pool.

    At most ``workers * prefetch`` results are in flight, so memory stays bounded for big datasets.
    With ``workers == 1`` it is a plain loop. When *should_stop* returns true no further items are
    submitted; already running ones finish, queued ones are cancelled and the iterator ends.
    An exception in ``fn`` is re-raised at its position (later items are cancelled).
    ``processes=True`` runs *fn* in worker processes (*fn* and items must be picklable).
    """
    n = render_processes(workers) if processes else render_workers(workers)
    stop = should_stop or (lambda: False)
    if n <= 1:
        for item in items:
            if stop():
                return
            yield fn(item)
        return

    it = iter(items)
    pending: deque[Future[R]] = deque()
    limit = n * max(1, int(prefetch))
    pool: Executor
    if processes:
        pool = ProcessPoolExecutor(max_workers=n, mp_context=_mp_context(), initializer=_process_worker_init)
    else:
        pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="render")
    with (nullcontext() if processes else _cv2_single_threaded(True)), pool as ex:
        try:
            exhausted = False
            while True:
                while not exhausted and len(pending) < limit and not stop():
                    try:
                        item = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(ex.submit(fn, item))
                if not pending:
                    return
                yield pending.popleft().result()
                if stop():
                    return
        finally:
            for f in pending:
                f.cancel()
