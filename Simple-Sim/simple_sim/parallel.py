"""Ordered, bounded thread-pool map for CPU-bound image rendering.

The OpenCV 2D renderer is numpy/OpenCV heavy and releases the GIL for most of its work, and every
sample is fully determined by its own seeded ``rng``, so rendering samples in threads gives the
same images as a sequential loop. Threads (not processes) keep this usable from the Tk GUI and
avoid pickling large configs; OpenCV's own thread pool is switched off meanwhile so the workers do
not oversubscribe the CPU.

Set ``SIMPLE_SIM_RENDER_WORKERS`` to override the worker count (``1`` = sequential).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

MAX_DEFAULT_WORKERS = 8  # measured: 2D rendering stops scaling at ~8 threads (GIL-bound Python parts)

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
) -> Iterator[R]:
    """Yield ``fn(item)`` for every item, in input order, computed by a thread pool.

    At most ``workers * prefetch`` results are in flight, so memory stays bounded for big datasets.
    With ``workers == 1`` it is a plain loop. When *should_stop* returns true no further items are
    submitted; already running ones finish, queued ones are cancelled and the iterator ends.
    An exception in ``fn`` is re-raised at its position (later items are cancelled).
    """
    n = render_workers(workers)
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
    with _cv2_single_threaded(True), ThreadPoolExecutor(max_workers=n, thread_name_prefix="render") as ex:
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
