"""Tiny publish/subscribe bus so tabs never call each other directly (tk-free)."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class Topics:
    """Well-known topic names. Payloads are keyword arguments."""

    DATASET_CHANGED = "dataset.changed"            # path: Path | None
    DATASET_CATALOG_CHANGED = "dataset.catalog"    # datasets added/renamed/deleted
    MODEL_CHANGED = "model.changed"                # path: Path | None
    MODELS_CHANGED = "model.catalog"               # checkpoints added/removed
    THEME_CHANGED = "app.theme"                    # mode: "light" | "dark"
    STATUS = "app.status"                          # text: str, level: info|success|warning|error
    TASK_STARTED = "task.started"                  # task: TaskHandle
    TASK_PROGRESS = "task.progress"                # task: TaskHandle
    TASK_FINISHED = "task.finished"                # task: TaskHandle


Handler = Callable[..., Any]


class EventBus:
    """Synchronous bus. ``publish`` from a worker thread is marshalled to the UI
    thread when a ``post`` function (e.g. ``UiDispatcher.post``) is set."""

    def __init__(self, post: Callable[[Callable[[], None]], None] | None = None) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._post = post
        self._ui_thread = threading.get_ident()

    def set_post(self, post: Callable[[Callable[[], None]], None] | None) -> None:
        self._post = post

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subs[topic].remove(handler)
                except ValueError:
                    pass

        return unsubscribe

    def publish(self, topic: str, **payload: Any) -> None:
        if self._post is not None and threading.get_ident() != self._ui_thread:
            self._post(lambda: self._deliver(topic, payload))
            return
        self._deliver(topic, payload)

    def _deliver(self, topic: str, payload: dict[str, Any]) -> None:
        with self._lock:
            handlers = list(self._subs.get(topic, ()))
        for handler in handlers:
            try:
                handler(**payload)
            except Exception:
                # One broken subscriber must not break the others.
                import traceback

                traceback.print_exc()
