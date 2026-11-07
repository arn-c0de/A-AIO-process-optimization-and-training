"""Stat-first file caches (tk-free, stdlib only).

``FileCache`` keeps a value derived from one or more files and recomputes it only when one of the
files' ``stat`` signature (inode, size, mtime, ctime) changed. A cache hit costs one ``os.stat`` per
file and never opens it; that is what the GUI needs for manifests, label files and YAML configs it
used to re-read on every selection / poll (plan item S6).

Values are shared between callers: treat them as read-only. Loader errors propagate and are not
cached; a missing file has the signature ``None`` (so "file appeared" invalidates the entry too).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any

Signature = tuple[int, int, int, int]


def file_signature(path: Path | str) -> Signature | None:
    """``(inode, size, mtime_ns, ctime_ns)`` of ``path`` or ``None`` if it cannot be stat'ed."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class FileCache:
    """LRU of ``key -> (signatures of the dependency files, value)``; thread-safe."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max(1, int(max_entries))
        self._data: OrderedDict[Hashable, tuple[tuple[Signature | None, ...], Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def cached(self, key: Hashable, paths: Iterable[Path | str], load: Callable[[], Any]) -> Any:
        """Value for ``key``; ``load()`` runs (outside the lock) when any of ``paths`` changed."""
        sig = tuple(file_signature(p) for p in paths)
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and entry[0] == sig:
                self._data.move_to_end(key)
                self.hits += 1
                return entry[1]
            self.misses += 1
        value = load()
        with self._lock:
            self._data[key] = (sig, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
        return value

    def get(self, path: Path | str, load: Callable[[Path], Any], *, missing: Any = None) -> Any:
        """``load(path)`` cached on ``path``'s signature; ``missing`` when the file does not exist."""
        p = Path(path)
        if file_signature(p) is None:
            self.invalidate(("file", str(p)))
            return missing
        return self.cached(("file", str(p)), (p,), lambda: load(p))

    def invalidate(self, key: Hashable | None = None) -> None:
        with self._lock:
            if key is None:
                self._data.clear()
            else:
                self._data.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
