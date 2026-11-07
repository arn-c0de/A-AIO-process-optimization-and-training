"""Parsed ``meta.jsonl`` / ``labels.jsonl`` / ``splits/*.txt`` of a dataset, read once per run.

:class:`DatasetIndex` is the shared read side of a dataset directory (plan item L11). The files
are parsed with the same validation as :func:`simple_sim.rows.read_jsonl`,
:func:`simple_sim.splits.read_split` and :func:`simple_sim.manifest.load_dataset_manifest`, and the result is kept in a small per-process cache keyed by
the file's path and ``stat`` signature (inode, size, mtime, ctime). A rewritten, appended or
replaced file is therefore parsed again on the next access, while e.g. ``batch_predict --split all``
(three datasets, ``dataset_info``, the profile ground truth) parses each JSONL once.

Rows are shared between all users of the cache: treat them, the returned tuples and the
``id_to_*`` mappings as read-only. Pure stdlib + rows/manifest; no torch/cv2.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from simple_sim.dataset_files import DATASET_MANIFEST, LABELS_JSONL, META_JSONL, SPLITS_DIR
from simple_sim.manifest import DatasetManifest, load_dataset_manifest
from simple_sim.rows import LabelRow, MetaRow, read_jsonl
from simple_sim.splits import read_split

SPLIT_NAMES = ("train", "val", "test")

#: Parsed files kept per process (each dataset contributes meta, labels and up to 3 split files).
CACHE_MAX_FILES = 32

_CACHE: OrderedDict[tuple[str, str], tuple[tuple[int, int, int, int], Any]] = OrderedDict()
_LOCK = threading.Lock()


def _signature(path: Path) -> tuple[int, int, int, int]:
    st = os.stat(path)
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _cached(path: Path, kind: str, load: Callable[[Path], Any]) -> Any:
    """``load(path)``, reused while the file's stat signature is unchanged.

    A missing file raises ``FileNotFoundError`` from the loader, as without the cache.
    """
    key = (str(Path(path).absolute()), kind)
    try:
        sig = _signature(path)
    except OSError:
        with _LOCK:
            _CACHE.pop(key, None)
        return load(path)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == sig:
            _CACHE.move_to_end(key)
            return hit[1]
    value = load(path)
    with _LOCK:
        _CACHE[key] = (sig, value)
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_MAX_FILES:
            _CACHE.popitem(last=False)
    return value


def clear_cache() -> None:
    """Drop all cached files (tests; long-running processes that want the memory back)."""
    with _LOCK:
        _CACHE.clear()


def _load_meta(path: Path) -> tuple[tuple[MetaRow, ...], Mapping[str, MetaRow]]:
    rows = tuple(read_jsonl(path, MetaRow))
    # Same last-row-wins semantics as ``{row.id: row for row in rows}`` at the former call sites.
    # A plain dict (not a MappingProxyType) so datasets holding it stay picklable for DataLoader
    # workers; callers must not mutate it.
    return rows, {row.id: row for row in rows}


def _load_labels(path: Path) -> tuple[LabelRow, ...]:
    return tuple(read_jsonl(path, LabelRow))


def _load_split(path: Path) -> tuple[str, ...]:
    return tuple(read_split(path))


class DatasetIndex:
    """Read-only view of one dataset directory; every accessor goes through the per-file cache.

    Construction is free; files are parsed on first access, so a missing file only raises where it
    is needed (``FileNotFoundError`` from the same readers as before).
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)

    @property
    def meta_path(self) -> Path:
        return self.data_dir / META_JSONL

    @property
    def labels_path(self) -> Path:
        return self.data_dir / LABELS_JSONL

    def split_path(self, split: str) -> Path:
        return self.data_dir / SPLITS_DIR / f"{split}.txt"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / DATASET_MANIFEST

    def manifest(self) -> DatasetManifest:
        """Validated ``dataset_manifest.json`` (``FileNotFoundError`` / ``ValueError`` as
        :func:`~simple_sim.manifest.read_dataset_manifest`); one shared, read-only instance per
        file version."""
        return _cached(self.manifest_path, "manifest", load_dataset_manifest)

    def profile_ids(self) -> tuple[str, ...]:
        """Profile ids declared by the manifest (see :attr:`DatasetManifest.profile_ids`)."""
        return self.manifest().profile_ids

    def meta_rows(self) -> tuple[MetaRow, ...]:
        return _cached(self.meta_path, "meta", _load_meta)[0]

    def id_to_meta(self) -> Mapping[str, MetaRow]:
        return _cached(self.meta_path, "meta", _load_meta)[1]

    def label_rows(self) -> tuple[LabelRow, ...]:
        return _cached(self.labels_path, "labels", _load_labels)

    def split_ids(self, split: str) -> tuple[str, ...]:
        """Sample ids of ``splits/<split>.txt`` (``FileNotFoundError`` if absent)."""
        return _cached(self.split_path(split), "split", _load_split)

    def split_sizes(self) -> dict[str, int]:
        """``{train, val, test}`` -> number of ids (0 for a missing split file)."""
        sizes: dict[str, int] = {}
        for s in SPLIT_NAMES:
            try:
                sizes[s] = len(self.split_ids(s))
            except FileNotFoundError:
                sizes[s] = 0
        return sizes

    def class_distribution(self) -> dict[str, int]:
        """``class_name`` -> label count, in first-seen order of ``labels.jsonl``."""
        dist: dict[str, int] = {}
        for r in self.label_rows():
            dist[r.class_name] = dist.get(r.class_name, 0) + 1
        return dist

    def profile_map(self) -> dict[str, str]:
        """Sample id -> ``profile_id`` for labels that have one (v2 labels)."""
        return {lr.id: lr.profile_id for lr in self.label_rows() if lr.profile_id}
