"""Discovery and querying of evaluation reports (``report_*.json`` / ``batch_report_*.json``).

Shared by ``scripts/batch_predict.py``, ``scripts/arena_report.py`` and the GUI so that
report lookup and metric extraction behave identically everywhere. Pure stdlib.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from simple_sim.io_utils import parse_iso_ts, read_json

REPORT_GLOBS = ("report_*.json", "batch_report_*.json")

#: Parsed report files kept per process (see :func:`_read_report`).
REPORT_CACHE_MAX = 4096
_REPORT_CACHE: OrderedDict[str, tuple[tuple[int, int, int], Any]] = OrderedDict()
_REPORT_LOCK = threading.Lock()


def _read_report(p: Path, st: os.stat_result) -> Any:
    """``read_json(p)``, reused while the file's inode, size and mtime are unchanged (plan S5)."""
    sp = str(p)
    sig = (st.st_ino, st.st_size, st.st_mtime_ns)
    with _REPORT_LOCK:
        hit = _REPORT_CACHE.get(sp)
        if hit is not None and hit[0] == sig:
            _REPORT_CACHE.move_to_end(sp)
            return hit[1]
    obj = read_json(p)
    with _REPORT_LOCK:
        _REPORT_CACHE[sp] = (sig, obj)
        _REPORT_CACHE.move_to_end(sp)
        while len(_REPORT_CACHE) > REPORT_CACHE_MAX:
            _REPORT_CACHE.popitem(last=False)
    return obj


def clear_report_cache() -> None:
    """Drop the parsed-report cache of :func:`load_reports`."""
    with _REPORT_LOCK:
        _REPORT_CACHE.clear()


PATH_KEY = "_path"


class Report(Mapping[str, Any]):
    """One evaluation report file: its *path* and its parsed JSON *data* (plan A2).

    Replaces the dicts with an injected ``"_path"`` key. For consumers that still index it like
    that dict, a report is a read-only :class:`~collections.abc.Mapping` over *data* that also
    answers ``report["_path"]`` / ``report.get("_path")`` with ``str(report.path)`` (iterated last,
    where the old key was appended; a ``"_path"`` key inside the file is shadowed, as before).
    ``dict(report)`` / :meth:`to_dict` give the old dict. New code uses ``report.path`` and
    ``report.data``.

    *data* is the parsed file shared with the :func:`load_reports` cache: it is exposed read-only
    at the top level, and nested values must not be mutated.
    """

    __slots__ = ("_data", "_path", "_path_str")

    def __init__(self, path: Path | str, data: Mapping[str, Any]) -> None:
        self._path = Path(path)
        self._path_str = str(path)
        self._data = data

    @property
    def path(self) -> Path:
        return self._path

    @property
    def data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._data) if isinstance(self._data, dict) else self._data

    def __getitem__(self, key: str) -> Any:
        if key == PATH_KEY:
            return self._path_str
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        for k in self._data:
            if k != PATH_KEY:
                yield k
        yield PATH_KEY

    def __len__(self) -> int:
        return len(self._data) + (PATH_KEY not in self._data)

    def __contains__(self, key: object) -> bool:
        return key == PATH_KEY or key in self._data

    def to_dict(self) -> dict[str, Any]:
        """The former ``load_reports`` dict: a shallow copy of *data* plus ``"_path"``."""
        out = dict(self._data)
        out[PATH_KEY] = self._path_str
        return out

    def __repr__(self) -> str:
        return f"Report(path={self._path_str!r}, keys={len(self._data)})"


def report_path(report: Mapping[str, Any]) -> str | None:
    """Source path of *report*: :attr:`Report.path` or the legacy ``"_path"`` key of a plain dict."""
    if isinstance(report, Report):
        return str(report.path)
    p = report.get(PATH_KEY)
    return str(p) if p else None


def report_search_dirs(sim_root: Path) -> list[Path]:
    """All directories where eval/batch reports are written, de-duplicated, in stable order."""
    sim_root = Path(sim_root)
    dirs: list[Path] = [sim_root / "outputs" / "models", sim_root / "outputs" / "models" / "history"]
    sim_data = sim_root / "outputs" / "sim_data"
    datasets: list[Path] = []
    runs = sim_data / "runs"
    versions = sim_data / "versions"
    if runs.is_dir():
        datasets.extend(sorted(d for d in runs.iterdir() if d.is_dir()))
    if versions.is_dir():
        datasets.extend(sorted(d for d in versions.glob("*/*") if d.is_dir()))
    for ds in datasets:
        dirs.append(ds / "predictions")
        dirs.append(ds / "predictions" / "weights_tab")

    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def load_reports(dirs: list[Path], model_stem: str | None = None) -> list[Report]:
    """Load valid reports (JSON objects with ``metrics``) from *dirs*, newest file first.

    With *model_stem*, only ``report_<stem>.json`` and ``batch_report_<stem>_*.json`` are read.
    Each item is a :class:`Report` (``report.path``, ``report.data``; still answers the legacy
    ``report["_path"]``).

    Parsed files are cached per path and stat signature, so repeated scans (status, arena, GUI
    refreshes) only parse new or changed reports. Every call returns fresh :class:`Report`
    objects over the cached data, which is read-only (see :class:`Report`).
    """
    globs = (f"report_{model_stem}.json", f"batch_report_{model_stem}_*.json") if model_stem else REPORT_GLOBS
    reports: list[Report] = []
    seen: set[str] = set()
    for d in dirs:
        if not Path(d).is_dir():
            continue
        stats = {p: p.stat() for g in globs for p in Path(d).glob(g)}
        for p in sorted(stats, key=lambda x: stats[x].st_mtime, reverse=True):
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            obj = _read_report(p, stats[p])
            if isinstance(obj, dict) and "metrics" in obj:
                reports.append(Report(sp, obj))
    return reports


def metric_value(report: Mapping[str, Any], key: str, critical_class: str | None = None) -> float | None:
    """Scalar metric from a report: any top-level metric key, or ``critical_fn_rate`` for *critical_class*."""
    metrics = report.get("metrics") or {}
    try:
        if key == "critical_fn_rate":
            if not critical_class:
                return None
            v = (metrics.get("critical_fn_rates") or {}).get(critical_class)
        else:
            v = metrics.get(key)
        return None if v is None else float(v)
    except (TypeError, ValueError, AttributeError):
        return None


def report_timestamp(report: Mapping[str, Any]) -> float:
    """Report time in epoch seconds: ``timestamp`` field, else file mtime, else ``0.0``."""
    ts = parse_iso_ts(report.get("timestamp"))
    if ts:
        return ts
    p = report_path(report)
    if p:
        try:
            return os.path.getmtime(str(p))
        except OSError:
            pass
    return 0.0


def best_report(reports: list[Report], key: str, critical_class: str | None = None) -> Report | None:
    """Report with the highest value of *key* (``None`` if no report has it)."""
    scored = [(v, r) for r in reports if (v := metric_value(r, key, critical_class)) is not None]
    return max(scored, key=lambda t: t[0])[1] if scored else None
