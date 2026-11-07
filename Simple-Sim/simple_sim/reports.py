"""Discovery and querying of evaluation reports (``report_*.json`` / ``batch_report_*.json``).

Shared by ``scripts/batch_predict.py``, ``scripts/arena_report.py`` and the GUI so that
report lookup and metric extraction behave identically everywhere. Pure stdlib.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
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


def load_reports(dirs: list[Path], model_stem: str | None = None) -> list[dict[str, Any]]:
    """Load valid reports (dicts with ``metrics``) from *dirs*, newest file first.

    With *model_stem*, only ``report_<stem>.json`` and ``batch_report_<stem>_*.json`` are read.
    Each report gets its source path under the ``_path`` key.

    Parsed files are cached per path and stat signature, so repeated scans (status, arena, GUI
    refreshes) only parse new or changed reports. Every call returns fresh top-level dicts; nested
    values are shared with the cache and must not be mutated.
    """
    globs = (f"report_{model_stem}.json", f"batch_report_{model_stem}_*.json") if model_stem else REPORT_GLOBS
    reports: list[dict[str, Any]] = []
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
                obj = dict(obj)
                obj["_path"] = sp
                reports.append(obj)
    return reports


def metric_value(report: dict[str, Any], key: str, critical_class: str | None = None) -> float | None:
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


def report_timestamp(report: dict[str, Any]) -> float:
    """Report time in epoch seconds: ``timestamp`` field, else file mtime, else ``0.0``."""
    ts = parse_iso_ts(report.get("timestamp"))
    if ts:
        return ts
    p = report.get("_path")
    if p:
        try:
            return os.path.getmtime(str(p))
        except OSError:
            pass
    return 0.0


def best_report(reports: list[dict[str, Any]], key: str, critical_class: str | None = None) -> dict[str, Any] | None:
    """Report with the highest value of *key* (``None`` if no report has it)."""
    scored = [(v, r) for r in reports if (v := metric_value(r, key, critical_class)) is not None]
    return max(scored, key=lambda t: t[0])[1] if scored else None
