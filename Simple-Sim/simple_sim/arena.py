"""Model "Arena" ranking report (markdown + static SVG charts).

Reads ``outputs/models/arena.json`` (tracked models), scans eval/batch reports
(see :mod:`simple_sim.reports`), ranks the models and writes ``ARENA_REPORT.md`` plus charts in
``ARENA_REPORT_assets/`` (rendered by :mod:`simple_sim.arena_markdown`). Pure stdlib; used by
``scripts/arena_report.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from simple_sim.arena_markdown import report_markdown
from simple_sim.coerce import as_int
from simple_sim.io_utils import atomic_write_text, path_size_bytes, read_json, rel_to
from simple_sim.reports import Report, is_partial_report, load_reports, metric_value, report_path, report_search_dirs, report_timestamp

REPORT_NAME = "ARENA_REPORT.md"


@dataclass
class BestByDataset:
    """Best (test-split preferred) report of one tracked model on one dataset."""

    dataset_name: str
    dataset_path: str
    split: str
    accuracy: float | None
    f1: float | None
    seen: int | None
    dataset_size_bytes: int | None
    last_ts: float
    report_path: str
    extra: dict[str, Any]


@dataclass(frozen=True)
class ModelRanking:
    """One row of the overall ranking (:func:`overall_ranking`): a tracked model's averages."""

    model_abs: str
    model_rel: str
    model_name: str
    type: str  # "Bundle" | "Single"
    avg_acc: float | None
    avg_f1: float | None
    datasets: int
    best_ds: str
    worst_ds: str
    last_ts: float

    def to_dict(self) -> dict[str, Any]:
        """The former ranking dict (same keys, same order)."""
        return asdict(self)


#: Keys of a ranking in :func:`write_arena_report`'s ``top`` summary.
TOP_KEYS = ("model_name", "model_rel", "type", "avg_acc", "avg_f1", "datasets")


def detect_sim_root(start: Path) -> Path | None:
    """Walk up from *start* to the first directory containing ``outputs/models`` and ``scripts``."""
    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "outputs" / "models").exists() and (p / "scripts").exists():
            return p
    return None


def _resolve(sim_root: Path, path: str) -> Path | None:
    """Absolute, resolved path for a (possibly sim-root-relative) path string; ``None`` if empty."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = sim_root / p
    try:
        return p.resolve()
    except OSError:
        return p


def load_arena(sim_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return ``(tracked, missing)``: tracked entries keyed by absolute model path, and missing paths."""
    obj = read_json(sim_root / "outputs" / "models" / "arena.json")
    items = obj.get("arena_models") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return {}, []

    tracked: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rp = str(it.get("path") or "").strip()
        abs_p = _resolve(sim_root, rp)
        if abs_p is None:
            continue
        if not abs_p.exists():
            missing.append(rp)
        tracked[str(abs_p)] = dict(it)
    return tracked, missing


def _critical_fn_summary(report: Mapping[str, Any]) -> str:
    d = (report.get("metrics") or {}).get("critical_fn_rates") or {}
    if not isinstance(d, dict) or not d:
        return "-"
    parts = []
    for k in sorted(str(k) for k in d)[:3]:
        try:
            parts.append(f"{k}:{float(d.get(k)):.4f}")
        except (TypeError, ValueError):
            continue
    return " ".join(parts) if parts else "-"


def _match_tracked(tracked: dict[str, dict[str, Any]], model_abs: Path) -> str | None:
    """Tracked key for a report's model: the model itself or the bundle dir containing the ``.pt``."""
    key = str(model_abs)
    if key in tracked:
        return key
    parent = model_abs.parent
    if str(parent) in tracked and parent.is_dir() and model_abs.suffix == ".pt":
        return str(parent)
    return None


def dataset_display_name(ds_path: str) -> str:
    """Arena name of a dataset path: its dir name, ``<name>/<version>`` for a snapshot under ``versions/``.

    Snapshots of different datasets share version names (``v1``); the bare dir name merged them.
    """
    if not ds_path:
        return "-"
    p = Path(ds_path)
    if p.parent.parent.name == "versions" and p.parent.name:
        return f"{p.parent.name}/{p.name}"
    return p.name


def best_rows_by_model(
    sim_root: Path, tracked: dict[str, dict[str, Any]], reports: list[Report] | list[Mapping[str, Any]]
) -> dict[str, list[BestByDataset]]:
    """Best report per (tracked model, dataset name), grouped by tracked model key.

    *reports* are :class:`~simple_sim.reports.Report` objects (or legacy dicts with ``"_path"``).
    """
    per_model_dataset: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in reports:
        abs_mp = _resolve(sim_root, str(r.get("model_path") or "").strip())
        if abs_mp is None:
            continue
        matched = _match_tracked(tracked, abs_mp)
        if matched is None:
            continue
        ds_path = str(r.get("dataset_path") or "").strip()
        ds_name = dataset_display_name(ds_path)
        per_model_dataset.setdefault((matched, ds_name), []).append(r)

    ds_size_cache: dict[str, int] = {}
    model_to_rows: dict[str, list[BestByDataset]] = {}
    for (model_abs, ds_name), rs in per_model_dataset.items():
        # Rank on the test split when available; train/val reports would inflate "best" scores.
        # Reports without a split key (older eval.py) are test reports.
        rs = [rr for rr in rs if str(rr.get("split") or "test") == "test"] or rs
        # A --max-samples quick check (or an eval with excluded samples) must not outrank a full run.
        rs = [rr for rr in rs if not is_partial_report(rr)] or rs
        # Timestamps parsed once per report (ISO parse / stat), not inside the sort key (plan S5).
        stamps = [report_timestamp(rr) for rr in rs]

        def key(i: int) -> tuple[float, float, float]:
            acc = metric_value(rs[i], "accuracy")
            f1 = metric_value(rs[i], "macro_f1")
            return (acc if acc is not None else -1.0, f1 if f1 is not None else -1.0, stamps[i])

        best = rs[max(range(len(rs)), key=key)]
        ds_path_s = str(best.get("dataset_path") or "")
        ds_size_b: int | None = None
        ds_abs = _resolve(sim_root, ds_path_s)
        if ds_abs is not None:
            ds_key = str(ds_abs)
            if ds_key not in ds_size_cache:
                ds_size_cache[ds_key] = path_size_bytes(ds_abs)
            ds_size_b = ds_size_cache[ds_key]
        model_to_rows.setdefault(model_abs, []).append(
            BestByDataset(
                dataset_name=ds_name,
                dataset_path=ds_path_s,
                split=str(best.get("split") or "test"),
                accuracy=metric_value(best, "accuracy"),
                f1=metric_value(best, "macro_f1"),
                seen=as_int(best.get("seen_samples") or best.get("test_size"), None),
                dataset_size_bytes=ds_size_b,
                last_ts=max(stamps),
                report_path=report_path(best) or "",
                extra={
                    "critical_fn": _critical_fn_summary(best),
                    "epoch": best.get("checkpoint_epoch") or best.get("epoch"),
                    "val_acc": best.get("checkpoint_val_accuracy") or best.get("val_accuracy"),
                    "img_per_s": best.get("img_per_s"),
                },
            )
        )
    return model_to_rows


def overall_ranking(sim_root: Path, model_to_rows: dict[str, list[BestByDataset]]) -> list[ModelRanking]:
    """Per-model averages, sorted by avg accuracy, avg F1, recency, then name."""
    overall: list[ModelRanking] = []
    for model_abs, rows in model_to_rows.items():
        with_acc = [r for r in rows if r.accuracy is not None]
        f1s = [r.f1 for r in rows if r.f1 is not None]
        best_ds = worst_ds = "-"
        if with_acc:
            best_r = max(with_acc, key=lambda r: float(r.accuracy))
            worst_r = min(with_acc, key=lambda r: float(r.accuracy))
            best_ds = f"{best_r.dataset_name} ({best_r.accuracy:.4f})"
            worst_ds = f"{worst_r.dataset_name} ({worst_r.accuracy:.4f})"
        p = Path(model_abs)
        overall.append(
            ModelRanking(
                model_abs=model_abs,
                model_rel=rel_to(sim_root, p),
                model_name=p.name,
                type="Bundle" if p.is_dir() else "Single",
                avg_acc=sum(r.accuracy for r in with_acc) / len(with_acc) if with_acc else None,
                avg_f1=sum(f1s) / len(f1s) if f1s else None,
                datasets=len({r.dataset_name for r in rows}),
                best_ds=best_ds,
                worst_ds=worst_ds,
                last_ts=max((r.last_ts for r in rows), default=0.0),
            )
        )
    overall.sort(
        key=lambda it: (
            -(float(it.avg_acc) if it.avg_acc is not None else -1.0),
            -(float(it.avg_f1) if it.avg_f1 is not None else -1.0),
            -float(it.last_ts or 0.0),
            str(it.model_name),
        )
    )
    return overall


def render_markdown(
    sim_root: Path,
    tracked: dict[str, dict[str, Any]],
    reports: list[Report] | list[Mapping[str, Any]],
    *,
    missing: list[str],
    assets_dir: Path | None = None,
) -> tuple[str, list[ModelRanking]]:
    """Render the report markdown (writing charts to *assets_dir*); return ``(markdown, overall_ranking)``."""
    model_to_rows = best_rows_by_model(sim_root, tracked, reports)
    overall = overall_ranking(sim_root, model_to_rows)
    md = report_markdown(model_to_rows, overall, missing=missing, assets_dir=assets_dir or (sim_root / "ARENA_REPORT_assets"))
    return md, overall


def write_arena_report(sim_root: Path, out_path: Path | None = None, *, top_n: int = 5) -> dict[str, Any]:
    """Write the Arena report for *sim_root* (default ``<sim_root>/ARENA_REPORT.md``).

    Charts go to ``<out_dir>/<out_stem>_assets/``. Returns a summary with ``out_path``,
    ``n_models``, ``n_missing`` and the ``top`` ranked models (name/type/avg_acc/avg_f1/datasets).
    """
    sim_root = Path(sim_root)
    out_path = Path(out_path) if out_path else sim_root / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracked, missing = load_arena(sim_root)
    overall: list[ModelRanking] = []
    if not tracked:
        atomic_write_text(
            out_path,
            "# Model Arena Report\n"
            f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "No Arena-tracked models found.\n",
        )
    else:
        reports = load_reports(report_search_dirs(sim_root))
        md, overall = render_markdown(
            sim_root, tracked, reports, missing=missing, assets_dir=out_path.parent / f"{out_path.stem}_assets"
        )
        atomic_write_text(out_path, md)

    return {
        "out_path": str(out_path),
        "n_tracked": len(tracked),
        "n_models": len(overall),
        "n_missing": len(missing),
        "top": [
            {k: getattr(it, k) for k in TOP_KEYS}
            for it in overall[:top_n]
        ],
    }
