"""Model "Arena" ranking report (markdown + static SVG charts).

Reads ``outputs/models/arena.json`` (tracked models), scans eval/batch reports
(see :mod:`simple_sim.reports`) and renders ``ARENA_REPORT.md`` plus charts in
``ARENA_REPORT_assets/``. Pure stdlib; used by ``scripts/arena_report.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from simple_sim.coerce import as_int
from simple_sim.io_utils import format_bytes, path_size_bytes, read_json, rel_to
from simple_sim.model_bundle import list_bundle_checkpoints, read_bundle_meta
from simple_sim.reports import Report, load_reports, metric_value, report_path, report_search_dirs, report_timestamp
from simple_sim.svg_charts import write_svg_barh, write_svg_heatmap, write_svg_pie_with_legend

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
    for k in sorted(str(k) for k in d.keys())[:3]:
        try:
            parts.append(f"{k}:{float(d.get(k)):.4f}")
        except (TypeError, ValueError):
            continue
    return " ".join(parts) if parts else "-"


def _bundle_details(bundle_dir: Path) -> dict[str, Any]:
    profiles: list[str] = []
    checkpoints = 0
    created = "-"
    try:
        meta = read_bundle_meta(bundle_dir)
        if meta is not None:
            profiles = sorted(meta.checkpoints.keys())
            checkpoints = len(meta.checkpoints)
            created = meta.created_at or "-"
        else:
            pts = [p for p in list_bundle_checkpoints(bundle_dir, include_last=True) if p.is_file()]
            checkpoints = len(pts)
            profiles = sorted({p.stem.replace("_last", "") for p in pts})
        if created == "-":
            created = datetime.fromtimestamp(bundle_dir.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        pass
    return {"profiles": profiles, "checkpoints": checkpoints, "created": created}


def _fmt_ts(ts: float) -> str:
    return "-" if ts <= 0 else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _anchor_slug(s: str) -> str:
    """URL-safe anchor slug (lowercase alnum, runs of other chars collapsed to '-')."""
    out: list[str] = []
    prev_dash = False
    for ch in (s or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-") or "section"


def _fmt4(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "-"


def _rank_cells(i: int, name: str) -> tuple[str, str]:
    """Rank label and (bold for top-3) markdown-escaped model name."""
    model_s = _md_escape(name)
    if i <= 3:
        return f"Top-{i}", f"**{model_s}**"
    return str(i), model_s


def _table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _match_tracked(tracked: dict[str, dict[str, Any]], model_abs: Path) -> str | None:
    """Tracked key for a report's model: the model itself or the bundle dir containing the ``.pt``."""
    key = str(model_abs)
    if key in tracked:
        return key
    parent = model_abs.parent
    if str(parent) in tracked and parent.is_dir() and model_abs.suffix == ".pt":
        return str(parent)
    return None


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
        ds_name = Path(ds_path).name if ds_path else "-"
        per_model_dataset.setdefault((matched, ds_name), []).append(r)

    ds_size_cache: dict[str, int] = {}
    model_to_rows: dict[str, list[BestByDataset]] = {}
    for (model_abs, ds_name), rs in per_model_dataset.items():
        # Rank on the test split when available; train/val reports would inflate "best" scores.
        # Reports without a split key (older eval.py) are test reports.
        rs = [rr for rr in rs if str(rr.get("split") or "test") == "test"] or rs
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


def _top_with_other(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Sort by size desc; keep the chart readable with top 8 + "Other" when more than 9."""
    items = sorted(((str(n), int(v)) for n, v in items if v is not None and int(v) > 0), key=lambda t: -t[1])
    if len(items) > 9:
        items = items[:8] + [("Other", sum(v for _n, v in items[8:]))]
    return items


def write_charts(
    assets_dir: Path,
    overall: list[ModelRanking],
    *,
    dataset_sizes: list[tuple[str, int]],
    heatmap_models: list[str],
    heatmap_datasets: list[str],
    heatmap_values: list[list[float | None]],
) -> list[tuple[str, str]]:
    """Write the SVG charts into *assets_dir*; return ``(title, md_relative_path)`` per chart."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    charts: list[tuple[str, str]] = []

    def add(title: str, filename: str) -> None:
        charts.append((title, f"{assets_dir.name}/{filename}"))

    # overall is already sorted by avg_acc desc; F1 needs its own (stable) sort.
    for metric_key, title, filename, sort_desc in (
        ("avg_acc", "Top Avg Accuracy", "top_avg_accuracy.svg", False),
        ("avg_f1", "Top Avg F1", "top_avg_f1.svg", True),
    ):
        items = [(it.model_name, getattr(it, metric_key)) for it in overall if getattr(it, metric_key) is not None]
        if sort_desc:
            items.sort(key=lambda t: -float(t[1]))
        items = items[:15]
        if items:
            write_svg_barh(
                assets_dir / filename,
                title=f"{title} (Top 15)",
                labels=[str(n) for n, _v in items],
                values=[float(v) for _n, v in items],
            )
            add(title, filename)

    model_sizes = []
    for it in overall:
        p = Path(str(it.model_abs))
        model_sizes.append((str(it.model_name or p.name), path_size_bytes(p)))
    size_items = _top_with_other(model_sizes)
    if size_items:
        write_svg_pie_with_legend(assets_dir / "model_size_pie.svg", title="Model Storage Breakdown", items=size_items)
        add("Model Storage Breakdown", "model_size_pie.svg")

    ds_items = _top_with_other(dataset_sizes)
    if ds_items:
        write_svg_pie_with_legend(assets_dir / "dataset_size_pie.svg", title="Dataset Storage Breakdown", items=ds_items)
        add("Dataset Storage Breakdown", "dataset_size_pie.svg")

    if heatmap_models and heatmap_datasets and heatmap_values:
        write_svg_heatmap(
            assets_dir / "accuracy_heatmap.svg",
            title="Accuracy Heatmap (Top Models x Datasets)",
            row_labels=heatmap_models,
            col_labels=heatmap_datasets,
            values=heatmap_values,
        )
        add("Accuracy Heatmap", "accuracy_heatmap.svg")
    return charts


def _dataset_stats(rows: list[tuple[str, BestByDataset]]) -> tuple[int, int]:
    """Max dataset size (bytes) and max sample count seen across a dataset's rows."""
    size = max((int(r.dataset_size_bytes) for _m, r in rows if r.dataset_size_bytes is not None), default=0)
    samples = max((int(r.seen) for _m, r in rows if r.seen is not None), default=0)
    return size, samples


def render_markdown(
    sim_root: Path,
    tracked: dict[str, dict[str, Any]],
    reports: list[Report] | list[Mapping[str, Any]],
    *,
    missing: list[str],
    assets_dir: Path | None = None,
) -> tuple[str, list[ModelRanking]]:
    """Render the report markdown (writing charts to *assets_dir*); return ``(markdown, overall_ranking)``."""
    assets_dir = assets_dir or (sim_root / "ARENA_REPORT_assets")
    model_to_rows = best_rows_by_model(sim_root, tracked, reports)
    overall = overall_ranking(sim_root, model_to_rows)

    dataset_to_models: dict[str, list[tuple[str, BestByDataset]]] = {}
    for model_abs, rows in model_to_rows.items():
        for r in rows:
            dataset_to_models.setdefault(r.dataset_name, []).append((model_abs, r))

    dataset_sizes_map: dict[str, int] = {}
    dataset_samples_map: dict[str, int] = {}
    for ds_name, rows in dataset_to_models.items():
        size, samples = _dataset_stats(rows)
        if size > 0:
            dataset_sizes_map[ds_name] = size
        if samples > 0:
            dataset_samples_map[ds_name] = samples

    dataset_chart_items: list[tuple[str, int]] = []
    for ds_name, size_b in sorted(dataset_sizes_map.items(), key=lambda t: -int(t[1])):
        samples = dataset_samples_map.get(ds_name, 0)
        dataset_chart_items.append((f"{ds_name} ({samples} samples)" if samples > 0 else ds_name, int(size_b)))

    # Heatmap: top models by overall ranking x top datasets by sample count.
    top_model_abs = [str(it.model_abs) for it in overall[:12]]
    top_datasets = [ds for ds, _cnt in sorted(dataset_samples_map.items(), key=lambda t: -int(t[1]))]
    if not top_datasets:
        top_datasets = sorted(dataset_to_models.keys())
    top_datasets = top_datasets[:12]
    acc_by_pair = {(m, r.dataset_name): r.accuracy for m, rows in model_to_rows.items() for r in rows}

    charts = write_charts(
        assets_dir,
        overall,
        dataset_sizes=dataset_chart_items,
        heatmap_models=[Path(m).name for m in top_model_abs],
        heatmap_datasets=top_datasets,
        heatmap_values=[[acc_by_pair.get((m, ds)) for ds in top_datasets] for m in top_model_abs],
    )

    dataset_order = sorted(dataset_to_models.keys())
    model_order = [str(it.model_abs) for it in overall]

    used_anchor_ids: set[str] = set()

    def unique_anchor(base: str) -> str:
        base = _anchor_slug(base)
        cand = base
        idx = 2
        while cand in used_anchor_ids:
            cand = f"{base}-{idx}"
            idx += 1
        used_anchor_ids.add(cand)
        return cand

    aid_warnings = unique_anchor("warnings")
    aid_charts = unique_anchor("charts")
    aid_overall = unique_anchor("overall-ranking")
    aid_per_dataset = unique_anchor("per-dataset-breakdown")
    aid_bundle = unique_anchor("bundle-details")
    aid_per_model = unique_anchor("per-model-detail-cards")
    aid_history = unique_anchor("history")
    dataset_anchor = {ds: unique_anchor(f"dataset-{ds}") for ds in dataset_order}
    model_anchor = {m: unique_anchor(f"model-{Path(m).name}") for m in model_order}

    lines: list[str] = []
    lines.append("# Model Arena Report")
    lines.append(f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Quick Navigation")
    lines.append("")
    if missing:
        lines.append(f"- [Warnings](#{aid_warnings})")
    if charts:
        lines.append(f"- [Charts](#{aid_charts})")
    lines.append(f"- [Overall Ranking](#{aid_overall})")
    lines.append(f"- [Per-Dataset Breakdown](#{aid_per_dataset})")
    lines.append(f"- [Bundle Details](#{aid_bundle})")
    lines.append(f"- [Per-Model Detail Cards](#{aid_per_model})")
    lines.append(f"- [History](#{aid_history})")
    lines.append("")

    for summary, keys, anchors, name_of in (
        ("Datasets", dataset_order, dataset_anchor, lambda k: k),
        ("Models", model_order, model_anchor, lambda k: Path(k).name),
    ):
        if not keys:
            continue
        lines.append("<details>")
        lines.append(f"<summary>{summary}</summary>")
        lines.append("")
        for k in keys:
            lines.append(f"- [{_md_escape(name_of(k))}](#{anchors[k]})")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append("")

    if missing:
        lines.append(f'<a id="{aid_warnings}"></a>')
        lines.append("## Warnings")
        lines.append("")
        lines.append("Tracked paths missing on disk:")
        for rp in missing[:20]:
            lines.append(f"- `{rp}`")
        if len(missing) > 20:
            lines.append(f"- ... (+{len(missing) - 20} more)")
        lines.append("")

    if charts:
        lines.append(f'<a id="{aid_charts}"></a>')
        lines.append("## Charts")
        lines.append("")
        for title, rel_path in charts:
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{_md_escape(title)}]({rel_path})")
            lines.append("")

    lines.append(f'<a id="{aid_overall}"></a>')
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Model | Type | Avg Accuracy | Avg F1 | Datasets Tested | Best Dataset | Worst Dataset | Last Run |")
    lines.append("|---:|---|---|---:|---:|---:|---|---|---|")
    for i, it in enumerate(overall, start=1):
        rank_s, model_s = _rank_cells(i, str(it.model_name))
        lines.append(
            _table_row(
                [
                    rank_s,
                    model_s,
                    str(it.type),
                    _fmt4(it.avg_acc),
                    _fmt4(it.avg_f1),
                    str(int(it.datasets)),
                    _md_escape(str(it.best_ds)),
                    _md_escape(str(it.worst_ds)),
                    _md_escape(_fmt_ts(float(it.last_ts or 0.0))),
                ]
            )
        )
    if not overall:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.append("")

    lines.append(f'<a id="{aid_per_dataset}"></a>')
    lines.append("## Per-Dataset Breakdown")
    lines.append("")
    for ds_name in dataset_order:
        rows = dataset_to_models[ds_name]
        rows.sort(key=lambda t: (-(t[1].accuracy or -1.0), -(t[1].f1 or -1.0), -t[1].last_ts, Path(t[0]).name))
        ds_size_b, ds_samples_n = _dataset_stats(rows)
        lines.append(f'<a id="{dataset_anchor[ds_name]}"></a>')
        lines.append(f"### Dataset: {ds_name}")
        lines.append(f"- Size on disk: {format_bytes(ds_size_b) if ds_size_b > 0 else '-'}")
        lines.append(f"- Total samples: {ds_samples_n if ds_samples_n > 0 else '-'}")
        lines.append("")
        lines.append("| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |")
        lines.append("|---:|---|---:|---:|---|---:|---:|---|")
        for i, (model_abs, r) in enumerate(rows, start=1):
            rank_s, model_s = _rank_cells(i, Path(model_abs).name)
            size_s = format_bytes(int(r.dataset_size_bytes)) if r.dataset_size_bytes is not None else "-"
            lines.append(
                _table_row(
                    [
                        rank_s,
                        model_s,
                        _fmt4(r.accuracy),
                        _fmt4(r.f1),
                        _md_escape(r.split),
                        str(r.seen) if r.seen is not None else "-",
                        _md_escape(size_s),
                        _md_escape(_fmt_ts(r.last_ts)),
                    ]
                )
            )
        lines.append("")

    lines.append(f'<a id="{aid_bundle}"></a>')
    lines.append("## Bundle Details")
    lines.append("")
    bundles = [it for it in overall if it.type == "Bundle"]
    for it in bundles:
        bdir = Path(it.model_abs)
        det = _bundle_details(bdir)
        lines.append(f"### {_md_escape(bdir.name)}")
        lines.append(f"- **Profiles:** {', '.join(_md_escape(p) for p in det.get('profiles') or []) or '-'}")
        lines.append(f"- **Checkpoints:** {det.get('checkpoints') or 0}")
        lines.append(f"- **Created:** {_md_escape(str(det.get('created') or '-'))}")
        lines.append("")
    if not bundles:
        lines.append("- (No bundle models tracked)")
        lines.append("")

    lines.append(f'<a id="{aid_per_model}"></a>')
    lines.append("## Per-Model Detail Cards")
    lines.append("")
    for it in overall:
        model_abs = it.model_abs
        rows_m = sorted(model_to_rows.get(model_abs, []), key=lambda r: r.dataset_name)
        lines.append(f'<a id="{model_anchor[model_abs]}"></a>')
        lines.append(f"### {_md_escape(Path(model_abs).name)} ({it.type})")
        lines.append("")
        lines.append("| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for r in rows_m:
            epoch = r.extra.get("epoch")
            val_acc = r.extra.get("val_acc")
            img_s = r.extra.get("img_per_s")
            epoch_s = (
                str(int(epoch))
                if epoch is not None and str(epoch).isdigit()
                else (str(epoch) if epoch is not None else "-")
            )
            try:
                val_acc_s = f"{float(val_acc):.4f}" if val_acc is not None else "-"
            except (TypeError, ValueError):
                val_acc_s = "-"
            try:
                img_s_s = f"{float(img_s):.1f} img/s" if img_s is not None else "-"
            except (TypeError, ValueError):
                img_s_s = "-"
            size_s = format_bytes(int(r.dataset_size_bytes)) if r.dataset_size_bytes is not None else "-"
            lines.append(
                _table_row(
                    [
                        _md_escape(r.dataset_name),
                        _md_escape(r.split),
                        _fmt4(r.accuracy),
                        _fmt4(r.f1),
                        _md_escape(str(r.extra.get("critical_fn") or "-")),
                        str(r.seen) if r.seen is not None else "-",
                        _md_escape(size_s),
                        _md_escape(epoch_s),
                        val_acc_s,
                        _md_escape(img_s_s),
                    ]
                )
            )
        if not rows_m:
            lines.append("| - | - | - | - | - | - | - | - | - | - |")
        lines.append("")

    lines.append(f'<a id="{aid_history}"></a>')
    lines.append("## History")
    lines.append("")
    lines.append(
        f"- {datetime.now().strftime('%Y-%m-%d')}: Report generated ({len(overall)} models, {len(dataset_to_models)} datasets)"
    )
    lines.append("")
    return "\n".join(lines), overall


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
        out_path.write_text(
            "# Model Arena Report\n"
            f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "No Arena-tracked models found.\n",
            encoding="utf-8",
        )
    else:
        reports = load_reports(report_search_dirs(sim_root))
        md, overall = render_markdown(
            sim_root, tracked, reports, missing=missing, assets_dir=out_path.parent / f"{out_path.stem}_assets"
        )
        out_path.write_text(md, encoding="utf-8")

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
