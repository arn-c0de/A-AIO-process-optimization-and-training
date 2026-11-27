"""Markdown and SVG charts of the model Arena report (:mod:`simple_sim.arena` ranks, this module renders).

Pure stdlib; :func:`report_markdown` is the entry point, the ``_md_*`` helpers build one section each.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simple_sim.io_utils import format_bytes, path_size_bytes
from simple_sim.model_bundle import list_bundle_checkpoints, read_bundle_meta
from simple_sim.svg_charts import write_svg_barh, write_svg_heatmap, write_svg_pie_with_legend

if TYPE_CHECKING:
    from simple_sim.arena import BestByDataset, ModelRanking

__all__ = ["report_markdown", "write_charts"]


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
            profiles = sorted({p.stem.removesuffix("_last") for p in pts})
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


def _top_with_other(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Sort by size desc; keep the chart readable with top 8 + "Other" when more than 9."""
    items = sorted(((str(n), int(v)) for n, v in items if v is not None and int(v) > 0), key=lambda t: -t[1])
    if len(items) > 9:
        items = [*items[:8], ("Other", sum(v for _n, v in items[8:]))]
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


DatasetRows = dict[str, list[tuple[str, "BestByDataset"]]]


class _Anchors:
    """Unique HTML anchor ids of the report sections, datasets and models."""

    def __init__(self, dataset_order: list[str], model_order: list[str]) -> None:
        self._used: set[str] = set()
        self.warnings = self._unique("warnings")
        self.charts = self._unique("charts")
        self.overall = self._unique("overall-ranking")
        self.per_dataset = self._unique("per-dataset-breakdown")
        self.bundle = self._unique("bundle-details")
        self.per_model = self._unique("per-model-detail-cards")
        self.history = self._unique("history")
        self.dataset = {ds: self._unique(f"dataset-{ds}") for ds in dataset_order}
        self.model = {m: self._unique(f"model-{Path(m).name}") for m in model_order}

    def _unique(self, base: str) -> str:
        base = _anchor_slug(base)
        cand = base
        idx = 2
        while cand in self._used:
            cand = f"{base}-{idx}"
            idx += 1
        self._used.add(cand)
        return cand


def _section_head(anchor: str, title: str) -> list[str]:
    return [f'<a id="{anchor}"></a>', f"## {title}", ""]


def _fmt_float(value: Any, fmt: str) -> str:
    """``fmt.format(float(value))``, or ``"-"`` for ``None`` / a non-numeric value."""
    try:
        return fmt.format(float(value)) if value is not None else "-"
    except (TypeError, ValueError):
        return "-"


def _fmt_size(size_bytes: int | None) -> str:
    return format_bytes(int(size_bytes)) if size_bytes is not None else "-"


def _group_by_dataset(model_to_rows: dict[str, list[BestByDataset]]) -> DatasetRows:
    dataset_to_models: DatasetRows = {}
    for model_abs, rows in model_to_rows.items():
        for r in rows:
            dataset_to_models.setdefault(r.dataset_name, []).append((model_abs, r))
    return dataset_to_models


def _write_report_charts(
    assets_dir: Path,
    overall: list[ModelRanking],
    model_to_rows: dict[str, list[BestByDataset]],
    dataset_to_models: DatasetRows,
) -> list[tuple[str, str]]:
    """Chart inputs (dataset sizes, accuracy heatmap) from the ranking rows -> :func:`write_charts`."""
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

    return write_charts(
        assets_dir,
        overall,
        dataset_sizes=dataset_chart_items,
        heatmap_models=[Path(m).name for m in top_model_abs],
        heatmap_datasets=top_datasets,
        heatmap_values=[[acc_by_pair.get((m, ds)) for ds in top_datasets] for m in top_model_abs],
    )


def _md_navigation(
    anchors: _Anchors, dataset_order: list[str], model_order: list[str], *, has_warnings: bool, has_charts: bool
) -> list[str]:
    lines = ["## Quick Navigation", ""]
    if has_warnings:
        lines.append(f"- [Warnings](#{anchors.warnings})")
    if has_charts:
        lines.append(f"- [Charts](#{anchors.charts})")
    lines += [
        f"- [Overall Ranking](#{anchors.overall})",
        f"- [Per-Dataset Breakdown](#{anchors.per_dataset})",
        f"- [Bundle Details](#{anchors.bundle})",
        f"- [Per-Model Detail Cards](#{anchors.per_model})",
        f"- [History](#{anchors.history})",
        "",
    ]
    for summary, keys, ids, name_of in (
        ("Datasets", dataset_order, anchors.dataset, lambda k: k),
        ("Models", model_order, anchors.model, lambda k: Path(k).name),
    ):
        if not keys:
            continue
        lines += ["<details>", f"<summary>{summary}</summary>", ""]
        lines += [f"- [{_md_escape(name_of(k))}](#{ids[k]})" for k in keys]
        lines += ["", "</details>", ""]
    lines.append("")
    return lines


def _md_warnings(anchor: str, missing: list[str], limit: int = 20) -> list[str]:
    lines = [*_section_head(anchor, "Warnings"), "Tracked paths missing on disk:"]
    lines += [f"- `{rp}`" for rp in missing[:limit]]
    if len(missing) > limit:
        lines.append(f"- ... (+{len(missing) - limit} more)")
    lines.append("")
    return lines


def _md_charts(anchor: str, charts: list[tuple[str, str]]) -> list[str]:
    lines = _section_head(anchor, "Charts")
    for title, rel_path in charts:
        lines += [f"### {title}", "", f"![{_md_escape(title)}]({rel_path})", ""]
    return lines


def _md_overall(anchor: str, overall: list[ModelRanking]) -> list[str]:
    lines = [
        *_section_head(anchor, "Overall Ranking"),
        "| Rank | Model | Type | Avg Accuracy | Avg F1 | Datasets Tested | Best Dataset | Worst Dataset | Last Run |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
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
    return lines


def _md_per_dataset(anchors: _Anchors, dataset_order: list[str], dataset_to_models: DatasetRows) -> list[str]:
    lines = _section_head(anchors.per_dataset, "Per-Dataset Breakdown")
    for ds_name in dataset_order:
        rows = sorted(
            dataset_to_models[ds_name],
            key=lambda t: (-(-1.0 if t[1].accuracy is None else t[1].accuracy),
                           -(-1.0 if t[1].f1 is None else t[1].f1), -t[1].last_ts, Path(t[0]).name),
        )
        ds_size_b, ds_samples_n = _dataset_stats(rows)
        lines += [
            f'<a id="{anchors.dataset[ds_name]}"></a>',
            f"### Dataset: {ds_name}",
            f"- Size on disk: {format_bytes(ds_size_b) if ds_size_b > 0 else '-'}",
            f"- Total samples: {ds_samples_n if ds_samples_n > 0 else '-'}",
            "",
            "| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |",
            "|---:|---|---:|---:|---|---:|---:|---|",
        ]
        for i, (model_abs, r) in enumerate(rows, start=1):
            rank_s, model_s = _rank_cells(i, Path(model_abs).name)
            lines.append(
                _table_row(
                    [
                        rank_s,
                        model_s,
                        _fmt4(r.accuracy),
                        _fmt4(r.f1),
                        _md_escape(r.split),
                        str(r.seen) if r.seen is not None else "-",
                        _md_escape(_fmt_size(r.dataset_size_bytes)),
                        _md_escape(_fmt_ts(r.last_ts)),
                    ]
                )
            )
        lines.append("")
    return lines


def _md_bundles(anchor: str, overall: list[ModelRanking]) -> list[str]:
    lines = _section_head(anchor, "Bundle Details")
    bundles = [it for it in overall if it.type == "Bundle"]
    for it in bundles:
        bdir = Path(it.model_abs)
        det = _bundle_details(bdir)
        lines += [
            f"### {_md_escape(bdir.name)}",
            f"- **Profiles:** {', '.join(_md_escape(p) for p in det.get('profiles') or []) or '-'}",
            f"- **Checkpoints:** {det.get('checkpoints') or 0}",
            f"- **Created:** {_md_escape(str(det.get('created') or '-'))}",
            "",
        ]
    if not bundles:
        lines += ["- (No bundle models tracked)", ""]
    return lines


def _model_card_row(r: BestByDataset) -> str:
    epoch = r.extra.get("epoch")
    epoch_s = (
        str(int(epoch))
        if epoch is not None and str(epoch).isdigit()
        else (str(epoch) if epoch is not None else "-")
    )
    return _table_row(
        [
            _md_escape(r.dataset_name),
            _md_escape(r.split),
            _fmt4(r.accuracy),
            _fmt4(r.f1),
            _md_escape(str(r.extra.get("critical_fn") or "-")),
            str(r.seen) if r.seen is not None else "-",
            _md_escape(_fmt_size(r.dataset_size_bytes)),
            _md_escape(epoch_s),
            _fmt_float(r.extra.get("val_acc"), "{:.4f}"),
            _md_escape(_fmt_float(r.extra.get("img_per_s"), "{:.1f} img/s")),
        ]
    )


def _md_per_model(
    anchors: _Anchors, overall: list[ModelRanking], model_to_rows: dict[str, list[BestByDataset]]
) -> list[str]:
    lines = _section_head(anchors.per_model, "Per-Model Detail Cards")
    for it in overall:
        model_abs = it.model_abs
        rows_m = sorted(model_to_rows.get(model_abs, []), key=lambda r: r.dataset_name)
        lines += [
            f'<a id="{anchors.model[model_abs]}"></a>',
            f"### {_md_escape(Path(model_abs).name)} ({it.type})",
            "",
            "| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        lines += [_model_card_row(r) for r in rows_m]
        if not rows_m:
            lines.append("| - | - | - | - | - | - | - | - | - | - |")
        lines.append("")
    return lines


def report_markdown(
    model_to_rows: dict[str, list[BestByDataset]],
    overall: list[ModelRanking],
    *,
    missing: list[str],
    assets_dir: Path,
) -> str:
    """The report markdown of a ranking (:func:`simple_sim.arena.best_rows_by_model` /
    :func:`~simple_sim.arena.overall_ranking`); writes the charts to *assets_dir*."""
    dataset_to_models = _group_by_dataset(model_to_rows)
    charts = _write_report_charts(assets_dir, overall, model_to_rows, dataset_to_models)

    dataset_order = sorted(dataset_to_models.keys())
    model_order = [str(it.model_abs) for it in overall]
    anchors = _Anchors(dataset_order, model_order)
    now = datetime.now()

    lines: list[str] = ["# Model Arena Report", f"> Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines += _md_navigation(anchors, dataset_order, model_order, has_warnings=bool(missing), has_charts=bool(charts))
    if missing:
        lines += _md_warnings(anchors.warnings, missing)
    if charts:
        lines += _md_charts(anchors.charts, charts)
    lines += _md_overall(anchors.overall, overall)
    lines += _md_per_dataset(anchors, dataset_order, dataset_to_models)
    lines += _md_bundles(anchors.bundle, overall)
    lines += _md_per_model(anchors, overall, model_to_rows)
    lines += [
        *_section_head(anchors.history, "History"),
        f"- {now.strftime('%Y-%m-%d')}: Report generated ({len(overall)} models, {len(dataset_to_models)} datasets)",
        "",
    ]
    return "\n".join(lines)
