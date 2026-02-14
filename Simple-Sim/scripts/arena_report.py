#!/usr/bin/env python3
"""Generate a markdown "Arena" ranking report for tracked models.

Reads:
  outputs/models/arena.json

Scans:
  report_*.json, batch_report_*.json in common outputs + per-dataset predictions dirs

Writes:
  ARENA_REPORT.md (in sim root)
  ARENA_REPORT_assets/*.svg (static charts referenced by the markdown)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _detect_sim_root(start: Path) -> Optional[Path]:
    """Walk up from start until we find a directory that looks like a Simple-Sim root."""
    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "outputs" / "models").exists() and (p / "scripts").exists():
            return p
    return None


def _report_search_dirs(sim_root: Path) -> List[Path]:
    def add_dataset_prediction_dirs(dirs: List[Path], ds: Path) -> None:
        dirs.append(ds / "predictions")
        dirs.append(ds / "predictions" / "weights_tab")

    dirs: List[Path] = []
    dirs.append(sim_root / "outputs" / "models")
    dirs.append(sim_root / "outputs" / "models" / "history")

    sim_data = sim_root / "outputs" / "sim_data"
    runs = sim_data / "runs"
    versions = sim_data / "versions"
    try:
        if runs.exists():
            for ds in runs.iterdir():
                if ds.is_dir():
                    add_dataset_prediction_dirs(dirs, ds)
    except Exception:
        pass
    try:
        if versions.exists():
            for ds in versions.glob("*/*"):
                if ds.is_dir():
                    add_dataset_prediction_dirs(dirs, ds)
    except Exception:
        pass

    out: List[Path] = []
    seen = set()
    for d in dirs:
        try:
            rp = str(d.resolve())
        except Exception:
            rp = str(d)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(d)
    return out


def _load_reports(dirs: List[Path]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    seen_paths = set()
    for d in dirs:
        if not d.exists():
            continue
        cand: List[Path] = []
        cand.extend(d.glob("report_*.json"))
        cand.extend(d.glob("batch_report_*.json"))
        for p in sorted(cand, key=lambda x: x.stat().st_mtime if x.exists() else 0.0, reverse=True):
            sp = str(p)
            if sp in seen_paths:
                continue
            seen_paths.add(sp)
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict) or "metrics" not in obj:
                continue
            obj["_path"] = sp
            reports.append(obj)
    return reports


def _metric(report: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = (report.get("metrics") or {}).get(key)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _ts(report: Dict[str, Any]) -> float:
    s = str(report.get("timestamp") or "")
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        pass
    p = report.get("_path")
    if p:
        try:
            return os.path.getmtime(str(p))
        except Exception:
            pass
    return 0.0


def _fmt_ts(ts: float) -> str:
    if ts <= 0:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _basename(p: Any) -> str:
    try:
        return Path(str(p)).name
    except Exception:
        return str(p)


def _resolve_model_path(sim_root: Path, model_path: str) -> Optional[Path]:
    if not model_path:
        return None
    p = Path(model_path)
    if not p.is_absolute():
        p = sim_root / p
    try:
        return p.resolve()
    except Exception:
        return p


def _rel(sim_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(sim_root.resolve()))
    except Exception:
        return str(p)


def _load_arena(sim_root: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    p = sim_root / "outputs" / "models" / "arena.json"
    if not p.exists():
        return {}, []
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}, []
    if not isinstance(obj, dict):
        return {}, []

    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    items = obj.get("arena_models")
    if not isinstance(items, list):
        return {}, []
    for it in items:
        if not isinstance(it, dict):
            continue
        rp = str(it.get("path") or "").strip()
        if not rp:
            continue
        abs_p = _resolve_model_path(sim_root, rp)
        if abs_p is None:
            continue
        if not abs_p.exists():
            missing.append(rp)
        out[str(abs_p)] = dict(it)
    return out, missing


@dataclass
class BestByDataset:
    dataset_name: str
    dataset_path: str
    split: str
    accuracy: Optional[float]
    f1: Optional[float]
    seen: Optional[int]
    last_ts: float
    report_path: str
    extra: Dict[str, Any]


def _intish(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _critical_fn_summary(report: Dict[str, Any]) -> str:
    try:
        d = (report.get("metrics") or {}).get("critical_fn_rates") or {}
        if not isinstance(d, dict) or not d:
            return "-"
        keys = sorted(str(k) for k in d.keys())
        parts = []
        for k in keys[:3]:
            try:
                parts.append(f"{k}:{float(d.get(k)):.4f}")
            except Exception:
                continue
        return " ".join(parts) if parts else "-"
    except Exception:
        return "-"


def _arena_type(sim_root: Path, model_abs: Path) -> str:
    try:
        return "Bundle" if model_abs.is_dir() else "Single"
    except Exception:
        return "Single"


def _bundle_details(sim_root: Path, bundle_dir: Path) -> Dict[str, Any]:
    try:
        sys.path.insert(0, str(sim_root))
        from simple_sim.model_bundle import read_bundle_meta
    except Exception:
        read_bundle_meta = None  # type: ignore

    meta = None
    if read_bundle_meta is not None:
        try:
            meta = read_bundle_meta(bundle_dir)
        except Exception:
            meta = None

    profiles: List[str] = []
    checkpoints = 0
    created = "-"
    try:
        if meta is not None:
            profiles = sorted(meta.checkpoints.keys())
            checkpoints = len(meta.checkpoints)
            created = meta.created_at or "-"
        else:
            pts = [p for p in bundle_dir.glob("*.pt") if p.is_file()]
            checkpoints = len(pts)
            profiles = sorted({p.stem.replace("_last", "") for p in pts})
        created = created if created != "-" else datetime.fromtimestamp(bundle_dir.stat().st_mtime).strftime("%Y-%m-%d")
    except Exception:
        pass

    return {"profiles": profiles, "checkpoints": checkpoints, "created": created}


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _xml_escape(s: str) -> str:
    # Minimal escaping for SVG/XML text nodes/attributes.
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _write_svg_barh(
    out_path: Path,
    *,
    title: str,
    labels: List[str],
    values: List[float],
    value_fmt: str = "{:.4f}",
) -> None:
    """Write a simple horizontal bar chart as SVG (no external deps)."""
    if not labels or not values or len(labels) != len(values):
        return

    # Layout tuned for markdown rendering on Git hosts.
    w = 980
    top_pad = 64
    bot_pad = 28
    left_pad = 260
    right_pad = 90
    row_h = 26
    bar_h = 14

    n = len(labels)
    h = top_pad + bot_pad + n * row_h

    vmax = max(values) if values else 0.0
    vmin = min(values) if values else 0.0
    if vmax <= 0:
        vmax = 1.0
    # If values are extremely close, avoid dividing by ~0. Keep scaling stable.
    span = max(1e-12, vmax - min(0.0, vmin))

    plot_w = w - left_pad - right_pad

    def x_for(v: float) -> float:
        # Start bars at 0 baseline.
        vv = max(0.0, float(v))
        return left_pad + (vv / (vmax)) * plot_w

    # Dark theme: readable on typical Git markdown (light background) without needing CSS/JS.
    # Background is black, text is white.
    bg = "#0b0f14"
    axis = "#30363d"
    text = "#f0f6fc"
    bar = "#2f81f7"
    bar2 = "#79c0ff"

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>')

    # Title
    lines.append(
        f'<text x="{left_pad}" y="34" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
        f'font-size="20" font-weight="700" fill="{text}">{_xml_escape(title)}</text>'
    )

    # Axis baseline
    y0 = top_pad - 8
    lines.append(f'<line x1="{left_pad}" y1="{y0}" x2="{w - right_pad}" y2="{y0}" stroke="{axis}" stroke-width="1"/>')

    for i, (lab, val) in enumerate(zip(labels, values), start=0):
        y = top_pad + i * row_h
        # Row separator
        lines.append(
            f'<line x1="{left_pad}" y1="{y + row_h - 1}" x2="{w - right_pad}" y2="{y + row_h - 1}" '
            f'stroke="{axis}" stroke-width="1" opacity="0.35"/>'
        )

        # Label
        lines.append(
            f'<text x="{left_pad - 10}" y="{y + 16}" text-anchor="end" '
            f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
            f'font-size="12" fill="{text}">{_xml_escape(lab)}</text>'
        )

        # Bar
        x_end = x_for(val)
        bw = max(0.0, x_end - left_pad)
        y_bar = y + (row_h - bar_h) / 2.0
        lines.append(
            f'<rect x="{left_pad}" y="{y_bar:.1f}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{bar if i % 2 == 0 else bar2}"/>'
        )

        # Value
        try:
            vs = value_fmt.format(float(val))
        except Exception:
            vs = str(val)
        lines.append(
            f'<text x="{w - right_pad + 6}" y="{y + 16}" text-anchor="start" '
            f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
            f'font-size="12" fill="{text}">{_xml_escape(vs)}</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_bytes(n: int) -> str:
    try:
        nn = float(max(0, int(n)))
    except Exception:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while nn >= 1024.0 and u < len(units) - 1:
        nn /= 1024.0
        u += 1
    if u == 0:
        return f"{int(nn)} {units[u]}"
    return f"{nn:.2f} {units[u]}"


def _path_size_bytes(p: Path) -> int:
    """Return on-disk size for a file or directory (best-effort)."""
    try:
        if p.is_file():
            return int(p.stat().st_size)
    except Exception:
        pass

    total = 0
    try:
        if p.is_dir():
            for root, _dirs, files in os.walk(str(p)):
                for fn in files:
                    fp = Path(root) / fn
                    try:
                        if fp.is_symlink():
                            continue
                        total += int(fp.stat().st_size)
                    except Exception:
                        continue
    except Exception:
        return 0
    return total


def _svg_arc_path(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    # Angles are radians. 0 at +x, increasing clockwise for SVG Y-down coords if we invert sin.
    x0 = cx + r * math.cos(a0)
    y0 = cy + r * math.sin(a0)
    x1 = cx + r * math.cos(a1)
    y1 = cy + r * math.sin(a1)
    large = 1 if (a1 - a0) % (2 * math.pi) > math.pi else 0
    sweep = 1
    return f"M {cx:.1f} {cy:.1f} L {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 {large} {sweep} {x1:.1f} {y1:.1f} Z"


def _write_svg_pie_with_legend(
    out_path: Path,
    *,
    title: str,
    items: List[Tuple[str, int]],
) -> None:
    """Write a pie chart SVG with legend for byte sizes."""
    items = [(str(n), int(v)) for (n, v) in items if v is not None and int(v) > 0]
    if not items:
        return

    total = sum(v for _n, v in items)
    if total <= 0:
        return

    # Layout.
    w = 980
    h = 420
    cx = 250.0
    cy = 230.0
    r = 140.0
    title_y = 34
    legend_x = 520.0
    legend_y = 92.0
    legend_row = 22.0

    bg = "#0b0f14"
    text = "#f0f6fc"
    axis = "#30363d"
    palette = [
        "#0969da",
        "#1f883d",
        "#bf8700",
        "#cf222e",
        "#8250df",
        "#d4a72c",
        "#0f766e",
        "#fb7185",
        "#8b949e",
    ]

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>')
    lines.append(
        f'<text x="260" y="{title_y}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
        f'font-size="20" font-weight="700" fill="{text}">{_xml_escape(title)}</text>'
    )

    # Pie slices
    a = -math.pi / 2.0  # start at 12 o'clock
    for i, (_name, val) in enumerate(items):
        frac = float(val) / float(total)
        da = frac * 2.0 * math.pi
        a0 = a
        a1 = a + da
        color = palette[i % len(palette)]
        # For SVG coords (y down), invert angle direction by swapping sign on sin via using -angles,
        # but easier is to rotate in clockwise by using +sin; starting angle already aligned.
        path = _svg_arc_path(cx, cy, r, a0, a1)
        lines.append(f'<path d="{path}" fill="{color}" stroke="{bg}" stroke-width="2"/>')
        a = a1

    # Outline circle for crispness
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{axis}" stroke-width="1"/>')

    # Legend
    lines.append(
        f'<text x="{legend_x}" y="{legend_y - 24:.1f}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
        f'font-size="12" font-weight="700" fill="{text}">Legend (size, %)</text>'
    )
    for i, (name, val) in enumerate(items):
        color = palette[i % len(palette)]
        y = legend_y + i * legend_row
        pct = (100.0 * float(val) / float(total)) if total else 0.0
        label = f"{name}  {_format_bytes(val)}  ({pct:.1f}%)"
        lines.append(f'<rect x="{legend_x}" y="{y - 10:.1f}" width="12" height="12" rx="2" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 18:.1f}" y="{y:.1f}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
            f'font-size="12" fill="{text}">{_xml_escape(label)}</text>'
        )

    # Total
    lines.append(
        f'<text x="{legend_x}" y="{h - 22}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial" '
        f'font-size="12" fill="{text}">Total: {_xml_escape(_format_bytes(total))}</text>'
    )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_charts(sim_root: Path, overall: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Write SVG charts and return markdown-relative paths to embed."""
    assets_dir = sim_root / "ARENA_REPORT_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    charts: List[Tuple[str, str]] = []

    # Top-N by avg accuracy (overall is already sorted by avg_acc desc).
    acc_items = [(it["model_name"], it["avg_acc"]) for it in overall if it.get("avg_acc") is not None]
    acc_items = acc_items[:15]
    if acc_items:
        labels = [str(a[0]) for a in acc_items]
        values = [float(a[1]) for a in acc_items]
        p = assets_dir / "top_avg_accuracy.svg"
        _write_svg_barh(p, title="Top Avg Accuracy (Top 15)", labels=labels, values=values, value_fmt="{:.4f}")
        charts.append(("Top Avg Accuracy", "ARENA_REPORT_assets/top_avg_accuracy.svg"))

    # Top-N by avg F1.
    f1_items = [(it["model_name"], it["avg_f1"]) for it in overall if it.get("avg_f1") is not None]
    f1_items.sort(key=lambda t: -float(t[1]))
    f1_items = f1_items[:15]
    if f1_items:
        labels = [str(a[0]) for a in f1_items]
        values = [float(a[1]) for a in f1_items]
        p = assets_dir / "top_avg_f1.svg"
        _write_svg_barh(p, title="Top Avg F1 (Top 15)", labels=labels, values=values, value_fmt="{:.4f}")
        charts.append(("Top Avg F1", "ARENA_REPORT_assets/top_avg_f1.svg"))

    # Model storage breakdown (pie chart).
    size_items: List[Tuple[str, int]] = []
    for it in overall:
        try:
            p = Path(str(it["model_abs"]))
        except Exception:
            continue
        sz = _path_size_bytes(p)
        if sz <= 0:
            continue
        size_items.append((str(it.get("model_name") or p.name), int(sz)))

    size_items.sort(key=lambda t: -int(t[1]))
    # Keep chart readable: top 8 + "Other".
    if len(size_items) > 9:
        top = size_items[:8]
        other = sum(v for _n, v in size_items[8:])
        size_items = top + [("Other", int(other))]

    if size_items:
        p = assets_dir / "model_size_pie.svg"
        _write_svg_pie_with_legend(p, title="Model Storage Breakdown", items=size_items)
        charts.append(("Model Storage Breakdown", "ARENA_REPORT_assets/model_size_pie.svg"))

    return charts


def _render(
    sim_root: Path,
    tracked: Dict[str, Dict[str, Any]],
    reports: List[Dict[str, Any]],
    *,
    missing: List[str],
) -> str:
    per_model_dataset: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for r in reports:
        mp = str(r.get("model_path") or "").strip()
        abs_mp = _resolve_model_path(sim_root, mp)
        if abs_mp is None:
            continue
        k_model = str(abs_mp)
        if k_model not in tracked:
            continue

        ds_path = str(r.get("dataset_path") or "").strip()
        ds_name = _basename(ds_path) if ds_path else "-"
        per_model_dataset.setdefault((k_model, ds_name), []).append(r)

    best_rows: Dict[Tuple[str, str], BestByDataset] = {}
    for (model_abs, ds_name), rs in per_model_dataset.items():

        def key(rr: Dict[str, Any]) -> Tuple[float, float, float]:
            acc = _metric(rr, "accuracy")
            f1 = _metric(rr, "macro_f1")
            return (
                acc if acc is not None else -1.0,
                f1 if f1 is not None else -1.0,
                _ts(rr),
            )

        rr_best = max(rs, key=key)
        last_ts = max(_ts(rr) for rr in rs) if rs else 0.0
        best_rows[(model_abs, ds_name)] = BestByDataset(
            dataset_name=ds_name,
            dataset_path=str(rr_best.get("dataset_path") or ""),
            split=str(rr_best.get("split") or "test"),
            accuracy=_metric(rr_best, "accuracy"),
            f1=_metric(rr_best, "macro_f1"),
            seen=_intish(rr_best.get("seen_samples") or rr_best.get("test_size")),
            last_ts=last_ts,
            report_path=str(rr_best.get("_path") or ""),
            extra={
                "critical_fn": _critical_fn_summary(rr_best),
                "epoch": rr_best.get("checkpoint_epoch") or rr_best.get("epoch"),
                "val_acc": rr_best.get("checkpoint_val_accuracy") or rr_best.get("val_accuracy"),
                "img_per_s": rr_best.get("img_per_s"),
            },
        )

    model_to_rows: Dict[str, List[BestByDataset]] = {}
    for (model_abs, _ds), row in best_rows.items():
        model_to_rows.setdefault(model_abs, []).append(row)

    overall: List[Dict[str, Any]] = []
    for model_abs, rows in model_to_rows.items():
        accs = [r.accuracy for r in rows if r.accuracy is not None]
        f1s = [r.f1 for r in rows if r.f1 is not None]
        avg_acc = sum(accs) / len(accs) if accs else None
        avg_f1 = sum(f1s) / len(f1s) if f1s else None
        best_ds = "-"
        worst_ds = "-"
        if accs:
            best_r = max([r for r in rows if r.accuracy is not None], key=lambda r: float(r.accuracy))
            worst_r = min([r for r in rows if r.accuracy is not None], key=lambda r: float(r.accuracy))
            best_ds = f"{best_r.dataset_name} ({best_r.accuracy:.4f})"
            worst_ds = f"{worst_r.dataset_name} ({worst_r.accuracy:.4f})"
        overall.append(
            {
                "model_abs": model_abs,
                "model_rel": _rel(sim_root, Path(model_abs)),
                "model_name": Path(model_abs).name,
                "type": _arena_type(sim_root, Path(model_abs)),
                "avg_acc": avg_acc,
                "avg_f1": avg_f1,
                "datasets": len({r.dataset_name for r in rows}),
                "best_ds": best_ds,
                "worst_ds": worst_ds,
                "last_ts": max((r.last_ts for r in rows), default=0.0),
            }
        )

    overall.sort(
        key=lambda it: (
            -(float(it["avg_acc"]) if it["avg_acc"] is not None else -1.0),
            -(float(it["avg_f1"]) if it["avg_f1"] is not None else -1.0),
            -float(it["last_ts"] or 0.0),
            str(it["model_name"]),
        )
    )

    dataset_to_models: Dict[str, List[Tuple[str, BestByDataset]]] = {}
    for model_abs, rows in model_to_rows.items():
        for r in rows:
            dataset_to_models.setdefault(r.dataset_name, []).append((model_abs, r))

    charts = _write_charts(sim_root, overall)

    lines: List[str] = []
    lines.append("# Model Arena Report")
    lines.append(f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if missing:
        lines.append("## Warnings")
        lines.append("")
        lines.append("Tracked paths missing on disk:")
        for rp in missing[:20]:
            lines.append(f"- `{rp}`")
        if len(missing) > 20:
            lines.append(f"- ... (+{len(missing) - 20} more)")
        lines.append("")

    if charts:
        lines.append("## Charts")
        lines.append("")
        for title, rel_path in charts:
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{_md_escape(title)}]({rel_path})")
            lines.append("")

    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Model | Type | Avg Accuracy | Avg F1 | Datasets Tested | Best Dataset | Worst Dataset | Last Run |")
    lines.append("|---:|---|---|---:|---:|---:|---|---|---|")
    for i, it in enumerate(overall, start=1):
        avg_acc = f"{it['avg_acc']:.4f}" if it["avg_acc"] is not None else "-"
        avg_f1 = f"{it['avg_f1']:.4f}" if it["avg_f1"] is not None else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    _md_escape(str(it["model_name"])),
                    str(it["type"]),
                    avg_acc,
                    avg_f1,
                    str(int(it["datasets"])),
                    _md_escape(str(it["best_ds"])),
                    _md_escape(str(it["worst_ds"])),
                    _md_escape(_fmt_ts(float(it["last_ts"] or 0.0))),
                ]
            )
            + " |"
        )
    if not overall:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.append("")

    lines.append("## Per-Dataset Breakdown")
    lines.append("")
    for ds_name in sorted(dataset_to_models.keys()):
        rows = dataset_to_models[ds_name]
        rows.sort(
            key=lambda t: (-(t[1].accuracy or -1.0), -(t[1].f1 or -1.0), -t[1].last_ts, Path(t[0]).name)
        )
        lines.append(f"### Dataset: {ds_name}")
        lines.append("")
        lines.append("| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |")
        lines.append("|---:|---|---:|---:|---|---:|---|")
        for i, (model_abs, r) in enumerate(rows, start=1):
            acc_s = f"{r.accuracy:.4f}" if r.accuracy is not None else "-"
            f1_s = f"{r.f1:.4f}" if r.f1 is not None else "-"
            seen_s = str(r.seen) if r.seen is not None else "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        _md_escape(Path(model_abs).name),
                        acc_s,
                        f1_s,
                        _md_escape(r.split),
                        seen_s,
                        _md_escape(_fmt_ts(r.last_ts)),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Bundle Details")
    lines.append("")
    any_bundle = False
    for it in overall:
        if it["type"] != "Bundle":
            continue
        any_bundle = True
        bdir = Path(it["model_abs"])
        det = _bundle_details(sim_root, bdir)
        lines.append(f"### {_md_escape(bdir.name)}")
        lines.append(f"- **Profiles:** {', '.join(_md_escape(p) for p in det.get('profiles') or []) or '-'}")
        lines.append(f"- **Checkpoints:** {det.get('checkpoints') or 0}")
        lines.append(f"- **Created:** {_md_escape(str(det.get('created') or '-'))}")
        lines.append("")
    if not any_bundle:
        lines.append("- (No bundle models tracked)")
        lines.append("")

    lines.append("## Per-Model Detail Cards")
    lines.append("")
    for it in overall:
        model_abs = it["model_abs"]
        rows = model_to_rows.get(model_abs, [])
        rows.sort(key=lambda r: r.dataset_name)
        lines.append(f"### {_md_escape(Path(model_abs).name)} ({it['type']})")
        lines.append("")
        lines.append("| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
        for r in rows:
            acc_s = f"{r.accuracy:.4f}" if r.accuracy is not None else "-"
            f1_s = f"{r.f1:.4f}" if r.f1 is not None else "-"
            seen_s = str(r.seen) if r.seen is not None else "-"

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
            except Exception:
                val_acc_s = "-"
            try:
                img_s_s = f"{float(img_s):.1f} img/s" if img_s is not None else "-"
            except Exception:
                img_s_s = "-"

            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(r.dataset_name),
                        _md_escape(r.split),
                        acc_s,
                        f1_s,
                        _md_escape(str(r.extra.get("critical_fn") or "-")),
                        seen_s,
                        _md_escape(epoch_s),
                        val_acc_s,
                        _md_escape(img_s_s),
                    ]
                )
                + " |"
            )
        if not rows:
            lines.append("| - | - | - | - | - | - | - | - | - |")
        lines.append("")

    lines.append("## History")
    lines.append("")
    lines.append(
        f"- {datetime.now().strftime('%Y-%m-%d')}: Report generated ({len(overall)} models, {len(dataset_to_models)} datasets)"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate ARENA_REPORT.md for Arena-tracked models")
    ap.add_argument("--sim-root", default=None, help="Simple-Sim root (default: auto-detect)")
    args = ap.parse_args()

    sim_root = Path(args.sim_root).resolve() if args.sim_root else None
    if sim_root is None:
        sim_root = _detect_sim_root(Path.cwd()) or _detect_sim_root(Path(__file__).resolve())
    if sim_root is None:
        raise SystemExit("Could not auto-detect --sim-root (expected outputs/models/ under repo)")

    tracked, missing = _load_arena(sim_root)
    if not tracked:
        out_path = sim_root / "ARENA_REPORT.md"
        out_path.write_text(
            "# Model Arena Report\n"
            f"> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "No Arena-tracked models found.\n",
            encoding="utf-8",
        )
        print(f"[arena] no tracked models; wrote {out_path}")
        return

    dirs = _report_search_dirs(sim_root)
    reports = _load_reports(dirs)
    md = _render(sim_root, tracked, reports, missing=missing)

    out_path = sim_root / "ARENA_REPORT.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"[arena] wrote {out_path}")


if __name__ == "__main__":
    main()
