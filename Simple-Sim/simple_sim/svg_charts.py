"""Dependency-free SVG chart writers (dark theme) used by the Arena report.

Each ``write_svg_*`` function writes one standalone SVG file and silently does nothing
when the input is empty or inconsistent.
"""

from __future__ import annotations

import math
from pathlib import Path

from simple_sim.io_utils import format_bytes

FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial"

# Dark palette shared by all charts.
BG = "#0b0f14"
AXIS = "#30363d"
TEXT = "#f0f6fc"


def xml_escape(s: str) -> str:
    """Minimal escaping for SVG/XML text nodes and attribute values."""
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _svg_header(w: int, h: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>',
    ]


def _title(x: object, y: object, title: str) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" '
        f'font-size="20" font-weight="700" fill="{TEXT}">{xml_escape(title)}</text>'
    )


def _finish(out_path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_barh(
    out_path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    value_fmt: str = "{:.4f}",
) -> None:
    """Horizontal bar chart; the first three bars are highlighted gold/silver/bronze."""
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
    h = top_pad + bot_pad + len(labels) * row_h

    vmax = max(values)
    if vmax <= 0:
        vmax = 1.0
    plot_w = w - left_pad - right_pad

    bar_colors = ("#2f81f7", "#79c0ff")
    top_colors = ("#f2cc60", "#b1bac4", "#c69063")

    lines = _svg_header(w, h)
    lines.append(_title(left_pad, 34, title))

    y0 = top_pad - 8
    lines.append(f'<line x1="{left_pad}" y1="{y0}" x2="{w - right_pad}" y2="{y0}" stroke="{AXIS}" stroke-width="1"/>')

    for i, (lab, val) in enumerate(zip(labels, values)):
        y = top_pad + i * row_h
        lines.append(
            f'<line x1="{left_pad}" y1="{y + row_h - 1}" x2="{w - right_pad}" y2="{y + row_h - 1}" '
            f'stroke="{AXIS}" stroke-width="1" opacity="0.35"/>'
        )
        lines.append(
            f'<text x="{left_pad - 10}" y="{y + 16}" text-anchor="end" '
            f'font-family="{FONT}" '
            f'font-size="12" fill="{TEXT}">{xml_escape(lab)}</text>'
        )

        # Bars start at a 0 baseline.
        x_end = left_pad + (max(0.0, float(val)) / vmax) * plot_w
        bw = max(0.0, x_end - left_pad)
        y_bar = y + (row_h - bar_h) / 2.0
        bar_color = top_colors[i] if i < 3 else bar_colors[i % 2]
        lines.append(
            f'<rect x="{left_pad}" y="{y_bar:.1f}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{bar_color}"/>'
        )

        try:
            vs = value_fmt.format(float(val))
        except (TypeError, ValueError):
            vs = str(val)
        lines.append(
            f'<text x="{w - right_pad + 6}" y="{y + 16}" text-anchor="start" '
            f'font-family="{FONT}" '
            f'font-size="12" fill="{TEXT}">{xml_escape(vs)}</text>'
        )

    _finish(out_path, lines)


def _shorten(s: str, limit: int) -> str:
    t = str(s or "").strip()
    return t if len(t) <= limit else t[: max(0, limit - 3)] + "..."


def heat_color(v: float) -> str:
    """Dark-theme friendly blue->cyan colour for a value in ``[0, 1]`` (clamped)."""
    vv = max(0.0, min(1.0, float(v)))
    c0 = (28, 46, 74)      # low
    c1 = (47, 129, 247)    # mid
    c2 = (121, 192, 255)   # high
    lo, hi, t = (c0, c1, vv / 0.5) if vv < 0.5 else (c1, c2, (vv - 0.5) / 0.5)
    r, g, b = (int(lo[k] + (hi[k] - lo[k]) * t) for k in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg_heatmap(
    out_path: Path,
    *,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float | None]],
) -> None:
    """Heatmap of accuracy values (rows x cols); ``None`` cells are drawn as N/A."""
    if not row_labels or not col_labels or not values:
        return
    rows = len(row_labels)
    cols = len(col_labels)
    if len(values) != rows or any(len(r) != cols for r in values):
        return

    cell_w = 90
    cell_h = 30
    left_pad = 260
    top_pad = 290
    right_pad = 40
    bottom_pad = 36
    w = left_pad + cols * cell_w + right_pad
    h = top_pad + rows * cell_h + bottom_pad
    na_fill = "#161b22"

    lines = _svg_header(w, h)
    lines.append(_title(left_pad, 34, title))
    lines.append(
        f'<text x="{left_pad}" y="54" font-family="{FONT}" '
        f'font-size="11" fill="{TEXT}" opacity="0.82">Cells show Accuracy per model/dataset (N/A if no run found).</text>'
    )

    for j, col in enumerate(col_labels):
        x = left_pad + j * cell_w + cell_w * 0.5
        lines.append(
            f'<text x="{x:.1f}" y="{top_pad - 26}" text-anchor="end" '
            f'transform="rotate(-42 {x:.1f},{top_pad - 26})" '
            f'font-family="{FONT}" '
            f'font-size="12" fill="{TEXT}">{xml_escape(_shorten(col, 26))}</text>'
        )

    for i, row in enumerate(row_labels):
        y = top_pad + i * cell_h
        lines.append(
            f'<text x="{left_pad - 10}" y="{y + 18}" text-anchor="end" '
            f'font-family="{FONT}" '
            f'font-size="11" fill="{TEXT}">{xml_escape(_shorten(row, 30))}</text>'
        )
        for j in range(cols):
            x = left_pad + j * cell_w
            v = values[i][j]
            fill = na_fill if v is None else heat_color(float(v))
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="3" fill="{fill}" stroke="{AXIS}" stroke-width="1"/>'
            )
            cx = x + (cell_w - 2) * 0.5
            if v is not None:
                lines.append(
                    f'<text x="{cx:.1f}" y="{y + 17}" text-anchor="middle" '
                    f'font-family="{FONT}" '
                    f'font-size="13" fill="#000000">{float(v):.3f}</text>'
                )
            else:
                lines.append(
                    f'<text x="{cx:.1f}" y="{y + 17}" text-anchor="middle" '
                    f'font-family="{FONT}" '
                    f'font-size="12" fill="#000000" opacity="0.7">-</text>'
                )

    _finish(out_path, lines)


def svg_arc_path(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    """SVG path for a pie slice from angle *a0* to *a1* (radians, clockwise in SVG coords)."""
    x0 = cx + r * math.cos(a0)
    y0 = cy + r * math.sin(a0)
    x1 = cx + r * math.cos(a1)
    y1 = cy + r * math.sin(a1)
    large = 1 if (a1 - a0) % (2 * math.pi) > math.pi else 0
    return f"M {cx:.1f} {cy:.1f} L {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 {large} 1 {x1:.1f} {y1:.1f} Z"


PIE_PALETTE = (
    "#0969da",
    "#1f883d",
    "#bf8700",
    "#cf222e",
    "#8250df",
    "#d4a72c",
    "#0f766e",
    "#fb7185",
    "#8b949e",
)


def write_svg_pie_with_legend(out_path: Path, *, title: str, items: list[tuple[str, int]]) -> None:
    """Pie chart of byte sizes with a legend (name, human size, percent) and total."""
    items = [(str(n), int(v)) for (n, v) in items if v is not None and int(v) > 0]
    if not items:
        return
    total = sum(v for _n, v in items)

    w = 980
    h = 420
    cx = 250.0
    cy = 230.0
    r = 140.0
    legend_x = 520.0
    legend_y = 92.0
    legend_row = 22.0

    lines = _svg_header(w, h)
    lines.append(_title(260, 34, title))

    a = -math.pi / 2.0  # start at 12 o'clock
    for i, (_name, val) in enumerate(items):
        frac = float(val) / float(total)
        a1 = a + frac * 2.0 * math.pi
        color = PIE_PALETTE[i % len(PIE_PALETTE)]
        if frac >= 0.9999:
            # A full-circle arc has identical start/end points and SVG would drop it.
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="{BG}" stroke-width="2"/>')
        else:
            lines.append(f'<path d="{svg_arc_path(cx, cy, r, a, a1)}" fill="{color}" stroke="{BG}" stroke-width="2"/>')
        a = a1

    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{AXIS}" stroke-width="1"/>')

    lines.append(
        f'<text x="{legend_x}" y="{legend_y - 24:.1f}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" fill="{TEXT}">Legend (size, %)</text>'
    )
    for i, (name, val) in enumerate(items):
        color = PIE_PALETTE[i % len(PIE_PALETTE)]
        y = legend_y + i * legend_row
        pct = 100.0 * float(val) / float(total)
        label = f"{name}  {format_bytes(val)}  ({pct:.1f}%)"
        lines.append(f'<rect x="{legend_x}" y="{y - 10:.1f}" width="12" height="12" rx="2" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 18:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="12" fill="{TEXT}">{xml_escape(label)}</text>'
        )

    lines.append(
        f'<text x="{legend_x}" y="{h - 22}" font-family="{FONT}" '
        f'font-size="12" fill="{TEXT}">Total: {xml_escape(format_bytes(total))}</text>'
    )
    _finish(out_path, lines)
