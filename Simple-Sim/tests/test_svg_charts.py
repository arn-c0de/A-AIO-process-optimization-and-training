"""Tests for simple_sim.svg_charts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from simple_sim.svg_charts import heat_color, write_svg_barh, write_svg_heatmap, write_svg_pie_with_legend, xml_escape


def _parse(p: Path) -> ET.Element:
    return ET.fromstring(p.read_text(encoding="utf-8").split("\n", 1)[1])


def test_xml_escape() -> None:
    assert xml_escape("<a & 'b'>") == "&lt;a &amp; &apos;b&apos;&gt;"
    assert xml_escape(None) == ""  # type: ignore[arg-type]


def test_heat_color_endpoints_and_clamp() -> None:
    assert heat_color(0.0) == "#1c2e4a"
    assert heat_color(1.0) == heat_color(5.0) == "#79c0ff"
    assert heat_color(-1) == heat_color(0.0)


def test_barh_writes_valid_svg_with_top3_colors(tmp_path: Path) -> None:
    out = tmp_path / "bar.svg"
    write_svg_barh(out, title="T & <x>", labels=["a", "b", "c", "d"], values=[0.9, 0.8, 0.7, 0.6])
    root = _parse(out)
    fills = [r.get("fill") for r in root.iter("{http://www.w3.org/2000/svg}rect")][1:]
    assert fills[:3] == ["#f2cc60", "#b1bac4", "#c69063"]
    assert "T &amp; &lt;x&gt;" in out.read_text(encoding="utf-8")


def test_writers_skip_invalid_input(tmp_path: Path) -> None:
    write_svg_barh(tmp_path / "a.svg", title="t", labels=["a"], values=[])
    write_svg_heatmap(tmp_path / "b.svg", title="t", row_labels=["r"], col_labels=["c"], values=[[0.1, 0.2]])
    write_svg_pie_with_legend(tmp_path / "c.svg", title="t", items=[("x", 0)])
    assert list(tmp_path.iterdir()) == []


def test_heatmap_and_pie(tmp_path: Path) -> None:
    write_svg_heatmap(tmp_path / "h.svg", title="H", row_labels=["m1", "m2"], col_labels=["d1"], values=[[0.5], [None]])
    _parse(tmp_path / "h.svg")
    write_svg_pie_with_legend(tmp_path / "p.svg", title="P", items=[("only", 2048)])
    txt = (tmp_path / "p.svg").read_text(encoding="utf-8")
    assert "<circle" in txt and "only  2.00 KB  (100.0%)" in txt
    _parse(tmp_path / "p.svg")
