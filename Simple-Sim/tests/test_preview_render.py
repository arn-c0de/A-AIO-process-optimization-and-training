"""Tests for the tk-free debug preview renderer (simple_sim.preview.render) and its CLI --json."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from simple_sim.cli_utils import parse_result
from simple_sim.preview.render import (
    PreviewOptions,
    backends_for_mode,
    collect_existing_previews,
    discover_profiles,
    parse_roi,
    parse_selection,
    render_previews,
    sanitize_dir_name,
)

SIM_ROOT = Path(__file__).resolve().parent.parent


def test_parse_helpers() -> None:
    assert parse_selection("1,3-4,3", 5) == [0, 2, 3]
    assert parse_selection("all", 3) == [0, 1, 2]
    with pytest.raises(ValueError):
        parse_selection("6", 5)
    assert parse_roi(" 64x32@0.02 ") == (64, 32, 0.02)
    with pytest.raises(ValueError):
        parse_roi("64x32")
    assert sanitize_dir_name("a b/c@1") == "a_b_c@1"
    assert backends_for_mode("both", {"blender_3d", "opencv_2d"}) == ["opencv_2d", "blender_3d"]
    assert backends_for_mode("blender_3d", {"opencv_2d"}) == []


def test_render_2d_previews_deterministic(tmp_path: Path) -> None:
    entries = [e for e in discover_profiles(SIM_ROOT / "configs" / "profiles", backend_mode="opencv_2d")][:1]
    assert entries
    opts = PreviewOptions(backend_mode="opencv_2d", roi_override=(64, 64, 0.04), states="OK,MISSING")
    kwargs = dict(opts=opts, sim_root=SIM_ROOT, configs_dir=SIM_ROOT / "configs", image_filters=None)
    run_a = render_previews(entries, out_root=tmp_path / "a", **kwargs)
    render_previews(entries, out_root=tmp_path / "b", **kwargs)
    assert [r[2] for r in run_a.previews] == ["OK", "MISSING"]
    assert run_a.jobs_path is None and run_a.index_path.exists()
    for rec in run_a.previews:
        assert (tmp_path / "a" / rec[3]).read_bytes() == (tmp_path / "b" / rec[3]).read_bytes()
    assert sorted(collect_existing_previews(tmp_path / "a")) == sorted(run_a.previews)


def test_dry_run_3d_writes_jobs_only(tmp_path: Path) -> None:
    entries = discover_profiles(SIM_ROOT / "configs" / "profiles", backend_mode="blender_3d")[:1]
    if not entries:
        pytest.skip("no 3D profiles")
    run = render_previews(
        entries, opts=PreviewOptions(backend_mode="blender_3d", states="OK"), sim_root=SIM_ROOT,
        configs_dir=SIM_ROOT / "configs", out_root=tmp_path, image_filters=None, dry_run=True,
    )
    assert run.jobs_path is not None and run.jobs_path.read_text(encoding="utf-8").count("\n") == 1
    assert not list(tmp_path.glob("previews/**/*.png"))


def test_cli_json_headless(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SIM_ROOT / "scripts" / "render_debug_previews.py"), "--backend", "opencv_2d",
         "--profiles", "chip_0603_resistor@1", "--states", "OK", "--roi", "48x48@0.04", "--out", str(tmp_path),
         "--settings-file", str(tmp_path / "settings.json"), "--json"],
        capture_output=True, text=True, check=True,
    )
    res = parse_result(proc.stdout)
    assert res["ok"] and res["rendered"] and len(res["images"]) == 1 and Path(res["images"][0]).exists()

    bad = subprocess.run(
        [sys.executable, str(SIM_ROOT / "scripts" / "render_debug_previews.py"), "--profiles", "nope@1",
         "--out", str(tmp_path), "--json"],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1 and parse_result(bad.stdout)["ok"] is False
