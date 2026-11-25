"""Tests for the Arena report (simple_sim.arena + scripts/arena_report.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from simple_sim.arena import load_arena, write_arena_report
from simple_sim.cli_utils import parse_result

from tests.helpers import build_sim_root
from tests.helpers import write_report as _report
from simple_sim.arena import best_rows_by_model
import json
from simple_sim import reports

SIM_ROOT = Path(__file__).resolve().parent.parent


def test_load_arena_reports_missing(tmp_path: Path) -> None:
    root = build_sim_root(tmp_path)
    tracked, missing = load_arena(root)
    assert len(tracked) == 4
    assert missing == ["outputs/models/gone.pt"]


def test_write_arena_report_ranking_and_assets(tmp_path: Path) -> None:
    root = build_sim_root(tmp_path)
    out = tmp_path / "custom" / "REPORT.md"
    res = write_arena_report(root, out)
    assert res["out_path"] == str(out)
    assert res["n_models"] == 3
    assert [t["model_name"] for t in res["top"]] == ["b.pt", "a.pt", "multi.bundle"]
    assert res["top"][1]["avg_acc"] == (0.91 + 0.71) / 2  # train report ignored
    md = out.read_text(encoding="utf-8")
    assert "| Top-1 | **b.pt** | Single |" in md
    assert "### multi.bundle" in md and "- **Profiles:** chip@1" in md
    assert "Tracked paths missing on disk:" in md
    assert "z.pt" not in md
    assets = out.parent / "REPORT_assets"
    assert f"]({assets.name}/top_avg_accuracy.svg)" in md
    for name in ("top_avg_accuracy.svg", "top_avg_f1.svg", "model_size_pie.svg", "dataset_size_pie.svg", "accuracy_heatmap.svg"):
        assert (assets / name).read_text(encoding="utf-8").startswith("<?xml")


def test_write_arena_report_no_tracked(tmp_path: Path) -> None:
    res = write_arena_report(tmp_path)
    assert res["n_tracked"] == 0
    assert "No Arena-tracked models found." in (tmp_path / "ARENA_REPORT.md").read_text(encoding="utf-8")


def test_cli_json(tmp_path: Path) -> None:
    root = build_sim_root(tmp_path)
    out = tmp_path / "r.md"
    proc = subprocess.run(
        [sys.executable, str(SIM_ROOT / "scripts" / "arena_report.py"), "--sim-root", str(root), "--out", str(out), "--json"],
        capture_output=True, text=True, check=True,
    )
    res = parse_result(proc.stdout)
    assert res["ok"] is True and res["command"] == "arena-report"
    assert res["n_models"] == 3 and out.exists()


def test_arena_prefers_a_full_run_over_a_quick_check(tmp_path):
    root = build_sim_root(tmp_path)
    hist = root / "outputs" / "models" / "history"
    ds1 = "outputs/sim_data/runs/ds_one"
    _report(hist / "batch_report_a_test_quick.json", model="outputs/models/a.pt", dataset=ds1, acc=1.0, f1=1.0,
            ts="2026-01-06T10:00:00")
    quick = hist / "batch_report_a_test_quick.json"
    data = json.loads(quick.read_text())
    data.update(max_samples=16, seen_samples=16)
    quick.write_text(json.dumps(data))

    tracked, _missing = load_arena(root)
    rows = best_rows_by_model(root, tracked, reports.load_reports(reports.report_search_dirs(root)))
    a_rows = {r.dataset_name: r for r in rows[str((root / "outputs" / "models" / "a.pt").resolve())]}
    assert a_rows["ds_one"].accuracy == 0.91
