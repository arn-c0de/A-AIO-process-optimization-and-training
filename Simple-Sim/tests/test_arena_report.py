"""Tests for the Arena report (simple_sim.arena + scripts/arena_report.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from simple_sim.arena import load_arena, write_arena_report
from simple_sim.cli_utils import parse_result

SIM_ROOT = Path(__file__).resolve().parent.parent


def _report(path: Path, *, model: str, dataset: str, acc: float, f1: float, split: str = "test", ts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "model_path": model,
                "dataset_path": dataset,
                "split": split,
                "seen_samples": 40,
                "checkpoint_epoch": 3,
                "checkpoint_val_accuracy": 0.9,
                "img_per_s": 123.4,
                "metrics": {"accuracy": acc, "macro_f1": f1, "critical_fn_rates": {"MISSING": 0.1}},
            }
        ),
        encoding="utf-8",
    )


def build_sim_root(root: Path) -> Path:
    """Create a small fake Simple-Sim tree with tracked models, a bundle and reports."""
    models = root / "outputs" / "models"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    (models / "a.pt").write_bytes(b"x" * 3000)
    (models / "b.pt").write_bytes(b"x" * 1500)
    bundle = models / "multi.bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "chip@1.pt").write_bytes(b"x" * 500)
    (bundle / "bundle.json").write_text(
        json.dumps({"bundle_version": 1, "created_at": "2026-01-01", "checkpoints": {"chip@1": "chip@1.pt"}}),
        encoding="utf-8",
    )
    for name in ("ds_one", "ds_two"):
        ds = root / "outputs" / "sim_data" / "runs" / name
        (ds / "predictions").mkdir(parents=True, exist_ok=True)
        (ds / "meta.jsonl").write_text("{}\n" * 10, encoding="utf-8")
    (models / "arena.json").write_text(
        json.dumps(
            {
                "arena_models": [
                    {"path": "outputs/models/a.pt"},
                    {"path": "outputs/models/b.pt"},
                    {"path": "outputs/models/multi.bundle"},
                    {"path": "outputs/models/gone.pt"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ds1 = "outputs/sim_data/runs/ds_one"
    ds2 = "outputs/sim_data/runs/ds_two"
    hist = models / "history"
    _report(models / "report_a.json", model="outputs/models/a.pt", dataset=ds1, acc=0.91, f1=0.90, ts="2026-01-02T10:00:00")
    _report(hist / "batch_report_a_test_1.json", model="outputs/models/a.pt", dataset=ds2, acc=0.71, f1=0.70, ts="2026-01-03T10:00:00")
    # Train-split report must not win over the test one.
    _report(hist / "batch_report_a_train_1.json", model="outputs/models/a.pt", dataset=ds2, acc=0.99, f1=0.99, split="train", ts="2026-01-03T11:00:00")
    _report(hist / "batch_report_b_test_1.json", model="outputs/models/b.pt", dataset=ds1, acc=0.95, f1=0.94, ts="2026-01-04T10:00:00")
    _report(
        root / "outputs" / "sim_data" / "runs" / "ds_one" / "predictions" / "batch_report_chip_test_1.json",
        model="outputs/models/multi.bundle/chip@1.pt", dataset=ds1, acc=0.5, f1=0.4, ts="2026-01-05T10:00:00",
    )
    # Untracked model: ignored.
    _report(hist / "batch_report_z_test_1.json", model="outputs/models/z.pt", dataset=ds1, acc=1.0, f1=1.0, ts="2026-01-05T10:00:00")
    return root


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
    assert res["ok"] is True and res["command"] == "arena_report"
    assert res["n_models"] == 3 and out.exists()
