"""Reports table helpers of the weights tab (gui.utils.report_rows, Tk-free)."""

from __future__ import annotations

from pathlib import Path

from gui.utils.report_rows import report_cells, report_matches_model


def test_report_matches_model_by_name_stem_or_bundle_dir(tmp_path: Path) -> None:
    model = tmp_path / "models" / "run_a.pt"
    assert report_matches_model({"model_path": "elsewhere/run_a.pt"}, model)
    assert report_matches_model({"model_path": "x/other.pt", "model_stem": "run_a"}, model)
    assert not report_matches_model({"model_path": "x/other.pt", "model_stem": "other"}, model)
    assert not report_matches_model({}, model)

    bundle = tmp_path / "models" / "all.bundle"
    bundle.mkdir(parents=True)
    assert report_matches_model({"model_path": str(bundle / "0603.pt")}, bundle)
    assert not report_matches_model({"model_path": str(tmp_path / "models" / "0603.pt")}, bundle)


def test_report_cells_format_and_fallbacks() -> None:
    report = {"timestamp": "2026-11-10T14:05:06.123456", "split": "test", "seen_samples": 120.0,
              "metrics": {"accuracy": 0.98765, "macro_f1": 0.5}, "model_path": "outputs/models/m.pt",
              "dataset_path": "outputs/sim_data/runs/r1", "_path": "outputs/models/history/rep.json"}
    assert report_cells(report, lambda p: f"rel:{p.name}") == (
        "2026-11-10 14:05:06", "test", "120", "0.9877", "0.5000", "m.pt", "r1", "rel:rep.json")
    assert report_cells({}, str) == ("-", "-", "-", "-", "-", "-", "-", "-")
    assert report_cells({"test_size": 7, "_path": "-"}, str)[2::5] == ("7", "-")
