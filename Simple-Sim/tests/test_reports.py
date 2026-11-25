"""Evaluation reports (simple_sim.reports, batch-predict): search dirs, history, partial runs, profile notes."""

from __future__ import annotations

import json
from pathlib import Path

from simple_sim import reports
from simple_sim.generation.dataset import generate_dataset
from tests.helpers import ROOT, build_sim_root
from tests.helpers import tiny_run_config as _config
from tests.helpers import write_report as _report


def test_report_search_dirs_cover_model_subfolders(tmp_path):
    models = tmp_path / "outputs" / "models"
    (models / "versions" / "run_a").mkdir(parents=True)
    (models / "imports").mkdir()
    dirs = reports.report_search_dirs(tmp_path)
    for want in (models, models / "history", models / "imports", models / "versions" / "run_a",
                 models / "versions" / "run_a" / "history"):
        assert want in dirs
    assert len(dirs) == len(set(dirs))


def test_batch_predict_history_skips_quick_checks(tmp_path):
    from simple_sim.batch_predict import history_rows

    def write(name: str, acc: float, **extra) -> None:
        (tmp_path / name).write_text(json.dumps({
            "timestamp": "2026-11-10T10:00:00", "model_path": "m.pt", "model_stem": "m", "dataset_path": "ds",
            "split": "test", "metrics": {"accuracy": acc, "macro_f1": acc}, **extra}), encoding="utf-8")

    write("report_m_full.json", 0.8, seen_samples=100, dataset_split_sizes={"test": 100})
    write("report_m_quick.json", 1.0, seen_samples=1, max_samples=1, dataset_split_sizes={"test": 100})
    rows = history_rows([tmp_path], scope="model", split_mode="same", metric="accuracy", critical_class=None,
                        model_path=Path("m.pt"), data_dir=Path("ds"), split="test", exclude_path=tmp_path / "none.json")
    assert [v for v, _r in rows] == [0.8]


def test_batch_predict_notes_a_profile_mismatch(tmp_path):
    from simple_sim.batch_predict import _profile_mismatch

    data = tmp_path / "ds"
    generate_dataset(_config(), data, project_root=ROOT)
    manifest = json.loads((data / "dataset_manifest.json").read_text(encoding="utf-8"))
    own = manifest["component_profile"]["profile_id"]
    assert _profile_mismatch(own, data) is None
    assert _profile_mismatch("", data) is None and _profile_mismatch("multi", data) is None
    assert _profile_mismatch("other_part@1", data) == {"model": "other_part@1", "dataset": own}
    assert _profile_mismatch("other_part@1", tmp_path / "missing") is None


def test_is_partial_report():
    full = {"split": "test", "seen_samples": 40, "dataset_split_sizes": {"test": 40}}
    assert not reports.is_partial_report(full)
    assert not reports.is_partial_report({"metrics": {}})  # legacy reports count as complete
    assert reports.is_partial_report({**full, "seen_samples": 16})
    assert reports.is_partial_report({**full, "max_samples": 16})
    assert reports.is_partial_report({"seen_samples": 30, "test_size": 30, "excluded_samples": 10})
    assert reports.is_partial_report({"seen_samples": 10, "test_size": 30})


def test_status_best_accuracy_ignores_train_split_and_quick_checks(tmp_path):
    from simple_sim.cli import collect_status

    root = build_sim_root(tmp_path)
    hist = root / "outputs" / "models" / "history"
    quick = hist / "batch_report_b_test_quick.json"
    _report(quick, model="outputs/models/b.pt", dataset="outputs/sim_data/runs/ds_one", acc=1.0, f1=1.0,
            ts="2026-01-07T10:00:00")
    data = json.loads(quick.read_text())
    data["max_samples"] = 8
    quick.write_text(json.dumps(data))
    best = collect_status(root)["reports"]["best_accuracy"]
    # 0.99 is a train-split report, 1.0 the quick check and the untracked z.pt report (a full test run).
    assert best["accuracy"] == 1.0 and "batch_report_z_test_1" in best["path"]
