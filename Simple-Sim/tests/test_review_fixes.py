"""Regression tests for the fixes of the 2026-11-10 code review (see CHANGELOG)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from simple_sim import io_utils, reports
from simple_sim.arena import best_rows_by_model, dataset_display_name, load_arena
from simple_sim.dataset_store import atomic_dataset_dir
from simple_sim.metrics import compute_metrics
from simple_sim.rows import MetaRow, read_jsonl
from simple_sim.splits import read_split, write_splits
from tests.test_arena_report import _report, build_sim_root

# ---------------------------------------------------------------------------
# io_utils / splits / rows / metrics
# ---------------------------------------------------------------------------


def test_atomic_writes_use_unique_temp_files_and_leave_none(tmp_path):
    target = tmp_path / "state.json"
    errors: list[BaseException] = []

    def write(i: int) -> None:
        try:
            for _ in range(50):
                io_utils.atomic_write_json(target, {"writer": i, "pad": "x" * 2000})
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors  # a shared "<name>.tmp" made concurrent writers fail on the rename
    assert json.loads(target.read_text())["pad"] == "x" * 2000
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_atomic_write_removes_its_temp_file_on_failure(tmp_path):
    def rows():
        yield {"a": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        io_utils.atomic_write_jsonl(tmp_path / "rows.jsonl", rows())
    assert list(tmp_path.iterdir()) == []


def test_write_splits_replaces_files_atomically(tmp_path):
    write_splits(tmp_path, {"train": ["a", "b"], "val": [], "test": ["c"]})
    write_splits(tmp_path, {"train": ["a"], "val": ["b"], "test": ["c"]})
    assert read_split(tmp_path / "splits" / "train.txt") == ["a"]
    assert read_split(tmp_path / "splits" / "val.txt") == ["b"]
    assert sorted(p.name for p in (tmp_path / "splits").iterdir()) == ["test.txt", "train.txt", "val.txt"]


def test_read_jsonl_rejects_non_object_lines(tmp_path):
    path = tmp_path / "meta.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path, MetaRow)


def test_critical_fn_rate_is_absent_for_a_class_without_samples():
    m = compute_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), ["OK", "MISSING", "MISALIGNED"],
                        critical_classes=["MISSING", "MISALIGNED"])
    assert m["critical_fn_rates"] == {"MISSING": 0.0}  # MISALIGNED: support 0 -> no rate, not a perfect 0.0
    assert reports.metric_value({"metrics": m}, "critical_fn_rate", "MISALIGNED") is None


# ---------------------------------------------------------------------------
# dataset_store
# ---------------------------------------------------------------------------


def test_atomic_dataset_dir_swaps_and_keeps_the_old_dataset_on_failure(tmp_path):
    out = tmp_path / "ds"
    with atomic_dataset_dir(out) as tmp:
        (tmp / "marker").write_text("v1")
    assert (out / "marker").read_text() == "v1"

    with pytest.raises(RuntimeError), atomic_dataset_dir(out) as tmp:
        (tmp / "marker").write_text("broken")
        raise RuntimeError("render failed")
    assert (out / "marker").read_text() == "v1"

    with atomic_dataset_dir(out) as tmp:
        (tmp / "marker").write_text("v2")
    assert (out / "marker").read_text() == "v2"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]  # no .tmp / .old left


def test_atomic_dataset_dir_does_not_lose_the_new_dataset_when_the_old_cannot_be_deleted(tmp_path, monkeypatch):
    out = tmp_path / "ds"
    with atomic_dataset_dir(out) as tmp:
        (tmp / "marker").write_text("v1")

    import shutil

    real_rmtree = shutil.rmtree

    def failing_rmtree(path, *args, **kwargs):
        if Path(path).name == "ds.old":
            if kwargs.get("ignore_errors"):
                return
            raise PermissionError("busy")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("simple_sim.dataset_store.shutil.rmtree", failing_rmtree)
    with atomic_dataset_dir(out) as tmp:
        (tmp / "marker").write_text("v2")
    assert (out / "marker").read_text() == "v2"


# ---------------------------------------------------------------------------
# reports / arena / status
# ---------------------------------------------------------------------------


def test_is_partial_report():
    full = {"split": "test", "seen_samples": 40, "dataset_split_sizes": {"test": 40}}
    assert not reports.is_partial_report(full)
    assert not reports.is_partial_report({"metrics": {}})  # legacy reports count as complete
    assert reports.is_partial_report({**full, "seen_samples": 16})
    assert reports.is_partial_report({**full, "max_samples": 16})
    assert reports.is_partial_report({"seen_samples": 30, "test_size": 30, "excluded_samples": 10})
    assert reports.is_partial_report({"seen_samples": 10, "test_size": 30})


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


def test_arena_names_snapshots_by_dataset_and_version():
    assert dataset_display_name("outputs/sim_data/runs/run_0001") == "run_0001"
    assert dataset_display_name("outputs/sim_data/versions/run_A/v1") == "run_A/v1"
    assert dataset_display_name("outputs/sim_data/versions/run_B/v1") == "run_B/v1"
    assert dataset_display_name("") == "-"


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


# ---------------------------------------------------------------------------
# batch-predict / eval
# ---------------------------------------------------------------------------


def test_max_samples_subset_is_spread_over_the_split():
    pytest.importorskip("torch")
    from simple_sim.batch_predict import evenly_spaced_indices

    assert evenly_spaced_indices(5, 10) == [0, 1, 2, 3, 4]
    idx = evenly_spaced_indices(100, 4)
    assert idx == [0, 33, 66, 99]  # first N would be the first class only (ids are sorted class by class)
    assert len(set(evenly_spaced_indices(7, 6))) == 6
    assert evenly_spaced_indices(100, 1) == [0]


def test_eval_keeps_the_report_of_another_dataset(tmp_path):
    pytest.importorskip("torch")
    from simple_sim.evaluation import _archive_report_of_other_dataset

    report = tmp_path / "report_m.json"
    report.write_text(json.dumps({"dataset_path": str(tmp_path / "ds_a"), "metrics": {"accuracy": 0.9}}))
    _archive_report_of_other_dataset(report, tmp_path / "ds_a")  # same dataset: replaced in place later
    assert report.exists() and not (tmp_path / "history").exists()
    _archive_report_of_other_dataset(report, tmp_path / "ds_b")
    assert not report.exists()
    kept = tmp_path / "history" / "report_m_ds_a.json"
    assert json.loads(kept.read_text())["metrics"]["accuracy"] == 0.9
    _archive_report_of_other_dataset(report, tmp_path / "ds_b")  # nothing there: no error


# ---------------------------------------------------------------------------
# GUI: flags
# ---------------------------------------------------------------------------


def test_flag_manager_batches_and_survives_a_bad_line(tmp_path):
    from gui.utils.flag_manager import FlagManager

    fm = FlagManager(tmp_path)
    assert fm.add_flags([("a", "OUTLIER", "r1"), ("b", "DUPLICATE", "r2"), ("a", "OUTLIER", "again")]) == 2
    path = tmp_path / "flags.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    path.write_text(lines[0] + "\n{broken\n" + lines[1] + "\n")
    again = FlagManager(tmp_path)
    assert sorted(again.get_all_flagged_samples()) == ["a", "b"]  # the line after the bad one is kept
    again.clear()
    assert path.read_text() == "" and FlagManager(tmp_path).get_all_flagged_samples() == []
    assert [p.name for p in tmp_path.iterdir()] == ["flags.jsonl"]


def test_validation_flags_go_to_the_validated_dataset(tmp_path, monkeypatch):
    from gui.tabs.validation import logic as vlogic

    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    ds_a.mkdir()
    ds_b.mkdir()
    monkeypatch.setattr(
        vlogic, "run_validation_suite",
        lambda *_a, **_k: {"outliers": [{"sample_id": "s1", "flag": "OUTLIER", "reason": "r"}],
                           "duplicates": [("s2", "s3", 1)]},
    )
    logic = vlogic.ValidationLogic(tmp_path)
    logic.set_dataset(ds_b)  # the selection moved to B while A is being validated
    logic.run_validation(ds_a, {"outliers": True})
    assert not (ds_b / "flags.jsonl").exists()
    assert sorted(vlogic.FlagManager(ds_a).get_all_flagged_samples()) == ["s1", "s2"]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def _empty_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "splits").mkdir()
    for name in ("meta.jsonl", "labels.jsonl"):
        (root / name).write_text("", encoding="utf-8")
    (root / "config.yaml").write_text("run: {seed: 1}\n", encoding="utf-8")
    for split in ("train", "val", "test"):
        (root / "splits" / f"{split}.txt").write_text("", encoding="utf-8")
    return root


def test_validate_fails_an_empty_dataset_and_foreign_split_ids(tmp_path, capsys):
    from tools.validate_dataset import run_validation

    ds = _empty_dataset(tmp_path / "ds")
    ok, checks = run_validation(ds)
    by_name = {c["name"]: c["status"] for c in checks}
    assert not ok and by_name["non_empty"] == "fail"

    (ds / "splits" / "train.txt").write_text("bogus/id/train/999999\n", encoding="utf-8")
    ok, checks = run_validation(ds)
    by_name = {c["name"]: c["status"] for c in checks}
    assert not ok and by_name["split_membership"] == "fail"
    capsys.readouterr()


def test_backfill_manifest_survives_an_empty_run_section(tmp_path, capsys):
    from tools.backfill_manifest import STATUS_FAILED, backfill_dataset

    ds = _empty_dataset(tmp_path / "ds")
    (ds / "config.yaml").write_text("run:\n", encoding="utf-8")
    # No AttributeError on `run: null`; the dataset is reported as failed (unknown profile) instead.
    assert backfill_dataset(ds, "no_such_profile@1", tmp_path) == STATUS_FAILED
    capsys.readouterr()


def test_create_multi_dataset_removes_a_failed_output(tmp_path, capsys):
    from tools.create_multi_dataset import create_multi_dataset

    src = _empty_dataset(tmp_path / "src")
    (src / "config.yaml").write_text("classes: {OK: 1}\n", encoding="utf-8")
    (src / "meta.jsonl").write_text("not json\n", encoding="utf-8")
    out = tmp_path / "merged"
    with pytest.raises((SystemExit, ValueError)):
        create_multi_dataset([src], out)
    assert not out.exists()  # a partial output used to block the retry ("already exists")
    capsys.readouterr()


def test_deepvision_tools_are_registered_commands():
    from simple_sim.cli import COMMANDS

    for name, script in (("dv-crops", "tools/dv_crops.py"), ("board-backgrounds", "tools/board_backgrounds.py"),
                         ("export-deepvision", "tools/export_deepvision.py")):
        assert COMMANDS[name].script == script
