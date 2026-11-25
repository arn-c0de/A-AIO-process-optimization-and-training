"""Dataset directories on disk: atomic replace, refusal to replace non-datasets, validation of empty datasets, merge cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from simple_sim.dataset_store import atomic_dataset_dir, check_replaceable
from simple_sim.generation.dataset import generate_dataset
from tests.helpers import ROOT
from tests.helpers import empty_dataset as _empty_dataset
from tests.helpers import tiny_run_config as _config


def test_generate_refuses_to_replace_a_directory_that_is_no_dataset(tmp_path):
    runs = tmp_path / "runs"
    (runs / "run_A").mkdir(parents=True)
    (runs / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="not a dataset"):
        generate_dataset(_config(), runs, project_root=ROOT)
    assert (runs / "notes.txt").read_text(encoding="utf-8") == "keep me" and (runs / "run_A").is_dir()
    assert not runs.with_name("runs.tmp").exists()

    check_replaceable(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    check_replaceable(tmp_path / "empty")
    old = tmp_path / "old_ds"
    (old / "images").mkdir(parents=True)
    with atomic_dataset_dir(old) as tmp:  # a (manifest-less) dataset is still replaced
        (tmp / "meta.jsonl").write_text("", encoding="utf-8")
    assert (old / "meta.jsonl").exists()


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


def test_validate_fails_an_empty_dataset_and_foreign_split_ids(tmp_path, capsys):
    from simple_sim.dataset_validation import run_validation

    ds = _empty_dataset(tmp_path / "ds")
    ok, checks = run_validation(ds)
    by_name = {c["name"]: c["status"] for c in checks}
    assert not ok and by_name["non_empty"] == "fail"

    (ds / "splits" / "train.txt").write_text("bogus/id/train/999999\n", encoding="utf-8")
    ok, checks = run_validation(ds)
    by_name = {c["name"]: c["status"] for c in checks}
    assert not ok and by_name["split_membership"] == "fail"
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
