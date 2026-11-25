"""Tk-free tests of GUI tab logic classes (pipeline config choice, weights deletion guard, validation flags)."""

from __future__ import annotations

import yaml


def test_pipeline_uses_the_chosen_config_of_a_profile(tmp_path):
    from types import SimpleNamespace

    from gui.tabs.pipeline.logic import PipelineLogic

    (tmp_path / "configs").mkdir()
    for name, look in (("run_a.yaml", "legacy"), ("run_a_dv.yaml", "dv_top")):
        (tmp_path / "configs" / name).write_text(yaml.safe_dump({
            "run": {"run_id": name[:-5], "component_profile": "part@1"},
            "render": {"backend": "opencv_2d", "look": look}}), encoding="utf-8")
    logic = SimpleNamespace(sim_root=tmp_path)
    chosen = PipelineLogic.chosen_config_for_profile
    info = chosen(logic, "configs/run_a_dv.yaml", "part@1", "opencv_2d")
    assert info == {"path": tmp_path / "configs" / "run_a_dv.yaml", "run_id": "run_a_dv"}
    assert chosen(logic, "configs/run_a_dv.yaml", "part@1", None) is not None
    assert chosen(logic, "configs/run_a_dv.yaml", "other@1", None) is None      # another profile
    assert chosen(logic, "configs/run_a_dv.yaml", "part@1", "blender_3d") is None  # another backend
    assert chosen(logic, "configs/missing.yaml", "part@1", None) is None and chosen(logic, "", "part@1", None) is None


def test_weights_tab_only_deletes_models_in_subfolders(tmp_path):
    from types import SimpleNamespace

    from gui.tabs.weights.logic import WeightsLogic

    models = tmp_path / "outputs" / "models"
    (models / "versions" / "run_a").mkdir(parents=True)
    (models / "imports").mkdir()
    logic = SimpleNamespace(sim_root=tmp_path)
    deletable = WeightsLogic.is_deletable_checkpoint
    assert deletable(logic, models / "versions" / "run_a" / "v1.pt") and deletable(logic, models / "imports" / "x.pt")
    assert not deletable(logic, models / "run_a.pt")  # the active model of a dataset
    assert not deletable(logic, tmp_path / "elsewhere.pt")


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


def _merge_source(root, name, *, size, rows, profile_id="p@1"):
    """A dataset with *rows* ``(sample id, split, class)``, ROI *size* and a v1 manifest of *profile_id*."""
    import cv2
    import numpy as np

    from simple_sim.manifest import write_dataset_manifest
    from simple_sim.rows import LabelRow, write_jsonl
    from tests.helpers import meta_row

    ds = root / name
    (ds / "images").mkdir(parents=True)
    meta, labels = [], []
    for i, (sid, split, cls) in enumerate(rows):
        image = f"images/{i:06d}.png"
        cv2.imwrite(str(ds / image), np.full((size, size, 3), 40 + i, np.uint8))
        meta.append(meta_row(sid, image, split=split))
        labels.append(LabelRow(schema_version=1, id=sid, class_name=cls))  # v1 labels: the manifest names the profile
    write_jsonl(ds / "meta.jsonl", meta)
    write_jsonl(ds / "labels.jsonl", labels)
    (ds / "config.yaml").write_text(yaml.safe_dump({"classes": {"OK": 1, "MISSING": 1},
                                                    "roi": {"width_px": size, "height_px": size}}), encoding="utf-8")
    write_dataset_manifest(ds, "r", profile_id=profile_id, profile_hash="h", profile_path="", meta_rows=meta,
                           label_rows=labels, splits={"train": [], "val": [], "test": []}, git_commit="x")
    return ds


def test_gui_merge_uses_the_library_tolerantly(tmp_path):
    """The GUI merge (strict=False) keeps the old tolerance: label classes outside the config classes."""
    import queue

    import cv2
    import pytest

    from gui.tabs.pipeline.logic import PipelineLogic
    from simple_sim.dataset_merge import MergeError, merge_datasets
    from simple_sim.rows import LabelRow, read_jsonl

    a = _merge_source(tmp_path, "a", size=48, rows=[("r/A/train/000000", "train", "OK"),
                                                    ("r/A/test/000001", "test", "SOLDER_BALL")])
    b = _merge_source(tmp_path, "b", size=32, rows=[("r/A/val/000000", "val", "MISSING")], profile_id="q@1")
    with pytest.raises(MergeError):  # the CLI default stays strict
        merge_datasets([a, b], tmp_path / "strict")
    assert not (tmp_path / "strict").exists()

    logic = PipelineLogic(tmp_path, queue.Queue(), queue.Queue(), tmp_path / "events.jsonl")
    logged: list[str] = []
    out = tmp_path / "merged"
    result = logic.merge_datasets_for_training([a, b], out, logged.append, logged.append)
    assert result.roi == (32, 32) and len(result.meta_rows) == 3
    assert {row.class_name for row in read_jsonl(out / "labels.jsonl", LabelRow)} == {"OK", "SOLDER_BALL", "MISSING"}
    assert sorted(p.name for p in (out / "splits").iterdir()) == ["test.txt", "train.txt", "val.txt"]
    assert result.manifest_version == 2 and len(result.splits["test"]) == 1
    assert all(cv2.imread(str(out / row.image_path)).shape[:2] == (32, 32) for row in result.meta_rows)
    assert yaml.safe_load((out / "config.yaml").read_text())["roi"] == {"width_px": 32, "height_px": 32}
    assert any("3 samples" in line for line in logged)

    (b / "dataset_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="dataset_manifest.json"):
        logic.merge_datasets_for_training([a, b], tmp_path / "again", logged.append, logged.append)
    assert not (tmp_path / "again").exists()
    with pytest.raises(FileExistsError):
        logic.merge_datasets_for_training([a], out, logged.append, logged.append)


def test_pipeline_monitor_maps_telemetry_events_to_the_status_line(tmp_path):
    from types import SimpleNamespace

    from gui.tabs.pipeline.monitor_mixin import PipelineMonitorMixin

    class Var:
        def __init__(self) -> None:
            self.value = ""

        def set(self, value: str) -> None:
            self.value = value

    ui = SimpleNamespace(var_phase=Var(), var_ips=Var(), var_epoch=Var(), var_last=Var())
    mon = PipelineMonitorMixin()
    mon.ui, mon.state, mon._recent_imgs = ui, SimpleNamespace(dataset_dir=None), []
    pushed: list[tuple[str, object]] = []
    mon._push_image = lambda rel, root=None: pushed.append((rel, root))

    tmp_out, out = tmp_path / "ds.tmp", tmp_path / "ds"
    mon._handle_event({"event": "gen_start", "output_dir": str(out)})
    assert ui.var_phase.value == "phase: generating" and mon._monitor_dir == out
    mon._handle_event({"event": "gen_progress", "samp_per_s": 2.5, "image_root": str(tmp_out), "last_image_path": "a.png"})
    assert ui.var_ips.value == "img/s: 2.5" and pushed[-1] == ("a.png", tmp_out)
    mon._recent_imgs = [str(tmp_out / "a.png")]
    mon._handle_event({"event": "gen_done", "output_dir": str(out)})
    assert ui.var_phase.value == "phase: generated" and mon._recent_imgs == [str(out / "a.png")]
    assert mon._monitor_image_root is None
    mon._handle_event({"event": "gen_progress"})
    assert ui.var_ips.value == "img/s: -"

    mon._handle_event({"event": "train_start", "dataset_dir": str(tmp_path / "train")})
    assert mon._monitor_dir == tmp_path / "train"
    mon._handle_event({"event": "epoch_start", "epoch": 2, "epochs": 5})
    mon._handle_event({"event": "train_batch", "img_per_s": 10.0, "last_id": "s1", "last_image_path": "b.png"})
    assert (ui.var_epoch.value, ui.var_ips.value, ui.var_last.value) == ("epoch: 2/5", "img/s: 10.0", "last: s1")
    mon._handle_event({"event": "eval_batch", "img_per_s": 4.0, "last_id": "ignored", "last_image_path": "c.png"})
    assert (ui.var_phase.value, ui.var_last.value, pushed[-1][0]) == ("phase: evaluating", "last: s1", "c.png")
    mon._handle_event({"event": "validate_done", "ok": False})
    assert ui.var_phase.value == "phase: validated_fail"
    mon._handle_event({"event": "something_else"})
    assert ui.var_phase.value == "phase: validated_fail"


def test_board_detection_reuses_the_classifier_until_models_or_device_change(tmp_path, monkeypatch):
    from pathlib import Path

    from gui.tabs.board_detection import logic as bd_logic

    built: list[tuple[Path, Path, str]] = []

    class FakeClassifier:
        def __init__(self, *, profile_model_path, defect_bundle_path, device, min_profile_confidence):
            built.append((profile_model_path, defect_bundle_path, device))
            self.min_profile_confidence = min_profile_confidence

        def predict(self, image_path):
            return (image_path, self.min_profile_confidence)

    monkeypatch.setattr(bd_logic, "TwoStageClassifier", FakeClassifier)
    logic = bd_logic.BoardDetectionLogic()
    pm, bundle, img = tmp_path / "p.pt", tmp_path / "b.bundle", tmp_path / "i.png"
    assert logic.predict(pm, bundle, img, "cpu", 0.5) == (img, 0.5)
    assert logic.predict(pm, bundle, img, "cpu", 0.7) == (img, 0.7)  # same models: threshold updated in place
    assert len(built) == 1
    logic.predict(pm, bundle, img, "cuda", 0.7)
    logic.predict(tmp_path / "other.pt", bundle, img, "cuda", 0.7)
    assert [b[2] for b in built] == ["cpu", "cuda", "cuda"]


def test_weights_rename_moves_favorite_and_arena_entries():
    from gui.tabs.weights.logic import rekey_tracked_entry

    entries = {"outputs/models/versions/a.pt": {"path": "outputs/models/versions/a.pt", "note": "x"}, "other": {"path": "other"}}
    assert rekey_tracked_entry(entries, "outputs/models/versions/a.pt", "outputs/models/versions/b.pt")
    moved = entries["outputs/models/versions/b.pt"]
    assert moved["path"] == "outputs/models/versions/b.pt" and moved["note"] == "x" and moved["renamed"]
    assert "outputs/models/versions/a.pt" not in entries and "other" in entries
    assert not rekey_tracked_entry(entries, "missing.pt", "new.pt")


def test_weights_import_unpacks_a_bundle_zip_and_rejects_unsafe_archives(tmp_path):
    import queue
    import zipfile

    import pytest

    from gui.tabs.weights.logic import WeightsLogic

    logic = WeightsLogic(tmp_path, queue.Queue())
    archive = tmp_path / "mine.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("export/mine.bundle/p1.pt", b"x")
        zf.writestr("export/mine.bundle/bundle.json", "{}")
    first, second = logic.import_model(archive), logic.import_model(archive)
    assert first.parent.name == "bundles" and first.name.endswith("_mine.bundle") and (first / "p1.pt").is_file()
    assert second != first and second.name.endswith("_001.bundle") and (second / "p1.pt").is_file()
    assert not [p for p in first.parent.iterdir() if p.name.startswith(".tmp_import_")]

    for name, members in (("evil.zip", {"../evil.pt": b"x"}), ("empty.zip", {}), ("nopt.zip", {"a.bundle/readme.txt": b"x"})):
        with zipfile.ZipFile(tmp_path / name, "w") as zf:
            for member, data in members.items():
                zf.writestr(member, data)
        with pytest.raises(ValueError):
            logic.import_model(tmp_path / name)

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"ckpt")
    imported = logic.import_model(ckpt)
    assert imported.parent.name == "imports" and imported.read_bytes() == b"ckpt"
