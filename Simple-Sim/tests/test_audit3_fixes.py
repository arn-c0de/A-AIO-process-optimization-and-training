"""Regression tests for the third audit of 2026-11-10 (see CHANGELOG: extend, resume, reports, tools)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from simple_sim import reports
from simple_sim.dataset_store import atomic_dataset_dir, check_replaceable
from simple_sim.generation.dataset import generate_dataset
from simple_sim.generation.render_options import look_differences
from simple_sim.images import list_synthetic_images
from simple_sim.manifest import RendererMismatchError, write_multi_profile_manifest
from simple_sim.rows import LabelRow, extend_jsonl_pair, read_jsonl, validate_jsonl_pair

ROOT = Path(__file__).resolve().parents[1]


def _config(**classes: int) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))
    cfg["roi"] = {"width_px": 64, "height_px": 64, "mm_per_px": 0.04}
    cfg["classes"] = classes or {"OK": 4, "MISSING": 4}
    return cfg


# ---------------------------------------------------------------------------
# generate / extend
# ---------------------------------------------------------------------------


def test_extend_refuses_another_roi_size_or_scale(tmp_path):
    cfg = _config()
    assert look_differences(cfg, cfg) == []
    other = {**cfg, "roi": {**cfg["roi"], "width_px": 32, "height_px": 32}}
    assert any(d.startswith("roi.width_px") for d in look_differences(cfg, other))
    scaled = {**cfg, "roi": {**cfg["roi"], "mm_per_px": 0.08}}
    assert any(d.startswith("roi.mm_per_px") for d in look_differences(cfg, scaled))

    data = tmp_path / "ds"
    generate_dataset(cfg, data, project_root=ROOT)
    with pytest.raises(RendererMismatchError, match="roi.width_px"):
        generate_dataset(other, data, extend=True, project_root=ROOT)
    assert len(read_jsonl(data / "labels.jsonl", LabelRow)) == 8


def test_extend_with_a_new_class_adds_it_to_the_dataset_config(tmp_path):
    data = tmp_path / "ds"
    generate_dataset(_config(OK=4, MISSING=4), data, project_root=ROOT)
    res = generate_dataset(_config(TOMBSTONE=6), data, extend=True, project_root=ROOT)
    assert res["extend"] is True
    stored = yaml.safe_load((data / "config.yaml").read_text(encoding="utf-8"))
    # Training reads its class list from config.yaml: without the entry the new samples were dropped.
    assert stored["classes"] == {"OK": 4, "MISSING": 4, "TOMBSTONE": 6}
    validate_jsonl_pair(data / "meta.jsonl", data / "labels.jsonl")
    assert len(read_jsonl(data / "labels.jsonl", LabelRow)) == 14
    assert not list(data.glob("*.tmp"))


def test_extend_into_an_empty_directory_is_a_fresh_dataset(tmp_path):
    data = tmp_path / "ds"
    data.mkdir()
    res = generate_dataset(_config(), data, extend=True, project_root=ROOT)
    assert res["extend"] is False and (data / "dataset_manifest.json").exists()


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


def test_extend_jsonl_pair_replaces_both_files_or_none(tmp_path):
    meta, labels = tmp_path / "meta.jsonl", tmp_path / "labels.jsonl"
    meta.write_text('{"id": "a"}', encoding="utf-8")  # no trailing newline
    labels.write_text('{"id": "a"}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot_meta.jsonl"
    os.link(meta, snapshot)
    row = LabelRow(schema_version=1, id="b", class_name="OK")

    extend_jsonl_pair(meta, [row], labels, [row])
    assert [json.loads(line)["id"] for line in meta.read_text(encoding="utf-8").splitlines()] == ["a", "b"]
    assert len(labels.read_text(encoding="utf-8").splitlines()) == 2
    assert snapshot.read_text(encoding="utf-8") == '{"id": "a"}'  # a hardlinked copy keeps the old rows

    before = (meta.read_bytes(), labels.read_bytes())
    with pytest.raises(TypeError):
        extend_jsonl_pair(meta, [row], labels, [object()])  # labels cannot be written
    assert (meta.read_bytes(), labels.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.jsonl", "meta.jsonl", "snapshot_meta.jsonl"]


def test_validate_jsonl_pair_reports_duplicate_ids(tmp_path):
    data = tmp_path / "ds"
    generate_dataset(_config(), data, project_root=ROOT)
    for name in ("meta.jsonl", "labels.jsonl"):
        lines = (data / name).read_text(encoding="utf-8").splitlines()
        (data / name).write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample ids"):
        validate_jsonl_pair(data / "meta.jsonl", data / "labels.jsonl")


def test_extending_a_multi_profile_manifest_keeps_foreign_keys(tmp_path):
    existing = {
        "manifest_version": 2, "created_at": "2026-01-01T00:00:00", "run_id": "r",
        "generator": {"version": "1", "git_commit": None, "script": "tools/create_multi_dataset.py", "renderer_version": "2"},
        "dataset_stats": {"total_samples": 1}, "extend_history": [{"samples_added": 1}],
        "real_crops": [{"added": 3}],
    }
    rows = [LabelRow(schema_version=2, id=f"s{i}", class_name="OK", profile_id="p@1") for i in range(3)]
    write_multi_profile_manifest(
        output_dir=tmp_path, run_id="r", profiles=[{"profile_id": "p@1", "profile_hash": "sha256:x", "profile_path": "p"}],
        meta_rows=rows, label_rows=rows, splits={"train": ["s0", "s1"], "val": ["s2"], "test": []},
        script_name="scripts/generate.py", existing=existing, renderer_version="3", git_commit="abc",
    )
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["real_crops"] == [{"added": 3}]
    assert manifest["generator"]["script"] == "scripts/generate.py"  # the script that wrote it last
    assert manifest["generator"]["renderer_version"] == "3"
    assert manifest["created_at"] == "2026-01-01T00:00:00" and len(manifest["extend_history"]) == 2
    assert manifest["dataset_stats"]["total_samples"] == 3


def test_gen_progress_comes_from_the_render_pass(tmp_path, monkeypatch):
    from simple_sim.generation import dataset as dataset_mod
    from simple_sim.generation import render_backends

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(render_backends, "emit", lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(dataset_mod, "emit", lambda name, **kw: events.append((name, kw)))
    data = tmp_path / "ds"
    generate_dataset(_config(OK=6, MISSING=6), data, project_root=ROOT)
    progress = [kw for name, kw in events if name == "gen_progress"]
    assert [p["done"] for p in progress] == [10, 12] and progress[-1]["total"] == 12
    # The images are written below image_root (<out>.tmp for a new dataset) when the event is sent.
    assert all(Path(p["image_root"]).name == "ds.tmp" for p in progress)
    assert (data / progress[-1]["last_image_path"]).exists()


def test_profile_dataset_does_not_need_a_domain_named_domain_a(tmp_path):
    from simple_sim.generation.profile_dataset import generate_profile_dataset

    cfg = yaml.safe_load((ROOT / "configs" / "run_profile_cls_quick.yaml").read_text(encoding="utf-8"))
    (name, domain), = list(cfg["domains"].items())[:1] if len(cfg["domains"]) == 1 else [next(iter(cfg["domains"].items()))]
    cfg["domains"] = {"lab": domain}
    cfg["splits"] = {k: (["lab"] if k.endswith("_domains") else v) for k, v in cfg["splits"].items()}
    cfg["roi"] = {**cfg["roi"], "width_px": 64, "height_px": 64}
    cfg["classes"] = dict.fromkeys(list(cfg["classes"])[:2], 2)
    res = generate_profile_dataset(cfg, tmp_path / "ds", project_root=ROOT)
    assert res["total_samples"] > 0 and name != "lab"


# ---------------------------------------------------------------------------
# training / reports
# ---------------------------------------------------------------------------


def test_resume_best_score_only_counts_for_the_same_dataset(tmp_path):
    from simple_sim.training import _other_dataset, _require_samples, hash_file

    ds = tmp_path / "A"
    ds.mkdir()
    manifest = ds / "dataset_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    same = {"trained_on_dataset": str(ds), "dataset_manifest_hash": hash_file(manifest)}
    assert not _other_dataset(same, ds, manifest)
    assert not _other_dataset({}, ds, manifest)  # old checkpoints: no entries, treated as the same dataset
    assert _other_dataset({**same, "trained_on_dataset": str(tmp_path / "B")}, ds, manifest)
    manifest.write_text('{"extended": true}', encoding="utf-8")
    assert _other_dataset(same, ds, manifest)  # extended since: the val split is another one

    _require_samples(ds, train=3, val=1)
    with pytest.raises(ValueError, match="val"):
        _require_samples(ds, train=3, val=0)


def test_resume_into_a_new_out_keeps_the_best_epoch_of_this_run(tmp_path, monkeypatch):
    from simple_sim import training
    from simple_sim.models import load_checkpoint

    cfg = _config(OK=4, MISSING=4, MISALIGNED=4)
    cfg["train"].update({"epochs": 1, "batch_size": 4, "pretrained": False, "num_workers": 0})
    data = tmp_path / "ds"
    generate_dataset(cfg, data, project_root=ROOT)

    scores = iter([0.9, 0.6, 0.8, 0.7])
    real_fit = training.fit_epoch

    def fit(run, epoch):
        stats = real_fit(run, epoch)
        stats["val_macro_f1"] = next(scores)
        return stats

    monkeypatch.setattr(training, "fit_epoch", fit)
    first = tmp_path / "first.pt"
    training.train_model(data, first, device="cpu")
    second = tmp_path / "second.pt"
    res = training.train_model(data, second, resume=first, extra_epochs=3, device="cpu", out_mode="best")
    # Same dataset: the resumed 0.9 stays the best score of the lineage, but the new file holds the
    # best epoch of this run (0.8, epoch 3), not just the first one written (0.6).
    assert res["best_val_f1"] == pytest.approx(0.9)
    assert load_checkpoint(second)["epoch"] == 3
    assert load_checkpoint(first)["epoch"] == 1


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


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_dv_crops_reexport_removes_crops_whose_review_changed(tmp_path):
    from tools.dv_crops import crop_stem, remove_stale

    out = tmp_path / "out"
    for folder, name in (("OK", "b1_R1.png"), ("OK", "b1_R2.png"), ("TOMBSTONE@grabstein", "b1_R3.png"), ("OK", "mine.png")):
        (out / folder).mkdir(parents=True, exist_ok=True)
        (out / folder / name).write_bytes(b"x")
    stems = {"b1_R1", "b1_R2", "b1_R3"}
    keep = {out / "MISSING" / "b1_R1.png", out / "OK" / "b1_R2.png", out / "TOMBSTONE" / "b1_R3.png"}
    assert remove_stale(out, stems, keep) == 2
    assert sorted(p.name for p in (out / "OK").iterdir()) == ["b1_R2.png", "mine.png"]  # foreign files stay
    assert not (out / "TOMBSTONE@grabstein").exists()
    assert crop_stem({"run": "b2", "id": "R1"}) == "b2_R1" and crop_stem({"board": "b1", "run": "x", "id": "R9"}) == "b1_R9"


def test_sim_to_real_tools_skip_mixed_in_real_crops_and_balance_evenly(tmp_path):
    from tools.sim_to_real import one_nn_accuracy

    data = tmp_path / "ds"
    (data / "images" / "real").mkdir(parents=True)
    (data / "meta.jsonl").write_text("", encoding="utf-8")
    for rel in ("images/000000.png", "images/000001.png", "images/real/000002.png"):
        (data / rel).write_bytes(b"x")
    assert [p.name for p in list_synthetic_images(data / "images")] == ["000000.png", "000001.png"]
    assert [p.name for p in list_synthetic_images(data)] == ["000000.png", "000001.png"]
    assert [p.name for p in list_synthetic_images(data / "images" / "real")] == ["000002.png"]  # asked for directly

    # Synthetic features sorted class by class: the first half looks like the real ones, the second
    # does not. A prefix of the larger set would see the first class only (accuracy ~ chance).
    rng = np.random.default_rng(0)

    def cluster(axis: int, n: int) -> np.ndarray:
        x = np.eye(8)[[axis] * n] + rng.normal(0.0, 0.05, (n, 8))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    real, syn = cluster(0, 6), np.concatenate([cluster(0, 6), cluster(1, 6)])
    assert one_nn_accuracy(real, syn) > one_nn_accuracy(real, syn[:6]) + 0.1
    assert one_nn_accuracy(syn, real) == pytest.approx(one_nn_accuracy(real, syn))


def test_calibrate_camera_converts_the_blur_offset_to_sensor_pixels(monkeypatch):
    from tools import calibrate_camera as cc

    assert cc.capture_scale({"sensor_px_per_mm": [15.5, 16.5], "capture_scale": [1.0, 1.0]}, 0.01613) == pytest.approx(0.258, abs=1e-3)
    assert cc.capture_scale({"sensor_px_per_mm": [0.0, 0.0], "capture_scale": [0.4, 0.6]}, 0.01) == pytest.approx(0.5)
    assert cc.capture_scale({}, None) == 1.0

    monkeypatch.setattr(cc, "blur_offset", lambda _r, _s: 1.0)
    monkeypatch.setattr(cc, "noise_sigma", lambda _i: 1.0)
    monkeypatch.setattr(cc, "sharpness", lambda _i: 1.0)
    img = [np.zeros((8, 8), np.float32)]
    cur = {"blur_sigma": [0.5, 0.5]}
    assert cc.fit_camera(img, img, cur)["camera"]["blur_sigma"][0] == pytest.approx(np.hypot(0.5, 1.0), abs=1e-3)
    assert cc.fit_camera(img, img, cur, scale=0.25)["camera"]["blur_sigma"][0] == pytest.approx(np.hypot(0.5, 0.25), abs=1e-3)


def test_eval_real_reads_the_roi_of_an_ensemble():
    from tools.eval_real import roi_size

    assert roi_size({"config": {"roi": {"width_px": 64, "height_px": 48}}}, 256) == (64, 48)
    ensemble = {"models": [{"checkpoint": {"config": {"roi": {"width_px": 128, "height_px": 128}}}}]}
    assert roi_size(ensemble, 256) == (128, 128)
    assert roi_size({}, 256) == (256, 256)


# ---------------------------------------------------------------------------
# renderer version 4
# ---------------------------------------------------------------------------


def test_render_stream_is_independent_of_the_sampling_stream():
    from simple_sim.generation.common import sample_record  # noqa: F401 (the sampler uses default_rng(seed))
    from simple_sim.rng import STREAM_BAKE, STREAM_POST, render_seed

    seeds = np.arange(1, 400)
    sampling = np.array([np.random.default_rng(int(s)).random() for s in seeds])
    for stream in (None, STREAM_BAKE, STREAM_POST):
        key = [render_seed(int(s)) if stream is None else render_seed(int(s), stream) for s in seeds]
        render = np.array([np.random.default_rng(k).random() for k in key])
        # The first render draw used to equal the first sampling draw (correlation 1.0).
        assert abs(np.corrcoef(sampling, render)[0, 1]) < 0.2


def test_render_record_does_not_replay_the_sampling_draws(monkeypatch):
    from simple_sim.generators.opencv2d import spec as spec_mod

    seen: dict[str, float] = {}
    monkeypatch.setattr(spec_mod, "render_sample", lambda _s, _n, _d, _a, rng, **_kw: seen.setdefault("first", rng.random()))
    spec_mod.render_record(None, {"seed": 1234, "nominal": {}, "defect": {}, "augment": {}})
    assert seen["first"] != np.random.default_rng(1234).random()
    assert seen["first"] == np.random.default_rng([1234, 1]).random()


def test_realism_tail_is_two_sided_for_signed_ranges_only():
    from simple_sim.generators.opencv2d.realism import _tri_sample

    rng = np.random.default_rng(0)
    signed = np.array([_tri_sample(rng, -4.0, 0.0, 4.0) for _ in range(40000)])
    assert abs(signed.mean()) < 0.05 and signed.min() < -3.5 and signed.max() > 3.5
    magnitude = np.array([_tri_sample(rng, 0.0, 1.0, 3.0) for _ in range(20000)])
    assert magnitude.min() >= 0.55 and magnitude.max() == 3.0  # the tail of a magnitude still goes up


def test_corner_lift_and_sot23_gap_scale_with_the_supersampling():
    import cv2

    from simple_sim.board.footprints import footprint_pads
    from simple_sim.generators.opencv2d.geometry import lifted_corner_geometry

    lifts = []
    for ss in (1, 2, 4):
        box = cv2.boxPoints(((200.0 * ss, 200.0 * ss), (120.0 * ss, 60.0 * ss), 0.0))
        geo = lifted_corner_geometry(box, (200 * ss, 200 * ss), 60 * ss, 50.0, 0, scale=ss)
        lifts.append(float(np.linalg.norm(geo["lifted"][0] - box[0])) / ss)
    assert lifts[0] == pytest.approx(lifts[1]) == pytest.approx(lifts[2])  # was 8.3 / 4.1 / 2.1 camera px

    nominal = {"pad_width": 20, "pad_height": 30, "pad_spacing": 80, "pad_spacing_y": 30}
    gaps = []
    for ss in (1, 2):
        scaled = {k: v * ss for k, v in nominal.items()}
        p1, p2, _p3 = footprint_pads("sot23", scaled, (256 * ss, 256 * ss), scale=ss)
        gaps.append((p2.y - (p1.y + p1.h)) / ss)
    assert gaps[0] == pytest.approx(gaps[1]) == pytest.approx(10.0)


def test_body_geometry_keeps_the_sub_pixel_shift():
    from simple_sim.generators.opencv2d.geometry import body_geometry

    defect = {"type": "MISALIGNED", "shift_x": 1.6, "shift_y": -1.6, "rotation_deg": 0.0, "tilt_deg": 0.0}
    geo = body_geometry({"component_length": 60, "component_width": 30}, defect, None, (256, 256))
    assert geo["center"] == (130, 126)                     # whole pixels: moved by hypot(2, 2) = 2.83
    assert geo["center_f"] == pytest.approx((129.6, 126.4))  # what the label saw: hypot(1.6, 1.6) = 2.26


# ---------------------------------------------------------------------------
# batch-predict profile note, GUI helpers
# ---------------------------------------------------------------------------


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
