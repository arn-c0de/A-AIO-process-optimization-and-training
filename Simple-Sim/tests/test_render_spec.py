"""``RenderSpec`` / ``render_sample`` (render one image from a run config) and the parallel 2D render path."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from simple_sim.config import domain_camera, load_config, merge_profile_render
from simple_sim.filter_settings import normalize_image_filters
from simple_sim.generation.common import render_2d_records, sample_domain, sample_record
from simple_sim.generators.opencv2d import RenderSpec, render_record, render_roi, render_sample
from simple_sim.parallel import PROCESS_MIN_ITEMS_PER_WORKER, render_executor
from simple_sim.profile_hash import load_profile

ROOT = Path(__file__).resolve().parent.parent


def _setup(run: str):
    cfg = load_config(ROOT / "configs" / f"{run}.yaml")
    profile = load_profile(cfg["run"]["component_profile"], ROOT / "configs" / "profiles")
    return cfg, profile


def _records(cfg, profile, n: int, classes=("OK", "TOMBSTONE", "SOLDER_BRIDGE", "MISSING")):
    out = []
    for i in range(n):
        rec, _ = sample_record(i, classes[i % len(classes)], config=cfg, run_seed=cfg["run"]["seed"],
                               geometry_ranges=profile.get("geometry_ranges"), tolerances=profile.get("tolerances"),
                               image_filters=normalize_image_filters(None), enable_cardinal_rotation_90=True,
                               footprint=profile["component"]["footprint"], domain=sample_domain(cfg, i))
        out.append(rec)
    return out


def test_from_run_config_merges_profile_and_domain_cameras():
    cfg, profile = _setup("run_0001_xdomain")
    spec = RenderSpec.from_run_config(cfg, profile)
    assert spec.roi_size == (cfg["roi"]["width_px"], cfg["roi"]["height_px"])
    assert spec.footprint == profile["component"]["footprint"]
    assert spec.render == merge_profile_render(cfg["render"], profile, cfg["roi"])
    for name, dcfg in cfg["domains"].items():
        camera = domain_camera(spec.render, dcfg)
        assert spec.render_config(name).get("camera") == (camera if camera is not None else spec.render.get("camera"))
    assert spec.render_config("unknown_domain") == dict(spec.render)
    assert pickle.loads(pickle.dumps(spec)) == spec


@pytest.mark.parametrize("run", ["run_0001", "run_0001_xdomain", "run_sot23"])
def test_render_sample_equals_the_hand_merged_render_roi(run):
    cfg, profile = _setup(run)
    spec = RenderSpec.from_run_config(cfg, profile)
    render = merge_profile_render(cfg["render"], profile, cfg["roi"])
    for rec in _records(cfg, profile, 4):
        camera = domain_camera(render, cfg["domains"][rec["domain"]])
        expected = render_roi(rec["nominal"], rec["defect"], rec["augment"], spec.roi_size,
                              {**render, "camera": camera} if camera is not None else render, profile.get("tolerances"),
                              np.random.default_rng(rec["seed"]), profile["component"]["footprint"])
        got = render_sample(spec, rec["nominal"], rec["defect"], rec["augment"], np.random.default_rng(rec["seed"]),
                            domain=rec["domain"])
        assert np.array_equal(got, expected)
        assert np.array_equal(render_record(spec, rec), expected)


def test_process_and_thread_rendering_write_the_same_images(tmp_path, monkeypatch):
    cfg, profile = _setup("run_qfn32")
    spec = RenderSpec.from_run_config(cfg, profile)
    recs = _records(cfg, profile, 6)
    tasks = [(spec, r, None) for r in recs]
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "2")
    seen = []
    render_2d_records(tasks, tmp_path / "thread", executor="thread", progress=lambda done, total: seen.append((done, total)))
    render_2d_records(tasks, tmp_path / "process", executor="process")
    assert seen[-1] == (6, 6)
    for r in recs:
        a = cv2.imread(str(tmp_path / "thread" / r["image_path"]), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(tmp_path / "process" / r["image_path"]), cv2.IMREAD_UNCHANGED)
        assert a is not None and np.array_equal(a, b)
        assert np.array_equal(a, render_record(spec, r))


def test_process_rendering_continues_a_passed_generator(tmp_path, monkeypatch):
    """Tasks may carry their own generator (generate-profile-dataset continues the sampling rng)."""
    cfg, profile = _setup("run_0001")
    spec = RenderSpec.from_run_config(cfg, profile)
    recs = _records(cfg, profile, 3)
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "2")
    render_2d_records([(spec, r, np.random.default_rng([r["seed"], 9])) for r in recs], tmp_path, executor="process")
    for r in recs:
        expected = render_record(spec, r, np.random.default_rng([r["seed"], 9]))
        assert np.array_equal(cv2.imread(str(tmp_path / r["image_path"]), cv2.IMREAD_UNCHANGED), expected)


def test_render_executor_choice(monkeypatch):
    monkeypatch.delenv("SIMPLE_SIM_RENDER_EXECUTOR", raising=False)
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "4")
    assert render_executor(10) == "thread"  # too few images to pay for starting processes
    assert render_executor(4 * PROCESS_MIN_ITEMS_PER_WORKER) == "process"
    assert render_executor(10_000, "thread") == "thread"
    monkeypatch.setenv("SIMPLE_SIM_RENDER_EXECUTOR", "process")
    assert render_executor(1) == "process"
    monkeypatch.setenv("SIMPLE_SIM_RENDER_EXECUTOR", "gpu")
    with pytest.raises(ValueError, match="render executor"):
        render_executor(1)
    monkeypatch.delenv("SIMPLE_SIM_RENDER_EXECUTOR")
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "1")
    assert render_executor(10_000) == "thread"  # one worker: no pool at all
