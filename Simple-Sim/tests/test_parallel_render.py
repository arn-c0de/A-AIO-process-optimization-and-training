"""Parallel 2D rendering: ordered thread map, thread-local supersampling, fast alpha blend."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.generators.opencv2d.photoreal.imageops import blend  # noqa: E402
from simple_sim.parallel import imap_ordered, render_workers  # noqa: E402
from tests.test_photoreal import CLASSES, FOOTPRINTS, _render  # noqa: E402


def test_imap_ordered_keeps_order_and_matches_sequential():
    def slow_square(x):
        time.sleep(0.001 * ((7 * x) % 5))  # finish out of order
        return x * x

    items = list(range(40))
    assert list(imap_ordered(slow_square, items, workers=6)) == [x * x for x in items]
    assert list(imap_ordered(slow_square, items, workers=1)) == [x * x for x in items]
    assert list(imap_ordered(slow_square, [], workers=4)) == []


def test_imap_ordered_stop_and_errors():
    stop = threading.Event()
    seen = []
    for v in imap_ordered(lambda x: x, range(1000), workers=4, should_stop=stop.is_set):
        seen.append(v)
        if v == 5:
            stop.set()
    assert seen == list(range(6))

    def boom(x):
        if x == 3:
            raise ValueError("bad item")
        return x

    with pytest.raises(ValueError, match="bad item"):
        list(imap_ordered(boom, range(10), workers=3))


def test_render_workers(monkeypatch):
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "3")
    assert render_workers() == 3
    assert render_workers(5) == 5
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "x")
    assert 1 <= render_workers() <= 8


def test_parallel_renders_with_mixed_supersampling_match_sequential():
    """Each render has its own RenderContext: concurrent renders with different factors do not interfere."""
    jobs = [(fp, cls, seed, ss) for fp in FOOTPRINTS for cls in CLASSES[:4] for seed in (1, 2) for ss in (1, 2, 4)]
    sequential = [_render(fp, cls, seed=seed, supersampling=ss) for fp, cls, seed, ss in jobs]
    with ThreadPoolExecutor(8) as ex:
        parallel = list(ex.map(lambda j: _render(j[0], j[1], seed=j[2], supersampling=j[3]), jobs))
    assert all(np.array_equal(a, b) for a, b in zip(sequential, parallel))


def test_blend_matches_full_image_blend():
    rng = np.random.default_rng(0)
    for per_pixel in (False, True):
        img = rng.uniform(0, 255, (64, 80, 3)).astype(np.float32)
        alpha = np.zeros((64, 80), dtype=np.float32)
        alpha[10:30, 5:40] = rng.uniform(0, 1, (20, 35)).astype(np.float32)
        color = rng.uniform(0, 255, (64, 80, 3)).astype(np.float32) if per_pixel else np.array([1.0, 2.0, 3.0], np.float32)
        expected = img * (1.0 - alpha[..., None]) + color * alpha[..., None]
        got = img.copy()
        blend(got, color, alpha)
        assert np.array_equal(got, expected.astype(np.float32))
    untouched = img.copy()
    blend(untouched, color, np.zeros((64, 80), dtype=np.float32))
    assert np.array_equal(untouched, img)


def test_generate_dataset_parallel_matches_sequential(tmp_path, monkeypatch):
    """Threaded and multi-process 2D dataset rendering write the same images as a sequential run."""
    import cv2
    import yaml

    from simple_sim.generation.dataset import generate_dataset

    root = Path(__file__).parent.parent
    cfg = yaml.safe_load((root / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {name: 3 for name in list(cfg["classes"])[:4]}
    cfg["roi"].update(width_px=96, height_px=96)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    outs = {}
    for workers, executor in (("1", "thread"), ("4", "thread"), ("2", "process")):
        monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", workers)
        monkeypatch.setenv("SIMPLE_SIM_RENDER_EXECUTOR", executor)
        out = tmp_path / f"ds_{workers}"
        generate_dataset(cfg_path, out, project_root=root)
        outs[workers] = out
    imgs = sorted(p.relative_to(outs["1"]) for p in (outs["1"] / "images").glob("*.png"))
    assert len(imgs) == 12
    for rel in imgs:
        ref = cv2.imread(str(outs["1"] / rel))
        assert np.array_equal(ref, cv2.imread(str(outs["4"] / rel))) and np.array_equal(ref, cv2.imread(str(outs["2"] / rel)))
    labels = (outs["1"] / "labels.jsonl").read_text()
    assert labels == (outs["4"] / "labels.jsonl").read_text() == (outs["2"] / "labels.jsonl").read_text()


def test_parallel_board_bake_matches_sequential(tmp_path, monkeypatch):
    """Boards baked in parallel threads are identical to a sequential bake (same files, same scene)."""
    import cv2

    from simple_sim.generators.blender3d.bake import prepare_jobs
    from tests.test_blender_bake import _job

    jobs = [_job(cls) for cls in ("OK", "MISSING", "TOMBSTONE", "SOLDER_BALL")]
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "1")
    seq = prepare_jobs(jobs, tmp_path / "seq")
    monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", "4")
    par = prepare_jobs(jobs, tmp_path / "par")
    for a, b in zip(seq, par):
        ta, tb = a["board"].pop("texture"), b["board"].pop("texture")
        assert a == b
        assert np.array_equal(cv2.imread(ta), cv2.imread(tb))


def test_generate_profile_dataset_parallel_matches_sequential(tmp_path, monkeypatch):
    """Mixed-profile datasets keep one sampling rng per record into the render: the images must not
    depend on the worker count either (separate path from ``generate_dataset``)."""
    import cv2
    import yaml

    from simple_sim.generation.profile_dataset import generate_profile_dataset

    root = Path(__file__).parent.parent
    cfg = yaml.safe_load((root / "configs" / "run_profile_cls_quick.yaml").read_text(encoding="utf-8"))
    cfg["run"]["component_profiles"] = cfg["run"]["component_profiles"][:2]
    cfg["classes"] = {name: 2 for name in list(cfg["classes"])[:3]}
    cfg["roi"].update(width_px=96, height_px=96)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    outs = {}
    for workers, executor in (("1", "thread"), ("4", "thread"), ("2", "process")):
        monkeypatch.setenv("SIMPLE_SIM_RENDER_WORKERS", workers)
        monkeypatch.setenv("SIMPLE_SIM_RENDER_EXECUTOR", executor)
        out = tmp_path / f"ds_{workers}"
        generate_profile_dataset(cfg_path, out, project_root=root)
        outs[workers] = out
    imgs = sorted(p.relative_to(outs["1"]) for p in (outs["1"] / "images").rglob("*.png"))
    assert len(imgs) >= 6
    for rel in imgs:
        ref = cv2.imread(str(outs["1"] / rel))
        assert np.array_equal(ref, cv2.imread(str(outs["4"] / rel))) and np.array_equal(ref, cv2.imread(str(outs["2"] / rel)))
    labels = (outs["1"] / "labels.jsonl").read_text()
    assert labels == (outs["4"] / "labels.jsonl").read_text() == (outs["2"] / "labels.jsonl").read_text()


def test_process_rendering_falls_back_to_threads_without_a_main_guard(tmp_path):
    """A script that calls generate_dataset at module level (no ``__main__`` guard) breaks spawned
    workers; generation must still finish, with the same images as a thread run."""
    import subprocess
    import sys

    import cv2

    root = Path(__file__).parent.parent
    script = tmp_path / "gen.py"
    script.write_text(f"""
import os, sys
sys.path.insert(0, {str(root)!r})
os.environ["SIMPLE_SIM_RENDER_WORKERS"] = "2"
os.environ["SIMPLE_SIM_RENDER_EXECUTOR"] = sys.argv[1]
from pathlib import Path
from simple_sim.config import load_config
from simple_sim.generation.dataset import generate_dataset
cfg = load_config(Path({str(root / 'configs' / 'run_0001.yaml')!r}))
cfg["classes"] = {{"OK": 2, "MISSING": 1}}
cfg["roi"].update(width_px=64, height_px=64)
generate_dataset(cfg, Path(sys.argv[2]), project_root=Path({str(root)!r}))
""", encoding="utf-8")
    for executor in ("process", "thread"):
        proc = subprocess.run([sys.executable, str(script), executor, str(tmp_path / executor)], cwd=tmp_path,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
    imgs = sorted((tmp_path / "thread" / "images").glob("*.png"))
    assert len(imgs) == 3
    for p in imgs:
        assert np.array_equal(cv2.imread(str(p)), cv2.imread(str(tmp_path / "process" / "images" / p.name)))
