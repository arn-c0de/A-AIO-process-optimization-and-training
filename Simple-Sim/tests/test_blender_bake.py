"""Board bake for the Blender backend: shared layout with 2D, no label leak, optional real render."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.blender3d.bake import BOARD_SCALE, prepare_jobs
from simple_sim.generators.opencv2d import sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import render_board

TOL = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}


def _job(defect_type, footprint="chip_2pad", seed=3):
    rng = np.random.default_rng(seed)
    return {"image_path": f"images/{defect_type}.png", "seed": seed, "mm_per_px": 0.01, "roi_width_px": 64,
            "roi_height_px": 64, "footprint": footprint, "component_height_mm": 0.45,
            "nominal": sample_nominal_geometry({}, rng, {"pad_spacing_y": [36.0, 50.0]}), "defect": sample_defect_params(defect_type, rng, TOL),
            "augment": {"rotation_deg": 0.0}, "render_3d": {}, "component_type": "chip_resistor", "tolerances": TOL}


def test_board_does_not_depend_on_the_defect():
    """Silkscreen/traces must not follow a misplaced part (would leak the label)."""
    rng = np.random.default_rng(0)
    nominal = sample_nominal_geometry({}, rng)
    boards = []
    for d in ({"type": "OK", "shift_x": 0.0, "shift_y": 0.0, "rotation_deg": 0.0, "tilt_deg": 0.0},
              {"type": "MISALIGNED", "shift_x": 20.0, "shift_y": -15.0, "rotation_deg": 25.0, "tilt_deg": 0.0}):
        img, _ = render_board(nominal, d, {}, TOL, np.random.default_rng(7), "chip_2pad", (128, 128))
        boards.append(img)
    assert np.array_equal(boards[0], boards[1])


def test_prepare_jobs_bakes_texture_and_layout(tmp_path):
    jobs = prepare_jobs([_job("OK"), _job("MISSING"), _job("TOMBSTONE")], tmp_path)
    for job in jobs:
        board = job["board"]
        assert Path(board["texture"]).is_file() and board["scale"] == BOARD_SCALE
        assert board["family"] == "resistor" and len(board["pads"]) == 2
    assert jobs[0]["board"]["body"]["is_tombstone"] is False
    assert jobs[1]["board"]["body"] is None
    assert jobs[2]["board"]["body"]["is_tombstone"] is True and jobs[2]["board"]["body"]["raised_sign"] in (1.0, -1.0)
    json.dumps(jobs)  # must stay JSONL-serialisable for the Blender process


@pytest.mark.parametrize("footprint", ["chip_2pad", "sot23"])
def test_blender_script_renders(tmp_path, footprint):
    pytest.importorskip("bpy")
    jobs = prepare_jobs([_job("OK", footprint)], tmp_path / "work")
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("\n".join(json.dumps(j) for j in jobs), encoding="utf-8")
    script = Path(__file__).parent.parent / "simple_sim" / "blender" / "render_batch.py"
    subprocess.run([sys.executable, str(script), "--", "--jobs", str(jobs_path), "--out_root", str(tmp_path),
                    "--samples", "4", "--engine", "CYCLES"], check=True, capture_output=True)
    assert (tmp_path / "images" / "OK.png").is_file()


def test_bake_supersampling_capped_for_large_boards():
    from simple_sim.generators.blender3d.bake import bake_supersampling

    assert bake_supersampling(2, 8000) == 2  # the default factor is never capped
    assert bake_supersampling(1, 512) == 1
    assert bake_supersampling(4, 512) == 4   # 256 px ROI: Ultra kept
    assert bake_supersampling(4, 2048) == 2  # 1024 px ROI (SOIC-16): capped to the pixel budget


def test_bake_fields_never_override_the_roi_scale():
    """``roi.mm_per_px`` is the one scale of a Blender job: a stale ``render.mm_per_px`` must not win."""
    from simple_sim.generation.blender import blender_job
    from simple_sim.generators.blender3d.bake import bake_fields

    fields = bake_fields({"mm_per_px": 0.05, "board": {"routing": "escape"}, "mask_gloss": 0.0}, {"mm_per_px": 0.01})
    assert "mm_per_px" not in fields
    assert fields["board"] == {"routing": "escape"} and fields["mask_gloss"] == 0.0
    record = {"image_path": "a.png", "seed": 1, "nominal": {}, "defect": {}, "augment": {}}
    job = blender_job(record, mm_per_px=0.01, roi_width=64, roi_height=64, footprint="chip_2pad",
                      component_height_mm=0.45, render_3d={}, bake=fields)
    assert job["mm_per_px"] == 0.01


def test_preview_jobs_bake_with_the_run_render_settings():
    """3D previews bake the same board as the dataset (board model, camera, gloss from ``render``)."""
    from simple_sim.preview.render import RenderSettings, build_preview_jobs
    from simple_sim.profile_hash import load_profile

    root = Path(__file__).parent.parent
    profile = load_profile("chip_0603_resistor_3d@2", root / "configs" / "profiles")
    render = {"board": {"routing": "escape"}, "camera": {"preset": "phone"}, "mask_gloss": 0.0}
    settings = RenderSettings(roi_width_px=64, roi_height_px=64, mm_per_px=0.01, blender_executable="blender",
                              cycles_samples=8, device="CPU", render=render)
    jobs, _ = build_preview_jobs(profile_id="chip_0603_resistor_3d@2", profile=profile, settings=settings,
                                 seed_base=1, defect_types=["OK"])
    assert jobs[0]["board"] == {"routing": "escape"}
    assert jobs[0]["camera"] == {"preset": "phone"} and jobs[0]["mask_gloss"] == 0.0
    assert jobs[0]["mm_per_px"] == 0.01


def _fake_blender(tmp_path, body: str) -> Path:
    """A stand-in "blender" executable (a Python script) for the runner tests."""
    exe = tmp_path / "fake_blender"
    exe.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def test_run_blender_should_stop_terminates_the_process(tmp_path):
    import time

    from simple_sim.generators.blender3d import runner

    pid_file = tmp_path / "pid"
    exe = _fake_blender(tmp_path, f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)")
    t0 = time.monotonic()
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return pid_file.exists()  # stop as soon as the fake Blender is running

    with pytest.raises(runner.BlenderCancelled):
        runner._run_blender(prepared=[], prepared_path=tmp_path / "jobs.jsonl", output_root=tmp_path,
                            blender_executable=str(exe), cycles_samples=1, device="CPU", engine="X", extra_env=None,
                            should_stop=stop)
    assert time.monotonic() - t0 < 20
    pid = int(pid_file.read_text())
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # terminated and reaped, not orphaned


def test_run_blender_without_should_stop_is_unchanged(tmp_path):
    from simple_sim.generators.blender3d import runner

    ok = _fake_blender(tmp_path, "pass")
    runner._run_blender(prepared=[], prepared_path=tmp_path / "jobs.jsonl", output_root=tmp_path,
                        blender_executable=str(ok), cycles_samples=1, device="CPU", engine="X", extra_env=None)
    bad = _fake_blender(tmp_path, "raise SystemExit(3)")
    for should_stop in (None, lambda: False):
        with pytest.raises(RuntimeError, match="exit code 3"):
            runner._run_blender(prepared=[], prepared_path=tmp_path / "jobs.jsonl", output_root=tmp_path,
                                blender_executable=str(bad), cycles_samples=1, device="CPU", engine="X", extra_env=None,
                                should_stop=should_stop)


def test_blender_script_is_found_without_a_checkout_root():
    """The batch script is package-relative (no ``sim_root``)."""
    from simple_sim.generators.blender3d import runner

    path = runner.script_path()
    assert path.name == "render_batch.py" and path.parent.name == "blender" and path.is_file()
