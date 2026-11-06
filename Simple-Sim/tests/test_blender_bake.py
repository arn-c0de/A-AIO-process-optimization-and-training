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
