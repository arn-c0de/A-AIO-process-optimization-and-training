"""Camera calibration recovers a known camera setting from 'real' images rendered with it."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry

_spec = importlib.util.spec_from_file_location("calibrate_camera", Path(__file__).parent.parent / "tools" / "calibrate_camera.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

TOL = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}
CFG = {"component_color": [20, 20, 20], "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180],
       "component_type": "chip_resistor"}


def _render(camera, n=24):
    out = []
    for i in range(n):  # same seeds -> same content, only the camera differs
        rng = np.random.default_rng(i)
        nominal = sample_nominal_geometry({}, rng)
        defect = sample_defect_params(("OK", "MISSING", "MISALIGNED", "TOMBSTONE")[i % 4], rng, TOL)
        img = render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), {**CFG, "camera": camera}, TOL, rng)
        out.append(img.mean(axis=2).astype(np.float32))
    return out


def test_identical_camera_needs_no_change():
    syn = _render(None)
    result = cc.fit_camera(syn, syn)
    assert result["stats"]["extra_blur"] == 0.0 and result["stats"]["noise_variance_ratio"] == pytest.approx(1.0)


def test_recovers_blur_and_noise_of_a_known_camera():
    real = _render({"blur_sigma": [1.3, 1.5], "read_noise": [4.0, 16.0], "shot_noise": 0.16})  # 4x default noise
    result = cc.fit_camera(real, _render(None))
    lo, hi = result["camera"]["blur_sigma"]
    assert (lo + hi) / 2 == pytest.approx(1.4, abs=0.15)
    assert result["stats"]["noise_variance_ratio"] == pytest.approx(4.0, rel=0.3)
