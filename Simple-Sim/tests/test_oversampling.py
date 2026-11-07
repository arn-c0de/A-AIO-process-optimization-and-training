"""render.camera.max_oversampling: cheaper zoomed-out views (opt-in, default off)."""

import numpy as np
import pytest

from simple_sim.config import _validate_camera
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal.camera import camera_settings, effective_supersampling
from simple_sim.generation.common import sample_defect_params

from tests.test_photoreal import CFG, FOOTPRINTS, TOL


def test_effective_supersampling_caps_zoomed_out_views_only():
    cam = camera_settings(None)
    assert effective_supersampling(2, {"zoom": 3.0}, cam) == 2                      # off by default
    cam = camera_settings({"max_oversampling": 2.0})
    assert effective_supersampling(2, {"zoom": 1.0}, cam) == 2                      # 2 x 1 <= 2: unchanged
    assert effective_supersampling(2, {"zoom": 0.74}, cam) == 2                     # magnified: never raised
    assert effective_supersampling(2, {"zoom": 2.0}, cam) == 1                      # 2 x 2 > 2 -> 1
    assert effective_supersampling(4, {"zoom": 1.5}, cam) == 1                      # floor(2 / 1.5)
    assert effective_supersampling(4, {"zoom": 1.0}, camera_settings({"max_oversampling": 3})) == 3
    assert effective_supersampling(2, {"zoom": 10.0}, cam) == 1                     # never below 1
    _validate_camera({"max_oversampling": 0})
    _validate_camera({"max_oversampling": 2.5})
    with pytest.raises(ValueError, match="max_oversampling"):
        _validate_camera({"max_oversampling": -1})


def test_cap_is_a_no_op_without_zoom_and_changes_only_zoomed_out_renders():
    rng = np.random.default_rng(3)
    nominal = sample_nominal_geometry({}, rng, FOOTPRINTS["chip_2pad"])
    defect = sample_defect_params("OK", rng, TOL)
    plain = render_roi(nominal, defect, {}, (96, 96), CFG, TOL, np.random.default_rng(1), footprint="chip_2pad")
    capped = render_roi(nominal, defect, {}, (96, 96), {**CFG, "camera": {"max_oversampling": 2.0}}, TOL,
                        np.random.default_rng(1), footprint="chip_2pad")
    assert np.array_equal(plain, capped)                                            # zoom 1: identical
    zoomed = {**CFG, "camera": {"zoom": [2.0, 2.0]}}
    a = render_roi(nominal, defect, {}, (96, 96), zoomed, TOL, np.random.default_rng(1), footprint="chip_2pad")
    b = render_roi(nominal, defect, {}, (96, 96), {**zoomed, "camera": {"zoom": [2.0, 2.0], "max_oversampling": 2.0}},
                   TOL, np.random.default_rng(1), footprint="chip_2pad")
    assert a.shape == b.shape == (96, 96, 3) and not np.array_equal(a, b)           # rendered at ss 1 instead of 2
    assert abs(float(a.mean()) - float(b.mean())) < 6.0                             # same picture, less oversampling
