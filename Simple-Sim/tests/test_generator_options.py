"""``generators/options.py`` (I7) must agree with the renderer's own definitions.

The photoreal modules still define their option constants; this test fails when the two drift.
"""

from __future__ import annotations

import pytest

from simple_sim.generators import options
from simple_sim.generators.opencv2d import render as render2d
from simple_sim.generators.opencv2d.photoreal import camera, context, look, meniscus


def test_option_constants_match_renderer() -> None:
    assert options.SUPERSAMPLING_CHOICES == context.SUPERSAMPLING_CHOICES
    assert options.DEFAULT_SUPERSAMPLING == context.DEFAULT_SUPERSAMPLING
    assert options.SOLDER_STYLES == meniscus.SOLDER_STYLES
    assert options.DEFAULT_SOLDER_STYLE == meniscus.DEFAULT_SOLDER_STYLE
    assert options.LOOKS == look.LOOKS
    assert options.DEFAULT_LOOK == look.DEFAULT_LOOK
    assert options.POST_FILTER_MODES == render2d.POST_FILTER_MODES
    assert options.DEFAULT_POST_FILTERS == render2d.DEFAULT_POST_FILTERS


def test_camera_constants_match_renderer() -> None:
    assert options.CAMERA_DEFAULTS == camera.CAMERA_DEFAULTS
    assert list(options.CAMERA_DEFAULTS) == list(camera.CAMERA_DEFAULTS)
    assert options.CAMERA_PRESET_NAMES == tuple(camera.CAMERA_PRESETS)
    assert options.DEFAULT_CAMERA_PRESET in camera.CAMERA_PRESETS
    assert options.CHANNEL_AXES == camera.CHANNEL_AXES
    assert options.CAMERA_RANGE_LIMITS == camera._RANGE_LIMITS


_CAMERAS = [
    None, {}, {"preset": "aoi"}, {"preset": "phone"}, {"preset": "quins"}, {"preset": "nope"},
    "aoi", [], {"bogus": 1}, {"zoom": [1.0, 1.5]}, {"zoom": [0.5, 1.0]}, {"zoom": [2.0, 1.0]},
    {"tilt_deg": [0.0, 61.0]}, {"tilt_deg": [0, 30]}, {"jpeg_quality": [80, 101]},
    {"capture_scale": [0.01, 0.5]}, {"capture_scale": [0.2, 1.0]}, {"blur_sigma": 0.5},
    {"blur_sigma": [0.5]}, {"blur_sigma": ["a", 1]}, {"read_noise": [-1.0, 1.0]},
    {"shot_noise": -0.1}, {"shot_noise": "x"}, {"shot_noise": 0.0},
    {"channel_axis": "z"}, {"channel_axis": "x"}, {"max_oversampling": -1}, {"max_oversampling": 2.5},
    {"preset": "phone", "sensor_px_per_mm": [10.0, 12.0], "crop_part_lengths": [3.0, 3.0]},
]


def _outcome(fn, cam):
    try:
        fn(cam)
    except ValueError as e:
        return str(e)
    return None


@pytest.mark.parametrize("cam", _CAMERAS, ids=[repr(c) for c in _CAMERAS])
def test_camera_validator_matches_renderer(cam) -> None:
    assert _outcome(options.validate_camera_config, cam) == _outcome(camera.validate_camera_config, cam)
