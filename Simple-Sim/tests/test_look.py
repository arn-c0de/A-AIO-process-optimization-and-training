"""``render.look``: target looks of the photoreal renderer (legacy default, QUINS AOI scans)."""

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import (LOOKS, RenderContext, camera_config, camera_model, camera_settings,
                                                      get_look, mask_gloss, rework_probability)
from simple_sim.generators.opencv2d.photoreal.look import LEGACY_LOOK, QUINS_LOOK

from tests.test_photoreal import CFG as BASE_CFG
from tests.test_photoreal import CLASSES, FOOTPRINTS, TOL

CFG = {**BASE_CFG, "mm_per_px": 0.01}


def _render(footprint="chip_2pad", defect_type="OK", seed=1, **cfg):
    rng = np.random.default_rng(seed)
    nominal = sample_nominal_geometry({}, rng, FOOTPRINTS[footprint])
    defect = sample_defect_params(defect_type, rng, TOL)
    return render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), {**CFG, **cfg}, TOL, rng, footprint=footprint)


def test_legacy_is_the_default_and_explicit_legacy_is_identical():
    assert get_look(None) is LEGACY_LOOK and get_look({"look": "legacy"}) is LEGACY_LOOK
    assert RenderContext().look is LEGACY_LOOK
    assert np.array_equal(_render(seed=3), _render(seed=3, look="legacy"))


def test_unknown_look_is_rejected():
    from simple_sim.config import _validate_look

    for name in LOOKS:
        _validate_look(name)
    _validate_look(None)
    with pytest.raises(ValueError, match="render.look"):
        _validate_look("phone")
    with pytest.raises(ValueError, match="render.look"):
        get_look({"look": "nope"})


@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
@pytest.mark.parametrize("defect_type", CLASSES)
def test_quins_renders_every_footprint_and_class(footprint, defect_type):
    img = _render(footprint, defect_type, look="quins")
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert 20 < img.mean() < 200


def test_quins_is_deterministic_and_differs_from_legacy():
    a, b = _render(seed=5, look="quins"), _render(seed=5, look="quins")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _render(seed=5))


def test_quins_context_uses_scanner_joints_unless_solder_is_set():
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    ctx = RenderContext.for_render({"look": "quins"}, rng)
    assert rng.bit_generator.state == state  # side stream only: the image's random stream is untouched
    assert ctx.look is QUINS_LOOK and ctx.solder_look is not None and ctx.solder_look.azimuths == ()
    assert RenderContext.for_render({"look": "quins", "solder": "legacy"}, rng).solder_look is None


def test_look_defaults_yield_to_explicit_settings():
    assert camera_config({}) is None
    assert camera_config({"look": "quins"}) == {"preset": "quins"}
    assert camera_config({"look": "quins", "camera": {"read_noise": [1, 2]}}) == {"preset": "quins", "read_noise": [1, 2]}
    assert camera_config({"look": "quins", "camera": {"preset": "phone"}}) == {"preset": "phone"}
    assert mask_gloss({}, 0.5) == 0.5 and mask_gloss({"look": "quins"}, 0.5) == 0.0
    assert mask_gloss({"look": "quins", "mask_gloss": 0.7}, 0.5) == 0.7
    assert rework_probability({}, 0.25) == 0.25 and rework_probability({"look": "quins"}, 0.25) == QUINS_LOOK.rework
    assert rework_probability({"look": "quins", "rework_artifacts": 1.0}, 0.25) == 1.0


def test_quins_body_colours_replace_only_the_dark_default():
    dark, brown = np.array((20, 20, 20), np.float32), np.array((60, 90, 140), np.float32)
    assert np.array_equal(LEGACY_LOOK.body_colour("qfn", dark), np.full(3, 34.0, np.float32))
    assert np.array_equal(QUINS_LOOK.body_colour("resistor", dark), QUINS_LOOK.bodies["resistor"])
    assert np.array_equal(QUINS_LOOK.body_colour("sot", dark), QUINS_LOOK.bodies["*"])
    assert float(QUINS_LOOK.bodies["led"].mean()) > 150  # white LED housing


def test_quins_mask_matches_the_measured_scan_colour():
    """The bare board of a MISSING part: mask median close to the QUINS scans (BGR ~ (67, 87, 49))."""
    meds = np.array([np.median(_render(defect_type="MISSING", seed=s, look="quins").reshape(-1, 3), axis=0)
                     for s in range(6)])
    assert np.all(np.abs(np.median(meds, axis=0) - (67, 87, 49)) < 14)


def test_sensor_resolution_equals_the_matching_capture_scale():
    img = np.random.default_rng(2).integers(0, 255, (96, 128, 3)).astype(np.uint8)
    by_sensor = camera_model(img, np.random.default_rng(1), {"sensor_px_per_mm": [25.0, 25.0]}, mm_per_px=0.01)
    by_scale = camera_model(img, np.random.default_rng(1), {"capture_scale": [0.25, 0.25]})
    assert np.array_equal(by_sensor, by_scale)
    with pytest.raises(ValueError, match="mm_per_px"):
        camera_model(img, np.random.default_rng(1), {"sensor_px_per_mm": [25.0, 25.0]})


def test_channel_shift_separates_blue_and_red_at_edges():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, 32:] = 200
    quiet = {"read_noise": [0, 0], "shot_noise": 0, "blur_sigma": [0.3, 0.3]}
    plain = camera_model(img, np.random.default_rng(1), quiet).astype(int)
    split = camera_model(img, np.random.default_rng(1), {**quiet, "channel_shift": [2.0, 2.0]}).astype(int)
    assert np.abs(plain[..., 0] - plain[..., 2]).max() <= 2
    assert np.abs(split[..., 0] - split[..., 2]).max() > 40


def test_quins_camera_preset_validates():
    from simple_sim.config import _validate_camera

    _validate_camera({"preset": "quins", "sensor_px_per_mm": [20, 26], "channel_shift": [0, 1]})
    assert camera_settings({"preset": "quins"})["sensor_px_per_mm"] == [22.0, 25.5]
    with pytest.raises(ValueError):
        _validate_camera({"channel_shift": [1, 0]})


def test_look_stats_tool(tmp_path):
    import cv2

    from tools.look_stats import folder_stats, format_table

    for s in range(3):
        cv2.imwrite(str(tmp_path / f"{s}.png"), _render(seed=s, look="quins"))
    stats = folder_stats(tmp_path, limit=10, size=128)
    assert stats["images"] == 3 and len(stats["mask_h"]) == 3 and 60 < stats["mask_h"][1] < 85
    assert "mask_h" in format_table({"q": stats})


def test_crop_part_lengths_sets_the_field_of_view():
    from simple_sim.generators.opencv2d.photoreal import apply_view
    from simple_sim.generators.opencv2d.photoreal.camera import fit_crop

    pose = {"tilt": 0.0, "azimuth": 0.0, "zoom": 1.0, "roll": 0.0}
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert fit_crop(pose, camera_settings(None), rng, 62.0, (256, 256)) is pose  # off: unchanged, no draw
    assert rng.bit_generator.state == state
    fitted = fit_crop(pose, camera_settings({"preset": "quins"}), rng, 62.0, (256, 256))
    assert fitted["zoom"] == pytest.approx(3.0 * 62.0 / 256.0)
    canvas = np.zeros((187, 187, 3), np.uint8)
    assert apply_view(canvas, fitted, (256, 256)).shape == (256, 256, 3)  # magnified crop, ROI size


def test_quins_neighbour_density_and_legacy_default():
    assert LEGACY_LOOK.crowd == 1.0 and QUINS_LOOK.crowd > 1.0
