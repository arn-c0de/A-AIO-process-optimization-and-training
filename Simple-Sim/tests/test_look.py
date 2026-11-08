"""``render.look``: target looks of the photoreal renderer (legacy default, QUINS AOI scans)."""

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import (LOOKS, RenderContext, camera_config, camera_model, camera_settings,
                                                      get_look, mask_gloss, rework_probability)
from simple_sim.generators.opencv2d.photoreal.look import DV_BOTTOM_LOOK, DV_TOP_LOOK, LEGACY_LOOK, QUINS_LOOK
from simple_sim.generators.opencv2d.photoreal.meniscus import SOLDER_TINT as LEGACY_TINT

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
    assert ctx.look is QUINS_LOOK and ctx.solder_look is not None
    # Two line lights along the image x axis; the board is shaded before the camera roll, so the
    # azimuths follow the roll and the bright bands stay vertical in the finished crop.
    assert ctx.solder_look.azimuths == (0.0, pytest.approx(np.pi)) and ctx.solder_look.tint != LEGACY_TINT
    rolled = RenderContext.for_render({"look": "quins"}, rng, roll_deg=90.0)
    assert rolled.solder_look.azimuths == (pytest.approx(np.pi / 2), pytest.approx(1.5 * np.pi))
    assert RenderContext.for_render({"look": "quins", "solder": "legacy"}, rng).solder_look is None
    assert RenderContext.for_render({"solder": "meniscus"}, rng).solder_look.tint == LEGACY_TINT


def test_quins_scanner_light_lights_the_pad_rims_facing_x_whatever_the_rotation():
    """Side lights: on an empty pad the left and right rims of the solder dome are brighter than the
    top and bottom rims, in every cardinal rotation and in both post-filter modes (the lights are
    rotated with the camera roll / the final image rotation)."""
    def side_ratio(img: np.ndarray, nominal: dict, rot: float, size: int = 128) -> float:
        s = size / (3.0 * nominal["component_length"])  # crop_part_lengths = 3 of the quins preset
        grey = img.astype(np.float32).mean(axis=2)
        a = np.radians(rot)
        along = np.array([np.cos(a), -np.sin(a)])  # pad axis after the image-CCW rotation (y down)
        lr, tb = [], []
        for sign in (-1, 1):
            cx, cy = size / 2.0 + sign * along * nominal["pad_spacing"] / 2.0 * s
            pw, ph = (nominal["pad_width"], nominal["pad_height"]) if rot % 180 == 0 else (nominal["pad_height"], nominal["pad_width"])
            hw, hh = pw * s / 2.0, ph * s / 2.0
            x0, x1, y0, y1 = int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)
            rows, cols = slice(int(cy - hh / 2), int(cy + hh / 2)), slice(int(cx - hw / 2), int(cx + hw / 2))
            lr += [grey[rows, x0 - 1:x0 + 3].mean(), grey[rows, x1 - 3:x1 + 1].mean()]
            tb += [grey[y0 - 1:y0 + 3, cols].mean(), grey[y1 - 3:y1 + 1, cols].mean()]
        return float(np.mean(lr) / np.mean(tb))

    rng = np.random.default_rng(7)
    nominal = sample_nominal_geometry({}, rng, FOOTPRINTS["chip_2pad"])
    defect = sample_defect_params("MISSING", rng, TOL)
    ratios = {}
    for rot in (0.0, 90.0, 180.0, 270.0):
        for post in ("camera", "legacy"):
            for seed in (3, 4):
                img = render_roi(nominal, defect, {"rotation_deg": rot}, (128, 128),
                                 {**CFG, "look": "quins", "post_filters": post}, TOL, np.random.default_rng(seed), footprint="chip_2pad")
                ratios[(rot, post, seed)] = side_ratio(img, nominal, rot)
    assert min(ratios.values()) > 1.15, ratios


def test_quins_look_softens_shadows_and_prints_little_silkscreen():
    assert LEGACY_LOOK.shadow == 1.0 and LEGACY_LOOK.silkscreen_skip == 0.45
    assert 0.0 <= QUINS_LOOK.shadow < 0.5 and QUINS_LOOK.silkscreen_skip > 0.7


def test_channel_axis_fixes_the_scanner_direction():
    from simple_sim.config import _validate_camera

    img = np.zeros((64, 64, 3), np.uint8)
    img[:, 32:] = 200   # vertical edge: a split along x separates B and R, a split along y does not
    quiet = {"read_noise": [0, 0], "shot_noise": 0, "blur_sigma": [0.3, 0.3], "channel_shift": [2.0, 2.0]}
    along_x = camera_model(img, np.random.default_rng(1), {**quiet, "channel_axis": "x"}).astype(int)
    along_y = camera_model(img, np.random.default_rng(1), {**quiet, "channel_axis": "y"}).astype(int)
    assert np.abs(along_x[..., 0] - along_x[..., 2]).max() > 40
    assert np.abs(along_y[..., 0] - along_y[..., 2]).max() <= 2
    # post_filters: legacy turns the finished image: a fixed axis is pre-rotated the other way (like
    # the scanner lights), so after a 90 degree turn the split still runs along x in the final image.
    turned = camera_model(img, np.random.default_rng(1), {**quiet, "channel_axis": "x"}, axis_roll_deg=90.0).astype(int)
    assert np.abs(turned[..., 0] - turned[..., 2]).max() <= 2
    turned = camera_model(img, np.random.default_rng(1), {**quiet, "channel_axis": "y"}, axis_roll_deg=90.0).astype(int)
    assert np.abs(turned[..., 0] - turned[..., 2]).max() > 40
    assert np.array_equal(camera_model(img, np.random.default_rng(1), {**quiet, "channel_axis": "x"}, axis_roll_deg=0.0), along_x)
    assert camera_settings({"preset": "quins"})["channel_axis"] == "x" and camera_settings(None)["channel_axis"] == "random"
    _validate_camera({"channel_axis": "y"})
    with pytest.raises(ValueError, match="channel_axis"):
        _validate_camera({"channel_axis": "diagonal"})


def test_look_defaults_yield_to_explicit_settings():
    assert camera_config({}) is None
    assert camera_config({"look": "quins"}) == {"preset": "quins"}
    assert camera_config({"look": "quins", "camera": {"read_noise": [1, 2]}}) == {"preset": "quins", "read_noise": [1, 2]}
    assert camera_config({"look": "quins", "camera": {"preset": "phone"}}) == {"preset": "phone"}
    # D30: the look carries numeric defaults (legacy: the former renderer constants).
    assert mask_gloss({}) == LEGACY_LOOK.mask_gloss == 0.5 and mask_gloss({"look": "quins"}) == 0.0
    assert mask_gloss({"look": "quins", "mask_gloss": 0.7}) == 0.7
    assert rework_probability({}) == LEGACY_LOOK.rework == 0.15 and rework_probability({"look": "quins"}) == QUINS_LOOK.rework
    assert rework_probability({"look": "quins", "rework_artifacts": 1.0}) == 1.0
    assert mask_gloss({}, 0.9) == 0.5  # the old *default* argument only applies to a look without a value


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


def test_neighbour_density_lives_in_packing_only():
    """Density comes from ``Look.packing`` (see the packing test); the former ``Look.crowd`` factor
    (always 1.0) is folded into ``Packing.parts``."""
    import dataclasses

    assert "crowd" not in {f.name for f in dataclasses.fields(LEGACY_LOOK)}


def test_quins_uses_dense_packing_and_legacy_layout_is_unchanged():
    from simple_sim.board import LEGACY_PACKING, neighbour_layout
    from simple_sim.generators.opencv2d.photoreal.look import DENSE_PACKING

    assert LEGACY_LOOK.packing is LEGACY_PACKING and QUINS_LOOK.packing is DENSE_PACKING
    pads = [{"x": 100, "y": 110, "w": 30, "h": 36}, {"x": 160, "y": 110, "w": 30, "h": 36}]
    keepout = (100, 110, 190, 146)
    a = neighbour_layout(pads, keepout, (256, 290), np.random.default_rng(3))
    b = neighbour_layout(pads, keepout, (256, 290), np.random.default_rng(3), 1.0, LEGACY_PACKING)
    assert a == b
    counts = [len(neighbour_layout(pads, keepout, (256, 290), np.random.default_rng(s), 1.0, p))
              for p in (LEGACY_PACKING, DENSE_PACKING) for s in range(20)]
    assert sum(counts[20:]) > sum(counts[:20])


# --- DeepVision app looks (dv_top: phone photo of the TOP side, dv_bottom: flatbed-like BOTTOM capture) -------

DV_LOOKS = {"dv_top": DV_TOP_LOOK, "dv_bottom": DV_BOTTOM_LOOK}


def test_dv_looks_are_registered_after_quins_and_legacy_quins_are_untouched():
    assert LOOKS == ("legacy", "quins", "dv_top", "dv_bottom")
    for name, look in DV_LOOKS.items():
        assert get_look({"look": name}) is look and look.name == name and look.camera_preset == name
        assert look.solder == "meniscus" and look.flat_caps
    # Additive: the earlier looks keep their registry objects and camera presets.
    assert get_look({"look": "quins"}) is QUINS_LOOK and QUINS_LOOK.camera_preset == "quins"
    assert camera_settings({"preset": "quins"})["sensor_px_per_mm"] == [22.0, 25.5]


@pytest.mark.parametrize("look", list(DV_LOOKS))
@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
@pytest.mark.parametrize("defect_type", CLASSES)
def test_dv_looks_render_every_footprint_and_class(look, footprint, defect_type):
    img = _render(footprint, defect_type, look=look)
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert 20 < img.mean() < 200


@pytest.mark.parametrize("look", list(DV_LOOKS))
def test_dv_looks_are_deterministic_and_differ_from_the_other_looks(look):
    a, b = _render(seed=5, look=look), _render(seed=5, look=look)
    assert np.array_equal(a, b)
    others = [n for n in LOOKS if n != look]
    for other in others:
        assert not np.array_equal(a, _render(seed=5, look=other)), other


def test_dv_camera_presets_model_the_app_crops():
    """~16 px/mm crops of 3 body lengths (the app's 49/50 px crops), JPEG, soft focus; no draw-free gaps."""
    from simple_sim.config import _validate_camera
    from simple_sim.generators.opencv2d.photoreal.camera import fit_crop

    for name in DV_LOOKS:
        _validate_camera({"preset": name})
        cam = camera_settings({"preset": name})
        assert cam["sensor_px_per_mm"] == [15.5, 16.5] and cam["crop_part_lengths"] == [3.0, 3.1]
        assert cam["jpeg_quality"][1] < 100 and cam["blur_sigma"][0] > 0
        pose = fit_crop({"tilt": 0.0, "azimuth": 0.0, "zoom": 1.0, "roll": 0.0}, cam, np.random.default_rng(0), 62.0, (256, 256))
        assert 3.0 * 62.0 / 256.0 <= pose["zoom"] <= 3.1 * 62.0 / 256.0
    assert camera_settings({"preset": "dv_top"})["tilt_deg"][1] > camera_settings({"preset": "dv_bottom"})["tilt_deg"][1]
    assert camera_config({"look": "dv_top"}) == {"preset": "dv_top"}
    assert camera_config({"look": "dv_bottom", "camera": {"preset": "aoi"}}) == {"preset": "aoi"}


def test_dv_joint_lights_flash_lobe_follows_the_roll_and_flatbed_is_a_ring():
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    top = RenderContext.for_render({"look": "dv_top"}, rng)
    assert rng.bit_generator.state == state  # side stream only
    assert len(top.solder_look.azimuths) == 1 and top.solder_look.tint != LEGACY_TINT
    rolled = RenderContext.for_render({"look": "dv_top"}, rng, roll_deg=90.0)
    assert rolled.solder_look.azimuths[0] == pytest.approx(top.solder_look.azimuths[0] + np.pi / 2)
    bottom = RenderContext.for_render({"look": "dv_bottom"}, rng)
    assert bottom.solder_look.azimuths == () and bottom.solder_look.sky > 100
    assert RenderContext.for_render({"look": "dv_top", "solder": "legacy"}, rng).solder_look is None


def test_dv_look_defaults_and_materials():
    assert mask_gloss({"look": "dv_top"}) == DV_TOP_LOOK.mask_gloss > mask_gloss({"look": "dv_bottom"}) > 0.0
    assert mask_gloss({"look": "dv_top", "mask_gloss": 0.0}) == 0.0
    assert rework_probability({"look": "dv_bottom"}) == DV_BOTTOM_LOOK.rework < LEGACY_LOOK.rework
    # Dark 0402 resistor bodies (the dark 1 % of the real crops), bright flat tin terminals, dense rows of parts
    # with as many resistors as MLCCs (the xpa pnp: 278 RESC1005 vs 272 CAPC1005).
    dark = np.array((20, 20, 20), np.float32)
    for look in DV_LOOKS.values():
        assert float(look.body_colour("resistor", dark).mean()) < 60 and float(look.end_cap.mean()) > 190
        assert look.packing.capacitor_share == 0.5 and look.packing.parts >= QUINS_LOOK.packing.parts
        assert look.shadow == 1.0 and look.silkscreen_skip >= 0.8
    # The TOP mask palette is brighter and yellower than the dark olive BOTTOM / quins one.
    top_mean = np.mean(DV_TOP_LOOK.mask_palette, axis=0)
    bottom_mean = np.mean(DV_BOTTOM_LOOK.mask_palette, axis=0)
    assert top_mean.sum() > bottom_mean.sum() and top_mean[2] > bottom_mean[2]


def test_dv_bottom_mask_matches_the_measured_app_colour():
    """Bare board of a MISSING part: median near the BOTTOM app crops' mask, BGR ~ (70, 89, 48) (p25-p75
    (64, 84, 37)-(95, 119, 58))."""
    meds = np.array([np.median(_render(defect_type="MISSING", seed=s, look="dv_bottom").reshape(-1, 3), axis=0)
                     for s in range(6)])
    assert np.all(np.abs(np.median(meds, axis=0) - (70, 89, 48)) < 16)
