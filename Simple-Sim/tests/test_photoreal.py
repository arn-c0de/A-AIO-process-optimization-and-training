"""Photoreal 2D base renderer: determinism, class separability, footprint coverage."""

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.draw import _compute_pad_positions

TOL = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}
CFG = {"component_color": [20, 20, 20], "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180]}
FOOTPRINTS = {
    "chip_2pad": {},
    "sot23": {"pad_spacing_y": [36.0, 50.0]},
    "qfn_32": {"pad_width": [15.0, 18.0], "pad_spacing": [173.0, 187.0], "pad_spacing_y": [173.0, 187.0],
               "component_length": [162.0, 172.0], "component_width": [162.0, 172.0]},
    "soic_16": {"pad_width": [10.0, 12.0], "pad_spacing": [120.0, 130.0], "component_length": [90.0, 100.0],
                "component_width": [70.0, 80.0]},
}
CLASSES = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE", "SOLDER_BRIDGE", "CORNER_LIFT"]


def _render(footprint, defect_type, seed=1, **cfg):
    rng = np.random.default_rng(seed)
    nominal = sample_nominal_geometry({}, rng, FOOTPRINTS[footprint])
    defect = sample_defect_params(defect_type, rng, TOL)
    return render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), {**CFG, **cfg}, TOL, rng, footprint=footprint)


@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
@pytest.mark.parametrize("defect_type", CLASSES)
def test_all_footprints_and_classes_render(footprint, defect_type):
    img = _render(footprint, defect_type)
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert 10 < img.mean() < 200


def test_photoreal_is_deterministic_per_seed():
    assert np.array_equal(_render("chip_2pad", "OK", seed=5), _render("chip_2pad", "OK", seed=5))
    assert not np.array_equal(_render("chip_2pad", "OK", seed=5), _render("chip_2pad", "OK", seed=6))


def test_missing_differs_from_ok_and_has_no_dark_body():
    ok, missing = _render("chip_2pad", "OK"), _render("chip_2pad", "MISSING")
    assert np.abs(ok.astype(int) - missing).sum() > 0
    centre = slice(44, 84)
    assert (ok[centre, centre].max(axis=2) < 60).sum() > (missing[centre, centre].max(axis=2) < 60).sum()


def test_capacitor_type_renders_tan_body():
    res = _render("chip_2pad", "OK", component_type="chip_resistor")
    cap = _render("chip_2pad", "OK", component_type="chip_capacitor")
    c = cap[60:68, 60:68].reshape(-1, 3).mean(axis=0)
    assert c[2] > c[0] + 30  # tan: red channel well above blue
    assert np.abs(res.astype(int) - cap).sum() > 0


def test_flat_style_still_available():
    img = _render("chip_2pad", "OK", style="flat")
    assert tuple(img[2, 2]) == (40, 90, 40)


def test_qfn_footprint_has_pad_rows_and_thermal_pad():
    nominal = {"pad_width": 16, "pad_height": 40, "pad_spacing": 180, "pad_spacing_y": 180,
               "component_length": 168, "component_width": 168}
    pads = _compute_pad_positions("qfn_32", nominal, (256, 256))
    assert len(pads) == 33
    assert pads[-1]["w"] == pads[-1]["h"] == round(168 * 0.6)


def test_texture_dir_replaces_substrate(tmp_path):
    import cv2

    tex = np.full((300, 400, 3), (200, 40, 160), dtype=np.uint8)  # magenta: impossible for the procedural mask
    cv2.imwrite(str(tmp_path / "board.png"), tex)
    (tmp_path / "notes.txt").write_text("ignored")
    img = _render("chip_2pad", "MISSING", texture_dir=str(tmp_path))
    corner = img[:10, :10].reshape(-1, 3).mean(axis=0)
    assert corner[0] > 150 and corner[1] < 90 and corner[2] > 120


def test_texture_dir_is_validated(tmp_path):
    from simple_sim.config import _validate_render_2d

    render = {"backend": "opencv_2d", "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180],
              "solder_mask_alpha": 0.3}
    _validate_render_2d(render, 2)
    with pytest.raises(ValueError, match="existing directory"):
        _validate_render_2d({**render, "texture_dir": str(tmp_path / "nope")}, 2)
    with pytest.raises(ValueError, match="no images"):
        _validate_render_2d({**render, "texture_dir": str(tmp_path)}, 2)


def test_tombstone_pose_shows_top_and_raised_end_face():
    """Pivot about the anchored bottom edge: top face L*cos(t) starts H*sin(t) behind the anchor."""
    from simple_sim.generators.opencv2d.photoreal import _THICKNESS_RATIO, _tombstone_pose

    geo = {"length": 60.0, "width": 30.0, "tilt_deg": 80.0, "rotation_deg": 0.0, "center": (64, 64)}
    pose = _tombstone_pose(geo, selector=0, img_shape=(128, 128))
    t = np.radians(80.0)
    assert pose["top_len"] == pytest.approx(60.0 * np.cos(t), rel=1e-5)
    assert pose["end_len"] == pytest.approx(30.0 * _THICKNESS_RATIO * np.sin(t), rel=1e-5)
    xs = pose["box"][:, 0]
    anchor_x = pose["anchor"][0]  # raised towards +x for selector 0
    assert xs.min() == pytest.approx(anchor_x - pose["end_len"], abs=1.0)
    assert xs.max() == pytest.approx(anchor_x + pose["top_len"], abs=1.0)


def test_tombstoned_chip_shows_bright_end_cap():
    img = _render("chip_2pad", "TOMBSTONE", seed=1).astype(float)
    ok = _render("chip_2pad", "OK", seed=1).astype(float)
    centre = slice(40, 88)
    # The upright tin end cap faces the camera: the part area is clearly brighter than a flat black resistor.
    assert img[centre, centre].mean() > ok[centre, centre].mean()
