"""Photoreal 2D base renderer: determinism, class separability, footprint coverage."""

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.geometry import pad_positions

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
CLASSES = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE", "SOLDER_BRIDGE", "CORNER_LIFT", "SOLDER_BALL", "EXCESS_SOLDER"]


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
    pads = pad_positions("qfn_32", nominal, (256, 256))
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
    from simple_sim.config import _validate_render

    render = {"backend": "opencv_2d", "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180],
              "solder_mask_alpha": 0.3}
    _validate_render(render, 2, "defect")
    with pytest.raises(ValueError, match="existing directory"):
        _validate_render({**render, "texture_dir": str(tmp_path / "nope")}, 2, "defect")
    with pytest.raises(ValueError, match="no images"):
        _validate_render({**render, "texture_dir": str(tmp_path)}, 2, "defect")


def test_tombstone_pose_shows_top_and_raised_end_face():
    """Pivot about the anchored bottom edge: top face L*cos(t) starts H*sin(t) behind the anchor."""
    from simple_sim.generators.opencv2d.photoreal.parts import THICKNESS_RATIO, tombstone_pose

    geo = {"length": 60.0, "width": 30.0, "tilt_deg": 80.0, "rotation_deg": 0.0, "center": (64, 64)}
    pose = tombstone_pose(geo, selector=0, img_shape=(128, 128))
    t = np.radians(80.0)
    assert pose["top_len"] == pytest.approx(60.0 * np.cos(t), rel=1e-5)
    assert pose["end_len"] == pytest.approx(30.0 * THICKNESS_RATIO * np.sin(t), rel=1e-5)
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


def test_tombstone_tilt_spans_threshold_to_upright():
    from simple_sim.defects import classify_defect

    rng = np.random.default_rng(0)
    tol = {**TOL, "tombstone_tilt_deg": 30.0}
    tilts = [sample_defect_params("TOMBSTONE", rng, tol)["tilt_deg"] for _ in range(400)]
    assert min(tilts) >= 30.0 and max(tilts) <= 90.0
    assert min(tilts) < 35.0 and max(tilts) > 85.0
    assert all(classify_defect({}, {"type": "TOMBSTONE", "shift_x": 0.0, "shift_y": 0.0, "rotation_deg": 0.0,
                                    "tilt_deg": t}, tol) == "TOMBSTONE" for t in tilts)


def test_rework_residue_stays_out_of_the_keepout():

    from simple_sim.generators.opencv2d.photoreal.context import RenderContext
    from simple_sim.generators.opencv2d.photoreal.rework import draw_rework

    img = np.full((256, 256, 3), 100.0, dtype=np.float32)
    keepout = (96, 112, 160, 144)
    for seed in range(20):
        draw_rework(RenderContext(), img, keepout, 20.0, np.random.default_rng(seed), 1.0)
    x0, y0, x1, y1 = keepout
    assert np.all(img[y0:y1, x0:x1] == 100.0)  # label-relevant area untouched
    assert np.abs(img - 100.0).max() > 20      # but residue was drawn elsewhere


def test_rework_can_be_disabled():
    a = _render("chip_2pad", "OK", seed=3, rework_artifacts=0.0)
    b = _render("chip_2pad", "OK", seed=3, rework_artifacts=0.0)
    assert np.array_equal(a, b)


def test_solder_defect_layout():
    from simple_sim.generators.opencv2d.photoreal import solder_defect_layout

    pads = [{"x": 20, "y": 40, "w": 30, "h": 40}, {"x": 110, "y": 40, "w": 30, "h": 40}]
    body = np.array([[35, 45], [125, 45], [125, 75], [35, 75]], dtype=np.float32)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert 1 <= len(solder_defect_layout("EXCESS_SOLDER", pads, body, rng, ss=2)["excess"]) <= 2
        balls = solder_defect_layout("SOLDER_BALL", pads, body, rng, ss=2)["balls"]
        assert 1 <= len(balls) <= 4
        for x, y, r in balls:  # never on a pad, never under the body
            assert not any(p["x"] - r < x < p["x"] + p["w"] + r and p["y"] - r < y < p["y"] + p["h"] + r for p in pads)
            assert cv2_point_outside(body, x, y, r)
    assert solder_defect_layout("OK", pads, body, np.random.default_rng(0), ss=2) == {"excess": [], "balls": []}


def cv2_point_outside(poly, x, y, r):
    import cv2

    return cv2.pointPolygonTest(poly, (x, y), True) <= -r


def test_flat_style_draws_solder_defects():
    ok = _render("chip_2pad", "OK", seed=4, style="flat")
    ball = _render("chip_2pad", "SOLDER_BALL", seed=4, style="flat")
    assert np.abs(ok.astype(int) - ball).sum() > 0


def test_phone_camera_preset():
    from simple_sim.config import _validate_camera
    from simple_sim.generators.opencv2d.photoreal import apply_view, camera_settings, view_canvas

    _validate_camera({"preset": "phone", "tilt_deg": [10.0, 30.0]})
    cam = camera_settings({"preset": "phone", "jpeg_quality": [90, 90]})
    assert cam["tilt_deg"] == [5.0, 40.0] and cam["jpeg_quality"] == [90, 90]
    # AOI default: no extra canvas, view is the identity
    flat = {"tilt": 0.0, "azimuth": 0.0, "zoom": 1.0}
    assert view_canvas((128, 96), flat) == (128, 96)
    img = np.random.default_rng(0).integers(0, 255, (96, 128, 3)).astype(np.uint8)
    assert np.array_equal(apply_view(img, flat, (128, 96)), img)
    # Zoomed out + oblique: larger canvas, output at ROI size
    pose = {"tilt": 40.0, "azimuth": 30.0, "zoom": 2.0}
    cw, ch = view_canvas((128, 96), pose)
    assert cw > 2 * 128 and ch > 2 * 96
    assert apply_view(np.zeros((ch, cw, 3), np.uint8), pose, (128, 96)).shape == (96, 128, 3)
    phone = _render("chip_2pad", "OK", seed=2, camera={"preset": "phone"})
    assert phone.shape == (128, 128, 3)
    assert not np.array_equal(phone, _render("chip_2pad", "OK", seed=2))


@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
@pytest.mark.parametrize("factor", [1, 3, 4])
def test_supersampling_factors_render(footprint, factor):
    """``render.supersampling`` 1-4 renders every footprint; the default (2) is unchanged."""
    img = _render(footprint, "TOMBSTONE", supersampling=factor)
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert np.array_equal(_render(footprint, "OK", seed=3), _render(footprint, "OK", seed=3, supersampling=2))


def test_supersampling_validation():
    from simple_sim.config import _validate_supersampling

    _validate_supersampling(None)
    _validate_supersampling(4)
    for bad in (0, 5, 2.5, True, "4"):
        with pytest.raises(ValueError):
            _validate_supersampling(bad)


def test_low_res_capture_keeps_roi_size_and_softens():
    """``camera.capture_scale`` < 1 samples at sensor resolution and upscales back to the ROI."""
    from simple_sim.generators.opencv2d.photoreal import camera_model

    img = np.zeros((96, 128, 3), np.uint8)
    img[:, 64:] = 220
    full = camera_model(img, np.random.default_rng(1), {"read_noise": [0, 0], "shot_noise": 0})
    small = camera_model(img, np.random.default_rng(1), {"capture_scale": [0.25, 0.25], "read_noise": [0, 0],
                                                       "shot_noise": 0, "chroma_shift": [1.0, 1.0]})
    assert small.shape == full.shape == img.shape
    edge = lambda x: np.abs(np.diff(x[48, :, 1].astype(int))).max()  # noqa: E731
    assert edge(small) < edge(full)


def test_camera_and_gloss_validation():
    from simple_sim.config import _validate_camera

    _validate_camera({"preset": "phone", "capture_scale": [0.2, 0.7], "chroma_shift": [0, 1]})
    for bad in ({"capture_scale": [0.0, 0.5]}, {"capture_scale": [0.5, 1.5]}, {"chroma_shift": [1, 0]}):
        with pytest.raises(ValueError):
            _validate_camera(bad)


def test_mask_gloss_is_optional_and_deterministic():
    a = _render("chip_2pad", "OK", seed=4, mask_gloss=1.0)
    assert np.array_equal(a, _render("chip_2pad", "OK", seed=4, mask_gloss=1.0))
    assert not np.array_equal(a, _render("chip_2pad", "OK", seed=4, mask_gloss=0.0))


def test_mask_colour_matches_real_boards():
    """Bare mask is an olive/bottle green with moderate saturation (real crops: HSV S ~90-110), not lime/cyan."""
    import cv2

    img = _render("chip_2pad", "MISSING", seed=7, mask_gloss=0.0, rework_artifacts=0.0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    green = hsv[(hsv[:, 0] >= 35) & (hsv[:, 0] <= 95)]
    assert 60 < np.median(green[:, 1]) < 140


def test_sot23_body_is_vertical_and_leads_move_with_it():
    """SOT-23: body long axis along the pin-1/2 column; leads are rigid with the body (a shifted part's feet miss the pads)."""
    from simple_sim.generators.opencv2d.photoreal import lead_layout, nominal_body_angle

    assert nominal_body_angle("sot") == 90.0 and nominal_body_angle("resistor") == 0.0
    nom = {"pad_width": 22, "pad_height": 18, "pad_spacing": 50, "pad_spacing_y": 42,
           "component_length": 64, "component_width": 29}
    pads = pad_positions("sot23", nom, (128, 128))
    leads = lead_layout("sot", pads, (64, 64), 64.0, 29.0, ss=2)
    assert len(leads) == 3 and all(l["axis"] == "v" for l in leads)  # all leave the long sides
    assert sorted(sum(l["side"] == s for l in leads) for s in (-1.0, 1.0)) == [1, 2]  # pins 1+2 on one side, pin 3 opposite
    assert all(l["tip"] > l["edge"] for l in leads)
    img = _render("sot23", "OK", seed=3)
    assert img.shape == (128, 128, 3)


NEW_FORM_PROFILES = ["chip_array_8_resistor@1", "chip_array_8_capacitor@1", "sot363_ic@1", "sot23_6_ic@1",
                     "chip_0402_resistor@1", "chip_0402_capacitor@1", "chip_1210_capacitor@1", "chip_0603_ferrite@1",
                     "chip_2520_power_inductor@1", "chip_0603_led@1", "tssop20_ic@1", "tssop14_ic@1"]


@pytest.mark.parametrize("footprint,n", [("soic_16", 16), ("sot_6", 6), ("chip_array_8", 8), ("tssop_14", 14),
                                         ("tssop_20", 20)])
def test_dual_row_footprints(footprint, n):
    """Two pad columns, pins numbered counter-clockwise (1..n/2 down the left, n/2+1..n up the right)."""
    from simple_sim.board.footprints import footprint_pads

    nom = {"pad_width": 13, "pad_height": 20, "pad_spacing": 46, "pad_spacing_y": 67,
           "component_length": 45, "component_width": 90}
    pads = footprint_pads(footprint, nom, (256, 256))
    assert len(pads) == n
    left = [p for p in pads if p.x < 128]
    assert len(left) == n // 2 and [p.pin for p in left] == [str(i + 1) for i in range(n // 2)]
    ys = sorted(p.y for p in left)
    assert all(b - a >= 12 for a, b in zip(ys, ys[1:]))  # pads of a column never overlap (rounding: 1 px)


def test_new_families():
    from simple_sim.board.footprints import family_of

    assert family_of("sot_6") == "sot6"
    assert family_of("chip_array_8", "resistor_array") == "array_r"
    assert family_of("chip_array_8", "capacitor_array") == "array_c"
    assert family_of("tssop_20") == "soic"
    assert family_of("chip_2pad", "ferrite_bead") == "inductor"
    assert family_of("chip_2pad", "power_inductor") == "power_inductor"
    assert family_of("chip_2pad", "led") == "led"
    assert family_of("chip_2pad", "chip_capacitor") == "capacitor"
    assert family_of("chip_2pad", "chip_resistor") == "resistor"


def test_netlist_handles_every_family():
    """Every family has net rules (routing: escape must not KeyError on new families)."""
    from simple_sim.board.footprints import FAMILIES
    from simple_sim.board.netlist import CONNECTED, PASSIVE_GND

    for fam in FAMILIES:
        assert fam in PASSIVE_GND or CONNECTED.get(fam, 1.0) > 0


@pytest.mark.parametrize("profile_id", NEW_FORM_PROFILES)
def test_new_form_profiles_render_every_class(profile_id):
    import yaml
    from pathlib import Path

    prof = yaml.safe_load((Path(__file__).parent.parent / "configs" / "profiles" / f"{profile_id}.yaml").read_text())
    tol = prof["tolerances"]
    for i, cls in enumerate(prof["defect_set"]):
        rng = np.random.default_rng(i)
        nominal = sample_nominal_geometry({}, rng, prof["geometry_ranges"])
        defect = sample_defect_params(cls, rng, tol)
        cfg = {**CFG, "component_type": prof["component"]["type"],
               "component_color": prof["render"]["component_color_bgr"],
               "marking_probability": prof["render"].get("marking_probability", 1.0)}
        img = render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), cfg, tol, rng,
                         footprint=prof["component"]["footprint"])
        assert img.shape == (128, 128, 3) and 10 < img.mean() < 220


def test_resistor_array_has_dark_body_and_bright_terminals():
    import yaml
    from pathlib import Path

    prof = yaml.safe_load((Path(__file__).parent.parent / "configs" / "profiles" / "chip_array_8_resistor@1.yaml").read_text())
    rng = np.random.default_rng(0)
    nominal = sample_nominal_geometry({}, rng, prof["geometry_ranges"])
    defect = sample_defect_params("OK", rng, prof["tolerances"])
    cfg = {**CFG, "component_type": "resistor_array", "component_color": [22, 22, 22], "mask_gloss": 0.0}
    img = render_roi(nominal, defect, {"rotation_deg": 0.0}, (256, 256), cfg, prof["tolerances"], rng,
                     footprint="chip_array_8")
    centre = img[100:156, 118:138].reshape(-1, 3).max(axis=1)
    assert (centre < 60).mean() > 0.3   # black body between the terminal rows
    assert (img[100:156, 100:156].max(axis=2) > 170).mean() > 0.05  # metal terminals / joints


def test_power_inductor_and_led_look():
    """Moulded inductor: light grey block without bright end caps; LED: white housing around a yellowish lens."""
    import yaml
    from pathlib import Path

    def render(pid):
        prof = yaml.safe_load((Path(__file__).parent.parent / "configs" / "profiles" / f"{pid}.yaml").read_text())
        rng = np.random.default_rng(1)
        nominal = sample_nominal_geometry({}, rng, prof["geometry_ranges"])
        defect = sample_defect_params("OK", rng, prof["tolerances"])
        cfg = {**CFG, "component_type": prof["component"]["type"], "component_color": prof["render"]["component_color_bgr"],
               "mask_gloss": 0.0, "rework_artifacts": 0.0}
        return render_roi(nominal, defect, {"rotation_deg": 0.0}, (256, 256), cfg, prof["tolerances"], rng,
                          footprint="chip_2pad")

    ind = render("chip_2520_power_inductor@1")[118:138, 108:148].reshape(-1, 3).astype(int)
    assert 110 < ind.mean() < 200 and (ind.max(axis=1) - ind.min(axis=1)).mean() < 25  # grey, unsaturated
    led = render("chip_0603_led@1")[120:136, 116:140].reshape(-1, 3).astype(int)
    assert (led[:, 2] - led[:, 0]).mean() > 20  # yellowish lens: red well above blue
