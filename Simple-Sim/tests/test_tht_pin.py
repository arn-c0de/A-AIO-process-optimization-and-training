"""``tht_pin`` (through-hole pin row, Blender only): footprint, defect layout, bake scene, Blender build, 2D refusal."""

import random

import numpy as np
import pytest
import yaml

from simple_sim.board.footprints import (BLENDER_ONLY_FOOTPRINTS, BLENDER_THT, FAMILIES, family_of, footprint_pads, scene_part,
                                         tht_pins)
from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import render_board
from simple_sim.generators.opencv2d.photoreal.solder_defects import tht_layout

from tests.test_blender_part_params import ROOT, load_script, trace

PROFILE = yaml.safe_load((ROOT / "configs" / "profiles" / "tht_pin_3d@1.yaml").read_text(encoding="utf-8"))
NOMINAL = {"pad_width": 42.0, "pad_height": 42.0, "pad_spacing": 64.0, "component_length": 190.0, "component_width": 42.0}


def test_family_and_footprint():
    assert "tht_pin" in FAMILIES and FAMILIES["tht_pin"].blender.body == BLENDER_THT and BLENDER_ONLY_FOOTPRINTS == ("tht_pin",)
    assert family_of("tht_pin") == "tht_pin" and family_of("tht_pin_5") == "tht_pin"
    assert tht_pins("tht_pin") == 3 and tht_pins("tht_pin_5") == 5 and tht_pins("tht_pin_x") == 3
    pads = footprint_pads("tht_pin", NOMINAL, (256, 256))
    assert [p.pin for p in pads] == ["1", "2", "3"] and all(p.w == p.h == 42 for p in pads)
    assert [p.center[0] for p in pads] == [64.0, 128.0, 192.0] and {p.center[1] for p in pads} == {128.0}
    assert len(footprint_pads("tht_pin_5", NOMINAL, (400, 256))) == 5
    part = scene_part("tht_pin")
    assert part["body"] == "tht" and part["marking"] is None and part["pin"] == {"diameter": 0.38, "height": 1.0, "solder_height": 0.45}
    assert "pin" not in scene_part("resistor")  # the other families' scene part is unchanged
    assert PROFILE["profile"]["supported_render_backends"] == ["blender_3d"] and "TOMBSTONE" not in PROFILE["defect_set"]


def test_tht_layout_is_deterministic_per_selector():
    assert tht_layout("OK", 3, 7) == {"missing_pins": [], "bridge": None, "unwetted": False}
    assert tht_layout("MISSING", 3, 7)["missing_pins"] == [1] and tht_layout("MISSING", 3, 8)["missing_pins"] == [2]
    assert tht_layout("SOLDER_BRIDGE", 3, 4)["bridge"] == [0, 1] and tht_layout("SOLDER_BRIDGE", 3, 5)["bridge"] == [1, 2]
    assert tht_layout("CORNER_LIFT", 3, 0)["unwetted"] is True
    assert tht_layout("SOLDER_BRIDGE", 1, 0)["bridge"] is None and tht_layout("MISSING", 0, 3)["missing_pins"] == []


@pytest.mark.parametrize("defect_type", ["OK", "MISSING", "SOLDER_BRIDGE", "CORNER_LIFT", "EXCESS_SOLDER"])
def test_bake_scene_keeps_the_row_and_carries_the_tht_layout(defect_type):
    rng = np.random.default_rng(1)
    nominal = sample_nominal_geometry({}, rng, PROFILE["geometry_ranges"])
    defect = sample_defect_params(defect_type, np.random.default_rng(2), PROFILE["tolerances"])
    config = {"component_type": "tht_pin_header", "mm_per_px": 0.04, "look": "dv_bottom"}
    texture, scene = render_board(nominal, defect, config, PROFILE["tolerances"], np.random.default_rng(3), "tht_pin", (256, 256))
    assert texture.shape == (256, 256, 3) and scene["family"] == "tht_pin" and scene["part"] == scene_part("tht_pin")
    assert scene["body"] is not None and len(scene["pads"]) == 3   # the row stays even for MISSING (one pin is absent)
    assert scene["tht"] == tht_layout(defect_type, 3, __import__("simple_sim.generators.opencv2d.geometry", fromlist=["param_selector"]).param_selector(defect))
    # round pad with a dark hole in the middle: the pad centre is dark, the ring around it bright
    cx, cy = (int(round(v + 128)) for v in scene["pads"][1][:2])
    assert texture[cy, cx].max() < 60 and texture[cy, cx + 15].mean() > 100   # hole radius 0.45 x 21 px, ring to 21 px
    # a texture of the same seed with a chip footprint is a different board (sanity: the tht branch ran)
    assert np.array_equal(texture, render_board(nominal, defect, config, PROFILE["tolerances"], np.random.default_rng(3),
                                                "tht_pin", (256, 256))[0])


def test_opencv_2d_refuses_the_footprint():
    rng = np.random.default_rng(1)
    nominal = sample_nominal_geometry({}, rng, PROFILE["geometry_ranges"])
    defect = sample_defect_params("OK", rng, PROFILE["tolerances"])
    cfg = {"component_color": [20, 20, 20], "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180], "mm_per_px": 0.04}
    with pytest.raises(ValueError, match="Blender backend only"):
        render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), cfg, PROFILE["tolerances"], rng, footprint="tht_pin")
    with pytest.raises(ValueError, match="Blender backend only"):
        render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), {**cfg, "style": "flat"}, PROFILE["tolerances"], rng, footprint="tht_pin")


def _case(tht: dict, body_center=(0.0, 0.0), rotation=0.0):
    pads = [[-64.0, 0.0, 42.0, 42.0], [0.0, 0.0, 42.0, 42.0], [64.0, 0.0, 42.0, 42.0]]
    body = {"center": list(body_center), "length": 190.0, "width": 42.0, "rotation_deg": rotation, "lead_frame_deg": 0.0,
            "tilt_deg": 0.0, "is_tombstone": False}
    return {"board": {"family": "tht_pin", "pads": pads, "body": body, "part": scene_part("tht_pin"), "tht": tht},
            "mm_per_px": 0.04, "component_height_mm": 2.5, "seed": 1}


def test_blender_build_pins_holes_bridge_and_contacts():
    mod = load_script()
    ok = trace(mod, _case(tht_layout("OK", 3, 0)))
    kinds = [e[0] for e in ok["log"]]
    assert kinds.count("cylinder") == 3 and "lead" not in kinds and "box" not in kinds and "text" not in kinds
    pin_r = 0.5 * 0.38 * (42.0 * 0.04)          # pin diameter / pad diameter x pad diameter in mm
    assert all(c["tht"] and c["pin_r"] == pytest.approx(pin_r) and not c["missing"] and not c["unwetted"] for c in ok["contacts"])
    assert ok["solder_height"] == pytest.approx(0.45 * 42.0 * 0.04 / 2.0)
    assert [round(c["cx"], 3) for c in ok["contacts"]] == [-2.56, 0.0, 2.56]  # pad centres in mm
    assert ok["rng_after"] == random.Random(1).random()                        # no marking: rng untouched

    missing = trace(mod, _case(tht_layout("MISSING", 3, 7)))   # pin 1 absent
    kinds = [e[0] for e in missing["log"]]
    assert kinds.count("cylinder") == 3                        # 2 pins + 1 hole
    assert [c["missing"] for c in missing["contacts"]] == [False, True, False]

    unwetted = trace(mod, _case(tht_layout("CORNER_LIFT", 3, 0)))
    assert [e[0] for e in unwetted["log"]].count("cylinder") == 6 and all(c["unwetted"] for c in unwetted["contacts"])

    bridge = trace(mod, _case(tht_layout("SOLDER_BRIDGE", 3, 5)))  # pins 1-2
    leads = [e for e in bridge["log"] if e[0] == "lead"]
    assert len(leads) == 1 and leads[0][1] == "tht_bridge"
    assert leads[0][2][0] == pytest.approx(0.0) and leads[0][3][0] == pytest.approx(2.56)

    shifted = trace(mod, _case(tht_layout("OK", 3, 0), body_center=(5.0, -2.5), rotation=10.0))
    assert shifted["root"][0] == pytest.approx([0.2, 0.1, 0.0]) and shifted["contacts"][0]["cx"] != pytest.approx(-2.56)


def test_tht_joint_mesh_is_a_cone_around_the_pin():
    """The solder ring mesh: pin height at the pin radius, ~0 at the ring edge, none for missing / unwetted pins."""
    mod = load_script()
    captured = {}
    mod._mesh_object = lambda name, verts, faces, mat, smooth=True: captured.update(name=name, verts=verts, faces=faces) or name
    pad = (0.0, 0.0, 1.68, 1.68)
    contact = {"tht": True, "cx": 0.0, "cy": 0.0, "pin_r": 0.32, "missing": False, "unwetted": False}
    assert mod._fillet("solder_0", pad, contact, 0.38, False, None) == "solder_0"
    verts = captured["verts"]
    assert len(verts) == 13 * 48 and len(captured["faces"]) == 12 * 48
    radii = sorted({round(np.hypot(x, y), 4) for x, y, _ in verts})
    assert radii[0] == pytest.approx(0.32) and radii[-1] == pytest.approx(0.84)   # pin radius .. ring radius
    z_at = {round(np.hypot(x, y), 4): z for x, y, z in verts}
    assert z_at[radii[0]] == pytest.approx(0.381) and z_at[radii[-1]] == pytest.approx(0.001)
    assert 0.001 < z_at[radii[6]] < 0.381
    assert mod._fillet("solder_1", pad, {**contact, "missing": True}, 0.38, False, None) is None
    assert mod._fillet("solder_2", pad, {**contact, "unwetted": True}, 0.38, False, None) is None
    mod._fillet("solder_3", pad, contact, 0.38, False, None, excess=True)
    assert max(z for _, _, z in captured["verts"]) == pytest.approx(0.001 + 1.5 * 0.38)
