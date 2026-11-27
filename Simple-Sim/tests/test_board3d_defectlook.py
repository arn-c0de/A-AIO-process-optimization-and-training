"""board3d defect look and the prep-board cameras: shaped defect ops, fault avoidance, fitted shapes, photo
registration, still sources, the defect-fit score (no Blender)."""

import json
import random

import cv2
import numpy as np
import pytest

from simple_sim.board3d.scene import apply_defects, job_component
from simple_sim.board3d.substrate import BoardRaster
from simple_sim.classes import DefectClass


def _comp(ref, x, y, fam="capacitor", pkg="CAPC1005", l=1.0, w=0.5, h=0.5):
    return {"refdes": ref, "family": fam, "familyStatus": "builder", "package": pkg, "stage": "smd", "dnp": False,
            "pose": {"x": x, "y": y, "rotDeg": 0.0, "z": 0.0, "tiltDeg": [0, 0]}, "body": {"l": l, "w": w, "h": h}}


def _board(n=6):
    return {"components": [_comp(f"C{i}", 5 + 4 * i, 10) for i in range(n)]}


def test_shaped_ops_reach_the_job():
    b = {"components": [_comp("C1", 10, 10), _comp("C2", 11.3, 10), _comp("C3", 20, 10), _comp("C4", 25, 10)]}
    out = apply_defects(b, [{"op": "remove", "refdes": "C1", "residue": "tinned", "domeMm": 0.1},
                            {"op": "excess", "refdes": "C3", "grow": 1.5, "heightScale": 3.0},
                            {"op": "tombstone", "refdes": "C4", "tiltDeg": 80, "raisedSign": -1, "domeMm": 0.12},
                            {"op": "bridge", "refdes": "C2", "pad": 1, "to": "C3", "toPad": 0, "widthScale": 1.1, "heightMm": 0.3}])
    by = {c["refdes"]: job_component(c) for c in out["components"]}
    assert by["C1"]["missing"] and by["C1"]["residue"] == {"kind": "tinned", "domeMm": 0.1}
    assert by["C3"]["excessGrow"] == 1.5 and by["C3"]["excessHeight"] == 3.0 and by["C3"]["excess"] == [0, 1]
    assert by["C4"]["tombstone"]["domeMm"] == 0.12
    assert "residue" not in by["C2"] and "excessGrow" not in by["C2"]          # untouched parts keep the old job
    (br,) = out["bridges"]
    assert br["widthScale"] == 1.1 and br["heightMm"] == 0.3
    plain = apply_defects(b, [{"op": "remove", "refdes": "C1"}])
    assert "residue" not in job_component(plain["components"][0])           # default: printed paste as before


def test_sampler_draws_shapes_in_range():
    from simple_sim.board3d.dataset import sample_op
    c = _comp("C1", 10, 10)
    rng = random.Random(0)
    miss = [sample_op(c, DefectClass.MISSING, rng) for _ in range(200)]
    tinned = [m for m in miss if m.get("residue") == "tinned"]
    assert 0.45 < len(tinned) / len(miss) < 0.75 and all(0.04 <= m["domeMm"] <= 0.16 for m in tinned)
    exc = sample_op(c, DefectClass.EXCESS_SOLDER, rng)
    assert 1.2 <= exc["grow"] <= 1.6 and 1.8 <= exc["heightScale"] <= 3.2
    b = {"components": [c, _comp("C2", 11.3, 10)]}
    br = sample_op(c, DefectClass.SOLDER_BRIDGE, rng, b)
    assert 0.45 <= br["widthScale"] <= 1.15 and 0.1 <= br["heightMm"] <= 0.38


def test_plan_sheet_avoids_real_faults():
    from simple_sim.board3d.dataset import near_any, plan_sheet
    b = _board()
    win = BoardRaster(0, 0, 30, 20, 10)
    fault = (b["components"][2]["pose"]["x"], 10.0)
    for seed in range(10):
        plan = plan_sheet(b, win, 10, 1.0, random.Random(seed), avoid=[fault])
        assert "C2" not in [p["component"]["refdes"] for p in plan]
    assert near_any(b["components"][2], [(fault[0] + 1.4, 10.0)])          # inside the 3.04 mm field (+0.5)
    assert not near_any(b["components"][2], [(fault[0] + 2.5, 10.0)])


def test_fitted_shapes_replace_half_the_parameters():
    from simple_sim.board3d.dataset import with_fitted_shape
    shapes = {"bridge": [{"widthScale": 2.0, "heightMm": 1.0}]}
    rng = random.Random(0)
    ops = [with_fitted_shape({"op": "bridge", "refdes": "C1", "widthScale": 0.5, "heightMm": 0.2}, shapes, rng) for _ in range(400)]
    fitted = [o for o in ops if o["widthScale"] > 1.5]
    assert 0.4 < len(fitted) / len(ops) < 0.6
    assert all(1.8 <= o["widthScale"] <= 2.2 and o["refdes"] == "C1" for o in fitted)
    assert with_fitted_shape(None, shapes, rng) is None
    assert with_fitted_shape({"op": "move"}, shapes, rng) == {"op": "move"}


def test_shape_params_keep_only_fits_that_beat_nominal():
    from simple_sim.board3d.defectfit import shape_params
    rows = [{"beatsNominal": True, "best": {"op": {"op": "tombstone", "refdes": "R1", "tiltDeg": 70, "raisedSign": 1, "domeMm": 0.1}}},
            {"beatsNominal": False, "best": {"op": {"op": "tombstone", "refdes": "R2", "tiltDeg": 50}}},
            {"beatsNominal": True, "best": {"op": {"op": "move", "refdes": "R3", "dx": 0.1}}}]
    assert shape_params(rows) == {"tombstone": [{"tiltDeg": 70, "domeMm": 0.1}]}


def test_crop_score_prefers_the_same_picture():
    from simple_sim.board3d.defectfit import crop_score
    rng = np.random.default_rng(0)
    real = np.full((64, 64, 3), (60, 140, 60), np.uint8)
    cv2.rectangle(real, (20, 26), (44, 38), (220, 220, 220), -1)
    same = np.clip(real.astype(int) + rng.integers(-4, 5, real.shape), 0, 255).astype(np.uint8)
    moved = np.full_like(real, (60, 140, 60))
    cv2.rectangle(moved, (8, 4), (32, 16), (220, 220, 220), -1)
    assert crop_score(same, real)["score"] > crop_score(moved, real)["score"] + 0.3


def test_register_plausibility_and_coarse_fit():
    from simple_sim.board3d.photoreg import coarse, plausible
    r = BoardRaster(0, 0, 40, 30, 10)
    ok = np.linalg.inv(r.pixel_to_board())                                  # the raster itself as the "photo"
    assert plausible(ok, r, (r.rows, r.cols))
    flipped = ok.copy()
    flipped[2] = [0.05, 0, 1]                                               # horizon through the board
    assert not plausible(flipped, r, (r.rows, r.cols))
    rng = np.random.default_rng(1)
    mosaic = cv2.GaussianBlur(rng.integers(0, 255, (r.rows, r.cols, 3)).astype(np.uint8), (0, 0), 1.2)
    shift = np.array([[1, 0, 30], [0, 1, 20], [0, 0, 1]], float)            # photo = mosaic moved by (30, 20) px
    photo = cv2.warpPerspective(mosaic, shift, (r.cols + 60, r.rows + 40))
    b, st = coarse(photo, mosaic, r, scales=(1.0,))
    centre = cv2.perspectiveTransform(np.array([[[20.0, 15.0]]]), b)[0, 0]
    want = cv2.perspectiveTransform(np.array([[[20.0, 15.0]]]), shift @ ok)[0, 0]
    assert np.abs(centre - want).max() < 1.0 and st["inliers"] >= 12


def test_board_faults_take_the_newest_review(tmp_path):
    from tools.board3d_register import board_faults
    lines = [{"board": "prep1", "id": "a", "label": "fault", "fault": "grabstein", "centerMm": [1, 2], "at": "2026-10-01"},
             {"board": "prep1", "id": "a", "label": "ok", "centerMm": [1, 2], "at": "2026-10-02"},          # re-judged ok
             {"board": "prep1", "id": "b", "label": "fault", "fault": "loetbruecke", "centerMm": [3, 4], "at": "2026-10-01"},
             {"board": "prep1", "id": "c", "label": "fault", "centerMm": [5, 6], "retracted": True},
             {"board": "other", "id": "d", "label": "fault", "fault": "fehlt", "centerMm": [7, 8]}]
    p = tmp_path / "top.ndjson"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert board_faults(p, "prep1") == [{"id": "b", "centerMm": [3.0, 4.0], "fault": "loetbruecke"}]


def test_still_source_registers_a_named_source(tmp_path):
    from simple_sim.board3d.render import SOURCES, still_source
    g = tmp_path / "IMG1" / "geometry.json"
    g.parent.mkdir()
    g.write_text("{}")
    name = still_source(g)
    clean = still_source(g, clean_faults=True)
    assert name == "still-IMG1" and clean == "still-IMG1+clean"
    assert SOURCES[name]["geometry"] == str(g) and SOURCES[clean]["cleanFaults"] is True
    assert SOURCES[name]["folder"] != SOURCES[clean]["folder"]
    for n in (name, clean):
        SOURCES.pop(n)


@pytest.mark.parametrize("bright", [200, 240])
def test_scan_solder_colour_takes_the_bright_lands(tmp_path, bright):
    from simple_sim.board3d import render
    img = np.full((50, 50, 3), (40, 120, 40), np.uint8)
    pads = np.zeros((50, 50), np.uint8)
    pads[10:40, 10:40] = 255
    img[10:40, 10:40] = 120                                                # dull land
    img[10:40, 10:14] = bright                                             # the lit crest
    parts = np.zeros_like(pads)
    paths = {}
    for k, v in (("scan_rgb", img), ("pads_mask", pads), ("parts_mask", parts)):
        paths[k] = str(tmp_path / f"{k}.png")
        cv2.imwrite(paths[k], v)
    render._SOLDER_COLOUR.pop(paths["scan_rgb"], None)
    assert render.scan_solder_colour({"maps": paths}) == (bright, bright, bright)
