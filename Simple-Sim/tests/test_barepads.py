"""Bare-board photo -> real SMD lands (simple_sim.board3d.barepads) - synthetic board only."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from simple_sim.board3d import barepads as bp
from simple_sim.board3d.footprints import footprint_for

W, H = 80.0, 60.0                                    # board mm
HOLES = [(4, 4), (76, 4), (76, 56), (4, 50), (14, 4), (60, 44), (14, 40)]
GREEN, COPPER = (60, 110, 40), (120, 190, 215)        # BGR: solder mask, ENIG


def _board() -> dict:
    comps = [{"refdes": f"MT{i}", "family": "mounting_hole", "package": "MTGP640H320", "pnp": {"x": x, "y": y, "rotDeg": 0},
              "pose": {"x": x, "y": y, "rotDeg": 0}, "body": {"l": 6.4, "w": 6.4, "h": 0.0}} for i, (x, y) in enumerate(HOLES)]
    k = 0
    for gx in np.arange(22, 60, 4.0):
        for gy in np.arange(12, 38, 3.0):
            rot = 90.0 if k % 3 == 0 else 0.0
            comps.append({"refdes": f"C{k}", "family": "capacitor", "package": "CAPC1608", "pnp": {"x": gx, "y": gy, "rotDeg": rot},
                          "pose": {"x": gx, "y": gy, "rotDeg": rot}, "body": {"l": 1.6, "w": 0.8, "h": 0.8}})
            k += 1
    return {"widthMm": W, "heightMm": H, "components": comps}


def _truth(c: dict) -> list[tuple[float, float, float, float]]:
    """Real lands: the estimate shifted 0.06 mm outwards and 20 % wider (what a photo would show)."""
    return [(cx + 0.06 * np.sign(cx), cy, w * 1.2, h) for cx, cy, w, h in footprint_for(c)["pads"]]


def _top_view(board: dict, s: float = 20.0) -> tuple[np.ndarray, dict]:
    raster = {"x0": 0.0, "y0": 0.0, "widthMm": W, "heightMm": H, "pxPerMm": s}
    img = np.full((int(H * s), int(W * s), 3), GREEN, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(300):                                  # silkscreen-ish clutter for the dense match
        x, y = rng.uniform(0, W * s), rng.uniform(0, H * s)
        cv2.line(img, (int(x), int(y)), (int(x + rng.uniform(-40, 40)), int(y + rng.uniform(-40, 40))), (225, 230, 230), 2)
    for c in board["components"]:
        p = c["pose"]
        if c["family"] == "mounting_hole":
            cx, cy = bp.to_px([[p["x"], p["y"]]], raster)[0]
            cv2.circle(img, (int(cx), int(cy)), int(3.2 * s), COPPER, -1)
            cv2.circle(img, (int(cx), int(cy)), int(1.6 * s), (20, 20, 20), -1)
            continue
        for cx, cy, w, h in _truth(c):
            r = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
            cv2.fillPoly(img, [bp.to_px(bp.rotate_points(r, p["rotDeg"]) + [p["x"], p["y"]], raster).round().astype(np.int32)], COPPER)
    return img, raster


def _photo(top: np.ndarray, mirror: bool = False) -> np.ndarray:
    h, w = top.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[180, 140], [w + 120, 90], [w + 260, h + 230], [120, h + 170]])
    if mirror:
        dst = dst[[1, 0, 3, 2]]
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.GaussianBlur(cv2.warpPerspective(top, M, (w + 400, h + 400), borderValue=(90, 90, 90)), (0, 0), 1.0)


def test_register_and_measure_lands():
    board = _board()
    top, raster = _top_view(board)
    model, stats = bp.register(_photo(top), board, scan=top, raster=raster)
    assert stats["holeHits"] >= 5 and stats["scan"][-1]["medianShiftMm"] < 0.08
    warped = bp.warp_to_board(_photo(top), model, raster)
    lands, _mask, ms = bp.measure_lands(warped, board, raster)
    assert ms["measured"] >= 0.9 * ms["pads"]
    errs, wide = [], []
    for c in board["components"]:
        if c["family"] != "capacitor":
            continue
        for m, t in zip(lands[c["refdes"]], _truth(c)):
            assert m is not None
            got = bp.land_in_part(m, c["pose"])
            errs.append(np.hypot(got[0] - t[0], got[1] - t[1]))
            wide.append(got[2] - t[2])
    assert np.median(errs) < 0.06 and abs(np.median(wide)) < 0.1


def test_mirrored_photo_is_not_taken_for_this_side():
    board = _board()
    top, raster = _top_view(board)
    with pytest.raises(ValueError):
        bp.register(_photo(top, mirror=True), board)


def test_lands_follow_the_board_not_the_part():
    land = [10.0, 5.0, 0.6, 0.8, 90.0]
    assert np.allclose(bp.land_in_part(land, {"x": 10.0, "y": 5.0, "rotDeg": 90.0}), (0, 0, 0.6, 0.8))
    # a part fitted 0.1 mm off and turned to 0 deg: the land stays where it is on the board
    cx, cy, w, h = bp.land_in_part(land, {"x": 10.1, "y": 5.0, "rotDeg": 0.0})
    assert np.allclose((cx, cy, w, h), (-0.1, 0.0, 0.8, 0.6))


def test_footprint_uses_measured_lands_by_pad_index():
    c = {"refdes": "C1", "family": "capacitor", "package": "CAPC1608", "pose": {"x": 5.0, "y": 5.0, "rotDeg": 0.0},
         "body": {"l": 1.6, "w": 0.8, "h": 0.8}}
    est = footprint_for(c)["pads"]
    c["lands"] = [[4.2, 5.0, 0.9, 1.0, 0.0], None]
    pads = footprint_for(c)["pads"]
    assert np.allclose(pads[0], (-0.8, 0.0, 0.9, 1.0)) and pads[1] == est[1]
    c["lands"] = [[4.2, 5.0, 0.9, 1.0, 0.0]]                 # wrong count: ignored
    assert footprint_for(c)["pads"] == est


def test_fuse_lands_takes_the_median():
    a = {"C1": [[1.0, 1.0, 0.5, 0.5, 0.0], None]}
    b = {"C1": [[1.2, 1.0, 0.7, 0.5, 0.0], [2.0, 2.0, 0.5, 0.5, 0.0]]}
    f = bp.fuse_lands([a, b])
    assert np.allclose(f["C1"][0], [1.1, 1.0, 0.6, 0.5, 0.0]) and f["C1"][1] == [2.0, 2.0, 0.5, 0.5, 0.0]


def test_paste_printed_photo_is_measured_as_paste():
    board = _board()
    top, raster = _top_view(board)
    copper = np.all(top == COPPER, axis=2)
    grey = top.copy()
    grey[copper] = (165, 165, 160)                       # printed solder paste: grey, not warm
    lands, _mask, ms = bp.measure_lands(grey, board, raster)
    assert ms["finish"] == "paste" and ms["measured"] >= 0.9 * ms["pads"]
    assert bp.measure_lands(top, board, raster)[2]["finish"] == "copper"


def test_ring_diameter_and_edge_bias_against_the_scan():
    s = 24.0
    sharp = np.full((400, 400, 3), GREEN, np.uint8)
    cv2.circle(sharp, (200, 200), int(3.2 * s), COPPER, -1)
    cv2.circle(sharp, (200, 200), int(1.6 * s), (20, 20, 20), -1)
    lum = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    d = bp._ring_outer_diameter(lum, np.ones(lum.shape, bool), 200, 200, 2.2 * s, 4.6 * s)
    assert abs(d - 2 * int(3.2 * s)) < 1.0                       # drawn radius 76 px
    spread = cv2.dilate(sharp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # lands bleed 2 px per edge
    raster = {"x0": 0.0, "y0": 0.0, "widthMm": 400 / s, "heightMm": 400 / s, "pxPerMm": s}
    bias, n = bp.edge_bias(spread, {"components": []}, raster, scan=sharp)
    assert n == 1 and abs(bias - 2 / s) < 0.03
