"""T1: board substrate images in board millimetres.

:func:`rectify` warps the flatbed scan (``boardToImage``: board mm -> scan px) to a top-down board image at a
fixed ``px_per_mm``: pixel ``(u, v)`` is board point ``(x0 + (u + 0.5) / s, y1 - (v + 0.5) / s)``
(``y1`` = top edge, board y up -> image rows down). The same convention is used for every texture of the board3d,
so ``uv = ((x - x0) / width_mm, (y - y0) / height_mm)`` in Blender (v up) addresses it directly.

:func:`part_mask` rasterises the scene's part footprints (body + pads + margin) and :func:`bare_board` removes
the parts from the SMD scan: masked pixels are filled from the surrounding solder mask (OpenCV Telea
inpainting) and the pad areas are re-painted with the pad colour measured on the visible pad rims
(:func:`pad_colour`). That is the T1 fallback (source 2 of the plan); real bare-pad crops (DNP sites,
:mod:`simple_sim.board3d.padatlas`) replace it per package where they exist.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from simple_sim.board3d.footprints import footprint_for


class BoardRaster:
    """Board mm <-> raster pixels for a window ``[x0, x0 + w] x [y0, y0 + h]`` at ``s`` px/mm."""

    def __init__(self, x0: float, y0: float, width_mm: float, height_mm: float, px_per_mm: float):
        self.x0, self.y0, self.w, self.h, self.s = float(x0), float(y0), float(width_mm), float(height_mm), float(px_per_mm)
        self.cols, self.rows = round(self.w * self.s), round(self.h * self.s)

    def to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        return np.c_[(pts[:, 0] - self.x0) * self.s - 0.5, (self.y0 + self.h - pts[:, 1]) * self.s - 0.5]

    def pixel_to_board(self) -> np.ndarray:
        """3x3 matrix raster px -> board mm."""
        s = self.s
        return np.array([[1 / s, 0, self.x0 + 0.5 / s], [0, -1 / s, self.y0 + self.h - 0.5 / s], [0, 0, 1]])

    def meta(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "widthMm": self.w, "heightMm": self.h, "pxPerMm": self.s}


def rectify(image: np.ndarray, board_to_image: list[float], raster: BoardRaster) -> np.ndarray:
    """Warp *image* (e.g. the flatbed scan) onto *raster* (bicubic)."""
    raster_to_image = np.asarray(board_to_image, float).reshape(3, 3) @ raster.pixel_to_board()
    return cv2.warpPerspective(image, raster_to_image, (raster.cols, raster.rows),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _rot(points: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return points @ np.array([[c, s], [-s, c]])


def _rect_pts(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def component_polygons(c: dict[str, Any], margin_mm: float = 0.0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(body polygons, pad polygons) of one scene component in board mm at its pose, grown by *margin_mm*."""
    if not c.get("pose") or not c.get("body"):
        return [], []
    pose, body = c["pose"], c["body"]
    fp = footprint_for(c)
    centre = np.array([pose["x"], pose["y"]])
    place = lambda pts: _rot(pts, pose["rotDeg"]) + centre  # noqa: E731
    bodies = [place(_rect_pts(0, 0, body["l"] + 2 * margin_mm, body["w"] + 2 * margin_mm))] if body.get("h") != 0.0 else []
    pads = [place(_rect_pts(px, py, pw + 2 * margin_mm, ph + 2 * margin_mm)) for px, py, pw, ph in fp["pads"]]
    return bodies, pads


#: inpaint margin around the bodies (mm): just enough for the body's dark rim in the scan
BODY_MARGIN_MM = 0.06


def part_mask(board3d: dict[str, Any], raster: BoardRaster, stages: set[str], margin_mm: float = BODY_MARGIN_MM,
              keep_lands: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(inpaint mask, lands mask), uint8 0/255, for every placed part of *stages* (measured fits applied).

    The inpaint mask covers the **bodies** only (+ *margin_mm*): the real solder joints stay in the board texture,
    where they are exactly right for the nominal board and still right when a part goes missing (the paste stays
    on the lands) or moves (the 3D joints are drawn over them). Removing the lands as well (``keep_lands=False``,
    the first version) left green rings wherever the estimated lands were smaller than the real joints: first product TOP
    full board, parts Delta E 4.36 with lands removed. DNP sites stay out entirely (their real bare lands)."""
    from simple_sim.board3d.scene import apply_fit
    board3d = apply_fit(board3d)
    parts = np.zeros((raster.rows, raster.cols), np.uint8)
    pads = np.zeros_like(parts)
    for c in board3d["components"]:
        if c["stage"] not in stages or c["dnp"]:
            continue
        bodies, pad_polys = component_polygons(c, margin_mm)
        for poly in pad_polys:
            q = raster.to_px(poly).round().astype(np.int32)
            cv2.fillPoly(pads, [q], 255)
            if not keep_lands:
                cv2.fillPoly(parts, [q], 255)
        for poly in bodies:
            cv2.fillPoly(parts, [raster.to_px(poly).round().astype(np.int32)], 255)
    return parts, pads


def pad_colour(image: np.ndarray, pads: np.ndarray, parts: np.ndarray) -> tuple[int, int, int]:
    """Median colour of bright, unsaturated pixels inside pads (solder / HASL / ENIG), BGR."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sel = (pads > 0) & (hsv[..., 1] < 70) & (hsv[..., 2] > 120)
    if sel.sum() < 50:
        return (190, 190, 185)
    return tuple(int(v) for v in np.median(image[sel], axis=0))


def push_pull_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Smooth fill of *mask* from the known pixels (push-pull: mask-weighted Gaussian pyramid down, then each level's
    holes filled from the coarser one). No streaks: Telea inpainting drew X-shaped smears over every large body
    (coin cell, FPGA), visible at body rims and wherever a tilted camera looks past a part."""
    img = image.astype(np.float32)
    w = (mask == 0).astype(np.float32)
    levels = [(img * w[..., None], w)]
    while min(levels[-1][1].shape) > 8:
        i, k = levels[-1]
        levels.append((cv2.pyrDown(i), cv2.pyrDown(k)))
    i, k = levels[-1]
    fallback = (img * w[..., None]).sum((0, 1)) / max(w.sum(), 1.0)
    est = np.where(k[..., None] > 1e-6, i / np.maximum(k, 1e-6)[..., None], fallback)
    for i, k in reversed(levels[:-1]):
        up = cv2.pyrUp(est, dstsize=(k.shape[1], k.shape[0]))
        val = i / np.maximum(k, 1e-6)[..., None]
        a = np.clip(k * 2.0, 0.0, 1.0)[..., None]      # fully known -> own value; empty -> the coarser estimate
        est = a * val + (1 - a) * up
    return est


def board_grain(image: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fine structure for the filled areas: the solder mask's own high-pass (weave, sheen, vias) copied from other,
    part-free spots of the same board (several random offsets, first valid one per pixel), clipped to the bare
    mask's amplitude so no silkscreen or part edge comes along. Without it the fill is a flat plastic patch."""
    g = image.astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    valid = cv2.dilate(mask, np.ones((9, 9), np.uint8)) == 0
    if valid.sum() < 1000:
        return np.zeros_like(res)
    lim = 1.5 * float(res[valid].std())
    res = np.clip(res, -lim, lim)
    rng = np.random.default_rng(seed)
    out = np.zeros_like(res)
    todo = mask > 0
    h, w = mask.shape
    for _ in range(24):
        if not todo.any():
            break
        dy, dx = int(rng.integers(-h // 3, h // 3)), int(rng.integers(-w // 3, w // 3))
        src, ok = np.roll(res, (dy, dx), (0, 1)), np.roll(valid, (dy, dx), (0, 1))
        take = todo & ok
        out[take] = src[take]
        todo &= ~take
    return out


def bare_board(image: np.ndarray, parts: np.ndarray, pads: np.ndarray, pad_bgr: tuple[int, int, int] | None = None,
               repaint_pads: bool = False) -> np.ndarray:
    """Remove the parts from a rectified board image: fill the part mask smoothly from the surrounding board
    (:func:`push_pull_fill`) and give it the board's grain (:func:`board_grain`). With *repaint_pads* the pads
    (copper finish) come back in a flat measured colour with a slight edge blur; off by default, because in the
    render the solder meshes cover the lands (and carry the scan pixels there)."""
    # the fill does not draw from a ring around the bodies (their dark rim and shadow would tint it); the ring itself
    # stays as scanned (it is the joints' and the body edge's real surroundings)
    ring = max(3, int(round(0.004 * max(image.shape[:2]))))
    fill = push_pull_fill(image, cv2.dilate(parts, np.ones((ring, ring), np.uint8))) + board_grain(image, parts)
    sel = parts > 0
    out = image.copy()
    out[sel] = np.clip(fill[sel] + 0.5, 0, 255).astype(np.uint8)
    if not repaint_pads:
        return out
    colour = pad_bgr or pad_colour(image, pads, parts)
    pad_core = cv2.erode(pads, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(pad_core.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None]
    out = (out * (1 - alpha) + np.array(colour, np.float32) * alpha).astype(np.uint8)
    return out
