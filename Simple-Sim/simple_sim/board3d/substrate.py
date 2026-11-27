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

from simple_sim.board3d.camera import as_mat3
from simple_sim.board3d.footprints import footprint_for


class BoardRaster:
    """Board mm <-> raster pixels for a window ``[x0, x0 + w] x [y0, y0 + h]`` at ``s`` px/mm."""

    def __init__(self, x0: float, y0: float, width_mm: float, height_mm: float, px_per_mm: float):
        self.x0, self.y0, self.w, self.h, self.s = float(x0), float(y0), float(width_mm), float(height_mm), float(px_per_mm)
        self.cols, self.rows = round(self.w * self.s), round(self.h * self.s)

    @classmethod
    def from_meta(cls, meta: dict[str, float], px_per_mm: float | None = None) -> BoardRaster:
        """The raster of a stored ``{x0, y0, widthMm, heightMm, pxPerMm}`` dict (:meth:`meta`), optionally at another
        resolution."""
        return cls(meta["x0"], meta["y0"], meta["widthMm"], meta["heightMm"], meta["pxPerMm"] if px_per_mm is None else px_per_mm)

    def to_px(self, pts: Any) -> np.ndarray:
        """Board mm (N x 2) -> raster px (pixel centres at integers, y down)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        return np.c_[(pts[:, 0] - self.x0) * self.s - 0.5, (self.y0 + self.h - pts[:, 1]) * self.s - 0.5]

    def to_mm(self, px: Any) -> np.ndarray:
        """Raster px (N x 2) -> board mm (inverse of :meth:`to_px`)."""
        px = np.asarray(px, float).reshape(-1, 2)
        return np.c_[self.x0 + (px[:, 0] + 0.5) / self.s, self.y0 + self.h - (px[:, 1] + 0.5) / self.s]

    def fill(self, img: np.ndarray, poly: Any, value: Any = 255) -> None:
        """Fill the board-mm polygon *poly* (N x 2) into *img* (raster-sized) with *value*, in place."""
        cv2.fillPoly(img, [self.to_px(poly).round().astype(np.int32)], value)

    def pixel_to_board(self) -> np.ndarray:
        """3x3 matrix raster px -> board mm."""
        s = self.s
        return np.array([[1 / s, 0, self.x0 + 0.5 / s], [0, -1 / s, self.y0 + self.h - 0.5 / s], [0, 0, 1]])

    def meta(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "widthMm": self.w, "heightMm": self.h, "pxPerMm": self.s}


def rectify(image: np.ndarray, board_to_image: list[float], raster: BoardRaster) -> np.ndarray:
    """Warp *image* (e.g. the flatbed scan) onto *raster* (bicubic)."""
    raster_to_image = as_mat3(board_to_image) @ raster.pixel_to_board()
    return cv2.warpPerspective(image, raster_to_image, (raster.cols, raster.rows),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def board_outline_mm(board3d: dict[str, Any]) -> np.ndarray:
    """4 x 2 corners of the board outline in board mm (counter-clockwise from the origin)."""
    w, h = board3d["widthMm"], board3d["heightMm"]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def board_mask(board3d: dict[str, Any], raster: BoardRaster, value: float = 255, dtype: Any = np.uint8) -> np.ndarray:
    """The board outline filled with *value* on *raster* (0 outside)."""
    mask = np.zeros((raster.rows, raster.cols), dtype)
    raster.fill(mask, board_outline_mm(board3d), value)
    return mask


def rotate_points(points: Any, deg: float) -> np.ndarray:
    """Points (N x 2) turned counter-clockwise by *deg* about the origin."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray(points, float) @ np.array([[c, s], [-s, c]])


def _rect_pts(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def component_polygons(c: dict[str, Any], margin_mm: float = 0.0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(body polygons, pad polygons) of one scene component in board mm at its pose, grown by *margin_mm*."""
    if not c.get("pose") or not c.get("body"):
        return [], []
    pose, body = c["pose"], c["body"]
    fp = footprint_for(c)
    centre = np.array([pose["x"], pose["y"]])
    place = lambda pts: rotate_points(pts, pose["rotDeg"]) + centre  # noqa: E731
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
            raster.fill(pads, poly)
            if not keep_lands:
                raster.fill(parts, poly)
        for poly in bodies:
            raster.fill(parts, poly)
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


#: the clean board (:func:`clean_board`) is CLEAN_BOARD_SCALE x the substrate raster
CLEAN_BOARD_SCALE = 2


def _crisp(mask: np.ndarray, scale: int, sigma: float = 0.7) -> np.ndarray:
    """Binary *mask* upsampled with smooth, narrow edges (alpha 0..1)."""
    up = cv2.resize(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    return np.clip((up - 0.35) / 0.3, 0.0, 1.0)


def clean_board(bare: np.ndarray, pads: np.ndarray, scale: int = CLEAN_BOARD_SCALE) -> tuple[np.ndarray, dict]:
    """The bare board in a clean look at *scale* x: solder mask and traces edge-preserving smoothed (the photo
    noise goes, the trace edges stay); the light, unsaturated features that stand out of their neighbourhood as
    crisp masks - near a land of the footprints (*pads*: rule-of-thumb lands, only a hint) the solder on it, elsewhere
    the silkscreen - each in its measured colour. Returns (BGR image, stats)."""
    base = cv2.bilateralFilter(bare, 9, 18, 4)
    base = cv2.bilateralFilter(base, 9, 12, 3)
    lab = cv2.cvtColor(bare, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 6.0)
    bright = (hp > 16) & (chroma < 22) & (lum > 120)
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, cc, st, _c = cv2.connectedComponentsWithStats(bright, 8)
    keep = np.zeros_like(bright)
    for i in range(1, n):
        if st[i][4] >= 4:
            keep[cc == i] = 1
    near_land = cv2.dilate((pads > 127).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    solder = (keep > 0) & near_land
    silk = (keep > 0) & ~near_land
    out = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    stats = {"scale": scale}
    for name, m, gain in (("solder", solder, 1.05), ("silk", silk, 1.08)):
        col = np.clip(np.median(bare[m], 0) * gain, 0, 255) if m.any() else None
        stats[name] = {"share": round(float(m.mean()), 4), "bgr": None if col is None else [round(float(v)) for v in col]}
        if col is None:
            continue
        a = _crisp(m, scale)[..., None]
        out = out * (1 - a) + col.astype(np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), stats


def resample(image: np.ndarray, src: BoardRaster, dst: BoardRaster, interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """*image* on raster *src* resampled onto raster *dst* (both board mm, y up, axis-aligned)."""
    # dst px -> board mm -> src px
    k = src.s / dst.s
    m = np.float32([[k, 0, (dst.x0 - src.x0) * src.s + 0.5 * k - 0.5],
                    [0, k, (src.y0 + src.h - (dst.y0 + dst.h)) * src.s + 0.5 * k - 0.5]])
    return cv2.warpAffine(image, m, (dst.cols, dst.rows), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def match_colours(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, n: int = 64) -> np.ndarray:
    """*src* with each channel mapped so its quantiles over *mask* match *ref*'s (a scanner to phone colour transfer)."""
    out = np.empty_like(src)
    qs = np.linspace(0.5, 99.5, n)
    for ch in range(3):
        a = np.percentile(src[..., ch][mask], qs)
        b = np.percentile(ref[..., ch][mask], qs)
        a = np.maximum.accumulate(a + np.arange(n) * 1e-6)
        lut = np.clip(np.interp(np.arange(256), a, b), 0, 255).astype(np.uint8)
        out[..., ch] = lut[src[..., ch]]
    return out


def scan_board(photo_bare: np.ndarray, photo_raster: BoardRaster, scan_bare: np.ndarray, scan_raster: BoardRaster,
               parts: np.ndarray, scale: int = CLEAN_BOARD_SCALE) -> tuple[np.ndarray, dict]:
    """The clean board from the flatbed scan (24 px/mm, no perspective, no glare, no tall part in the way) on the
    photo raster at *scale* x, its colours matched to the phone photos on the bare board both show; outside the scan
    (a bracket overhanging the board) the photos' clean board (:func:`clean_board` without the lands hint)."""
    dst = BoardRaster(photo_raster.x0, photo_raster.y0, photo_raster.w, photo_raster.h, photo_raster.s * scale)
    scan = resample(scan_bare, scan_raster, dst)
    cover = resample(np.full(scan_bare.shape[:2], 255, np.uint8), scan_raster, dst, cv2.INTER_NEAREST) > 0
    cover = cv2.erode(cover.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    photo_up = cv2.resize(photo_bare, (dst.cols, dst.rows), interpolation=cv2.INTER_CUBIC)
    free = cover & (cv2.resize(parts, (dst.cols, dst.rows), interpolation=cv2.INTER_NEAREST) < 128)
    scan = match_colours(scan, photo_up, free)
    fallback, _st = clean_board(photo_bare, np.zeros(photo_bare.shape[:2], np.uint8), scale)
    a = cv2.GaussianBlur(cover.astype(np.float32), (0, 0), 2.0)[..., None]
    out = np.clip(scan.astype(np.float32) * a + fallback.astype(np.float32) * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    return out, {"scale": scale, "source": "scan", "coverage": round(float(cover.mean()), 3)}


#: plated holes (:func:`find_holes`): radius range (mm; smaller are vias), the dark core and the bright ring
HOLE_R_MM, HOLE_DARK_L, RING_MIN_L = (0.35, 2.0), 60, 120


def find_holes(scan: np.ndarray, raster: BoardRaster) -> list[list[float]]:
    """Plated through holes in the flatbed scan of the bare (SMD-only) board: round dark cores inside a bright copper
    ring. Returns ``[x, y, r]`` (board mm)."""
    lum = cv2.cvtColor(scan, cv2.COLOR_BGR2LAB)[..., 0]
    dark = cv2.morphologyEx((lum < HOLE_DARK_L).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _cc, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    s = raster.s
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        r = (w + h) / 4
        if not HOLE_R_MM[0] * s <= r <= HOLE_R_MM[1] * s or abs(w - h) > 0.25 * max(w, h) or area < 0.6 * np.pi * r * r:
            continue
        cx, cy = cen[i]
        y0, y1, x0, x1 = (max(int(v), 0) for v in (cy - 1.7 * r, cy + 1.7 * r, cx - 1.7 * r, cx + 1.7 * r))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(gx - cx, gy - cy)
        ring = lum[y0:y1, x0:x1][(rr > 1.15 * r) & (rr < 1.6 * r)]
        if ring.size and ring.mean() >= RING_MIN_L:
            out.append([round(raster.x0 + (cx + 0.5) / s, 3), round(raster.y0 + raster.h - (cy + 0.5) / s, 3), round(r / s, 3)])
    return out
