"""Real SMD lands from photos of the bare (unpopulated) board.

There are no CAD footprints, so :func:`simple_sim.board3d.footprints.footprint_for` places pads by rule of thumb.
A phone photo of the bare board shows every land as bright copper finish (ENIG / HASL) on the solder mask; this
module registers such a photo to board millimetres and measures the lands:

1. **coarse** (:func:`coarse_homographies`): the board's mounting holes (``family == "mounting_hole"``) are the
   large bright rings of the photo; every 4 ring <-> hole assignment gives a homography, ranked by how many holes
   land on a ring (one-to-one).
2. **lands ICP** (:func:`refine_on_lands`): the estimated pad centres are matched to the land blobs of the photo
   (:func:`land_mask`) with shrinking radii; homography + cubic correction (lens distortion, board bow).
3. **dense** (:func:`refine_on_scan`, optional): the photo warped to the board raster is compared tile by tile
   with the flatbed scan of the substrate (same raster): phase correlation on edge images, a smooth cubic shift
   field on top. The scan is metrically exact, so this removes what the approximate pad estimates leave.
4. **measure** (:func:`measure_lands`): per component and pad index, the land blob under the estimated pad, as
   an axis-aligned rectangle in the part frame (``[cx, cy, w, h]`` like ``footprint_for``'s pads); ``None`` where
   nothing reliable was found (blob shared by two pads, outside the photo, ...), so pad indices stay stable.

The mapping is ``board mm -> photo px``: ``photo = H(q + shift(q)) + cubic(q)`` (:class:`PhotoModel`).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

#: mounting-hole rings in the photo: radius range in px (6.4 mm lands at ~10-35 px/mm)
RING_RADIUS_PX = (30.0, 115.0)
#: bare copper finish in Lab (OpenCV 8-bit, a/b centred): bright, not green, warm
LAND_L_MIN, LAND_A_MIN, LAND_B_MIN = 125, -7, 9


def _basis(q: np.ndarray, ref: tuple[float, float], span: float, deg: int = 3) -> np.ndarray:
    u, v = ((np.asarray(q, float) - ref) / span).T
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], 1)


@dataclass
class PhotoModel:
    """board mm -> photo px: homography *H*, a cubic px correction and a cubic board-mm pre-shift."""

    H: np.ndarray
    ref: tuple[float, float] = (0.0, 0.0)
    span: float = 70.0
    px_coef: np.ndarray | None = None
    shift_coef: list[np.ndarray] = field(default_factory=list)

    def __call__(self, q: Any) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1, 2)
        s = q.copy()
        for c in self.shift_coef:
            s = s + _basis(q, self.ref, self.span) @ c
        p = cv2.perspectiveTransform(s[None], self.H)[0]
        return p if self.px_coef is None else p + _basis(q, self.ref, self.span) @ self.px_coef

    def to_json(self) -> dict[str, Any]:
        return {"H": np.asarray(self.H).tolist(), "ref": list(self.ref), "span": self.span,
                "pxCoef": None if self.px_coef is None else self.px_coef.tolist(),
                "shiftCoef": [c.tolist() for c in self.shift_coef]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PhotoModel:
        return cls(np.array(d["H"]), tuple(d["ref"]), float(d["span"]),
                   None if d.get("pxCoef") is None else np.array(d["pxCoef"]), [np.array(c) for c in d.get("shiftCoef", [])])


# --- masks -------------------------------------------------------------------------------------------------------

#: printed solder paste (grey) instead of bare copper: only the warmth condition is relaxed
PASTE_B_MIN = -8


def land_mask(img: np.ndarray, b_min: int = LAND_B_MIN) -> np.ndarray:
    """255 where the image shows bare copper finish (lands, rings), 0 on solder mask and silkscreen. With
    ``b_min=PASTE_B_MIN`` grey solder-paste deposits count too (a photo taken after the paste print)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    m = ((lab[..., 0] > LAND_L_MIN) & (lab[..., 1] - 128 > LAND_A_MIN) & (lab[..., 2] - 128 > b_min))
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def rings(img: np.ndarray) -> np.ndarray:
    """``(n, 3)`` x, y, r of round land blobs in the radius range of mounting-hole rings, largest first."""
    mask = land_mask(img)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < math.pi * RING_RADIUS_PX[0] ** 2 * 0.3 or not 0.7 < w / h < 1.4:
            continue
        cs, _ = cv2.findContours((lbl[y:y + h, x:x + w] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if RING_RADIUS_PX[0] < r < RING_RADIUS_PX[1] and cv2.contourArea(c) / (math.pi * r * r) > 0.7:
            out.append((x + cx, y + cy, r))
    return np.array(sorted(out, key=lambda t: -t[2])).reshape(-1, 3)


# --- registration --------------------------------------------------------------------------------------------------

def coarse_homographies(ring_xy: np.ndarray, holes_mm: np.ndarray, *, max_rings: int = 10, keep: int = 8,
                        handed: int | None = -1) -> list[tuple[int, float, np.ndarray, float]]:
    """``(hits, mean error px, H, px/mm)`` from 4 ring <-> hole assignments, best first, near-duplicates dropped.

    *handed*: sign the Jacobian determinant must have. Board y points up, image rows down, so a photo looking at the
    side the coordinates are given for has ``-1``; the other side of the board (seen mirrored) has ``+1``. A hole
    pattern alone often fits both, so this keeps a bottom photo from passing as the top. ``None``: either."""
    big = np.asarray(ring_xy, np.float32)[:max_rings, :2]
    M = np.asarray(holes_mm, np.float32)
    if len(big) < 4 or len(M) < 4:
        return []
    c0 = M.mean(0)
    probe = np.array([[c0, c0 + [10, 0], c0 + [0, 10]]], np.float64)
    hyps = []
    for quad in itertools.combinations(range(len(M)), 4):
        for pq in itertools.permutations(range(len(big)), 4):
            H = cv2.getPerspectiveTransform(M[list(quad)], big[list(pq)])
            pr = cv2.perspectiveTransform(M[None].astype(np.float64), H)[0]
            if not np.isfinite(pr).all():
                continue
            q = cv2.perspectiveTransform(probe, H)[0]
            ex, ey = q[1] - q[0], q[2] - q[0]
            if handed is not None and np.sign(ex[0] * ey[1] - ex[1] * ey[0]) != handed:
                continue
            sc = np.array([np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[0])]) / 10
            if not 8 < sc.mean() < 40 or sc.max() / sc.min() > 1.6:
                continue
            dd = np.linalg.norm(pr[:, None] - big[None], axis=2)
            j, d = dd.argmin(1), dd.min(1)
            hit = d < sc.mean()
            if hit.sum() < 5 or len(set(j[hit])) < hit.sum():          # two holes on one ring: degenerate
                continue
            hyps.append((int(hit.sum()), float(d[hit].mean()), H, float(sc.mean())))
    hyps.sort(key=lambda t: (-t[0], t[1]))
    out: list[tuple[int, float, np.ndarray, float]] = []
    for h in hyps:
        if not any(np.allclose(h[2] / h[2][2, 2], g[2] / g[2][2, 2], atol=1e-2) for g in out):
            out.append(h)
        if len(out) == keep:
            break
    return out


def split_lands(mask: np.ndarray, erode_px: int = 2) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected lands with touching neighbours separated: label the eroded cores, then give every mask pixel the
    label of its nearest core (adjacent lands whose gap the photo blur closed). Returns ``(n, labels, stats)`` like
    ``cv2.connectedComponentsWithStats``."""
    from scipy import ndimage

    core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_px) if erode_px else mask
    n, cl = cv2.connectedComponents(core)
    if n <= 1:
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask)
        return n, lbl, st
    _, idx = ndimage.distance_transform_edt(cl == 0, return_indices=True)
    lbl = np.where(mask > 0, cl[idx[0], idx[1]], 0).astype(np.int32)
    ys, xs = np.nonzero(lbl)
    ls = lbl[ys, xs]
    x0 = np.full(n, np.iinfo(np.int64).max); y0 = x0.copy()
    x1 = np.zeros(n, np.int64); y1 = x1.copy()
    np.minimum.at(x0, ls, xs); np.minimum.at(y0, ls, ys)
    np.maximum.at(x1, ls, xs); np.maximum.at(y1, ls, ys)
    st = np.stack([x0, y0, x1 - x0 + 1, y1 - y0 + 1, np.bincount(ls, minlength=n)], 1)
    st[0] = 0
    st[st[:, 4] == 0, :4] = 0
    return n, lbl, st


def blob_centres(mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    return cen[1:][st[1:, 4] >= min_area]


def refine_on_lands(model: PhotoModel, pads_mm: np.ndarray, blobs_px: np.ndarray, px_per_mm: float,
                    radii: tuple[float, ...] = (1.2, 0.8, 0.5, 0.35, 0.25)) -> tuple[PhotoModel, list[dict[str, float]]]:
    """ICP of the estimated pad centres onto the land blobs: homography + trimmed cubic px correction per radius."""
    from scipy.spatial import cKDTree

    tree, hist = cKDTree(blobs_px), []
    for r in radii:
        d, j = tree.query(model(pads_mm), distance_upper_bound=r * px_per_mm)
        ok = np.isfinite(d)
        if ok.sum() < 20:
            break
        src, dst = pads_mm[ok], blobs_px[j[ok]]
        H, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, r * px_per_mm * 0.5)
        if H is None:
            break
        m = PhotoModel(H, model.ref, model.span)
        res, keep = dst - m(src), inl.ravel().astype(bool)
        coef = np.zeros((10, 2))
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(_basis(src[keep], m.ref, m.span), res[keep], rcond=None)
            e = np.linalg.norm(res - _basis(src, m.ref, m.span) @ coef, axis=1)
            keep = e < max(2.5 * float(np.median(e[keep])), 0.08 * px_per_mm)
        model = PhotoModel(H, m.ref, m.span, coef)
        hist.append({"radiusMm": r, "matched": int(ok.sum()), "kept": int(keep.sum()),
                     "medianMm": float(np.median(e[keep]) / px_per_mm)})
    return model, hist


def raster_points(raster: dict[str, float]) -> np.ndarray:
    """Board mm of every pixel centre of a substrate raster (``x0, y0, widthMm, heightMm, pxPerMm``)."""
    s = raster["pxPerMm"]
    cols, rows = round(raster["widthMm"] * s), round(raster["heightMm"] * s)
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    x = raster["x0"] + (u + 0.5) / s
    y = raster["y0"] + raster["heightMm"] - (v + 0.5) / s
    return np.stack([x, y], -1)


def warp_to_board(img: np.ndarray, model: PhotoModel, raster: dict[str, float]) -> np.ndarray:
    """The photo resampled onto the substrate raster (black outside the photo)."""
    q = raster_points(raster)
    p = model(q.reshape(-1, 2)).astype(np.float32).reshape(*q.shape[:2], 2)
    return cv2.remap(img, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _edges(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return e / (np.percentile(e, 99) + 1e-6)


def tile_shifts(warped: np.ndarray, ref: np.ndarray, raster: dict[str, float], *, tile_mm: float = 6.0,
                step_mm: float = 4.0, min_resp: float = 0.12) -> np.ndarray:
    """``(n, 5)`` x, y (tile centre, mm), dx, dy (mm the photo content is off), response, per textured tile."""
    a, b, s = _edges(warped), _edges(ref), raster["pxPerMm"]
    t, st = int(tile_mm * s), int(step_mm * s)
    win = cv2.createHanningWindow((t, t), cv2.CV_32F)
    valid = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    out = []
    for v0 in range(0, a.shape[0] - t, st):
        for u0 in range(0, a.shape[1] - t, st):
            if valid[v0:v0 + t, u0:u0 + t].mean() < 0.98:
                continue
            pa, pb = a[v0:v0 + t, u0:u0 + t], b[v0:v0 + t, u0:u0 + t]
            if pa.mean() < 0.03 or pb.mean() < 0.03:
                continue
            (du, dv), resp = cv2.phaseCorrelate(pb, pa, win)
            if resp < min_resp or abs(du) > t / 4 or abs(dv) > t / 4:
                continue
            x = raster["x0"] + (u0 + t / 2) / s
            y = raster["y0"] + raster["heightMm"] - (v0 + t / 2) / s
            out.append((x, y, du / s, -dv / s, resp))
    return np.array(out).reshape(-1, 5)


def refine_on_scan(img: np.ndarray, model: PhotoModel, scan: np.ndarray, raster: dict[str, float],
                   iterations: int = 4) -> tuple[PhotoModel, list[dict[str, float]]]:
    """Dense refinement against the metric substrate scan: a cubic board-mm shift field per iteration."""
    hist = []
    for _ in range(iterations):
        sh = tile_shifts(warp_to_board(img, model, raster), scan, raster)
        if len(sh) < 20:
            break
        xy, d, w = sh[:, :2], sh[:, 2:4], sh[:, 4]
        keep = np.ones(len(xy), bool)
        coef = np.zeros((10, 2))
        for _ in range(4):
            A = _basis(xy[keep], model.ref, model.span) * w[keep, None]
            coef, *_ = np.linalg.lstsq(A, d[keep] * w[keep, None], rcond=None)
            e = np.linalg.norm(_basis(xy, model.ref, model.span) @ coef - d, axis=1)
            keep = e < max(3 * float(np.median(e[keep])), 0.05)
        mag = np.linalg.norm(d, axis=1)
        hist.append({"tiles": len(sh), "medianShiftMm": float(np.median(mag)), "p90ShiftMm": float(np.percentile(mag, 90)),
                     "fitResidualMm": float(np.median(e[keep]))})
        model = PhotoModel(model.H, model.ref, model.span, model.px_coef, [*model.shift_coef, coef])
    sh = tile_shifts(warp_to_board(img, model, raster), scan, raster)
    if len(sh):
        mag = np.linalg.norm(sh[:, 2:4], axis=1)
        hist.append({"tiles": len(sh), "medianShiftMm": float(np.median(mag)), "p90ShiftMm": float(np.percentile(mag, 90))})
    return model, hist


# --- measuring --------------------------------------------------------------------------------------------------

def _rot(points: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray(points, float) @ np.array([[c, s], [-s, c]])


def to_px(xy: Any, raster: dict[str, float]) -> np.ndarray:
    xy, s = np.asarray(xy, float).reshape(-1, 2), raster["pxPerMm"]
    return np.c_[(xy[:, 0] - raster["x0"]) * s - 0.5, (raster["y0"] + raster["heightMm"] - xy[:, 1]) * s - 0.5]


def to_mm(px: Any, raster: dict[str, float]) -> np.ndarray:
    px, s = np.asarray(px, float).reshape(-1, 2), raster["pxPerMm"]
    return np.c_[raster["x0"] + (px[:, 0] + 0.5) / s, raster["y0"] + raster["heightMm"] - (px[:, 1] + 0.5) / s]


def _same_land_pairs(img: np.ndarray, markers: np.ndarray, mask: np.ndarray, valley: float) -> set[int]:
    """Labels (>= 2) of the watershed whose border with another label crosses no darker gap: one land, two pads."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    ys, xs = np.nonzero((markers == -1) & (mask > 0))
    if not len(ys):
        return set()
    h, w = markers.shape
    nb = np.stack([markers[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1)
    nb = np.where(nb >= 2, nb, 0)
    hi = nb.max(1)
    lo = np.where(nb > 0, nb, np.iinfo(np.int32).max).min(1)
    two = (hi > 0) & (lo != hi) & (lo < np.iinfo(np.int32).max)
    pairs: dict[tuple[int, int], list[float]] = {}
    for a, b, v in zip(lo[two], hi[two], L[ys[two], xs[two]]):
        pairs.setdefault((int(a), int(b)), []).append(float(v))
    labels = np.unique(np.concatenate([lo[two], hi[two]])) if two.any() else np.array([], int)
    level = {int(k): float(np.median(L[(markers == k) & (mask > 0)])) for k in labels}
    same: set[int] = set()
    for (a, b), vals in pairs.items():
        if len(vals) >= 2 and float(np.median(vals)) > valley * min(level[a], level[b]):
            same |= {a, b}
    return same


def _ring_outer_diameter(lum: np.ndarray, valid: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> float | None:
    """Outer diameter (px) of a bright ring at half contrast between ring and surroundings (radial median)."""
    x0, y0 = int(max(0, cx - r_out)), int(max(0, cy - r_out))
    x1, y1 = int(min(lum.shape[1], cx + r_out)), int(min(lum.shape[0], cy + r_out))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - cx, yy - cy)
    L, ok = lum[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    bins = np.arange(r_in, r_out, 0.25)
    prof = np.array([np.median(L[(r >= a) & (r < a + 0.25) & ok]) if ((r >= a) & (r < a + 0.25) & ok).sum() > 8 else np.nan
                     for a in bins])
    if np.isnan(prof).mean() > 0.2:
        return None
    ring, out = np.nanmax(prof[: len(prof) // 3]), np.nanmedian(prof[-8:])
    above = np.nonzero(prof > (ring + out) / 2)[0]
    return None if not len(above) or ring - out < 20 else 2 * (bins[above[-1]] + 0.125)


def edge_bias(warped: np.ndarray, board3d: dict[str, Any], raster: dict[str, float],
              scan: np.ndarray | None = None) -> tuple[float, int]:
    """(mm, rings used): how far a half-contrast edge of this photo lies outside the copper edge. Blur, glare and the
    lens spread bright lands on the dark mask; the lands are shrunk by this per edge.

    Reference: the same rings (mounting holes, large THT lands) measured the same way on the metric flatbed *scan*,
    so only what the photo adds is removed; without a scan the mounting holes' nominal land diameter
    (``MTGP640...``: 6.40 mm). ``(0, 0)`` without usable rings."""
    import re

    s = raster["pxPerMm"]
    lum = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.7)
    valid = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    errs: list[float] = []
    if scan is not None:
        slum = cv2.GaussianBlur(cv2.cvtColor(scan, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.7)
        sval = np.ones(slum.shape, bool)
        for cx, cy, r in rings(scan):
            ref = _ring_outer_diameter(slum, sval, cx, cy, 0.7 * r, 1.45 * r)
            got = _ring_outer_diameter(lum, valid, cx, cy, 0.7 * r, 1.45 * r)
            if ref is not None and got is not None and abs(got - ref) / s < 0.5:
                errs.append((got - ref) / s / 2)
        return (float(np.median(errs)), len(errs)) if errs else (0.0, 0)
    for c in board3d.get("components", []):
        m = re.match(r"MTGP(\d{3})", str(c.get("package", "")).upper())
        if c.get("family") != "mounting_hole" or not m or not c.get("pose"):
            continue
        nominal = int(m[1]) / 100
        cx, cy = to_px([[c["pose"]["x"], c["pose"]["y"]]], raster)[0]
        d = _ring_outer_diameter(lum, valid, cx, cy, 0.35 * nominal * s, 0.72 * nominal * s)
        if d is not None and abs(d / s - nominal) < 0.5:
            errs.append((d / s - nominal) / 2)
    return (float(np.median(errs)), len(errs)) if errs else (0.0, 0)


def land_finish(img: np.ndarray, centres_mm: list[np.ndarray], raster: dict[str, float]) -> str:
    """``"copper"`` (warm ENIG / bare copper) or ``"paste"`` (grey print) from the colour at the estimated pad
    centres that are not solder mask."""
    if not centres_mm:
        return "copper"
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    px = to_px(np.array(centres_mm), raster).round().astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < img.shape[1]) & (px[:, 1] < img.shape[0])
    v = lab[px[ok, 1], px[ok, 0]]
    v = v[(v[:, 0] > LAND_L_MIN) & (v[:, 1] - 128 > LAND_A_MIN)]
    return "copper" if len(v) and float(np.median(v[:, 2] - 128)) > LAND_B_MIN else "paste"


def _half_contrast(region: np.ndarray, lum: np.ndarray, markers: np.ndarray, label: int, grow_px: int) -> np.ndarray:
    """*region* grown to the land edge at half contrast: the colour mask cuts the blurred rim short (it needs the
    full copper hue), the edge of a blurred step is where the brightness is halfway between land and solder mask."""
    x, y, w, h = cv2.boundingRect(region)
    g = grow_px + 2
    x0, y0 = max(0, x - g), max(0, y - g)
    x1, y1 = min(region.shape[1], x + w + g), min(region.shape[0], y + h + g)
    r = region[y0:y1, x0:x1]
    ring = cv2.dilate(r, np.ones((3, 3), np.uint8), iterations=grow_px) - r
    outside = cv2.dilate(r, np.ones((3, 3), np.uint8), iterations=grow_px + 2) - cv2.dilate(r, np.ones((3, 3), np.uint8),
                                                                                           iterations=grow_px)
    L = lum[y0:y1, x0:x1]
    if not r.any() or not outside.any():
        return region
    mid = 0.5 * (float(np.median(L[r > 0])) + float(np.median(L[outside > 0])))
    own = markers[y0:y1, x0:x1]
    add = (ring > 0) & (L > mid) & ((own == label) | (own <= 1))          # own basin, border or solder mask
    out = region.copy()
    out[y0:y1, x0:x1] |= add.astype(np.uint8)
    return out


def measure_lands(warped: np.ndarray, board3d: dict[str, Any], raster: dict[str, float], *,
                  max_area_ratio: float = 40.0, valley: float = 0.9, bias_mm: float = 0.0,
                  ) -> tuple[dict[str, list[list[float] | None]], np.ndarray, dict[str, Any]]:
    """(lands per refdes, land mask of the warped photo, stats). A land is ``[x, y, w, h, rotDeg]`` in board mm:
    the centre on the board, the extent along the part axes of the pose it was measured at (*rotDeg*). Lands are
    fixed to the board, not to the part: :func:`land_in_part` places them relative to any later (fitted) pose.

    Every estimated pad puts a marker on the land under it; a marker-controlled watershed splits lands the photo
    blur merged (0402 neighbours, fine-pitch rows) at the line where they meet. Where that line shows no darker
    valley (median brightness above *valley* x the darker side), the two pads really share one land (a wrong
    estimate): both stay unmeasured, like pads outside the photo or on a land far larger than the estimate.
    *bias_mm* (:func:`edge_bias`) is taken off every edge."""
    from simple_sim.board3d.substrate import component_polygons

    valid = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    comps = {c["refdes"]: c for c in board3d.get("components", []) if not c.get("dnp") and c.get("pose")}
    finish = land_finish(warped, [p.mean(0) for c in comps.values() for p in component_polygons(c)[1]], raster)
    mask = land_mask(warped, LAND_B_MIN if finish == "copper" else PASTE_B_MIN)
    mask[~valid] = 0
    markers = np.where(mask > 0, 0, 1).astype(np.int32)              # 1 = solder mask (background)
    slots: list[tuple[str, int, int]] = []                           # (refdes, pad index, est. area px)
    for ref, c in comps.items():
        for k, p in enumerate(component_polygons(c)[1]):
            poly = to_px(p, raster)
            m = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
            if not m.any() or not valid[m > 0].all():
                continue
            ctr = poly.mean(0)
            inner = np.zeros_like(m)
            cv2.fillPoly(inner, [((poly - ctr) * 0.4 + ctr).round().astype(np.int32)], 1)
            seed = (inner > 0) & (mask > 0) & (markers == 0)
            if not seed.any():
                seed = (m > 0) & (mask > 0) & (markers == 0)
            if not seed.any():
                continue
            markers[seed] = len(slots) + 2
            slots.append((ref, k, int(m.sum())))
    cv2.watershed(warped, markers)
    same = _same_land_pairs(warped, markers, mask, valley)
    lum = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 0.7)
    grow_px = max(1, round(0.12 * raster["pxPerMm"]))
    lands: dict[str, list[list[float] | None]] = {ref: [None] * len(component_polygons(c)[1]) for ref, c in comps.items()}
    found = shared = 0
    for i, (ref, k, est_area) in enumerate(slots):
        label = i + 2
        if label in same:
            shared += 1
            continue
        region = ((markers == label) & (mask > 0)).astype(np.uint8)
        area = int(region.sum())
        if area == 0 or area > max_area_ratio * est_area:
            continue
        region = _half_contrast(region, lum, markers, label, grow_px)
        cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = comps[ref]
        centre, rot = np.array([c["pose"]["x"], c["pose"]["y"]]), float(c["pose"]["rotDeg"])
        pts = _rot(to_mm(max(cs, key=cv2.contourArea).reshape(-1, 2), raster) - centre, -rot)
        lo, hi = pts.min(0), pts.max(0)
        x, y = (_rot(((lo + hi) / 2)[None], rot) + centre)[0]
        w_mm, h_mm = (max(0.05, float(v) - 2 * bias_mm) for v in (hi - lo))
        lands[ref][k] = [round(float(x), 4), round(float(y), 4), round(w_mm, 4), round(h_mm, 4), rot]
        found += 1
    total = sum(len(v) for v in lands.values())
    return lands, mask, {"pads": total, "measured": found, "shared": shared, "finish": finish}


def land_in_part(land: list[float], pose: dict[str, Any]) -> tuple[float, float, float, float]:
    """A board-frame land ``[x, y, w, h, rotDeg]`` as a pad ``(cx, cy, w, h)`` in the frame of *pose*."""
    x, y, w, h, rot = land
    cx, cy = _rot(np.array([[x - pose["x"], y - pose["y"]]]), -float(pose["rotDeg"]))[0]
    turn = (float(pose["rotDeg"]) - rot) % 180.0
    if 45.0 < turn < 135.0:
        w, h = h, w
    return float(cx), float(cy), float(w), float(h)


def fuse_lands(per_photo: list[dict[str, list[list[float] | None]]]) -> dict[str, list[list[float] | None]]:
    """Median of every land over the photos that measured it."""
    out: dict[str, list[list[float] | None]] = {}
    for ref in sorted({r for d in per_photo for r in d}):
        n = max(len(d.get(ref, [])) for d in per_photo)
        pads: list[list[float] | None] = []
        for k in range(n):
            vals = [d[ref][k] for d in per_photo if k < len(d.get(ref, [])) and d[ref][k] is not None]
            pads.append(None if not vals else [round(float(v), 4) for v in np.median(np.array(vals), 0)])
        out[ref] = pads
    return out


def register(img: np.ndarray, board3d: dict[str, Any], *, scan: np.ndarray | None = None,
             raster: dict[str, float] | None = None, min_land_share: float = 0.15) -> tuple[PhotoModel, dict[str, Any]]:
    """Register a bare-board photo of *board3d*'s side: coarse (mounting holes) -> lands ICP -> dense (scan).

    A hole assignment counts only if the lands ICP then holds at least *min_land_share* of the estimated pads."""
    from simple_sim.board3d.substrate import component_polygons

    holes = np.array([[c["pnp"]["x"], c["pnp"]["y"]] for c in board3d["components"] if c.get("family") == "mounting_hole"])
    if len(holes) < 5 and scan is not None and raster is not None:
        # the side's placements carry no mounting holes (they are often placed on one side only): take the rings
        # of the metric scan itself
        found = rings(scan)
        if len(found) >= len(holes):
            holes = to_mm(found[:, :2], raster)
    pads = np.array([p.mean(0) for c in board3d["components"] if not c.get("dnp") for p in component_polygons(c)[1]])
    ref = (board3d.get("widthMm", 100) / 2, board3d.get("heightMm", 100) / 2)
    blobs = blob_centres(land_mask(img))
    best: tuple[int, PhotoModel, dict[str, Any]] | None = None
    # placement coordinates of both sides are given as seen from the top: a photo of the bottom side is mirrored
    handed = 1 if board3d.get("side") == "bottom" else -1
    for hits, err, H, scale in coarse_homographies(rings(img), holes, handed=handed):
        model, hist = refine_on_lands(PhotoModel(H, ref), pads, blobs, scale)
        if not hist or hist[-1]["kept"] < max(20, min_land_share * len(pads)):
            continue                      # the holes fit, the lands do not: other side, other board, symmetric holes
        score = hist[-1]["kept"]
        if best is None or score > best[0]:
            best = (score, model, {"holeHits": hits, "holeErrorPx": err, "pxPerMm": scale, "lands": hist})
    if best is None:
        raise ValueError("bare photo: no registration (need >= 5 of the board's mounting holes and lands that match them; "
                         "a photo of the other board side does not match)")
    _, model, stats = best
    if scan is not None and raster is not None:
        model, stats["scan"] = refine_on_scan(img, model, scan, raster)
    return model, stats
