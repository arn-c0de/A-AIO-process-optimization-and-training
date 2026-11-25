"""T3: a sharp, glare-free top texture of the fully populated board from many single phone photos.

The ``mounted`` source (SimpleAOI's render of one photo, 16 px/mm) is soft and carries that photo's lamp reflection.
DeepVision keeps, per AOI run, the still (``still.jpg``, 3008 x 4000 = ~20 px/mm on the board) and its registration
(``geometry.json``): the homography ``boardToSource`` and the local residual field ``boardField`` (lens distortion,
board bow; ``modules/aoi-engine/src/board-field.js``: a board point q shows in the photo at H(q - d(q)), median
residual 0.02 mm after the field against 0.08-0.10 mm without).

:func:`warp_photo` maps one still onto the board raster **height-aware**: every raster pixel gets the top height of
the part above it (``height_map``), the camera pose comes from the homography (:mod:`simple_sim.board3d.camera`) and
the point (x - dx, y - dy, z) is projected with the full camera, so a 10 mm connector top lands where it is instead of
up to ~3 mm off (plane-only warp, camera ~220 mm above the board). :func:`build_mosaic` normalises each warped photo
to a common exposure and light field (ratio of heavily blurred images: vignetting and the lamp's fall-off cancel) and
fuses per pixel with a low-biased trimmed mean (25th-60th percentile): glare is a bright outlier in a few photos,
noise averages out, registration keeps the detail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.board3d.camera import choose_focal, pose_from_homography
from simple_sim.board3d.footprints import footprint_for
from simple_sim.board3d.substrate import BoardRaster, component_polygons

#: neighbourhood of the residual field (DeepVision FIELD_DEFAULTS): 6 mm, else 10 mm, at least 4 anchors
FIELD_RADII_MM, FIELD_MIN_POINTS = (6.0, 10.0), 4
#: fusion tiles (mm) and how sharply the best photo of a tile wins (softmax temperature on the score)
TILE_MM, TEMPERATURE = 3.0, 0.15
#: light-field normalisation blur (mm)
LIGHT_BLUR_MM = 8.0


def height_map(board3d: dict[str, Any], raster: BoardRaster, stages: set[str]) -> np.ndarray:
    """Top height (mm) of the tallest part body over every raster pixel, 0 on the bare board."""
    from simple_sim.board3d.scene import apply_fit
    board3d = apply_fit(board3d)
    hm = np.zeros((raster.rows, raster.cols), np.float32)
    comps = [c for c in board3d["components"] if c["stage"] in stages and not c["dnp"] and c.get("pose") and c.get("body")]
    for c in sorted(comps, key=lambda c: footprint_for(c)["height"]):      # taller parts drawn last win
        h = float(footprint_for(c)["height"] or 0.0)
        if h <= 0:
            continue
        bodies, _ = component_polygons(c)
        for poly in bodies:
            cv2.fillPoly(hm, [raster.to_px(poly).round().astype(np.int32)], h)
    return hm


def field_grid(field: list[float] | None, raster: BoardRaster, step_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Residual shift d (mm) of :mod:`board-field` on the raster: neighbourhood median at a ``step_mm`` grid, bilinear
    in between; zero without a field or where too few anchors are near."""
    dx = np.zeros((raster.rows, raster.cols), np.float32)
    dy = np.zeros_like(dx)
    if not field:
        return dx, dy
    f = np.asarray(field, float).reshape(-1, 4)
    xs = np.arange(raster.x0, raster.x0 + raster.w + step_mm, step_mm)
    ys = np.arange(raster.y0, raster.y0 + raster.h + step_mm, step_mm)
    gx, gy = np.zeros((len(ys), len(xs)), np.float32), np.zeros((len(ys), len(xs)), np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            d2 = (f[:, 0] - x) ** 2 + (f[:, 1] - y) ** 2
            for r in FIELD_RADII_MM:
                sel = d2 <= r * r
                if sel.sum() >= FIELD_MIN_POINTS:
                    gx[j, i], gy[j, i] = np.median(f[sel, 2]), np.median(f[sel, 3])
                    break
    # grid node (i, j) sits at board (xs[i], ys[j]); raster rows run down from y0 + h
    u = (np.arange(raster.cols) + 0.5) / raster.s / step_mm
    v = (raster.h - (np.arange(raster.rows) + 0.5) / raster.s) / step_mm
    mu, mv = np.meshgrid(u.astype(np.float32), v.astype(np.float32))
    for src, dst in ((gx, dx), (gy, dy)):
        dst[:] = cv2.remap(src, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return dx, dy


def photo_camera(geo: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K, R, t) of a photo from its homography (``X_cam = R X_board + t``, board z towards the camera)."""
    h = np.asarray(geo["boardToSource"], float).reshape(3, 3)
    w, hh = (int(v) for v in geo["sourceSize"])
    f, _src, _est = choose_focal(h, w, hh)
    k = np.array([[f, 0, w / 2], [0, f, hh / 2], [0, 0, 1]])
    r, t = pose_from_homography(h, k)
    return k, r, t


def warp_photo(image: np.ndarray, geo: dict[str, Any], raster: BoardRaster, heights: np.ndarray | None) -> np.ndarray:
    """*image* (the still) as a top view on *raster*: residual field and part heights applied (see module doc)."""
    k, r, t = photo_camera(geo)
    u, v = np.meshgrid(np.arange(raster.cols, dtype=np.float32), np.arange(raster.rows, dtype=np.float32))
    x = raster.x0 + (u + 0.5) / raster.s
    y = raster.y0 + raster.h - (v + 0.5) / raster.s
    dx, dy = field_grid(geo.get("boardField"), raster)
    z = heights if heights is not None else np.zeros_like(x)
    pts = np.stack([x - dx, y - dy, z], -1).reshape(-1, 3).astype(np.float64)
    cam = pts @ r.T + t
    pix = cam @ k.T
    mx = (pix[:, 0] / pix[:, 2]).reshape(x.shape).astype(np.float32)
    my = (pix[:, 1] / pix[:, 2]).reshape(x.shape).astype(np.float32)
    if heights is None:
        # plane only: the homography itself (identical to the camera at z = 0 up to its residual)
        h = np.asarray(geo["boardToSource"], float).reshape(3, 3)
        q = np.stack([x - dx, y - dy, np.ones_like(x)], -1).reshape(-1, 3) @ h.T
        mx, my = (q[:, 0] / q[:, 2]).reshape(x.shape).astype(np.float32), (q[:, 1] / q[:, 2]).reshape(x.shape).astype(np.float32)
    return cv2.remap(image, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _blur_norm(img: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Mask-weighted Gaussian blur (normalised convolution)."""
    w = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma_px)
    b = cv2.GaussianBlur(img.astype(np.float32) * mask[..., None], (0, 0), sigma_px)
    return b / np.maximum(w, 1e-4)[..., None]


def select_photos(dv_root: Path, *, side: str = "top", min_px_per_mm: float = 17.0, max_tilt_deg: float = 12.0,
                  limit: int = 40, exclude: set[str] | None = None) -> list[Path]:
    """Refined geometries with a residual field and a still, sharp enough and near-frontal; the sharpest first."""
    from simple_sim.board3d.camera import blender_camera
    out = []
    for g in sorted(Path(dv_root).glob("store/simple/runs/*/auto/*/geometry.json")):
        if exclude and any(e in str(g) for e in exclude):
            continue
        try:
            geo = json.loads(g.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if geo.get("side", side) != side or not geo.get("refined") or not geo.get("boardField") or not (g.parent / "still.jpg").is_file():
            continue
        h = np.asarray(geo["boardToSource"], float).reshape(3, 3)
        w, hh = (int(v) for v in geo["sourceSize"])
        a, b = h @ [60, 50, 1], h @ [61, 50, 1]
        scale = float(np.hypot(*(b[:2] / b[2] - a[:2] / a[2])))
        try:
            tilt = blender_camera(geo["boardToSource"], w, hh)["tiltDeg"]
        except ValueError:
            continue
        if scale >= min_px_per_mm and tilt <= max_tilt_deg:
            out.append((-scale, tilt, g))
    return [g for *_k, g in sorted(out)[:limit]]


def build_mosaic(board3d: dict[str, Any], raster: BoardRaster, geometries: list[Path], *, stages: set[str],
                 log=print) -> tuple[np.ndarray, dict[str, Any]]:
    """Fused top view on *raster* from the stills of *geometries*; returns (BGR image, report). The stills come from
    many units: what most of them lack (e.g. a coin cell, standoffs) is missing here too."""
    heights = height_map(board3d, raster, stages)
    board = np.zeros((raster.rows, raster.cols), np.float32)
    q = raster.to_px(np.array([[0, 0], [board3d["widthMm"], 0], [board3d["widthMm"], board3d["heightMm"]], [0, board3d["heightMm"]]]))
    cv2.fillPoly(board, [q.round().astype(np.int32)], 1.0)
    flat = board * (heights == 0)
    flat = cv2.erode(flat, np.ones((5, 5), np.uint8))
    stack, used = [], []
    for g in geometries:
        geo = json.loads(Path(g).read_text(encoding="utf-8"))
        img = cv2.imread(str(Path(g).parent / "still.jpg"), cv2.IMREAD_COLOR)
        if img is None or list(img.shape[1::-1]) != [int(v) for v in geo["sourceSize"]]:
            continue
        stack.append(warp_photo(img, geo, raster, heights))
        used.append(str(g))
        log(f"  warped {len(stack)}/{len(geometries)}: {Path(g).parents[2].name}")
    if not stack:
        raise ValueError("no usable photo (still.jpg + geometry.json with boardField)")
    # light field: every photo to the median photo's low-pass (exposure, white balance, vignetting, lamp fall-off)
    sigma = LIGHT_BLUR_MM * raster.s
    small = [cv2.resize(s, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) for s in stack]
    flat_s = cv2.resize(flat, (small[0].shape[1], small[0].shape[0]), interpolation=cv2.INTER_NEAREST)
    ref = np.median(np.stack(small), 0)
    ref_low = _blur_norm(ref, flat_s, sigma / 4)
    gains = []
    for i, s in enumerate(small):
        valid = flat_s * (s.max(2) > 0)
        ratio = ref_low / np.maximum(_blur_norm(s, valid, sigma / 4), 1.0)
        ratio = np.clip(ratio, 0.4, 2.5)
        full = cv2.resize(ratio, (raster.cols, raster.rows), interpolation=cv2.INTER_LINEAR)
        valid_full = stack[i].max(2) > 0
        stack[i] = np.where(valid_full[..., None], np.clip(stack[i] * full, 0, 255), 0).astype(np.uint8)
        gains.append([round(float(v), 3) for v in ratio.reshape(-1, 3).mean(0)])
    out, fusion = fuse(stack, raster.s)
    report = {"photos": used, "count": len(stack), "raster": raster.meta(), "fusion": fusion, "lightBlurMm": LIGHT_BLUR_MM,
              "heightAware": True, "field": True, "lightGains": gains}
    return out, report



def _median(stack: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of the valid (non-black) samples, in row strips."""
    rows = stack[0].shape[0]
    out = np.zeros_like(stack[0])
    for r0 in range(0, rows, 128):
        strip = np.stack([s[r0:r0 + 128] for s in stack]).astype(np.float32)
        strip[np.broadcast_to(strip.max(3, keepdims=True) == 0, strip.shape)] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(strip, 0)
        out[r0:r0 + 128] = np.clip(np.nan_to_num(med) + 0.5, 0, 255).astype(np.uint8)
    return out


def fuse(stack: list[np.ndarray], px_per_mm: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Best photo per tile, softly blended. Averaging registered photos still blurs (0.02-0.06 mm residual = up to
    1 px, and units differ slightly); a single sharp photo per region keeps the detail. Score per tile and photo:
    edge energy (Laplacian variance) divided by the disagreement with the per-pixel median (glare, reflections,
    a cable across, another unit's label all disagree). Weights are a softmax of the log score, upsampled
    bilinearly, so tile seams fade over a tile."""
    med = _median(stack)
    tile = max(8, int(round(TILE_MM * px_per_mm)))
    rows, cols = med.shape[:2]
    ty, tx = -(-rows // tile), -(-cols // tile)
    scores = np.full((len(stack), ty, tx), -np.inf, np.float32)
    med_f = med.astype(np.float32)
    for i, s in enumerate(stack):
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.7), cv2.CV_32F) ** 2
        diff = np.abs(s.astype(np.float32) - med_f).mean(2)
        valid = (s.max(2) > 0).astype(np.float32)
        pad = ((0, ty * tile - rows), (0, tx * tile - cols))
        def tiles(a):
            return np.pad(a, pad).reshape(ty, tile, tx, tile).sum((1, 3))
        n = tiles(valid)
        ok = n >= 0.9 * tile * tile
        energy = tiles(lap * valid) / np.maximum(n, 1)
        disagree = tiles(diff * valid) / np.maximum(n, 1)
        # agreement first (the photos are of real, partly faulty boards: splashes, flux, dirt are sharp too)
        sc = 0.5 * np.log1p(energy) - 4.0 * np.log1p(disagree / 3.0)
        scores[i][ok] = sc[ok]
    best = scores.max(0)
    finite = np.isfinite(scores)
    w = np.zeros_like(scores)
    with np.errstate(invalid="ignore"):
        w[finite] = np.exp((scores - best[None])[finite] / TEMPERATURE)
    no_photo = w.sum(0) == 0
    out = np.zeros((rows, cols, 3), np.float32)
    acc = np.zeros((rows, cols), np.float32)
    for i, s in enumerate(stack):
        wi = cv2.resize(w[i], (tx * tile, ty * tile), interpolation=cv2.INTER_LINEAR)[:rows, :cols]
        wi = wi * (s.max(2) > 0)
        out += s.astype(np.float32) * wi[..., None]
        acc += wi
    fused = out / np.maximum(acc, 1e-6)[..., None]
    fused = np.where((acc > 1e-6)[..., None], fused, med_f)
    winners = np.bincount(scores.argmax(0)[~no_photo].ravel(), minlength=len(stack))
    return np.clip(fused + 0.5, 0, 255).astype(np.uint8), {"method": "best tile (edge energy / disagreement), soft blend",
                                                          "tileMm": TILE_MM, "temperature": TEMPERATURE,
                                                          "tilesWon": winners.tolist()}
