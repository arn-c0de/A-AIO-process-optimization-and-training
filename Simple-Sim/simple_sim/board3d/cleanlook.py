"""The clean look of the part textures: one flat body colour per face with the print as a crisp mask, at
``CLEAN_SCALE`` x the photo atlas of :mod:`simple_sim.board3d.parttex` (its layout and report).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

__all__ = ["CLEAN_SCALE", "NO_PRINT_KINDS", "PRINT_MIN_IOU", "body_box", "clean_atlas", "clean_tile", "print_mask"]

#: the clean atlas is CLEAN_SCALE x the photo atlas: flat body colours with the print as a crisp mask
CLEAN_SCALE = 2
#: print (:func:`print_mask`): Otsu threshold on the high-pass at least PRINT_MIN_DL, print share within PRINT_SHARE,
#: blobs above PRINT_MAX_BLOB of the inside are glare
PRINT_MIN_DL, PRINT_SHARE, PRINT_MAX_BLOB, PRINT_GAIN = 6.0, (0.02, 0.30), 0.08, 1.8
#: share of the body size along its outline that is never print
PRINT_EDGE = 0.10


def print_mask(body: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mask float 0..1 at the crop's size, body colour BGR, ink colour BGR) of a fused top-of-body crop.

    The print contrasts with the body (light laser marking on a dark mould, dark ink on a light body): on the
    high-pass of the luminance (slow shading cancels) inside the body (a band along the outline - edge shading, a
    QFN's terminals, the neighbour - is left out), Otsu splits print from body. A fixed noise threshold missed the
    soft, dense marking of the photo atlas (contrast 15-20 against a noise MAD of ~7). The split counts only when it
    is real (threshold >= PRINT_MIN_DL, print share PRINT_SHARE); blobs over PRINT_MAX_BLOB of the inside are glare."""
    lab = cv2.cvtColor(body, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum = lab[..., 0]
    rows, cols = lum.shape
    band = max(2, int(round(PRINT_EDGE * min(rows, cols))))
    inner = np.zeros((rows, cols), bool)
    inner[band:rows - band, band:cols - band] = True
    if inner.sum() < 16:
        inner[:] = True
    base = np.median(body[inner], 0)
    base_l = float(np.median(lum[inner]))
    sgn = 1.0 if base_l < 128 else -1.0
    hp = sgn * (lum - cv2.GaussianBlur(lum, (0, 0), max(3.0, 0.08 * min(rows, cols))))
    v8 = np.clip(np.clip(hp[inner], 0, None) * 4, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t /= 4.0
    raw = (hp > max(t, PRINT_MIN_DL)) & inner
    share = raw.sum() / max(inner.sum(), 1)
    empty = np.zeros((rows, cols), np.float32)
    if t < PRINT_MIN_DL or not PRINT_SHARE[0] <= share <= PRINT_SHARE[1]:
        return empty, base, base
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_cc, stats, _c = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros_like(raw)
    for i in range(1, n):
        if 3 <= stats[i][4] <= PRINT_MAX_BLOB * inner.sum():
            keep[lab_cc == i] = 1
    if not keep.any():
        return empty, base, base
    # the fused photo blurs the strokes into the body: their mean is too close to it; a sharp marking has
    # about PRINT_GAIN x that contrast
    ink = np.clip(base + PRINT_GAIN * (np.median(body[keep > 0], 0) - base), 0, 255)
    return keep.astype(np.float32), base, ink


#: kinds that never get a print: a crystal's welded lid (its rim is no print), connectors (the photos show cables,
#: brackets and latches on them, no legible marking)
NO_PRINT_KINDS = ("crystal", "connector", "box")
#: the print is dropped where two disjoint halves of the photos both find one but not the same (IoU, 1 px
#: tolerance): real marking is reproducible, blur and noise are not
PRINT_MIN_IOU = 0.35


def body_box(part_size: tuple[float, float, float], bounds: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """(c0, r0, c1, r1) of the body outline inside a top tile."""
    l, w, _h = part_size
    u0, u1, v0, v1 = bounds
    rows, cols = shape
    return (int(round((-l / 2 - u0) / (u1 - u0) * cols)), int(round((v1 - w / 2) / (v1 - v0) * rows)),
            int(round((l / 2 - u0) / (u1 - u0) * cols)), int(round((v1 + w / 2) / (v1 - v0) * rows)))


def clean_tile(tile: np.ndarray, body_box: tuple[int, int, int, int] | None, scale: int = CLEAN_SCALE, *,
               with_print: bool = True) -> np.ndarray:
    """A face tile in the clean look at *scale* x: one colour (the body's) with its print as a crisp, anti-aliased
    overlay. *body_box* (c0, r0, c1, r1) is the body outline inside a top tile (None: the whole tile, walls)."""
    rows, cols = tile.shape[:2]
    c0, r0, c1, r1 = body_box or (0, 0, cols, rows)
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, cols), min(r1, rows)
    out_size = (cols * scale, rows * scale)
    if c1 - c0 < 4 or r1 - r0 < 4:
        return cv2.resize(tile, out_size, interpolation=cv2.INTER_CUBIC)
    body = tile[r0:r1, c0:c1]
    if body_box is None:
        base = np.median(body.reshape(-1, 3), 0)
        return np.broadcast_to(np.clip(base, 0, 255).astype(np.uint8), (rows * scale, cols * scale, 3)).copy()
    mask, base, ink = print_mask(body)
    if not with_print:
        mask[:] = 0.0
    full = np.zeros((rows, cols), np.float32)
    full[r0:r1, c0:c1] = mask
    # crisp edges at the higher resolution: smooth the binary mask, upsample, re-threshold with a narrow ramp
    up = cv2.resize(cv2.GaussianBlur(full, (0, 0), 0.7), out_size, interpolation=cv2.INTER_CUBIC)
    alpha = np.clip((up - 0.35) / 0.3, 0.0, 1.0)[..., None]
    flat = np.empty((rows * scale, cols * scale, 3), np.float32)
    flat[:] = base
    return np.clip(flat * (1 - alpha) + ink.astype(np.float32) * alpha + 0.5, 0, 255).astype(np.uint8)


def clean_atlas(atlas: np.ndarray, report: dict[str, Any], scale: int = CLEAN_SCALE) -> np.ndarray:
    """The clean-look atlas (same layout x *scale*) from the photo atlas and its report."""
    out = np.zeros((atlas.shape[0] * scale, atlas.shape[1] * scale, 3), np.uint8)
    for part in report["parts"].values():
        for f, info in part["faces"].items():
            if "rect" not in info:
                continue
            x, y, tw, th = info["rect"]
            tile = atlas[y:y + th, x:x + tw]
            box_ = None
            if f == "top" and part.get("kind") == "can":
                # electrolytic cans: the large print and the polarity bar are well resolved: keep the photo, sharper
                big = cv2.resize(tile, (tw * scale, th * scale), interpolation=cv2.INTER_CUBIC)
                out[y * scale:(y + th) * scale, x * scale:(x + tw) * scale] = big
                continue
            if f == "top":
                box_ = body_box(part["size"], info["bounds"], (th, tw))
            # a crystal's seam-welded lid: its rim is no print, one colour; elsewhere the print only where it is
            # reproducible across the photos (print_agreement; older reports without it: kept)
            iou = part.get("printIoU")
            legible = iou is None or iou >= PRINT_MIN_IOU        # None: no evidence against the print
            out[y * scale:(y + th) * scale, x * scale:(x + tw) * scale] = clean_tile(
                tile, box_, scale, with_print=part.get("kind") not in NO_PRINT_KINDS and legible)
    return out
