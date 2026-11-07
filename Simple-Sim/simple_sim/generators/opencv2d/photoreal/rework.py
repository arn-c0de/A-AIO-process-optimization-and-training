"""Rework residue (flux, smeared contamination, patch wires): a nuisance kept clear of the inspected part.
"""

from __future__ import annotations

import cv2
import numpy as np

from .context import RenderContext
from .imageops import blend, smooth_noise


REWORK_PROBABILITY = 0.15  # default share of images with rework residue (``render.rework_artifacts``)


FLUX = np.array((55, 105, 140), dtype=np.float32)   # amber, translucent


DIRT = np.array((43, 114, 131), dtype=np.float32)   # olive-yellow smeared contamination, unmixed from a board photo


def draw_rework(ctx: RenderContext, img: np.ndarray, keepout: tuple[int, int, int, int], ref: float, rng: np.random.Generator,
            probability: float) -> None:
    """Residue seen on the production boards: flux residue, smeared yellow contamination, patch wires.

    A nuisance, not a defect class: always kept clear of the inspected part (its keep-out plus a gap)
    so the label is unaffected; the model learns to ignore it.
    No random draw when disabled (``probability <= 0``).
    """
    if probability <= 0.0 or rng.random() >= probability:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(np.clip(v, 0, lim)) for v, lim in zip(keepout, (w, h, w, h)))
    protected = img[y0:y1, x0:x1].copy()

    def clear(pts: np.ndarray, margin: float) -> bool:
        m = 4 * ctx.ss + margin  # gap + stroke half-width + blur
        return not np.any((pts[:, 0] > keepout[0] - m) & (pts[:, 0] < keepout[2] + m)
                          & (pts[:, 1] > keepout[1] - m) & (pts[:, 1] < keepout[3] + m))

    def stroke(margin: float) -> np.ndarray | None:
        """Smooth curve (quadratic Bezier) clear of the keep-out, or None."""
        for _try in range(10):
            ctrl = np.stack([rng.uniform(-0.1 * w, 1.1 * w, 3), rng.uniform(-0.1 * h, 1.1 * h, 3)], axis=1)
            t = np.linspace(0.0, 1.0, 60)[:, None]
            curve = (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t**2 * ctrl[2]
            if clear(curve, margin):
                return curve.astype(np.int32)
        return None

    for kind in rng.permutation(["flux", "dirt", "wire"])[: int(rng.integers(1, 3))]:
        layer = np.zeros((h, w), dtype=np.uint8)
        if kind == "flux":
            r = float(rng.uniform(0.08, 0.2)) * min(h, w)
            for _try in range(10):
                c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
                ring = c + r * np.stack([np.cos(np.linspace(0, 2 * np.pi, 24)), np.sin(np.linspace(0, 2 * np.pi, 24))], 1)
                if clear(ring, 6 * ctx.ss):
                    break
            else:
                continue
            cv2.ellipse(layer, (int(c[0]), int(c[1])), (int(r), int(r * rng.uniform(0.5, 1.0))),
                        float(rng.uniform(0, 180)), 0, 360, 255, -1)
            blob = (layer > 0) & (smooth_noise((h, w), 0.6 * r, rng) > -0.7)  # irregular outline
            a = cv2.GaussianBlur(blob.astype(np.float32), (0, 0), 2 * ctx.ss)
            rim = np.clip(cv2.GaussianBlur(blob.astype(np.float32), (0, 0), 1.0 * ctx.ss) - a, 0, 1)  # dried edge
            blend(img, FLUX * float(rng.uniform(0.85, 1.1)), a * float(rng.uniform(0.25, 0.45)))
            img *= (1.0 - 0.5 * rim)[..., None]
        elif kind == "dirt":
            width = int(round(float(rng.uniform(0.35, 0.8)) * ref))
            curve = stroke(width / 2 + 4 * ctx.ss)
            if curve is None:
                continue
            cv2.polylines(layer, [curve], False, 255, width, lineType=cv2.LINE_AA)
            a = cv2.GaussianBlur(layer, (0, 0), 1.5 * ctx.ss).astype(np.float32) / 255.0
            blend(img, DIRT * float(rng.uniform(0.85, 1.1)), a * float(rng.uniform(0.5, 0.7)))
        else:
            wd = max(ctx.ss, int(round(float(rng.uniform(0.08, 0.15)) * ref)))
            curve = stroke(wd + 5 * ctx.ss)  # incl. the offset shadow
            if curve is None:
                continue
            shadow = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(shadow, [curve + int(1.5 * ctx.ss)], False, 255, wd + ctx.ss, lineType=cv2.LINE_AA)
            img *= (1.0 - 0.35 * cv2.GaussianBlur(shadow, (0, 0), ctx.ss).astype(np.float32)[..., None] / 255.0)
            cv2.polylines(layer, [curve], False, 255, wd, lineType=cv2.LINE_AA)
            col = np.array((215, 218, 220) if rng.random() < 0.7 else (70, 120, 200), dtype=np.float32)  # white / enamel
            blend(img, col, layer.astype(np.float32) / 255.0)
            core = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(core, [curve - int(0.3 * wd)], False, 255, max(1, wd // 3), lineType=cv2.LINE_AA)
            blend(img, np.full(3, 250.0, dtype=np.float32), core.astype(np.float32) / 255.0 * 0.6)  # highlight
    img[y0:y1, x0:x1] = protected  # hard guarantee: the inspected part's area is never touched
