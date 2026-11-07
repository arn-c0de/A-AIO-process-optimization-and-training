"""Silkscreen print next to the inspected part (outline, pin-1 dot, reference designator).
"""

from __future__ import annotations

import cv2
import numpy as np

from simple_sim.board.footprints import KIND_IC, part_family

from .context import RenderContext
from .imageops import blend, smooth_noise, text_layer
from .palette import CHIPS, SILKSCREEN


def draw_silkscreen(ctx: RenderContext, img: np.ndarray, keepout: tuple[int, int, int, int], family: str, rng: np.random.Generator) -> None:
    if rng.random() < ctx.look.silkscreen_skip:  # dense production boards often have no print next to small parts
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = keepout
    gap = int(3 * ctx.ss)
    layer = np.zeros((h, w), dtype=np.float32)
    thick = max(1, int(round(1.2 * ctx.ss)))
    outline = np.zeros((h, w), dtype=np.uint8)
    if family in CHIPS and rng.random() < 0.6:
        # Two short bars flanking the body (typical passives outline).
        cv2.line(outline, (x0 - gap, y0 - gap), (x0 - gap, y1 + gap), 255, thick)
        cv2.line(outline, (x1 + gap, y0 - gap), (x1 + gap, y1 + gap), 255, thick)
    else:
        cv2.rectangle(outline, (x0 - gap, y0 - gap), (x1 + gap, y1 + gap), 255, thick)
        if part_family(family).kind == KIND_IC:  # moulded ICs: pin-1 dot
            cv2.circle(outline, (x0 - gap - 3 * ctx.ss, y0 - gap - 3 * ctx.ss), int(1.5 * ctx.ss), 255, -1)
    layer = np.maximum(layer, outline.astype(np.float32) / 255.0)

    refdes = f"{part_family(family).refdes}{int(rng.integers(1, 400))}"
    text_h = 9.0 * ctx.ss
    above = y0 - gap - text_h * 1.2
    cy = above if above > text_h else y1 + gap + text_h * 1.2
    cx = x0 + (x1 - x0) * float(rng.uniform(0.25, 0.75))
    angle = 0.0 if rng.random() < 0.7 else 90.0
    layer = np.maximum(layer, text_layer((h, w), refdes, (cx, cy), text_h, angle, max(1, ctx.ss)))

    layer = cv2.GaussianBlur(layer, (0, 0), 0.5 * ctx.ss)
    ink = 0.85 + 0.12 * smooth_noise((h, w), 2 * ctx.ss, rng)  # screen-printed ink is blotchy, not uniform
    blend(img, SILKSCREEN, np.clip(layer * ink, 0, 1))
