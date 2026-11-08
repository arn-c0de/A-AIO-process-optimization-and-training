"""Pad finish and solder joints (legacy fillets, or the meniscus model of ``render.solder``).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import meniscus as solder_model
from .context import RenderContext
from .imageops import blend, grid, shade_solder, skip_smooth_noise, smooth_noise, solder_env
from .palette import SOLDER


def shade_blob(ctx: RenderContext, hmap: np.ndarray, board: np.ndarray, spec_power: float, spec_gain: float) -> np.ndarray:
    """Shade a free solder shape (ball, excess bulge) with the joint model of this render."""
    look = ctx.solder_look
    if look is None:
        return shade_solder(hmap, SOLDER, solder_env(board), spec_power, spec_gain)
    return solder_model.shade(hmap, board, look)


def thermal_pad_index(pads: list[dict[str, int]]) -> int | None:
    """Index of an exposed thermal pad (QFN): the largest pad, if clearly larger than the others."""
    if len(pads) < 3:
        return None
    areas = [p["w"] * p["h"] for p in pads]
    i = max(range(len(areas)), key=areas.__getitem__)
    return i if areas[i] > 3 * sorted(areas)[len(areas) // 2] else None


def pads_and_solder(ctx: RenderContext, img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], body_mask: np.ndarray,
                     rng: np.random.Generator, finish: np.ndarray, *, with_solder: bool = True,
                     excess: frozenset[int] = frozenset()) -> None:
    """Pad finish and solder; pads in *excess* get a high, convex (bulging) joint instead of a fillet."""
    h, w = img.shape[:2]
    solder = SOLDER * float(rng.uniform(0.9, 1.05))
    mask_edge = base * 0.65
    env = solder_env(base)  # what slopes of the solder mirror: the surrounding board
    thermal = thermal_pad_index(pads)
    look = ctx.solder_look if with_solder else None
    # Surface irregularity: real fillets are lumpy, not mathematically smooth. (The meniscus model has
    # its own relief; it only keeps the random stream in step.)
    if look is None:
        lumps = smooth_noise((h, w), 8 * ctx.ss, rng) * 0.06
    else:
        skip_smooth_noise((h, w), 8 * ctx.ss, rng)
    # render.solder: meniscus draws from a side stream: *rng* advances exactly as for legacy joints.
    joint_rng = solder_model.side_stream(rng, 1) if look is not None else None

    for i, pad in enumerate(pads):
        x0, y0 = max(0, pad["x"]), max(0, pad["y"])
        x1, y1 = min(w, pad["x"] + pad["w"]), min(h, pad["y"] + pad["h"])
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        # Solder-mask opening: dark rim then copper finish.
        cv2.rectangle(img, (x0 - ctx.ss, y0 - ctx.ss), (x1 + ctx.ss - 1, y1 + ctx.ss - 1), mask_edge.tolist(), ctx.ss)
        region = img[y0:y1, x0:x1]
        region[:] = finish
        if not with_solder:
            continue

        ph, pw = region.shape[:2]
        covered = body_mask[y0:y1, x0:x1] > 0
        amp = min(0.55 * min(pw, ph), 9.0 * ctx.ss)  # solder height; big (thermal) pads stay flat
        # Thermal pads are printed through a window-pane stencil (separate solder islands).
        cells = (max(2, round(3 * pw / min(pw, ph))), max(2, round(3 * ph / min(pw, ph)))) if i == thermal else (1, 1)
        if not covered.any() and ctx.missing_residue <= 0.0:
            if look is not None:
                solder_model.skip_joint_draws(region.shape[:2], covered, look, joint_rng, ctx.ss)
            continue  # render.missing_residue 0: bare pad finish, no reflowed paste (random stream kept in step)
        if look is not None:
            solder_model.render_joint(region, base, covered, amp, look, joint_rng, ctx.ss, excess=i in excess, cells=cells,
                                      residue=ctx.missing_residue)
            continue  # region is a view: already written
        yy, xx = grid(ph, pw)
        edge = np.minimum(np.minimum(xx, pw - 1 - xx), np.minimum(yy, ph - 1 - yy))
        rim = np.clip(edge / (1.5 * ctx.ss), 0.0, 1.0)  # copper visible at the pad rim
        if covered.any():
            dist =cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 3)
            t = 1.0 - dist / max(1.0, float(dist.max()))
            if i in excess:  # too much solder: convex bulge, higher than the terminal
                hmap = np.where(covered, 1.0, t ** 0.5) * rim * amp * 1.5
            else:
                hmap = np.where(covered, 1.0, t ** 1.2) * rim * amp  # concave fillet rising to the terminal
            spec_power, spec_gain = 40.0, 0.9
        else:
            # Reflowed paste, no part: one flat island per stencil window, bare finish in between.
            nx, ny = cells
            u, v = ((xx / pw * nx) % 1.0) * 2 - 1, ((yy / ph * ny) % 1.0) * 2 - 1
            dome = np.clip(1.0 - (np.abs(u) ** 3 + np.abs(v) ** 3) * (0.85 if nx * ny == 1 else 1.2), 0, 1)
            hmap = dome * rim * min(amp * 0.35, 4.0 * ctx.ss) * ctx.missing_residue  # render.missing_residue (1 = as before)
            spec_power, spec_gain = 40.0, 0.5
        hmap = hmap * (1.0 + lumps[y0:y1, x0:x1])
        shaded = shade_solder(hmap, solder, env, spec_power, spec_gain)
        alpha = np.clip(hmap / (0.08 * amp), 0.0, 1.0)
        blend(region, shaded, alpha)
        img[y0:y1, x0:x1] = region


def draw_tht_pads(ctx: RenderContext, img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], finish: np.ndarray,
                  hole_ratio: float = 0.45) -> None:
    """Round plated through-hole pads (annular rings) with a dark hole, drawn over the rectangular pads
    :func:`pads_and_solder` painted (bake for ``tht_pin``: the solder meniscus and the pins are Blender geometry).
    No random draw."""
    h, w = img.shape[:2]
    mask_edge = (base * 0.65).tolist()
    hole = (18.0, 18.0, 20.0)
    for pad in pads:
        x0, y0 = max(0, pad["x"]), max(0, pad["y"])
        x1, y1 = min(w, pad["x"] + pad["w"]), min(h, pad["y"] + pad["h"])
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        # back to bare mask (the rectangle plus its dark rim), then ring, finish and hole
        cv2.rectangle(img, (x0 - ctx.ss, y0 - ctx.ss), (x1 + ctx.ss - 1, y1 + ctx.ss - 1), base.tolist(), -1)
        cx, cy = pad["x"] + pad["w"] / 2.0, pad["y"] + pad["h"] / 2.0
        r = min(pad["w"], pad["h"]) / 2.0
        centre = (int(round(cx)), int(round(cy)))
        cv2.circle(img, centre, int(round(r)) + ctx.ss, mask_edge, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, centre, int(round(r)), finish.tolist(), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, centre, max(1, int(round(r * hole_ratio))), hole, -1, lineType=cv2.LINE_AA)
