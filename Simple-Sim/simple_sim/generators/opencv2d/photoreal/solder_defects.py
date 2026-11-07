"""Solder defects of the inspected part: where they are (shared with the 3D bake) and how they look.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from simple_sim.board.placement import pad_ref
from simple_sim.classes import DefectClass

from .context import RenderContext
from .imageops import blend, grid, rect_alpha, rect_window
from .joints import shade_blob, thermal_pad_index
from .palette import ARRAY_FAMILIES, CHIPS, LEADED


def solder_defect_layout(defect_type: str, pads: list[dict[str, int]], body: np.ndarray | None,
                         rng: np.random.Generator, *, ss: int) -> dict[str, Any]:
    """Where the inspected part's solder defects are; shared by the 2D renderer and the 3D bake.

    ``excess``: indices of pads with too much (bulging) solder. ``balls``: ``(x, y, r)`` solder balls on
    the mask next to the joints, never on a pad or under the body (*body*: its 4 corner points).
    *ss*: supersampling factor of the pixel coordinates (minimum ball size).
    """
    out: dict[str, Any] = {"excess": [], "balls": []}
    thermal = thermal_pad_index(pads)
    joints = [i for i in range(len(pads)) if i != thermal]
    if defect_type == DefectClass.EXCESS_SOLDER:
        k = int(rng.integers(1, min(len(joints), 4) + 1))
        out["excess"] = sorted(int(i) for i in rng.choice(joints, size=k, replace=False))
    elif defect_type == DefectClass.SOLDER_BALL:
        ref = pad_ref([pads[i] for i in joints])
        for _ in range(int(rng.integers(1, 5))):
            r = max(float(rng.uniform(0.12, 0.25)) * ref, float(rng.uniform(2.5, 4.5)) * ss)  # >= ~5-9 camera px across
            for _try in range(30):
                p = pads[joints[int(rng.integers(len(joints)))]]
                a, d = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0.6, 1.3)) * max(p["w"], p["h"])
                x, y = p["x"] + p["w"] / 2.0 + d * np.cos(a), p["y"] + p["h"] / 2.0 + d * np.sin(a)
                on_pad = any(q["x"] - r < x < q["x"] + q["w"] + r and q["y"] - r < y < q["y"] + q["h"] + r for q in pads)
                under = body is not None and cv2.pointPolygonTest(body.astype(np.float32), (x, y), True) > -r
                if not on_pad and not under:
                    out["balls"].append((float(x), float(y), r))
                    break
    return out


def draw_solder_balls(ctx: RenderContext, img: np.ndarray, base: np.ndarray, balls: list[tuple[float, float, float]]) -> None:
    """Small mirror-like solder spheres with a contact shadow."""
    h, w = img.shape[:2]
    for x, y, r in balls:
        x0, y0, x1, y1 = int(x - 2 * r), int(y - 2 * r), int(x + 2 * r) + 1, int(y + 2 * r) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        gy, gx = grid(h, w)
        yy, xx = gy[cy0:cy1, cx0:cx1], gx[cy0:cy1, cx0:cx1]
        d2 = ((xx - x) ** 2 + (yy - y) ** 2) / (r * r)
        region = img[cy0:cy1, cx0:cx1]
        shadow = np.clip(1.0 - (((xx - x - 0.4 * r) ** 2 + (yy - y - 0.4 * r) ** 2) / (r * r) - 0.6) / 0.8, 0, 1)
        region *= (1.0 - 0.4 * shadow)[..., None]
        hmap = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0)) * r
        blend(region, shade_blob(ctx, hmap, base, 60.0, 0.9), np.clip((1.0 - d2) * r, 0.0, 1.0))


def draw_solder_bulge(ctx: RenderContext, img: np.ndarray, base: np.ndarray, center, length: float, width: float,
                      angle_deg: float) -> None:
    """Shiny convex solder blob (too much solder), e.g. over a chip's end cap or a buried lead foot."""
    win = rect_window(img.shape[:2], center, length, width, angle_deg)
    if win is None:
        return
    y0, y1, x0, x1 = win
    u, v, m = rect_alpha(y1 - y0, x1 - x0, center, length, width, angle_deg, y0, x0)
    bulge = np.clip(1.0 - (u / (length / 2.0)) ** 2 - (v / (width / 2.0)) ** 2, 0.0, 1.0)
    hmap = np.sqrt(bulge) * 0.4 * min(length, width)
    blend(img[y0:y1, x0:x1], shade_blob(ctx, hmap, base, 60.0, 0.85), np.clip(bulge * 4.0, 0, 1) * m)


def draw_excess(ctx: RenderContext, img: np.ndarray, base: np.ndarray, family: str, pads: list[dict[str, int]], excess: list[int],
                 center: tuple[float, float], length: float, width: float, angle_deg: float) -> None:
    """Too much solder on the pads in *excess*: chips get it over the end cap, leaded parts over the lead foot."""
    a = np.radians(angle_deg)
    axis = np.array([np.cos(a), np.sin(a)])
    for i in excess:
        p = pads[i]
        pc = np.array([p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0])
        if family in CHIPS:
            side = 1.0 if (pc - np.array(center)) @ axis >= 0 else -1.0
            draw_solder_bulge(ctx, img, base, np.array(center) + axis * side * length * 0.42, length * 0.3, width * 1.06, angle_deg)
        elif family in LEADED or family in ARRAY_FAMILIES:
            draw_solder_bulge(ctx, img, base, pc, p["w"] * 0.95, p["h"] * 0.95, 0.0)
        elif family == "qfn":  # bulges out beyond the package edge
            out = pc - np.array(center)
            out = np.where(np.abs(out) >= np.abs(out).max(), np.sign(out), 0.0)
            draw_solder_bulge(ctx, img, base, pc + out * 0.35 * np.array([p["w"], p["h"]]), p["w"] * 1.1, p["h"] * 1.1, 0.0)
