"""Drawing primitives for OpenCV 2D renderer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from simple_sim.board.footprints import DEFAULT_FOOTPRINT
from simple_sim.classes import DefectClass

from .geometry import body_geometry, draw_solder_bridge, lifted_corner_geometry, pad_positions, param_selector


def draw_substrate(img: np.ndarray, config: dict[str, Any]) -> None:
    """Draw green PCB substrate background."""
    img[:] = config["substrate_color"]


def draw_pads(img: np.ndarray, nominal: dict[str, float], color: tuple[int, int, int], footprint: str = DEFAULT_FOOTPRINT) -> None:
    pads = pad_positions(footprint, nominal, img.shape[:2])
    for pad in pads:
        cv2.rectangle(img, (pad["x"], pad["y"]), (pad["x"] + pad["w"], pad["y"] + pad["h"]), color, -1)


def draw_solder(img: np.ndarray, nominal: dict[str, float], footprint: str = DEFAULT_FOOTPRINT) -> None:
    solder_color = (200, 200, 200)
    pads = pad_positions(footprint, nominal, img.shape[:2])
    h, w = img.shape[:2]
    for pad in pads:
        x, y, pw, ph = pad["x"], pad["y"], pad["w"], pad["h"]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, int(x + pw))
        y1 = min(h, int(y + ph))
        if x1 <= x0 or y1 <= y0:
            continue

        solder_region = img[y0:y1, x0:x1].copy()
        overlay = np.full_like(solder_region, solder_color)
        blended = cv2.addWeighted(solder_region, 0.7, overlay, 0.3, 0)
        if blended is None:
            continue
        img[y0:y1, x0:x1] = blended


def _darken_poly(img: np.ndarray, pts: np.ndarray, factor: float) -> None:
    """Multiply pixels inside polygon *pts* by *factor* (soft contact shadow)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * (1.0 - factor)
    img[:] = np.clip(img.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def _draw_corner_lift(
    img: np.ndarray,
    box: np.ndarray,
    center: tuple[int, int],
    comp_width: int,
    color: tuple[int, int, int],
    tilt_deg: float,
    selector: int,
) -> None:
    """Draw the body with one corner raised: shadow under it, skewed quad, lit corner."""
    geo = lifted_corner_geometry(box, center, comp_width, tilt_deg, selector)
    _darken_poly(img, geo["shadow"], 0.5)

    body = np.round(geo["lifted"]).astype(np.int32)
    cv2.fillPoly(img, [body], color)

    # Highlight on the raised (tilted towards the light) corner facet.
    facet = np.round(geo["facet"]).astype(np.int32)
    bright = tuple(int(min(255, c * 1.45 + 20)) for c in color)
    overlay = img.copy()
    cv2.fillPoly(overlay, [facet], bright)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, dst=img)
    cv2.line(img, tuple(facet[1]), tuple(facet[2]), tuple(int(c * 0.6) for c in color), 1, lineType=cv2.LINE_AA)

    edge = tuple(int(min(255, max(0, c * 1.2))) for c in color)
    cv2.polylines(img, [body], True, edge, 1)


def draw_component(
    img: np.ndarray,
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    color: tuple[int, int, int],
    tolerances: dict[str, Any] | None = None,
    footprint: str = DEFAULT_FOOTPRINT,
) -> None:
    defect_type = defect_params["type"]
    if defect_type == DefectClass.MISSING:
        return

    geo = body_geometry(nominal, defect_params, tolerances, img.shape[:2])
    center = geo["center"]
    if geo["is_tombstone"]:
        comp_width = int(geo["width"] * 0.2)
        comp_height = int(geo["length"] * 1.5)
    else:
        comp_width, comp_height = geo["width"], geo["length"]

    rect_size = (comp_height, comp_width)
    box = cv2.boxPoints((center, rect_size, geo["rotation_deg"]))

    if defect_type == DefectClass.CORNER_LIFT:
        _draw_corner_lift(img, box, center, comp_width, color, geo["tilt_deg"], param_selector(defect_params))
        return

    box = box.astype(np.int32)

    cv2.fillPoly(img, [box], color)
    edge = tuple(int(min(255, max(0, c * 1.2))) for c in color)
    cv2.polylines(img, [box], True, edge, 1)

    if defect_type == DefectClass.SOLDER_BRIDGE:
        # Drawn on top of the body so a bridge routed along the body edge stays visible.
        draw_solder_bridge(
            img, nominal, footprint, center, rect_size, param_selector(defect_params), geo["tilt_deg"],
        )
