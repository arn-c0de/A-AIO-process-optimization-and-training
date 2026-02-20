"""Drawing primitives for OpenCV 2D renderer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_substrate(img: np.ndarray, config: Dict[str, Any]) -> None:
    """Draw green PCB substrate background."""
    img[:] = config["substrate_color"]


def _compute_pad_positions(footprint: str, nominal: Dict[str, float], img_shape: Tuple[int, int]) -> List[Dict[str, int]]:
    h, w = img_shape
    center_x, center_y = w // 2, h // 2

    pad_width = int(nominal["pad_width"])
    pad_height = int(nominal["pad_height"])
    pad_spacing = int(nominal["pad_spacing"])

    if footprint == "sot23":
        pad_spacing_y = int(nominal["pad_spacing_y"])
        small_h = int(pad_height * 0.8)
        min_gap_px = 10
        effective_spacing_y = max(pad_spacing_y, small_h + min_gap_px)

        p1_x = center_x - pad_spacing // 2 - pad_width // 2
        p1_y = center_y - effective_spacing_y // 2 - small_h // 2
        p2_x = center_x - pad_spacing // 2 - pad_width // 2
        p2_y = center_y + effective_spacing_y // 2 - small_h // 2
        p3_x = center_x + pad_spacing // 2 - pad_width // 2
        p3_y = center_y - pad_height // 2
        return [
            {"x": p1_x, "y": p1_y, "w": pad_width, "h": small_h},
            {"x": p2_x, "y": p2_y, "w": pad_width, "h": small_h},
            {"x": p3_x, "y": p3_y, "w": pad_width, "h": pad_height},
        ]

    if footprint.startswith("qfn"):
        pad_spacing_y = int(nominal["pad_spacing_y"])
        left_x = center_x - pad_spacing // 2 - pad_width // 2
        left_y = center_y - pad_height // 2
        right_x = center_x + pad_spacing // 2 - pad_width // 2
        right_y = center_y - pad_height // 2
        top_x = center_x - pad_height // 2
        top_y = center_y - pad_spacing_y // 2 - pad_width // 2
        bot_x = center_x - pad_height // 2
        bot_y = center_y + pad_spacing_y // 2 - pad_width // 2
        return [
            {"x": left_x, "y": left_y, "w": pad_width, "h": pad_height},
            {"x": right_x, "y": right_y, "w": pad_width, "h": pad_height},
            {"x": top_x, "y": top_y, "w": pad_height, "h": pad_width},
            {"x": bot_x, "y": bot_y, "w": pad_height, "h": pad_width},
        ]

    if footprint == "soic_16":
        comp_w = int(nominal.get("component_width", 0))
        pad_spacing_y = int(nominal.get("pad_spacing_y", 0))
        y_span = max(pad_width * 7, pad_spacing_y if pad_spacing_y > 0 else int(comp_w * 0.72))
        pitch = y_span / 7.0
        y0 = center_y - (y_span / 2.0)

        left_x = center_x - pad_spacing // 2 - pad_height // 2
        right_x = center_x + pad_spacing // 2 - pad_height // 2

        pads: List[Dict[str, int]] = []
        for i in range(8):
            cy = int(round(y0 + (i * pitch)))
            y = cy - pad_width // 2
            pads.append({"x": left_x, "y": y, "w": pad_height, "h": pad_width})
            pads.append({"x": right_x, "y": y, "w": pad_height, "h": pad_width})
        return pads

    left_x = center_x - pad_spacing // 2 - pad_width // 2
    left_y = center_y - pad_height // 2
    right_x = center_x + pad_spacing // 2 - pad_width // 2
    right_y = center_y - pad_height // 2
    return [
        {"x": left_x, "y": left_y, "w": pad_width, "h": pad_height},
        {"x": right_x, "y": right_y, "w": pad_width, "h": pad_height},
    ]


def draw_pads(img: np.ndarray, nominal: Dict[str, float], color: Tuple[int, int, int], footprint: str = "chip_2pad") -> None:
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
    for pad in pads:
        cv2.rectangle(img, (pad["x"], pad["y"]), (pad["x"] + pad["w"], pad["y"] + pad["h"]), color, -1)


def draw_solder(img: np.ndarray, nominal: Dict[str, float], footprint: str = "chip_2pad") -> None:
    solder_color = (200, 200, 200)
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
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


def draw_component(
    img: np.ndarray,
    nominal: Dict[str, float],
    defect_params: Dict[str, Any],
    color: Tuple[int, int, int],
    tolerances: Optional[Dict[str, Any]] = None,
) -> None:
    if defect_params["type"] == "MISSING":
        return

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    comp_center_x = center_x + int(defect_params["shift_x"])
    comp_center_y = center_y + int(defect_params["shift_y"])

    tilt_deg = abs(defect_params["tilt_deg"])
    tombstone_thresh = float(tolerances["tombstone_tilt_deg"]) if tolerances is not None else 75.0

    if tilt_deg >= tombstone_thresh:
        comp_width = int(nominal["component_width"] * 0.2)
        comp_height = int(nominal["component_length"] * 1.5)
    else:
        comp_width = int(nominal["component_width"])
        comp_height = int(nominal["component_length"])

    rect_size = (comp_height, comp_width)
    box = cv2.boxPoints(((comp_center_x, comp_center_y), rect_size, defect_params["rotation_deg"]))
    box = box.astype(np.int32)

    cv2.fillPoly(img, [box], color)
    edge = tuple(int(min(255, max(0, c * 1.2))) for c in color)
    cv2.polylines(img, [box], True, edge, 1)
