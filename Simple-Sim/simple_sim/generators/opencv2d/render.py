"""High-level ROI rendering pipeline for OpenCV 2D backend."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .draw import _compute_pad_positions, body_geometry, draw_component, draw_pads, draw_solder, draw_substrate
from .filters import apply_post_filters
from .photoreal import render_photoreal_base, solder_defect_layout


def _draw_flat_solder_defects(img, nominal, defect_params, tolerances, footprint, rng) -> None:
    """SOLDER_BALL / EXCESS_SOLDER for the legacy flat style (same layout as photoreal)."""
    if defect_params["type"] not in ("SOLDER_BALL", "EXCESS_SOLDER"):
        return
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
    geo = body_geometry(nominal, defect_params, tolerances, img.shape[:2])
    rect = cv2.boxPoints(((float(geo["center"][0]), float(geo["center"][1])), (geo["length"], geo["width"]), geo["rotation_deg"]))
    layout = solder_defect_layout(defect_params["type"], pads, rect, rng)
    for i in layout["excess"]:
        p = pads[i]
        cv2.ellipse(img, (p["x"] + p["w"] // 2, p["y"] + p["h"] // 2), (int(p["w"] * 0.65), int(p["h"] * 0.65)),
                    0, 0, 360, (235, 235, 235), -1, lineType=cv2.LINE_AA)
    for x, y, r in layout["balls"]:
        cv2.circle(img, (int(x), int(y)), max(2, int(r)), (220, 220, 220), -1, lineType=cv2.LINE_AA)


def render_roi(
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    augment: dict[str, float],
    roi_size: tuple[int, int],
    config: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str = "chip_2pad",
) -> np.ndarray:
    """Render complete ROI with defects and augmentation."""
    width, height = roi_size
    if str(config.get("style", "photoreal")) == "photoreal":
        img = render_photoreal_base(nominal, defect_params, config, tolerances, rng, footprint, roi_size)
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        draw_substrate(img, config)
        draw_pads(img, nominal, tuple(config["copper_color"]), footprint=footprint)
        draw_solder(img, nominal, footprint=footprint)
        draw_component(
            img, nominal, defect_params, tuple(config["component_color"]), tolerances=tolerances, footprint=footprint
        )
        _draw_flat_solder_defects(img, nominal, defect_params, tolerances, footprint, rng)

    img = apply_post_filters(img, augment, rng)

    if abs(augment.get("rotation_deg", 0.0)) > 0.1:
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, augment["rotation_deg"], 1.0)
        img = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return img
