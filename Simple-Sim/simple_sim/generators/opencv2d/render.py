"""High-level ROI rendering pipeline for OpenCV 2D backend."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .draw import draw_component, draw_pads, draw_solder, draw_substrate
from .filters import apply_post_filters
from .photoreal import render_photoreal_base


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

    img = apply_post_filters(img, augment, rng)

    if abs(augment.get("rotation_deg", 0.0)) > 0.1:
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, augment["rotation_deg"], 1.0)
        img = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return img
