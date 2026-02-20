"""High-level ROI rendering pipeline for OpenCV 2D backend."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .draw import draw_component, draw_pads, draw_solder, draw_substrate
from .filters import (
    apply_blur,
    apply_brightness,
    apply_chromatic_aberration,
    apply_color_temperature,
    apply_contrast,
    apply_dust_particles,
    apply_hue_shift,
    apply_jpeg_compression,
    apply_lens_distortion,
    apply_motion_blur,
    apply_noise,
    apply_perspective_transform,
    apply_reflection,
    apply_saturation,
    apply_shadow,
    apply_sharpen,
    apply_vignetting,
)


def render_roi(
    nominal: Dict[str, float],
    defect_params: Dict[str, Any],
    augment: Dict[str, float],
    roi_size: Tuple[int, int],
    config: Dict[str, Any],
    tolerances: Optional[Dict[str, Any]],
    rng: np.random.Generator,
    footprint: str = "chip_2pad",
) -> np.ndarray:
    """Render complete ROI with defects and augmentation."""
    width, height = roi_size
    img = np.zeros((height, width, 3), dtype=np.uint8)

    draw_substrate(img, config)
    draw_pads(img, nominal, tuple(config["copper_color"]), footprint=footprint)
    draw_solder(img, nominal, footprint=footprint)
    draw_component(img, nominal, defect_params, tuple(config["component_color"]), tolerances=tolerances)

    img = apply_blur(img, augment.get("blur_sigma", 0.0))
    img = apply_noise(img, augment.get("noise_stddev", 0.0), rng)
    img = apply_brightness(img, augment.get("brightness_factor", 1.0))
    img = apply_contrast(img, augment.get("contrast_factor", 1.0))

    img = apply_saturation(img, augment.get("saturation_factor", 1.0))
    img = apply_hue_shift(img, augment.get("hue_shift_deg", 0.0))
    img = apply_color_temperature(img, augment.get("color_temperature_kelvin", 5500))

    img = apply_vignetting(img, augment.get("vignetting_strength", 0.0))
    img = apply_chromatic_aberration(img, augment.get("chromatic_strength", 0.0))
    img = apply_lens_distortion(img, augment.get("distortion_k1", 0.0), augment.get("distortion_k2", 0.0))

    img = apply_motion_blur(img, augment.get("motion_blur_strength", 0.0), augment.get("motion_blur_angle", 0.0))
    img = apply_sharpen(img, augment.get("sharpen_strength", 0.0))

    img = apply_shadow(img, augment.get("shadow_strength", 0.0), augment.get("shadow_size", 0.2), rng)
    img = apply_reflection(img, augment.get("reflection_strength", 0.0), augment.get("reflection_size", 0.15), rng)
    img = apply_dust_particles(img, augment.get("dust_density", 0.0), augment.get("dust_size", 2.0), rng)

    img = apply_jpeg_compression(img, augment.get("jpeg_quality", 100))

    img = apply_perspective_transform(
        img,
        augment.get("perspective_strength", 0.0),
        augment.get("perspective_angle_x", 0.0),
        augment.get("perspective_angle_y", 0.0),
    )

    if abs(augment.get("rotation_deg", 0.0)) > 0.1:
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, augment["rotation_deg"], 1.0)
        img = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return img
