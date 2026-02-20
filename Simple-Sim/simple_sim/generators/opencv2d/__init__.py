"""OpenCV 2D renderer split into focused modules."""

from .augment import apply_image_filter_overrides, sample_augment_params
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
from .geometry import sample_nominal_geometry
from .render import render_roi
from simple_sim.generators.filter_settings import normalize_image_filters

__all__ = [
    "apply_image_filter_overrides",
    "sample_augment_params",
    "sample_nominal_geometry",
    "draw_substrate",
    "draw_pads",
    "draw_solder",
    "draw_component",
    "apply_blur",
    "apply_noise",
    "apply_brightness",
    "apply_contrast",
    "apply_perspective_transform",
    "apply_motion_blur",
    "apply_chromatic_aberration",
    "apply_vignetting",
    "apply_saturation",
    "apply_hue_shift",
    "apply_sharpen",
    "apply_lens_distortion",
    "apply_jpeg_compression",
    "apply_shadow",
    "apply_reflection",
    "apply_dust_particles",
    "apply_color_temperature",
    "render_roi",
    "normalize_image_filters",
]
