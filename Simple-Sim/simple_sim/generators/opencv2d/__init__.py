"""OpenCV 2D renderer split into focused modules."""

from .augment import apply_image_filter_overrides, clean_augment, sample_augment_params, sample_rotation_deg
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
    apply_post_filters,
    apply_reflection,
    apply_saturation,
    apply_shadow,
    apply_sharpen,
    apply_vignetting,
)
from .geometry import sample_nominal_geometry
from .render import render_roi
from .spec import RenderSpec, render_record, render_record_with_meta, render_sample
from simple_sim.filter_settings import normalize_image_filters

__all__ = [
    "apply_image_filter_overrides",
    "sample_augment_params",
    "sample_rotation_deg",
    "clean_augment",
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
    "apply_post_filters",
    "render_roi",
    "RenderSpec",
    "render_sample",
    "render_record",
    "render_record_with_meta",
    "normalize_image_filters",
]
