"""Augmentation sampling and filter-override logic for OpenCV 2D renderer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from simple_sim.generators.filter_settings import normalize_image_filters
from .realism import apply_realism_groups


def apply_image_filter_overrides(augment: Dict[str, float], image_filters: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Apply toggle + strength overrides to sampled augment params."""
    aug = dict(augment or {})
    src_keys = set(image_filters.keys()) if isinstance(image_filters, dict) else set()
    filt_base = normalize_image_filters(image_filters)
    if not filt_base.get("enable", True):
        aug["blur_sigma"] = 0.0
        aug["noise_stddev"] = 0.0
        aug["brightness_factor"] = 1.0
        aug["contrast_factor"] = 1.0
        aug["rotation_deg"] = 0.0
        aug["perspective_strength"] = 0.0
        aug["perspective_angle_x"] = 0.0
        aug["perspective_angle_y"] = 0.0
        aug["motion_blur_strength"] = 0.0
        aug["motion_blur_angle"] = 0.0
        aug["saturation_factor"] = 1.0
        aug["hue_shift_deg"] = 0.0
        aug["shadow_strength"] = 0.0
        aug["shadow_size"] = 0.2
        aug["reflection_strength"] = 0.0
        aug["reflection_size"] = 0.15
        aug["vignetting_strength"] = 0.0
        aug["chromatic_strength"] = 0.0
        aug["jpeg_quality"] = 100
        aug["color_temperature_kelvin"] = 5500
        aug["distortion_k1"] = 0.0
        aug["distortion_k2"] = 0.0
        aug["dust_density"] = 0.0
        aug["dust_size"] = 2.0
        aug["sharpen_strength"] = 0.0
        return aug

    filt = dict(filt_base)
    filt["__source_keys"] = src_keys
    if str(filt.get("filter_mode", "custom")).strip().lower() == "realism" and bool(filt.get("realism_enabled", False)):
        return apply_realism_groups(aug, filt)

    def _rand_float(base_key: str, default_value: float) -> float:
        base = float(filt.get(base_key, default_value))
        if not bool(filt.get(f"{base_key}_randomize", False)):
            return base
        lo = float(filt.get(f"{base_key}_min", base))
        hi = float(filt.get(f"{base_key}_max", base))
        return float(np.random.uniform(min(lo, hi), max(lo, hi)))

    def _rand_int(base_key: str, default_value: int) -> int:
        base = int(filt.get(base_key, default_value))
        if not bool(filt.get(f"{base_key}_randomize", False)):
            return base
        lo = int(filt.get(f"{base_key}_min", base))
        hi = int(filt.get(f"{base_key}_max", base))
        return int(np.random.randint(min(lo, hi), max(lo, hi) + 1))

    rotation_strength = _rand_float("rotation_strength", 1.0)
    blur_strength = _rand_float("blur_strength", 1.0)
    grain_strength = _rand_float("grain_strength", 1.0)
    brightness_strength = _rand_float("brightness_strength", 1.0)
    contrast_strength = _rand_float("contrast_strength", 1.0)
    perspective_strength = _rand_float("perspective_strength", 1.0)
    motion_blur_strength = _rand_float("motion_blur_strength", 1.0)
    saturation_factor = _rand_float("saturation_factor", 1.0)
    hue_shift_deg = _rand_float("hue_shift_deg", 0.0)
    shadow_strength = _rand_float("shadow_strength", 0.3)
    reflection_strength = _rand_float("reflection_strength", 0.5)
    vignetting_strength = _rand_float("vignetting_strength", 1.0)
    chromatic_strength = _rand_float("chromatic_strength", 1.0)
    jpeg_quality = _rand_int("jpeg_quality", 85)
    color_temperature_kelvin = _rand_int("color_temperature_kelvin", 5500)
    distortion_k1 = _rand_float("distortion_k1", 0.0)
    dust_density = _rand_float("dust_density", 0.3)
    sharpen_strength = _rand_float("sharpen_strength", 1.0)

    aug["blur_sigma"] = 0.0 if not filt.get("enable_blur", True) else float(aug.get("blur_sigma", 0.0)) * blur_strength
    aug["noise_stddev"] = 0.0 if not filt.get("enable_grain", True) else float(aug.get("noise_stddev", 0.0)) * grain_strength

    if not filt.get("enable_brightness", True):
        aug["brightness_factor"] = 1.0
    else:
        bf = float(aug.get("brightness_factor", 1.0))
        aug["brightness_factor"] = 1.0 + ((bf - 1.0) * brightness_strength)

    if not filt.get("enable_contrast", True):
        aug["contrast_factor"] = 1.0
    else:
        cf = float(aug.get("contrast_factor", 1.0))
        aug["contrast_factor"] = 1.0 + ((cf - 1.0) * contrast_strength)

    if not filt.get("enable_rotation", True):
        if bool(filt.get("cardinal_rotation_90", True)):
            rot = float(aug.get("rotation_deg", 0.0))
            aug["rotation_deg"] = float(round(rot / 90.0) * 90.0)
        else:
            aug["rotation_deg"] = 0.0
    else:
        aug["rotation_deg"] = float(aug.get("rotation_deg", 0.0)) * rotation_strength

    if not filt.get("enable_perspective", True):
        aug["perspective_strength"] = 0.0
    else:
        aug["perspective_strength"] = perspective_strength
        aug["perspective_angle_x"] = float(aug.get("perspective_angle_x", 0.0))
        aug["perspective_angle_y"] = float(aug.get("perspective_angle_y", 0.0))

    if not filt.get("enable_motion_blur", True):
        aug["motion_blur_strength"] = 0.0
    else:
        aug["motion_blur_strength"] = motion_blur_strength
        aug["motion_blur_angle"] = float(aug.get("motion_blur_angle", 0.0))

    aug["chromatic_strength"] = 0.0 if not filt.get("enable_chromatic_aberration", True) else chromatic_strength
    aug["vignetting_strength"] = 0.0 if not filt.get("enable_vignetting", True) else vignetting_strength

    if not filt.get("enable_saturation", True):
        aug["saturation_factor"] = 1.0
    else:
        aug["saturation_factor"] = float(aug.get("saturation_factor", 1.0)) * saturation_factor

    aug["hue_shift_deg"] = 0.0 if not filt.get("enable_hue_shift", True) else float(aug.get("hue_shift_deg", 0.0)) + hue_shift_deg
    aug["sharpen_strength"] = 0.0 if not filt.get("enable_sharpen", True) else sharpen_strength

    if not filt.get("enable_lens_distortion", True):
        aug["distortion_k1"] = 0.0
        aug["distortion_k2"] = 0.0
    else:
        aug["distortion_k1"] = distortion_k1
        aug["distortion_k2"] = float(filt.get("distortion_k2", 0.0))

    aug["jpeg_quality"] = 100 if not filt.get("enable_jpeg_compression", True) else jpeg_quality

    if not filt.get("enable_shadow", True):
        aug["shadow_strength"] = 0.0
    else:
        aug["shadow_strength"] = shadow_strength
        aug["shadow_size"] = float(filt.get("shadow_size", 0.2))

    if not filt.get("enable_reflection", True):
        aug["reflection_strength"] = 0.0
    else:
        aug["reflection_strength"] = reflection_strength
        aug["reflection_size"] = float(filt.get("reflection_size", 0.15))

    if not filt.get("enable_dust", True):
        aug["dust_density"] = 0.0
    else:
        aug["dust_density"] = dust_density
        aug["dust_size"] = float(filt.get("dust_size", 2.0))

    aug["color_temperature_kelvin"] = 5500 if not filt.get("enable_color_temperature", True) else color_temperature_kelvin
    return aug


def sample_augment_params(
    augment_config: Dict[str, Any],
    domain_config: Dict[str, Any],
    rng: np.random.Generator,
    *,
    enable_cardinal_rotation_90: bool = True,
) -> Dict[str, float]:
    """Sample augmentation parameters."""
    blur_min, blur_max = domain_config["blur_sigma"]
    noise_min, noise_max = domain_config["noise_stddev"]
    bright_min, bright_max = domain_config["lighting_brightness"]
    contrast_min, contrast_max = augment_config["contrast_factor_range"]
    rot_min, rot_max = augment_config["rotation_deg_range"]

    base_orientation_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0])) if enable_cardinal_rotation_90 else 0.0
    rotation_jitter_deg = float(rng.uniform(rot_min, rot_max))

    return {
        "blur_sigma": float(rng.uniform(blur_min, blur_max)),
        "noise_stddev": float(rng.uniform(noise_min, noise_max)),
        "brightness_factor": float(rng.uniform(bright_min, bright_max)),
        "contrast_factor": float(rng.uniform(contrast_min, contrast_max)),
        "rotation_deg": float(base_orientation_deg + rotation_jitter_deg),
        "perspective_angle_x": float(rng.uniform(-10.0, 10.0)),
        "perspective_angle_y": float(rng.uniform(-10.0, 10.0)),
        "motion_blur_angle": float(rng.uniform(0.0, 360.0)),
        "saturation_factor": float(rng.uniform(0.8, 1.2)),
        "hue_shift_deg": float(rng.uniform(-15.0, 15.0)),
        "color_temperature_kelvin": int(rng.choice([3000, 4000, 5500, 6500])),
    }
