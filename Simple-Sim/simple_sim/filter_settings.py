"""Shared image-filter defaults and normalization for GUI + generator."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from simple_sim.coerce import as_bool, as_float, as_int

log = logging.getLogger(__name__)

_FLOAT_RANGES: dict[str, tuple[float, float, float]] = {
    "rotation_strength": (1.0, 0.0, 4.0),
    "blur_strength": (1.0, 0.0, 4.0),
    "grain_strength": (1.0, 0.0, 4.0),
    "brightness_strength": (1.0, 0.0, 4.0),
    "contrast_strength": (1.0, 0.0, 4.0),
    "perspective_strength": (1.0, 0.0, 2.0),
    "perspective_angle_x": (0.0, -15.0, 15.0),
    "perspective_angle_y": (0.0, -15.0, 15.0),
    "motion_blur_strength": (1.0, 0.0, 3.0),
    "motion_blur_angle": (0.0, 0.0, 360.0),
    "saturation_factor": (1.0, 0.5, 1.5),
    "hue_shift_deg": (0.0, -30.0, 30.0),
    "shadow_strength": (0.3, 0.0, 0.8),
    "shadow_size": (0.2, 0.1, 0.5),
    "reflection_strength": (0.5, 0.0, 1.0),
    "reflection_size": (0.15, 0.05, 0.3),
    "vignetting_strength": (1.0, 0.0, 2.0),
    "chromatic_strength": (1.0, 0.0, 2.0),
    "distortion_k1": (0.0, -0.3, 0.3),
    "distortion_k2": (0.0, -0.1, 0.1),
    "dust_density": (0.3, 0.0, 1.0),
    "dust_size": (2.0, 1.0, 5.0),
    "sharpen_strength": (1.0, 0.0, 2.0),
}

_INT_RANGES: dict[str, tuple[int, int, int]] = {
    "jpeg_quality": (85, 50, 95),
    "color_temperature_kelvin": (5500, 2500, 7500),
}

_BOOL_DEFAULTS: dict[str, bool] = {
    "enable": True,
    "cardinal_rotation_90": True,
    "enable_rotation": True,
    "enable_blur": True,
    "enable_grain": True,
    "enable_brightness": True,
    "enable_contrast": True,
    "enable_perspective": False,
    "enable_motion_blur": False,
    "enable_saturation": True,
    "enable_hue_shift": True,
    "enable_shadow": False,
    "enable_reflection": False,
    "enable_vignetting": False,
    "enable_chromatic_aberration": False,
    "enable_jpeg_compression": False,
    "enable_color_temperature": False,
    "enable_lens_distortion": False,
    "enable_dust": False,
    "enable_sharpen": False,
    "realism_enabled": False,
    "realism_constraints_enabled": True,
}

_STR_DEFAULTS: dict[str, str] = {
    "filter_mode": "custom",
    "realism_profile_id": "profile_industrial_cam",
}

_REALISM_FLOAT_RANGES: dict[str, tuple[float, float, float]] = {
    "realism_k_prob_0": (0.60, 0.0, 1.0),
    "realism_k_prob_1": (0.35, 0.0, 1.0),
    "realism_k_prob_2": (0.05, 0.0, 1.0),
    "realism_group_G1_prob": (0.35, 0.0, 1.0),
    "realism_group_G2_prob": (0.15, 0.0, 1.0),
    "realism_group_G3_prob": (0.20, 0.0, 1.0),
    "realism_group_G4_prob": (0.25, 0.0, 1.0),
    "realism_group_G5_prob": (0.12, 0.0, 1.0),
    "realism_group_G6_prob": (0.18, 0.0, 1.0),
    "realism_group_G7_prob": (0.10, 0.0, 1.0),
    "realism_group_G8_prob": (0.08, 0.0, 1.0),
}

# Secondary parameters (angles/sizes/k2) are fixed values, not randomizable strengths.
_NON_RANDOMIZABLE_FLOATS = {
    "perspective_angle_x",
    "perspective_angle_y",
    "motion_blur_angle",
    "shadow_size",
    "reflection_size",
    "distortion_k2",
    "dust_size",
}

_RANDOMIZABLE_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    key: (lo, hi) for key, (_default, lo, hi) in _FLOAT_RANGES.items() if key not in _NON_RANDOMIZABLE_FLOATS
}

_RANDOMIZABLE_INT_RANGES: dict[str, tuple[int, int]] = {
    key: (lo, hi) for key, (_default, lo, hi) in _INT_RANGES.items()
}

# Augment values that leave the image untouched (no-op for every apply_* filter).
NEUTRAL_AUGMENT: dict[str, float] = {
    "blur_sigma": 0.0,
    "noise_stddev": 0.0,
    "brightness_factor": 1.0,
    "contrast_factor": 1.0,
    "rotation_deg": 0.0,
    "perspective_strength": 0.0,
    "perspective_angle_x": 0.0,
    "perspective_angle_y": 0.0,
    "motion_blur_strength": 0.0,
    "motion_blur_angle": 0.0,
    "saturation_factor": 1.0,
    "hue_shift_deg": 0.0,
    "shadow_strength": 0.0,
    "shadow_size": 0.2,
    "reflection_strength": 0.0,
    "reflection_size": 0.15,
    "vignetting_strength": 0.0,
    "chromatic_strength": 0.0,
    "jpeg_quality": 100,
    "color_temperature_kelvin": 5500,
    "distortion_k1": 0.0,
    "distortion_k2": 0.0,
    "dust_density": 0.0,
    "dust_size": 2.0,
    "sharpen_strength": 0.0,
}


def normalize_image_filters(image_filters: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional IMAGE_FILTERS payload with safe defaults."""
    src = image_filters or {}
    out: dict[str, Any] = {}

    for key, default in _BOOL_DEFAULTS.items():
        out[key] = as_bool(src.get(key, default), default)

    for key, default in _STR_DEFAULTS.items():
        out[key] = str(src.get(key, default) or default)

    for key, (default, lo, hi) in _FLOAT_RANGES.items():
        out[key] = as_float(src.get(key, default), default, lo, hi)

    for key, (default, lo, hi) in _REALISM_FLOAT_RANGES.items():
        out[key] = as_float(src.get(key, default), default, lo, hi)

    for key, (default, lo, hi) in _INT_RANGES.items():
        out[key] = as_int(src.get(key, default), default, lo, hi)

    for key, (lo, hi) in _RANDOMIZABLE_FLOAT_RANGES.items():
        base = float(out[key])
        out[f"{key}_randomize"] = as_bool(src.get(f"{key}_randomize", False), False)
        out[f"{key}_min"] = as_float(src.get(f"{key}_min", base), base, lo, hi)
        out[f"{key}_max"] = as_float(src.get(f"{key}_max", base), base, lo, hi)

    for key, (lo, hi) in _RANDOMIZABLE_INT_RANGES.items():
        base = int(out[key])
        out[f"{key}_randomize"] = as_bool(src.get(f"{key}_randomize", False), False)
        out[f"{key}_min"] = as_int(src.get(f"{key}_min", base), base, lo, hi)
        out[f"{key}_max"] = as_int(src.get(f"{key}_max", base), base, lo, hi)

    mode = str(out.get("filter_mode", "custom")).strip().lower()
    if mode not in {"custom", "realism"}:
        out["filter_mode"] = "custom"
    else:
        out["filter_mode"] = mode

    k0 = float(out.get("realism_k_prob_0", 0.60))
    k1 = float(out.get("realism_k_prob_1", 0.35))
    k2 = float(out.get("realism_k_prob_2", 0.05))
    s = k0 + k1 + k2
    if s > 1e-9:
        out["realism_k_prob_0"] = k0 / s
        out["realism_k_prob_1"] = k1 / s
        out["realism_k_prob_2"] = k2 / s
    else:
        out["realism_k_prob_0"] = 0.60
        out["realism_k_prob_1"] = 0.35
        out["realism_k_prob_2"] = 0.05

    return out


def default_filter_values_for_popup() -> dict[str, Any]:
    """Default values encoded as popup-friendly strings/bools."""
    normalized = normalize_image_filters(None)
    out: dict[str, Any] = {}
    for key, value in normalized.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int):
            out[key] = str(value)
        elif isinstance(value, float):
            out[key] = f"{value:.2f}"
        else:
            out[key] = value
    return out


def clean_filter_values_for_popup() -> dict[str, Any]:
    """Neutral/no-op preset for clean image generation."""
    clean = default_filter_values_for_popup()
    for key in list(clean.keys()):
        if key.startswith("enable_") or key == "cardinal_rotation_90":
            clean[key] = False

    # Multipliers (*_strength scaling a sampled value) are neutral at 1.0 for brightness/contrast
    # and 0.0 otherwise; every other randomizable key is an augment value with a NEUTRAL_AUGMENT entry.
    neutral_multipliers = {"brightness_strength": 1.0, "contrast_strength": 1.0}
    neutral_values: dict[str, str] = {}
    for key in list(_RANDOMIZABLE_FLOAT_RANGES) + list(_RANDOMIZABLE_INT_RANGES):
        value = NEUTRAL_AUGMENT.get(key, neutral_multipliers.get(key, 0.0))
        neutral_values[key] = str(value) if isinstance(value, int) else f"{value:.2f}"
    clean.update(neutral_values)

    for key, value in neutral_values.items():
        clean[f"{key}_randomize"] = False
        clean[f"{key}_min"] = value
        clean[f"{key}_max"] = value

    return clean


def load_image_filters_env(var_name: str = "IMAGE_FILTERS") -> dict[str, Any]:
    """Read image-filter settings as JSON from an environment variable (GUI -> generator handoff)."""
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return normalize_image_filters(None)
    try:
        parsed = json.loads(raw)
    except ValueError:
        log.warning(f"Invalid {var_name} JSON, using defaults.")
        return normalize_image_filters(None)
    return normalize_image_filters(parsed if isinstance(parsed, dict) else None)


def augment_for_meta(augment: dict[str, Any]) -> dict[str, float]:
    """Return the schema-compatible augment subset stored in MetaRow."""
    return {
        "blur_sigma": float(augment.get("blur_sigma", 0.0) or 0.0),
        "noise_stddev": float(augment.get("noise_stddev", 0.0) or 0.0),
        "brightness_factor": float(augment.get("brightness_factor", 1.0) or 1.0),
        "contrast_factor": float(augment.get("contrast_factor", 1.0) or 1.0),
        "rotation_deg": float(augment.get("rotation_deg", 0.0) or 0.0),
    }
