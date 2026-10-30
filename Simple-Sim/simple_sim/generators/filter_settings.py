"""Shared image-filter defaults and normalization for GUI + generator."""

from __future__ import annotations

import json
import os
from typing import Any


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
    "enable_motion_blur": True,
    "enable_saturation": True,
    "enable_hue_shift": True,
    "enable_shadow": True,
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

_RANDOMIZABLE_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "rotation_strength": (0.0, 4.0),
    "blur_strength": (0.0, 4.0),
    "grain_strength": (0.0, 4.0),
    "brightness_strength": (0.0, 4.0),
    "contrast_strength": (0.0, 4.0),
    "perspective_strength": (0.0, 2.0),
    "motion_blur_strength": (0.0, 3.0),
    "saturation_factor": (0.5, 1.5),
    "hue_shift_deg": (-30.0, 30.0),
    "shadow_strength": (0.0, 0.8),
    "reflection_strength": (0.0, 1.0),
    "vignetting_strength": (0.0, 2.0),
    "chromatic_strength": (0.0, 2.0),
    "distortion_k1": (-0.3, 0.3),
    "dust_density": (0.0, 1.0),
    "sharpen_strength": (0.0, 2.0),
}

_RANDOMIZABLE_INT_RANGES: dict[str, tuple[int, int]] = {
    "jpeg_quality": (50, 95),
    "color_temperature_kelvin": (2500, 7500),
}



def _to_bool(src: dict[str, Any], name: str, default: bool) -> bool:
    v = src.get(name, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default



def _to_float(src: dict[str, Any], name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(src.get(name, default))
    except Exception:
        v = default
    return max(lo, min(hi, v))



def _to_int(src: dict[str, Any], name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(src.get(name, default))
    except Exception:
        v = default
    return max(lo, min(hi, v))



def normalize_image_filters(image_filters: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional IMAGE_FILTERS payload with safe defaults."""
    src = image_filters or {}
    out: dict[str, Any] = {}

    for key, default in _BOOL_DEFAULTS.items():
        out[key] = _to_bool(src, key, default)

    for key, default in _STR_DEFAULTS.items():
        out[key] = str(src.get(key, default) or default)

    for key, (default, lo, hi) in _FLOAT_RANGES.items():
        out[key] = _to_float(src, key, default, lo, hi)

    for key, (default, lo, hi) in _REALISM_FLOAT_RANGES.items():
        out[key] = _to_float(src, key, default, lo, hi)

    for key, (default, lo, hi) in _INT_RANGES.items():
        out[key] = _to_int(src, key, default, lo, hi)

    for key, (lo, hi) in _RANDOMIZABLE_FLOAT_RANGES.items():
        base = float(out[key])
        out[f"{key}_randomize"] = _to_bool(src, f"{key}_randomize", False)
        out[f"{key}_min"] = _to_float(src, f"{key}_min", base, lo, hi)
        out[f"{key}_max"] = _to_float(src, f"{key}_max", base, lo, hi)

    for key, (lo, hi) in _RANDOMIZABLE_INT_RANGES.items():
        base = int(out[key])
        out[f"{key}_randomize"] = _to_bool(src, f"{key}_randomize", False)
        out[f"{key}_min"] = _to_int(src, f"{key}_min", base, lo, hi)
        out[f"{key}_max"] = _to_int(src, f"{key}_max", base, lo, hi)

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

    neutral_values = {
        "rotation_strength": "0.00",
        "blur_strength": "0.00",
        "grain_strength": "0.00",
        "brightness_strength": "1.00",
        "contrast_strength": "1.00",
        "perspective_strength": "0.00",
        "motion_blur_strength": "0.00",
        "saturation_factor": "1.00",
        "hue_shift_deg": "0.00",
        "shadow_strength": "0.00",
        "reflection_strength": "0.00",
        "vignetting_strength": "0.00",
        "chromatic_strength": "0.00",
        "jpeg_quality": "100",
        "color_temperature_kelvin": "5500",
        "distortion_k1": "0.00",
        "dust_density": "0.00",
        "sharpen_strength": "0.00",
    }
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
    except Exception:
        print(f"[warn] Invalid {var_name} JSON, using defaults.")
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
