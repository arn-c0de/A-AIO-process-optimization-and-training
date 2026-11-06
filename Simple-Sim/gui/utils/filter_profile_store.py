"""Shared helpers for filter popup profile persistence and extra-key handling."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from simple_sim.coerce import as_bool
from simple_sim.filter_profiles import is_filter_popup_extra_key


def extract_filter_popup_extras(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(k, str) and is_filter_popup_extra_key(k):
            out[k] = v
    return out


def load_filter_popup_extras_from_profiles_json(raw_profiles: str, raw_active: str = "Default") -> dict[str, Any]:
    extras: dict[str, Any] = {}
    raw_profiles = str(raw_profiles or "").strip()
    raw_active = str(raw_active or "").strip() or "Default"
    if not raw_profiles:
        return extras
    try:
        obj = json.loads(raw_profiles)
        if not isinstance(obj, dict):
            return extras
        active_profile = obj.get(raw_active)
        if not isinstance(active_profile, dict) and obj:
            first_key = sorted(obj.keys())[0]
            active_profile = obj.get(first_key)
        if isinstance(active_profile, dict):
            extras = extract_filter_popup_extras(active_profile)
    except Exception:
        return {}
    return extras



def float_or_default(value: Any, *, default: float) -> float:
    """``float(value)`` (whitespace-stripped), or *default* if it does not parse."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)


# Pipeline image-filter settings as stored in filter profiles / SettingsStore ("pipeline.filter.<key>").
# (key, kind, default) with kind in {"bool", "float", "int"}; the UI variable is ``var_filter_<key>``.
FILTER_SETTING_SPECS: tuple[tuple[str, str, Any], ...] = (
    # Existing filters
    ("cardinal_rotation_90", "bool", True),
    ("enable_rotation", "bool", True),
    ("enable_blur", "bool", True),
    ("enable_grain", "bool", True),
    ("enable_brightness", "bool", True),
    ("enable_contrast", "bool", True),
    ("rotation_strength", "float", 1.0),
    ("blur_strength", "float", 1.0),
    ("grain_strength", "float", 1.0),
    ("brightness_strength", "float", 1.0),
    ("contrast_strength", "float", 1.0),
    # High priority filters
    ("enable_perspective", "bool", False),
    ("perspective_strength", "float", 1.0),
    ("enable_motion_blur", "bool", False),
    ("motion_blur_strength", "float", 1.0),
    ("enable_saturation", "bool", True),
    ("saturation_factor", "float", 1.0),
    ("enable_hue_shift", "bool", True),
    ("hue_shift_deg", "float", 0.0),
    ("enable_shadow", "bool", False),
    ("shadow_strength", "float", 0.3),
    ("enable_reflection", "bool", False),
    ("reflection_strength", "float", 0.5),
    # Medium/low priority filters
    ("enable_vignetting", "bool", False),
    ("vignetting_strength", "float", 1.0),
    ("enable_chromatic_aberration", "bool", False),
    ("chromatic_strength", "float", 1.0),
    ("enable_jpeg_compression", "bool", False),
    ("jpeg_quality", "int", 85),
    ("enable_color_temperature", "bool", False),
    ("color_temperature_kelvin", "int", 5500),
    ("enable_lens_distortion", "bool", False),
    ("distortion_k1", "float", 0.0),
    ("enable_dust", "bool", False),
    ("dust_density", "float", 0.3),
    ("enable_sharpen", "bool", False),
    ("sharpen_strength", "float", 1.0),
)


def _default_text(kind: str, default: Any) -> str:
    return f"{default:.2f}" if kind == "float" else str(int(default))


def formatted_filter_settings(raw: Callable[[str], Any]) -> dict[str, Any]:
    """Settings dict from current UI values (``raw(key)``): bools as bool, numbers as display strings."""
    out: dict[str, Any] = {}
    for key, kind, default in FILTER_SETTING_SPECS:
        v = raw(key)
        if kind == "bool":
            out[key] = bool(v)
        elif kind == "float":
            out[key] = f"{float_or_default(v, default=default):.2f}"
        else:
            out[key] = f"{int(float_or_default(v, default=default))}"
    return out


def normalized_filter_settings(lookup: Callable[[str, Any], Any]) -> dict[str, Any]:
    """Settings dict from a stored mapping (``lookup(key, default)``), filling per-key defaults."""
    out: dict[str, Any] = {}
    for key, kind, default in FILTER_SETTING_SPECS:
        if kind == "bool":
            out[key] = bool(lookup(key, default))
        else:
            out[key] = str(lookup(key, _default_text(kind, default)))
    return out


# Keys sent to the generator via IMAGE_FILTERS (UI variable ``var_filter_<key>``).
FILTER_PAYLOAD_BOOL_KEYS: tuple[str, ...] = (
    "cardinal_rotation_90", "enable_rotation", "enable_blur", "enable_grain", "enable_brightness",
    "enable_contrast", "enable_perspective", "enable_motion_blur", "enable_saturation", "enable_hue_shift",
    "enable_shadow", "enable_reflection", "enable_vignetting", "enable_chromatic_aberration",
    "enable_jpeg_compression", "enable_color_temperature", "enable_lens_distortion", "enable_dust",
    "enable_sharpen",
)
FILTER_PAYLOAD_FLOAT_KEYS: tuple[tuple[str, float], ...] = (
    ("rotation_strength", 1.0), ("blur_strength", 1.0), ("grain_strength", 1.0),
    ("brightness_strength", 1.0), ("contrast_strength", 1.0), ("perspective_strength", 1.0),
    ("perspective_angle_x", 0.0), ("perspective_angle_y", 0.0), ("motion_blur_strength", 1.0),
    ("motion_blur_angle", 0.0), ("saturation_factor", 1.0), ("hue_shift_deg", 0.0),
    ("shadow_strength", 0.3), ("shadow_size", 0.2), ("reflection_strength", 0.5),
    ("reflection_size", 0.15), ("vignetting_strength", 1.0), ("chromatic_strength", 1.0),
    ("distortion_k1", 0.0), ("distortion_k2", 0.0), ("dust_density", 0.3), ("dust_size", 2.0),
    ("sharpen_strength", 1.0),
)
FILTER_PAYLOAD_INT_KEYS: tuple[tuple[str, int], ...] = (
    ("jpeg_quality", 85),
    ("color_temperature_kelvin", 5500),
)


def image_filters_payload(raw: Callable[[str], Any], extras: dict[str, Any]) -> dict[str, Any]:
    """Typed IMAGE_FILTERS payload from UI values (``raw(key)``) plus filter-popup *extras*."""
    bool_payload = {key: bool(raw(key)) for key in FILTER_PAYLOAD_BOOL_KEYS}
    payload: dict[str, Any] = {"enable": any(bool_payload.values())}
    payload.update(bool_payload)
    for key, default in FILTER_PAYLOAD_FLOAT_KEYS:
        payload[key] = float_or_default(raw(key), default=default)
    for key, default in FILTER_PAYLOAD_INT_KEYS:
        payload[key] = int(float_or_default(raw(key), default=float(default)))

    for k, v in extras.items():
        if not isinstance(k, str) or not is_filter_popup_extra_key(k):
            continue
        if k.endswith("_randomize"):
            payload[k] = as_bool(v, default=False)
        elif k == "filter_mode":
            payload[k] = str(v)
        elif k.startswith("realism_"):
            if k.endswith("_enabled"):
                payload[k] = as_bool(v, default=False)
            elif k.endswith("_profile_id"):
                payload[k] = str(v)
            else:
                payload[k] = float_or_default(str(v), default=0.0)
        else:
            payload[k] = float_or_default(str(v), default=0.0)

    # Realism mode must not be blocked by custom enable_* toggles.
    mode = str(payload.get("filter_mode", "custom") or "custom").strip().lower()
    if mode == "realism" and bool(payload.get("realism_enabled", False)):
        payload["enable"] = True
    return payload
