"""Shared helpers for filter popup profile persistence and extra-key handling."""

from __future__ import annotations

import json
from typing import Any, Dict


def is_filter_popup_extra_key(key: str) -> bool:
    return (
        key.endswith("_randomize")
        or key.endswith("_min")
        or key.endswith("_max")
        or key == "filter_mode"
        or key.startswith("realism_")
    )


def parse_bool_like(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def extract_filter_popup_extras(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(k, str) and is_filter_popup_extra_key(k):
            out[k] = v
    return out


def load_filter_popup_extras_from_profiles_json(raw_profiles: str, raw_active: str = "Default") -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
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

