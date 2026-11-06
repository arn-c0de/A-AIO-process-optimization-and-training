"""Image-filter profiles persisted in the shared GUI ``settings.json`` (tk-free).

The GUI filter popup stores named filter profiles as a JSON string under
``pipeline.filter_profiles_json`` (+ ``pipeline.filter_profile_active``) and mirrors the
active profile as flat ``pipeline.filter.<key>`` entries. These helpers read/write that
format so CLI tools (e.g. ``scripts/render_debug_previews.py``) stay in sync with the GUI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simple_sim.filter_settings import default_filter_values_for_popup, normalize_image_filters
from simple_sim.io_utils import atomic_write_json, read_json

PROFILES_KEY = "pipeline.filter_profiles_json"
ACTIVE_KEY = "pipeline.filter_profile_active"
FLAT_PREFIX = "pipeline.filter."
DEFAULT_PROFILE = "Default"


def is_filter_popup_extra_key(key: str) -> bool:
    """Keys only the filter popup/profile store knows (randomize ranges, mode, realism knobs)."""
    return (
        key.endswith("_randomize")
        or key.endswith("_min")
        or key.endswith("_max")
        or key == "filter_mode"
        or key.startswith("realism_")
    )


def shared_settings_path(sim_root: Path, explicit_path: str | None = None) -> Path:
    """``explicit_path`` if given, else ``<sim_root>/outputs/gui/settings.json`` (resolved)."""
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return (Path(sim_root) / "outputs" / "gui" / "settings.json").resolve()


def encode_store_value(key: str, value: Any) -> Any:
    """Encode one normalized filter value in the popup/profile-store format (strings for numbers)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if key.startswith("realism_") else f"{value:.2f}"
    return value


def to_profile_store(d: dict[str, Any] | None, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return *d* merged over *existing*, encoded like the GUI filter popup stores it.

    Every key the shared normalizer knows is (re)encoded; all other keys are kept verbatim
    so filters a tool does not know about survive a load/save round trip.
    """
    merged: dict[str, Any] = {}
    for src in (existing, d):
        for k, v in (src or {}).items():
            if isinstance(k, str):
                merged[k] = v
    out = dict(merged)
    for k, v in normalize_image_filters(merged).items():
        out[k] = encode_store_value(k, v)
    return out


def flat_filter_setting_keys() -> list[str]:
    """Filter keys mirrored as flat ``pipeline.filter.<key>`` entries in settings.json."""
    return [k for k in default_filter_values_for_popup() if k != "enable" and not is_filter_popup_extra_key(k)]


def normalized_with_extras(values: dict[str, Any]) -> dict[str, Any]:
    """Normalized filter values plus any unknown keys from *values* (kept verbatim)."""
    out = normalize_image_filters(values)
    for k, v in values.items():
        if isinstance(k, str) and k not in out:
            out[k] = v
    return out


def load_shared_filter_profiles(settings_path: Path) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    """Return ``(profiles, active_name, current_normalized_filters)`` from settings.json.

    Without stored profiles, a ``Default`` profile is built from the legacy flat keys.
    """
    data = read_json(settings_path, {})
    if not isinstance(data, dict):
        data = {}
    raw_profiles = str(data.get(PROFILES_KEY, "") or "").strip()
    active = str(data.get(ACTIVE_KEY, "") or "").strip() or DEFAULT_PROFILE
    profiles: dict[str, dict[str, Any]] = {}
    if raw_profiles:
        try:
            obj = json.loads(raw_profiles)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            # Keep stored profiles verbatim (no key stripping).
            profiles = {k: dict(v) for k, v in obj.items() if isinstance(k, str) and isinstance(v, dict)}

    if not profiles:
        legacy = {k: data[FLAT_PREFIX + k] for k in flat_filter_setting_keys() if FLAT_PREFIX + k in data}
        profiles = {DEFAULT_PROFILE: to_profile_store(legacy)}
        active = DEFAULT_PROFILE
    if active not in profiles:
        active = sorted(profiles.keys(), key=lambda s: s.lower())[0]
    return profiles, active, normalized_with_extras(profiles.get(active, {}))


def save_shared_filter_profiles(
    settings_path: Path, profiles: dict[str, dict[str, Any]], active: str, current: dict[str, Any]
) -> None:
    """Store *profiles*/*active* and mirror *current* as flat keys, keeping all other settings."""
    data = read_json(settings_path, {})
    if not isinstance(data, dict):
        data = {}
    data[PROFILES_KEY] = json.dumps(profiles, ensure_ascii=True)
    data[ACTIVE_KEY] = str(active)

    cur = normalize_image_filters(current)
    for k in flat_filter_setting_keys():
        v = cur[k]
        # Flat keys always use 2 decimals (unlike realism_* in the profile store).
        data[FLAT_PREFIX + k] = f"{v:.2f}" if isinstance(v, float) else encode_store_value(k, v)
    settings_path = Path(settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(settings_path, data)
