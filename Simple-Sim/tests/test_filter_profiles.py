"""Tests for simple_sim.filter_profiles (settings.json-backed filter profile store)."""

from __future__ import annotations

import json
from pathlib import Path

from simple_sim.filter_profiles import (
    flat_filter_setting_keys,
    load_shared_filter_profiles,
    save_shared_filter_profiles,
    to_profile_store,
)


def test_missing_settings_gives_default_profile(tmp_path: Path) -> None:
    profiles, active, current = load_shared_filter_profiles(tmp_path / "nope.json")
    assert active == "Default" and list(profiles) == ["Default"]
    assert current["enable_blur"] is True


def test_legacy_flat_keys_are_used(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"pipeline.filter.enable_blur": False, "pipeline.filter.blur_strength": "0.50"}), encoding="utf-8")
    profiles, active, current = load_shared_filter_profiles(p)
    assert current["enable_blur"] is False and current["blur_strength"] == 0.5
    assert profiles["Default"]["blur_strength"] == "0.50"


def test_round_trip_keeps_unknown_keys_and_other_settings(tmp_path: Path) -> None:
    p = tmp_path / "gui" / "settings.json"
    stored = {"B": {"enable_blur": False, "my_custom_key": "x"}, "a": {}}
    p.parent.mkdir()
    p.write_text(json.dumps({"other.setting": 1, "pipeline.filter_profiles_json": json.dumps(stored),
                             "pipeline.filter_profile_active": "missing"}), encoding="utf-8")
    profiles, active, current = load_shared_filter_profiles(p)
    assert active == "a"  # unknown active -> first by case-insensitive name
    profiles["B"] = to_profile_store({"enable_grain": False}, profiles["B"])
    assert profiles["B"]["my_custom_key"] == "x" and profiles["B"]["enable_grain"] is False
    save_shared_filter_profiles(p, profiles, "B", profiles["B"])
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["other.setting"] == 1
    assert data["pipeline.filter_profile_active"] == "B"
    assert data["pipeline.filter.enable_grain"] is False
    assert set(k[len("pipeline.filter."):] for k in data if k.startswith("pipeline.filter.")) == set(flat_filter_setting_keys())
    profiles2, active2, current2 = load_shared_filter_profiles(p)
    assert active2 == "B" and current2["my_custom_key"] == "x" and current2["enable_blur"] is False


def test_to_profile_store_encoding() -> None:
    out = to_profile_store({"blur_strength": 0.5, "jpeg_quality": 90, "realism_k_prob_0": 1.0})
    assert out["blur_strength"] == "0.50"
    assert out["jpeg_quality"] == "90"
    assert len(out["realism_k_prob_0"].split(".")[1]) == 4
