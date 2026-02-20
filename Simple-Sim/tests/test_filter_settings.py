from simple_sim.generators.filter_settings import (
    clean_filter_values_for_popup,
    default_filter_values_for_popup,
    normalize_image_filters,
)


def test_normalize_image_filters_has_core_keys() -> None:
    out = normalize_image_filters(None)
    assert out["enable"] is True
    assert "rotation_strength" in out
    assert "rotation_strength_min" in out
    assert "rotation_strength_max" in out
    assert out["filter_mode"] == "custom"
    assert "realism_group_G1_prob" in out


def test_popup_default_and_clean_presets() -> None:
    defaults = default_filter_values_for_popup()
    clean = clean_filter_values_for_popup()
    assert defaults["enable_rotation"] is True
    assert clean["enable_rotation"] is False
    assert clean["blur_strength"] == "0.00"
