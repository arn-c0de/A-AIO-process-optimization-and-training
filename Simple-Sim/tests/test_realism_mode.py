import os

from simple_sim.generators.opencv2d.augment import apply_image_filter_overrides


def _base_augment() -> dict:
    return {
        "blur_sigma": 0.0,
        "noise_stddev": 0.0,
        "brightness_factor": 1.0,
        "contrast_factor": 1.0,
        "rotation_deg": 0.0,
        "motion_blur_strength": 0.0,
        "reflection_strength": 0.0,
        "jpeg_quality": 100,
    }


def test_realism_mode_keeps_glare_jpeg_incompatible() -> None:
    filt = {
        "enable": True,
        "filter_mode": "realism",
        "realism_enabled": True,
    }
    for _ in range(300):
        out = apply_image_filter_overrides(_base_augment(), filt)
        # Reflection/glare events should not co-exist with heavy JPEG degradation.
        assert not (float(out.get("reflection_strength", 0.0)) > 1e-6 and int(out.get("jpeg_quality", 100)) < 100)


def test_realism_mode_constraints_toggle_allows_glare_jpeg_combo_when_disabled() -> None:
    filt = {
        "enable": True,
        "filter_mode": "realism",
        "realism_enabled": True,
        "realism_constraints_enabled": False,
        "realism_k_prob_0": 0.0,
        "realism_k_prob_1": 0.0,
        "realism_k_prob_2": 1.0,
        "realism_group_G5_prob": 1.0,
        "realism_group_G7_prob": 1.0,
    }
    seen_combo = False
    for _ in range(200):
        out = apply_image_filter_overrides(_base_augment(), filt)
        if float(out.get("reflection_strength", 0.0)) > 1e-6 and int(out.get("jpeg_quality", 100)) < 100:
            seen_combo = True
            break
    assert seen_combo


def test_realism_mode_k0_baseline_is_sparse_when_forced() -> None:
    filt = {
        "enable": True,
        "filter_mode": "realism",
        "realism_enabled": True,
        "realism_k_prob_0": 1.0,
        "realism_k_prob_1": 0.0,
        "realism_k_prob_2": 0.0,
    }
    for _ in range(80):
        out = apply_image_filter_overrides(_base_augment(), filt)
        assert float(out.get("motion_blur_strength", 0.0)) <= 1e-6
        assert float(out.get("reflection_strength", 0.0)) <= 1e-6
        assert int(out.get("jpeg_quality", 100)) == 100


def test_realism_mode_uses_yaml_profile_preset(tmp_path) -> None:
    cfg = tmp_path / "realism_profiles.yaml"
    cfg.write_text(
        "\n".join(
            [
                "profiles:",
                "  profile_test_yaml:",
                "    k0: 0.0",
                "    k1: 1.0",
                "    k2: 0.0",
                "    groups:",
                "      G1: 0.0",
                "      G2: 1.0",
                "      G3: 0.0",
                "      G4: 0.0",
                "      G5: 0.0",
                "      G6: 0.0",
                "      G7: 0.0",
                "      G8: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    old = os.environ.get("SIMPLE_SIM_REALISM_PRESETS_PATH")
    os.environ["SIMPLE_SIM_REALISM_PRESETS_PATH"] = str(cfg)
    try:
        filt = {
            "enable": True,
            "filter_mode": "realism",
            "realism_enabled": True,
            "realism_profile_id": "profile_test_yaml",
            # Do not override with explicit realism_k/group values.
        }
        for _ in range(40):
            out = apply_image_filter_overrides(_base_augment(), filt)
            assert float(out.get("motion_blur_strength", 0.0)) > 1e-6
            assert float(out.get("reflection_strength", 0.0)) <= 1e-6
            assert int(out.get("jpeg_quality", 100)) == 100
    finally:
        if old is None:
            os.environ.pop("SIMPLE_SIM_REALISM_PRESETS_PATH", None)
        else:
            os.environ["SIMPLE_SIM_REALISM_PRESETS_PATH"] = old
