import numpy as np
import pytest

from simple_sim.filter_settings import NEUTRAL_AUGMENT
from simple_sim.generators.opencv2d.augment import apply_image_filter_overrides


@pytest.mark.parametrize("base", [0.0, 90.0, 180.0, 270.0])
def test_rotation_strength_scales_only_jitter(base: float) -> None:
    aug = {"rotation_deg": base + 3.0}
    filt = {"enable": True, "enable_rotation": True, "rotation_strength": 0.5}
    out = apply_image_filter_overrides(aug, filt, np.random.default_rng(0))
    assert out["rotation_deg"] == pytest.approx(base + 1.5)


def test_rotation_strength_zero_keeps_cardinal_base() -> None:
    aug = {"rotation_deg": 270.0 - 2.0}
    filt = {"enable": True, "enable_rotation": True, "rotation_strength": 0.0}
    out = apply_image_filter_overrides(aug, filt, np.random.default_rng(0))
    assert out["rotation_deg"] == pytest.approx(270.0)


def test_filters_disabled_resets_to_neutral() -> None:
    aug = {"blur_sigma": 2.0, "rotation_deg": 95.0, "jpeg_quality": 60}
    out = apply_image_filter_overrides(aug, {"enable": False}, np.random.default_rng(0))
    for key, neutral in NEUTRAL_AUGMENT.items():
        assert out[key] == neutral


def test_rotation_sampler_matches_sample_augment_params() -> None:
    """``sample_rotation_deg`` is the rotation draw of ``sample_augment_params`` (same two draws, same stream)."""
    from simple_sim.generators.opencv2d.augment import clean_augment, sample_augment_params, sample_rotation_deg

    aug_cfg = {"contrast_factor_range": [0.9, 1.1], "rotation_deg_range": [-4.0, 4.0]}
    dom_cfg = {"blur_sigma": [0.0, 1.0], "noise_stddev": [0.0, 5.0], "lighting_brightness": [0.9, 1.1]}
    for cardinal in (True, False):
        full = sample_augment_params(aug_cfg, dom_cfg, np.random.default_rng(11), enable_cardinal_rotation_90=cardinal)
        rng = np.random.default_rng(11)
        rot = sample_rotation_deg(rng, (-4.0, 4.0), enable_cardinal_rotation_90=cardinal)
        assert rot == full["rotation_deg"]
        assert float(rng.uniform(0.0, 1.0)) == full["blur_sigma"]  # the stream continues where the sampler left it
    clean = clean_augment(95.0)
    assert clean == {"blur_sigma": 0.0, "noise_stddev": 0.0, "brightness_factor": 1.0, "contrast_factor": 1.0, "rotation_deg": 95.0}
    assert all(clean[k] == NEUTRAL_AUGMENT[k] for k in clean if k != "rotation_deg")
