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
