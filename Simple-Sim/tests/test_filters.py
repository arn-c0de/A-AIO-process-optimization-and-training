import numpy as np

from simple_sim.filter_settings import NEUTRAL_AUGMENT
from simple_sim.generators.opencv2d.filters import apply_post_filters, apply_sharpen
import pytest


def test_sharpen_keeps_uniform_image_unchanged() -> None:
    img = np.full((32, 32, 3), 120, dtype=np.uint8)
    out = apply_sharpen(img, 0.5)
    assert out.shape == img.shape
    # A normalised sharpen kernel (sum == 1) must not brighten/saturate flat regions.
    assert abs(float(out.mean()) - 120.0) < 1e-6
    assert np.array_equal(out, img)


def test_sharpen_increases_local_contrast_on_edges() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, 16:] = 200
    out = apply_sharpen(img, 1.0)
    assert float(out.astype(np.float32).std()) > float(img.astype(np.float32).std())


def test_post_filters_neutral_is_noop() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    out = apply_post_filters(img, dict(NEUTRAL_AUGMENT), np.random.default_rng(1))
    assert np.array_equal(out, img)


def test_motion_blur_keeps_brightness_and_position_for_every_angle():
    from simple_sim.generators.opencv2d.filters import apply_motion_blur

    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    for strength in (0.5, 1.0, 1.5):
        for angle in (0.0, 45.0, 90.0, 135.0):
            out = apply_motion_blur(img, strength, angle)
            assert abs(float(out[16:48, 16:48].mean()) - 120.0) <= 1.0, (strength, angle)
    dot = np.zeros((65, 65), dtype=np.float32)
    dot[32, 32] = 1.0
    ys, xs = np.nonzero(apply_motion_blur(dot, 0.5, 90.0) > 0)  # even kernel size used to shift the image
    assert abs(float(ys.mean()) - 32.0) < 0.26 and abs(float(xs.mean()) - 32.0) < 0.26


def test_chromatic_aberration_acts_below_half_strength():
    from simple_sim.generators.opencv2d.filters import apply_chromatic_aberration

    img = np.zeros((16, 32, 3), dtype=np.uint8)
    img[:, 16:] = 200
    assert not np.array_equal(apply_chromatic_aberration(img, 0.2), img)


def test_filter_angles_from_the_settings_are_applied():
    from simple_sim.generators.opencv2d.augment import apply_image_filter_overrides

    sampled = {"perspective_angle_x": 3.0, "perspective_angle_y": -2.0, "motion_blur_angle": 200.0, "rotation_deg": -50.0}
    filt = {"enable_perspective": True, "perspective_angle_x": 10.0, "enable_motion_blur": True, "motion_blur_angle": 45.0,
            "rotation_strength": 0.5}
    out = apply_image_filter_overrides(dict(sampled), filt, np.random.default_rng(0))
    assert out["perspective_angle_x"] == 10.0 and out["motion_blur_angle"] == 45.0
    assert out["perspective_angle_y"] == -2.0  # not set: the sampled angle stays
    assert out["rotation_deg"] == pytest.approx(-70.0)  # jitter scaled around the cardinal base (-90)
