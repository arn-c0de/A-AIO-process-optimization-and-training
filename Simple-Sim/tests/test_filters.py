import numpy as np

from simple_sim.generators.filter_settings import NEUTRAL_AUGMENT
from simple_sim.generators.opencv2d.filters import apply_post_filters, apply_sharpen


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
