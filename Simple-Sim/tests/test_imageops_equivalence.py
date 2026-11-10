"""Bit-identity of the hand-written photoreal primitives with the NumPy / full-frame code they replace.

``imageops.gradient``, ``heightmap_normals`` and the windowed ``contact_shadow`` claim the exact
float results of ``np.gradient``, ``np.linalg.norm`` and a full-frame shadow rolled with ``np.roll``.
The golden tests only pin that on the rendered images; these tests pin it on the primitives,
including the border cases (1-2 px wide maps, masks touching every frame edge and corner).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from simple_sim.generators.opencv2d.photoreal.context import RenderContext
from simple_sim.generators.opencv2d.photoreal.imageops import _rolled_spans, blend, contact_shadow, gradient, heightmap_normals
from simple_sim.generators.opencv2d.photoreal.look import QUINS_LOOK

DTYPES = (np.float32, np.float64)
SHAPES = ((2, 2), (2, 7), (7, 2), (3, 3), (2, 1000), (17, 23), (64, 48))


def _hmap(shape: tuple[int, int], dtype, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 3.0, shape) + rng.uniform(-50, 50)).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_gradient_equals_np_gradient(shape, dtype):
    f = _hmap(shape, dtype)
    gy, gx = gradient(f)
    ry, rx = np.gradient(f)
    assert gy.dtype == ry.dtype and gx.dtype == rx.dtype
    assert np.array_equal(gy, ry) and np.array_equal(gx, rx)


@pytest.mark.parametrize("shape", ((1, 5), (5, 1), (1, 1)))
def test_gradient_of_one_pixel_wide_map_fails_like_np_gradient(shape):
    f = np.zeros(shape, np.float32)
    with pytest.raises(ValueError):
        np.gradient(f)
    with pytest.raises(ValueError):
        gradient(f)


def test_gradient_of_integer_map_falls_back_to_np_gradient():
    f = np.arange(12, dtype=np.int32).reshape(3, 4) ** 2
    for a, b in zip(gradient(f), np.gradient(f)):
        assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_heightmap_normals_equal_linalg_norm(shape, dtype):
    hmap = _hmap(shape, dtype, seed=1)
    gy, gx = np.gradient(hmap)
    ref = np.dstack([-gx, -gy, np.ones_like(hmap)])
    ref /= np.linalg.norm(ref, axis=2, keepdims=True)
    out = heightmap_normals(hmap)
    assert out.dtype == ref.dtype and np.array_equal(out, ref)


def _shadow_reference(ctx: RenderContext, img: np.ndarray, mask: np.ndarray, strength: float, spread: float) -> None:
    """The full-frame contact shadow: dilate, blur, ``1 - strength * shadow``, ``np.roll`` away from the light."""
    if not mask.any():
        return
    strength = strength * ctx.look.shadow
    shift = int(1.5 * ctx.ss)
    sh = cv2.dilate(mask, np.ones((int(spread), int(spread)), np.uint8))
    sh = cv2.GaussianBlur(sh, (0, 0), spread * 0.8).astype(np.float32) / 255.0
    img *= np.roll(1.0 - strength * sh, (shift, shift), axis=(0, 1))[..., None]


def _mask_positions(h: int, w: int, size: int) -> list[tuple[int, int]]:
    ys, xs = (0, (h - size) // 2, h - size), (0, (w - size) // 2, w - size)
    return [(y, x) for y in ys for x in xs]  # corners, edge midpoints, centre


@pytest.mark.parametrize("ss", (1, 2, 3, 4))
@pytest.mark.parametrize("frame", ((48, 64), (13, 11), (64, 7)))
@pytest.mark.parametrize("size", (1, 3, 9))
def test_contact_shadow_window_equals_full_frame(ss, frame, size):
    h, w = frame
    size = min(size, h, w)
    rng = np.random.default_rng(ss * 100 + size)
    for look_ctx in (RenderContext(ss=ss), RenderContext(ss=ss, look=QUINS_LOOK)):
        for y, x in _mask_positions(h, w, size):
            mask = np.zeros((h, w), np.uint8)
            mask[y:y + size, x:x + size] = 255
            img = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
            ref = img.copy()
            contact_shadow(look_ctx, img, mask, 0.35, 3 * ss)
            _shadow_reference(look_ctx, ref, mask, 0.35, 3 * ss)
            assert np.array_equal(img, ref), (ss, frame, size, y, x, look_ctx.look.name)


@pytest.mark.parametrize("frame", ((40, 50), (6, 9)))
def test_contact_shadow_full_and_empty_masks(frame):
    h, w = frame
    ctx = RenderContext(ss=2)
    img = np.full((h, w, 3), 120.0, np.float32)
    out = img.copy()
    contact_shadow(ctx, out, np.zeros((h, w), np.uint8), 0.35, 6)
    assert np.array_equal(out, img)  # empty mask: untouched
    mask = np.full((h, w), 255, np.uint8)
    ref = img.copy()
    contact_shadow(ctx, out, mask, 0.35, 6)
    _shadow_reference(ctx, ref, mask, 0.35, 6)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("n", (1, 2, 5, 17))
def test_rolled_spans_match_np_roll(n):
    idx = np.arange(n)
    for shift in range(2 * n + 2):
        rolled = np.roll(idx, shift)
        for a0 in range(n):
            for a1 in range(a0 + 1, n + 1):
                target = np.full(n, -1)
                for start, stop, off in _rolled_spans(a0, a1, shift, n):
                    target[start:stop] = idx[a0 + off:a0 + off + stop - start]
                expect = np.where((rolled >= a0) & (rolled < a1), rolled, -1)
                assert np.array_equal(target, expect), (n, shift, a0, a1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_blend_bounding_box_equals_full_frame(dtype):
    rng = np.random.default_rng(3)
    h, w = 31, 37
    for y, x in _mask_positions(h, w, 4):
        alpha = np.zeros((h, w), dtype)
        alpha[y:y + 4, x:x + 4] = rng.uniform(0.0, 1.0, (4, 4))
        img = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
        for color in (np.array([10.0, 200.0, 90.0], np.float32), rng.uniform(0, 255, (h, w, 3)).astype(np.float32)):
            out, ref = img.copy(), img.copy()
            blend(out, color, alpha)
            ref *= 1.0 - alpha[..., None]
            ref += color * alpha[..., None]
            assert np.array_equal(out, ref)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_heightmap_normals_of_one_pixel_wide_maps(shape, dtype):
    """Tiny solder features at small ROIs give 1-px height maps: flat along that axis, no crash."""
    from simple_sim.generators.opencv2d.photoreal.imageops import heightmap_normals

    hmap = np.arange(np.prod(shape), dtype=dtype).reshape(shape) * 0.5
    n = heightmap_normals(hmap)
    assert n.shape == (*shape, 3) and n.dtype == dtype and np.all(np.isfinite(n))
    np.testing.assert_allclose(np.linalg.norm(n, axis=2), 1.0, rtol=1e-6)
    if shape[0] == 1:
        assert np.all(n[..., 1] == 0)
    if shape[1] == 1:
        assert np.all(n[..., 0] == 0)
    if shape == (1, 7):
        np.testing.assert_allclose(n[..., 0], -0.5 / np.sqrt(1.25), rtol=1e-6)


def test_small_roi_tombstones_render():
    """Regression: a 32 px ROI of run_0001 crashed in np.gradient on TOMBSTONE / solder classes."""
    from pathlib import Path

    from simple_sim.config import load_config, merge_profile_render
    from simple_sim.filter_settings import normalize_image_filters
    from simple_sim.generation.common import sample_record
    from simple_sim.generators.opencv2d import render_roi
    from simple_sim.profile_hash import load_profile

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "configs" / "run_0001.yaml")
    cfg["roi"].update(width_px=32, height_px=32)
    prof = load_profile(cfg["run"]["component_profile"], root / "configs" / "profiles")
    render = merge_profile_render(cfg["render"], prof, cfg.get("roi"))
    fp = prof["component"]["footprint"]
    for cls in ("TOMBSTONE", "SOLDER_BALL", "EXCESS_SOLDER"):
        for i in range(40):
            rec, _ = sample_record(i, cls, config=cfg, run_seed=cfg["run"]["seed"], geometry_ranges=prof.get("geometry_ranges"),
                                   tolerances=prof.get("tolerances"), image_filters=normalize_image_filters(None),
                                   enable_cardinal_rotation_90=True, footprint=fp)
            img = render_roi(rec["nominal"], rec["defect"], rec["augment"], (32, 32), render, prof.get("tolerances"),
                             np.random.default_rng(i), footprint=fp)
            assert img.shape == (32, 32, 3)
