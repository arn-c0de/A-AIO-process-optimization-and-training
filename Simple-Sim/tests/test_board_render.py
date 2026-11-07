"""Renderer side of the board model: rule scale from the drawn pads, keep-out along the body axis."""

from __future__ import annotations

import numpy as np
import pytest

from simple_sim.generators.opencv2d import photoreal
from simple_sim.generators.opencv2d.photoreal import copper, parts
from simple_sim.generators.opencv2d.photoreal.context import RenderContext


def _pads(size_px: int) -> list[dict[str, int]]:
    return [{"x": 0, "y": 0, "w": size_px, "h": size_px + 5}, {"x": 40, "y": 0, "w": size_px + 2, "h": size_px}]


def test_rule_scale_follows_the_drawn_pad_not_mm_per_px():
    # 0603 at the profile scale (31 px pads at 0.01 mm/px): the presets' reference -> 1.
    assert photoreal.board_rule_scale(_pads(31), 0.01, "resistor") == pytest.approx(1.0, abs=1e-6)
    # Same drawing, supersampled 2x (px_to_mm halves, pads double): same scale.
    assert photoreal.board_rule_scale(_pads(62), 0.005, "resistor") == pytest.approx(1.0, abs=1e-6)
    # QFN-32 at 0.03 mm/px (8 px pads): IC boost, not the hairlines of an unscaled preset.
    qfn = photoreal.board_rule_scale(_pads(8), 0.03, "qfn")
    assert qfn == pytest.approx(8 * 0.03 / photoreal.RULES_REF_PAD_MM * photoreal.FINE_PITCH_RULE_BOOST)
    lo, hi = photoreal.RULE_SCALE_LIMITS
    assert photoreal.board_rule_scale(_pads(1), 0.001, "resistor") == lo
    assert photoreal.board_rule_scale(_pads(500), 0.1, "soic") == hi
    assert photoreal.board_rule_scale([], 0.01, "resistor") == 1.0


def test_body_extent_follows_the_body_angle():
    assert parts.body_half_extent(30.0, 10.0, 0.0) == pytest.approx((30.0, 10.0))
    assert parts.body_half_extent(30.0, 10.0, 90.0) == pytest.approx((10.0, 30.0))
    ex, ey = parts.body_half_extent(30.0, 10.0, 45.0)
    assert ex == pytest.approx(40.0 / np.sqrt(2.0)) and ey == pytest.approx(ex)


def test_draw_via_window_matches_full_canvas():
    """The window-local via is pixel-identical to drawing on the whole canvas (incl. clipped borders)."""
    rng = np.random.default_rng(3)
    base = np.array([40.0, 90.0, 30.0], dtype=np.float32)
    for ss in (1, 2, 4):
        ctx = RenderContext(ss=ss)
        for _ in range(60):
            h, w = (int(v) for v in rng.integers(20, 160, 2))
            img = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
            center = (int(rng.integers(-8, w + 8)), int(rng.integers(-8, h + 8)))
            r, tented = int(rng.integers(1, 20)), bool(rng.integers(2))
            full, local = img.copy(), img.copy()
            copper.draw_via_window(ctx, full, base, center, r, tented)  # the drawing on the full canvas
            copper.draw_via(ctx, local, base, center, r, tented)
            assert np.array_equal(full, local)


@pytest.mark.parametrize("footprint,mm_per_px", [("sot23", 0.01), ("qfn_32", 0.03), ("chip_2pad", 0.01)])
def test_escape_warn_mode_renders(footprint, mm_per_px):
    """``drc: warn`` logs violations but always returns an image (strict mode raises instead)."""
    from tests.test_photoreal import _render

    board = {"routing": "escape", "drc": "warn", "pour_probability": 1.0}
    img = _render(footprint, "OK", seed=4, board=board, mm_per_px=mm_per_px)
    assert img.shape == (128, 128, 3)
    assert np.array_equal(img, _render(footprint, "OK", seed=4, board=board, mm_per_px=mm_per_px))
