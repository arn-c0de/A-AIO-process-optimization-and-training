"""Meniscus solder joints (``render.solder: meniscus``, simple_sim/generators/opencv2d/solder.py)."""

import numpy as np
import pytest

from simple_sim.generators.opencv2d import solder
from simple_sim.generators.opencv2d.photoreal import _solder_look
from tests.test_photoreal import CLASSES, FOOTPRINTS, _render


def test_default_style_is_legacy():
    assert solder.solder_style(None) == solder.solder_style({}) == solder.solder_style({"solder": None}) == "legacy"
    assert solder.solder_style({"solder": "meniscus"}) == "meniscus"
    with pytest.raises(ValueError, match="render.solder"):
        solder.solder_style({"solder": "shiny"})


def test_legacy_is_the_default_render():
    assert np.array_equal(_render("chip_2pad", "OK", seed=3), _render("chip_2pad", "OK", seed=3, solder="legacy"))


@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
@pytest.mark.parametrize("defect_type", CLASSES)
def test_meniscus_renders_all_footprints_and_classes(footprint, defect_type):
    img = _render(footprint, defect_type, solder="meniscus")
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert 10 < img.mean() < 200
    assert _solder_look() is None  # the thread's joint model is reset after the render


def test_meniscus_is_deterministic_and_only_changes_the_joints():
    """The joints draw from a side stream: board, part and camera noise stay those of the legacy render."""
    a = _render("chip_2pad", "OK", seed=5, solder="meniscus")
    assert np.array_equal(a, _render("chip_2pad", "OK", seed=5, solder="meniscus"))
    legacy = _render("chip_2pad", "OK", seed=5)
    changed = np.abs(a.astype(int) - legacy.astype(int)).max(axis=2) > 0
    assert 0.01 < changed.mean() < 0.4


def test_side_stream_does_not_advance_the_generator():
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    first = solder.side_stream(rng, 1).random()
    assert rng.bit_generator.state == state
    assert solder.side_stream(rng, 1).random() == first and solder.side_stream(rng, 2).random() != first


def test_fillet_is_concave_from_terminal_to_toe():
    rng = np.random.default_rng(0)
    look = solder.sample_look(rng)
    look = solder.SolderLook(**{**look.__dict__, "lumps": 0.0, "grain": 0.0, "lip": 0.0})
    ph, pw, ss = 80, 60, 2
    covered = np.zeros((ph, pw), bool)
    covered[:, :20] = True
    wet = np.zeros((ph, pw), np.float32)
    wet[1:-1, 1:-1] = 1.0  # contact line at the pad edge
    h = solder.fillet_height(covered, wet, 18.0, look, rng, ss)[ph // 2, 20:-1]
    assert np.all(np.diff(h) <= 1e-4)                 # falls from the terminal to the toe
    slope = -np.diff(h)
    assert slope[:5].mean() > 2.0 * slope[-10:].mean()  # steep at the terminal, flat at the toe (concave)
    excess = solder.fillet_height(covered, wet, 18.0, look, rng, ss, excess=True)[ph // 2, 20:-1]
    assert excess[len(excess) // 2] > h[len(h) // 2]  # too much solder: convex and higher


def test_meniscus_joint_is_a_high_contrast_mirror():
    """Real joints show bright highlights next to dark reflections of the mask and the part body."""
    rng = np.random.default_rng(1)
    look = solder.sample_look(rng)
    region = np.full((76, 60, 3), (165.0, 170.0, 175.0), np.float32)
    covered = np.zeros((76, 60), bool)
    covered[8:68, :24] = True
    solder.render_joint(region, np.array((64.0, 89.0, 58.0)), covered, 18.0, look, rng, 2)
    lum = region[~covered].mean(axis=1)
    assert np.percentile(lum, 95) - np.percentile(lum, 5) > 80
