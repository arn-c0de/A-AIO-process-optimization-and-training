"""``render.missing_residue``: solder residue on pads without a part (default 1.0 = bit-identical to before)."""

import numpy as np
import pytest

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import RenderContext
from simple_sim.generators.opencv2d.photoreal.context import missing_residue

from tests.test_photoreal import CFG, FOOTPRINTS, TOL


def _render(defect_type="MISSING", seed=2, **cfg):
    rng = np.random.default_rng(seed)
    nominal = sample_nominal_geometry({}, rng, FOOTPRINTS["chip_2pad"])
    defect = sample_defect_params(defect_type, rng, TOL)
    return render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), {**CFG, "mm_per_px": 0.01, **cfg}, TOL, rng)


def test_default_and_explicit_one_are_identical():
    assert missing_residue(None) == 1.0 and missing_residue({"missing_residue": 1}) == 1.0
    assert RenderContext.for_render({}, np.random.default_rng(0)).missing_residue == 1.0
    for cfg in ({}, {"solder": "meniscus"}, {"look": "dv_bottom"}):
        assert np.array_equal(_render(**cfg), _render(missing_residue=1.0, **cfg))


# dv_bottom without JPEG / sharpening: block artefacts would spread a local change over the whole image.
_DV_NO_JPEG = {"look": "dv_bottom", "camera": {"preset": "dv_bottom", "jpeg_quality": [100, 100], "sharpen": [0.0, 0.0]}}


@pytest.mark.parametrize("cfg", [{}, {"solder": "meniscus"}, _DV_NO_JPEG])
def test_residue_changes_only_the_empty_pads(cfg):
    base, bare, heavy = _render(**cfg), _render(missing_residue=0.0, **cfg), _render(missing_residue=2.0, **cfg)
    assert not np.array_equal(base, bare) and not np.array_equal(base, heavy)
    # The pads of the centred chip lie in the middle band; the board outside it is untouched (a few levels of
    # lens blur spill at the band edge).
    outside = np.ones((128, 128), bool)
    outside[26:102, 6:122] = False
    assert np.abs(base.astype(int) - bare)[outside].max() <= 4 and np.abs(base.astype(int) - heavy)[outside].max() <= 4
    # An OK part covers its pads: no empty pad, so the option changes nothing.
    assert np.array_equal(_render("OK", **cfg), _render("OK", missing_residue=0.0, **cfg))


def test_validation():
    from simple_sim.config import _validate_missing_residue

    _validate_missing_residue(None)
    _validate_missing_residue(0)
    _validate_missing_residue(2.5)
    for bad in (-0.1, 5, "1", True):
        with pytest.raises(ValueError, match="missing_residue"):
            _validate_missing_residue(bad)
    with pytest.raises(ValueError, match="missing_residue"):
        missing_residue({"missing_residue": 9})


def test_extend_check_sees_the_option():
    from simple_sim.generation.render_options import look_options

    base = {"render": {"backend": "opencv_2d"}}
    assert look_options(base)["render.missing_residue"] == 1.0
    assert look_options({"render": {"backend": "opencv_2d", "missing_residue": 1.5}})["render.missing_residue"] == 1.5
