"""Test defect sampling and 2D rendering of defect classes."""

import numpy as np

from simple_sim.defects import classify_defect, sample_defect_params
from simple_sim.generators.opencv2d.draw import draw_component, draw_pads, draw_solder

TOL = {'ok_shift_px': 2.0, 'ok_rotation_deg': 3.0, 'tombstone_tilt_deg': 75.0}
NOMINAL = {'pad_width': 30, 'pad_height': 35, 'pad_spacing': 55, 'component_length': 60, 'component_width': 30}


def test_misaligned_includes_pure_shift_samples():
    rng = np.random.default_rng(0)
    pure_shift = 0
    for _ in range(200):
        d = sample_defect_params('MISALIGNED', rng, TOL)
        assert classify_defect(NOMINAL, d, TOL) == 'MISALIGNED'
        if abs(d['rotation_deg']) <= TOL['ok_rotation_deg']:
            pure_shift += 1
            assert np.hypot(d['shift_x'], d['shift_y']) > TOL['ok_shift_px']
    assert pure_shift > 50


def _render(defect, footprint="chip_2pad"):
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = (40, 110, 40)
    draw_pads(img, NOMINAL, (60, 140, 200), footprint=footprint)
    draw_solder(img, NOMINAL, footprint=footprint)
    draw_component(img, NOMINAL, defect, (40, 40, 40), tolerances=TOL, footprint=footprint)
    return img


def test_solder_bridge_and_corner_lift_render_differently_from_ok():
    base = {'shift_x': 0.3, 'shift_y': -0.2, 'rotation_deg': 1.0}
    ok = _render({**base, 'type': 'OK', 'tilt_deg': 0.0})
    bridge = _render({**base, 'type': 'SOLDER_BRIDGE', 'tilt_deg': 1.0})
    lift = _render({**base, 'type': 'CORNER_LIFT', 'tilt_deg': 50.0})
    assert np.abs(bridge.astype(int) - ok).sum() > 0
    assert np.abs(lift.astype(int) - ok).sum() > 0
    # Bridge adds bright solder pixels not present in the OK render.
    assert (bridge.min(axis=2) > 180).sum() > (ok.min(axis=2) > 180).sum()
    # Deterministic for identical params.
    assert np.array_equal(bridge, _render({**base, 'type': 'SOLDER_BRIDGE', 'tilt_deg': 1.0}))


def test_shift_is_rounded_not_truncated():
    a = _render({'type': 'MISALIGNED', 'shift_x': 2.6, 'shift_y': 0.0, 'rotation_deg': 0.0, 'tilt_deg': 0.0})
    b = _render({'type': 'MISALIGNED', 'shift_x': 3.0, 'shift_y': 0.0, 'rotation_deg': 0.0, 'tilt_deg': 0.0})
    assert np.array_equal(a, b)
