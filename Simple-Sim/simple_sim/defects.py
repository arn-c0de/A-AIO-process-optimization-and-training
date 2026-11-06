"""Defect classification and parameter sampling."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

_Sample = tuple[float, float, float, float]  # shift_x, shift_y, rotation_deg, tilt_deg

# ---------------------------------------------------------------------------
# Sampling constants
# ---------------------------------------------------------------------------

# Safety margin: keep OK samples this fraction inside tolerance bounds to avoid
# edge-case label flips due to rounding / floating-point drift.
_OK_SAFETY_MARGIN = 0.7

# Fallback bounds when no profile tolerances are available.
_DEFAULT_OK_SHIFT_PX = 2.0
_DEFAULT_OK_ROT_DEG = 3.0
_DEFAULT_MISALIGNED_SHIFT = 5.0
_DEFAULT_MISALIGNED_ROT = 8.0
_DEFAULT_TOMBSTONE_TILT = 75.0

# Scale factor applied to the tolerance-derived minimum for the upper bound of
# misaligned sampling ranges.
_MISALIGNED_SHIFT_SCALE = 5.0
_MISALIGNED_ROT_SCALE = 6.0

# Numerical epsilon to avoid divide-by-zero when normalising a shift vector.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _tolerance_bounds(tolerances: dict[str, Any] | None) -> tuple[float, float, float]:
    """``(ok_shift_px, ok_rotation_deg, tombstone_tilt_deg)`` with defaults for missing keys.

    Shared by classify_defect() and sample_defect_params() so partial/missing tolerances
    classify and sample consistently.
    """
    tol = tolerances or {}
    return (
        float(tol.get('ok_shift_px', _DEFAULT_OK_SHIFT_PX)),
        float(tol.get('ok_rotation_deg', _DEFAULT_OK_ROT_DEG)),
        float(tol.get('tombstone_tilt_deg', _DEFAULT_TOMBSTONE_TILT)),
    )

def classify_defect(nominal: dict[str, float], defect_params: dict[str, Any], tolerances: dict[str, Any] | None) -> str:
    """Classify defect based on tolerances.

    Deterministic labeling based on threshold rules.

    Args:
        nominal: Nominal geometry parameters
        defect_params: Sampled defect parameters
        tolerances: Tolerance thresholds from config

    Returns:
        Class name: 'OK', 'MISALIGNED', 'TOMBSTONE' or the sampled type for MISSING, SOLDER_BRIDGE,
        CORNER_LIFT, SOLDER_BALL and EXCESS_SOLDER
    """
    defect_type = defect_params['type']

    if defect_type in ('SOLDER_BRIDGE', 'CORNER_LIFT', 'MISSING', 'SOLDER_BALL', 'EXCESS_SOLDER'):
        return defect_type

    ok_shift_px, ok_rotation_deg, tombstone_tilt_deg = _tolerance_bounds(tolerances)

    tilt_deg = abs(defect_params['tilt_deg'])
    if tilt_deg >= tombstone_tilt_deg:
        return 'TOMBSTONE'

    shift_x = defect_params['shift_x']
    shift_y = defect_params['shift_y']
    rotation_deg = abs(defect_params['rotation_deg'])
    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
    if shift_magnitude > ok_shift_px or rotation_deg > ok_rotation_deg:
        return 'MISALIGNED'

    return 'OK'


def sample_defect_params(
    defect_type: str,
    rng: np.random.Generator,
    tolerances: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Sample defect parameters based on type.

    Args:
        defect_type: Target defect class ('OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE',
                     'SOLDER_BRIDGE', 'CORNER_LIFT', 'SOLDER_BALL', 'EXCESS_SOLDER')
        rng: NumPy random generator
        tolerances: Optional tolerance thresholds from config for footprint-specific sampling

    Returns:
        Dictionary with defect parameters
    """
    ok_shift_px, ok_rotation_deg, tombstone_tilt_deg = _tolerance_bounds(tolerances)

    samplers: dict[str, Callable[..., _Sample]] = {
        'OK':           _sample_ok,
        'MISSING':      _sample_missing,
        'MISALIGNED':   _sample_misaligned,
        'TOMBSTONE':    _sample_tombstone,
        'SOLDER_BRIDGE': _sample_solder_bridge,
        'CORNER_LIFT':  _sample_corner_lift,
        # Solder defects on a correctly placed part (placement like OK; the renderer draws the solder).
        'SOLDER_BALL':  _sample_ok,
        'EXCESS_SOLDER': _sample_ok,
    }

    if defect_type not in samplers:
        raise ValueError(f"Unknown defect type: {defect_type}")

    shift_x, shift_y, rotation_deg, tilt_deg = samplers[defect_type](
        rng, ok_shift_px, ok_rotation_deg, tombstone_tilt_deg
    )

    return {
        'type': defect_type,
        'shift_x': float(shift_x),
        'shift_y': float(shift_y),
        'rotation_deg': float(rotation_deg),
        'tilt_deg': float(tilt_deg),
    }


# ---------------------------------------------------------------------------
# Private per-type samplers
# signature: (rng, ok_shift_px, ok_rotation_deg, tombstone_tilt_deg) -> (sx, sy, rot, tilt)
# ---------------------------------------------------------------------------

def _sample_ok(rng, ok_shift_px, ok_rotation_deg, _tombstone):
    if ok_shift_px is not None and ok_rotation_deg is not None:
        shift_bound = max(0.0, _OK_SAFETY_MARGIN * ok_shift_px)
        rot_bound = max(0.0, _OK_SAFETY_MARGIN * ok_rotation_deg)
    else:
        shift_bound = _DEFAULT_OK_SHIFT_PX
        rot_bound = _DEFAULT_OK_ROT_DEG
    sx = rng.uniform(-shift_bound, shift_bound)
    sy = rng.uniform(-shift_bound, shift_bound)
    rot = rng.uniform(-rot_bound, rot_bound)
    return sx, sy, rot, 0.0


def _sample_missing(rng, _ok_shift, _ok_rot, _tombstone):
    return 0.0, 0.0, 0.0, 0.0


def _sample_misaligned(rng, ok_shift_px, ok_rotation_deg, _tombstone):
    min_shift = (ok_shift_px + 1.0) if ok_shift_px is not None else _DEFAULT_MISALIGNED_SHIFT
    max_shift = max(min_shift, _MISALIGNED_SHIFT_SCALE * min_shift)
    min_rot = (ok_rotation_deg + 1.0) if ok_rotation_deg is not None else _DEFAULT_MISALIGNED_ROT
    max_rot = max(min_rot, _MISALIGNED_ROT_SCALE * min_rot)

    if rng.random() < 0.5:
        # Misalignment via shift; rotation stays inside tolerance (pure shift)
        sx = rng.uniform(-max_shift, max_shift)
        sy = rng.uniform(-max_shift, max_shift)
        magnitude = np.sqrt(sx**2 + sy**2)
        if magnitude < min_shift:
            scale = min_shift / (magnitude + _EPS)
            sx *= scale
            sy *= scale
        if ok_rotation_deg is not None:
            rot_bound = max(0.0, _OK_SAFETY_MARGIN * ok_rotation_deg)
        else:
            rot_bound = _DEFAULT_OK_ROT_DEG
        rot = rng.uniform(-rot_bound, rot_bound)
    else:
        # Misalignment via rotation; shift stays inside tolerance
        if ok_shift_px is not None:
            shift_bound = max(0.0, _OK_SAFETY_MARGIN * ok_shift_px)
        else:
            shift_bound = _DEFAULT_OK_SHIFT_PX
        sx = rng.uniform(-shift_bound, shift_bound)
        sy = rng.uniform(-shift_bound, shift_bound)
        rot = rng.uniform(-max_rot, max_rot)
        if abs(rot) < min_rot:
            rot = min_rot if rot >= 0 else -min_rot

    return sx, sy, rot, 0.0


def _sample_tombstone(rng, _ok_shift, _ok_rot, tombstone_tilt_deg):
    # Anywhere from the class threshold up to fully upright (real tombstones end up at all angles).
    tilt_min = tombstone_tilt_deg if tombstone_tilt_deg is not None else _DEFAULT_TOMBSTONE_TILT
    tilt = rng.uniform(tilt_min, 90.0)
    sx = rng.uniform(-5.0, 5.0)
    sy = rng.uniform(-5.0, 5.0)
    rot = rng.uniform(-10.0, 10.0)
    return sx, sy, rot, tilt


def _sample_solder_bridge(rng, _ok_shift, _ok_rot, _tombstone):
    sx = rng.uniform(-1.0, 1.0)
    sy = rng.uniform(-1.0, 1.0)
    rot = rng.uniform(-5.0, 5.0)
    tilt = rng.uniform(0.0, 2.0)
    return sx, sy, rot, tilt


def _sample_corner_lift(rng, _ok_shift, _ok_rot, _tombstone):
    sx = rng.uniform(-2.0, 2.0)
    sy = rng.uniform(-2.0, 2.0)
    rot = rng.uniform(-3.0, 3.0)
    tilt = rng.uniform(40.0, 65.0)
    return sx, sy, rot, tilt
