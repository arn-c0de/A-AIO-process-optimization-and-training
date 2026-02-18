"""Defect classification and parameter sampling."""

import numpy as np
from typing import Dict, Any, Optional

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

def classify_defect(nominal: Dict[str, float], defect_params: Dict[str, Any], tolerances: Dict[str, Any]) -> str:
    """Classify defect based on tolerances.

    Deterministic labeling based on threshold rules.

    Args:
        nominal: Nominal geometry parameters
        defect_params: Sampled defect parameters
        tolerances: Tolerance thresholds from config

    Returns:
        Class name: 'OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE'
    """
    defect_type = defect_params['type']

    if defect_type in ('SOLDER_BRIDGE', 'CORNER_LIFT', 'MISSING'):
        return defect_type

    tilt_deg = abs(defect_params['tilt_deg'])
    if tilt_deg >= tolerances['tombstone_tilt_deg']:
        return 'TOMBSTONE'

    shift_x = defect_params['shift_x']
    shift_y = defect_params['shift_y']
    rotation_deg = abs(defect_params['rotation_deg'])
    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
    if shift_magnitude > tolerances['ok_shift_px'] or rotation_deg > tolerances['ok_rotation_deg']:
        return 'MISALIGNED'

    return 'OK'


def sample_defect_params(
    defect_type: str,
    rng: np.random.Generator,
    tolerances: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Sample defect parameters based on type.

    Args:
        defect_type: Target defect class ('OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE',
                     'SOLDER_BRIDGE', 'CORNER_LIFT')
        rng: NumPy random generator
        tolerances: Optional tolerance thresholds from config for footprint-specific sampling

    Returns:
        Dictionary with defect parameters
    """
    ok_shift_px = float(tolerances['ok_shift_px']) if tolerances else None
    ok_rotation_deg = float(tolerances['ok_rotation_deg']) if tolerances else None
    tombstone_tilt_deg = float(tolerances['tombstone_tilt_deg']) if tolerances else None

    samplers = {
        'OK':           _sample_ok,
        'MISSING':      _sample_missing,
        'MISALIGNED':   _sample_misaligned,
        'TOMBSTONE':    _sample_tombstone,
        'SOLDER_BRIDGE': _sample_solder_bridge,
        'CORNER_LIFT':  _sample_corner_lift,
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
        # Misalignment via shift
        sx = rng.uniform(-max_shift, max_shift)
        sy = rng.uniform(-max_shift, max_shift)
        magnitude = np.sqrt(sx**2 + sy**2)
        if magnitude < min_shift:
            scale = min_shift / (magnitude + _EPS)
            sx *= scale
            sy *= scale
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
    tilt_min = tombstone_tilt_deg if tombstone_tilt_deg is not None else _DEFAULT_TOMBSTONE_TILT
    tilt_max = min(90.0, tilt_min + 10.0)
    tilt = rng.uniform(tilt_min, tilt_max)
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
