"""Defect classification and parameter sampling."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from simple_sim.classes import TYPE_LABELLED_CLASSES

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
_DEFAULT_TOMBSTONE_TILT = 75.0

# Scale factor applied to the tolerance-derived minimum for the upper bound of
# misaligned sampling ranges.
_MISALIGNED_SHIFT_SCALE = 5.0
_MISALIGNED_ROT_SCALE = 6.0

# Numerical epsilon to avoid divide-by-zero when normalising a shift vector.
_EPS = 1e-6

# ``defects.placement_sampling``: how OK / MISALIGNED placements are drawn.
#   legacy      OK stays inside 0.7 x tolerance, MISALIGNED starts at tolerance + 1 px: nothing is ever
#               rendered in the band around the decision boundary.
#   continuous  OK fills the tolerance up to the boundary, MISALIGNED starts right at it with most
#               samples close to it (the hard cases) and a tail out to the legacy maximum; the shift
#               direction is uniform. Labels are still decided by :func:`classify_defect`.
PLACEMENT_SAMPLING = ("legacy", "continuous")
DEFAULT_PLACEMENT_SAMPLING = "legacy"
# continuous: OK placements stop this close to the boundary (labels use ``>``; keeps float drift out).
_BOUNDARY_GAP = 0.98
# continuous: share of the MISALIGNED range covered by half the samples (u**2 density, see _beyond()).
_MISALIGNED_TAIL_SCALE = 5.0


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

    if defect_type in TYPE_LABELLED_CLASSES:
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
    tolerances: dict[str, Any] | None = None,
    *,
    placement: str = DEFAULT_PLACEMENT_SAMPLING,
) -> dict[str, Any]:
    """Sample defect parameters based on type.

    Args:
        defect_type: Target defect class ('OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE',
                     'SOLDER_BRIDGE', 'CORNER_LIFT', 'SOLDER_BALL', 'EXCESS_SOLDER')
        rng: NumPy random generator
        tolerances: Optional tolerance thresholds from config for footprint-specific sampling
        placement: ``defects.placement_sampling`` (see :data:`PLACEMENT_SAMPLING`); affects the
            placement of OK, MISALIGNED, SOLDER_BALL and EXCESS_SOLDER only

    Returns:
        Dictionary with defect parameters
    """
    if placement not in PLACEMENT_SAMPLING:
        raise ValueError(f"defects.placement_sampling must be one of {PLACEMENT_SAMPLING}, got {placement!r}")
    ok_shift_px, ok_rotation_deg, tombstone_tilt_deg = _tolerance_bounds(tolerances)
    continuous = placement == "continuous"
    ok_sampler = _sample_ok_continuous if continuous else _sample_ok

    samplers: dict[str, Callable[..., _Sample]] = {
        'OK':           ok_sampler,
        'MISSING':      _sample_missing,
        'MISALIGNED':   _sample_misaligned_continuous if continuous else _sample_misaligned,
        'TOMBSTONE':    _sample_tombstone,
        'SOLDER_BRIDGE': _sample_solder_bridge,
        'CORNER_LIFT':  _sample_corner_lift,
        # Solder defects on a correctly placed part (placement like OK; the renderer draws the solder).
        'SOLDER_BALL':  ok_sampler,
        'EXCESS_SOLDER': ok_sampler,
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
    # _tolerance_bounds() always fills in defaults, so the bounds are never None here.
    shift_bound = max(0.0, _OK_SAFETY_MARGIN * ok_shift_px)
    rot_bound = max(0.0, _OK_SAFETY_MARGIN * ok_rotation_deg)
    sx = rng.uniform(-shift_bound, shift_bound)
    sy = rng.uniform(-shift_bound, shift_bound)
    rot = rng.uniform(-rot_bound, rot_bound)
    return sx, sy, rot, 0.0


def _sample_missing(rng, _ok_shift, _ok_rot, _tombstone):
    return 0.0, 0.0, 0.0, 0.0


def _sample_misaligned(rng, ok_shift_px, ok_rotation_deg, _tombstone):
    min_shift = ok_shift_px + 1.0
    max_shift = max(min_shift, _MISALIGNED_SHIFT_SCALE * min_shift)
    min_rot = ok_rotation_deg + 1.0
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
        rot_bound = max(0.0, _OK_SAFETY_MARGIN * ok_rotation_deg)
        rot = rng.uniform(-rot_bound, rot_bound)
    else:
        # Misalignment via rotation; shift stays inside tolerance
        shift_bound = max(0.0, _OK_SAFETY_MARGIN * ok_shift_px)
        sx = rng.uniform(-shift_bound, shift_bound)
        sy = rng.uniform(-shift_bound, shift_bound)
        rot = rng.uniform(-max_rot, max_rot)
        if abs(rot) < min_rot:
            rot = min_rot if rot >= 0 else -min_rot

    return sx, sy, rot, 0.0


def _uniform_disc(rng, r_max: float) -> tuple[float, float]:
    """Point uniformly distributed in a disc of radius *r_max* (uniform direction)."""
    r = r_max * np.sqrt(rng.random())
    a = rng.uniform(0.0, 2.0 * np.pi)
    return float(r * np.cos(a)), float(r * np.sin(a))


def _beyond(rng, lo: float, hi: float) -> float:
    """Value in ``(lo, hi]`` with density concentrated at *lo*: half the samples fall into the first
    quarter of the range (``lo + (hi - lo) * u**2``), so the boundary cases dominate but the tail keeps
    the gross defects the legacy sampler produced."""
    return lo + (hi - lo) * float(rng.random()) ** 2


def _sample_ok_continuous(rng, ok_shift_px, ok_rotation_deg, _tombstone):
    """OK placement that fills the whole tolerance (uniform over the disc / rotation interval)."""
    sx, sy = _uniform_disc(rng, _BOUNDARY_GAP * ok_shift_px)
    rot = rng.uniform(-_BOUNDARY_GAP * ok_rotation_deg, _BOUNDARY_GAP * ok_rotation_deg)
    return sx, sy, rot, 0.0


def _sample_misaligned_continuous(rng, ok_shift_px, ok_rotation_deg, _tombstone):
    """MISALIGNED placement starting right at the tolerance boundary (see :data:`PLACEMENT_SAMPLING`).

    Half the samples are pure shifts, half pure rotations, as in the legacy sampler; the other
    quantity stays an OK value, so the label hinges on one measurement.
    """
    max_shift = _MISALIGNED_TAIL_SCALE * (ok_shift_px + 1.0)  # legacy upper bounds
    max_rot = _MISALIGNED_ROT_SCALE * (ok_rotation_deg + 1.0)
    if rng.random() < 0.5:
        mag = _beyond(rng, ok_shift_px / _BOUNDARY_GAP, max_shift)  # strictly > tolerance
        a = rng.uniform(0.0, 2.0 * np.pi)
        sx, sy = float(mag * np.cos(a)), float(mag * np.sin(a))
        rot = rng.uniform(-_BOUNDARY_GAP * ok_rotation_deg, _BOUNDARY_GAP * ok_rotation_deg)
    else:
        sx, sy = _uniform_disc(rng, _BOUNDARY_GAP * ok_shift_px)
        rot = _beyond(rng, ok_rotation_deg / _BOUNDARY_GAP, max_rot) * (1.0 if rng.random() < 0.5 else -1.0)
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
