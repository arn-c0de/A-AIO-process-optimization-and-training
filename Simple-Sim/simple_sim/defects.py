"""Defect classification and parameter sampling."""

import numpy as np
from typing import Dict, Any, Optional


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
    shift_x = defect_params['shift_x']
    shift_y = defect_params['shift_y']
    rotation_deg = abs(defect_params['rotation_deg'])
    tilt_deg = abs(defect_params['tilt_deg'])

    # MISSING: component not present
    if defect_type == 'MISSING':
        return 'MISSING'

    # TOMBSTONE: tilt exceeds threshold
    if tilt_deg >= tolerances['tombstone_tilt_deg']:
        return 'TOMBSTONE'

    # MISALIGNED: shift or rotation exceeds threshold
    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
    if shift_magnitude > tolerances['ok_shift_px'] or rotation_deg > tolerances['ok_rotation_deg']:
        return 'MISALIGNED'

    # OK: within tolerances
    return 'OK'


def sample_defect_params(
    defect_type: str,
    rng: np.random.Generator,
    tolerances: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Sample defect parameters based on type.

    Args:
        defect_type: Target defect class ('OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE')
        rng: NumPy random generator
        tolerances: Optional tolerance thresholds from config for footprint-specific sampling

    Returns:
        Dictionary with defect parameters
    """
    ok_shift_px = None
    ok_rotation_deg = None
    tombstone_tilt_deg = None
    if tolerances is not None:
        ok_shift_px = float(tolerances['ok_shift_px'])
        ok_rotation_deg = float(tolerances['ok_rotation_deg'])
        tombstone_tilt_deg = float(tolerances['tombstone_tilt_deg'])

    if defect_type == 'OK':
        # Small jitter within tolerance. If tolerances are provided, stay comfortably inside them.
        if ok_shift_px is not None and ok_rotation_deg is not None:
            # Keep well inside tolerance to avoid edge-case label flips due to rounding.
            shift_bound = max(0.0, 0.7 * ok_shift_px)
            rot_bound = max(0.0, 0.7 * ok_rotation_deg)
            shift_x = rng.uniform(-shift_bound, shift_bound)
            shift_y = rng.uniform(-shift_bound, shift_bound)
            rotation_deg = rng.uniform(-rot_bound, rot_bound)
        else:
            shift_x = rng.uniform(-2.0, 2.0)
            shift_y = rng.uniform(-2.0, 2.0)
            rotation_deg = rng.uniform(-3.0, 3.0)
        tilt_deg = 0.0

    elif defect_type == 'MISSING':
        # No component, other params don't matter
        shift_x = 0.0
        shift_y = 0.0
        rotation_deg = 0.0
        tilt_deg = 0.0

    elif defect_type == 'MISALIGNED':
        # Large shift or rotation, but no tilt
        # If tolerances are provided, ensure we exceed them (not hardcoded constants).
        min_misaligned_shift = (ok_shift_px + 1.0) if ok_shift_px is not None else 5.0
        max_misaligned_shift = max(min_misaligned_shift, 5.0 * min_misaligned_shift)
        min_misaligned_rot = (ok_rotation_deg + 1.0) if ok_rotation_deg is not None else 8.0
        max_misaligned_rot = max(min_misaligned_rot, 6.0 * min_misaligned_rot)

        if rng.random() < 0.5:
            # Large shift
            shift_x = rng.uniform(-max_misaligned_shift, max_misaligned_shift)
            shift_y = rng.uniform(-max_misaligned_shift, max_misaligned_shift)
            # Ensure magnitude exceeds tolerance-derived minimum.
            magnitude = np.sqrt(shift_x**2 + shift_y**2)
            if magnitude < min_misaligned_shift:
                scale = min_misaligned_shift / (magnitude + 1e-6)
                shift_x *= scale
                shift_y *= scale
        else:
            # Small shift
            if ok_shift_px is not None:
                # Keep shift inside tolerance; misalignment will come from rotation.
                shift_bound = max(0.0, 0.7 * ok_shift_px)
                shift_x = rng.uniform(-shift_bound, shift_bound)
                shift_y = rng.uniform(-shift_bound, shift_bound)
            else:
                shift_x = rng.uniform(-2.0, 2.0)
                shift_y = rng.uniform(-2.0, 2.0)

        # Large rotation
        rotation_deg = rng.uniform(-max_misaligned_rot, max_misaligned_rot)
        # Ensure absolute rotation exceeds tolerance-derived minimum.
        if abs(rotation_deg) < min_misaligned_rot:
            rotation_deg = min_misaligned_rot if rotation_deg >= 0 else -min_misaligned_rot

        tilt_deg = 0.0

    elif defect_type == 'TOMBSTONE':
        # Large tilt (>= threshold). If tolerances are provided, use them.
        tilt_min = tombstone_tilt_deg if tombstone_tilt_deg is not None else 75.0
        tilt_max = min(90.0, tilt_min + 10.0)
        tilt_deg = rng.uniform(tilt_min, tilt_max)

        # Keep shift/rotation modest; defect is driven by tilt.
        shift_x = rng.uniform(-5.0, 5.0)
        shift_y = rng.uniform(-5.0, 5.0)
        rotation_deg = rng.uniform(-10.0, 10.0)

    else:
        raise ValueError(f"Unknown defect type: {defect_type}")

    return {
        'type': defect_type,
        'shift_x': float(shift_x),
        'shift_y': float(shift_y),
        'rotation_deg': float(rotation_deg),
        'tilt_deg': float(tilt_deg)
    }
