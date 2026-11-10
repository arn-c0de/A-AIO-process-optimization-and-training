"""Draw defect visualization overlays on images."""

from __future__ import annotations
import numpy as np
import cv2
from typing import Any
import math


_FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours per defect class.
_GREEN, _RED, _YELLOW, _MAGENTA, _ORANGE, _CYAN = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0)


def _label(img: np.ndarray, text: str, *, color: tuple[int, int, int], xy: tuple[int, int] = (10, 30)) -> None:
    """Readable label with a subtle backing box (works on bright/dark ROIs)."""
    h, w = img.shape[:2]
    x, y = xy
    scale, thick, pad = 0.8, 2, 4
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    cv2.rectangle(img, (max(0, x - pad), max(0, y - th - pad)), (min(w - 1, x + tw + pad), min(h - 1, y + pad)), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thick)


def _draw_ok(img: np.ndarray, cx: int, cy: int, _params: dict[str, Any]) -> None:
    """Green checkmark at the component center."""
    size = 30
    knee = (cx - size // 6, cy + size // 2)
    cv2.line(img, (cx - size // 2, cy), knee, _GREEN, 3)
    cv2.line(img, knee, (cx + size // 2, cy - size // 2), _GREEN, 3)
    _label(img, 'OK', color=_GREEN, xy=(cx + size, max(25, cy - size // 2)))


def _draw_missing(img: np.ndarray, cx: int, cy: int, _params: dict[str, Any]) -> None:
    """Red X at the expected location."""
    half = 40 // 2
    cv2.line(img, (cx - half, cy - half), (cx + half, cy + half), _RED, 3)
    cv2.line(img, (cx - half, cy + half), (cx + half, cy - half), _RED, 3)
    _label(img, 'MISSING', color=_RED, xy=(cx + 40, max(25, cy)))


def _draw_misaligned(img: np.ndarray, cx: int, cy: int, params: dict[str, Any]) -> None:
    """Yellow arrow along the shift vector plus a rotation arc (each only when significant)."""
    shift_x, shift_y = params.get('shift_x', 0.0), params.get('shift_y', 0.0)
    rotation_deg = params.get('rotation_deg', 0.0)
    shift_magnitude = math.sqrt(shift_x**2 + shift_y**2)
    if shift_magnitude > 1.0:
        scale = min(3.0, 50.0 / (shift_magnitude + 1e-6))  # scaled for visibility
        arrow_end = (cx + int(shift_x * scale), cy + int(shift_y * scale))
        cv2.arrowedLine(img, (cx, cy), arrow_end, _YELLOW, 2, tipLength=0.3)
        cv2.putText(img, f'shift: {shift_magnitude:.1f}px', (arrow_end[0] + 10, arrow_end[1]), _FONT, 0.6, _YELLOW, 2)
    if abs(rotation_deg) > 3.0:
        arc_radius, start_angle = 40, -90  # arc starts at the top
        cv2.ellipse(img, (cx, cy), (arc_radius, arc_radius), 0, start_angle, start_angle + rotation_deg, _YELLOW, 2)
        cv2.putText(img, f'rot: {rotation_deg:.1f}°', (cx + arc_radius + 10, cy - arc_radius), _FONT, 0.6, _YELLOW, 2)
    _label(img, 'MISALIGNED', color=_YELLOW)


def _draw_tombstone(img: np.ndarray, cx: int, cy: int, params: dict[str, Any]) -> None:
    """Magenta vertical bar plus a rectangle rotated by the tilt angle."""
    tilt_deg = params.get('tilt_deg', 0.0)
    rect_width, rect_height = 40, 60
    cv2.line(img, (cx, cy - rect_height // 2), (cx, cy + rect_height // 2), _MAGENTA, 6)

    tilt_rad = math.radians(tilt_deg)
    cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
    corners = np.array([
        [-rect_width // 2, -rect_height // 2],
        [rect_width // 2, -rect_height // 2],
        [rect_width // 2, rect_height // 2],
        [-rect_width // 2, rect_height // 2],
    ], dtype=np.float32)
    rotated = np.zeros_like(corners)
    rotated[:, 0] = corners[:, 0] * cos_t - corners[:, 1] * sin_t + cx
    rotated[:, 1] = corners[:, 0] * sin_t + corners[:, 1] * cos_t + cy
    cv2.polylines(img, [rotated.astype(np.int32)], True, _MAGENTA, 2)

    _label(img, 'TOMBSTONE', color=_MAGENTA)
    _label(img, f'tilt: {tilt_deg:.1f}°', color=_MAGENTA, xy=(10, 60))


def _draw_solder_bridge(img: np.ndarray, cx: int, cy: int, _params: dict[str, Any]) -> None:
    """Red bar across the component area (no footprint-specific pad geometry)."""
    w = img.shape[1]
    cv2.line(img, (max(10, cx - int(0.35 * w)), cy), (min(w - 10, cx + int(0.35 * w)), cy), _RED, 10)
    _label(img, 'SOLDER_BRIDGE', color=_RED)


def _draw_corner_lift(img: np.ndarray, cx: int, cy: int, params: dict[str, Any]) -> None:
    """Orange marker at the top-right corner plus the tilt angle."""
    h, w = img.shape[:2]
    size = max(18, min(w, h) // 8)
    x, y = min(w - 5, cx + size), max(5, cy - size)
    pts = np.array([[x, y], [x + size, y], [x + size, y + size]], dtype=np.int32)
    cv2.polylines(img, [pts], True, _ORANGE, 4)
    _label(img, 'CORNER_LIFT', color=_ORANGE)
    _label(img, f"tilt: {params.get('tilt_deg', 0.0):.1f}°", color=_ORANGE, xy=(10, 60))


_DEFECT_DRAWERS = {
    'OK': _draw_ok,
    'MISSING': _draw_missing,
    'MISALIGNED': _draw_misaligned,
    'TOMBSTONE': _draw_tombstone,
    'SOLDER_BRIDGE': _draw_solder_bridge,
    'CORNER_LIFT': _draw_corner_lift,
}


def draw_defect_overlay(img: np.ndarray, defect_params: dict[str, Any],
                        nominal: dict[str, float]) -> np.ndarray:
    """Draw defect visualization overlay on image.

    Args:
        img: Image array (BGR format)
        defect_params: Defect parameters from metadata
        nominal: Nominal geometry parameters

    Returns:
        Image with overlay drawn (copy, original not modified)
    """
    img_overlay = img.copy()
    h, w = img.shape[:2]
    defect_type = str(defect_params.get('type', 'OK') or 'OK')
    drawer = _DEFECT_DRAWERS.get(defect_type)
    if drawer is None:
        # Fallback: always show the class label even if there is no bespoke icon yet.
        _label(img_overlay, defect_type, color=_CYAN)
    else:
        drawer(img_overlay, w // 2, h // 2, defect_params)
    return img_overlay


def draw_two_stage_overlay(
    img: np.ndarray,
    gt_class: str,
    pred_class: str,
    defect_conf: float,
    gt_profile: str,
    pred_profile: str,
    profile_conf: float,
) -> np.ndarray:
    """Draw two-stage prediction overlay showing both profile and defect match info.

    Args:
        img: Image array (BGR format)
        gt_class: Ground-truth defect class
        pred_class: Predicted defect class
        defect_conf: Defect prediction confidence (0-1)
        gt_profile: Ground-truth profile id
        pred_profile: Predicted profile id
        profile_conf: Profile prediction confidence (0-1)

    Returns:
        Image with two-stage overlay
    """
    img_overlay = img.copy()
    h, w = img.shape[:2]

    profile_ok = (gt_profile == pred_profile)
    defect_ok = (gt_class == pred_class)

    # Border color: both correct → green, profile wrong → orange, defect wrong → red
    if defect_ok and profile_ok:
        border_color = (0, 255, 0)      # green
    elif not defect_ok:
        border_color = (0, 0, 255)      # red (defect wrong, regardless of profile)
    else:
        border_color = (0, 165, 255)    # orange (profile wrong only)

    border_thickness = 8
    cv2.rectangle(img_overlay, (0, 0), (w - 1, h - 1), border_color, border_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top section: profile info
    prof_mark = "Y" if profile_ok else "X"
    prof_text = f"Profile: {pred_profile} ({profile_conf:.0%}) {prof_mark} (GT: {gt_profile})"
    cv2.putText(img_overlay, prof_text, (10, 30), font, 0.6, (255, 255, 255), 2)

    # Bottom section: defect info
    def_mark = "Y" if defect_ok else "X"
    def_text = f"Defect: {pred_class} ({defect_conf:.0%}) {def_mark} (GT: {gt_class})"
    cv2.putText(img_overlay, def_text, (10, h - 15), font, 0.6, (255, 255, 255), 2)

    return img_overlay


def draw_prediction_overlay(img: np.ndarray, ground_truth: str, predicted: str,
                            confidence: float) -> np.ndarray:
    """Draw prediction comparison overlay on image.

    Args:
        img: Image array (BGR format)
        ground_truth: True class label
        predicted: Predicted class label
        confidence: Prediction confidence (0-1)

    Returns:
        Image with prediction overlay
    """
    img_overlay = img.copy()
    h, w = img.shape[:2]

    # Determine color based on correctness
    is_correct = (ground_truth == predicted)
    color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, Red if wrong

    # Draw semi-transparent border
    border_thickness = 8
    cv2.rectangle(img_overlay, (0, 0), (w-1, h-1), color, border_thickness)

    # Status label at top
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    cv2.putText(img_overlay, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Prediction info at bottom
    pred_text = f'Pred: {predicted} ({confidence:.1%})'
    cv2.putText(img_overlay, pred_text, (10, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    gt_text = f'Truth: {ground_truth}'
    cv2.putText(img_overlay, gt_text, (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img_overlay
