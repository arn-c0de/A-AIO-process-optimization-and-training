"""Draw defect visualization overlays on images."""

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Any
import math


def draw_defect_overlay(img: np.ndarray, defect_params: Dict[str, Any],
                        nominal: Dict[str, float]) -> np.ndarray:
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
    center_x, center_y = w // 2, h // 2

    defect_type = defect_params.get('type', 'OK')
    shift_x = defect_params.get('shift_x', 0.0)
    shift_y = defect_params.get('shift_y', 0.0)
    rotation_deg = defect_params.get('rotation_deg', 0.0)
    tilt_deg = defect_params.get('tilt_deg', 0.0)

    if defect_type == 'OK':
        # Green checkmark at component center
        color = (0, 255, 0)  # Green
        thickness = 3
        size = 30

        # Checkmark: two lines forming a check
        check_pt1 = (center_x - size//2, center_y)
        check_pt2 = (center_x - size//6, center_y + size//2)
        check_pt3 = (center_x + size//2, center_y - size//2)

        cv2.line(img_overlay, check_pt1, check_pt2, color, thickness)
        cv2.line(img_overlay, check_pt2, check_pt3, color, thickness)

        # Label
        cv2.putText(img_overlay, 'OK', (center_x + size, center_y - size//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    elif defect_type == 'MISSING':
        # Red X mark at expected location
        color = (0, 0, 255)  # Red
        thickness = 3
        size = 40

        # X mark: two diagonal lines
        cv2.line(img_overlay,
                (center_x - size//2, center_y - size//2),
                (center_x + size//2, center_y + size//2),
                color, thickness)
        cv2.line(img_overlay,
                (center_x - size//2, center_y + size//2),
                (center_x + size//2, center_y - size//2),
                color, thickness)

        # Label
        cv2.putText(img_overlay, 'MISSING', (center_x + size, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    elif defect_type == 'MISALIGNED':
        # Yellow arrow showing shift vector + rotation arc
        color = (0, 255, 255)  # Yellow
        thickness = 2

        # Draw arrow from center to shifted position
        shift_magnitude = math.sqrt(shift_x**2 + shift_y**2)
        if shift_magnitude > 1.0:  # Only draw if significant shift
            # Scale for visibility
            scale = min(3.0, 50.0 / (shift_magnitude + 1e-6))
            arrow_dx = int(shift_x * scale)
            arrow_dy = int(shift_y * scale)

            arrow_end = (center_x + arrow_dx, center_y + arrow_dy)
            cv2.arrowedLine(img_overlay, (center_x, center_y), arrow_end,
                           color, thickness, tipLength=0.3)

            # Label with shift magnitude
            label = f'shift: {shift_magnitude:.1f}px'
            cv2.putText(img_overlay, label,
                       (arrow_end[0] + 10, arrow_end[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw rotation arc if significant
        if abs(rotation_deg) > 3.0:
            arc_radius = 40
            # Arc from 0 to rotation angle
            start_angle = -90  # Start at top
            end_angle = start_angle + rotation_deg

            cv2.ellipse(img_overlay, (center_x, center_y),
                       (arc_radius, arc_radius), 0,
                       start_angle, end_angle, color, thickness)

            # Rotation label
            rot_label = f'rot: {rotation_deg:.1f}°'
            cv2.putText(img_overlay, rot_label,
                       (center_x + arc_radius + 10, center_y - arc_radius),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Main label
        cv2.putText(img_overlay, 'MISALIGNED',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    elif defect_type == 'TOMBSTONE':
        # Purple vertical bar with tilt angle label
        color = (255, 0, 255)  # Magenta/Purple
        thickness = 6

        # Draw thick vertical bar indicating tombstoned component
        bar_height = 60
        cv2.line(img_overlay,
                (center_x, center_y - bar_height//2),
                (center_x, center_y + bar_height//2),
                color, thickness)

        # Draw tilt indicator (skewed rectangle)
        tilt_rad = math.radians(tilt_deg)
        rect_width = 40
        rect_height = 60

        # Calculate corners of tilted rectangle
        cos_t = math.cos(tilt_rad)
        sin_t = math.sin(tilt_rad)

        corners = np.array([
            [-rect_width//2, -rect_height//2],
            [rect_width//2, -rect_height//2],
            [rect_width//2, rect_height//2],
            [-rect_width//2, rect_height//2]
        ], dtype=np.float32)

        # Rotate corners
        rotated = np.zeros_like(corners)
        rotated[:, 0] = corners[:, 0] * cos_t - corners[:, 1] * sin_t
        rotated[:, 1] = corners[:, 0] * sin_t + corners[:, 1] * cos_t

        # Translate to center
        rotated[:, 0] += center_x
        rotated[:, 1] += center_y

        # Draw tilted rectangle
        pts = rotated.astype(np.int32)
        cv2.polylines(img_overlay, [pts], True, color, 2)

        # Labels
        cv2.putText(img_overlay, 'TOMBSTONE',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img_overlay, f'tilt: {tilt_deg:.1f}°',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
