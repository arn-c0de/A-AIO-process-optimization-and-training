"""2D OpenCV-based PCB defect rendering."""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def sample_nominal_geometry(
    roi_config: Dict[str, Any],
    rng: np.random.Generator,
    geometry_ranges: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Sample nominal component geometry.

    Args:
        roi_config: ROI configuration section
        rng: NumPy random generator
        geometry_ranges: Optional geometry sampling ranges from config

    Returns:
        Dictionary with nominal geometry parameters
    """
    # Defaults: 0603 footprint typical dimensions (in pixels at 0.01 mm/px)
    # Real 0603: 1.6mm x 0.8mm component, pads ~0.8mm x 1.0mm
    # At 0.01 mm/px: component ~60x30 px, pads ~30x35 px
    defaults = {
        'pad_width': [25.0, 35.0],
        'pad_height': [30.0, 40.0],
        'pad_spacing': [50.0, 65.0],
        'component_length': [55.0, 65.0],
        'component_width': [25.0, 35.0],
    }
    gr = geometry_ranges or {}

    def _range(name: str):
        r = gr.get(name, defaults[name])
        return float(r[0]), float(r[1])

    pad_width = rng.uniform(*_range('pad_width'))
    pad_height = rng.uniform(*_range('pad_height'))
    pad_spacing = rng.uniform(*_range('pad_spacing'))
    component_length = rng.uniform(*_range('component_length'))
    component_width = rng.uniform(*_range('component_width'))

    result = {
        'pad_width': float(pad_width),
        'pad_height': float(pad_height),
        'pad_spacing': float(pad_spacing),
        'component_length': float(component_length),
        'component_width': float(component_width)
    }

    # Optional: sample pad_spacing_y for multi-pad footprints (e.g. SOT-23)
    if 'pad_spacing_y' in gr:
        lo, hi = float(gr['pad_spacing_y'][0]), float(gr['pad_spacing_y'][1])
        result['pad_spacing_y'] = float(rng.uniform(lo, hi))

    return result


def sample_augment_params(augment_config: Dict[str, Any], domain_config: Dict[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """Sample augmentation parameters.

    Args:
        augment_config: Augmentation configuration section
        domain_config: Domain-specific configuration
        rng: NumPy random generator

    Returns:
        Dictionary with augmentation parameters
    """
    # Domain-specific blur and noise
    blur_min, blur_max = domain_config['blur_sigma']
    blur_sigma = rng.uniform(blur_min, blur_max)

    noise_min, noise_max = domain_config['noise_stddev']
    noise_stddev = rng.uniform(noise_min, noise_max)

    # Domain-specific brightness
    bright_min, bright_max = domain_config['lighting_brightness']
    brightness_factor = rng.uniform(bright_min, bright_max)

    # Global contrast augmentation
    contrast_min, contrast_max = augment_config['contrast_factor_range']
    contrast_factor = rng.uniform(contrast_min, contrast_max)

    # Global rotation augmentation
    rot_min, rot_max = augment_config['rotation_deg_range']
    rotation_deg = rng.uniform(rot_min, rot_max)

    return {
        'blur_sigma': float(blur_sigma),
        'noise_stddev': float(noise_stddev),
        'brightness_factor': float(brightness_factor),
        'contrast_factor': float(contrast_factor),
        'rotation_deg': float(rotation_deg)
    }


def draw_substrate(img: np.ndarray, config: Dict[str, Any]) -> None:
    """Draw green PCB substrate background.

    Args:
        img: Image array to modify (in-place)
        config: Render configuration
    """
    substrate_color = config['substrate_color']  # BGR
    img[:] = substrate_color


def _compute_pad_positions(
    footprint: str,
    nominal: Dict[str, float],
    img_shape: Tuple[int, int]
) -> List[Dict[str, int]]:
    """Compute pad rectangles based on footprint type.

    Args:
        footprint: Footprint identifier ('chip_2pad', 'sot23', etc.)
        nominal: Nominal geometry parameters
        img_shape: (height, width) of the image

    Returns:
        List of dicts with keys 'x', 'y', 'w', 'h' for each pad rectangle
    """
    h, w = img_shape
    center_x, center_y = w // 2, h // 2

    pad_width = int(nominal['pad_width'])
    pad_height = int(nominal['pad_height'])
    pad_spacing = int(nominal['pad_spacing'])

    if footprint == 'sot23':
        pad_spacing_y = int(nominal['pad_spacing_y'])
        # Pads 1+2 on the left side get 80% height
        small_h = int(pad_height * 0.8)

        # Ensure minimum gap of 10 pixels between left pads (Pad1 and Pad2)
        # Gap = pad_spacing_y - small_h, so ensure pad_spacing_y > small_h + min_gap
        min_gap_px = 10
        effective_spacing_y = max(pad_spacing_y, small_h + min_gap_px)

        # Pad 1 (left-top)
        p1_x = center_x - pad_spacing // 2 - pad_width // 2
        p1_y = center_y - effective_spacing_y // 2 - small_h // 2

        # Pad 2 (left-bottom)
        p2_x = center_x - pad_spacing // 2 - pad_width // 2
        p2_y = center_y + effective_spacing_y // 2 - small_h // 2

        # Pad 3 (right-center, full size)
        p3_x = center_x + pad_spacing // 2 - pad_width // 2
        p3_y = center_y - pad_height // 2

        return [
            {'x': p1_x, 'y': p1_y, 'w': pad_width, 'h': small_h},
            {'x': p2_x, 'y': p2_y, 'w': pad_width, 'h': small_h},
            {'x': p3_x, 'y': p3_y, 'w': pad_width, 'h': pad_height},
        ]
    elif footprint.startswith('qfn'):
        pad_spacing_y = int(nominal['pad_spacing_y'])
        # 4 pads around perimeter: left, right, top, bottom
        # Left pad (vertical orientation)
        left_x = center_x - pad_spacing // 2 - pad_width // 2
        left_y = center_y - pad_height // 2

        # Right pad (vertical orientation)
        right_x = center_x + pad_spacing // 2 - pad_width // 2
        right_y = center_y - pad_height // 2

        # Top pad (horizontal orientation — dimensions swapped)
        top_x = center_x - pad_height // 2
        top_y = center_y - pad_spacing_y // 2 - pad_width // 2

        # Bottom pad (horizontal orientation — dimensions swapped)
        bot_x = center_x - pad_height // 2
        bot_y = center_y + pad_spacing_y // 2 - pad_width // 2

        return [
            {'x': left_x,  'y': left_y,  'w': pad_width,  'h': pad_height},
            {'x': right_x, 'y': right_y, 'w': pad_width,  'h': pad_height},
            {'x': top_x,   'y': top_y,   'w': pad_height,  'h': pad_width},
            {'x': bot_x,   'y': bot_y,   'w': pad_height,  'h': pad_width},
        ]
    else:
        # Default: chip_2pad — two symmetric pads (left/right)
        left_x = center_x - pad_spacing // 2 - pad_width // 2
        left_y = center_y - pad_height // 2

        right_x = center_x + pad_spacing // 2 - pad_width // 2
        right_y = center_y - pad_height // 2

        return [
            {'x': left_x, 'y': left_y, 'w': pad_width, 'h': pad_height},
            {'x': right_x, 'y': right_y, 'w': pad_width, 'h': pad_height},
        ]


def draw_pads(img: np.ndarray, nominal: Dict[str, float], color: Tuple[int, int, int],
              footprint: str = 'chip_2pad') -> None:
    """Draw copper pads.

    Args:
        img: Image array to modify (in-place)
        nominal: Nominal geometry parameters
        color: BGR color tuple
        footprint: Footprint identifier for pad layout
    """
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
    for pad in pads:
        cv2.rectangle(img,
                      (pad['x'], pad['y']),
                      (pad['x'] + pad['w'], pad['y'] + pad['h']),
                      color, -1)


def draw_solder(img: np.ndarray, nominal: Dict[str, float],
                footprint: str = 'chip_2pad') -> None:
    """Draw solder paste highlights on pads.

    Args:
        img: Image array to modify (in-place)
        nominal: Nominal geometry parameters
        footprint: Footprint identifier for pad layout
    """
    solder_color = (200, 200, 200)  # Light gray/white
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
    h, w = img.shape[:2]
    for pad in pads:
        x, y, pw, ph = pad['x'], pad['y'], pad['w'], pad['h']
        # Clamp to image bounds; some footprints (e.g. qfn) can produce pads that partially
        # fall outside the ROI depending on geometry ranges.
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, int(x + pw))
        y1 = min(h, int(y + ph))
        if x1 <= x0 or y1 <= y0:
            continue

        solder_region = img[y0:y1, x0:x1].copy()
        overlay = np.full_like(solder_region, solder_color)
        # Avoid passing a pre-allocated dst to OpenCV here; some array layouts can trip the bindings.
        blended = cv2.addWeighted(solder_region, 0.7, overlay, 0.3, 0)
        if blended is None:
            continue
        img[y0:y1, x0:x1] = blended


def draw_component(
    img: np.ndarray,
    nominal: Dict[str, float],
    defect_params: Dict[str, Any],
    color: Tuple[int, int, int],
    tolerances: Optional[Dict[str, Any]] = None
) -> None:
    """Draw component with defects applied.

    Args:
        img: Image array to modify (in-place)
        nominal: Nominal geometry parameters
        defect_params: Defect parameters
        color: BGR color tuple
        tolerances: Optional tolerance thresholds from config (used for tombstone rendering threshold)
    """
    # Skip drawing if MISSING
    if defect_params['type'] == 'MISSING':
        return

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Apply defect shifts
    shift_x = int(defect_params['shift_x'])
    shift_y = int(defect_params['shift_y'])
    comp_center_x = center_x + shift_x
    comp_center_y = center_y + shift_y

    # Handle TOMBSTONE (tilted component)
    tilt_deg = abs(defect_params['tilt_deg'])
    tombstone_thresh = 75.0
    if tolerances is not None:
        tombstone_thresh = float(tolerances['tombstone_tilt_deg'])

    if tilt_deg >= tombstone_thresh:
        # Tombstone: thin vertical rectangle
        comp_width = int(nominal['component_width'] * 0.2)
        comp_height = int(nominal['component_length'] * 1.5)
    else:
        # Normal orientation
        comp_width = int(nominal['component_width'])
        comp_height = int(nominal['component_length'])

    # Create rotated rectangle
    rotation_deg = defect_params['rotation_deg']
    rect_center = (comp_center_x, comp_center_y)
    rect_size = (comp_height, comp_width)  # (length, width) for horizontal component
    angle = rotation_deg

    # Get rotated rectangle points
    box = cv2.boxPoints(((rect_center[0], rect_center[1]), rect_size, angle))
    box = box.astype(np.int32)

    # Draw filled polygon
    cv2.fillPoly(img, [box], color)

    # Add subtle edge highlight for depth
    edge = tuple(int(min(255, max(0, c * 1.2))) for c in color)
    cv2.polylines(img, [box], True, edge, 1)


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur.

    Args:
        img: Input image
        sigma: Blur sigma

    Returns:
        Blurred image
    """
    if sigma <= 0:
        return img

    # Kernel size must be odd
    ksize = int(sigma * 6)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_noise(img: np.ndarray, stddev: float, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian noise.

    Args:
        img: Input image
        stddev: Noise standard deviation
        rng: NumPy random generator

    Returns:
        Noisy image
    """
    if stddev <= 0:
        return img

    noise = rng.normal(0, stddev, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Apply brightness adjustment.

    Args:
        img: Input image
        factor: Brightness multiplier

    Returns:
        Adjusted image
    """
    if abs(factor - 1.0) < 1e-6:
        return img

    adjusted = img.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Apply contrast adjustment around mid-gray.

    Args:
        img: Input image
        factor: Contrast multiplier (1.0 = no-op)

    Returns:
        Adjusted image
    """
    if abs(factor - 1.0) < 1e-6:
        return img

    adjusted = (img.astype(np.float32) - 128.0) * factor + 128.0
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def render_roi(
    nominal: Dict[str, float],
    defect_params: Dict[str, Any],
    augment: Dict[str, float],
    roi_size: Tuple[int, int],
    config: Dict[str, Any],
    tolerances: Optional[Dict[str, Any]],
    rng: np.random.Generator,
    footprint: str = 'chip_2pad'
) -> np.ndarray:
    """Render complete ROI with defects and augmentation.

    Args:
        nominal: Nominal geometry parameters
        defect_params: Defect parameters
        augment: Augmentation parameters
        roi_size: (width, height) in pixels
        config: Render configuration
        tolerances: Optional tolerance thresholds from config (used for tombstone rendering threshold)
        rng: NumPy random generator
        footprint: Footprint identifier for pad layout

    Returns:
        Rendered image (BGR, uint8)
    """
    width, height = roi_size
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw base substrate
    draw_substrate(img, config)

    # Draw copper pads
    copper_color = tuple(config['copper_color'])
    draw_pads(img, nominal, copper_color, footprint=footprint)

    # Draw solder paste
    draw_solder(img, nominal, footprint=footprint)

    # Draw component with defects
    component_color = tuple(config['component_color'])
    draw_component(img, nominal, defect_params, component_color, tolerances=tolerances)

    # Apply augmentation
    img = apply_blur(img, augment['blur_sigma'])
    img = apply_noise(img, augment['noise_stddev'], rng)
    img = apply_brightness(img, augment['brightness_factor'])
    img = apply_contrast(img, augment['contrast_factor'])

    # Apply rotation augmentation
    if abs(augment['rotation_deg']) > 0.1:
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, augment['rotation_deg'], 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return img
