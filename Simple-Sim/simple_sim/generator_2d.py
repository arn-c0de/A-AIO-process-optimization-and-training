"""2D OpenCV-based PCB defect rendering."""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def normalize_image_filters(image_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize optional IMAGE_FILTERS payload with safe defaults."""
    src = image_filters or {}

    def _bool(name: str, default: bool) -> bool:
        v = src.get(name, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _float(name: str, default: float, lo: float = 0.0, hi: float = 5.0) -> float:
        try:
            v = float(src.get(name, default))
        except Exception:
            v = float(default)
        return max(lo, min(hi, v))

    def _int(name: str, default: int, lo: int = 0, hi: int = 10000) -> int:
        try:
            v = int(src.get(name, default))
        except Exception:
            v = int(default)
        return max(lo, min(hi, v))

    return {
        # Core toggles
        "enable": _bool("enable", True),
        "cardinal_rotation_90": _bool("cardinal_rotation_90", True),

        # Existing filters (already active by default)
        "enable_rotation": _bool("enable_rotation", True),
        "enable_blur": _bool("enable_blur", True),
        "enable_grain": _bool("enable_grain", True),
        "enable_brightness": _bool("enable_brightness", True),
        "enable_contrast": _bool("enable_contrast", True),
        "rotation_strength": _float("rotation_strength", 1.0, 0.0, 4.0),
        "blur_strength": _float("blur_strength", 1.0, 0.0, 4.0),
        "grain_strength": _float("grain_strength", 1.0, 0.0, 4.0),
        "brightness_strength": _float("brightness_strength", 1.0, 0.0, 4.0),
        "contrast_strength": _float("contrast_strength", 1.0, 0.0, 4.0),

        # High priority new filters (active by default)
        "enable_perspective": _bool("enable_perspective", True),
        "perspective_strength": _float("perspective_strength", 1.0, 0.0, 2.0),
        "perspective_angle_x": _float("perspective_angle_x", 0.0, -15.0, 15.0),
        "perspective_angle_y": _float("perspective_angle_y", 0.0, -15.0, 15.0),

        "enable_motion_blur": _bool("enable_motion_blur", True),
        "motion_blur_strength": _float("motion_blur_strength", 1.0, 0.0, 3.0),
        "motion_blur_angle": _float("motion_blur_angle", 0.0, 0.0, 360.0),

        "enable_saturation": _bool("enable_saturation", True),
        "saturation_factor": _float("saturation_factor", 1.0, 0.5, 1.5),

        "enable_hue_shift": _bool("enable_hue_shift", True),
        "hue_shift_deg": _float("hue_shift_deg", 0.0, -30.0, 30.0),

        "enable_shadow": _bool("enable_shadow", True),
        "shadow_strength": _float("shadow_strength", 0.3, 0.0, 0.8),
        "shadow_size": _float("shadow_size", 0.2, 0.1, 0.5),

        "enable_reflection": _bool("enable_reflection", True),
        "reflection_strength": _float("reflection_strength", 0.5, 0.0, 1.0),
        "reflection_size": _float("reflection_size", 0.15, 0.05, 0.3),

        # Medium priority new filters (inactive by default)
        "enable_vignetting": _bool("enable_vignetting", False),
        "vignetting_strength": _float("vignetting_strength", 1.0, 0.0, 2.0),

        "enable_chromatic_aberration": _bool("enable_chromatic_aberration", False),
        "chromatic_strength": _float("chromatic_strength", 1.0, 0.0, 2.0),

        "enable_jpeg_compression": _bool("enable_jpeg_compression", False),
        "jpeg_quality": _int("jpeg_quality", 85, 50, 95),

        "enable_color_temperature": _bool("enable_color_temperature", False),
        "color_temperature_kelvin": _int("color_temperature_kelvin", 5500, 2500, 7500),

        # Low priority new filters (inactive by default)
        "enable_lens_distortion": _bool("enable_lens_distortion", False),
        "distortion_k1": _float("distortion_k1", 0.0, -0.3, 0.3),
        "distortion_k2": _float("distortion_k2", 0.0, -0.1, 0.1),

        "enable_dust": _bool("enable_dust", False),
        "dust_density": _float("dust_density", 0.3, 0.0, 1.0),
        "dust_size": _float("dust_size", 2.0, 1.0, 5.0),

        "enable_sharpen": _bool("enable_sharpen", False),
        "sharpen_strength": _float("sharpen_strength", 1.0, 0.0, 2.0),
    }


def apply_image_filter_overrides(augment: Dict[str, float], image_filters: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Apply toggle + strength overrides to sampled augment params."""
    aug = dict(augment or {})
    filt = normalize_image_filters(image_filters)
    if not filt.get("enable", True):
        return aug

    if not filt.get("enable_blur", True):
        aug["blur_sigma"] = 0.0
    else:
        aug["blur_sigma"] = float(aug.get("blur_sigma", 0.0)) * float(filt.get("blur_strength", 1.0))

    if not filt.get("enable_grain", True):
        aug["noise_stddev"] = 0.0
    else:
        aug["noise_stddev"] = float(aug.get("noise_stddev", 0.0)) * float(filt.get("grain_strength", 1.0))

    if not filt.get("enable_brightness", True):
        aug["brightness_factor"] = 1.0
    else:
        bf = float(aug.get("brightness_factor", 1.0))
        bs = float(filt.get("brightness_strength", 1.0))
        aug["brightness_factor"] = 1.0 + ((bf - 1.0) * bs)

    if not filt.get("enable_contrast", True):
        aug["contrast_factor"] = 1.0
    else:
        cf = float(aug.get("contrast_factor", 1.0))
        cs = float(filt.get("contrast_strength", 1.0))
        aug["contrast_factor"] = 1.0 + ((cf - 1.0) * cs)

    if not filt.get("enable_rotation", True):
        aug["rotation_deg"] = 0.0
    else:
        aug["rotation_deg"] = float(aug.get("rotation_deg", 0.0)) * float(filt.get("rotation_strength", 1.0))

    # New filters - apply overrides
    if not filt.get("enable_perspective", True):
        aug["perspective_strength"] = 0.0
    else:
        aug["perspective_strength"] = float(filt.get("perspective_strength", 1.0))
        aug["perspective_angle_x"] = float(aug.get("perspective_angle_x", 0.0))
        aug["perspective_angle_y"] = float(aug.get("perspective_angle_y", 0.0))

    if not filt.get("enable_motion_blur", True):
        aug["motion_blur_strength"] = 0.0
    else:
        aug["motion_blur_strength"] = float(filt.get("motion_blur_strength", 1.0))
        aug["motion_blur_angle"] = float(aug.get("motion_blur_angle", 0.0))

    if not filt.get("enable_chromatic_aberration", True):
        aug["chromatic_strength"] = 0.0
    else:
        aug["chromatic_strength"] = float(filt.get("chromatic_strength", 1.0))

    if not filt.get("enable_vignetting", True):
        aug["vignetting_strength"] = 0.0
    else:
        aug["vignetting_strength"] = float(filt.get("vignetting_strength", 1.0))

    if not filt.get("enable_saturation", True):
        aug["saturation_factor"] = 1.0
    else:
        sf = float(aug.get("saturation_factor", 1.0))
        aug["saturation_factor"] = sf * float(filt.get("saturation_factor", 1.0))

    if not filt.get("enable_hue_shift", True):
        aug["hue_shift_deg"] = 0.0
    else:
        aug["hue_shift_deg"] = float(aug.get("hue_shift_deg", 0.0))

    if not filt.get("enable_sharpen", True):
        aug["sharpen_strength"] = 0.0
    else:
        aug["sharpen_strength"] = float(filt.get("sharpen_strength", 1.0))

    if not filt.get("enable_lens_distortion", True):
        aug["distortion_k1"] = 0.0
        aug["distortion_k2"] = 0.0
    else:
        aug["distortion_k1"] = float(filt.get("distortion_k1", 0.0))
        aug["distortion_k2"] = float(filt.get("distortion_k2", 0.0))

    if not filt.get("enable_jpeg_compression", True):
        aug["jpeg_quality"] = 100
    else:
        aug["jpeg_quality"] = int(filt.get("jpeg_quality", 85))

    if not filt.get("enable_shadow", True):
        aug["shadow_strength"] = 0.0
    else:
        aug["shadow_strength"] = float(filt.get("shadow_strength", 0.3))
        aug["shadow_size"] = float(filt.get("shadow_size", 0.2))

    if not filt.get("enable_reflection", True):
        aug["reflection_strength"] = 0.0
    else:
        aug["reflection_strength"] = float(filt.get("reflection_strength", 0.5))
        aug["reflection_size"] = float(filt.get("reflection_size", 0.15))

    if not filt.get("enable_dust", True):
        aug["dust_density"] = 0.0
    else:
        aug["dust_density"] = float(filt.get("dust_density", 0.3))
        aug["dust_size"] = float(filt.get("dust_size", 2.0))

    if not filt.get("enable_color_temperature", True):
        aug["color_temperature_kelvin"] = 5500
    else:
        aug["color_temperature_kelvin"] = int(filt.get("color_temperature_kelvin", 5500))

    return aug


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


def sample_augment_params(
    augment_config: Dict[str, Any],
    domain_config: Dict[str, Any],
    rng: np.random.Generator,
    *,
    enable_cardinal_rotation_90: bool = True,
) -> Dict[str, float]:
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

    # Global rotation augmentation:
    # Optionally sample a coarse orientation in 90-degree steps so models see all sides,
    # then add the configured fine jitter range.
    rot_min, rot_max = augment_config['rotation_deg_range']
    base_orientation_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0])) if enable_cardinal_rotation_90 else 0.0
    rotation_jitter_deg = float(rng.uniform(rot_min, rot_max))
    rotation_deg = base_orientation_deg + rotation_jitter_deg

    # Sample new filter parameters
    perspective_angle_x = float(rng.uniform(-10.0, 10.0))
    perspective_angle_y = float(rng.uniform(-10.0, 10.0))
    motion_blur_angle = float(rng.uniform(0.0, 360.0))
    saturation_factor = float(rng.uniform(0.8, 1.2))
    hue_shift_deg = float(rng.uniform(-15.0, 15.0))
    color_temperature_kelvin = int(rng.choice([3000, 4000, 5500, 6500]))

    return {
        # Existing parameters
        'blur_sigma': float(blur_sigma),
        'noise_stddev': float(noise_stddev),
        'brightness_factor': float(brightness_factor),
        'contrast_factor': float(contrast_factor),
        'rotation_deg': float(rotation_deg),

        # New filter parameters
        'perspective_angle_x': perspective_angle_x,
        'perspective_angle_y': perspective_angle_y,
        'motion_blur_angle': motion_blur_angle,
        'saturation_factor': saturation_factor,
        'hue_shift_deg': hue_shift_deg,
        'color_temperature_kelvin': color_temperature_kelvin,
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
    elif footprint == 'soic_16':
        # SOIC-16: 8 pads on left + 8 pads on right.
        comp_w = int(nominal.get('component_width', 0))
        pad_spacing_y = int(nominal.get('pad_spacing_y', 0))
        y_span = max(pad_width * 7, pad_spacing_y if pad_spacing_y > 0 else int(comp_w * 0.72))
        pitch = y_span / 7.0
        y0 = center_y - (y_span / 2.0)

        left_x = center_x - pad_spacing // 2 - pad_height // 2
        right_x = center_x + pad_spacing // 2 - pad_height // 2

        pads: List[Dict[str, int]] = []
        for i in range(8):
            cy = int(round(y0 + (i * pitch)))
            y = cy - pad_width // 2
            pads.append({'x': left_x, 'y': y, 'w': pad_height, 'h': pad_width})
            pads.append({'x': right_x, 'y': y, 'w': pad_height, 'h': pad_width})
        return pads
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


def apply_perspective_transform(img: np.ndarray, strength: float, angle_x: float, angle_y: float) -> np.ndarray:
    """Apply perspective transformation to simulate camera angles.

    Args:
        img: Input image
        strength: Overall strength multiplier (0.0-2.0)
        angle_x: Tilt around X-axis in degrees (-15 to +15)
        angle_y: Tilt around Y-axis in degrees (-15 to +15)

    Returns:
        Transformed image
    """
    if strength <= 0 or (abs(angle_x) < 0.1 and abs(angle_y) < 0.1):
        return img

    h, w = img.shape[:2]

    # Scale angles by strength
    angle_x = angle_x * strength
    angle_y = angle_y * strength

    # Convert to radians
    angle_x_rad = np.deg2rad(angle_x)
    angle_y_rad = np.deg2rad(angle_y)

    # Compute perspective transformation matrix
    # Create a simple perspective effect by shifting corners
    shift_x = w * 0.15 * np.tan(angle_y_rad)
    shift_y = h * 0.15 * np.tan(angle_x_rad)

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [shift_x, shift_y],
        [w - shift_x, shift_y],
        [w + shift_x, h - shift_y],
        [-shift_x, h - shift_y]
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return result


def apply_motion_blur(img: np.ndarray, strength: float, angle: float) -> np.ndarray:
    """Apply motion blur with directional component.

    Args:
        img: Input image
        strength: Blur strength (0.0-3.0)
        angle: Motion direction in degrees (0-360)

    Returns:
        Motion blurred image
    """
    if strength <= 0:
        return img

    # Kernel size based on strength
    ksize = int(strength * 10) + 1
    if ksize < 3:
        return img

    # Create motion blur kernel
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
    kernel = kernel / ksize

    # Rotate kernel to match angle
    center = (ksize // 2, ksize // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (ksize, ksize))

    # Apply kernel
    result = cv2.filter2D(img, -1, kernel)
    return result


def apply_chromatic_aberration(img: np.ndarray, strength: float) -> np.ndarray:
    """Apply chromatic aberration (color fringing).

    Args:
        img: Input image
        strength: Aberration strength (0.0-2.0)

    Returns:
        Image with chromatic aberration
    """
    if strength <= 0:
        return img

    h, w = img.shape[:2]
    shift = int(strength * 2)

    if shift == 0:
        return img

    # Split channels
    b, g, r = cv2.split(img)

    # Shift red and blue channels slightly
    M_r = np.float32([[1, 0, shift], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift], [0, 1, 0]])

    r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Merge back
    result = cv2.merge([b_shifted, g, r_shifted])
    return result


def apply_vignetting(img: np.ndarray, strength: float) -> np.ndarray:
    """Apply vignetting (edge darkening).

    Args:
        img: Input image
        strength: Vignetting strength (0.0-2.0)

    Returns:
        Vignetted image
    """
    if strength <= 0:
        return img

    h, w = img.shape[:2]

    # Create radial gradient mask
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]

    # Distance from center normalized
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = dist / max_dist

    # Create vignette mask (1.0 at center, darker at edges)
    vignette = 1.0 - (dist * strength * 0.7)
    vignette = np.clip(vignette, 0.2, 1.0)

    # Apply to all channels
    vignette = np.stack([vignette] * 3, axis=-1)
    result = (img.astype(np.float32) * vignette).astype(np.uint8)
    return result


def apply_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Apply saturation adjustment.

    Args:
        img: Input image (BGR)
        factor: Saturation multiplier (0.5-1.5, 1.0 = neutral)

    Returns:
        Adjusted image
    """
    if abs(factor - 1.0) < 1e-6:
        return img

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result


def apply_hue_shift(img: np.ndarray, shift_deg: float) -> np.ndarray:
    """Apply hue shift in HSV color space.

    Args:
        img: Input image (BGR)
        shift_deg: Hue shift in degrees (-30 to +30)

    Returns:
        Hue-shifted image
    """
    if abs(shift_deg) < 0.1:
        return img

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Shift hue (OpenCV uses 0-180 for hue)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_deg / 2.0) % 180

    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result


def apply_sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    """Apply sharpening filter.

    Args:
        img: Input image
        strength: Sharpening strength (0.0-2.0)

    Returns:
        Sharpened image
    """
    if strength <= 0:
        return img

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Scale kernel by strength
    kernel = (kernel - 1) * strength + 1
    kernel[1, 1] = 5 * strength

    # Normalize
    kernel = kernel / kernel.sum() * 5

    result = cv2.filter2D(img, -1, kernel)
    return result


def apply_lens_distortion(img: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Apply lens distortion (barrel/pincushion).

    Args:
        img: Input image
        k1: Radial distortion coefficient 1st order (-0.3 to +0.3)
        k2: Radial distortion coefficient 2nd order (-0.1 to +0.1)

    Returns:
        Distorted image
    """
    if abs(k1) < 1e-6 and abs(k2) < 1e-6:
        return img

    h, w = img.shape[:2]

    # Camera matrix (simple centered camera)
    fx = fy = w
    cx, cy = w // 2, h // 2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)

    # Undistort (which actually applies distortion when coeffs are inverted)
    result = cv2.undistort(img, camera_matrix, dist_coeffs)
    return result


def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression artifacts.

    Args:
        img: Input image
        quality: JPEG quality (50-95, lower = more artifacts)

    Returns:
        Compressed image
    """
    if quality >= 95:
        return img

    # Encode and decode as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result


def apply_shadow(img: np.ndarray, strength: float, size: float, rng: np.random.Generator) -> np.ndarray:
    """Apply shadow overlay.

    Args:
        img: Input image
        strength: Shadow darkness (0.0-0.8)
        size: Shadow size relative to image (0.1-0.5)
        rng: NumPy random generator

    Returns:
        Image with shadow
    """
    if strength <= 0 or size <= 0:
        return img

    h, w = img.shape[:2]

    # Random shadow position
    shadow_w = int(w * size)
    shadow_h = int(h * size)

    # Create shadow mask with gradient
    shadow_mask = np.ones((h, w), dtype=np.float32)

    # Random position for shadow
    x_pos = rng.integers(0, max(1, w - shadow_w))
    y_pos = rng.integers(0, max(1, h - shadow_h))

    # Create gradient shadow
    for i in range(shadow_h):
        for j in range(shadow_w):
            # Elliptical gradient
            dist = np.sqrt(((i - shadow_h/2) / (shadow_h/2))**2 +
                          ((j - shadow_w/2) / (shadow_w/2))**2)
            if dist < 1.0:
                shadow_val = 1.0 - (1.0 - dist) * strength
                if y_pos + i < h and x_pos + j < w:
                    shadow_mask[y_pos + i, x_pos + j] = min(shadow_mask[y_pos + i, x_pos + j], shadow_val)

    # Apply shadow
    shadow_mask = np.stack([shadow_mask] * 3, axis=-1)
    result = (img.astype(np.float32) * shadow_mask).astype(np.uint8)
    return result


def apply_reflection(img: np.ndarray, strength: float, size: float, rng: np.random.Generator) -> np.ndarray:
    """Apply reflection/glare overlay.

    Args:
        img: Input image
        strength: Reflection brightness (0.0-1.0)
        size: Reflection size relative to image (0.05-0.3)
        rng: NumPy random generator

    Returns:
        Image with reflection
    """
    if strength <= 0 or size <= 0:
        return img

    h, w = img.shape[:2]

    # Random reflection position
    refl_w = int(w * size)
    refl_h = int(h * size)

    # Create reflection overlay
    reflection = np.zeros((h, w), dtype=np.float32)

    # Random position
    x_pos = rng.integers(0, max(1, w - refl_w))
    y_pos = rng.integers(0, max(1, h - refl_h))

    # Create bright spot with gradient
    for i in range(refl_h):
        for j in range(refl_w):
            # Elliptical gradient
            dist = np.sqrt(((i - refl_h/2) / (refl_h/2))**2 +
                          ((j - refl_w/2) / (refl_w/2))**2)
            if dist < 1.0:
                refl_val = (1.0 - dist) * strength * 255
                if y_pos + i < h and x_pos + j < w:
                    reflection[y_pos + i, x_pos + j] = refl_val

    # Apply reflection
    reflection = np.stack([reflection] * 3, axis=-1)
    result = np.clip(img.astype(np.float32) + reflection, 0, 255).astype(np.uint8)
    return result


def apply_dust_particles(img: np.ndarray, density: float, size: float, rng: np.random.Generator) -> np.ndarray:
    """Apply dust/dirt particle overlay.

    Args:
        img: Input image
        density: Particle density (0.0-1.0)
        size: Average particle size in pixels (1-5)
        rng: NumPy random generator

    Returns:
        Image with dust particles
    """
    if density <= 0 or size <= 0:
        return img

    h, w = img.shape[:2]
    result = img.copy()

    # Number of particles based on density
    num_particles = int(w * h * density * 0.001)

    for _ in range(num_particles):
        # Random position
        x = rng.integers(0, w)
        y = rng.integers(0, h)

        # Random particle size
        particle_size = int(rng.uniform(size * 0.5, size * 1.5))

        # Random darkness (dust is usually dark)
        darkness = rng.uniform(0.3, 0.8)

        # Draw particle (small circle or irregular shape)
        if particle_size <= 1:
            if y < h and x < w:
                result[y, x] = (result[y, x].astype(np.float32) * darkness).astype(np.uint8)
        else:
            cv2.circle(result, (x, y), particle_size,
                      (result[min(y, h-1), min(x, w-1)].astype(np.float32) * darkness).astype(np.uint8).tolist(),
                      -1, lineType=cv2.LINE_AA)

    return result


def apply_color_temperature(img: np.ndarray, kelvin: int) -> np.ndarray:
    """Apply color temperature adjustment.

    Args:
        img: Input image (BGR)
        kelvin: Color temperature in Kelvin (2500-7500)

    Returns:
        Temperature-adjusted image
    """
    if kelvin == 5500:  # Neutral daylight
        return img

    # Simplified color temperature adjustment
    # Based on blackbody radiation approximation

    if kelvin < 5500:
        # Warm (more red/orange)
        factor = (5500 - kelvin) / 3000.0  # 0.0 to 1.0
        b_scale = 1.0 - factor * 0.3
        g_scale = 1.0 - factor * 0.1
        r_scale = 1.0
    else:
        # Cool (more blue)
        factor = (kelvin - 5500) / 2000.0  # 0.0 to 1.0
        b_scale = 1.0 + factor * 0.3
        g_scale = 1.0
        r_scale = 1.0 - factor * 0.2

    # Split and scale channels
    b, g, r = cv2.split(img.astype(np.float32))
    b = np.clip(b * b_scale, 0, 255)
    g = np.clip(g * g_scale, 0, 255)
    r = np.clip(r * r_scale, 0, 255)

    result = cv2.merge([b, g, r]).astype(np.uint8)
    return result


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

    # Apply basic augmentation (existing filters)
    img = apply_blur(img, augment.get('blur_sigma', 0.0))
    img = apply_noise(img, augment.get('noise_stddev', 0.0), rng)
    img = apply_brightness(img, augment.get('brightness_factor', 1.0))
    img = apply_contrast(img, augment.get('contrast_factor', 1.0))

    # Apply new filters (before rotation to avoid artifacts)
    # Color adjustments
    img = apply_saturation(img, augment.get('saturation_factor', 1.0))
    img = apply_hue_shift(img, augment.get('hue_shift_deg', 0.0))
    img = apply_color_temperature(img, augment.get('color_temperature_kelvin', 5500))

    # Optical effects
    img = apply_vignetting(img, augment.get('vignetting_strength', 0.0))
    img = apply_chromatic_aberration(img, augment.get('chromatic_strength', 0.0))
    img = apply_lens_distortion(img, augment.get('distortion_k1', 0.0), augment.get('distortion_k2', 0.0))

    # Motion and sharpness
    img = apply_motion_blur(img, augment.get('motion_blur_strength', 0.0), augment.get('motion_blur_angle', 0.0))
    img = apply_sharpen(img, augment.get('sharpen_strength', 0.0))

    # Overlays (shadow, reflection, dust)
    img = apply_shadow(img, augment.get('shadow_strength', 0.0), augment.get('shadow_size', 0.2), rng)
    img = apply_reflection(img, augment.get('reflection_strength', 0.0), augment.get('reflection_size', 0.15), rng)
    img = apply_dust_particles(img, augment.get('dust_density', 0.0), augment.get('dust_size', 2.0), rng)

    # Compression artifacts
    img = apply_jpeg_compression(img, augment.get('jpeg_quality', 100))

    # Geometric transforms (should be last before rotation)
    img = apply_perspective_transform(
        img,
        augment.get('perspective_strength', 0.0),
        augment.get('perspective_angle_x', 0.0),
        augment.get('perspective_angle_y', 0.0)
    )

    # Apply rotation augmentation (last step)
    if abs(augment.get('rotation_deg', 0.0)) > 0.1:
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, augment['rotation_deg'], 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return img
