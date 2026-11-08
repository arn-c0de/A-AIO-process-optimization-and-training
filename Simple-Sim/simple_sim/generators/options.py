"""``render.*`` option constants and the camera validator, without the renderer (plan I7).

``simple_sim.config.validate_config`` checks render options against these, so validating a
config does not import cv2 or the photoreal renderer. The renderer modules currently keep their
own definitions with the same values (``photoreal.context.SUPERSAMPLING_CHOICES``,
``photoreal.meniscus.SOLDER_STYLES``, ``photoreal.look.LOOKS``, ``opencv2d.render.POST_FILTER_MODES``,
``photoreal.camera.CAMERA_DEFAULTS`` / ``CAMERA_PRESETS`` / ``validate_camera_config``);
``tests/test_generator_options.py`` fails when the two drift apart.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CAMERA_DEFAULTS", "CAMERA_PRESET_NAMES", "CAMERA_RANGE_LIMITS", "CHANNEL_AXES", "DEFAULT_CAMERA_PRESET",
    "DEFAULT_LOOK", "DEFAULT_POST_FILTERS", "DEFAULT_SOLDER_STYLE", "DEFAULT_SUPERSAMPLING", "LOOKS", "MISSING_RESIDUE_MAX",
    "POST_FILTER_MODES", "SOLDER_STYLES", "SUPERSAMPLING_CHOICES", "validate_camera_config",
]

#: ``render.supersampling``: photoreal render scale (4 = "ultra": 16 sub-pixels per camera pixel).
SUPERSAMPLING_CHOICES = (1, 2, 3, 4)
DEFAULT_SUPERSAMPLING = 2

#: ``render.solder``: solder joint model.
SOLDER_STYLES = ("legacy", "meniscus")
DEFAULT_SOLDER_STYLE = "legacy"

#: ``render.look``: registered target looks, in registration order.
LOOKS = ("legacy", "quins", "dv_top", "dv_bottom")
DEFAULT_LOOK = "legacy"

#: ``render.missing_residue`` (2D photoreal): factor on the reflowed-paste domes of pads without a part
#: (1.0 = the look's default, 0 = bare pads); values above this are rejected.
MISSING_RESIDUE_MAX = 4.0

#: ``render.post_filters`` (2D photoreal): where the domain blur/noise and the rotation are applied.
POST_FILTER_MODES = ("legacy", "camera")
DEFAULT_POST_FILTERS = "legacy"

#: ``render.camera.preset`` names (the parameters live in ``photoreal.camera.CAMERA_PRESETS``).
CAMERA_PRESET_NAMES = ("aoi", "phone", "quins", "dv_top", "dv_bottom")
DEFAULT_CAMERA_PRESET = "aoi"

#: ``render.camera`` keys and their defaults (same as ``photoreal.camera.CAMERA_DEFAULTS``).
CAMERA_DEFAULTS: dict[str, Any] = {
    "blur_sigma": [0.55, 0.95],
    "read_noise": [1.0, 4.0],
    "shot_noise": 0.04,
    "tilt_deg": [0.0, 0.0],
    "zoom": [1.0, 1.0],
    "defocus": [0.0, 0.0],
    "sharpen": [0.0, 0.0],
    "jpeg_quality": [100, 100],
    "capture_scale": [1.0, 1.0],
    "chroma_shift": [0.0, 0.0],
    "sensor_px_per_mm": [0.0, 0.0],
    "channel_shift": [0.0, 0.0],
    "channel_axis": "random",
    "crop_part_lengths": [0.0, 0.0],
    "max_oversampling": 0.0,
}

CHANNEL_AXES = ("random", "x", "y")
CAMERA_RANGE_LIMITS = {"tilt_deg": 60.0, "jpeg_quality": 100.0, "capture_scale": 1.0}


def validate_camera_config(camera: Any) -> None:
    """Raise ``ValueError`` unless *camera* is a valid ``render.camera`` mapping (preset + overrides).

    Same checks and messages as ``photoreal.camera.validate_camera_config``.
    """
    if camera is None:
        return
    if not isinstance(camera, dict) or set(camera) - set(CAMERA_DEFAULTS) - {'preset'}:
        raise ValueError(f"render.camera must be a mapping with keys from {sorted(set(CAMERA_DEFAULTS) | {'preset'})}")
    if camera.get('preset', DEFAULT_CAMERA_PRESET) not in CAMERA_PRESET_NAMES:
        raise ValueError(f"render.camera.preset must be one of {sorted(CAMERA_PRESET_NAMES)}")
    for key, default in CAMERA_DEFAULTS.items():
        if key not in camera or not isinstance(default, list):
            continue
        rng = camera[key]
        limit = CAMERA_RANGE_LIMITS.get(key, float('inf'))
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng)
                and 0 <= rng[0] <= rng[1] <= limit):
            raise ValueError(f"render.camera.{key} must be [min, max] with 0 <= min <= max <= {CAMERA_RANGE_LIMITS.get(key, 'inf')}")
    if 'capture_scale' in camera and camera['capture_scale'][0] < 0.05:
        raise ValueError("render.camera.capture_scale must be in [0.05, 1] (sensor pixels per ROI pixel)")
    if 'zoom' in camera and camera['zoom'][0] < 1:
        raise ValueError("render.camera.zoom must be >= 1 (1 = profile scale, larger = farther away)")
    if 'shot_noise' in camera and not (isinstance(camera['shot_noise'], (int, float)) and camera['shot_noise'] >= 0):
        raise ValueError("render.camera.shot_noise must be a number >= 0")
    if 'channel_axis' in camera and camera['channel_axis'] not in CHANNEL_AXES:
        raise ValueError(f"render.camera.channel_axis must be one of {list(CHANNEL_AXES)}")
    if 'max_oversampling' in camera and not (isinstance(camera['max_oversampling'], (int, float))
                                             and camera['max_oversampling'] >= 0):
        raise ValueError("render.camera.max_oversampling must be a number >= 0 (0 = off)")
