"""Augmentation sampling and filter-override logic for OpenCV 2D renderer."""

from __future__ import annotations

from typing import Any

import numpy as np

from simple_sim.filter_settings import NEUTRAL_AUGMENT, normalize_image_filters
from .realism import RealismPresets, apply_realism_groups


def apply_image_filter_overrides(
    augment: dict[str, float],
    image_filters: dict[str, Any] | None,
    rng: np.random.Generator | None = None,
    *,
    realism_presets: RealismPresets | None = None,
) -> dict[str, float]:
    """Apply toggle + strength overrides to sampled augment params.

    *realism_presets*: group probabilities of the realism mode (default: the preset file, see
    :mod:`simple_sim.generators.opencv2d.realism`).

    Pass the per-sample ``rng`` to keep randomized filters reproducible; without it a
    fresh unseeded generator is used (e.g. for interactive previews).
    """
    if rng is None:
        rng = np.random.default_rng()
    aug = dict(augment or {})
    src_keys = set(image_filters.keys()) if isinstance(image_filters, dict) else set()
    filt_base = normalize_image_filters(image_filters)
    if not filt_base.get("enable", True):
        aug.update(NEUTRAL_AUGMENT)
        return aug

    filt = dict(filt_base)
    filt["__source_keys"] = src_keys
    if str(filt.get("filter_mode", "custom")).strip().lower() == "realism" and bool(filt.get("realism_enabled", False)):
        return apply_realism_groups(aug, filt, rng, presets=realism_presets)

    st = _sample_strengths(filt, rng)

    def off(name: str) -> bool:
        return not filt.get(f"enable_{name}", True)

    def fixed_angle(key: str) -> float:
        # An angle set in the filter settings (non-zero) replaces the per-sample random one; 0 (the
        # default) keeps the sampled angle.
        value = float(filt.get(key, 0.0) or 0.0) if key in src_keys else 0.0
        return value if value != 0.0 else float(aug.get(key, 0.0))

    def about_one(key: str, strength: float) -> float:
        return 1.0 + (float(aug.get(key, 1.0)) - 1.0) * strength

    aug["blur_sigma"] = 0.0 if off("blur") else float(aug.get("blur_sigma", 0.0)) * st["blur_strength"]
    aug["noise_stddev"] = 0.0 if off("grain") else float(aug.get("noise_stddev", 0.0)) * st["grain_strength"]
    aug["brightness_factor"] = 1.0 if off("brightness") else about_one("brightness_factor", st["brightness_strength"])
    aug["contrast_factor"] = 1.0 if off("contrast") else about_one("contrast_factor", st["contrast_strength"])
    aug["rotation_deg"] = _rotation_override(float(aug.get("rotation_deg", 0.0)), filt, off("rotation"), st["rotation_strength"])

    if off("perspective"):
        aug["perspective_strength"] = 0.0
    else:
        aug["perspective_strength"] = st["perspective_strength"]
        aug["perspective_angle_x"] = fixed_angle("perspective_angle_x")
        aug["perspective_angle_y"] = fixed_angle("perspective_angle_y")

    if off("motion_blur"):
        aug["motion_blur_strength"] = 0.0
    else:
        aug["motion_blur_strength"] = st["motion_blur_strength"]
        aug["motion_blur_angle"] = fixed_angle("motion_blur_angle")

    aug["chromatic_strength"] = 0.0 if off("chromatic_aberration") else st["chromatic_strength"]
    aug["vignetting_strength"] = 0.0 if off("vignetting") else st["vignetting_strength"]
    aug["saturation_factor"] = 1.0 if off("saturation") else float(aug.get("saturation_factor", 1.0)) * st["saturation_factor"]
    aug["hue_shift_deg"] = 0.0 if off("hue_shift") else float(aug.get("hue_shift_deg", 0.0)) + st["hue_shift_deg"]
    aug["sharpen_strength"] = 0.0 if off("sharpen") else st["sharpen_strength"]

    if off("lens_distortion"):
        aug["distortion_k1"] = 0.0
        aug["distortion_k2"] = 0.0
    else:
        aug["distortion_k1"] = st["distortion_k1"]
        aug["distortion_k2"] = float(filt.get("distortion_k2", 0.0))

    aug["jpeg_quality"] = 100 if off("jpeg_compression") else st["jpeg_quality"]

    # strength plus its size setting (the size only when the filter is on)
    for name, size_key, size_default in (("shadow", "shadow_size", 0.2), ("reflection", "reflection_size", 0.15), ("dust", "dust_size", 2.0)):
        key = "dust_density" if name == "dust" else f"{name}_strength"
        if off(name):
            aug[key] = 0.0
        else:
            aug[key] = st[key]
            aug[size_key] = float(filt.get(size_key, size_default))

    aug["color_temperature_kelvin"] = 5500 if off("color_temperature") else st["color_temperature_kelvin"]
    return aug


#: per-image filter strengths ``(key, default, integer)``. With ``<key>_randomize`` each is drawn from
#: ``[<key>_min, <key>_max]``; they are drawn in this order for every image, whichever filters are on, so the rng
#: stream (and with it every later random choice of the sample) does not depend on the filter toggles.
SAMPLED_STRENGTHS: tuple[tuple[str, float, bool], ...] = (
    ("rotation_strength", 1.0, False), ("blur_strength", 1.0, False), ("grain_strength", 1.0, False),
    ("brightness_strength", 1.0, False), ("contrast_strength", 1.0, False), ("perspective_strength", 1.0, False),
    ("motion_blur_strength", 1.0, False), ("saturation_factor", 1.0, False), ("hue_shift_deg", 0.0, False),
    ("shadow_strength", 0.3, False), ("reflection_strength", 0.5, False), ("vignetting_strength", 1.0, False),
    ("chromatic_strength", 1.0, False), ("jpeg_quality", 85, True), ("color_temperature_kelvin", 5500, True),
    ("distortion_k1", 0.0, False), ("dust_density", 0.3, False), ("sharpen_strength", 1.0, False),
)


def _sample_strengths(filt: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    """Every :data:`SAMPLED_STRENGTHS` value of *filt* for one image (fixed, or drawn when randomized)."""
    out: dict[str, float] = {}
    for key, default, integer in SAMPLED_STRENGTHS:
        cast = int if integer else float
        base = cast(filt.get(key, default))
        if not bool(filt.get(f"{key}_randomize", False)):
            out[key] = base
            continue
        lo, hi = cast(filt.get(f"{key}_min", base)), cast(filt.get(f"{key}_max", base))
        lo, hi = min(lo, hi), max(lo, hi)
        out[key] = int(rng.integers(lo, hi + 1)) if integer else float(rng.uniform(lo, hi))
    return out


def _rotation_override(rot: float, filt: dict[str, Any], disabled: bool, strength: float) -> float:
    """The sampled rotation after the rotation filter: off snaps to the cardinal angle (or 0 without
    ``cardinal_rotation_90``); on scales only the jitter, the cardinal 0/90/180/270 base orientation stays intact."""
    if disabled:
        return float(round(rot / 90.0) * 90.0) if bool(filt.get("cardinal_rotation_90", True)) else 0.0
    base = 90.0 * round(rot / 90.0)
    return base + (rot - base) * strength


CARDINAL_ROTATIONS_DEG = (0.0, 90.0, 180.0, 270.0)

#: The augment keys of a clean image (previews, galleries): neutral values of ``NEUTRAL_AUGMENT``.
CLEAN_AUGMENT_KEYS = ("blur_sigma", "noise_stddev", "brightness_factor", "contrast_factor")


def sample_rotation_deg(rng: np.random.Generator, rotation_range: tuple[float, float], *,
                        enable_cardinal_rotation_90: bool = True) -> float:
    """Global in-plane orientation: a cardinal base (one ``rng.choice`` of 0/90/180/270, skipped
    without *enable_cardinal_rotation_90*) plus one uniform jitter draw in *rotation_range*."""
    base_orientation_deg = float(rng.choice(list(CARDINAL_ROTATIONS_DEG))) if enable_cardinal_rotation_90 else 0.0
    rotation_jitter_deg = float(rng.uniform(rotation_range[0], rotation_range[1]))
    return float(base_orientation_deg + rotation_jitter_deg)


def clean_augment(rotation_deg: float = 0.0) -> dict[str, float]:
    """Augment of a clean image (no blur, grain, brightness or contrast change) at *rotation_deg*."""
    return {**{key: NEUTRAL_AUGMENT[key] for key in CLEAN_AUGMENT_KEYS}, "rotation_deg": float(rotation_deg)}


def sample_augment_params(
    augment_config: dict[str, Any],
    domain_config: dict[str, Any],
    rng: np.random.Generator,
    *,
    enable_cardinal_rotation_90: bool = True,
) -> dict[str, float]:
    """Sample augmentation parameters."""
    blur_min, blur_max = domain_config["blur_sigma"]
    noise_min, noise_max = domain_config["noise_stddev"]
    bright_min, bright_max = domain_config["lighting_brightness"]
    contrast_min, contrast_max = augment_config["contrast_factor_range"]
    rot_min, rot_max = augment_config["rotation_deg_range"]

    rotation_deg = sample_rotation_deg(rng, (rot_min, rot_max), enable_cardinal_rotation_90=enable_cardinal_rotation_90)

    return {
        "blur_sigma": float(rng.uniform(blur_min, blur_max)),
        "noise_stddev": float(rng.uniform(noise_min, noise_max)),
        "brightness_factor": float(rng.uniform(bright_min, bright_max)),
        "contrast_factor": float(rng.uniform(contrast_min, contrast_max)),
        "rotation_deg": rotation_deg,
        "perspective_angle_x": float(rng.uniform(-10.0, 10.0)),
        "perspective_angle_y": float(rng.uniform(-10.0, 10.0)),
        "motion_blur_angle": float(rng.uniform(0.0, 360.0)),
        # Mild colour jitter: +-15 deg hue turned the olive solder mask into lime or cyan, which no
        # real board/camera produces (real crops: hue ~135-150 deg, see photoreal.palette.MASK_PALETTE).
        "saturation_factor": float(rng.uniform(0.85, 1.1)),
        "hue_shift_deg": float(rng.uniform(-5.0, 5.0)),
        "color_temperature_kelvin": int(rng.choice([3000, 4000, 5500, 6500])),
    }
