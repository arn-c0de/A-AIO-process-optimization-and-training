"""Typed payload models for filter popup state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from .constants import DEFAULT_REALISM_PROFILE_ID, REALISM_GROUP_DEFAULTS, REALISM_K_DEFAULTS


def _to_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


@dataclass
class CustomFilterSettings:
    cardinal_rotation_90: bool = True
    enable_rotation: bool = True
    enable_blur: bool = True
    enable_grain: bool = True
    enable_brightness: bool = True
    enable_contrast: bool = True
    enable_perspective: bool = False
    enable_motion_blur: bool = True
    enable_saturation: bool = True
    enable_hue_shift: bool = True
    enable_shadow: bool = True
    enable_reflection: bool = False
    enable_vignetting: bool = False
    enable_chromatic_aberration: bool = False
    enable_jpeg_compression: bool = False
    enable_color_temperature: bool = False
    enable_lens_distortion: bool = False
    enable_dust: bool = False
    enable_sharpen: bool = False

    rotation_strength: float = 1.0
    blur_strength: float = 1.0
    grain_strength: float = 1.0
    brightness_strength: float = 1.0
    contrast_strength: float = 1.0
    perspective_strength: float = 1.0
    motion_blur_strength: float = 1.0
    saturation_factor: float = 1.0
    hue_shift_deg: float = 0.0
    shadow_strength: float = 0.3
    reflection_strength: float = 0.5
    vignetting_strength: float = 1.0
    chromatic_strength: float = 1.0
    distortion_k1: float = 0.0
    dust_density: float = 0.3
    sharpen_strength: float = 1.0

    jpeg_quality: int = 85
    color_temperature_kelvin: int = 5500


@dataclass
class RealismSettings:
    filter_mode: str = "custom"
    realism_enabled: bool = False
    realism_constraints_enabled: bool = True
    realism_profile_id: str = DEFAULT_REALISM_PROFILE_ID
    realism_k_prob_0: float = REALISM_K_DEFAULTS["realism_k_prob_0"]
    realism_k_prob_1: float = REALISM_K_DEFAULTS["realism_k_prob_1"]
    realism_k_prob_2: float = REALISM_K_DEFAULTS["realism_k_prob_2"]
    realism_group_G1_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G1_prob"]
    realism_group_G2_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G2_prob"]
    realism_group_G3_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G3_prob"]
    realism_group_G4_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G4_prob"]
    realism_group_G5_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G5_prob"]
    realism_group_G6_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G6_prob"]
    realism_group_G7_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G7_prob"]
    realism_group_G8_prob: float = REALISM_GROUP_DEFAULTS["realism_group_G8_prob"]


@dataclass
class FilterPopupPayload:
    custom: CustomFilterSettings = field(default_factory=CustomFilterSettings)
    realism: RealismSettings = field(default_factory=RealismSettings)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def known_keys(cls) -> set[str]:
        sample = cls()
        return set(sample.to_mapping().keys())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FilterPopupPayload":
        custom = CustomFilterSettings(
            cardinal_rotation_90=_to_bool(data.get("cardinal_rotation_90", True), True),
            enable_rotation=_to_bool(data.get("enable_rotation", True), True),
            enable_blur=_to_bool(data.get("enable_blur", True), True),
            enable_grain=_to_bool(data.get("enable_grain", True), True),
            enable_brightness=_to_bool(data.get("enable_brightness", True), True),
            enable_contrast=_to_bool(data.get("enable_contrast", True), True),
            enable_perspective=_to_bool(data.get("enable_perspective", False), False),
            enable_motion_blur=_to_bool(data.get("enable_motion_blur", True), True),
            enable_saturation=_to_bool(data.get("enable_saturation", True), True),
            enable_hue_shift=_to_bool(data.get("enable_hue_shift", True), True),
            enable_shadow=_to_bool(data.get("enable_shadow", True), True),
            enable_reflection=_to_bool(data.get("enable_reflection", False), False),
            enable_vignetting=_to_bool(data.get("enable_vignetting", False), False),
            enable_chromatic_aberration=_to_bool(data.get("enable_chromatic_aberration", False), False),
            enable_jpeg_compression=_to_bool(data.get("enable_jpeg_compression", False), False),
            enable_color_temperature=_to_bool(data.get("enable_color_temperature", False), False),
            enable_lens_distortion=_to_bool(data.get("enable_lens_distortion", False), False),
            enable_dust=_to_bool(data.get("enable_dust", False), False),
            enable_sharpen=_to_bool(data.get("enable_sharpen", False), False),
            rotation_strength=_to_float(data.get("rotation_strength", 1.0), 1.0),
            blur_strength=_to_float(data.get("blur_strength", 1.0), 1.0),
            grain_strength=_to_float(data.get("grain_strength", 1.0), 1.0),
            brightness_strength=_to_float(data.get("brightness_strength", 1.0), 1.0),
            contrast_strength=_to_float(data.get("contrast_strength", 1.0), 1.0),
            perspective_strength=_to_float(data.get("perspective_strength", 1.0), 1.0),
            motion_blur_strength=_to_float(data.get("motion_blur_strength", 1.0), 1.0),
            saturation_factor=_to_float(data.get("saturation_factor", 1.0), 1.0),
            hue_shift_deg=_to_float(data.get("hue_shift_deg", 0.0), 0.0),
            shadow_strength=_to_float(data.get("shadow_strength", 0.3), 0.3),
            reflection_strength=_to_float(data.get("reflection_strength", 0.5), 0.5),
            vignetting_strength=_to_float(data.get("vignetting_strength", 1.0), 1.0),
            chromatic_strength=_to_float(data.get("chromatic_strength", 1.0), 1.0),
            distortion_k1=_to_float(data.get("distortion_k1", 0.0), 0.0),
            dust_density=_to_float(data.get("dust_density", 0.3), 0.3),
            sharpen_strength=_to_float(data.get("sharpen_strength", 1.0), 1.0),
            jpeg_quality=_to_int(data.get("jpeg_quality", 85), 85),
            color_temperature_kelvin=_to_int(data.get("color_temperature_kelvin", 5500), 5500),
        )
        realism = RealismSettings(
            filter_mode=str(data.get("filter_mode", "custom") or "custom"),
            realism_enabled=_to_bool(data.get("realism_enabled", False), False),
            realism_constraints_enabled=_to_bool(data.get("realism_constraints_enabled", True), True),
            realism_profile_id=str(data.get("realism_profile_id", DEFAULT_REALISM_PROFILE_ID) or DEFAULT_REALISM_PROFILE_ID),
            realism_k_prob_0=_to_float(data.get("realism_k_prob_0", REALISM_K_DEFAULTS["realism_k_prob_0"]), REALISM_K_DEFAULTS["realism_k_prob_0"]),
            realism_k_prob_1=_to_float(data.get("realism_k_prob_1", REALISM_K_DEFAULTS["realism_k_prob_1"]), REALISM_K_DEFAULTS["realism_k_prob_1"]),
            realism_k_prob_2=_to_float(data.get("realism_k_prob_2", REALISM_K_DEFAULTS["realism_k_prob_2"]), REALISM_K_DEFAULTS["realism_k_prob_2"]),
            realism_group_G1_prob=_to_float(data.get("realism_group_G1_prob", REALISM_GROUP_DEFAULTS["realism_group_G1_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G1_prob"]),
            realism_group_G2_prob=_to_float(data.get("realism_group_G2_prob", REALISM_GROUP_DEFAULTS["realism_group_G2_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G2_prob"]),
            realism_group_G3_prob=_to_float(data.get("realism_group_G3_prob", REALISM_GROUP_DEFAULTS["realism_group_G3_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G3_prob"]),
            realism_group_G4_prob=_to_float(data.get("realism_group_G4_prob", REALISM_GROUP_DEFAULTS["realism_group_G4_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G4_prob"]),
            realism_group_G5_prob=_to_float(data.get("realism_group_G5_prob", REALISM_GROUP_DEFAULTS["realism_group_G5_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G5_prob"]),
            realism_group_G6_prob=_to_float(data.get("realism_group_G6_prob", REALISM_GROUP_DEFAULTS["realism_group_G6_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G6_prob"]),
            realism_group_G7_prob=_to_float(data.get("realism_group_G7_prob", REALISM_GROUP_DEFAULTS["realism_group_G7_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G7_prob"]),
            realism_group_G8_prob=_to_float(data.get("realism_group_G8_prob", REALISM_GROUP_DEFAULTS["realism_group_G8_prob"]), REALISM_GROUP_DEFAULTS["realism_group_G8_prob"]),
        )
        known = cls.known_keys()
        extras = {k: v for k, v in data.items() if isinstance(k, str) and k not in known}
        return cls(custom=custom, realism=realism, extras=extras)

    def to_mapping(self) -> Dict[str, Any]:
        c = self.custom
        r = self.realism
        out: Dict[str, Any] = {
            "cardinal_rotation_90": bool(c.cardinal_rotation_90),
            "enable_rotation": bool(c.enable_rotation),
            "enable_blur": bool(c.enable_blur),
            "enable_grain": bool(c.enable_grain),
            "enable_brightness": bool(c.enable_brightness),
            "enable_contrast": bool(c.enable_contrast),
            "enable_perspective": bool(c.enable_perspective),
            "enable_motion_blur": bool(c.enable_motion_blur),
            "enable_saturation": bool(c.enable_saturation),
            "enable_hue_shift": bool(c.enable_hue_shift),
            "enable_shadow": bool(c.enable_shadow),
            "enable_reflection": bool(c.enable_reflection),
            "enable_vignetting": bool(c.enable_vignetting),
            "enable_chromatic_aberration": bool(c.enable_chromatic_aberration),
            "enable_jpeg_compression": bool(c.enable_jpeg_compression),
            "enable_color_temperature": bool(c.enable_color_temperature),
            "enable_lens_distortion": bool(c.enable_lens_distortion),
            "enable_dust": bool(c.enable_dust),
            "enable_sharpen": bool(c.enable_sharpen),
            "rotation_strength": f"{c.rotation_strength:.2f}",
            "blur_strength": f"{c.blur_strength:.2f}",
            "grain_strength": f"{c.grain_strength:.2f}",
            "brightness_strength": f"{c.brightness_strength:.2f}",
            "contrast_strength": f"{c.contrast_strength:.2f}",
            "perspective_strength": f"{c.perspective_strength:.2f}",
            "motion_blur_strength": f"{c.motion_blur_strength:.2f}",
            "saturation_factor": f"{c.saturation_factor:.2f}",
            "hue_shift_deg": f"{c.hue_shift_deg:.2f}",
            "shadow_strength": f"{c.shadow_strength:.2f}",
            "reflection_strength": f"{c.reflection_strength:.2f}",
            "vignetting_strength": f"{c.vignetting_strength:.2f}",
            "chromatic_strength": f"{c.chromatic_strength:.2f}",
            "distortion_k1": f"{c.distortion_k1:.2f}",
            "dust_density": f"{c.dust_density:.2f}",
            "sharpen_strength": f"{c.sharpen_strength:.2f}",
            "jpeg_quality": str(int(c.jpeg_quality)),
            "color_temperature_kelvin": str(int(c.color_temperature_kelvin)),
            "filter_mode": str(r.filter_mode),
            "realism_enabled": bool(r.realism_enabled),
            "realism_constraints_enabled": bool(r.realism_constraints_enabled),
            "realism_profile_id": str(r.realism_profile_id),
            "realism_k_prob_0": f"{r.realism_k_prob_0:.4f}",
            "realism_k_prob_1": f"{r.realism_k_prob_1:.4f}",
            "realism_k_prob_2": f"{r.realism_k_prob_2:.4f}",
            "realism_group_G1_prob": f"{r.realism_group_G1_prob:.4f}",
            "realism_group_G2_prob": f"{r.realism_group_G2_prob:.4f}",
            "realism_group_G3_prob": f"{r.realism_group_G3_prob:.4f}",
            "realism_group_G4_prob": f"{r.realism_group_G4_prob:.4f}",
            "realism_group_G5_prob": f"{r.realism_group_G5_prob:.4f}",
            "realism_group_G6_prob": f"{r.realism_group_G6_prob:.4f}",
            "realism_group_G7_prob": f"{r.realism_group_G7_prob:.4f}",
            "realism_group_G8_prob": f"{r.realism_group_G8_prob:.4f}",
        }
        out.update(self.extras)
        return out
