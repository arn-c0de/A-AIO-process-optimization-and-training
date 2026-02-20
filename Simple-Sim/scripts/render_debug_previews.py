#!/usr/bin/env python3
"""Render small per-profile preview images for debugging 2D/3D profiles.

Goal: quickly verify that components/pads/defect states render correctly before AI training.

This script:
- Detects available profiles under `configs/profiles/` (2D: "opencv_2d", 3D: "blender_3d")
- Lets you choose backend mode (2D, 3D, or both), then select one or more profiles
- Renders exactly 1 image per defect state (OK/MISALIGNED/MISSING/TOMBSTONE, etc.)
- Writes previews under `<out>/previews/<profile_id>/<backend>/...` and an `<out>/index.html`

Examples:
  .venv/bin/python scripts/render_debug_previews.py
  .venv/bin/python scripts/render_debug_previews.py --backend opencv_2d --all
  .venv/bin/python scripts/render_debug_previews.py --backend both --profiles chip_0603_resistor@1,chip_0603_resistor_3d@1
  .venv/bin/python scripts/render_debug_previews.py --all
  .venv/bin/python scripts/render_debug_previews.py --profiles chip_0603_resistor_3d@1,sot23_transistor_3d@1
  .venv/bin/python scripts/render_debug_previews.py --config configs/run_qfn32_3d.yaml --profiles qfn32_ic_3d@1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Add Simple-Sim root to sys.path, like other scripts.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.components.filter_popup import open_filter_popup
from gui.utils.filter_profile_store import is_filter_popup_extra_key, parse_bool_like

from simple_sim.config import load_config, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generators.blender_3d import write_jobs_jsonl, render_blender_batch
from simple_sim.profile_hash import load_profile


BACKEND_2D = "opencv_2d"
BACKEND_3D = "blender_3d"
BACKEND_CHOICES = [BACKEND_2D, BACKEND_3D, "both", "auto"]

# (profile_id, backend, defect, rel_image_path)
PreviewRec = Tuple[str, str, str, str]


@dataclass(frozen=True)
class RenderSettings:
    roi_width_px: int
    roi_height_px: int
    mm_per_px: float
    blender_executable: str
    cycles_samples: int
    device: str


def _shared_settings_path(sim_root: Path, explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return (Path(sim_root) / "outputs" / "gui" / "settings.json").resolve()


def _read_settings_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_settings_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalize_filter_settings(d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    src = d or {}

    def _bool(name: str, default: bool) -> bool:
        v = src.get(name, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _f(name: str, default: float, lo: float = 0.0, hi: float = 4.0) -> float:
        try:
            x = float(src.get(name, default))
        except Exception:
            x = float(default)
        return max(lo, min(hi, x))

    def _s(name: str, default: str) -> str:
        return str(src.get(name, default) or default)

    return {
        # Existing filters
        "cardinal_rotation_90": _bool("cardinal_rotation_90", True),
        "enable_rotation": _bool("enable_rotation", True),
        "enable_blur": _bool("enable_blur", True),
        "enable_grain": _bool("enable_grain", True),
        "enable_brightness": _bool("enable_brightness", True),
        "enable_contrast": _bool("enable_contrast", True),
        "rotation_strength": _f("rotation_strength", 1.0),
        "blur_strength": _f("blur_strength", 1.0),
        "grain_strength": _f("grain_strength", 1.0),
        "brightness_strength": _f("brightness_strength", 1.0),
        "contrast_strength": _f("contrast_strength", 1.0),
        # High priority new filters
        "enable_perspective": _bool("enable_perspective", False),
        "perspective_strength": _f("perspective_strength", 1.0, 0.0, 2.0),
        "enable_motion_blur": _bool("enable_motion_blur", True),
        "motion_blur_strength": _f("motion_blur_strength", 1.0, 0.0, 3.0),
        "enable_saturation": _bool("enable_saturation", True),
        "saturation_factor": _f("saturation_factor", 1.0, 0.5, 1.5),
        "enable_hue_shift": _bool("enable_hue_shift", True),
        "hue_shift_deg": _f("hue_shift_deg", 0.0, -30.0, 30.0),
        "enable_shadow": _bool("enable_shadow", True),
        "shadow_strength": _f("shadow_strength", 0.3, 0.0, 0.8),
        "enable_reflection": _bool("enable_reflection", False),
        "reflection_strength": _f("reflection_strength", 0.5, 0.0, 1.0),
        # Medium/Low priority new filters
        "enable_vignetting": _bool("enable_vignetting", False),
        "vignetting_strength": _f("vignetting_strength", 1.0, 0.0, 2.0),
        "enable_chromatic_aberration": _bool("enable_chromatic_aberration", False),
        "chromatic_strength": _f("chromatic_strength", 1.0, 0.0, 2.0),
        "enable_jpeg_compression": _bool("enable_jpeg_compression", False),
        "jpeg_quality": int(_f("jpeg_quality", 85, 50, 95)),
        "enable_color_temperature": _bool("enable_color_temperature", False),
        "color_temperature_kelvin": int(_f("color_temperature_kelvin", 5500, 2500, 7500)),
        "enable_lens_distortion": _bool("enable_lens_distortion", False),
        "distortion_k1": _f("distortion_k1", 0.0, -0.3, 0.3),
        "enable_dust": _bool("enable_dust", False),
        "dust_density": _f("dust_density", 0.3, 0.0, 1.0),
        "enable_sharpen": _bool("enable_sharpen", False),
        "sharpen_strength": _f("sharpen_strength", 1.0, 0.0, 2.0),
        # Realism mode
        "filter_mode": _s("filter_mode", "custom"),
        "realism_enabled": _bool("realism_enabled", False),
        "realism_constraints_enabled": _bool("realism_constraints_enabled", True),
        "realism_profile_id": _s("realism_profile_id", "profile_industrial_cam"),
        "realism_k_prob_0": _f("realism_k_prob_0", 0.60, 0.0, 1.0),
        "realism_k_prob_1": _f("realism_k_prob_1", 0.35, 0.0, 1.0),
        "realism_k_prob_2": _f("realism_k_prob_2", 0.05, 0.0, 1.0),
        "realism_group_G1_prob": _f("realism_group_G1_prob", 0.35, 0.0, 1.0),
        "realism_group_G2_prob": _f("realism_group_G2_prob", 0.15, 0.0, 1.0),
        "realism_group_G3_prob": _f("realism_group_G3_prob", 0.20, 0.0, 1.0),
        "realism_group_G4_prob": _f("realism_group_G4_prob", 0.25, 0.0, 1.0),
        "realism_group_G5_prob": _f("realism_group_G5_prob", 0.12, 0.0, 1.0),
        "realism_group_G6_prob": _f("realism_group_G6_prob", 0.18, 0.0, 1.0),
        "realism_group_G7_prob": _f("realism_group_G7_prob", 0.10, 0.0, 1.0),
        "realism_group_G8_prob": _f("realism_group_G8_prob", 0.08, 0.0, 1.0),
    }


def _filter_settings_to_profile_store(d: Dict[str, Any]) -> Dict[str, Any]:
    n = _normalize_filter_settings(d)
    out = {
        # Existing filters
        "cardinal_rotation_90": bool(n["cardinal_rotation_90"]),
        "enable_rotation": bool(n["enable_rotation"]),
        "enable_blur": bool(n["enable_blur"]),
        "enable_grain": bool(n["enable_grain"]),
        "enable_brightness": bool(n["enable_brightness"]),
        "enable_contrast": bool(n["enable_contrast"]),
        "rotation_strength": f"{float(n['rotation_strength']):.2f}",
        "blur_strength": f"{float(n['blur_strength']):.2f}",
        "grain_strength": f"{float(n['grain_strength']):.2f}",
        "brightness_strength": f"{float(n['brightness_strength']):.2f}",
        "contrast_strength": f"{float(n['contrast_strength']):.2f}",
        # High priority new filters
        "enable_perspective": bool(n["enable_perspective"]),
        "perspective_strength": f"{float(n['perspective_strength']):.2f}",
        "enable_motion_blur": bool(n["enable_motion_blur"]),
        "motion_blur_strength": f"{float(n['motion_blur_strength']):.2f}",
        "enable_saturation": bool(n["enable_saturation"]),
        "saturation_factor": f"{float(n['saturation_factor']):.2f}",
        "enable_hue_shift": bool(n["enable_hue_shift"]),
        "hue_shift_deg": f"{float(n['hue_shift_deg']):.2f}",
        "enable_shadow": bool(n["enable_shadow"]),
        "shadow_strength": f"{float(n['shadow_strength']):.2f}",
        "enable_reflection": bool(n["enable_reflection"]),
        "reflection_strength": f"{float(n['reflection_strength']):.2f}",
        # Medium/Low priority new filters
        "enable_vignetting": bool(n["enable_vignetting"]),
        "vignetting_strength": f"{float(n['vignetting_strength']):.2f}",
        "enable_chromatic_aberration": bool(n["enable_chromatic_aberration"]),
        "chromatic_strength": f"{float(n['chromatic_strength']):.2f}",
        "enable_jpeg_compression": bool(n["enable_jpeg_compression"]),
        "jpeg_quality": str(int(n["jpeg_quality"])),
        "enable_color_temperature": bool(n["enable_color_temperature"]),
        "color_temperature_kelvin": str(int(n["color_temperature_kelvin"])),
        "enable_lens_distortion": bool(n["enable_lens_distortion"]),
        "distortion_k1": f"{float(n['distortion_k1']):.2f}",
        "enable_dust": bool(n["enable_dust"]),
        "dust_density": f"{float(n['dust_density']):.2f}",
        "enable_sharpen": bool(n["enable_sharpen"]),
        "sharpen_strength": f"{float(n['sharpen_strength']):.2f}",
    }
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        if not is_filter_popup_extra_key(k):
            continue
        if k.endswith("_randomize") or k.endswith("_enabled"):
            out[k] = parse_bool_like(v, default=False)
        elif k == "filter_mode" or k.endswith("_profile_id"):
            out[k] = str(v)
        elif k.endswith("_min") or k.endswith("_max"):
            try:
                out[k] = f"{float(v):.2f}"
            except Exception:
                pass
        elif k.startswith("realism_"):
            try:
                out[k] = f"{float(v):.4f}"
            except Exception:
                pass
    return out


def _load_shared_filter_profiles(settings_path: Path) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    data = _read_settings_json(settings_path)
    raw_profiles = str(data.get("pipeline.filter_profiles_json", "") or "").strip()
    active = str(data.get("pipeline.filter_profile_active", "") or "").strip() or "Default"
    profiles: Dict[str, Dict[str, Any]] = {}
    if raw_profiles:
        try:
            obj = json.loads(raw_profiles)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        profiles[k] = _filter_settings_to_profile_store(v)
        except Exception:
            profiles = {}

    current = _normalize_filter_settings(
        {
            # Existing filters
            "cardinal_rotation_90": data.get("pipeline.filter.cardinal_rotation_90", True),
            "enable_rotation": data.get("pipeline.filter.enable_rotation", True),
            "enable_blur": data.get("pipeline.filter.enable_blur", True),
            "enable_grain": data.get("pipeline.filter.enable_grain", True),
            "enable_brightness": data.get("pipeline.filter.enable_brightness", True),
            "enable_contrast": data.get("pipeline.filter.enable_contrast", True),
            "rotation_strength": data.get("pipeline.filter.rotation_strength", 1.0),
            "blur_strength": data.get("pipeline.filter.blur_strength", 1.0),
            "grain_strength": data.get("pipeline.filter.grain_strength", 1.0),
            "brightness_strength": data.get("pipeline.filter.brightness_strength", 1.0),
            "contrast_strength": data.get("pipeline.filter.contrast_strength", 1.0),
            # High priority new filters
            "enable_perspective": data.get("pipeline.filter.enable_perspective", False),
            "perspective_strength": data.get("pipeline.filter.perspective_strength", 1.0),
            "enable_motion_blur": data.get("pipeline.filter.enable_motion_blur", True),
            "motion_blur_strength": data.get("pipeline.filter.motion_blur_strength", 1.0),
            "enable_saturation": data.get("pipeline.filter.enable_saturation", True),
            "saturation_factor": data.get("pipeline.filter.saturation_factor", 1.0),
            "enable_hue_shift": data.get("pipeline.filter.enable_hue_shift", True),
            "hue_shift_deg": data.get("pipeline.filter.hue_shift_deg", 0.0),
            "enable_shadow": data.get("pipeline.filter.enable_shadow", True),
            "shadow_strength": data.get("pipeline.filter.shadow_strength", 0.3),
            "enable_reflection": data.get("pipeline.filter.enable_reflection", False),
            "reflection_strength": data.get("pipeline.filter.reflection_strength", 0.5),
            # Medium/Low priority new filters
            "enable_vignetting": data.get("pipeline.filter.enable_vignetting", False),
            "vignetting_strength": data.get("pipeline.filter.vignetting_strength", 1.0),
            "enable_chromatic_aberration": data.get("pipeline.filter.enable_chromatic_aberration", False),
            "chromatic_strength": data.get("pipeline.filter.chromatic_strength", 1.0),
            "enable_jpeg_compression": data.get("pipeline.filter.enable_jpeg_compression", False),
            "jpeg_quality": data.get("pipeline.filter.jpeg_quality", 85),
            "enable_color_temperature": data.get("pipeline.filter.enable_color_temperature", False),
            "color_temperature_kelvin": data.get("pipeline.filter.color_temperature_kelvin", 5500),
            "enable_lens_distortion": data.get("pipeline.filter.enable_lens_distortion", False),
            "distortion_k1": data.get("pipeline.filter.distortion_k1", 0.0),
            "enable_dust": data.get("pipeline.filter.enable_dust", False),
            "dust_density": data.get("pipeline.filter.dust_density", 0.3),
            "enable_sharpen": data.get("pipeline.filter.enable_sharpen", False),
            "sharpen_strength": data.get("pipeline.filter.sharpen_strength", 1.0),
        }
    )
    if not profiles:
        profiles = {"Default": _filter_settings_to_profile_store(current)}
        active = "Default"
    if active not in profiles:
        active = sorted(profiles.keys(), key=lambda s: s.lower())[0]
    active_profile = profiles.get(active, {})
    current = _normalize_filter_settings(active_profile)
    if isinstance(active_profile, dict):
        for k, v in active_profile.items():
            if isinstance(k, str) and is_filter_popup_extra_key(k):
                current[k] = v
    return profiles, active, current


def _save_shared_filter_profiles(settings_path: Path, profiles: Dict[str, Dict[str, Any]], active: str, current: Dict[str, Any]) -> None:
    data = _read_settings_json(settings_path)
    data["pipeline.filter_profiles_json"] = json.dumps(profiles, ensure_ascii=True)
    data["pipeline.filter_profile_active"] = str(active)

    cur = _normalize_filter_settings(current)
    data["pipeline.filter.cardinal_rotation_90"] = bool(cur["cardinal_rotation_90"])
    data["pipeline.filter.enable_rotation"] = bool(cur["enable_rotation"])
    data["pipeline.filter.enable_blur"] = bool(cur["enable_blur"])
    data["pipeline.filter.enable_grain"] = bool(cur["enable_grain"])
    data["pipeline.filter.enable_brightness"] = bool(cur["enable_brightness"])
    data["pipeline.filter.enable_contrast"] = bool(cur["enable_contrast"])
    data["pipeline.filter.rotation_strength"] = f"{float(cur['rotation_strength']):.2f}"
    data["pipeline.filter.blur_strength"] = f"{float(cur['blur_strength']):.2f}"
    data["pipeline.filter.grain_strength"] = f"{float(cur['grain_strength']):.2f}"
    data["pipeline.filter.brightness_strength"] = f"{float(cur['brightness_strength']):.2f}"
    data["pipeline.filter.contrast_strength"] = f"{float(cur['contrast_strength']):.2f}"
    # High priority new filters
    data["pipeline.filter.enable_perspective"] = bool(cur["enable_perspective"])
    data["pipeline.filter.perspective_strength"] = f"{float(cur['perspective_strength']):.2f}"
    data["pipeline.filter.enable_motion_blur"] = bool(cur["enable_motion_blur"])
    data["pipeline.filter.motion_blur_strength"] = f"{float(cur['motion_blur_strength']):.2f}"
    data["pipeline.filter.enable_saturation"] = bool(cur["enable_saturation"])
    data["pipeline.filter.saturation_factor"] = f"{float(cur['saturation_factor']):.2f}"
    data["pipeline.filter.enable_hue_shift"] = bool(cur["enable_hue_shift"])
    data["pipeline.filter.hue_shift_deg"] = f"{float(cur['hue_shift_deg']):.2f}"
    data["pipeline.filter.enable_shadow"] = bool(cur["enable_shadow"])
    data["pipeline.filter.shadow_strength"] = f"{float(cur['shadow_strength']):.2f}"
    data["pipeline.filter.enable_reflection"] = bool(cur["enable_reflection"])
    data["pipeline.filter.reflection_strength"] = f"{float(cur['reflection_strength']):.2f}"
    # Medium/Low priority new filters
    data["pipeline.filter.enable_vignetting"] = bool(cur["enable_vignetting"])
    data["pipeline.filter.vignetting_strength"] = f"{float(cur['vignetting_strength']):.2f}"
    data["pipeline.filter.enable_chromatic_aberration"] = bool(cur["enable_chromatic_aberration"])
    data["pipeline.filter.chromatic_strength"] = f"{float(cur['chromatic_strength']):.2f}"
    data["pipeline.filter.enable_jpeg_compression"] = bool(cur["enable_jpeg_compression"])
    data["pipeline.filter.jpeg_quality"] = str(int(cur["jpeg_quality"]))
    data["pipeline.filter.enable_color_temperature"] = bool(cur["enable_color_temperature"])
    data["pipeline.filter.color_temperature_kelvin"] = str(int(cur["color_temperature_kelvin"]))
    data["pipeline.filter.enable_lens_distortion"] = bool(cur["enable_lens_distortion"])
    data["pipeline.filter.distortion_k1"] = f"{float(cur['distortion_k1']):.2f}"
    data["pipeline.filter.enable_dust"] = bool(cur["enable_dust"])
    data["pipeline.filter.dust_density"] = f"{float(cur['dust_density']):.2f}"
    data["pipeline.filter.enable_sharpen"] = bool(cur["enable_sharpen"])
    data["pipeline.filter.sharpen_strength"] = f"{float(cur['sharpen_strength']):.2f}"
    _write_settings_json(settings_path, data)


def _apply_filter_overrides_to_augment(augment: Dict[str, float], image_filters: Optional[Dict[str, Any]]) -> Dict[str, float]:
    from simple_sim.generators.opencv_2d import apply_image_filter_overrides  # type: ignore

    # Debug preview uses clean base augment values (blur/noise=0, brightness/contrast=1).
    # To make filter toggles visibly testable, inject a small baseline when a filter
    # is enabled but the sampled value is neutral.
    f = _normalize_filter_settings(image_filters or {})
    a = dict(augment or {})

    if bool(f.get("enable_blur", True)) and float(a.get("blur_sigma", 0.0) or 0.0) <= 1e-6:
        a["blur_sigma"] = 1.0
    if bool(f.get("enable_grain", True)) and float(a.get("noise_stddev", 0.0) or 0.0) <= 1e-6:
        a["noise_stddev"] = 8.0
    if bool(f.get("enable_brightness", True)) and abs(float(a.get("brightness_factor", 1.0) or 1.0) - 1.0) <= 1e-6:
        a["brightness_factor"] = 1.15
    if bool(f.get("enable_contrast", True)) and abs(float(a.get("contrast_factor", 1.0) or 1.0) - 1.0) <= 1e-6:
        a["contrast_factor"] = 1.20
    if bool(f.get("enable_rotation", True)) and abs(float(a.get("rotation_deg", 0.0) or 0.0)) <= 1e-6:
        a["rotation_deg"] = 15.0

    return apply_image_filter_overrides(a, f)


def _postprocess_rendered_previews(out_root: Path, jobs: Sequence[Dict[str, Any]]) -> None:
    import cv2  # type: ignore
    from simple_sim.generators.opencv_2d import (  # type: ignore
        apply_blur,
        apply_noise,
        apply_brightness,
        apply_contrast,
        apply_saturation,
        apply_hue_shift,
        apply_color_temperature,
        apply_vignetting,
        apply_chromatic_aberration,
        apply_lens_distortion,
        apply_motion_blur,
        apply_sharpen,
        apply_shadow,
        apply_reflection,
        apply_dust_particles,
        apply_jpeg_compression,
        apply_perspective_transform,
    )

    for j in jobs:
        aug = (j.get("augment") or {})
        blur_sigma = float(aug.get("blur_sigma", 0.0) or 0.0)
        noise_stddev = float(aug.get("noise_stddev", 0.0) or 0.0)
        brightness_factor = float(aug.get("brightness_factor", 1.0) or 1.0)
        contrast_factor = float(aug.get("contrast_factor", 1.0) or 1.0)
        saturation_factor = float(aug.get("saturation_factor", 1.0) or 1.0)
        hue_shift_deg = float(aug.get("hue_shift_deg", 0.0) or 0.0)
        color_temperature_kelvin = int(aug.get("color_temperature_kelvin", 5500) or 5500)
        vignetting_strength = float(aug.get("vignetting_strength", 0.0) or 0.0)
        chromatic_strength = float(aug.get("chromatic_strength", 0.0) or 0.0)
        distortion_k1 = float(aug.get("distortion_k1", 0.0) or 0.0)
        distortion_k2 = float(aug.get("distortion_k2", 0.0) or 0.0)
        motion_blur_strength = float(aug.get("motion_blur_strength", 0.0) or 0.0)
        motion_blur_angle = float(aug.get("motion_blur_angle", 0.0) or 0.0)
        sharpen_strength = float(aug.get("sharpen_strength", 0.0) or 0.0)
        shadow_strength = float(aug.get("shadow_strength", 0.0) or 0.0)
        shadow_size = float(aug.get("shadow_size", 0.2) or 0.2)
        reflection_strength = float(aug.get("reflection_strength", 0.0) or 0.0)
        reflection_size = float(aug.get("reflection_size", 0.15) or 0.15)
        dust_density = float(aug.get("dust_density", 0.0) or 0.0)
        dust_size = float(aug.get("dust_size", 2.0) or 2.0)
        jpeg_quality = int(aug.get("jpeg_quality", 100) or 100)
        perspective_strength = float(aug.get("perspective_strength", 0.0) or 0.0)
        perspective_angle_x = float(aug.get("perspective_angle_x", 0.0) or 0.0)
        perspective_angle_y = float(aug.get("perspective_angle_y", 0.0) or 0.0)
        rotation_deg = float(aug.get("rotation_deg", 0.0) or 0.0)
        if (
            blur_sigma <= 1e-6
            and noise_stddev <= 1e-6
            and abs(brightness_factor - 1.0) <= 1e-6
            and abs(contrast_factor - 1.0) <= 1e-6
            and abs(saturation_factor - 1.0) <= 1e-6
            and abs(hue_shift_deg) <= 1e-6
            and color_temperature_kelvin == 5500
            and vignetting_strength <= 1e-6
            and chromatic_strength <= 1e-6
            and abs(distortion_k1) <= 1e-6
            and abs(distortion_k2) <= 1e-6
            and motion_blur_strength <= 1e-6
            and sharpen_strength <= 1e-6
            and shadow_strength <= 1e-6
            and reflection_strength <= 1e-6
            and dust_density <= 1e-6
            and jpeg_quality >= 100
            and perspective_strength <= 1e-6
            and abs(rotation_deg) <= 1e-6
        ):
            continue
        p = (Path(out_root) / str(j.get("image_path", ""))).resolve()
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rng = np.random.default_rng(int(j.get("seed", 0)))
        img = apply_blur(img, blur_sigma)
        img = apply_noise(img, noise_stddev, rng)
        img = apply_brightness(img, brightness_factor)
        img = apply_contrast(img, contrast_factor)
        img = apply_saturation(img, saturation_factor)
        img = apply_hue_shift(img, hue_shift_deg)
        img = apply_color_temperature(img, color_temperature_kelvin)
        img = apply_vignetting(img, vignetting_strength)
        img = apply_chromatic_aberration(img, chromatic_strength)
        img = apply_lens_distortion(img, distortion_k1, distortion_k2)
        img = apply_motion_blur(img, motion_blur_strength, motion_blur_angle)
        img = apply_sharpen(img, sharpen_strength)
        img = apply_shadow(img, shadow_strength, shadow_size, rng)
        img = apply_reflection(img, reflection_strength, reflection_size, rng)
        img = apply_dust_particles(img, dust_density, dust_size, rng)
        img = apply_jpeg_compression(img, jpeg_quality)
        img = apply_perspective_transform(img, perspective_strength, perspective_angle_x, perspective_angle_y)
        if abs(rotation_deg) > 0.1:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(p), img)


def _sample_nominal_geometry(
    roi_config: Dict[str, Any],
    rng: np.random.Generator,
    geometry_ranges: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Local copy of the nominal sampler (kept here to avoid OpenCV dependency).

    Generator 2D imports OpenCV at module import time; previews only need the sampler.
    """
    defaults = {
        "pad_width": [25.0, 35.0],
        "pad_height": [30.0, 40.0],
        "pad_spacing": [50.0, 65.0],
        "component_length": [55.0, 65.0],
        "component_width": [25.0, 35.0],
    }
    gr = geometry_ranges or {}

    def _range(name: str) -> Tuple[float, float]:
        r = gr.get(name, defaults[name])
        return float(r[0]), float(r[1])

    pad_width = rng.uniform(*_range("pad_width"))
    pad_height = rng.uniform(*_range("pad_height"))
    pad_spacing = rng.uniform(*_range("pad_spacing"))
    component_length = rng.uniform(*_range("component_length"))
    component_width = rng.uniform(*_range("component_width"))

    result: Dict[str, float] = {
        "pad_width": float(pad_width),
        "pad_height": float(pad_height),
        "pad_spacing": float(pad_spacing),
        "component_length": float(component_length),
        "component_width": float(component_width),
    }

    # Optional for multi-pad footprints (e.g. SOT-23, QFN)
    if "pad_spacing_y" in gr:
        lo, hi = float(gr["pad_spacing_y"][0]), float(gr["pad_spacing_y"][1])
        result["pad_spacing_y"] = float(rng.uniform(lo, hi))

    return result


def _get_rotation_jitter_range(cfg: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """Read augment.rotation_deg_range from config, fallback to [0, 0]."""
    augment_cfg = (cfg or {}).get("augment") or {}
    rr = augment_cfg.get("rotation_deg_range", [0.0, 0.0])
    if not isinstance(rr, (list, tuple)) or len(rr) != 2:
        return (0.0, 0.0)
    lo = float(rr[0])
    hi = float(rr[1])
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _sample_preview_augment(
    rng: np.random.Generator,
    *,
    rotation_jitter_range: Tuple[float, float],
    enable_cardinal_rotation_90: bool = True,
) -> Dict[str, float]:
    """Preview augment: keep image clean but randomize global orientation like dataset generation."""
    base_orientation_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0])) if enable_cardinal_rotation_90 else 0.0
    rot_min, rot_max = rotation_jitter_range
    rotation_jitter_deg = float(rng.uniform(rot_min, rot_max))
    return {
        "blur_sigma": 0.0,
        "noise_stddev": 0.0,
        "brightness_factor": 1.0,
        "contrast_factor": 1.0,
        "rotation_deg": float(base_orientation_deg + rotation_jitter_deg),
    }


def _sanitize_dir_name(s: str) -> str:
    # Keep stable and filesystem friendly.
    return re.sub(r"[^a-zA-Z0-9._@+-]+", "_", s).strip("_") or "profile"


def _iter_profile_ids(profiles_dir: Path) -> List[str]:
    # Profile files are `<profile_id>.yaml` where profile_id contains '@'.
    ids: List[str] = []
    for p in sorted(Path(profiles_dir).glob("*.yaml")):
        if p.name.startswith("."):
            continue
        ids.append(p.stem)
    return ids


def _profile_backends(profile: Dict[str, Any]) -> set[str]:
    b = (profile.get("profile") or {}).get("supported_render_backends") or []
    out = set()
    for x in b:
        s = str(x)
        if s in {BACKEND_2D, BACKEND_3D}:
            out.add(s)
    return out


def _discover_profiles(profiles_dir: Path, *, backend_mode: str) -> List[Tuple[str, Dict[str, Any], set[str]]]:
    """Discover profiles supporting 2D and/or 3D."""
    backend_mode = str(backend_mode or "auto")
    out: List[Tuple[str, Dict[str, Any], set[str]]] = []
    for pid in _iter_profile_ids(profiles_dir):
        try:
            prof = load_profile(pid, profiles_dir)
        except Exception:
            continue
        backends = _profile_backends(prof)
        if not backends:
            continue
        if backend_mode == BACKEND_2D and BACKEND_2D not in backends:
            continue
        if backend_mode == BACKEND_3D and BACKEND_3D not in backends:
            continue
        out.append((pid, prof, backends))
    out.sort(key=lambda t: t[0])
    return out


def _parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = []
    for chunk in str(s).split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _parse_selection(selection: str, n: int) -> List[int]:
    """Parse '1,2,4-6,all' into 0-based indices."""
    selection = (selection or "").strip().lower()
    if selection in {"a", "all", "*"}:
        return list(range(n))

    idxs: List[int] = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            if lo <= 0 or hi <= 0:
                raise ValueError("Selection indices are 1-based and must be positive.")
            for k in range(min(lo, hi), max(lo, hi) + 1):
                idxs.append(k - 1)
        else:
            k = int(part)
            if k <= 0:
                raise ValueError("Selection indices are 1-based and must be positive.")
            idxs.append(k - 1)

    # Dedup while preserving order
    seen = set()
    final = []
    for i in idxs:
        if i < 0 or i >= n:
            raise ValueError(f"Selection index out of range: {i + 1} (valid: 1..{n})")
        if i in seen:
            continue
        seen.add(i)
        final.append(i)
    return final


def _find_matching_run_configs(configs_dir: Path, *, profile_id: str, backend: str) -> List[Path]:
    configs_dir = Path(configs_dir)
    backend = str(backend)
    hits: List[Path] = []
    for p in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = load_config(p)
        except Exception:
            continue
        if str((cfg.get("render") or {}).get("backend", "")) != backend:
            continue
        run = cfg.get("run") or {}
        if str(run.get("component_profile", "")) == str(profile_id):
            hits.append(p)
    return hits


def _auto_select_profiles_with_missing_previews(
    discovered: List[Tuple[str, Dict[str, Any], set[str]]],
    *,
    profiles_dir: Path,
    out_root: Path,
    backend_mode: str,
    states_override: str,
) -> List[str]:
    """Return profile IDs that are new/incomplete in preview output.

    A profile is selected when at least one expected preview image is missing for
    the requested backend mode and defect states.
    """
    selected: List[str] = []
    for pid, prof, backends in discovered:
        defect_types = _parse_csv_list(states_override) or list(prof.get("defect_set") or [])
        if not defect_types:
            defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

        if backend_mode == "both":
            backends_to_check = [b for b in [BACKEND_2D, BACKEND_3D] if b in backends]
        elif backend_mode in backends:
            backends_to_check = [backend_mode]
        else:
            backends_to_check = []

        if not backends_to_check:
            continue

        profile_dir = _sanitize_dir_name(pid)
        profile_path = (Path(profiles_dir) / f"{pid}.yaml").resolve()
        profile_mtime = profile_path.stat().st_mtime if profile_path.exists() else 0.0
        needs_render = False
        for backend in backends_to_check:
            for defect_type in defect_types:
                img_path = (out_root / f"previews/{profile_dir}/{backend}/{str(defect_type)}.png").resolve()
                if not img_path.exists():
                    needs_render = True
                    break
                # Re-render when profile changed after the preview image.
                if profile_mtime > img_path.stat().st_mtime:
                    needs_render = True
                    break
            if needs_render:
                break

        if needs_render:
            selected.append(pid)

    return selected


def _settings_from_config(cfg: Dict[str, Any]) -> RenderSettings:
    validate_config(cfg)
    roi = cfg["roi"]
    blender = (cfg.get("render") or {}).get("blender") or {}
    return RenderSettings(
        roi_width_px=int(roi["width_px"]),
        roi_height_px=int(roi["height_px"]),
        mm_per_px=float(roi["mm_per_px"]),
        blender_executable=str(blender.get("executable", "blender")),
        cycles_samples=int(blender.get("samples", 64)),
        device=str(blender.get("device", "CPU")),
    )


def _write_index_html(out_root: Path, previews: List[PreviewRec]) -> None:
    """previews: list of (profile_id, backend, defect, rel_image_path)."""
    out_root = Path(out_root)
    index_path = out_root / "index.html"

    # Group by profile id
    by_profile: Dict[str, List[Tuple[str, str, str]]] = {}
    for pid, backend, defect, rel_path in previews:
        by_profile.setdefault(pid, []).append((backend, defect, rel_path))

    def _escape(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    lines.append("<title>Simple-Sim Debug Previews</title>")
    lines.append("<style>")
    lines.append("body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial; margin:24px;}")
    lines.append("h1{font-size:20px;margin:0 0 16px 0}")
    lines.append("h2{font-size:16px;margin:20px 0 8px 0}")
    lines.append(".grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px;}")
    lines.append(".card{border:1px solid #ddd; border-radius:10px; padding:10px;}")
    lines.append(".cap{font-size:12px;color:#333;margin:0 0 8px 0; display:flex; gap:6px; flex-wrap:wrap; align-items:baseline;}")
    lines.append(".pill{font-size:11px; padding:2px 6px; border-radius:999px; border:1px solid #ddd; background:#fafafa;}")
    lines.append(".pill.backend{border-color:#ddd; background:#fff;}")
    lines.append(".pill.state{border-color:#cfd8ff; background:#f5f7ff;}")
    lines.append("img{width:100%; height:auto; border-radius:8px; background:#f6f6f6;}")
    lines.append("</style></head><body>")
    lines.append("<h1>Simple-Sim Debug Previews</h1>")

    for pid in sorted(by_profile.keys()):
        lines.append(f"<h2>{_escape(pid)}</h2>")
        lines.append("<div class='grid'>")
        items = by_profile[pid]
        items.sort(key=lambda t: (t[0], t[1]))
        for backend, defect, rel_path in items:
            lines.append("<div class='card'>")
            lines.append("<div class='cap'>")
            lines.append(f"<span class='pill'>{_escape(pid)}</span>")
            lines.append(f"<span class='pill backend'>{_escape(backend)}</span>")
            lines.append(f"<span class='pill state'>{_escape(defect)}</span>")
            lines.append("</div>")
            lines.append(f"<a href='{_escape(rel_path)}'><img loading='lazy' src='{_escape(rel_path)}'></a>")
            lines.append("</div>")
        lines.append("</div>")

    lines.append("</body></html>")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_existing_previews(out_root: Path) -> List[PreviewRec]:
    """Scan existing preview PNG files and return preview records."""
    out_root = Path(out_root)
    recs: List[PreviewRec] = []
    for img_path in sorted((out_root / "previews").glob("*/*/*.png")):
        try:
            rel = img_path.relative_to(out_root)
        except Exception:
            continue
        parts = rel.parts
        # Expected: previews/<profile_id>/<backend>/<defect>.png
        if len(parts) != 4 or parts[0] != "previews":
            continue
        pid = str(parts[1])
        backend = str(parts[2])
        defect = str(Path(parts[3]).stem)
        recs.append((pid, backend, defect, str(rel.as_posix())))
    return recs


def _open_tk_viewer(
    out_root: Path,
    previews: List[PreviewRec],
    *,
    sim_root: Optional[Path] = None,
    profiles_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
    all_profiles: Optional[List[Tuple[str, Dict[str, Any], set[str]]]] = None,
    render_settings: Optional[Dict[str, Any]] = None,
    selected_profile_ids: Optional[List[str]] = None,
) -> None:
    """Open an interactive Tkinter viewer for rendered previews with rerun capabilities."""
    # Import lazily so the script can still run on systems without Tk installed.
    try:
        import tkinter as tk
        from tkinter import ttk, simpledialog
    except Exception as exc:
        raise RuntimeError(f"Tkinter not available: {exc}") from exc

    try:
        from PIL import Image, ImageTk
    except Exception as exc:
        raise RuntimeError(f"Pillow (PIL) not available: {exc}") from exc

    import threading

    out_root = Path(out_root)
    settings_file = _shared_settings_path(Path(sim_root) if sim_root is not None else Path(__file__).parent.parent, str((render_settings or {}).get("settings_file", "") or ""))

    # State for interactive mode
    class ViewerState:
        def __init__(self):
            self.previews = previews
            self.is_rendering = False
            self.backend_filter = "both"

    state = ViewerState()

    def _load_items():
        """Load image items from current preview list."""
        items: List[Tuple[str, str, str, Path]] = []
        for pid, backend, defect, rel in state.previews:
            if state.backend_filter != "both" and backend != state.backend_filter:
                continue
            p = (out_root / rel).resolve()
            if p.exists():
                items.append((pid, backend, defect, p))
        return items

    root = tk.Tk()
    root.title(f"Simple-Sim Debug Previews")
    root.geometry("1200x900")

    top = ttk.Frame(root, padding=10)
    top.pack(fill="both", expand=True)

    # Header with controls
    header = ttk.Frame(top)
    header.pack(fill="x", pady=(0, 10))

    ttk.Label(header, text="Simple-Sim Debug Previews", font=("TkDefaultFont", 14, "bold")).pack(side="left")
    ttk.Label(
        header,
        text=str(out_root),
        foreground="#444",
    ).pack(side="left", padx=12)

    # Interactive controls (only if we have the necessary context)
    if all_profiles and sim_root and profiles_dir and configs_dir and render_settings:
        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(0, 10))

        def _profiles_for_backend(mode: str) -> List[str]:
            mode = str(mode or "both")
            if mode == "both":
                return [pid for pid, _prof, _b in all_profiles]
            return [pid for pid, _prof, b in all_profiles if mode in b]

        # Backend selection / filter
        ttk.Label(controls, text="Backend:").pack(side="left", padx=(0, 5))
        backend_var = tk.StringVar(value=str(render_settings.get("backend_mode", "both")))
        backend_combo = ttk.Combobox(
            controls,
            textvariable=backend_var,
            values=["both", BACKEND_2D, BACKEND_3D],
            width=10,
            state="readonly",
        )
        backend_combo.pack(side="left", padx=(0, 15))
        state.backend_filter = backend_var.get()

        # Profile selection
        ttk.Label(controls, text="Profile:").pack(side="left", padx=(0, 5))

        profile_var = tk.StringVar()
        profile_ids = _profiles_for_backend(state.backend_filter)
        profile_combo = ttk.Combobox(controls, textvariable=profile_var, values=profile_ids, width=35, state="readonly")
        if profile_ids:
            # Pre-select the first selected profile if it was explicitly chosen
            initial_profile = None
            if selected_profile_ids and len(selected_profile_ids) > 0:
                for pid in selected_profile_ids:
                    if pid in profile_ids:
                        initial_profile = pid
                        break
            if initial_profile:
                profile_var.set(initial_profile)
            else:
                profile_combo.current(0)
        profile_combo.pack(side="left", padx=(0, 15))

        def _on_backend_change(_evt=None):
            state.backend_filter = backend_var.get()
            ids = _profiles_for_backend(state.backend_filter)
            profile_combo["values"] = ids
            if ids:
                profile_combo.current(0)
            _refresh_images()

        backend_combo.bind("<<ComboboxSelected>>", _on_backend_change)

        # Seed controls
        ttk.Label(controls, text="Seed:").pack(side="left", padx=(0, 5))
        seed_var = tk.StringVar(value=str(render_settings.get("seed_base", 2026)))
        seed_entry = ttk.Entry(controls, textvariable=seed_var, width=8)
        seed_entry.pack(side="left", padx=(0, 10))

        variable_seeds_var = tk.BooleanVar(value=True)  # Default to variable for reruns
        variable_seeds_check = ttk.Checkbutton(controls, text="Variable", variable=variable_seeds_var)
        variable_seeds_check.pack(side="left", padx=(0, 15))

        shared_profiles, shared_active_profile, shared_current_filter = _load_shared_filter_profiles(settings_file)
        current_image_filters = dict(shared_current_filter)
        extra_filter_values: Dict[str, Any] = {
            k: v for k, v in current_image_filters.items()
            if isinstance(k, str) and is_filter_popup_extra_key(k)
        }

        filter_cardinal_rotation_90_var = tk.BooleanVar(value=bool(current_image_filters.get("cardinal_rotation_90", True)))
        filter_enable_rotation_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_rotation", True)))
        filter_enable_blur_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_blur", True)))
        filter_enable_grain_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_grain", True)))
        filter_enable_brightness_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_brightness", True)))
        filter_enable_contrast_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_contrast", True)))
        filter_rotation_strength_var = tk.DoubleVar(value=float(current_image_filters.get("rotation_strength", 1.0)))
        filter_blur_strength_var = tk.DoubleVar(value=float(current_image_filters.get("blur_strength", 1.0)))
        filter_grain_strength_var = tk.DoubleVar(value=float(current_image_filters.get("grain_strength", 1.0)))
        filter_brightness_strength_var = tk.DoubleVar(value=float(current_image_filters.get("brightness_strength", 1.0)))
        filter_contrast_strength_var = tk.DoubleVar(value=float(current_image_filters.get("contrast_strength", 1.0)))
        # High priority new filters
        filter_enable_perspective_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_perspective", False)))
        filter_perspective_strength_var = tk.DoubleVar(value=float(current_image_filters.get("perspective_strength", 1.0)))
        filter_enable_motion_blur_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_motion_blur", True)))
        filter_motion_blur_strength_var = tk.DoubleVar(value=float(current_image_filters.get("motion_blur_strength", 1.0)))
        filter_enable_saturation_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_saturation", True)))
        filter_saturation_factor_var = tk.DoubleVar(value=float(current_image_filters.get("saturation_factor", 1.0)))
        filter_enable_hue_shift_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_hue_shift", True)))
        filter_hue_shift_deg_var = tk.DoubleVar(value=float(current_image_filters.get("hue_shift_deg", 0.0)))
        filter_enable_shadow_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_shadow", True)))
        filter_shadow_strength_var = tk.DoubleVar(value=float(current_image_filters.get("shadow_strength", 0.3)))
        filter_enable_reflection_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_reflection", False)))
        filter_reflection_strength_var = tk.DoubleVar(value=float(current_image_filters.get("reflection_strength", 0.5)))
        # Medium/Low priority new filters
        filter_enable_vignetting_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_vignetting", False)))
        filter_vignetting_strength_var = tk.DoubleVar(value=float(current_image_filters.get("vignetting_strength", 1.0)))
        filter_enable_chromatic_aberration_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_chromatic_aberration", False)))
        filter_chromatic_strength_var = tk.DoubleVar(value=float(current_image_filters.get("chromatic_strength", 1.0)))
        filter_enable_jpeg_compression_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_jpeg_compression", False)))
        filter_jpeg_quality_var = tk.DoubleVar(value=float(current_image_filters.get("jpeg_quality", 85)))
        filter_enable_color_temperature_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_color_temperature", False)))
        filter_color_temperature_kelvin_var = tk.DoubleVar(value=float(current_image_filters.get("color_temperature_kelvin", 5500)))
        filter_enable_lens_distortion_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_lens_distortion", False)))
        filter_distortion_k1_var = tk.DoubleVar(value=float(current_image_filters.get("distortion_k1", 0.0)))
        filter_enable_dust_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_dust", False)))
        filter_dust_density_var = tk.DoubleVar(value=float(current_image_filters.get("dust_density", 0.3)))
        filter_enable_sharpen_var = tk.BooleanVar(value=bool(current_image_filters.get("enable_sharpen", False)))
        filter_sharpen_strength_var = tk.DoubleVar(value=float(current_image_filters.get("sharpen_strength", 1.0)))

        def _current_filters_from_vars() -> Dict[str, Any]:
            base = _normalize_filter_settings(
                {
                    "cardinal_rotation_90": bool(filter_cardinal_rotation_90_var.get()),
                    "enable_rotation": bool(filter_enable_rotation_var.get()),
                    "enable_blur": bool(filter_enable_blur_var.get()),
                    "enable_grain": bool(filter_enable_grain_var.get()),
                    "enable_brightness": bool(filter_enable_brightness_var.get()),
                    "enable_contrast": bool(filter_enable_contrast_var.get()),
                    "rotation_strength": float(filter_rotation_strength_var.get()),
                    "blur_strength": float(filter_blur_strength_var.get()),
                    "grain_strength": float(filter_grain_strength_var.get()),
                    "brightness_strength": float(filter_brightness_strength_var.get()),
                    "contrast_strength": float(filter_contrast_strength_var.get()),
                    # High priority new filters
                    "enable_perspective": bool(filter_enable_perspective_var.get()),
                    "perspective_strength": float(filter_perspective_strength_var.get()),
                    "enable_motion_blur": bool(filter_enable_motion_blur_var.get()),
                    "motion_blur_strength": float(filter_motion_blur_strength_var.get()),
                    "enable_saturation": bool(filter_enable_saturation_var.get()),
                    "saturation_factor": float(filter_saturation_factor_var.get()),
                    "enable_hue_shift": bool(filter_enable_hue_shift_var.get()),
                    "hue_shift_deg": float(filter_hue_shift_deg_var.get()),
                    "enable_shadow": bool(filter_enable_shadow_var.get()),
                    "shadow_strength": float(filter_shadow_strength_var.get()),
                    "enable_reflection": bool(filter_enable_reflection_var.get()),
                    "reflection_strength": float(filter_reflection_strength_var.get()),
                    # Medium/Low priority new filters
                    "enable_vignetting": bool(filter_enable_vignetting_var.get()),
                    "vignetting_strength": float(filter_vignetting_strength_var.get()),
                    "enable_chromatic_aberration": bool(filter_enable_chromatic_aberration_var.get()),
                    "chromatic_strength": float(filter_chromatic_strength_var.get()),
                    "enable_jpeg_compression": bool(filter_enable_jpeg_compression_var.get()),
                    "jpeg_quality": float(filter_jpeg_quality_var.get()),
                    "enable_color_temperature": bool(filter_enable_color_temperature_var.get()),
                    "color_temperature_kelvin": float(filter_color_temperature_kelvin_var.get()),
                    "enable_lens_distortion": bool(filter_enable_lens_distortion_var.get()),
                    "distortion_k1": float(filter_distortion_k1_var.get()),
                    "enable_dust": bool(filter_enable_dust_var.get()),
                    "dust_density": float(filter_dust_density_var.get()),
                    "enable_sharpen": bool(filter_enable_sharpen_var.get()),
                    "sharpen_strength": float(filter_sharpen_strength_var.get()),
                }
            )
            base.update(extra_filter_values)
            return base

        def _apply_filters_to_vars(d: Dict[str, Any]) -> None:
            x = _normalize_filter_settings(d)
            extra_filter_values.clear()
            for k, v in d.items():
                if isinstance(k, str) and is_filter_popup_extra_key(k):
                    extra_filter_values[k] = v
            filter_cardinal_rotation_90_var.set(bool(x["cardinal_rotation_90"]))
            filter_enable_rotation_var.set(bool(x["enable_rotation"]))
            filter_enable_blur_var.set(bool(x["enable_blur"]))
            filter_enable_grain_var.set(bool(x["enable_grain"]))
            filter_enable_brightness_var.set(bool(x["enable_brightness"]))
            filter_enable_contrast_var.set(bool(x["enable_contrast"]))
            filter_rotation_strength_var.set(float(x["rotation_strength"]))
            filter_blur_strength_var.set(float(x["blur_strength"]))
            filter_grain_strength_var.set(float(x["grain_strength"]))
            filter_brightness_strength_var.set(float(x["brightness_strength"]))
            filter_contrast_strength_var.set(float(x["contrast_strength"]))
            # High priority new filters
            filter_enable_perspective_var.set(bool(x["enable_perspective"]))
            filter_perspective_strength_var.set(float(x["perspective_strength"]))
            filter_enable_motion_blur_var.set(bool(x["enable_motion_blur"]))
            filter_motion_blur_strength_var.set(float(x["motion_blur_strength"]))
            filter_enable_saturation_var.set(bool(x["enable_saturation"]))
            filter_saturation_factor_var.set(float(x["saturation_factor"]))
            filter_enable_hue_shift_var.set(bool(x["enable_hue_shift"]))
            filter_hue_shift_deg_var.set(float(x["hue_shift_deg"]))
            filter_enable_shadow_var.set(bool(x["enable_shadow"]))
            filter_shadow_strength_var.set(float(x["shadow_strength"]))
            filter_enable_reflection_var.set(bool(x["enable_reflection"]))
            filter_reflection_strength_var.set(float(x["reflection_strength"]))
            # Medium/Low priority new filters
            filter_enable_vignetting_var.set(bool(x["enable_vignetting"]))
            filter_vignetting_strength_var.set(float(x["vignetting_strength"]))
            filter_enable_chromatic_aberration_var.set(bool(x["enable_chromatic_aberration"]))
            filter_chromatic_strength_var.set(float(x["chromatic_strength"]))
            filter_enable_jpeg_compression_var.set(bool(x["enable_jpeg_compression"]))
            filter_jpeg_quality_var.set(float(x["jpeg_quality"]))
            filter_enable_color_temperature_var.set(bool(x["enable_color_temperature"]))
            filter_color_temperature_kelvin_var.set(float(x["color_temperature_kelvin"]))
            filter_enable_lens_distortion_var.set(bool(x["enable_lens_distortion"]))
            filter_distortion_k1_var.set(float(x["distortion_k1"]))
            filter_enable_dust_var.set(bool(x["enable_dust"]))
            filter_dust_density_var.set(float(x["dust_density"]))
            filter_enable_sharpen_var.set(bool(x["enable_sharpen"]))
            filter_sharpen_strength_var.set(float(x["sharpen_strength"]))

        def _persist_filters(active_name: Optional[str] = None) -> None:
            nonlocal shared_active_profile
            cur = _current_filters_from_vars()
            if active_name:
                shared_active_profile = active_name
            shared_profiles[shared_active_profile] = _filter_settings_to_profile_store(cur)
            _save_shared_filter_profiles(
                settings_file,
                shared_profiles,
                shared_active_profile,
                cur,
            )

        def _open_filter_popup() -> None:
            """Open shared image filter popup."""
            def on_close(updated_profiles: Dict[str, Dict[str, Any]], updated_active: str) -> None:
                nonlocal shared_active_profile, shared_profiles
                shared_profiles = updated_profiles
                shared_active_profile = updated_active
                _apply_filters_to_vars(shared_profiles.get(shared_active_profile, {}))
                _save_shared_filter_profiles(
                    settings_file,
                    shared_profiles,
                    shared_active_profile,
                    _current_filters_from_vars(),
                )

            def show_messagebox(msg_type: str, title: str, message: str) -> None:
                if msg_type == "warning":
                    messagebox.showwarning(title, message, parent=root)
                elif msg_type == "error":
                    messagebox.showerror(title, message, parent=root)
                else:
                    messagebox.showinfo(title, message, parent=root)

            def ask_string(title: str, prompt: str, initial: str = "") -> Optional[str]:
                from tkinter import simpledialog
                return simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)

            def ask_yes_no(title: str, question: str) -> bool:
                return messagebox.askyesno(title, question, parent=root)

            def float_or_default(val: str, default: float) -> float:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            open_filter_popup(
                parent=root,
                profiles=shared_profiles,
                active_profile=shared_active_profile,
                on_close=on_close,
                get_current_values=lambda: _filter_settings_to_profile_store(_current_filters_from_vars()),
                set_current_values=_apply_filters_to_vars,
                float_or_default=float_or_default,
                show_messagebox=show_messagebox,
                ask_string=ask_string,
                ask_yes_no=ask_yes_no,
                sim_root=str(sim_root),
            )

        cardinal_rotation_90_var = filter_cardinal_rotation_90_var
        cardinal_rotation_90_check = ttk.Checkbutton(controls, text="90° Rotation", variable=cardinal_rotation_90_var)
        cardinal_rotation_90_check.pack(side="left", padx=(0, 15))
        ttk.Button(controls, text="Image Filters", command=_open_filter_popup).pack(side="left", padx=(0, 15))

        status_label = ttk.Label(controls, text="Ready", foreground="#060")
        status_label.pack(side="left", padx=(10, 10))

        def _start_render():
            if state.is_rendering:
                return

            selected_pid = profile_var.get()
            if not selected_pid:
                status_label.config(text="No profile selected", foreground="#b00")
                return

            state.is_rendering = True
            render_btn.config(state="disabled")
            status_label.config(text="Rendering...", foreground="#c60")
            _persist_filters(shared_active_profile)

            def _render_thread():
                try:
                    # Always reload profile from disk so edits are picked up without restart.
                    selected_prof = load_profile(selected_pid, profiles_dir)
                    selected_backends = _profile_backends(selected_prof)
                    if not selected_backends:
                        raise RuntimeError(f"Profile {selected_pid!r} has no supported backends")

                    # Extract render settings from the passed config and UI controls
                    forced_cfg = render_settings.get("forced_cfg")

                    # Get seed from UI
                    try:
                        seed_base = int(seed_var.get())
                    except ValueError:
                        seed_base = render_settings.get("seed_base", 2026)

                    use_variable_seeds = variable_seeds_var.get()

                    states_override = render_settings.get("states", "")
                    blender_override = render_settings.get("blender", "")
                    samples_override = render_settings.get("samples", 0)
                    device_override = render_settings.get("device", "")
                    roi_override = render_settings.get("roi_override")

                    # Get defect types
                    defect_types = _parse_csv_list(states_override) or list(selected_prof.get("defect_set") or [])
                    if not defect_types:
                        defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

                    # Determine which backend(s) to render in this run
                    mode = backend_var.get()
                    if mode == "both":
                        backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in selected_backends]
                    else:
                        backends_to_render = [mode] if mode in selected_backends else []
                    if not backends_to_render:
                        raise RuntimeError(f"Profile {selected_pid!r} does not support backend={mode!r}")

                    run_id = int(time.time() * 1000000) if use_variable_seeds else None
                    all_new_previews: List[PreviewRec] = []

                    for backend in backends_to_render:
                        # Select config for this profile/backend
                        if forced_cfg is not None:
                            cfg = forced_cfg
                            validate_config(cfg)
                            cfg_backend = str((cfg.get("render") or {}).get("backend", ""))
                            if cfg_backend != backend:
                                raise RuntimeError(f"Forced --config backend={cfg_backend!r} does not match requested backend={backend!r}")
                        else:
                            matches = _find_matching_run_configs(configs_dir, profile_id=selected_pid, backend=backend)
                            if matches:
                                cfg = load_config(matches[0])
                            else:
                                fallback = "configs/run_0001.yaml" if backend == BACKEND_2D else "configs/run_0001_3d.yaml"
                                cfg = load_config(sim_root / fallback)
                            validate_config(cfg)

                        if backend == BACKEND_3D:
                            settings = _settings_from_config(cfg)

                            # Apply overrides (3D only)
                            if roi_override is not None:
                                w, h, mpp = roi_override
                                settings = RenderSettings(
                                    roi_width_px=int(w),
                                    roi_height_px=int(h),
                                    mm_per_px=float(mpp),
                                    blender_executable=settings.blender_executable,
                                    cycles_samples=settings.cycles_samples,
                                    device=settings.device,
                                )

                            if blender_override:
                                settings = RenderSettings(
                                    roi_width_px=settings.roi_width_px,
                                    roi_height_px=settings.roi_height_px,
                                    mm_per_px=settings.mm_per_px,
                                    blender_executable=str(blender_override),
                                    cycles_samples=settings.cycles_samples,
                                    device=settings.device,
                                )

                            if samples_override and int(samples_override) > 0:
                                settings = RenderSettings(
                                    roi_width_px=settings.roi_width_px,
                                    roi_height_px=settings.roi_height_px,
                                    mm_per_px=settings.mm_per_px,
                                    blender_executable=settings.blender_executable,
                                    cycles_samples=int(samples_override),
                                    device=settings.device,
                                )

                            if device_override:
                                settings = RenderSettings(
                                    roi_width_px=settings.roi_width_px,
                                    roi_height_px=settings.roi_height_px,
                                    mm_per_px=settings.mm_per_px,
                                    blender_executable=settings.blender_executable,
                                    cycles_samples=settings.cycles_samples,
                                    device=str(device_override),
                                )

                            jobs, new_previews = _build_preview_jobs(
                                profile_id=selected_pid,
                                profile=selected_prof,
                                settings=settings,
                                out_root=out_root,
                                seed_base=int(seed_base),
                                defect_types=defect_types,
                                run_id=run_id,
                                backend=backend,
                                rotation_jitter_range=_get_rotation_jitter_range(cfg),
                                enable_cardinal_rotation_90=bool(_current_filters_from_vars().get("cardinal_rotation_90", True)),
                                image_filters=_current_filters_from_vars(),
                            )

                            jobs_path = out_root / "blender_jobs_previews.jsonl"
                            write_jobs_jsonl(jobs_path, jobs)
                            render_blender_batch(
                                sim_root=sim_root,
                                jobs_path=jobs_path,
                                output_root=out_root,
                                blender_executable=settings.blender_executable,
                                cycles_samples=settings.cycles_samples,
                                device=settings.device,
                            )
                            _postprocess_rendered_previews(out_root, jobs)
                            all_new_previews.extend(new_previews)
                        else:
                            # 2D OpenCV render
                            import cv2  # type: ignore
                            from simple_sim.generators.opencv_2d import render_roi  # type: ignore

                            roi = cfg["roi"]
                            w = int(roi["width_px"])
                            h = int(roi["height_px"])
                            mpp = float(roi["mm_per_px"])
                            if roi_override is not None:
                                w, h, mpp = int(roi_override[0]), int(roi_override[1]), float(roi_override[2])

                            roi_cfg = {"width_px": int(w), "height_px": int(h), "mm_per_px": float(mpp)}

                            render_cfg = dict(cfg["render"])
                            # schema v2: allow render.component_color to come from profile.
                            if "component_color" not in render_cfg and "render" in selected_prof:
                                render_cfg["component_color"] = selected_prof["render"]["component_color_bgr"]

                            footprint = str((selected_prof.get("component") or {}).get("footprint", "chip_2pad"))
                            geometry_ranges = selected_prof.get("geometry_ranges") or {}
                            tolerances = selected_prof.get("tolerances")
                            rotation_jitter_range = _get_rotation_jitter_range(cfg)

                            profile_dir = _sanitize_dir_name(selected_pid)
                            for defect_type in defect_types:
                                if run_id is not None:
                                    seed_str = f"{int(seed_base)}|{selected_pid}|{backend}|{str(defect_type)}|{int(run_id)}"
                                else:
                                    seed_str = f"{int(seed_base)}|{selected_pid}|{backend}|{str(defect_type)}"
                                hh = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
                                seed = int(hh[:12], 16)
                                rng = np.random.default_rng(seed)
                                augment = _sample_preview_augment(
                                    rng,
                                    rotation_jitter_range=rotation_jitter_range,
                                    enable_cardinal_rotation_90=bool(_current_filters_from_vars().get("cardinal_rotation_90", True)),
                                )
                                augment = _apply_filter_overrides_to_augment(augment, _current_filters_from_vars())

                                nominal = _sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
                                defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)
                                img = render_roi(
                                    nominal=nominal,
                                    defect_params=defect,
                                    augment=augment,
                                    roi_size=(int(w), int(h)),
                                    config=render_cfg,
                                    tolerances=tolerances,
                                    rng=rng,
                                    footprint=footprint,
                                )

                                rel_path = f"previews/{profile_dir}/{backend}/{str(defect_type)}.png"
                                out_path = (out_root / rel_path).resolve()
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                ok = cv2.imwrite(str(out_path), img)
                                if not ok:
                                    raise RuntimeError(f"Failed to write image: {out_path}")
                                all_new_previews.append((selected_pid, backend, str(defect_type), rel_path))

                    # Update state with new previews (replace old ones for this profile+backend(s))
                    rendered_b = set(backends_to_render)
                    state.previews = [rec for rec in state.previews if not (rec[0] == selected_pid and rec[1] in rendered_b)]
                    state.previews.extend(all_new_previews)

                    # Update index.html
                    _write_index_html(out_root, state.previews)

                    # Refresh UI
                    root.after(0, lambda: _refresh_images())
                    root.after(0, lambda: status_label.config(text="Render complete", foreground="#060"))

                except Exception as exc:
                    root.after(0, lambda: status_label.config(text=f"Error: {exc}", foreground="#b00"))
                finally:
                    state.is_rendering = False
                    root.after(0, lambda: render_btn.config(state="normal"))

            thread = threading.Thread(target=_render_thread, daemon=True)
            thread.start()

        def _start_render_all():
            """Render all available profiles."""
            if state.is_rendering:
                return

            if not all_profiles:
                status_label.config(text="No profiles available", foreground="#b00")
                return

            state.is_rendering = True
            render_btn.config(state="disabled")
            render_all_btn.config(state="disabled")
            status_label.config(text=f"Rendering all {len(all_profiles)} profiles...", foreground="#c60")
            _persist_filters(shared_active_profile)

            def _render_all_thread():
                try:
                    # Extract render settings
                    forced_cfg = render_settings.get("forced_cfg")

                    # Get seed from UI
                    try:
                        seed_base = int(seed_var.get())
                    except ValueError:
                        seed_base = render_settings.get("seed_base", 2026)

                    use_variable_seeds = variable_seeds_var.get()

                    states_override = render_settings.get("states", "")
                    blender_override = render_settings.get("blender", "")
                    samples_override = render_settings.get("samples", 0)
                    device_override = render_settings.get("device", "")
                    roi_override = render_settings.get("roi_override")

                    mode = backend_var.get()
                    all_new_previews: List[PreviewRec] = []

                    for idx, (pid, _prof_cached, _backends_cached) in enumerate(all_profiles, 1):
                        # Always reload profile from disk so edits are picked up without restart.
                        prof = load_profile(pid, profiles_dir)
                        backends = _profile_backends(prof)

                        # Skip profiles not in the current backend filter.
                        if mode != "both" and mode not in backends:
                            continue

                        root.after(0, lambda i=idx, p=pid: status_label.config(
                            text=f"Rendering {i}/{len(all_profiles)}: {p}", foreground="#c60"))

                        defect_types = _parse_csv_list(states_override) or list(prof.get("defect_set") or [])
                        if not defect_types:
                            defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

                        run_id = int(time.time() * 1000000) if use_variable_seeds else None

                        backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in backends] if mode == "both" else [mode]

                        for backend in backends_to_render:
                            # Get config for this profile/backend
                            if forced_cfg is not None:
                                cfg = forced_cfg
                                validate_config(cfg)
                                cfg_backend = str((cfg.get("render") or {}).get("backend", ""))
                                if cfg_backend != backend:
                                    raise RuntimeError(f"Forced --config backend={cfg_backend!r} does not match requested backend={backend!r}")
                            else:
                                matches = _find_matching_run_configs(configs_dir, profile_id=pid, backend=backend)
                                if matches:
                                    cfg = load_config(matches[0])
                                else:
                                    fallback = "configs/run_0001.yaml" if backend == BACKEND_2D else "configs/run_0001_3d.yaml"
                                    cfg = load_config(sim_root / fallback)
                                validate_config(cfg)

                            if backend == BACKEND_3D:
                                settings = _settings_from_config(cfg)

                                if roi_override is not None:
                                    w, h, mpp = roi_override
                                    settings = RenderSettings(
                                        roi_width_px=int(w),
                                        roi_height_px=int(h),
                                        mm_per_px=float(mpp),
                                        blender_executable=settings.blender_executable,
                                        cycles_samples=settings.cycles_samples,
                                        device=settings.device,
                                    )

                                if blender_override:
                                    settings = RenderSettings(
                                        roi_width_px=settings.roi_width_px,
                                        roi_height_px=settings.roi_height_px,
                                        mm_per_px=settings.mm_per_px,
                                        blender_executable=str(blender_override),
                                        cycles_samples=settings.cycles_samples,
                                        device=settings.device,
                                    )

                                if samples_override and int(samples_override) > 0:
                                    settings = RenderSettings(
                                        roi_width_px=settings.roi_width_px,
                                        roi_height_px=settings.roi_height_px,
                                        mm_per_px=settings.mm_per_px,
                                        blender_executable=settings.blender_executable,
                                        cycles_samples=int(samples_override),
                                        device=settings.device,
                                    )

                                if device_override:
                                    settings = RenderSettings(
                                        roi_width_px=settings.roi_width_px,
                                        roi_height_px=settings.roi_height_px,
                                        mm_per_px=settings.mm_per_px,
                                        blender_executable=settings.blender_executable,
                                        cycles_samples=settings.cycles_samples,
                                        device=str(device_override),
                                    )

                                jobs, new_previews = _build_preview_jobs(
                                    profile_id=pid,
                                    profile=prof,
                                    settings=settings,
                                    out_root=out_root,
                                    seed_base=int(seed_base),
                                    defect_types=defect_types,
                                    run_id=run_id,
                                    backend=backend,
                                    rotation_jitter_range=_get_rotation_jitter_range(cfg),
                                    enable_cardinal_rotation_90=bool(_current_filters_from_vars().get("cardinal_rotation_90", True)),
                                    image_filters=_current_filters_from_vars(),
                                )

                                jobs_path = out_root / f"blender_jobs_previews_{_sanitize_dir_name(pid)}.jsonl"
                                write_jobs_jsonl(jobs_path, jobs)
                                render_blender_batch(
                                    sim_root=sim_root,
                                    jobs_path=jobs_path,
                                    output_root=out_root,
                                    blender_executable=settings.blender_executable,
                                    cycles_samples=settings.cycles_samples,
                                    device=settings.device,
                                )
                                _postprocess_rendered_previews(out_root, jobs)
                                all_new_previews.extend(new_previews)
                            else:
                                import cv2  # type: ignore
                                from simple_sim.generators.opencv_2d import render_roi  # type: ignore

                                roi = cfg["roi"]
                                w = int(roi["width_px"])
                                h = int(roi["height_px"])
                                mpp = float(roi["mm_per_px"])
                                if roi_override is not None:
                                    w, h, mpp = int(roi_override[0]), int(roi_override[1]), float(roi_override[2])

                                roi_cfg = {"width_px": int(w), "height_px": int(h), "mm_per_px": float(mpp)}

                                render_cfg = dict(cfg["render"])
                                if "component_color" not in render_cfg and "render" in prof:
                                    render_cfg["component_color"] = prof["render"]["component_color_bgr"]

                                footprint = str((prof.get("component") or {}).get("footprint", "chip_2pad"))
                                geometry_ranges = prof.get("geometry_ranges") or {}
                                tolerances = prof.get("tolerances")
                                rotation_jitter_range = _get_rotation_jitter_range(cfg)

                                profile_dir = _sanitize_dir_name(pid)
                                for defect_type in defect_types:
                                    if run_id is not None:
                                        seed_str = f"{int(seed_base)}|{pid}|{backend}|{str(defect_type)}|{int(run_id)}"
                                    else:
                                        seed_str = f"{int(seed_base)}|{pid}|{backend}|{str(defect_type)}"
                                    hh = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
                                    seed = int(hh[:12], 16)
                                    rng = np.random.default_rng(seed)
                                    augment = _sample_preview_augment(
                                        rng,
                                        rotation_jitter_range=rotation_jitter_range,
                                        enable_cardinal_rotation_90=bool(_current_filters_from_vars().get("cardinal_rotation_90", True)),
                                    )
                                    augment = _apply_filter_overrides_to_augment(augment, _current_filters_from_vars())
                                    nominal = _sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
                                    defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)
                                    img = render_roi(
                                        nominal=nominal,
                                        defect_params=defect,
                                        augment=augment,
                                        roi_size=(int(w), int(h)),
                                        config=render_cfg,
                                        tolerances=tolerances,
                                        rng=rng,
                                        footprint=footprint,
                                    )
                                    rel_path = f"previews/{profile_dir}/{backend}/{str(defect_type)}.png"
                                    out_path = (out_root / rel_path).resolve()
                                    out_path.parent.mkdir(parents=True, exist_ok=True)
                                    ok = cv2.imwrite(str(out_path), img)
                                    if not ok:
                                        raise RuntimeError(f"Failed to write image: {out_path}")
                                    all_new_previews.append((pid, backend, str(defect_type), rel_path))

                    # Update state with all new previews
                    # Remove old previews for all rendered profiles
                    rendered_pids = {pid for pid, _prof, _b in all_profiles}
                    # Remove old previews for all rendered profiles (keep others)
                    state.previews = [rec for rec in state.previews if rec[0] not in rendered_pids]
                    state.previews.extend(all_new_previews)

                    # Update index.html
                    _write_index_html(out_root, state.previews)

                    # Refresh UI
                    root.after(0, lambda: _refresh_images())
                    root.after(0, lambda: status_label.config(text=f"Rendered all {len(all_profiles)} profiles", foreground="#060"))

                except Exception as exc:
                    root.after(0, lambda: status_label.config(text=f"Error: {exc}", foreground="#b00"))
                finally:
                    state.is_rendering = False
                    root.after(0, lambda: render_btn.config(state="normal"))
                    root.after(0, lambda: render_all_btn.config(state="normal"))

            thread = threading.Thread(target=_render_all_thread, daemon=True)
            thread.start()

        render_btn = ttk.Button(controls, text="Render", command=_start_render)
        render_btn.pack(side="left", padx=(0, 5))

        render_all_btn = ttk.Button(controls, text="Render ALL", command=_start_render_all)
        render_all_btn.pack(side="left", padx=(0, 10))

    # Container for images
    container = ttk.Frame(top)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, highlightthickness=0)
    vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)

    vbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas, padding=6)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_inner_configure(_evt=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(evt):
        canvas.itemconfigure(inner_id, width=evt.width)

    inner.bind("<Configure>", _on_inner_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    # Mouse wheel scrolling
    def _on_mousewheel(evt):
        if getattr(evt, "delta", 0):
            canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")

    def _on_button4(_evt):
        canvas.yview_scroll(-3, "units")

    def _on_button5(_evt):
        canvas.yview_scroll(3, "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_button4)
    canvas.bind_all("<Button-5>", _on_button5)

    # Build grid
    thumb_max = 240
    pad = 10
    cols = 4

    photo_refs: List[Any] = []
    zoom_photo_refs: List[Any] = []

    def _open_zoom_popup(path: Path, title: str) -> None:
        try:
            im = Image.open(path)
        except Exception:
            return

        win = tk.Toplevel(root)
        win.title(title)
        win.transient(root)

        sw = max(800, int(root.winfo_screenwidth() * 0.9))
        sh = max(600, int(root.winfo_screenheight() * 0.9))

        w, h = im.size
        scale = min(sw / max(1, w), sh / max(1, h), 1.0)
        vw = max(1, int(w * scale))
        vh = max(1, int(h * scale))
        if scale < 1.0:
            im = im.resize((vw, vh), Image.Resampling.LANCZOS)

        ph = ImageTk.PhotoImage(im)
        zoom_photo_refs.append(ph)

        frm = ttk.Frame(win, padding=8)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, image=ph).pack(fill="both", expand=True)

        # Keep popup size snug to image while respecting screen bounds.
        win.geometry(f"{min(vw + 24, sw)}x{min(vh + 24, sh)}")

    def _refresh_images():
        """Clear and rebuild the image grid."""
        nonlocal photo_refs

        # Clear existing widgets
        for widget in inner.winfo_children():
            widget.destroy()

        photo_refs.clear()

        items = _load_items()

        if not items:
            msg = "No images found yet. Click 'Render' to generate previews."
            ttk.Label(inner, text=msg, foreground="#b00").grid(row=0, column=0, pady=16)
            root.title(f"Simple-Sim Debug Previews (0 images)")
            return

        root.title(f"Simple-Sim Debug Previews ({len(items)} images)")

        items.sort(key=lambda t: (t[0], t[1], t[2]))
        for idx, (pid, backend, defect, img_path) in enumerate(items):
            r = idx // cols
            c = idx % cols

            card = ttk.Frame(inner, padding=pad)
            card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)

            cap = f"{pid}\n{backend}\n{defect}"
            ttk.Label(card, text=cap, justify="left").pack(anchor="w")

            try:
                im = Image.open(img_path)
                im.thumbnail((thumb_max, thumb_max))
                ph = ImageTk.PhotoImage(im)
            except Exception as exc:
                ttk.Label(card, text=f"[failed to load]\n{img_path.name}\n{exc}", foreground="#b00").pack()
                continue

            photo_refs.append(ph)
            lbl = ttk.Label(card, image=ph)
            lbl.pack()

            lbl.bind(
                "<Button-1>",
                lambda _e, p=img_path, t=f"{pid} | {backend} | {defect}": _open_zoom_popup(p, t),
            )

        for c in range(cols):
            inner.grid_columnconfigure(c, weight=1)

    # Initial load
    _refresh_images()

    root.mainloop()


def _build_preview_jobs(
    *,
    profile_id: str,
    profile: Dict[str, Any],
    settings: RenderSettings,
    out_root: Path,
    seed_base: int,
    defect_types: Sequence[str],
    run_id: Optional[int] = None,
    backend: str = BACKEND_3D,
    rotation_jitter_range: Tuple[float, float] = (0.0, 0.0),
    enable_cardinal_rotation_90: bool = True,
    image_filters: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[PreviewRec]]:
    """Return (jobs, preview_records). preview_records is for index.html.

    If run_id is provided, it will be included in seed generation to create
    different images on each run (like the real 3D pipeline).
    """
    footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
    geometry_ranges = profile.get("geometry_ranges") or {}
    tolerances = profile.get("tolerances") or {}
    component_height_mm = float(((profile.get("component") or {}).get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45)
    render_3d = profile.get("render_3d") or {}

    roi_cfg = {
        "width_px": int(settings.roi_width_px),
        "height_px": int(settings.roi_height_px),
        "mm_per_px": float(settings.mm_per_px),
    }

    jobs: List[Dict[str, Any]] = []
    previews: List[PreviewRec] = []

    profile_dir = _sanitize_dir_name(profile_id)
    for i, defect_type in enumerate(defect_types):
        # Include run_id to vary seeds across reruns (like real 3D pipeline)
        if run_id is not None:
            seed_str = f"{int(seed_base)}|{profile_id}|{str(defect_type)}|{int(run_id)}"
        else:
            seed_str = f"{int(seed_base)}|{profile_id}|{str(defect_type)}"
        h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        seed = int(h[:12], 16)
        rng = np.random.default_rng(seed)

        nominal = _sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
        defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)
        augment = _sample_preview_augment(
            rng,
            rotation_jitter_range=rotation_jitter_range,
            enable_cardinal_rotation_90=enable_cardinal_rotation_90,
        )
        augment = _apply_filter_overrides_to_augment(augment, image_filters)

        rel_path = f"previews/{profile_dir}/{backend}/{str(defect_type)}.png"
        job = {
            "image_path": rel_path,
            "seed": int(seed),
            "mm_per_px": float(settings.mm_per_px),
            "roi_width_px": int(settings.roi_width_px),
            "roi_height_px": int(settings.roi_height_px),
            "footprint": str(footprint),
            "component_height_mm": float(component_height_mm),
            "nominal": nominal,
            "defect": defect,
            "augment": augment,
            "render_3d": render_3d,
        }
        jobs.append(job)
        previews.append((profile_id, backend, str(defect_type), rel_path))

    return jobs, previews


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-profile debug previews (2D OpenCV and/or 3D Blender)")
    parser.add_argument("--profiles-dir", default="configs/profiles", help="Directory with component profile YAMLs")
    parser.add_argument("--configs-dir", default="configs", help="Directory with run_*.yaml configs (for auto matching)")
    parser.add_argument("--out", default="outputs/debug_previews", help="Output directory root")
    parser.add_argument("--settings-file", default="", help="Shared GUI settings JSON (default: outputs/gui/settings.json)")

    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto", help="Render backend: opencv_2d|blender_3d|both|auto")
    parser.add_argument("--profiles", default="", help="Comma-separated profile IDs to render")
    parser.add_argument("--all", action="store_true", help="Render all discovered profiles (within backend selection)")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt; auto-renders new/incomplete profiles when --profiles/--all is omitted")

    parser.add_argument("--config", default="", help="Optional run config YAML to use for ALL selected profiles (must match backend unless --backend=both)")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed for deterministic previews")
    parser.add_argument("--variable-seeds", action="store_true", help="Generate different images on each run (like real pipeline)")
    parser.add_argument(
        "--disable-cardinal-rotation-90",
        action="store_true",
        help="Disable random 0/90/180/270 base orientation in previews (keep only rotation_deg_range jitter)",
    )
    parser.add_argument("--states", default="", help="Comma-separated defect states to render (default: from profile.defect_set)")

    parser.add_argument("--blender", default="", help="(3D) Override blender executable (else from config)")
    parser.add_argument("--samples", type=int, default=0, help="(3D) Override Cycles samples (else from config)")
    parser.add_argument("--device", default="", help="(3D) Override device (CPU/GPU; else from config)")

    parser.add_argument("--roi", default="", help="Override ROI as WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")
    parser.add_argument("--dry-run", action="store_true", help="Only write index/jobs; do not render (2D or 3D)")
    parser.add_argument("--view", choices=["tk", "browser", "none"], default="tk", help="How to open previews after run (default: tk)")
    # Back-compat flags
    open_group = parser.add_mutually_exclusive_group()
    open_group.add_argument("--open", dest="open_browser", action="store_true", help="(legacy) open in browser after run")
    open_group.add_argument("--no-open", dest="open_browser", action="store_false", help="(legacy) do not auto-open after run")
    parser.set_defaults(open_browser=None)
    args = parser.parse_args()

    sim_root = Path(__file__).parent.parent
    profiles_dir = (sim_root / args.profiles_dir).resolve()
    configs_dir = (sim_root / args.configs_dir).resolve()
    out_root = (sim_root / args.out).resolve()
    settings_file = _shared_settings_path(sim_root, args.settings_file)
    shared_profiles, shared_active_profile, shared_filter_settings = _load_shared_filter_profiles(settings_file)

    discovered_all = _discover_profiles(profiles_dir, backend_mode="both")
    if not discovered_all:
        raise SystemExit(f"No profiles found under: {profiles_dir}")

    # Load settings: either one config for all, or auto match per profile/backend.
    forced_cfg: Optional[Dict[str, Any]] = None
    forced_backend: Optional[str] = None
    if args.config:
        forced_cfg = load_config(sim_root / args.config)
        validate_config(forced_cfg)
        forced_backend = str((forced_cfg.get("render") or {}).get("backend", ""))

    # Optional ROI override parsing
    roi_override: Optional[Tuple[int, int, float]] = None
    if args.roi:
        m = re.match(r"^\\s*(\\d+)x(\\d+)@([0-9]*\\.?[0-9]+)\\s*$", str(args.roi))
        if not m:
            raise SystemExit("Invalid --roi. Expected WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")
        roi_override = (int(m.group(1)), int(m.group(2)), float(m.group(3)))

    # Resolve backend mode
    backend_mode = str(args.backend or "auto")
    if backend_mode == "auto" and forced_backend:
        backend_mode = forced_backend
    if backend_mode != "auto" and forced_backend and backend_mode not in {"both", forced_backend}:
        raise SystemExit(f"--backend={backend_mode!r} conflicts with --config backend={forced_backend!r}")

    def _prompt_backend() -> str:
        print("\nSelect backend mode:\n")
        print(f"  1. 3D ({BACKEND_3D})")
        print(f"  2. 2D ({BACKEND_2D})")
        print("  3. Both (default)")
        print("\nPress Enter for default, or enter your choice:")
        sel = input("> ").strip().lower()
        if not sel:
            return "both"
        if sel in {"1", "3d", BACKEND_3D}:
            return BACKEND_3D
        if sel in {"2", "2d", BACKEND_2D}:
            return BACKEND_2D
        if sel in {"3", "both", "b"}:
            return "both"
        raise SystemExit("Invalid backend selection.")

    if backend_mode == "auto" and not args.non_interactive:
        backend_mode = _prompt_backend()
    elif backend_mode == "auto":
        backend_mode = "both"

    # Filter discovered profiles for CLI selection list
    if backend_mode in {BACKEND_2D, BACKEND_3D}:
        discovered = [(pid, prof, b) for (pid, prof, b) in discovered_all if backend_mode in b]
    else:
        discovered = list(discovered_all)

    if not discovered:
        raise SystemExit(f"No profiles match backend={backend_mode!r} under: {profiles_dir}")

    # Determine selected profile ids
    selected_ids: List[str] = []
    if args.all:
        selected_ids = [pid for pid, _prof, _b in discovered]
    else:
        selected_ids = _parse_csv_list(args.profiles)

    # Auto mode: if nothing explicitly selected, render profiles with missing previews.
    if not selected_ids:
        selected_ids = _auto_select_profiles_with_missing_previews(
            discovered,
            profiles_dir=profiles_dir,
            out_root=out_root,
            backend_mode=backend_mode,
            states_override=str(args.states or ""),
        )
        if selected_ids:
            print(f"[auto] Rendering {len(selected_ids)} new/incomplete profile(s): {', '.join(selected_ids)}")

    if not selected_ids and not args.non_interactive:
        print("\nAvailable profiles:\n")
        for i, (pid, prof, b) in enumerate(discovered, 1):
            desc = str((prof.get("profile") or {}).get("description", "") or "")
            b_str = "/".join(sorted(b))
            print(f"{i:2d}. {pid}  [{b_str}]" + (f"  ({desc})" if desc else ""))
        print("\nSelect profiles by number (e.g. 1,3-4) or 'all' (default):")
        print("Press Enter for all profiles, or enter your selection:")
        sel = input("> ").strip()
        if not sel:
            sel = "all"
        idxs = _parse_selection(sel, len(discovered))
        selected_ids = [discovered[i][0] for i in idxs]

    if not selected_ids:
        if args.non_interactive:
            # Keep index/view in sync even when nothing new needs rendering.
            existing = _collect_existing_previews(out_root)
            if existing:
                _write_index_html(out_root, existing)
                print(f"[auto] No new/incomplete profiles found. Reusing {len(existing)} existing preview(s).")
                if args.view == "browser":
                    import webbrowser
                    webbrowser.open((out_root / "index.html").resolve().as_uri())
                elif args.view == "tk":
                    viewer_render_settings = {
                        "forced_cfg": forced_cfg,
                        "seed_base": int(args.seed),
                        "states": args.states,
                        "enable_cardinal_rotation_90": bool(shared_filter_settings.get("cardinal_rotation_90", True)) if not bool(args.disable_cardinal_rotation_90) else False,
                        "blender": args.blender,
                        "samples": args.samples,
                        "device": args.device,
                        "roi_override": roi_override,
                        "backend_mode": backend_mode,
                        "settings_file": str(settings_file),
                    }
                    _open_tk_viewer(
                        out_root,
                        existing,
                        sim_root=sim_root,
                        profiles_dir=profiles_dir,
                        configs_dir=configs_dir,
                        all_profiles=discovered_all,
                        render_settings=viewer_render_settings,
                        selected_profile_ids=None,  # Auto mode, no explicit selection
                    )
            else:
                print("[auto] No new/incomplete profiles found and no existing previews available.")
            return
        raise SystemExit("No profiles selected and no new/incomplete profiles found.")

    pid_to_entry: Dict[str, Tuple[str, Dict[str, Any], set[str]]] = {pid: (pid, prof, b) for pid, prof, b in discovered_all}
    missing = [pid for pid in selected_ids if pid not in pid_to_entry]
    if missing:
        raise SystemExit(f"Unknown profiles: {missing}")

    selected_entries: List[Tuple[str, Dict[str, Any], set[str]]] = []
    for pid in selected_ids:
        _pid, prof, b = pid_to_entry[pid]
        selected_entries.append((pid, prof, b))

    previews_for_index: List[PreviewRec] = []
    all_jobs: List[Dict[str, Any]] = []
    per_profile_settings: List[RenderSettings] = []

    # Translate legacy flags into --view behavior if explicitly set.
    if args.open_browser is True:
        args.view = "browser"
    elif args.open_browser is False:
        args.view = "none"

    out_root.mkdir(parents=True, exist_ok=True)

    for pid, prof, backends in selected_entries:
        defect_types = _parse_csv_list(args.states) or list(prof.get("defect_set") or [])
        if not defect_types:
            defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

        run_id = int(time.time() * 1000000) if args.variable_seeds else None
        if backend_mode == "both":
            backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in backends]
        else:
            backends_to_render = [backend_mode] if backend_mode in backends else []
        if not backends_to_render:
            raise SystemExit(f"Profile {pid!r} does not support backend={backend_mode!r}")

        for backend in backends_to_render:
            if forced_cfg is not None:
                cfg = forced_cfg
                cfg_src = f"(forced) {args.config}"
                cfg_backend = str((cfg.get("render") or {}).get("backend", ""))
                if backend != "both" and cfg_backend != backend:
                    raise SystemExit(f"--config backend={cfg_backend!r} does not match requested backend={backend!r}")
            else:
                matches = _find_matching_run_configs(configs_dir, profile_id=pid, backend=backend)
                if matches:
                    cfg = load_config(matches[0])
                    try:
                        rel = str(matches[0].relative_to(sim_root))
                    except Exception:
                        rel = str(matches[0])
                    cfg_src = f"(auto) {rel}"
                else:
                    fallback = "configs/run_0001.yaml" if backend == BACKEND_2D else "configs/run_0001_3d.yaml"
                    cfg = load_config(sim_root / fallback)
                    cfg_src = f"(fallback) {fallback}"
                validate_config(cfg)

            if backend == BACKEND_2D:
                roi = cfg["roi"]
                w = int(roi["width_px"])
                h = int(roi["height_px"])
                mpp = float(roi["mm_per_px"])
                if roi_override is not None:
                    w, h, mpp = int(roi_override[0]), int(roi_override[1]), float(roi_override[2])

                print(f"\n[preview] Profile:  {pid}")
                print(f"[preview] Backend:  {backend}")
                print(f"[preview] Config:    {cfg_src}")
                print(f"[preview] ROI:       {w}x{h} @ {mpp} mm/px")
                print(f"[preview] States:    {defect_types}")

                profile_dir = _sanitize_dir_name(pid)
                for defect_type in defect_types:
                    rel_path = f"previews/{profile_dir}/{backend}/{str(defect_type)}.png"
                    previews_for_index.append((pid, backend, str(defect_type), rel_path))

                if args.dry_run:
                    continue

                import cv2  # type: ignore
                from simple_sim.generators.opencv_2d import render_roi  # type: ignore

                roi_cfg = {"width_px": int(w), "height_px": int(h), "mm_per_px": float(mpp)}
                render_cfg = dict(cfg["render"])
                if "component_color" not in render_cfg and "render" in prof:
                    render_cfg["component_color"] = prof["render"]["component_color_bgr"]
                footprint = str((prof.get("component") or {}).get("footprint", "chip_2pad"))
                geometry_ranges = prof.get("geometry_ranges") or {}
                tolerances = prof.get("tolerances")
                rotation_jitter_range = _get_rotation_jitter_range(cfg)

                out_root.mkdir(parents=True, exist_ok=True)
                for defect_type in defect_types:
                    if run_id is not None:
                        seed_str = f"{int(args.seed)}|{pid}|{backend}|{str(defect_type)}|{int(run_id)}"
                    else:
                        seed_str = f"{int(args.seed)}|{pid}|{backend}|{str(defect_type)}"
                    hh = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
                    seed = int(hh[:12], 16)
                    rng = np.random.default_rng(seed)
                    augment = _sample_preview_augment(
                        rng,
                        rotation_jitter_range=rotation_jitter_range,
                        enable_cardinal_rotation_90=(not bool(args.disable_cardinal_rotation_90) and bool(shared_filter_settings.get("cardinal_rotation_90", True))),
                    )
                    augment = _apply_filter_overrides_to_augment(augment, shared_filter_settings)
                    nominal = _sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
                    defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)
                    img = render_roi(
                        nominal=nominal,
                        defect_params=defect,
                        augment=augment,
                        roi_size=(int(w), int(h)),
                        config=render_cfg,
                        tolerances=tolerances,
                        rng=rng,
                        footprint=footprint,
                    )
                    out_path = (out_root / f"previews/{profile_dir}/{backend}/{str(defect_type)}.png").resolve()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    ok = cv2.imwrite(str(out_path), img)
                    if not ok:
                        raise SystemExit(f"Failed to write image: {out_path}")
            else:
                settings = _settings_from_config(cfg)
                if roi_override is not None:
                    w, h, mpp = roi_override
                    settings = RenderSettings(
                        roi_width_px=int(w),
                        roi_height_px=int(h),
                        mm_per_px=float(mpp),
                        blender_executable=settings.blender_executable,
                        cycles_samples=settings.cycles_samples,
                        device=settings.device,
                    )

                if args.blender:
                    settings = RenderSettings(
                        roi_width_px=settings.roi_width_px,
                        roi_height_px=settings.roi_height_px,
                        mm_per_px=settings.mm_per_px,
                        blender_executable=str(args.blender),
                        cycles_samples=settings.cycles_samples,
                        device=settings.device,
                    )
                if args.samples and int(args.samples) > 0:
                    settings = RenderSettings(
                        roi_width_px=settings.roi_width_px,
                        roi_height_px=settings.roi_height_px,
                        mm_per_px=settings.mm_per_px,
                        blender_executable=settings.blender_executable,
                        cycles_samples=int(args.samples),
                        device=settings.device,
                    )
                if args.device:
                    settings = RenderSettings(
                        roi_width_px=settings.roi_width_px,
                        roi_height_px=settings.roi_height_px,
                        mm_per_px=settings.mm_per_px,
                        blender_executable=settings.blender_executable,
                        cycles_samples=settings.cycles_samples,
                        device=str(args.device),
                    )

                print(f"\n[preview] Profile:  {pid}")
                print(f"[preview] Backend:  {backend}")
                print(f"[preview] Config:    {cfg_src}")
                print(f"[preview] ROI:       {settings.roi_width_px}x{settings.roi_height_px} @ {settings.mm_per_px} mm/px")
                print(f"[preview] Blender:   {settings.blender_executable} (samples={settings.cycles_samples}, device={settings.device})")
                print(f"[preview] States:    {defect_types}")

                per_profile_settings.append(settings)
                jobs, previews = _build_preview_jobs(
                    profile_id=pid,
                    profile=prof,
                    settings=settings,
                    out_root=out_root,
                    seed_base=int(args.seed),
                    defect_types=defect_types,
                    run_id=run_id,
                    backend=backend,
                    rotation_jitter_range=_get_rotation_jitter_range(cfg),
                    enable_cardinal_rotation_90=(not bool(args.disable_cardinal_rotation_90) and bool(shared_filter_settings.get("cardinal_rotation_90", True))),
                    image_filters=shared_filter_settings,
                )
                all_jobs.extend(jobs)
                previews_for_index.extend(previews)

    jobs_path = out_root / "blender_jobs_previews.jsonl"
    if all_jobs:
        write_jobs_jsonl(jobs_path, all_jobs)
    _write_index_html(out_root, previews_for_index)

    viewer_render_settings = {
        "forced_cfg": forced_cfg,
        "seed_base": int(args.seed),
        "states": args.states,
        "enable_cardinal_rotation_90": (not bool(args.disable_cardinal_rotation_90) and bool(shared_filter_settings.get("cardinal_rotation_90", True))),
        "blender": args.blender,
        "samples": args.samples,
        "device": args.device,
        "roi_override": roi_override,
        "backend_mode": backend_mode,
        "settings_file": str(settings_file),
    }

    if args.dry_run:
        if all_jobs:
            print(f"\n[dry-run] Wrote jobs:  {jobs_path}")
        print(f"[dry-run] Wrote index: {out_root / 'index.html'}")
        if args.view == "browser":
            import webbrowser
            webbrowser.open((out_root / "index.html").resolve().as_uri())
        elif args.view == "tk":
            _open_tk_viewer(
                out_root,
                previews_for_index,
                sim_root=sim_root,
                profiles_dir=profiles_dir,
                configs_dir=configs_dir,
                all_profiles=discovered_all,
                render_settings=viewer_render_settings,
                selected_profile_ids=selected_ids,
            )
        return

    if all_jobs:
        # One Blender invocation for all selected 3D jobs (fast).
        exes = sorted({s.blender_executable for s in per_profile_settings} or {"blender"})
        devices = sorted({s.device for s in per_profile_settings} or {"CPU"})
        samples_list = sorted({int(s.cycles_samples) for s in per_profile_settings} or {64})

        if len(exes) > 1 and not args.blender:
            print(f"\n[warn] Multiple blender executables across auto configs: {exes} (using: {exes[-1]})")
        if len(devices) > 1 and not args.device:
            print(f"[warn] Multiple devices across auto configs: {devices} (using: {devices[-1]})")
        if len(samples_list) > 1 and not args.samples:
            print(f"[warn] Multiple sample counts across auto configs: {samples_list} (using max: {max(samples_list)})")

        blender_executable = str(args.blender or exes[-1])
        device = str(args.device or devices[-1])
        cycles_samples = int(args.samples or max(samples_list))

        print(f"\n[render] Running Blender batch for {len(all_jobs)} previews...")
        render_blender_batch(
            sim_root=sim_root,
            jobs_path=jobs_path,
            output_root=out_root,
            blender_executable=blender_executable,
            cycles_samples=cycles_samples,
            device=device,
        )
        _postprocess_rendered_previews(out_root, all_jobs)

    index_path = (out_root / "index.html").resolve()
    print(f"[render] Done. Open: {index_path}")
    if args.view == "browser":
        import webbrowser
        webbrowser.open(index_path.as_uri())
    elif args.view == "tk":
        _open_tk_viewer(
            out_root,
            previews_for_index,
            sim_root=sim_root,
            profiles_dir=profiles_dir,
            configs_dir=configs_dir,
            all_profiles=discovered_all,
            render_settings=viewer_render_settings,
            selected_profile_ids=selected_ids,
        )


if __name__ == "__main__":
    main()
