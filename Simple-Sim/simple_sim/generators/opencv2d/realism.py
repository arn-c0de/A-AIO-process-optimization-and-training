"""Realism-mode grouped augmentation sampler.

Design goals:
- G0 always-on mild baseline.
- Sparse group mixing (K in {0,1,2}).
- Correlated, physically plausible group effects.
- Optional compatibility constraints.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence
from pathlib import Path

import numpy as np


_HEAVY_GROUPS = {"G2", "G5", "G7"}
_GROUPS: Sequence[str] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")

_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    # Baseline from issue #19
    "profile_industrial_cam": {
        "k0": 0.60,
        "k1": 0.35,
        "k2": 0.05,
        "G1": 0.35,
        "G2": 0.15,
        "G3": 0.20,
        "G4": 0.25,
        "G5": 0.12,
        "G6": 0.18,
        "G7": 0.10,
        "G8": 0.08,
    },
    # Slightly noisier/mobile-like variant
    "profile_mobile_phone": {
        "k0": 0.52,
        "k1": 0.40,
        "k2": 0.08,
        "G1": 0.30,
        "G2": 0.20,
        "G3": 0.16,
        "G4": 0.30,
        "G5": 0.10,
        "G6": 0.25,
        "G7": 0.14,
        "G8": 0.12,
    },
}
_PRESET_CACHE: Dict[str, Dict[str, float]] | None = None
_PRESET_CACHE_PATH: str | None = None


def _tri_sample(rng: np.random.Generator, min_val: float, nom_val: float, max_val: float) -> float:
    if rng.random() < 0.80:
        delta = (max_val - min_val) * 0.15
        lo = max(min_val, nom_val - delta)
        hi = min(max_val, nom_val + delta)
        return float(rng.triangular(lo, nom_val, hi))
    if rng.random() < 0.95:
        return float(rng.uniform(nom_val, max_val))
    return float(max_val)


def _sample_k(rng: np.random.Generator, p0: float, p1: float, p2: float) -> int:
    r = rng.random()
    if r < p0:
        return 0
    if r < (p0 + p1):
        return 1
    return 2


def _weighted_sample_no_replace(rng: np.random.Generator, candidates: Sequence[str], probs: Dict[str, float], k: int) -> List[str]:
    pool = list(candidates)
    out: List[str] = []
    while pool and len(out) < k:
        weights = np.array([max(0.0, float(probs.get(g, 0.0))) for g in pool], dtype=float)
        s = float(weights.sum())
        if s <= 1e-12:
            break
        weights /= s
        idx = int(rng.choice(len(pool), p=weights))
        out.append(pool.pop(idx))
    return out


def _choose_groups(rng: np.random.Generator, probs: Dict[str, float], k: int) -> List[str]:
    if k <= 0:
        return []
    gated = [g for g in _GROUPS if rng.random() < float(probs.get(g, 0.0))]
    if len(gated) >= k:
        return _weighted_sample_no_replace(rng, gated, probs, k)
    selected = list(gated)
    if len(selected) < k:
        remaining = [g for g in _GROUPS if g not in selected]
        selected.extend(_weighted_sample_no_replace(rng, remaining, probs, k - len(selected)))
    return selected[:k]


def _default_preset_path() -> Path:
    # .../Simple-Sim/simple_sim/generators/opencv2d/realism.py -> Simple-Sim/configs/realism_profiles.yaml
    return Path(__file__).resolve().parents[3] / "configs" / "realism_profiles.yaml"


def _normalize_yaml_preset(raw: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, float]:
    out = dict(fallback)
    try:
        out["k0"] = float(raw.get("k0", out["k0"]))
        out["k1"] = float(raw.get("k1", out["k1"]))
        out["k2"] = float(raw.get("k2", out["k2"]))
        groups = raw.get("groups", {})
        if isinstance(groups, dict):
            for g in _GROUPS:
                if g in groups:
                    out[g] = float(groups[g])
                elif f"{g}_prob" in groups:
                    out[g] = float(groups[f"{g}_prob"])
        for g in _GROUPS:
            key = f"{g}_prob"
            if key in raw:
                out[g] = float(raw[key])
            elif g in raw:
                out[g] = float(raw[g])
    except Exception:
        return dict(fallback)
    return out


def _load_profile_presets() -> Dict[str, Dict[str, float]]:
    global _PRESET_CACHE, _PRESET_CACHE_PATH

    cfg_path = str(os.environ.get("SIMPLE_SIM_REALISM_PRESETS_PATH", "")).strip()
    if not cfg_path:
        cfg_path = str(_default_preset_path())

    if _PRESET_CACHE is not None and _PRESET_CACHE_PATH == cfg_path:
        return _PRESET_CACHE

    presets = dict(_PROFILE_PRESETS)
    p = Path(cfg_path)
    if p.exists():
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            profiles = obj.get("profiles", obj) if isinstance(obj, dict) else {}
            if isinstance(profiles, dict):
                for pid, raw in profiles.items():
                    if not isinstance(pid, str) or not isinstance(raw, dict):
                        continue
                    base = presets.get(
                        pid,
                        presets.get("profile_industrial_cam", _PROFILE_PRESETS["profile_industrial_cam"]),
                    )
                    presets[pid] = _normalize_yaml_preset(raw, base)
        except Exception:
            pass

    _PRESET_CACHE = presets
    _PRESET_CACHE_PATH = cfg_path
    return presets


def _resolve_policy(filt: Dict[str, Any]) -> Dict[str, float]:
    src_keys = set(filt.get("__source_keys", set()) or set())
    pid = str(filt.get("realism_profile_id", "profile_industrial_cam") or "profile_industrial_cam")
    presets = _load_profile_presets()
    preset = presets.get(pid, presets.get("profile_industrial_cam", _PROFILE_PRESETS["profile_industrial_cam"]))
    out = dict(preset)
    if "realism_k_prob_0" in src_keys:
        out["k0"] = float(filt.get("realism_k_prob_0", out["k0"]))
    if "realism_k_prob_1" in src_keys:
        out["k1"] = float(filt.get("realism_k_prob_1", out["k1"]))
    if "realism_k_prob_2" in src_keys:
        out["k2"] = float(filt.get("realism_k_prob_2", out["k2"]))
    for g in _GROUPS:
        k = f"realism_group_{g}_prob"
        if k in src_keys:
            out[g] = float(filt.get(k, out[g]))
    s = out["k0"] + out["k1"] + out["k2"]
    if s > 1e-12:
        out["k0"] /= s
        out["k1"] /= s
        out["k2"] /= s
    else:
        out["k0"], out["k1"], out["k2"] = 0.60, 0.35, 0.05
    return out


def _apply_constraints(selected: List[str]) -> List[str]:
    out = list(selected)
    if "G5" in out and "G7" in out:
        out.remove("G7")
    heavies = [g for g in out if g in _HEAVY_GROUPS]
    if len(heavies) > 1:
        keep = heavies[0]
        out = [g for g in out if g not in _HEAVY_GROUPS or g == keep]
    return out


def apply_realism_groups(augment: Dict[str, float], filt: Dict[str, Any]) -> Dict[str, float]:
    """Apply correlated realism groups on top of sampled augment values."""
    aug = dict(augment or {})
    rng = np.random.default_rng()
    constraints_enabled = bool(filt.get("realism_constraints_enabled", True))
    policy = _resolve_policy(filt)

    # G0: always-on mild day-to-day drift.
    aug["blur_sigma"] = _tri_sample(rng, 0.10, 0.50, 1.20)
    aug["noise_stddev"] = _tri_sample(rng, 1.0, 3.0, 6.0)
    aug["brightness_factor"] = _tri_sample(rng, 0.96, 1.00, 1.05)
    aug["contrast_factor"] = _tri_sample(rng, 0.96, 1.00, 1.06)
    aug["rotation_deg"] = _tri_sample(rng, -4.0, 0.0, 4.0)
    aug["perspective_strength"] = 0.0
    aug["motion_blur_strength"] = 0.0
    aug["vignetting_strength"] = 0.0
    aug["reflection_strength"] = 0.0
    aug["chromatic_strength"] = 0.0
    aug["dust_density"] = 0.0
    aug["hue_shift_deg"] = 0.0
    aug["sharpen_strength"] = 0.0
    aug["distortion_k1"] = 0.0
    aug["distortion_k2"] = 0.0
    aug["jpeg_quality"] = 100
    aug["color_temperature_kelvin"] = 5500
    aug["shadow_strength"] = 0.0

    k = _sample_k(rng, policy["k0"], policy["k1"], policy["k2"])
    probs = {g: policy[g] for g in _GROUPS}
    selected = _choose_groups(rng, probs, k)
    if constraints_enabled:
        selected = _apply_constraints(selected)

    for g in selected:
        if g == "G1":
            # Focus / optics softness
            aug["blur_sigma"] = _tri_sample(rng, 0.4, 1.2, 2.0)
            aug["sharpen_strength"] = _tri_sample(rng, 0.0, 0.1, 0.3) if rng.random() < 0.10 else 0.0
        elif g == "G2":
            # Motion / conveyor
            aug["motion_blur_strength"] = _tri_sample(rng, 0.4, 1.0, 1.8)
            aug["motion_blur_angle"] = float(rng.uniform(0.0, 360.0))
            aug["noise_stddev"] = float(max(aug.get("noise_stddev", 0.0), _tri_sample(rng, 2.0, 4.0, 8.0)))
            aug["brightness_factor"] *= _tri_sample(rng, 0.94, 0.98, 1.02)
            if constraints_enabled and "G1" in selected:
                aug["blur_sigma"] = float(min(aug.get("blur_sigma", 0.0), 1.0))
        elif g == "G3":
            # Geometry / mounting
            aug["perspective_strength"] = _tri_sample(rng, 0.2, 0.5, 1.0)
            aug["perspective_angle_x"] = _tri_sample(rng, -5.0, 0.0, 5.0)
            aug["perspective_angle_y"] = _tri_sample(rng, -5.0, 0.0, 5.0)
            if rng.random() < 0.10:
                aug["distortion_k1"] = _tri_sample(rng, -0.03, 0.0, 0.03)
        elif g == "G4":
            # Illumination non-uniformity
            aug["shadow_strength"] = _tri_sample(rng, 0.10, 0.25, 0.45)
            aug["shadow_size"] = _tri_sample(rng, 0.10, 0.20, 0.35)
            aug["vignetting_strength"] = _tri_sample(rng, 0.05, 0.25, 0.50)
            if constraints_enabled and "G3" in selected:
                aug["shadow_strength"] = float(min(aug.get("shadow_strength", 0.0), 0.40))
        elif g == "G5":
            # Specular / glare
            aug["reflection_strength"] = _tri_sample(rng, 0.10, 0.30, 0.55)
            aug["reflection_size"] = _tri_sample(rng, 0.05, 0.12, 0.20)
            aug["contrast_factor"] *= _tri_sample(rng, 0.90, 0.96, 1.0)
        elif g == "G6":
            # Color pipeline / white balance
            aug["color_temperature_kelvin"] = int(round(_tri_sample(rng, 3200.0, 5500.0, 7000.0)))
            aug["saturation_factor"] = _tri_sample(rng, 0.90, 1.00, 1.10)
            if rng.random() < 0.10:
                aug["hue_shift_deg"] = _tri_sample(rng, -3.0, 0.0, 3.0)
        elif g == "G7":
            # Compression / transport
            aug["jpeg_quality"] = int(round(_tri_sample(rng, 70.0, 88.0, 95.0)))
            if rng.random() < 0.10:
                aug["chromatic_strength"] = _tri_sample(rng, 0.05, 0.10, 0.20)
        elif g == "G8":
            # Contamination
            aug["dust_density"] = _tri_sample(rng, 0.05, 0.15, 0.30)
            aug["dust_size"] = _tri_sample(rng, 1.0, 2.0, 3.0)
            aug["contrast_factor"] *= _tri_sample(rng, 0.90, 0.97, 1.0)

    if constraints_enabled and "G5" in selected:
        aug["jpeg_quality"] = 100

    return aug
