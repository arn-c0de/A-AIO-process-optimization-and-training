"""Realism-mode grouped augmentation sampler.

Design goals:
- G0 always-on mild baseline.
- Sparse group mixing (K in {0,1,2}).
- Correlated, physically plausible group effects.
- Optional compatibility constraints.

Group probabilities come from a preset (``realism_profile_id`` of the image filters). Presets are
the built-in :data:`BUILTIN_PRESETS` merged with a YAML file (``configs/realism_profiles.yaml``);
:func:`load_realism_presets` reads one, and callers may pass their own mapping as ``presets`` to
:func:`apply_realism_groups` (library use without a checkout). Without it the default file is used
(``$SIMPLE_SIM_REALISM_PRESETS_PATH`` overrides its path), read once per path and modification time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.filter_settings import NEUTRAL_AUGMENT
from simple_sim.io_utils import load_yaml

log = logging.getLogger(__name__)


_HEAVY_GROUPS = {"G2", "G5", "G7"}
_GROUPS: Sequence[str] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
# Group-driven effects that G0 resets to neutral before the selected groups re-enable them.
_G0_RESET_KEYS: Sequence[str] = (
    "perspective_strength",
    "motion_blur_strength",
    "vignetting_strength",
    "reflection_strength",
    "chromatic_strength",
    "dust_density",
    "hue_shift_deg",
    "sharpen_strength",
    "distortion_k1",
    "distortion_k2",
    "jpeg_quality",
    "color_temperature_kelvin",
    "shadow_strength",
)

BUILTIN_PRESETS: dict[str, dict[str, float]] = {
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
DEFAULT_PRESET = "profile_industrial_cam"
RealismPresets = Mapping[str, Mapping[str, float]]


def _tri_sample(rng: np.random.Generator, min_val: float, nom_val: float, max_val: float) -> float:
    if rng.random() < 0.80:
        delta = (max_val - min_val) * 0.15
        lo = max(min_val, nom_val - delta)
        hi = min(max_val, nom_val + delta)
        return float(rng.triangular(lo, nom_val, hi))
    # Tail: towards max for magnitudes (blur, noise, ...). A signed range (rotation, perspective,
    # distortion, hue shift: min < nominal < max around 0) takes either side; one-sided it biased
    # those to positive values (rotation jitter [-4, 4]: mean +0.4 deg, never below -1.2).
    end = max_val
    if min_val < 0.0 < max_val and min_val < nom_val and rng.random() < 0.5:
        end = min_val
    if rng.random() < 0.95:
        return float(rng.uniform(*sorted((nom_val, end))))
    return float(end)


def _sample_k(rng: np.random.Generator, p0: float, p1: float, p2: float) -> int:
    r = rng.random()
    if r < p0:
        return 0
    if r < (p0 + p1):
        return 1
    return 2


def _weighted_sample_no_replace(rng: np.random.Generator, candidates: Sequence[str], probs: dict[str, float], k: int) -> list[str]:
    pool = list(candidates)
    out: list[str] = []
    while pool and len(out) < k:
        weights = np.array([max(0.0, float(probs.get(g, 0.0))) for g in pool], dtype=float)
        s = float(weights.sum())
        if s <= 1e-12:
            break
        weights /= s
        idx = int(rng.choice(len(pool), p=weights))
        out.append(pool.pop(idx))
    return out


def _choose_groups(rng: np.random.Generator, probs: dict[str, float], k: int) -> list[str]:
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


def default_presets_path() -> Path:
    """``$SIMPLE_SIM_REALISM_PRESETS_PATH`` or the checkout's ``configs/realism_profiles.yaml``."""
    env = str(os.environ.get("SIMPLE_SIM_REALISM_PRESETS_PATH", "")).strip()
    # .../Simple-Sim/simple_sim/generators/opencv2d/realism.py -> Simple-Sim/configs/realism_profiles.yaml
    return Path(env) if env else Path(__file__).resolve().parents[3] / "configs" / "realism_profiles.yaml"


def _normalize_yaml_preset(raw: dict[str, Any], fallback: Mapping[str, float]) -> dict[str, float]:
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
    except (TypeError, ValueError) as exc:
        log.warning(f"Ignoring invalid realism preset values ({exc})")
        return dict(fallback)
    return out


def load_realism_presets(path: Path | str | None) -> dict[str, dict[str, float]]:
    """:data:`BUILTIN_PRESETS` merged with the ``profiles`` of the YAML file *path* (``None`` or a
    missing file: the built-ins). A preset missing in the file inherits from the built-in of the same
    id, else from :data:`DEFAULT_PRESET`. An unreadable file is logged and ignored."""
    presets = {k: dict(v) for k, v in BUILTIN_PRESETS.items()}
    if path is None or not Path(path).exists():
        return presets
    try:
        obj = load_yaml(Path(path)) or {}
    except Exception as exc:  # noqa: BLE001 - any YAML/IO error: fall back to the built-ins
        log.warning(f"Ignoring realism presets {path}: {exc}")
        return presets
    profiles = obj.get("profiles", obj) if isinstance(obj, dict) else {}
    if isinstance(profiles, dict):
        for pid, raw in profiles.items():
            if isinstance(pid, str) and isinstance(raw, dict):
                presets[pid] = _normalize_yaml_preset(raw, presets.get(pid, presets[DEFAULT_PRESET]))
    return presets


@lru_cache(maxsize=4)
def _cached_presets(path: str, mtime_ns: int) -> dict[str, dict[str, float]]:
    del mtime_ns  # cache key only: an edited file is read again
    return load_realism_presets(path)


def default_presets() -> dict[str, dict[str, float]]:
    """Presets of :func:`default_presets_path` (cached per path and modification time; do not mutate)."""
    path = default_presets_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    return _cached_presets(str(path), mtime)


def _resolve_policy(filt: dict[str, Any], presets: RealismPresets | None = None) -> dict[str, float]:
    src_keys = set(filt.get("__source_keys", set()) or set())
    pid = str(filt.get("realism_profile_id", DEFAULT_PRESET) or DEFAULT_PRESET)
    presets = default_presets() if presets is None else presets
    preset = presets.get(pid, presets.get(DEFAULT_PRESET, BUILTIN_PRESETS[DEFAULT_PRESET]))
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


def _apply_constraints(selected: list[str]) -> list[str]:
    out = list(selected)
    if "G5" in out and "G7" in out:
        out.remove("G7")
    heavies = [g for g in out if g in _HEAVY_GROUPS]
    if len(heavies) > 1:
        keep = heavies[0]
        out = [g for g in out if g not in _HEAVY_GROUPS or g == keep]
    return out


def apply_realism_groups(
    augment: dict[str, float],
    filt: dict[str, Any],
    rng: np.random.Generator | None = None,
    *,
    presets: RealismPresets | None = None,
) -> dict[str, float]:
    """Apply correlated realism groups on top of sampled augment values (*presets*: see the module doc)."""
    aug = dict(augment or {})
    if rng is None:
        rng = np.random.default_rng()
    constraints_enabled = bool(filt.get("realism_constraints_enabled", True))
    policy = _resolve_policy(filt, presets)

    # G0: always-on mild day-to-day drift.
    aug["blur_sigma"] = _tri_sample(rng, 0.10, 0.50, 1.20)
    aug["noise_stddev"] = _tri_sample(rng, 1.0, 3.0, 6.0)
    aug["brightness_factor"] = _tri_sample(rng, 0.96, 1.00, 1.05)
    aug["contrast_factor"] = _tri_sample(rng, 0.96, 1.00, 1.06)
    # Keep the cardinal 0/90/180/270 base orientation, only replace the small jitter.
    base_orientation_deg = (90.0 * round(float(aug.get("rotation_deg", 0.0) or 0.0) / 90.0)) % 360.0
    aug["rotation_deg"] = base_orientation_deg + _tri_sample(rng, -4.0, 0.0, 4.0)
    aug.update({k: NEUTRAL_AUGMENT[k] for k in _G0_RESET_KEYS})

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

    # Cross-group clamps run after the loop so they do not depend on group order.
    if constraints_enabled and "G1" in selected and "G2" in selected:
        # Optics blur + motion blur together: cap optics blur.
        aug["blur_sigma"] = float(min(aug.get("blur_sigma", 0.0), 1.0))
    if constraints_enabled and "G5" in selected:
        aug["jpeg_quality"] = 100

    return aug
