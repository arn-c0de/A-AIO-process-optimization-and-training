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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np

# Add Simple-Sim root to sys.path, like other scripts.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.config import load_config, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generators.blender_3d import write_jobs_jsonl, render_blender_batch
from simple_sim.generators.filter_settings import default_filter_values_for_popup, normalize_image_filters
from simple_sim.generators.opencv2d.geometry import sample_nominal_geometry
from simple_sim.profile_hash import load_profile

from gui.utils.filter_profile_store import is_filter_popup_extra_key, parse_bool_like


BACKEND_2D = "opencv_2d"
BACKEND_3D = "blender_3d"
BACKEND_CHOICES = [BACKEND_2D, BACKEND_3D, "both", "auto"]

# (profile_id, backend, defect, rel_image_path)
PreviewRec = tuple[str, str, str, str]


@dataclass(frozen=True)
class RenderSettings:
    roi_width_px: int
    roi_height_px: int
    mm_per_px: float
    blender_executable: str
    cycles_samples: int
    device: str
    engine: str = "BLENDER_EEVEE"


def _shared_settings_path(sim_root: Path, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return (Path(sim_root) / "outputs" / "gui" / "settings.json").resolve()


def _read_settings_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_settings_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalize_filter_settings(d: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize image-filter values with the defaults shared by GUI and generator."""
    return normalize_image_filters(d)


def _encode_filter_store_value(key: str, value: Any) -> Any:
    """Encode one normalized filter value in the popup/profile-store format."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if key.startswith("realism_") else f"{value:.2f}"
    return value


def _filter_settings_to_profile_store(d: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``d`` merged over ``existing`` and encoded like the GUI filter popup stores it.

    Every key the shared normalizer knows is (re)encoded; all other keys from the
    stored profile and from ``d`` are kept verbatim so filters this script does not
    know about survive a load/save round trip.
    """
    merged: dict[str, Any] = {}
    for src in (existing, d):
        for k, v in (src or {}).items():
            if isinstance(k, str):
                merged[k] = v
    out = dict(merged)
    for k, v in normalize_image_filters(merged).items():
        out[k] = _encode_filter_store_value(k, v)
    return out


def _flat_filter_setting_keys() -> list[str]:
    """Filter keys mirrored as flat ``pipeline.filter.<key>`` entries in settings.json."""
    return [
        k for k in default_filter_values_for_popup()
        if k != "enable" and not is_filter_popup_extra_key(k)
    ]


def _load_shared_filter_profiles(settings_path: Path) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    data = _read_settings_json(settings_path)
    raw_profiles = str(data.get("pipeline.filter_profiles_json", "") or "").strip()
    active = str(data.get("pipeline.filter_profile_active", "") or "").strip() or "Default"
    profiles: dict[str, dict[str, Any]] = {}
    if raw_profiles:
        try:
            obj = json.loads(raw_profiles)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        # Keep stored profiles verbatim (no key stripping).
                        profiles[k] = dict(v)
        except Exception:
            profiles = {}

    if not profiles:
        legacy = {
            k: data[f"pipeline.filter.{k}"]
            for k in _flat_filter_setting_keys()
            if f"pipeline.filter.{k}" in data
        }
        profiles = {"Default": _filter_settings_to_profile_store(legacy)}
        active = "Default"
    if active not in profiles:
        active = sorted(profiles.keys(), key=lambda s: s.lower())[0]
    active_profile = profiles.get(active, {})
    current = _normalize_filter_settings(active_profile)
    for k, v in active_profile.items():
        if isinstance(k, str) and k not in current:
            current[k] = v
    return profiles, active, current


def _save_shared_filter_profiles(settings_path: Path, profiles: dict[str, dict[str, Any]], active: str, current: dict[str, Any]) -> None:
    data = _read_settings_json(settings_path)
    data["pipeline.filter_profiles_json"] = json.dumps(profiles, ensure_ascii=True)
    data["pipeline.filter_profile_active"] = str(active)

    cur = _normalize_filter_settings(current)
    for k in _flat_filter_setting_keys():
        v = cur[k]
        if isinstance(v, bool):
            data[f"pipeline.filter.{k}"] = v
        elif isinstance(v, int):
            data[f"pipeline.filter.{k}"] = str(v)
        elif isinstance(v, float):
            data[f"pipeline.filter.{k}"] = f"{v:.2f}"
        else:
            data[f"pipeline.filter.{k}"] = v
    _write_settings_json(settings_path, data)


def _apply_filter_overrides_to_augment(augment: dict[str, float], image_filters: dict[str, Any] | None) -> dict[str, float]:
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


def _postprocess_rendered_previews(out_root: Path, jobs: Sequence[dict[str, Any]]) -> None:
    """Apply the dataset generator's Blender post-filters to preview renders.

    Reuses scripts/generate.py so previews match generated datasets. Rotation is not
    re-applied: render_batch.py already rotates the Blender scene by augment.rotation_deg.
    """
    from scripts.generate import _postprocess_blender_images

    _postprocess_blender_images(list(jobs), Path(out_root))


def _get_rotation_jitter_range(cfg: dict[str, Any] | None) -> tuple[float, float]:
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
    rotation_jitter_range: tuple[float, float],
    enable_cardinal_rotation_90: bool = True,
) -> dict[str, float]:
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


def _iter_profile_ids(profiles_dir: Path) -> list[str]:
    # Profile files are `<profile_id>.yaml` where profile_id contains '@'.
    ids: list[str] = []
    for p in sorted(Path(profiles_dir).glob("*.yaml")):
        if p.name.startswith("."):
            continue
        ids.append(p.stem)
    return ids


def _profile_backends(profile: dict[str, Any]) -> set[str]:
    b = (profile.get("profile") or {}).get("supported_render_backends") or []
    out = set()
    for x in b:
        s = str(x)
        if s in {BACKEND_2D, BACKEND_3D}:
            out.add(s)
    return out


def _discover_profiles(profiles_dir: Path, *, backend_mode: str) -> list[tuple[str, dict[str, Any], set[str]]]:
    """Discover profiles supporting 2D and/or 3D."""
    backend_mode = str(backend_mode or "auto")
    out: list[tuple[str, dict[str, Any], set[str]]] = []
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


def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    parts = []
    for chunk in str(s).split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _parse_selection(selection: str, n: int) -> list[int]:
    """Parse '1,2,4-6,all' into 0-based indices."""
    selection = (selection or "").strip().lower()
    if selection in {"a", "all", "*"}:
        return list(range(n))

    idxs: list[int] = []
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


def _find_matching_run_configs(configs_dir: Path, *, profile_id: str, backend: str) -> list[Path]:
    configs_dir = Path(configs_dir)
    backend = str(backend)
    hits: list[Path] = []
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
    discovered: list[tuple[str, dict[str, Any], set[str]]],
    *,
    profiles_dir: Path,
    out_root: Path,
    backend_mode: str,
    states_override: str,
) -> list[str]:
    """Return profile IDs that are new/incomplete in preview output.

    A profile is selected when at least one expected preview image is missing for
    the requested backend mode and defect states.
    """
    selected: list[str] = []
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


def _settings_from_config(cfg: dict[str, Any]) -> RenderSettings:
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
        engine=str(blender.get("engine", "BLENDER_EEVEE")),
    )


def _load_preview_config_for_backend(
    *,
    forced_cfg: dict[str, Any] | None,
    sim_root: Path,
    configs_dir: Path,
    profile_id: str,
    backend: str,
    forced_label: str = "",
) -> tuple[dict[str, Any], str]:
    """Return (config, source description) for one profile/backend.

    A forced config (--config) is used only for its own backend; other backends
    (e.g. with --backend=both) fall back to per-profile auto matching.
    """
    if forced_cfg is not None:
        cfg_backend = str((forced_cfg.get("render") or {}).get("backend", ""))
        if cfg_backend == backend:
            validate_config(forced_cfg)
            return forced_cfg, f"(forced) {forced_label}".rstrip()

    matches = _find_matching_run_configs(configs_dir, profile_id=profile_id, backend=backend)
    if matches:
        cfg = load_config(matches[0])
        try:
            rel = str(matches[0].relative_to(sim_root))
        except Exception:
            rel = str(matches[0])
        cfg_src = f"(auto) {rel}"
    else:
        fallback = "configs/run_0001.yaml" if backend == BACKEND_2D else "configs/run_0001_3d.yaml"
        cfg = load_config(Path(sim_root) / fallback)
        cfg_src = f"(fallback) {fallback}"
    validate_config(cfg)
    return cfg, cfg_src


def _apply_3d_settings_overrides(
    settings: RenderSettings,
    *,
    roi_override: tuple[int, int, float] | None,
    blender_override: str,
    samples_override: int,
    device_override: str,
) -> RenderSettings:
    if roi_override is not None:
        w, h, mpp = roi_override
        settings = replace(settings, roi_width_px=int(w), roi_height_px=int(h), mm_per_px=float(mpp))
    if blender_override:
        settings = replace(settings, blender_executable=str(blender_override))
    if samples_override and int(samples_override) > 0:
        settings = replace(settings, cycles_samples=int(samples_override))
    if device_override:
        settings = replace(settings, device=str(device_override))
    return settings


def _resolve_defect_types(profile: dict[str, Any], states_override: str | None) -> list[str]:
    defect_types = _parse_csv_list(states_override) or list(profile.get("defect_set") or [])
    return defect_types or ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]


def _preview_rel_path(profile_id: str, backend: str, defect_type: str) -> str:
    return f"previews/{_sanitize_dir_name(profile_id)}/{backend}/{str(defect_type)}.png"


def _resolve_2d_roi(cfg: dict[str, Any], roi_override: tuple[int, int, float] | None) -> tuple[int, int, float]:
    if roi_override is not None:
        return int(roi_override[0]), int(roi_override[1]), float(roi_override[2])
    roi = cfg["roi"]
    return int(roi["width_px"]), int(roi["height_px"]), float(roi["mm_per_px"])


def _render_2d_previews(
    *,
    profile_id: str,
    profile: dict[str, Any],
    cfg: dict[str, Any],
    out_root: Path,
    seed_base: int,
    defect_types: Sequence[str],
    run_id: int | None,
    roi_override: tuple[int, int, float] | None,
    image_filters: dict[str, Any] | None,
    enable_cardinal_rotation_90: bool,
) -> list[PreviewRec]:
    """Render one OpenCV 2D preview per defect state; return preview records."""
    import cv2  # type: ignore
    from simple_sim.generators.opencv_2d import render_roi  # type: ignore

    backend = BACKEND_2D
    w, h, mpp = _resolve_2d_roi(cfg, roi_override)
    roi_cfg = {"width_px": int(w), "height_px": int(h), "mm_per_px": float(mpp)}

    render_cfg = dict(cfg["render"])
    # schema v2: allow render.component_color to come from profile.
    if "component_color" not in render_cfg and "render" in profile:
        render_cfg["component_color"] = profile["render"]["component_color_bgr"]

    footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
    geometry_ranges = profile.get("geometry_ranges") or {}
    tolerances = profile.get("tolerances")
    rotation_jitter_range = _get_rotation_jitter_range(cfg)

    recs: list[PreviewRec] = []
    for defect_type in defect_types:
        if run_id is not None:
            seed_str = f"{int(seed_base)}|{profile_id}|{backend}|{str(defect_type)}|{int(run_id)}"
        else:
            seed_str = f"{int(seed_base)}|{profile_id}|{backend}|{str(defect_type)}"
        hh = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        seed = int(hh[:12], 16)
        rng = np.random.default_rng(seed)
        augment = _sample_preview_augment(
            rng,
            rotation_jitter_range=rotation_jitter_range,
            enable_cardinal_rotation_90=bool(enable_cardinal_rotation_90),
        )
        augment = _apply_filter_overrides_to_augment(augment, image_filters)
        nominal = sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
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
        rel_path = _preview_rel_path(profile_id, backend, str(defect_type))
        out_path = (Path(out_root) / rel_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), img):
            raise OSError(f"Failed to write image: {out_path}")
        recs.append((profile_id, backend, str(defect_type), rel_path))
    return recs


def _render_3d_previews(
    *,
    sim_root: Path,
    profile_id: str,
    profile: dict[str, Any],
    settings: RenderSettings,
    out_root: Path,
    seed_base: int,
    defect_types: Sequence[str],
    run_id: int | None,
    rotation_jitter_range: tuple[float, float],
    image_filters: dict[str, Any] | None,
    enable_cardinal_rotation_90: bool,
    jobs_path: Path,
) -> list[PreviewRec]:
    """Render one Blender 3D preview per defect state (single Blender batch)."""
    jobs, recs = _build_preview_jobs(
        profile_id=profile_id,
        profile=profile,
        settings=settings,
        out_root=out_root,
        seed_base=int(seed_base),
        defect_types=defect_types,
        run_id=run_id,
        backend=BACKEND_3D,
        rotation_jitter_range=rotation_jitter_range,
        enable_cardinal_rotation_90=bool(enable_cardinal_rotation_90),
        image_filters=image_filters,
    )
    write_jobs_jsonl(jobs_path, jobs)
    render_blender_batch(
        sim_root=sim_root,
        jobs_path=jobs_path,
        output_root=out_root,
        blender_executable=settings.blender_executable,
        cycles_samples=settings.cycles_samples,
        device=settings.device,
        engine=settings.engine,
    )
    _postprocess_rendered_previews(out_root, jobs)
    return recs


def _write_index_html(out_root: Path, previews: list[PreviewRec]) -> None:
    """previews: list of (profile_id, backend, defect, rel_image_path)."""
    out_root = Path(out_root)
    index_path = out_root / "index.html"

    # Group by profile id
    by_profile: dict[str, list[tuple[str, str, str]]] = {}
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


def _collect_existing_previews(out_root: Path) -> list[PreviewRec]:
    """Scan existing preview PNG files and return preview records."""
    out_root = Path(out_root)
    recs: list[PreviewRec] = []
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
    previews: list[PreviewRec],
    *,
    sim_root: Path | None = None,
    profiles_dir: Path | None = None,
    configs_dir: Path | None = None,
    all_profiles: list[tuple[str, dict[str, Any], set[str]]] | None = None,
    render_settings: dict[str, Any] | None = None,
    selected_profile_ids: list[str] | None = None,
) -> None:
    """Open an interactive Tkinter viewer for rendered previews with rerun capabilities."""
    # Import lazily so the script can still run on systems without Tk installed.
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as exc:
        raise RuntimeError(f"Tkinter not available: {exc}") from exc

    try:
        from PIL import Image, ImageTk
    except Exception as exc:
        raise RuntimeError(f"Pillow (PIL) not available: {exc}") from exc

    import queue
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
        items: list[tuple[str, str, str, Path]] = []
        for pid, backend, defect, rel in state.previews:
            if state.backend_filter != "both" and backend != state.backend_filter:
                continue
            p = (out_root / rel).resolve()
            if p.exists():
                items.append((pid, backend, defect, p))
        return items

    root = tk.Tk()
    root.title("Simple-Sim Debug Previews")
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

        def _profiles_for_backend(mode: str) -> list[str]:
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
        # Current filter values in profile-store (popup) format. Unknown keys are kept
        # so a save never strips filters this viewer does not know about.
        filter_values: dict[str, Any] = _filter_settings_to_profile_store(
            shared_current_filter,
            shared_profiles.get(shared_active_profile),
        )
        cardinal_rotation_90_var = tk.BooleanVar(
            value=parse_bool_like(filter_values.get("cardinal_rotation_90", True), default=True)
        )

        def _current_filter_store() -> dict[str, Any]:
            """Current filter values in profile-store format (main thread only)."""
            d = dict(filter_values)
            d["cardinal_rotation_90"] = bool(cardinal_rotation_90_var.get())
            return d

        def _current_filters_from_vars() -> dict[str, Any]:
            """Normalized current filter values (main thread only)."""
            d = _current_filter_store()
            out = _normalize_filter_settings(d)
            for k, v in d.items():
                out.setdefault(k, v)
            return out

        def _apply_filters_to_vars(d: dict[str, Any]) -> None:
            filter_values.clear()
            filter_values.update(_filter_settings_to_profile_store(d))
            cardinal_rotation_90_var.set(parse_bool_like(filter_values.get("cardinal_rotation_90", True), default=True))

        def _persist_filters(active_name: str | None = None) -> None:
            nonlocal shared_active_profile
            cur = _current_filter_store()
            if active_name:
                shared_active_profile = active_name
            shared_profiles[shared_active_profile] = _filter_settings_to_profile_store(
                cur,
                shared_profiles.get(shared_active_profile),
            )
            _save_shared_filter_profiles(
                settings_file,
                shared_profiles,
                shared_active_profile,
                cur,
            )

        def _open_filter_popup() -> None:
            """Open shared image filter popup."""
            from gui.components.filter_popup import open_filter_popup

            def on_close(updated_profiles: dict[str, dict[str, Any]], updated_active: str) -> None:
                nonlocal shared_active_profile, shared_profiles
                shared_profiles = updated_profiles
                shared_active_profile = updated_active
                _apply_filters_to_vars(shared_profiles.get(shared_active_profile, {}))
                _save_shared_filter_profiles(
                    settings_file,
                    shared_profiles,
                    shared_active_profile,
                    _current_filter_store(),
                )

            def show_messagebox(msg_type: str, title: str, message: str) -> None:
                if msg_type == "warning":
                    messagebox.showwarning(title, message, parent=root)
                elif msg_type == "error":
                    messagebox.showerror(title, message, parent=root)
                else:
                    messagebox.showinfo(title, message, parent=root)

            def ask_string(title: str, prompt: str, initial: str = "") -> str | None:
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
                get_current_values=lambda: _filter_settings_to_profile_store(_current_filter_store()),
                set_current_values=_apply_filters_to_vars,
                float_or_default=float_or_default,
                show_messagebox=show_messagebox,
                ask_string=ask_string,
                ask_yes_no=ask_yes_no,
                sim_root=str(sim_root),
            )

        cardinal_rotation_90_check = ttk.Checkbutton(controls, text="90° Rotation", variable=cardinal_rotation_90_var)
        cardinal_rotation_90_check.pack(side="left", padx=(0, 15))
        ttk.Button(controls, text="Image Filters", command=_open_filter_popup).pack(side="left", padx=(0, 15))

        status_label = ttk.Label(controls, text="Ready", foreground="#060")
        status_label.pack(side="left", padx=(10, 10))

        # Worker threads never touch Tk: they post callables here, and the main
        # thread runs them from a root.after poll loop.
        ui_queue: queue.Queue = queue.Queue()

        def _post_ui(fn) -> None:
            ui_queue.put(fn)

        def _poll_ui_queue() -> None:
            while True:
                try:
                    fn = ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                except Exception as exc:
                    print(f"[viewer] UI update failed: {exc}")
            root.after(100, _poll_ui_queue)

        root.after(100, _poll_ui_queue)

        def _snapshot_render_inputs() -> dict[str, Any]:
            """Read every Tk-backed render input on the main thread."""
            try:
                seed_base = int(seed_var.get())
            except ValueError:
                seed_base = int(render_settings.get("seed_base", 2026))
            image_filters = _current_filters_from_vars()
            return {
                "seed_base": seed_base,
                "use_variable_seeds": bool(variable_seeds_var.get()),
                "mode": str(backend_var.get()),
                "image_filters": image_filters,
                "enable_cardinal_rotation_90": bool(image_filters.get("cardinal_rotation_90", True)),
            }

        def _render_profile_backend(
            pid: str,
            prof: dict[str, Any],
            backend: str,
            defect_types: list[str],
            run_id: int | None,
            snap: dict[str, Any],
            jobs_path: Path,
        ) -> list[PreviewRec]:
            """Render one profile/backend in a worker thread (no Tk access)."""
            roi_override = render_settings.get("roi_override")
            cfg, _cfg_src = _load_preview_config_for_backend(
                forced_cfg=render_settings.get("forced_cfg"),
                sim_root=sim_root,
                configs_dir=configs_dir,
                profile_id=pid,
                backend=backend,
            )
            if backend == BACKEND_3D:
                settings = _apply_3d_settings_overrides(
                    _settings_from_config(cfg),
                    roi_override=roi_override,
                    blender_override=str(render_settings.get("blender", "") or ""),
                    samples_override=int(render_settings.get("samples", 0) or 0),
                    device_override=str(render_settings.get("device", "") or ""),
                )
                return _render_3d_previews(
                    sim_root=sim_root,
                    profile_id=pid,
                    profile=prof,
                    settings=settings,
                    out_root=out_root,
                    seed_base=int(snap["seed_base"]),
                    defect_types=defect_types,
                    run_id=run_id,
                    rotation_jitter_range=_get_rotation_jitter_range(cfg),
                    image_filters=snap["image_filters"],
                    enable_cardinal_rotation_90=snap["enable_cardinal_rotation_90"],
                    jobs_path=jobs_path,
                )
            return _render_2d_previews(
                profile_id=pid,
                profile=prof,
                cfg=cfg,
                out_root=out_root,
                seed_base=int(snap["seed_base"]),
                defect_types=defect_types,
                run_id=run_id,
                roi_override=roi_override,
                image_filters=snap["image_filters"],
                enable_cardinal_rotation_90=snap["enable_cardinal_rotation_90"],
            )

        def _finish_render(
            new_previews: list[PreviewRec],
            rendered_pairs: set[tuple[str, str]],
            message: str,
            error: str | None = None,
        ) -> None:
            """Main thread: merge results, rewrite index.html, refresh the grid."""
            if rendered_pairs:
                # Replace only (profile, backend) pairs that were actually rendered.
                state.previews = [rec for rec in state.previews if (rec[0], rec[1]) not in rendered_pairs]
                state.previews.extend(new_previews)
                _write_index_html(out_root, state.previews)
                _refresh_images()
            if error is None:
                status_label.config(text=message, foreground="#060")
            else:
                status_label.config(text=error, foreground="#b00")
            state.is_rendering = False
            render_btn.config(state="normal")
            render_all_btn.config(state="normal")

        def _begin_render(status_text: str) -> dict[str, Any]:
            state.is_rendering = True
            render_btn.config(state="disabled")
            render_all_btn.config(state="disabled")
            status_label.config(text=status_text, foreground="#c60")
            _persist_filters(shared_active_profile)
            return _snapshot_render_inputs()

        def _start_render():
            if state.is_rendering:
                return

            selected_pid = profile_var.get()
            if not selected_pid:
                status_label.config(text="No profile selected", foreground="#b00")
                return

            snap = _begin_render("Rendering...")

            def _render_thread():
                new_previews: list[PreviewRec] = []
                rendered_pairs: set[tuple[str, str]] = set()
                error: str | None = None
                try:
                    # Always reload profile from disk so edits are picked up without restart.
                    selected_prof = load_profile(selected_pid, profiles_dir)
                    selected_backends = _profile_backends(selected_prof)
                    if not selected_backends:
                        raise RuntimeError(f"Profile {selected_pid!r} has no supported backends")

                    defect_types = _resolve_defect_types(selected_prof, render_settings.get("states", ""))

                    # Determine which backend(s) to render in this run
                    mode = snap["mode"]
                    if mode == "both":
                        backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in selected_backends]
                    else:
                        backends_to_render = [mode] if mode in selected_backends else []
                    if not backends_to_render:
                        raise RuntimeError(f"Profile {selected_pid!r} does not support backend={mode!r}")

                    run_id = int(time.time() * 1000000) if snap["use_variable_seeds"] else None
                    for backend in backends_to_render:
                        new_previews.extend(
                            _render_profile_backend(
                                selected_pid,
                                selected_prof,
                                backend,
                                defect_types,
                                run_id,
                                snap,
                                out_root / "blender_jobs_previews.jsonl",
                            )
                        )
                        rendered_pairs.add((selected_pid, backend))
                except Exception as exc:
                    error = f"Error: {exc}"
                _post_ui(lambda: _finish_render(new_previews, rendered_pairs, "Render complete", error))

            thread = threading.Thread(target=_render_thread, daemon=True)
            thread.start()

        def _start_render_all():
            """Render all available profiles."""
            if state.is_rendering:
                return

            if not all_profiles:
                status_label.config(text="No profiles available", foreground="#b00")
                return

            total = len(all_profiles)
            snap = _begin_render(f"Rendering all {total} profiles...")

            def _render_all_thread():
                new_previews: list[PreviewRec] = []
                rendered_pairs: set[tuple[str, str]] = set()
                error: str | None = None
                try:
                    mode = snap["mode"]
                    for idx, (pid, _prof_cached, _backends_cached) in enumerate(all_profiles, 1):
                        # Always reload profile from disk so edits are picked up without restart.
                        prof = load_profile(pid, profiles_dir)
                        backends = _profile_backends(prof)

                        # Skip profiles not in the current backend filter.
                        if mode != "both" and mode not in backends:
                            continue

                        _post_ui(lambda i=idx, p=pid: status_label.config(
                            text=f"Rendering {i}/{total}: {p}", foreground="#c60"))

                        defect_types = _resolve_defect_types(prof, render_settings.get("states", ""))
                        run_id = int(time.time() * 1000000) if snap["use_variable_seeds"] else None
                        backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in backends] if mode == "both" else [mode]

                        for backend in backends_to_render:
                            new_previews.extend(
                                _render_profile_backend(
                                    pid,
                                    prof,
                                    backend,
                                    defect_types,
                                    run_id,
                                    snap,
                                    out_root / f"blender_jobs_previews_{_sanitize_dir_name(pid)}.jsonl",
                                )
                            )
                            rendered_pairs.add((pid, backend))
                except Exception as exc:
                    error = f"Error: {exc}"
                n_rendered = len({p for p, _b in rendered_pairs})
                _post_ui(lambda: _finish_render(new_previews, rendered_pairs, f"Rendered {n_rendered} profiles", error))

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

    photo_refs: list[Any] = []
    zoom_photo_refs: list[Any] = []

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
            root.title("Simple-Sim Debug Previews (0 images)")
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
    profile: dict[str, Any],
    settings: RenderSettings,
    out_root: Path,
    seed_base: int,
    defect_types: Sequence[str],
    run_id: int | None = None,
    backend: str = BACKEND_3D,
    rotation_jitter_range: tuple[float, float] = (0.0, 0.0),
    enable_cardinal_rotation_90: bool = True,
    image_filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[PreviewRec]]:
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

    jobs: list[dict[str, Any]] = []
    previews: list[PreviewRec] = []

    for defect_type in defect_types:
        # Include run_id to vary seeds across reruns (like real 3D pipeline)
        if run_id is not None:
            seed_str = f"{int(seed_base)}|{profile_id}|{str(defect_type)}|{int(run_id)}"
        else:
            seed_str = f"{int(seed_base)}|{profile_id}|{str(defect_type)}"
        h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        seed = int(h[:12], 16)
        rng = np.random.default_rng(seed)

        nominal = sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
        defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)
        augment = _sample_preview_augment(
            rng,
            rotation_jitter_range=rotation_jitter_range,
            enable_cardinal_rotation_90=enable_cardinal_rotation_90,
        )
        augment = _apply_filter_overrides_to_augment(augment, image_filters)

        rel_path = _preview_rel_path(profile_id, backend, str(defect_type))
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
    forced_cfg: dict[str, Any] | None = None
    forced_backend: str | None = None
    if args.config:
        forced_cfg = load_config(sim_root / args.config)
        validate_config(forced_cfg)
        forced_backend = str((forced_cfg.get("render") or {}).get("backend", ""))

    # Optional ROI override parsing
    roi_override: tuple[int, int, float] | None = None
    if args.roi:
        m = re.match(r"^\s*(\d+)x(\d+)@([0-9]*\.?[0-9]+)\s*$", str(args.roi))
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
    selected_ids: list[str] = []
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

    pid_to_entry: dict[str, tuple[str, dict[str, Any], set[str]]] = {pid: (pid, prof, b) for pid, prof, b in discovered_all}
    missing = [pid for pid in selected_ids if pid not in pid_to_entry]
    if missing:
        raise SystemExit(f"Unknown profiles: {missing}")

    selected_entries: list[tuple[str, dict[str, Any], set[str]]] = []
    for pid in selected_ids:
        _pid, prof, b = pid_to_entry[pid]
        selected_entries.append((pid, prof, b))

    previews_for_index: list[PreviewRec] = []
    all_jobs: list[dict[str, Any]] = []
    per_profile_settings: list[RenderSettings] = []

    # Translate legacy flags into --view behavior if explicitly set.
    if args.open_browser is True:
        args.view = "browser"
    elif args.open_browser is False:
        args.view = "none"

    out_root.mkdir(parents=True, exist_ok=True)

    enable_cardinal_rotation_90 = (
        not bool(args.disable_cardinal_rotation_90)
        and bool(shared_filter_settings.get("cardinal_rotation_90", True))
    )

    for pid, prof, backends in selected_entries:
        defect_types = _resolve_defect_types(prof, args.states)

        run_id = int(time.time() * 1000000) if args.variable_seeds else None
        if backend_mode == "both":
            backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in backends]
        else:
            backends_to_render = [backend_mode] if backend_mode in backends else []
        if not backends_to_render:
            raise SystemExit(f"Profile {pid!r} does not support backend={backend_mode!r}")

        for backend in backends_to_render:
            # A forced --config applies to its own backend; with --backend=both the
            # other backend is auto-matched per profile.
            cfg, cfg_src = _load_preview_config_for_backend(
                forced_cfg=forced_cfg,
                sim_root=sim_root,
                configs_dir=configs_dir,
                profile_id=pid,
                backend=backend,
                forced_label=str(args.config),
            )

            if backend == BACKEND_2D:
                w, h, mpp = _resolve_2d_roi(cfg, roi_override)

                print(f"\n[preview] Profile:  {pid}")
                print(f"[preview] Backend:  {backend}")
                print(f"[preview] Config:    {cfg_src}")
                print(f"[preview] ROI:       {w}x{h} @ {mpp} mm/px")
                print(f"[preview] States:    {defect_types}")

                for defect_type in defect_types:
                    previews_for_index.append((pid, backend, str(defect_type), _preview_rel_path(pid, backend, str(defect_type))))

                if args.dry_run:
                    continue

                try:
                    _render_2d_previews(
                        profile_id=pid,
                        profile=prof,
                        cfg=cfg,
                        out_root=out_root,
                        seed_base=int(args.seed),
                        defect_types=defect_types,
                        run_id=run_id,
                        roi_override=roi_override,
                        image_filters=shared_filter_settings,
                        enable_cardinal_rotation_90=enable_cardinal_rotation_90,
                    )
                except OSError as exc:
                    raise SystemExit(str(exc)) from exc
            else:
                settings = _apply_3d_settings_overrides(
                    _settings_from_config(cfg),
                    roi_override=roi_override,
                    blender_override=str(args.blender or ""),
                    samples_override=int(args.samples or 0),
                    device_override=str(args.device or ""),
                )

                print(f"\n[preview] Profile:  {pid}")
                print(f"[preview] Backend:  {backend}")
                print(f"[preview] Config:    {cfg_src}")
                print(f"[preview] ROI:       {settings.roi_width_px}x{settings.roi_height_px} @ {settings.mm_per_px} mm/px")
                print(
                    f"[preview] Blender:   {settings.blender_executable} "
                    f"(samples={settings.cycles_samples}, device={settings.device}, engine={settings.engine})"
                )
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
                    enable_cardinal_rotation_90=enable_cardinal_rotation_90,
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
        "enable_cardinal_rotation_90": enable_cardinal_rotation_90,
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
        engines = sorted({s.engine for s in per_profile_settings} or {"BLENDER_EEVEE"})

        if len(exes) > 1 and not args.blender:
            print(f"\n[warn] Multiple blender executables across auto configs: {exes} (using: {exes[-1]})")
        if len(devices) > 1 and not args.device:
            print(f"[warn] Multiple devices across auto configs: {devices} (using: {devices[-1]})")
        if len(samples_list) > 1 and not args.samples:
            print(f"[warn] Multiple sample counts across auto configs: {samples_list} (using max: {max(samples_list)})")
        if len(engines) > 1:
            print(f"[warn] Multiple render engines across auto configs: {engines} (using: {engines[-1]})")

        blender_executable = str(args.blender or exes[-1])
        device = str(args.device or devices[-1])
        cycles_samples = int(args.samples or max(samples_list))
        engine = str(engines[-1])

        print(f"\n[render] Running Blender batch for {len(all_jobs)} previews...")
        render_blender_batch(
            sim_root=sim_root,
            jobs_path=jobs_path,
            output_root=out_root,
            blender_executable=blender_executable,
            cycles_samples=cycles_samples,
            device=device,
            engine=engine,
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
