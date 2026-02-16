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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Add Simple-Sim root to sys.path, like other scripts.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.config import load_config, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generator_3d import write_jobs_jsonl, render_blender_batch
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
        from tkinter import ttk
    except Exception as exc:
        raise RuntimeError(f"Tkinter not available: {exc}") from exc

    try:
        from PIL import Image, ImageTk
    except Exception as exc:
        raise RuntimeError(f"Pillow (PIL) not available: {exc}") from exc

    import threading

    out_root = Path(out_root)

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

        cardinal_rotation_90_var = tk.BooleanVar(value=bool(render_settings.get("enable_cardinal_rotation_90", True)))
        cardinal_rotation_90_check = ttk.Checkbutton(controls, text="90° Rotation", variable=cardinal_rotation_90_var)
        cardinal_rotation_90_check.pack(side="left", padx=(0, 15))

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
                                enable_cardinal_rotation_90=bool(cardinal_rotation_90_var.get()),
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
                            all_new_previews.extend(new_previews)
                        else:
                            # 2D OpenCV render
                            import cv2  # type: ignore
                            from simple_sim.generator_2d import render_roi  # type: ignore

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
                                    enable_cardinal_rotation_90=bool(cardinal_rotation_90_var.get()),
                                )

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
                                    enable_cardinal_rotation_90=bool(cardinal_rotation_90_var.get()),
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
                                all_new_previews.extend(new_previews)
                            else:
                                import cv2  # type: ignore
                                from simple_sim.generator_2d import render_roi  # type: ignore

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
                                        enable_cardinal_rotation_90=bool(cardinal_rotation_90_var.get()),
                                    )
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

            def _open_file(path=img_path):
                import webbrowser
                webbrowser.open(path.as_uri())

            lbl.bind("<Button-1>", lambda _e, f=_open_file: f())

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
                        "enable_cardinal_rotation_90": not bool(args.disable_cardinal_rotation_90),
                        "blender": args.blender,
                        "samples": args.samples,
                        "device": args.device,
                        "roi_override": roi_override,
                        "backend_mode": backend_mode,
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
                from simple_sim.generator_2d import render_roi  # type: ignore

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
                        enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
                    )
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
                    enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
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
        "enable_cardinal_rotation_90": not bool(args.disable_cardinal_rotation_90),
        "blender": args.blender,
        "samples": args.samples,
        "device": args.device,
        "roi_override": roi_override,
        "backend_mode": backend_mode,
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
