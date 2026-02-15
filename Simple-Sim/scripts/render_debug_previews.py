#!/usr/bin/env python3
"""Render small per-profile preview images for debugging 3D (Blender) profiles.

Goal: quickly verify that components/pads/defect states render correctly before AI training.

This script:
- Detects available 3D profiles under `configs/profiles/` (supported_render_backends includes "blender_3d")
- Lets you select one or more profiles (interactive by default, or via --profiles/--all)
- For each selected profile, renders exactly 1 image per defect state (OK/MISALIGNED/MISSING/TOMBSTONE, etc.)
- Writes previews under `<out>/previews/<profile_id>/...` and an `<out>/index.html`

Examples:
  .venv/bin/python scripts/render_debug_previews.py
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


def _discover_3d_profiles(profiles_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for pid in _iter_profile_ids(profiles_dir):
        try:
            prof = load_profile(pid, profiles_dir)
        except Exception:
            # Skip invalid profiles; this is a debug helper.
            continue
        backends = prof.get("profile", {}).get("supported_render_backends") or []
        if "blender_3d" in backends:
            out.append((pid, prof))
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


def _find_matching_run_configs(configs_dir: Path, *, profile_id: str) -> List[Path]:
    configs_dir = Path(configs_dir)
    hits: List[Path] = []
    for p in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = load_config(p)
        except Exception:
            continue
        if str((cfg.get("render") or {}).get("backend", "")) != "blender_3d":
            continue
        run = cfg.get("run") or {}
        if str(run.get("component_profile", "")) == str(profile_id):
            hits.append(p)
    return hits


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


def _write_index_html(out_root: Path, previews: List[Tuple[str, str, str]]) -> None:
    """previews: list of (profile_id, defect, rel_image_path)."""
    out_root = Path(out_root)
    index_path = out_root / "index.html"

    # Group by profile id
    by_profile: Dict[str, List[Tuple[str, str]]] = {}
    for pid, defect, rel_path in previews:
        by_profile.setdefault(pid, []).append((defect, rel_path))

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
    lines.append(".pill.state{border-color:#cfd8ff; background:#f5f7ff;}")
    lines.append("img{width:100%; height:auto; border-radius:8px; background:#f6f6f6;}")
    lines.append("</style></head><body>")
    lines.append("<h1>Simple-Sim Debug Previews</h1>")

    for pid in sorted(by_profile.keys()):
        lines.append(f"<h2>{_escape(pid)}</h2>")
        lines.append("<div class='grid'>")
        items = by_profile[pid]
        items.sort(key=lambda t: t[0])
        for defect, rel_path in items:
            lines.append("<div class='card'>")
            lines.append("<div class='cap'>")
            lines.append(f"<span class='pill'>{_escape(pid)}</span>")
            lines.append(f"<span class='pill state'>{_escape(defect)}</span>")
            lines.append("</div>")
            lines.append(f"<a href='{_escape(rel_path)}'><img loading='lazy' src='{_escape(rel_path)}'></a>")
            lines.append("</div>")
        lines.append("</div>")

    lines.append("</body></html>")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _open_tk_viewer(
    out_root: Path,
    previews: List[Tuple[str, str, str]],
    *,
    sim_root: Optional[Path] = None,
    profiles_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
    all_profiles: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    render_settings: Optional[Dict[str, Any]] = None,
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

    state = ViewerState()

    def _load_items():
        """Load image items from current preview list."""
        items: List[Tuple[str, str, Path]] = []
        for pid, defect, rel in state.previews:
            p = (out_root / rel).resolve()
            if p.exists():
                items.append((pid, defect, p))
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

        # Profile selection
        ttk.Label(controls, text="Profile:").pack(side="left", padx=(0, 5))

        profile_var = tk.StringVar()
        profile_ids = [pid for pid, _ in all_profiles]
        profile_combo = ttk.Combobox(controls, textvariable=profile_var, values=profile_ids, width=35, state="readonly")
        if profile_ids:
            profile_combo.current(0)
        profile_combo.pack(side="left", padx=(0, 15))

        # Seed controls
        ttk.Label(controls, text="Seed:").pack(side="left", padx=(0, 5))
        seed_var = tk.StringVar(value=str(render_settings.get("seed_base", 2026)))
        seed_entry = ttk.Entry(controls, textvariable=seed_var, width=8)
        seed_entry.pack(side="left", padx=(0, 10))

        variable_seeds_var = tk.BooleanVar(value=True)  # Default to variable for reruns
        variable_seeds_check = ttk.Checkbutton(controls, text="Variable", variable=variable_seeds_var)
        variable_seeds_check.pack(side="left", padx=(0, 15))

        status_label = ttk.Label(controls, text="Ready", foreground="#060")
        status_label.pack(side="left", padx=(10, 10))

        def _start_render():
            if state.is_rendering:
                return

            selected_pid = profile_var.get()
            if not selected_pid:
                status_label.config(text="No profile selected", foreground="#b00")
                return

            # Find the profile
            selected_prof = None
            for pid, prof in all_profiles:
                if pid == selected_pid:
                    selected_prof = prof
                    break

            if not selected_prof:
                status_label.config(text="Profile not found", foreground="#b00")
                return

            state.is_rendering = True
            render_btn.config(state="disabled")
            status_label.config(text="Rendering...", foreground="#c60")

            def _render_thread():
                try:
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

                    # Get config for this profile
                    if forced_cfg is not None:
                        cfg = forced_cfg
                    else:
                        matches = _find_matching_run_configs(configs_dir, profile_id=selected_pid)
                        if matches:
                            cfg = load_config(matches[0])
                        else:
                            cfg = load_config(sim_root / "configs/run_0001_3d.yaml")

                    settings = _settings_from_config(cfg)

                    # Apply overrides
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

                    # Get defect types
                    defect_types = _parse_csv_list(states_override) or list(selected_prof.get("defect_set") or [])
                    if not defect_types:
                        defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

                    # Build and run jobs with unique run_id if variable seeds enabled
                    run_id = int(time.time() * 1000000) if use_variable_seeds else None
                    jobs, new_previews = _build_preview_jobs(
                        profile_id=selected_pid,
                        profile=selected_prof,
                        settings=settings,
                        out_root=out_root,
                        seed_base=int(seed_base),
                        defect_types=defect_types,
                        run_id=run_id,
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

                    # Update state with new previews (replace old ones for this profile)
                    # Remove old previews for this profile
                    state.previews = [(pid, df, rp) for pid, df, rp in state.previews if pid != selected_pid]
                    # Add new previews
                    state.previews.extend(new_previews)

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

                    all_new_previews = []

                    for idx, (pid, prof) in enumerate(all_profiles, 1):
                        root.after(0, lambda i=idx, p=pid: status_label.config(
                            text=f"Rendering {i}/{len(all_profiles)}: {p}", foreground="#c60"))

                        # Get config for this profile
                        if forced_cfg is not None:
                            cfg = forced_cfg
                        else:
                            matches = _find_matching_run_configs(configs_dir, profile_id=pid)
                            if matches:
                                cfg = load_config(matches[0])
                            else:
                                cfg = load_config(sim_root / "configs/run_0001_3d.yaml")

                        settings = _settings_from_config(cfg)

                        # Apply overrides
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

                        # Get defect types
                        defect_types = _parse_csv_list(states_override) or list(prof.get("defect_set") or [])
                        if not defect_types:
                            defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

                        # Build and run jobs with unique run_id if variable seeds enabled
                        run_id = int(time.time() * 1000000) if use_variable_seeds else None
                        jobs, new_previews = _build_preview_jobs(
                            profile_id=pid,
                            profile=prof,
                            settings=settings,
                            out_root=out_root,
                            seed_base=int(seed_base),
                            defect_types=defect_types,
                            run_id=run_id,
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

                    # Update state with all new previews
                    # Remove old previews for all rendered profiles
                    rendered_pids = {pid for pid, _ in all_profiles}
                    state.previews = [(pid, df, rp) for pid, df, rp in state.previews if pid not in rendered_pids]
                    # Add all new previews
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

        items.sort(key=lambda t: (t[0], t[1]))
        for idx, (pid, defect, img_path) in enumerate(items):
            r = idx // cols
            c = idx % cols

            card = ttk.Frame(inner, padding=pad)
            card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)

            cap = f"{pid}\n{defect}"
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
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]]]:
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
    previews: List[Tuple[str, str, str]] = []

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

        rel_path = f"previews/{profile_dir}/{str(defect_type)}.png"
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
            "render_3d": render_3d,
        }
        jobs.append(job)
        previews.append((profile_id, str(defect_type), rel_path))

    return jobs, previews


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-profile 3D debug previews (OK/MISALIGNED/etc.)")
    parser.add_argument("--profiles-dir", default="configs/profiles", help="Directory with component profile YAMLs")
    parser.add_argument("--configs-dir", default="configs", help="Directory with run_*.yaml configs (for auto matching)")
    parser.add_argument("--out", default="outputs/debug_previews", help="Output directory root")

    parser.add_argument("--profiles", default="", help="Comma-separated profile IDs to render (e.g. chip_..._3d@1,sot23_..._3d@1)")
    parser.add_argument("--all", action="store_true", help="Render all discovered 3D profiles")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt; requires --profiles or --all")

    parser.add_argument("--config", default="", help="Optional run config YAML to use for ALL selected profiles")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed for deterministic previews")
    parser.add_argument("--variable-seeds", action="store_true", help="Generate different images on each run (like real 3D pipeline)")

    parser.add_argument("--states", default="", help="Comma-separated defect states to render (default: from profile.defect_set)")

    parser.add_argument("--blender", default="", help="Override blender executable (else from config)")
    parser.add_argument("--samples", type=int, default=0, help="Override Cycles samples (else from config)")
    parser.add_argument("--device", default="", help="Override device (CPU/GPU; else from config)")

    parser.add_argument("--roi", default="", help="Override ROI as WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")
    parser.add_argument("--dry-run", action="store_true", help="Only write jobs file and index; do not invoke Blender")
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

    discovered = _discover_3d_profiles(profiles_dir)
    if not discovered:
        raise SystemExit(f"No 3D (blender_3d) profiles found under: {profiles_dir}")

    # Determine selected profile ids
    selected_ids: List[str] = []
    if args.all:
        selected_ids = [pid for pid, _ in discovered]
    else:
        selected_ids = _parse_csv_list(args.profiles)

    if not selected_ids and not args.non_interactive:
        print("\nAvailable 3D profiles:\n")
        for i, (pid, prof) in enumerate(discovered, 1):
            desc = str((prof.get("profile") or {}).get("description", "") or "")
            print(f"{i:2d}. {pid}" + (f"  ({desc})" if desc else ""))
        print("\nSelect profiles by number (e.g. 1,3-4) or 'all':")
        sel = input("> ").strip()
        idxs = _parse_selection(sel, len(discovered))
        selected_ids = [discovered[i][0] for i in idxs]

    if not selected_ids:
        raise SystemExit("No profiles selected. Use interactive mode or pass --profiles/--all.")

    # Load profiles (dict) for selected ids and validate they're 3D.
    selected: List[Tuple[str, Dict[str, Any]]] = []
    pid_to_profile: Dict[str, Dict[str, Any]] = {pid: prof for pid, prof in discovered}
    missing = [pid for pid in selected_ids if pid not in pid_to_profile]
    if missing:
        raise SystemExit(f"Unknown profiles: {missing}\nAvailable: {[pid for pid, _ in discovered]}")
    for pid in selected_ids:
        prof = pid_to_profile[pid]
        backends = prof.get("profile", {}).get("supported_render_backends") or []
        if "blender_3d" not in backends:
            raise SystemExit(f"Profile is not a 3D backend profile: {pid}")
        selected.append((pid, prof))

    # Load settings: either one config for all, or auto match per profile.
    forced_cfg: Optional[Dict[str, Any]] = None
    if args.config:
        forced_cfg = load_config(sim_root / args.config)

    # Optional ROI override parsing
    roi_override: Optional[Tuple[int, int, float]] = None
    if args.roi:
        m = re.match(r"^\\s*(\\d+)x(\\d+)@([0-9]*\\.?[0-9]+)\\s*$", str(args.roi))
        if not m:
            raise SystemExit("Invalid --roi. Expected WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")
        roi_override = (int(m.group(1)), int(m.group(2)), float(m.group(3)))

    previews_for_index: List[Tuple[str, str, str]] = []
    all_jobs: List[Dict[str, Any]] = []
    per_profile_settings: List[RenderSettings] = []

    for pid, prof in selected:
        if forced_cfg is not None:
            cfg = forced_cfg
            cfg_src = f"(forced) {args.config}"
        else:
            matches = _find_matching_run_configs(configs_dir, profile_id=pid)
            if matches:
                cfg = load_config(matches[0])
                try:
                    rel = str(matches[0].relative_to(sim_root))
                except Exception:
                    rel = str(matches[0])
                cfg_src = f"(auto) {rel}"
            else:
                # Fallback minimal config (validated later after we fill required sections)
                cfg = load_config(sim_root / "configs/run_0001_3d.yaml")
                cfg_src = "(fallback) configs/run_0001_3d.yaml"

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

        defect_types = _parse_csv_list(args.states) or list(prof.get("defect_set") or [])
        if not defect_types:
            defect_types = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

        print(f"\n[preview] Profile: {pid}")
        print(f"[preview] Config:   {cfg_src}")
        print(f"[preview] ROI:      {settings.roi_width_px}x{settings.roi_height_px} @ {settings.mm_per_px} mm/px")
        print(f"[preview] Blender:  {settings.blender_executable} (samples={settings.cycles_samples}, device={settings.device})")
        print(f"[preview] States:   {defect_types}")

        per_profile_settings.append(settings)

        # Use variable seeds if requested (like real 3D pipeline)
        run_id = int(time.time() * 1000000) if args.variable_seeds else None

        jobs, previews = _build_preview_jobs(
            profile_id=pid,
            profile=prof,
            settings=settings,
            out_root=out_root,
            seed_base=int(args.seed),
            defect_types=defect_types,
            run_id=run_id,
        )
        all_jobs.extend(jobs)
        previews_for_index.extend(previews)

    out_root.mkdir(parents=True, exist_ok=True)
    jobs_path = out_root / "blender_jobs_previews.jsonl"
    write_jobs_jsonl(jobs_path, all_jobs)
    _write_index_html(out_root, previews_for_index)

    # Translate legacy flags into --view behavior if explicitly set.
    if args.open_browser is True:
        args.view = "browser"
    elif args.open_browser is False:
        args.view = "none"

    # Prepare render settings dict for viewer
    viewer_render_settings = {
        "forced_cfg": forced_cfg,
        "seed_base": int(args.seed),
        "states": args.states,
        "blender": args.blender,
        "samples": args.samples,
        "device": args.device,
        "roi_override": roi_override,
    }

    if args.dry_run:
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
                all_profiles=discovered,
                render_settings=viewer_render_settings,
            )
        return

    # One Blender invocation for all selected profiles (fast).
    # Cycles samples/device/executable are process-wide (can't vary per job).
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

    # NOTE: The job file includes per-job ROI sizes; render_batch.py reads width/height per job.
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
            all_profiles=discovered,
            render_settings=viewer_render_settings,
        )


if __name__ == "__main__":
    main()
