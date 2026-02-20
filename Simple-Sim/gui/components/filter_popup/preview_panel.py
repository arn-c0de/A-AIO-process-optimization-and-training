"""Preview panel helpers for the filter popup."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any, Dict, Optional


def _open_zoom_image_popup(popup: Any, pil_img: Any, title: str) -> None:
    """Open a modal-ish window with a larger version of the selected preview image."""
    try:
        from PIL import Image, ImageTk  # type: ignore
    except ImportError:
        popup._preview_status_var.set("Pillow not installed.\nRun: pip install Pillow")
        return

    if pil_img is None:
        return

    prev = getattr(popup, "_zoom_img_window", None)
    if prev is not None and getattr(prev, "winfo_exists", lambda: False)():
        prev.destroy()

    win = tk.Toplevel(popup.top)
    win.title(title)
    win.transient(popup.top)
    win.lift()
    win.focus_force()

    max_w = max(240, int(win.winfo_screenwidth() * 0.9))
    max_h = max(240, int(win.winfo_screenheight() * 0.85))
    img = pil_img.copy()
    w, h = img.size
    if w > max_w or h > max_h:
        scale = min(max_w / float(w), max_h / float(h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    photo = ImageTk.PhotoImage(img, master=win)
    lbl = tk.Label(win, image=photo, anchor="center", bd=0, highlightthickness=0)
    lbl.image = photo  # keep reference
    lbl.pack(fill="both", expand=True, padx=8, pady=8)
    lbl.bind("<Double-Button-1>", lambda _evt: win.destroy())
    win.bind("<Escape>", lambda _evt: win.destroy())

    popup._zoom_photo = photo
    popup._zoom_pil = img
    popup._zoom_img_window = win

    # Grab can fail if called before the window is viewable on some Tk backends.
    def _safe_grab() -> None:
        try:
            if win.winfo_exists():
                win.grab_set()
        except tk.TclError:
            pass

    win.after_idle(_safe_grab)


def build_preview_panel(popup: Any, parent: ttk.Frame) -> None:
    """Build the live filter preview panel."""
    lf = ttk.LabelFrame(parent, text="Filter Preview", padding=8)
    lf.pack(fill="both", expand=True)

    sel_row = ttk.Frame(lf)
    sel_row.pack(fill="x", pady=(0, 4))

    ttk.Label(sel_row, text="Profile:").pack(side="left")
    popup._preview_profile_var = tk.StringVar()
    popup._preview_profile_combo = ttk.Combobox(
        sel_row,
        textvariable=popup._preview_profile_var,
        width=30,
        state="readonly",
    )
    popup._preview_profile_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)
    popup._preview_profile_combo.bind("<<ComboboxSelected>>", popup._on_preview_profile_selected)

    btn_row = ttk.Frame(lf)
    btn_row.pack(fill="x", pady=(4, 4))
    ttk.Button(btn_row, text="Render Preview", command=popup._render_preview).pack(side="left")

    popup._preview_status_var = tk.StringVar(value="")
    ttk.Label(lf, textvariable=popup._preview_status_var, foreground="#555", wraplength=310, justify="left"
              ).pack(anchor="w", pady=(0, 6))

    popup._preview_img_label = ttk.Label(lf, text="No preview yet", anchor="center")
    popup._preview_img_label.pack(fill="both", expand=True)

    populate_preview_profiles(popup)


def populate_preview_profiles(popup: Any) -> None:
    """Scan configs/profiles/ and populate the profile combobox."""
    if not getattr(popup, "_sim_root", None):
        popup._preview_status_var.set("Preview unavailable: sim_root not provided.")
        popup._preview_profile_combo["values"] = ()
        return

    try:
        import yaml as _yaml  # type: ignore
    except ImportError:
        popup._preview_status_var.set("PyYAML not installed – cannot load profiles.")
        return

    profiles_dir = Path(popup._sim_root) / "configs" / "profiles"
    if not profiles_dir.exists():
        popup._preview_status_var.set("profiles dir not found.")
        return

    vals_2d: list = []
    vals_3d: list = []
    for p in sorted(profiles_dir.glob("*.yaml")):
        try:
            data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            backends = list((data.get("profile") or {}).get("supported_render_backends") or [])
            pid = p.stem
            if "opencv_2d" in backends:
                vals_2d.append(pid)
            if "blender_3d" in backends:
                vals_3d.append(pid)
        except Exception:
            continue

    popup._preview_profiles_2d = set(vals_2d)
    popup._preview_profiles_3d = set(vals_3d)

    combo_vals: list = []
    if vals_2d:
        combo_vals.append("── 2D ──")
        combo_vals.extend(vals_2d)
    if vals_3d:
        combo_vals.append("── 3D ──")
        combo_vals.extend(vals_3d)

    popup._preview_profile_combo["values"] = combo_vals

    if vals_2d:
        popup._preview_profile_var.set(vals_2d[0])
        popup._preview_status_var.set("Click 'Render Preview' to see the current filter effect.")
    elif combo_vals:
        popup._preview_profile_var.set(combo_vals[0])
    else:
        popup._preview_status_var.set("No component profiles found.")


def on_preview_profile_selected(popup: Any, _evt: Optional[tk.Event] = None) -> None:
    """Skip separator header items in the combobox."""
    v = popup._preview_profile_var.get()
    if not v.startswith("──"):
        return
    vals = list(popup._preview_profile_combo["values"])
    try:
        idx = vals.index(v)
        for nxt in vals[idx + 1:]:
            if not nxt.startswith("──"):
                popup._preview_profile_var.set(nxt)
                return
    except ValueError:
        pass
    popup._preview_profile_var.set("")


def get_preview_filter_values(popup: Any) -> Dict[str, Any]:
    """Collect current filter values directly from the popup's internal vars."""
    result: Dict[str, Any] = {}
    for key, var in popup.filter_vars.items():
        if isinstance(var, tk.BooleanVar):
            result[key] = bool(var.get())
        elif isinstance(var, tk.DoubleVar):
            try:
                result[key] = float(var.get())
            except Exception:
                pass
        elif isinstance(var, tk.StringVar):
            result[key] = var.get()
    return result


def render_preview(popup: Any) -> None:
    """Start a background render with the current filter settings."""
    if popup._preview_is_rendering:
        return

    pid = popup._preview_profile_var.get().strip()
    if not pid or pid.startswith("──"):
        popup._preview_status_var.set("Select a valid profile first.")
        return

    is_3d_only = pid in popup._preview_profiles_3d and pid not in popup._preview_profiles_2d

    popup._preview_is_rendering = True
    popup._preview_status_var.set("Rendering…")

    filter_vals = get_preview_filter_values(popup)
    if is_3d_only:
        threading.Thread(target=popup._do_render_3d_preview, args=(pid, filter_vals), daemon=True).start()
    else:
        threading.Thread(target=popup._do_render_2d_preview, args=(pid, filter_vals), daemon=True).start()


def do_render_2d_preview(popup: Any, profile_id: str, filter_vals: Dict[str, Any]) -> None:
    """Background thread: render a 2D preview and schedule UI update."""
    try:
        import sys
        import numpy as np  # type: ignore

        sim_root = Path(popup._sim_root)  # type: ignore[arg-type]
        if str(sim_root) not in sys.path:
            sys.path.insert(0, str(sim_root))

        from simple_sim.profile_hash import load_profile  # type: ignore
        from simple_sim.generators.opencv_2d import (  # type: ignore
            render_roi,
            apply_image_filter_overrides,
            sample_nominal_geometry,
        )

        profiles_dir = sim_root / "configs" / "profiles"
        profile = load_profile(profile_id, profiles_dir)

        footprint: str = (profile.get("component") or {}).get("footprint", "chip_2pad")
        geometry_ranges = profile.get("geometry_ranges")
        tolerances = profile.get("tolerances")
        component_color = (profile.get("render") or {}).get("component_color_bgr", [20, 20, 20])

        render_cfg: Dict[str, Any] = {
            "substrate_color": [40, 90, 40],
            "copper_color": [60, 120, 180],
            "component_color": list(component_color),
        }

        rng = np.random.default_rng(42)
        nominal = sample_nominal_geometry({}, rng, geometry_ranges)
        defect_params: Dict[str, Any] = {
            "type": "OK",
            "shift_x": 0.0,
            "shift_y": 0.0,
            "rotation_deg": 0.0,
            "tilt_deg": 0.0,
        }

        base_aug: Dict[str, Any] = {
            "blur_sigma": 1.0,
            "noise_stddev": 8.0,
            "brightness_factor": 1.1,
            "contrast_factor": 1.1,
            "rotation_deg": 15.0,
        }
        augment = apply_image_filter_overrides(base_aug, filter_vals)

        roi_sizes: Dict[str, tuple] = {
            "soic_16": (600, 600),
            "qfn32": (320, 320),
            "sot23": (256, 256),
        }
        roi_size: tuple = roi_sizes.get(footprint, (256, 256))

        img_bgr = render_roi(
            nominal, defect_params, augment,
            roi_size, render_cfg, tolerances, rng,
            footprint=footprint,
        )

        popup.top.after(0, lambda img=img_bgr: popup._show_preview_image(img, profile_id))

    except Exception as exc:
        err = str(exc)
        popup.top.after(0, lambda e=err: popup._preview_status_var.set(f"Render error: {e}"))
    finally:
        popup.top.after(0, lambda: setattr(popup, "_preview_is_rendering", False))


def do_render_3d_preview(popup: Any, profile_id: str, filter_vals: Dict[str, Any]) -> None:
    """Background thread: render a 3D preview via Blender and schedule UI update."""
    try:
        import sys
        import numpy as np  # type: ignore

        sim_root = Path(popup._sim_root)  # type: ignore[arg-type]
        if str(sim_root) not in sys.path:
            sys.path.insert(0, str(sim_root))

        from simple_sim.profile_hash import load_profile  # type: ignore
        from simple_sim.config import load_config  # type: ignore
        from simple_sim.defects import sample_defect_params  # type: ignore
        from simple_sim.generators.blender_3d import write_jobs_jsonl, render_blender_batch  # type: ignore
        from simple_sim.generators.opencv_2d import (  # type: ignore
            apply_image_filter_overrides,
            sample_nominal_geometry,
        )

        profiles_dir = sim_root / "configs" / "profiles"
        profile = load_profile(profile_id, profiles_dir)

        blender_executable = "blender"
        cycles_samples = 32
        device = "CPU"
        roi_width_px = 320
        roi_height_px = 320
        mm_per_px = 0.02

        configs_dir = sim_root / "configs"
        for cfg_path in sorted(configs_dir.glob("*.yaml")):
            try:
                cfg = load_config(cfg_path)
            except Exception:
                continue
            if str((cfg.get("render") or {}).get("backend", "")) != "blender_3d":
                continue
            run = cfg.get("run") or {}
            if str(run.get("component_profile", "")) != str(profile_id):
                continue
            roi = cfg.get("roi") or {}
            roi_width_px = int(roi.get("width_px", roi_width_px))
            roi_height_px = int(roi.get("height_px", roi_height_px))
            mm_per_px = float(roi.get("mm_per_px", mm_per_px))
            blender_cfg = (cfg.get("render") or {}).get("blender") or {}
            blender_executable = str(blender_cfg.get("executable", blender_executable))
            cycles_samples = min(32, int(blender_cfg.get("samples", cycles_samples)))
            device = str(blender_cfg.get("device", device))
            break

        geometry_ranges = profile.get("geometry_ranges") or {}
        tolerances = profile.get("tolerances") or {}
        component_height_mm = float(((profile.get("component") or {}).get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45)
        render_3d = profile.get("render_3d") or {}
        footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))

        rng = np.random.default_rng(42)
        nominal = sample_nominal_geometry({}, rng, geometry_ranges)
        defect = sample_defect_params("OK", rng, tolerances=tolerances)

        base_augment: Dict[str, Any] = {
            "blur_sigma": 1.0,
            "noise_stddev": 8.0,
            "brightness_factor": 1.1,
            "contrast_factor": 1.1,
            "rotation_deg": 15.0,
        }
        augment = apply_image_filter_overrides(base_augment, filter_vals)

        out_root = sim_root / "outputs" / "filter_preview_3d"
        out_root.mkdir(parents=True, exist_ok=True)

        rel_path = f"preview_{profile_id}.png"
        job = {
            "image_path": rel_path,
            "seed": 42,
            "mm_per_px": mm_per_px,
            "roi_width_px": roi_width_px,
            "roi_height_px": roi_height_px,
            "footprint": footprint,
            "component_height_mm": component_height_mm,
            "nominal": nominal,
            "defect": defect,
            "augment": augment,
            "render_3d": render_3d,
        }

        jobs_path = out_root / f"jobs_{profile_id}.jsonl"
        write_jobs_jsonl(jobs_path, [job])

        render_blender_batch(
            sim_root=sim_root,
            jobs_path=jobs_path,
            output_root=out_root,
            blender_executable=blender_executable,
            cycles_samples=cycles_samples,
            device=device,
        )

        import cv2  # type: ignore
        out_img_path = out_root / rel_path
        if not out_img_path.exists():
            raise RuntimeError(f"Blender did not write output: {out_img_path}")

        img_bgr = cv2.imread(str(out_img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read rendered image: {out_img_path}")

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

        rng_post = np.random.default_rng(42)
        img_bgr = apply_blur(img_bgr, float(augment.get("blur_sigma", 0.0) or 0.0))
        img_bgr = apply_noise(img_bgr, float(augment.get("noise_stddev", 0.0) or 0.0), rng_post)
        img_bgr = apply_brightness(img_bgr, float(augment.get("brightness_factor", 1.0) or 1.0))
        img_bgr = apply_contrast(img_bgr, float(augment.get("contrast_factor", 1.0) or 1.0))
        img_bgr = apply_saturation(img_bgr, float(augment.get("saturation_factor", 1.0) or 1.0))
        img_bgr = apply_hue_shift(img_bgr, float(augment.get("hue_shift_deg", 0.0) or 0.0))
        img_bgr = apply_color_temperature(img_bgr, int(augment.get("color_temperature_kelvin", 5500) or 5500))
        img_bgr = apply_vignetting(img_bgr, float(augment.get("vignetting_strength", 0.0) or 0.0))
        img_bgr = apply_chromatic_aberration(img_bgr, float(augment.get("chromatic_strength", 0.0) or 0.0))
        img_bgr = apply_lens_distortion(img_bgr, float(augment.get("distortion_k1", 0.0) or 0.0), float(augment.get("distortion_k2", 0.0) or 0.0))
        img_bgr = apply_motion_blur(img_bgr, float(augment.get("motion_blur_strength", 0.0) or 0.0), float(augment.get("motion_blur_angle", 0.0) or 0.0))
        img_bgr = apply_sharpen(img_bgr, float(augment.get("sharpen_strength", 0.0) or 0.0))
        img_bgr = apply_shadow(img_bgr, float(augment.get("shadow_strength", 0.0) or 0.0), float(augment.get("shadow_size", 0.2) or 0.2), rng_post)
        img_bgr = apply_reflection(img_bgr, float(augment.get("reflection_strength", 0.0) or 0.0), float(augment.get("reflection_size", 0.15) or 0.15), rng_post)
        img_bgr = apply_dust_particles(img_bgr, float(augment.get("dust_density", 0.0) or 0.0), float(augment.get("dust_size", 2.0) or 2.0), rng_post)
        img_bgr = apply_jpeg_compression(img_bgr, int(augment.get("jpeg_quality", 100) or 100))
        img_bgr = apply_perspective_transform(img_bgr, float(augment.get("perspective_strength", 0.0) or 0.0), float(augment.get("perspective_angle_x", 0.0) or 0.0), float(augment.get("perspective_angle_y", 0.0) or 0.0))
        rotation_deg = float(augment.get("rotation_deg", 0.0) or 0.0)
        if abs(rotation_deg) > 0.1:
            h_img, w_img = img_bgr.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), rotation_deg, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, rot_mat, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

        popup.top.after(0, lambda img=img_bgr: popup._show_preview_image(img, profile_id, backend="3D"))

    except Exception as exc:
        err = str(exc)
        popup.top.after(0, lambda e=err: popup._preview_status_var.set(f"Render error: {e}"))
    finally:
        popup.top.after(0, lambda: setattr(popup, "_preview_is_rendering", False))


def show_preview_image(popup: Any, img_bgr: Any, profile_id: str = "", backend: str = "2D") -> None:
    """Display a rendered BGR numpy array in the preview label."""
    try:
        from PIL import Image, ImageTk  # type: ignore
        import numpy as np  # type: ignore

        img_rgb = img_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(img_rgb.astype(np.uint8))

        popup._preview_full_pil = pil_img.copy()

        max_px = 300
        w, h = pil_img.size
        if w > max_px or h > max_px:
            scale = max_px / max(w, h)
            pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        popup._preview_photo = ImageTk.PhotoImage(pil_img)
        popup._preview_img_label.configure(image=popup._preview_photo, text="")
        popup._preview_img_label.configure(cursor="hand2")
        popup._preview_img_label.bind(
            "<Button-1>",
            lambda _evt: _open_zoom_image_popup(
                popup,
                getattr(popup, "_preview_full_pil", None),
                f"Filter Preview - {profile_id} ({backend})",
            ),
        )
        popup._preview_status_var.set(f"Profile: {profile_id}  ·  defect: OK  ·  {backend}")

    except ImportError:
        popup._preview_status_var.set("Pillow not installed.\nRun: pip install Pillow")
    except Exception as exc:
        popup._preview_status_var.set(f"Display error: {exc}")


# ---------------------------------------------------------------------------
# Left-column profile preview panel
# ---------------------------------------------------------------------------

def build_left_preview_panel(popup: Any, parent: ttk.Frame) -> None:
    """Build a compact preview panel below the filter-profiles listbox.

    The dropdown lists all real component profiles from configs/profiles/ (both
    2D and 3D).  Render always uses the currently active filter settings from
    the popup's filter_vars – not any stored profile dict.
    """
    lf = ttk.LabelFrame(parent, text="Component Preview", padding=6)
    lf.pack(fill="x", pady=(8, 0))

    sel_row = ttk.Frame(lf)
    sel_row.pack(fill="x", pady=(0, 4))
    ttk.Label(sel_row, text="Profile:").pack(side="left")
    popup._left_preview_profile_var = tk.StringVar()
    popup._left_preview_profile_combo = ttk.Combobox(
        sel_row,
        textvariable=popup._left_preview_profile_var,
        width=16,
        state="readonly",
    )
    popup._left_preview_profile_combo.pack(side="left", padx=(4, 0), fill="x", expand=True)
    popup._left_preview_profile_combo.bind(
        "<<ComboboxSelected>>", popup._on_left_preview_profile_selected
    )

    ttk.Button(lf, text="Render", command=popup._render_left_preview).pack(anchor="w", pady=(0, 4))

    popup._left_preview_status_var = tk.StringVar(value="")
    ttk.Label(
        lf,
        textvariable=popup._left_preview_status_var,
        foreground="#555",
        wraplength=170,
        justify="left",
    ).pack(anchor="w", pady=(0, 4))

    popup._left_preview_img_label = ttk.Label(lf, text="No preview yet", anchor="center")
    popup._left_preview_img_label.pack(fill="both", expand=True)

    popup._left_preview_is_rendering = False
    popup._left_preview_photo = None
    popup._left_preview_profiles_2d: set = set()
    popup._left_preview_profiles_3d: set = set()

    populate_left_filter_profiles(popup)


def populate_left_filter_profiles(popup: Any) -> None:
    """Scan configs/profiles/ and fill the left combo with all component profiles (2D + 3D)."""
    combo = getattr(popup, "_left_preview_profile_combo", None)
    if combo is None:
        return
    status = getattr(popup, "_left_preview_status_var", None)

    if not getattr(popup, "_sim_root", None):
        if status:
            status.set("Preview unavailable: sim_root not set.")
        combo["values"] = ()
        return

    try:
        import yaml as _yaml  # type: ignore
    except ImportError:
        if status:
            status.set("PyYAML not installed.")
        return

    profiles_dir = Path(popup._sim_root) / "configs" / "profiles"
    if not profiles_dir.exists():
        if status:
            status.set(f"profiles dir not found:\n{profiles_dir}")
        return

    vals_2d: list = []
    vals_3d: list = []
    for p in sorted(profiles_dir.glob("*.yaml")):
        try:
            data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            backends = list((data.get("profile") or {}).get("supported_render_backends") or [])
            pid = p.stem
            if "opencv_2d" in backends:
                vals_2d.append(pid)
            if "blender_3d" in backends:
                vals_3d.append(pid)
        except Exception:
            continue

    popup._left_preview_profiles_2d = set(vals_2d)
    popup._left_preview_profiles_3d = set(vals_3d)

    combo_vals: list = []
    if vals_2d:
        combo_vals.append("── 2D ──")
        combo_vals.extend(vals_2d)
    if vals_3d:
        combo_vals.append("── 3D ──")
        combo_vals.extend(vals_3d)

    combo["values"] = combo_vals

    current = popup._left_preview_profile_var.get()
    if current not in combo_vals or current.startswith("──"):
        if vals_2d:
            popup._left_preview_profile_var.set(vals_2d[0])
        elif vals_3d:
            popup._left_preview_profile_var.set(vals_3d[0])

    if status:
        if combo_vals:
            status.set("Click 'Render' to preview with current filter settings.")
        else:
            status.set("No component profiles found.")


def on_left_preview_profile_selected(popup: Any, _evt: Optional[tk.Event] = None) -> None:
    """Skip separator header items in the left combo."""
    v = popup._left_preview_profile_var.get()
    if not v.startswith("──"):
        return
    vals = list(popup._left_preview_profile_combo["values"])
    try:
        idx = vals.index(v)
        for nxt in vals[idx + 1:]:
            if not nxt.startswith("──"):
                popup._left_preview_profile_var.set(nxt)
                return
    except ValueError:
        pass
    popup._left_preview_profile_var.set("")


def render_left_preview(popup: Any) -> None:
    """Render the selected component profile with the CURRENT active filter settings."""
    if getattr(popup, "_left_preview_is_rendering", False):
        return

    pid = popup._left_preview_profile_var.get().strip()
    if not pid or pid.startswith("──"):
        popup._left_preview_status_var.set("Select a valid component profile first.")
        return

    # Always use the live filter settings currently shown in the UI controls.
    filter_vals = get_preview_filter_values(popup)

    profiles_2d = getattr(popup, "_left_preview_profiles_2d", set())
    profiles_3d = getattr(popup, "_left_preview_profiles_3d", set())
    is_3d_only = pid in profiles_3d and pid not in profiles_2d

    popup._left_preview_is_rendering = True
    popup._left_preview_status_var.set("Rendering…")

    if is_3d_only:
        threading.Thread(
            target=_left_render_3d_thread,
            args=(popup, pid, filter_vals),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_left_render_2d_thread,
            args=(popup, pid, filter_vals),
            daemon=True,
        ).start()


def _left_render_2d_thread(popup: Any, profile_id: str, filter_vals: Dict[str, Any]) -> None:
    """Background thread: 2D render for the left preview panel."""
    try:
        import sys
        import numpy as np  # type: ignore

        sim_root = Path(popup._sim_root)
        if str(sim_root) not in sys.path:
            sys.path.insert(0, str(sim_root))

        from simple_sim.profile_hash import load_profile  # type: ignore
        from simple_sim.generators.opencv_2d import (  # type: ignore
            render_roi,
            apply_image_filter_overrides,
            sample_nominal_geometry,
        )

        profiles_dir = sim_root / "configs" / "profiles"
        profile = load_profile(profile_id, profiles_dir)

        footprint: str = (profile.get("component") or {}).get("footprint", "chip_2pad")
        geometry_ranges = profile.get("geometry_ranges")
        tolerances = profile.get("tolerances")
        component_color = (profile.get("render") or {}).get("component_color_bgr", [20, 20, 20])

        render_cfg: Dict[str, Any] = {
            "substrate_color": [40, 90, 40],
            "copper_color": [60, 120, 180],
            "component_color": list(component_color),
        }

        rng = np.random.default_rng(42)
        nominal = sample_nominal_geometry({}, rng, geometry_ranges)
        defect_params: Dict[str, Any] = {
            "type": "OK",
            "shift_x": 0.0,
            "shift_y": 0.0,
            "rotation_deg": 0.0,
            "tilt_deg": 0.0,
        }

        base_aug: Dict[str, Any] = {
            "blur_sigma": 1.0,
            "noise_stddev": 8.0,
            "brightness_factor": 1.1,
            "contrast_factor": 1.1,
            "rotation_deg": 15.0,
        }
        augment = apply_image_filter_overrides(base_aug, filter_vals)

        roi_sizes: Dict[str, tuple] = {
            "soic_16": (600, 600),
            "qfn32": (320, 320),
            "sot23": (256, 256),
        }
        roi_size: tuple = roi_sizes.get(footprint, (256, 256))

        img_bgr = render_roi(
            nominal, defect_params, augment,
            roi_size, render_cfg, tolerances, rng,
            footprint=footprint,
        )

        popup.top.after(0, lambda img=img_bgr: _show_left_preview_image(popup, img, profile_id, "2D"))

    except Exception as exc:
        err = str(exc)
        popup.top.after(0, lambda e=err: popup._left_preview_status_var.set(f"Render error: {e}"))
    finally:
        popup.top.after(0, lambda: setattr(popup, "_left_preview_is_rendering", False))


def _left_render_3d_thread(popup: Any, profile_id: str, filter_vals: Dict[str, Any]) -> None:
    """Background thread: 3D (Blender) render for the left preview panel."""
    try:
        import sys
        import numpy as np  # type: ignore

        sim_root = Path(popup._sim_root)
        if str(sim_root) not in sys.path:
            sys.path.insert(0, str(sim_root))

        from simple_sim.profile_hash import load_profile  # type: ignore
        from simple_sim.config import load_config  # type: ignore
        from simple_sim.defects import sample_defect_params  # type: ignore
        from simple_sim.generators.blender_3d import write_jobs_jsonl, render_blender_batch  # type: ignore
        from simple_sim.generators.opencv_2d import (  # type: ignore
            apply_image_filter_overrides,
            sample_nominal_geometry,
        )

        profiles_dir = sim_root / "configs" / "profiles"
        profile = load_profile(profile_id, profiles_dir)

        blender_executable = "blender"
        cycles_samples = 32
        device = "CPU"
        roi_width_px = 320
        roi_height_px = 320
        mm_per_px = 0.02

        configs_dir = sim_root / "configs"
        for cfg_path in sorted(configs_dir.glob("*.yaml")):
            try:
                cfg = load_config(cfg_path)
            except Exception:
                continue
            if str((cfg.get("render") or {}).get("backend", "")) != "blender_3d":
                continue
            if str((cfg.get("run") or {}).get("component_profile", "")) != str(profile_id):
                continue
            roi = cfg.get("roi") or {}
            roi_width_px = int(roi.get("width_px", roi_width_px))
            roi_height_px = int(roi.get("height_px", roi_height_px))
            mm_per_px = float(roi.get("mm_per_px", mm_per_px))
            blender_cfg = (cfg.get("render") or {}).get("blender") or {}
            blender_executable = str(blender_cfg.get("executable", blender_executable))
            cycles_samples = min(32, int(blender_cfg.get("samples", cycles_samples)))
            device = str(blender_cfg.get("device", device))
            break

        geometry_ranges = profile.get("geometry_ranges") or {}
        tolerances = profile.get("tolerances") or {}
        component_height_mm = float(
            ((profile.get("component") or {}).get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45
        )
        render_3d = profile.get("render_3d") or {}
        footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))

        rng = np.random.default_rng(42)
        nominal = sample_nominal_geometry({}, rng, geometry_ranges)
        defect = sample_defect_params("OK", rng, tolerances=tolerances)

        base_augment: Dict[str, Any] = {
            "blur_sigma": 1.0,
            "noise_stddev": 8.0,
            "brightness_factor": 1.1,
            "contrast_factor": 1.1,
            "rotation_deg": 15.0,
        }
        augment = apply_image_filter_overrides(base_augment, filter_vals)

        out_root = sim_root / "outputs" / "filter_preview_left"
        out_root.mkdir(parents=True, exist_ok=True)

        rel_path = f"preview_{profile_id}.png"
        job = {
            "image_path": rel_path,
            "seed": 42,
            "mm_per_px": mm_per_px,
            "roi_width_px": roi_width_px,
            "roi_height_px": roi_height_px,
            "footprint": footprint,
            "component_height_mm": component_height_mm,
            "nominal": nominal,
            "defect": defect,
            "augment": augment,
            "render_3d": render_3d,
        }

        jobs_path = out_root / f"jobs_{profile_id}.jsonl"
        write_jobs_jsonl(jobs_path, [job])

        render_blender_batch(
            sim_root=sim_root,
            jobs_path=jobs_path,
            output_root=out_root,
            blender_executable=blender_executable,
            cycles_samples=cycles_samples,
            device=device,
        )

        import cv2  # type: ignore
        out_img_path = out_root / rel_path
        if not out_img_path.exists():
            raise RuntimeError(f"Blender did not write output: {out_img_path}")
        img_bgr = cv2.imread(str(out_img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read rendered image: {out_img_path}")

        from simple_sim.generators.opencv_2d import (  # type: ignore
            apply_blur, apply_noise, apply_brightness, apply_contrast,
            apply_saturation, apply_hue_shift, apply_color_temperature,
            apply_vignetting, apply_chromatic_aberration, apply_lens_distortion,
            apply_motion_blur, apply_sharpen, apply_shadow, apply_reflection,
            apply_dust_particles, apply_jpeg_compression, apply_perspective_transform,
        )

        rng_post = np.random.default_rng(42)
        img_bgr = apply_blur(img_bgr, float(augment.get("blur_sigma", 0.0) or 0.0))
        img_bgr = apply_noise(img_bgr, float(augment.get("noise_stddev", 0.0) or 0.0), rng_post)
        img_bgr = apply_brightness(img_bgr, float(augment.get("brightness_factor", 1.0) or 1.0))
        img_bgr = apply_contrast(img_bgr, float(augment.get("contrast_factor", 1.0) or 1.0))
        img_bgr = apply_saturation(img_bgr, float(augment.get("saturation_factor", 1.0) or 1.0))
        img_bgr = apply_hue_shift(img_bgr, float(augment.get("hue_shift_deg", 0.0) or 0.0))
        img_bgr = apply_color_temperature(img_bgr, int(augment.get("color_temperature_kelvin", 5500) or 5500))
        img_bgr = apply_vignetting(img_bgr, float(augment.get("vignetting_strength", 0.0) or 0.0))
        img_bgr = apply_chromatic_aberration(img_bgr, float(augment.get("chromatic_strength", 0.0) or 0.0))
        img_bgr = apply_lens_distortion(img_bgr, float(augment.get("distortion_k1", 0.0) or 0.0), float(augment.get("distortion_k2", 0.0) or 0.0))
        img_bgr = apply_motion_blur(img_bgr, float(augment.get("motion_blur_strength", 0.0) or 0.0), float(augment.get("motion_blur_angle", 0.0) or 0.0))
        img_bgr = apply_sharpen(img_bgr, float(augment.get("sharpen_strength", 0.0) or 0.0))
        img_bgr = apply_shadow(img_bgr, float(augment.get("shadow_strength", 0.0) or 0.0), float(augment.get("shadow_size", 0.2) or 0.2), rng_post)
        img_bgr = apply_reflection(img_bgr, float(augment.get("reflection_strength", 0.0) or 0.0), float(augment.get("reflection_size", 0.15) or 0.15), rng_post)
        img_bgr = apply_dust_particles(img_bgr, float(augment.get("dust_density", 0.0) or 0.0), float(augment.get("dust_size", 2.0) or 2.0), rng_post)
        img_bgr = apply_jpeg_compression(img_bgr, int(augment.get("jpeg_quality", 100) or 100))
        img_bgr = apply_perspective_transform(img_bgr, float(augment.get("perspective_strength", 0.0) or 0.0), float(augment.get("perspective_angle_x", 0.0) or 0.0), float(augment.get("perspective_angle_y", 0.0) or 0.0))
        rotation_deg = float(augment.get("rotation_deg", 0.0) or 0.0)
        if abs(rotation_deg) > 0.1:
            h_img, w_img = img_bgr.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), rotation_deg, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, rot_mat, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

        popup.top.after(0, lambda img=img_bgr: _show_left_preview_image(popup, img, profile_id, "3D"))

    except Exception as exc:
        err = str(exc)
        popup.top.after(0, lambda e=err: popup._left_preview_status_var.set(f"Render error: {e}"))
    finally:
        popup.top.after(0, lambda: setattr(popup, "_left_preview_is_rendering", False))


def _show_left_preview_image(
    popup: Any, img_bgr: Any, profile_id: str, backend: str = "2D"
) -> None:
    """Display a rendered image in the left preview label."""
    try:
        from PIL import Image, ImageTk  # type: ignore
        import numpy as np  # type: ignore

        img_rgb = img_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(img_rgb.astype(np.uint8))

        popup._left_preview_full_pil = pil_img.copy()

        max_px = 180
        w, h = pil_img.size
        if w > max_px or h > max_px:
            scale = max_px / max(w, h)
            pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        popup._left_preview_photo = ImageTk.PhotoImage(pil_img)
        popup._left_preview_img_label.configure(image=popup._left_preview_photo, text="")
        popup._left_preview_img_label.configure(cursor="hand2")
        popup._left_preview_img_label.bind(
            "<Button-1>",
            lambda _evt: _open_zoom_image_popup(
                popup,
                getattr(popup, "_left_preview_full_pil", None),
                f"Component Preview - {profile_id} ({backend})",
            ),
        )
        popup._left_preview_status_var.set(f"Profile: {profile_id}  ·  {backend}")

    except ImportError:
        popup._left_preview_status_var.set("Pillow not installed.\nRun: pip install Pillow")
    except Exception as exc:
        popup._left_preview_status_var.set(f"Display error: {exc}")
