"""Shared Image Filter Popup Component.

This popup is used by:
- Main GUI pipeline tab (gui/tabs/pipeline/tab.py)
- Debug preview tool (scripts/render_debug_previews.py)

Both tools share the same filter settings via outputs/gui/settings.json.
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Set

from simple_sim.generators.filter_settings import (
    clean_filter_values_for_popup,
    default_filter_values_for_popup,
)

class FilterPopup:
    """Image filter configuration popup with profile management."""

    def __init__(
        self,
        parent: tk.Widget,
        profiles: Dict[str, Dict[str, Any]],
        active_profile: str,
        on_close: Callable[[Dict[str, Dict[str, Any]], str], None],
        get_current_values: Callable[[], Dict[str, Any]],
        set_current_values: Callable[[Dict[str, Any]], None],
        float_or_default: Callable[[str, float], float],
        show_messagebox: Callable[[str, str, str], None],
        ask_string: Callable[[str, str, str], Optional[str]],
        ask_yes_no: Callable[[str, str], bool],
        sim_root: Optional[str] = None,
    ):
        """Initialize filter popup.

        Args:
            parent: Parent widget
            profiles: Filter profiles dict (name -> settings)
            active_profile: Currently active profile name
            on_close: Callback(profiles, active_profile) when popup closes
            get_current_values: Function to get current filter values as dict
            set_current_values: Function to apply filter values from dict
            float_or_default: Function to parse float with default
            show_messagebox: Function to show message box (type, title, message)
            ask_string: Function to ask for string input (title, prompt, initial)
            ask_yes_no: Function to ask yes/no question (title, question)
            sim_root: Optional path to Simple-Sim root; enables the live filter
                      preview panel when provided.
        """
        self.profiles = profiles
        self.active_profile = active_profile
        self.on_close = on_close
        self.get_current_values = get_current_values
        self.set_current_values = set_current_values
        self.float_or_default = float_or_default
        self.show_messagebox = show_messagebox
        self.ask_string = ask_string
        self.ask_yes_no = ask_yes_no
        self._sim_root: Optional[str] = str(sim_root) if sim_root is not None else None

        # Preview-panel state (populated in _build_preview_panel)
        self._preview_profiles_2d: Set[str] = set()
        self._preview_profiles_3d: Set[str] = set()
        self._preview_is_rendering: bool = False
        self._preview_photo: Any = None  # keep PhotoImage reference alive

        # Create popup window
        self.top = tk.Toplevel(parent)
        self.top.title("Image Filters")
        self.top.transient(parent.winfo_toplevel())
        # Don't grab_set() so user can interact with main window
        self.top.resizable(True, True)
        min_w = 1310 if self._sim_root else 980
        self.top.minsize(min_w, 640)

        # Build UI
        self._build_ui()
        self._bind_mousewheel()

        # Set up close handler
        self.top.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _build_ui(self) -> None:
        """Build the popup UI."""
        root = ttk.Frame(self.top, padding=12)
        root.pack(fill="both", expand=True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        # Left: Profile list
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        self._build_profile_list(left)

        # Middle: Filter controls
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_filter_controls(right)

        # Right: Filter preview panel (only when sim_root is available)
        if self._sim_root:
            preview_col = ttk.Frame(root)
            preview_col.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
            root.columnconfigure(2, minsize=340)
            self._build_preview_panel(preview_col)

    def _build_profile_list(self, parent: ttk.Frame) -> None:
        """Build profile list UI."""
        ttk.Label(parent, text="Filter Profiles", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.lb_profiles = tk.Listbox(parent, height=10, width=22, exportselection=False)
        self.lb_profiles.pack(fill="y", pady=(6, 6))

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="New", width=7, command=self._new_profile).pack(side="left")
        ttk.Button(btn_frame, text="Save", width=9, command=self._save_profile).pack(side="left", padx=(4, 0))
        ttk.Button(btn_frame, text="Rename", width=11, command=self._rename_profile).pack(side="left", padx=(4, 0))
        ttk.Button(btn_frame, text="Delete", width=8, command=self._delete_profile).pack(side="left", padx=(4, 0))

        self.lb_profiles.bind("<<ListboxSelect>>", self._load_profile_from_selection)
        self.lb_profiles.bind("<Double-Button-1>", self._load_profile_from_selection)

        self._refresh_profile_list(select_name=self.active_profile)
        self.set_current_values(self.profiles.get(self.active_profile, {}))

    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        """Build filter controls UI."""
        ttk.Label(parent, text="Apply filters during image generation",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Configure strength and enable/disable filters below.").pack(anchor="w", pady=(2, 8))

        # Get current filter values
        current = self.get_current_values()

        # Cardinal rotation toggles
        row0 = ttk.Frame(parent)
        row0.pack(fill="x", pady=(0, 8))

        self.var_cardinal = tk.BooleanVar(value=current.get("cardinal_rotation_90", True))
        self.var_rotation = tk.BooleanVar(value=current.get("enable_rotation", True))

        ttk.Checkbutton(row0, text="Enable 90° base rotation", variable=self.var_cardinal).pack(side="left")
        ttk.Checkbutton(row0, text="Enable rotation jitter", variable=self.var_rotation).pack(side="left", padx=(12, 0))

        # Scrollable filter controls
        canvas = tk.Canvas(parent)
        self._scroll_canvas = canvas
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        controls_frame = ttk.Frame(canvas)

        controls_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store filter variables
        self.filter_vars: Dict[str, Any] = {
            "cardinal_rotation_90": self.var_cardinal,
            "enable_rotation": self.var_rotation,
        }

        # Add filter rows
        self._add_filter_sections(controls_frame, current)
        self.var_cardinal.trace_add("write", lambda *_args: self._update_rotation_row_state())
        self._update_rotation_row_state()

        # Bottom buttons
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Clean", command=self._set_clean_filters).pack(side="left")
        ttk.Button(btns, text="Reset Defaults", command=self._reset_defaults).pack(side="left")
        ttk.Button(btns, text="Reset to 1.0", command=self._reset_to_one).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Close", command=self._handle_close).pack(side="right")

    # ------------------------------------------------------------------
    # Filter preview panel
    # ------------------------------------------------------------------

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        """Build the live filter preview panel."""
        lf = ttk.LabelFrame(parent, text="Filter Preview", padding=8)
        lf.pack(fill="both", expand=True)

        # ── Profile selector ──────────────────────────────────────────
        sel_row = ttk.Frame(lf)
        sel_row.pack(fill="x", pady=(0, 4))

        ttk.Label(sel_row, text="Profile:").pack(side="left")
        self._preview_profile_var = tk.StringVar()
        self._preview_profile_combo = ttk.Combobox(
            sel_row,
            textvariable=self._preview_profile_var,
            width=30,
            state="readonly",
        )
        self._preview_profile_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)
        self._preview_profile_combo.bind("<<ComboboxSelected>>", self._on_preview_profile_selected)

        # ── Render button ─────────────────────────────────────────────
        btn_row = ttk.Frame(lf)
        btn_row.pack(fill="x", pady=(4, 4))
        ttk.Button(btn_row, text="Render Preview", command=self._render_preview).pack(side="left")

        # ── Status label ──────────────────────────────────────────────
        self._preview_status_var = tk.StringVar(value="")
        ttk.Label(lf, textvariable=self._preview_status_var, foreground="#555",
                  wraplength=310, justify="left").pack(anchor="w", pady=(0, 6))

        # ── Image display ─────────────────────────────────────────────
        self._preview_img_label = ttk.Label(lf, text="No preview yet", anchor="center")
        self._preview_img_label.pack(fill="both", expand=True)

        # Populate combo with discovered profiles
        self._populate_preview_profiles()

    def _populate_preview_profiles(self) -> None:
        """Scan configs/profiles/ and populate the profile combobox."""
        try:
            import yaml as _yaml  # type: ignore
        except ImportError:
            self._preview_status_var.set("PyYAML not installed – cannot load profiles.")
            return

        profiles_dir = Path(self._sim_root) / "configs" / "profiles"
        if not profiles_dir.exists():
            self._preview_status_var.set("profiles dir not found.")
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

        self._preview_profiles_2d = set(vals_2d)
        self._preview_profiles_3d = set(vals_3d)

        combo_vals: list = []
        if vals_2d:
            combo_vals.append("── 2D ──")
            combo_vals.extend(vals_2d)
        if vals_3d:
            combo_vals.append("── 3D ──")
            combo_vals.extend(vals_3d)

        self._preview_profile_combo["values"] = combo_vals

        # Pre-select first 2D profile
        if vals_2d:
            self._preview_profile_var.set(vals_2d[0])
            self._preview_status_var.set("Click 'Render Preview' to see the current filter effect.")
        elif combo_vals:
            self._preview_profile_var.set(combo_vals[0])
        else:
            self._preview_status_var.set("No component profiles found.")

    def _on_preview_profile_selected(self, _evt: Optional[tk.Event] = None) -> None:
        """Skip separator header items in the combobox."""
        v = self._preview_profile_var.get()
        if not v.startswith("──"):
            return
        # Jump to the next real entry after the header
        vals = list(self._preview_profile_combo["values"])
        try:
            idx = vals.index(v)
            for nxt in vals[idx + 1:]:
                if not nxt.startswith("──"):
                    self._preview_profile_var.set(nxt)
                    return
        except ValueError:
            pass
        self._preview_profile_var.set("")

    def _get_preview_filter_values(self) -> Dict[str, Any]:
        """Collect current filter values directly from the popup's internal vars."""
        result: Dict[str, Any] = {}
        for key, var in self.filter_vars.items():
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

    def _render_preview(self) -> None:
        """Start a background render with the current filter settings."""
        if self._preview_is_rendering:
            return

        pid = self._preview_profile_var.get().strip()
        if not pid or pid.startswith("──"):
            self._preview_status_var.set("Select a valid profile first.")
            return

        is_3d_only = pid in self._preview_profiles_3d and pid not in self._preview_profiles_2d

        self._preview_is_rendering = True
        self._preview_status_var.set("Rendering…")

        filter_vals = self._get_preview_filter_values()
        if is_3d_only:
            threading.Thread(
                target=self._do_render_3d_preview,
                args=(pid, filter_vals),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._do_render_2d_preview,
                args=(pid, filter_vals),
                daemon=True,
            ).start()

    def _do_render_2d_preview(self, profile_id: str, filter_vals: Dict[str, Any]) -> None:
        """Background thread: render a 2D preview and schedule UI update."""
        try:
            import sys
            import numpy as np  # type: ignore

            sim_root = Path(self._sim_root)  # type: ignore[arg-type]
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

            # Small baseline so individual filter effects are clearly visible.
            # rotation_deg must be non-zero so rotation_strength actually has an effect.
            base_aug: Dict[str, Any] = {
                "blur_sigma": 1.0,
                "noise_stddev": 8.0,
                "brightness_factor": 1.1,
                "contrast_factor": 1.1,
                "rotation_deg": 15.0,
            }
            augment = apply_image_filter_overrides(base_aug, filter_vals)

            # ROI size by footprint family
            _ROI_SIZES: Dict[str, tuple] = {
                "soic_16": (600, 600),
                "qfn32": (320, 320),
                "sot23": (256, 256),
            }
            roi_size: tuple = _ROI_SIZES.get(footprint, (256, 256))

            img_bgr = render_roi(
                nominal, defect_params, augment,
                roi_size, render_cfg, tolerances, rng,
                footprint=footprint,
            )

            self.top.after(0, lambda img=img_bgr: self._show_preview_image(img, profile_id))

        except Exception as exc:
            err = str(exc)
            self.top.after(0, lambda e=err: self._preview_status_var.set(f"Render error: {e}"))
        finally:
            self.top.after(0, lambda: setattr(self, "_preview_is_rendering", False))

    def _do_render_3d_preview(self, profile_id: str, filter_vals: Dict[str, Any]) -> None:
        """Background thread: render a 3D preview via Blender and schedule UI update."""
        try:
            import sys
            import numpy as np  # type: ignore

            sim_root = Path(self._sim_root)  # type: ignore[arg-type]
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

            # --- Defaults; overridden if a matching run config is found ---
            blender_executable = "blender"
            cycles_samples = 32  # keep low for a quick preview
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
                # Match found – extract render settings
                roi = cfg.get("roi") or {}
                roi_width_px = int(roi.get("width_px", roi_width_px))
                roi_height_px = int(roi.get("height_px", roi_height_px))
                mm_per_px = float(roi.get("mm_per_px", mm_per_px))
                blender_cfg = (cfg.get("render") or {}).get("blender") or {}
                blender_executable = str(blender_cfg.get("executable", blender_executable))
                # Cap samples to 32 for a fast popup preview
                cycles_samples = min(32, int(blender_cfg.get("samples", cycles_samples)))
                device = str(blender_cfg.get("device", device))
                break

            # --- Profile fields ---
            geometry_ranges = profile.get("geometry_ranges") or {}
            tolerances = profile.get("tolerances") or {}
            component_height_mm = float(
                ((profile.get("component") or {}).get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45
            )
            render_3d = profile.get("render_3d") or {}
            footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))

            # --- Sample scene ---
            rng = np.random.default_rng(42)
            nominal = sample_nominal_geometry({}, rng, geometry_ranges)
            defect = sample_defect_params("OK", rng, tolerances=tolerances)

            # Non-zero base rotation so rotation_strength / cardinal mode are visible.
            base_augment: Dict[str, Any] = {
                "blur_sigma": 1.0,
                "noise_stddev": 8.0,
                "brightness_factor": 1.1,
                "contrast_factor": 1.1,
                "rotation_deg": 15.0,
            }
            augment = apply_image_filter_overrides(base_augment, filter_vals)

            # --- Build job and run Blender ---
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

            # --- Load rendered image and apply image-space filters ---
            # Blender only uses rotation_deg for the 3D scene orientation;
            # all other filters (blur, noise, brightness, etc.) must be applied here.
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

            self.top.after(0, lambda img=img_bgr: self._show_preview_image(img, profile_id, backend="3D"))

        except Exception as exc:
            err = str(exc)
            self.top.after(0, lambda e=err: self._preview_status_var.set(f"Render error: {e}"))
        finally:
            self.top.after(0, lambda: setattr(self, "_preview_is_rendering", False))

    def _show_preview_image(self, img_bgr: Any, profile_id: str = "", backend: str = "2D") -> None:
        """Display a rendered BGR numpy array in the preview label."""
        try:
            from PIL import Image, ImageTk  # type: ignore
            import numpy as np  # type: ignore

            img_rgb = img_bgr[:, :, ::-1].copy()
            pil_img = Image.fromarray(img_rgb.astype(np.uint8))

            # Scale down to fit the panel (max 300 px on longest side)
            max_px = 300
            w, h = pil_img.size
            if w > max_px or h > max_px:
                scale = max_px / max(w, h)
                pil_img = pil_img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.LANCZOS,
                )

            self._preview_photo = ImageTk.PhotoImage(pil_img)
            self._preview_img_label.configure(image=self._preview_photo, text="")
            self._preview_status_var.set(f"Profile: {profile_id}  ·  defect: OK  ·  {backend}")

        except ImportError:
            self._preview_status_var.set(
                "Pillow not installed.\n"
                "Run: pip install Pillow"
            )
        except Exception as exc:
            self._preview_status_var.set(f"Display error: {exc}")

    # ------------------------------------------------------------------
    # Mouse-wheel scrolling
    # ------------------------------------------------------------------

    def _bind_mousewheel(self) -> None:
        """Route mouse wheel events inside popup to filter list scrolling."""
        self.top.bind("<MouseWheel>", self._on_mousewheel)
        # Linux/X11 wheel events
        self.top.bind("<Button-4>", self._on_mousewheel)
        self.top.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event: tk.Event) -> str:
        """Scroll popup canvas while cursor is inside popup widgets."""
        canvas = getattr(self, "_scroll_canvas", None)
        if canvas is None:
            return "break"
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
            return "break"
        if getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
            return "break"
        delta = int(getattr(event, "delta", 0))
        if delta != 0:
            # Windows/macOS: delta sign indicates direction.
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")
        return "break"

    def _add_filter_sections(self, parent: ttk.Frame, current: Dict[str, Any]) -> None:
        """Add all filter sections with controls."""
        self._rotation_row_state_cb: Optional[Callable[[], None]] = None

        def add_section(title: str) -> None:
            ttk.Label(parent, text=title, font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(6, 3))

        def add_filter_row(
            label: str,
            key_enable: str,
            key_strength: str,
            default_enable: bool,
            default_strength: float,
            min_v: float = 0.0,
            max_v: float = 2.0,
            enable_var: Optional[tk.BooleanVar] = None,
        ) -> None:
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=(3, 0))
            row_main = ttk.Frame(row)
            row_main.pack(fill="x")
            row_rand = ttk.Frame(row)
            row_rand.pack(fill="x", pady=(2, 0))

            var_enable = enable_var if enable_var is not None else tk.BooleanVar(value=current.get(key_enable, default_enable))
            var_strength = tk.DoubleVar(
                value=self.float_or_default(
                    str(current.get(key_strength, default_strength)),
                    default=default_strength,
                )
            )

            chk_enable = ttk.Checkbutton(row_main, text=label, variable=var_enable)
            chk_enable.pack(side="left")

            scale = ttk.Scale(row_main, from_=min_v, to=max_v, orient="horizontal",
                            variable=var_strength, length=170)
            scale.pack(side="left", padx=(10, 6))

            lbl = ttk.Label(row_main, width=5, anchor="e")
            lbl.pack(side="left")

            def update_label(*args):
                v = max(min_v, min(max_v, var_strength.get()))
                lbl.configure(text=f"{v:.2f}")

            var_strength.trace_add("write", lambda *args: update_label())
            update_label()

            # Per-filter randomization: when enabled, slider is disabled and
            # min/max values are used to sample a random value per image.
            key_randomize = f"{key_strength}_randomize"
            key_min = f"{key_strength}_min"
            key_max = f"{key_strength}_max"
            base_val = self.float_or_default(str(current.get(key_strength, default_strength)), default=default_strength)
            var_randomize = tk.BooleanVar(value=bool(current.get(key_randomize, False)))
            var_min = tk.DoubleVar(
                value=self.float_or_default(str(current.get(key_min, base_val)), default=base_val)
            )
            var_max = tk.DoubleVar(
                value=self.float_or_default(str(current.get(key_max, base_val)), default=base_val)
            )

            ttk.Label(row_rand, text="Random").pack(side="left")
            chk_random = ttk.Checkbutton(row_rand, text="Rnd", variable=var_randomize)
            chk_random.pack(side="left", padx=(6, 6))
            ttk.Label(row_rand, text="Min").pack(side="left")
            ent_min = ttk.Entry(row_rand, textvariable=var_min, width=7)
            ent_min.pack(side="left", padx=(4, 8))
            ttk.Label(row_rand, text="Max").pack(side="left")
            ent_max = ttk.Entry(row_rand, textvariable=var_max, width=7)
            ent_max.pack(side="left", padx=(4, 0))

            is_rotation_row = (key_strength == "rotation_strength")

            def _update_random_mode(*_args: Any) -> None:
                cardinal_only = bool(self.var_cardinal.get()) if is_rotation_row else False
                if cardinal_only:
                    var_enable.set(False)
                    if bool(var_randomize.get()):
                        var_randomize.set(False)
                    chk_enable.state(["disabled"])
                    chk_random.state(["disabled"])
                    scale.state(["disabled"])
                    ent_min.state(["disabled"])
                    ent_max.state(["disabled"])
                    return
                chk_enable.state(["!disabled"])
                chk_random.state(["!disabled"])
                if bool(var_randomize.get()):
                    scale.state(["disabled"])
                    ent_min.state(["!disabled"])
                    ent_max.state(["!disabled"])
                else:
                    scale.state(["!disabled"])
                    ent_min.state(["disabled"])
                    ent_max.state(["disabled"])

            var_randomize.trace_add("write", _update_random_mode)
            _update_random_mode()
            if is_rotation_row:
                self._rotation_row_state_cb = _update_random_mode

            self.filter_vars[key_enable] = var_enable
            self.filter_vars[key_strength] = var_strength
            self.filter_vars[key_randomize] = var_randomize
            self.filter_vars[key_min] = var_min
            self.filter_vars[key_max] = var_max

        # Existing filters
        add_section("━━━ Existing Filters ━━━")
        add_filter_row("Blur", "enable_blur", "blur_strength", True, 1.0)
        add_filter_row("Grain", "enable_grain", "grain_strength", True, 1.0, max_v=3.0)
        add_filter_row("Brightness", "enable_brightness", "brightness_strength", True, 1.0, max_v=2.0)
        add_filter_row("Contrast", "enable_contrast", "contrast_strength", True, 1.0, max_v=2.0)
        add_filter_row("Rotation", "enable_rotation", "rotation_strength", True, 1.0, max_v=2.0, enable_var=self.var_rotation)

        # High priority
        add_section("━━━ High Priority ━━━")
        add_filter_row("Perspective Transform", "enable_perspective", "perspective_strength", False, 1.0, max_v=2.0)
        add_filter_row("Motion Blur", "enable_motion_blur", "motion_blur_strength", True, 1.0, max_v=3.0)
        add_filter_row("Saturation", "enable_saturation", "saturation_factor", True, 1.0, min_v=0.5, max_v=1.5)
        add_filter_row("Hue Shift", "enable_hue_shift", "hue_shift_deg", True, 0.0, min_v=-30.0, max_v=30.0)
        add_filter_row("Shadow", "enable_shadow", "shadow_strength", True, 0.3, min_v=0.0, max_v=0.8)
        add_filter_row("Reflection/Glare", "enable_reflection", "reflection_strength", False, 0.5, min_v=0.0, max_v=1.0)

        # Medium priority
        add_section("━━━ Medium Priority (disabled) ━━━")
        add_filter_row("Vignetting", "enable_vignetting", "vignetting_strength", False, 1.0, max_v=2.0)
        add_filter_row("Chromatic Aberration", "enable_chromatic_aberration", "chromatic_strength", False, 1.0, max_v=2.0)
        add_filter_row("JPEG Quality", "enable_jpeg_compression", "jpeg_quality", False, 85.0, min_v=50.0, max_v=95.0)
        add_filter_row("Color Temperature", "enable_color_temperature", "color_temperature_kelvin", False, 5500.0, min_v=2500.0, max_v=7500.0)

        # Low priority
        add_section("━━━ Low Priority (disabled) ━━━")
        add_filter_row("Lens Distortion K1", "enable_lens_distortion", "distortion_k1", False, 0.0, min_v=-0.3, max_v=0.3)
        add_filter_row("Dust Particles", "enable_dust", "dust_density", False, 0.3, min_v=0.0, max_v=1.0)
        add_filter_row("Sharpen", "enable_sharpen", "sharpen_strength", False, 1.0, max_v=2.0)

    def _refresh_profile_list(self, select_name: Optional[str] = None) -> None:
        """Refresh profile list."""
        names = sorted(self.profiles.keys(), key=lambda s: s.lower())
        self.lb_profiles.delete(0, "end")
        for n in names:
            self.lb_profiles.insert("end", n)
        target = select_name or self.active_profile
        if target in names:
            idx = names.index(target)
            self.lb_profiles.selection_clear(0, "end")
            self.lb_profiles.selection_set(idx)
            self.lb_profiles.activate(idx)

    def _selected_profile_name(self) -> Optional[str]:
        """Get selected profile name."""
        sel = self.lb_profiles.curselection()
        if not sel:
            return None
        try:
            return str(self.lb_profiles.get(sel[0]))
        except Exception:
            return None

    def _persist_active_profile(self) -> None:
        """Save current filter values to active profile."""
        if self.active_profile in self.profiles:
            # First sync internal popup vars to external vars
            self._sync_to_external_vars()
            # Then get the values and save to profile
            self.profiles[self.active_profile] = self.get_current_values()

    def _sync_to_external_vars(self) -> None:
        """Sync internal popup variables to external variables via set_current_values."""
        current_popup_values = {}
        for key, var in self.filter_vars.items():
            if isinstance(var, tk.BooleanVar):
                current_popup_values[key] = var.get()
            elif isinstance(var, tk.DoubleVar):
                current_popup_values[key] = f"{var.get():.2f}"
            elif isinstance(var, tk.StringVar):
                current_popup_values[key] = var.get()
        # Push to external variables
        self.set_current_values(current_popup_values)

    def _update_internal_vars(self, data: Dict[str, Any]) -> None:
        """Update internal popup variables from profile data."""
        for key, var in self.filter_vars.items():
            if key in data:
                value = data[key]
                if isinstance(var, tk.BooleanVar):
                    if isinstance(value, str):
                        var.set(value.strip().lower() in {"1", "true", "yes", "on"})
                    else:
                        var.set(bool(value))
                elif isinstance(var, (tk.DoubleVar, tk.StringVar)):
                    try:
                        if isinstance(var, tk.DoubleVar):
                            var.set(float(value))
                        else:
                            var.set(str(value))
                    except (ValueError, TypeError):
                        pass

    def _load_profile_from_selection(self, _evt: Optional[tk.Event] = None) -> None:
        """Load selected profile."""
        name = self._selected_profile_name()
        if not name:
            return
        self._persist_active_profile()
        data = self.profiles.get(name)
        if isinstance(data, dict):
            self.set_current_values(data)
            self.active_profile = name
            # Also update internal popup variables
            self._update_internal_vars(data)
            self._persist_active_profile()

    def _new_profile(self) -> None:
        """Create new profile."""
        name = self.ask_string("New Filter Profile", "Profile name:", "")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self.profiles:
            self.show_messagebox("warning", "Exists", f"Profile '{name}' already exists.")
            return
        # Sync popup values to external vars first
        self._sync_to_external_vars()
        self.profiles[name] = self.get_current_values()
        self.active_profile = name
        self._refresh_profile_list(select_name=name)

    def _save_profile(self) -> None:
        """Save current values to selected profile."""
        name = self._selected_profile_name()
        if not name:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        # Sync popup values to external vars first
        self._sync_to_external_vars()
        self.profiles[name] = self.get_current_values()
        self.active_profile = name

    def _rename_profile(self) -> None:
        """Rename selected profile."""
        old = self._selected_profile_name()
        if not old:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        new = self.ask_string("Rename Filter Profile", "New name:", old)
        if not new:
            return
        new = new.strip()
        if not new or new == old:
            return
        if new in self.profiles:
            self.show_messagebox("warning", "Exists", f"Profile '{new}' already exists.")
            return
        self.profiles[new] = self.profiles.pop(old)
        if self.active_profile == old:
            self.active_profile = new
        self._persist_active_profile()
        self._refresh_profile_list(select_name=new)

    def _delete_profile(self) -> None:
        """Delete selected profile."""
        name = self._selected_profile_name()
        if not name:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        if len(self.profiles) <= 1:
            self.show_messagebox("warning", "Blocked", "At least one profile must remain.")
            return
        if not self.ask_yes_no("Delete Filter Profile", f"Delete profile '{name}'?"):
            return
        self.profiles.pop(name, None)
        if self.active_profile == name:
            self.active_profile = sorted(self.profiles.keys(), key=lambda s: s.lower())[0]
            data = self.profiles.get(self.active_profile, {})
            self.set_current_values(data)
            self._update_internal_vars(data)
        self._persist_active_profile()
        self._refresh_profile_list(select_name=self.active_profile)

    def _reset_defaults(self) -> None:
        """Reset all filters to default values."""
        defaults = self._get_default_values()
        self.set_current_values(defaults)
        self._update_internal_vars(defaults)
        self._persist_active_profile()

    def _reset_to_one(self) -> None:
        """Reset selected profile to all 1.0."""
        name = self._selected_profile_name()
        if not name:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        reset_vals = self._get_default_values()
        self.profiles[name] = reset_vals
        self.set_current_values(reset_vals)
        self._update_internal_vars(reset_vals)
        self._persist_active_profile()
        self._refresh_profile_list(select_name=name)

    def _set_clean_filters(self) -> None:
        """Disable all filters and use neutral values for clean image generation."""
        clean = self._get_clean_values()
        self.set_current_values(clean)
        self._update_internal_vars(clean)
        self._persist_active_profile()

    def _get_default_values(self) -> Dict[str, Any]:
        """Get default filter values."""
        return default_filter_values_for_popup()

    def _get_clean_values(self) -> Dict[str, Any]:
        """Get a neutral preset with all filters disabled."""
        return clean_filter_values_for_popup()

    def _update_rotation_row_state(self) -> None:
        """Enforce 90-degree-only mode when cardinal rotation is enabled."""
        if self._rotation_row_state_cb is not None:
            self._rotation_row_state_cb()

    def _handle_close(self) -> None:
        """Handle popup close."""
        self._persist_active_profile()
        self.on_close(self.profiles, self.active_profile)
        self.top.destroy()


def open_filter_popup(
    parent: tk.Widget,
    profiles: Dict[str, Dict[str, Any]],
    active_profile: str,
    on_close: Callable[[Dict[str, Dict[str, Any]], str], None],
    get_current_values: Callable[[], Dict[str, Any]],
    set_current_values: Callable[[Dict[str, Any]], None],
    float_or_default: Callable[[str, float], float],
    show_messagebox: Callable[[str, str, str], None],
    ask_string: Callable[[str, str, str], Optional[str]],
    ask_yes_no: Callable[[str, str], bool],
    sim_root: Optional[str] = None,
) -> None:
    """Open filter popup (convenience function).

    Args:
        parent: Parent widget
        profiles: Filter profiles dict
        active_profile: Active profile name
        on_close: Callback when closed
        get_current_values: Get current filter values
        set_current_values: Set filter values
        float_or_default: Parse float with default
        show_messagebox: Show message box
        ask_string: Ask for string input
        ask_yes_no: Ask yes/no question
        sim_root: Optional path to Simple-Sim root for the preview panel
    """
    FilterPopup(
        parent=parent,
        profiles=profiles,
        active_profile=active_profile,
        on_close=on_close,
        get_current_values=get_current_values,
        set_current_values=set_current_values,
        float_or_default=float_or_default,
        show_messagebox=show_messagebox,
        ask_string=ask_string,
        ask_yes_no=ask_yes_no,
        sim_root=sim_root,
    )
