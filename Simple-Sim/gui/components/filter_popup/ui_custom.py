"""Custom tab UI helpers for the filter popup."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, Callable

from .ui_realism import build_realism_controls


def build_filter_controls(popup: Any, parent: ttk.Frame) -> None:
    """Build filter controls UI."""
    ttk.Label(parent, text="Apply filters during image generation",
              font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    ttk.Label(parent, text="Configure Custom filters or switch to grouped Realism mode.").pack(anchor="w", pady=(2, 8))

    current = popup.get_current_values()

    popup.filter_vars = {}
    popup.var_filter_mode = tk.StringVar(value=str(current.get("filter_mode", "custom") or "custom"))
    if popup.var_filter_mode.get() not in {"custom", "realism"}:
        popup.var_filter_mode.set("custom")
    popup.var_realism_enabled = tk.BooleanVar(value=bool(current.get("realism_enabled", False)))
    popup.filter_vars["filter_mode"] = popup.var_filter_mode
    popup.filter_vars["realism_enabled"] = popup.var_realism_enabled

    mode_row = ttk.Frame(parent)
    mode_row.pack(fill="x", pady=(0, 8))
    ttk.Label(mode_row, text="Mode:").pack(side="left")
    ttk.Radiobutton(mode_row, text="Custom", variable=popup.var_filter_mode, value="custom").pack(side="left", padx=(8, 0))
    ttk.Radiobutton(mode_row, text="Realism", variable=popup.var_filter_mode, value="realism").pack(side="left", padx=(8, 0))
    ttk.Checkbutton(mode_row, text="Enable realism mode", variable=popup.var_realism_enabled).pack(side="left", padx=(16, 0))

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True)
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, minsize=360)
    body.rowconfigure(0, weight=1)

    notebook = ttk.Notebook(body)
    notebook.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    tab_custom = ttk.Frame(notebook)
    tab_realism = ttk.Frame(notebook)
    notebook.add(tab_custom, text="Custom")
    notebook.add(tab_realism, text="Realism")

    row0 = ttk.Frame(tab_custom)
    row0.pack(fill="x", pady=(0, 8))

    popup.var_cardinal = tk.BooleanVar(value=current.get("cardinal_rotation_90", True))
    popup.var_rotation = tk.BooleanVar(value=current.get("enable_rotation", True))

    ttk.Checkbutton(row0, text="Enable 90° base rotation", variable=popup.var_cardinal).pack(side="left")
    ttk.Checkbutton(row0, text="Enable rotation jitter", variable=popup.var_rotation).pack(side="left", padx=(12, 0))

    canvas = tk.Canvas(tab_custom)
    popup._scroll_canvas = canvas
    scrollbar = ttk.Scrollbar(tab_custom, orient="vertical", command=canvas.yview)
    controls_frame = ttk.Frame(canvas)

    controls_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=controls_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    popup.filter_vars.update({
        "cardinal_rotation_90": popup.var_cardinal,
        "enable_rotation": popup.var_rotation,
    })

    add_filter_sections(popup, controls_frame, current)
    popup.var_cardinal.trace_add("write", lambda *_args: popup._update_rotation_row_state())
    popup._update_rotation_row_state()

    build_realism_controls(popup, tab_realism, current)

    preview_inline = ttk.Frame(body, width=360)
    preview_inline.grid(row=0, column=1, sticky="nsew")
    preview_inline.grid_propagate(False)
    popup._build_preview_panel(preview_inline)

    btns = ttk.Frame(parent)
    btns.pack(fill="x", pady=(10, 0))

    ttk.Button(btns, text="Clean", command=popup._set_clean_filters).pack(side="left")
    ttk.Button(btns, text="Reset Defaults", command=popup._reset_defaults).pack(side="left")
    ttk.Button(btns, text="Reset to 1.0", command=popup._reset_to_one).pack(side="left", padx=(6, 0))
    ttk.Button(btns, text="Close", command=popup._handle_close).pack(side="right")


def add_filter_sections(popup: Any, parent: ttk.Frame, current: Dict[str, Any]) -> None:
    """Add all filter sections with controls."""
    popup._rotation_row_state_cb = None

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
            value=popup.float_or_default(
                str(current.get(key_strength, default_strength)),
                default=default_strength,
            )
        )

        chk_enable = ttk.Checkbutton(row_main, text=label, variable=var_enable)
        chk_enable.pack(side="left")

        scale = ttk.Scale(row_main, from_=min_v, to=max_v, orient="horizontal", variable=var_strength, length=170)
        scale.pack(side="left", padx=(10, 6))

        lbl = ttk.Label(row_main, width=5, anchor="e")
        lbl.pack(side="left")

        def update_label(*args: Any) -> None:
            v = max(min_v, min(max_v, var_strength.get()))
            lbl.configure(text=f"{v:.2f}")

        var_strength.trace_add("write", lambda *args: update_label())
        update_label()

        key_randomize = f"{key_strength}_randomize"
        key_min = f"{key_strength}_min"
        key_max = f"{key_strength}_max"
        base_val = popup.float_or_default(str(current.get(key_strength, default_strength)), default=default_strength)
        var_randomize = tk.BooleanVar(value=bool(current.get(key_randomize, False)))
        var_min = tk.DoubleVar(value=popup.float_or_default(str(current.get(key_min, base_val)), default=base_val))
        var_max = tk.DoubleVar(value=popup.float_or_default(str(current.get(key_max, base_val)), default=base_val))

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
            cardinal_only = bool(popup.var_cardinal.get()) if is_rotation_row else False
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
            popup._rotation_row_state_cb = _update_random_mode

        popup.filter_vars[key_enable] = var_enable
        popup.filter_vars[key_strength] = var_strength
        popup.filter_vars[key_randomize] = var_randomize
        popup.filter_vars[key_min] = var_min
        popup.filter_vars[key_max] = var_max

    add_section("━━━ Existing Filters ━━━")
    add_filter_row("Blur", "enable_blur", "blur_strength", True, 1.0)
    add_filter_row("Grain", "enable_grain", "grain_strength", True, 1.0, max_v=3.0)
    add_filter_row("Brightness", "enable_brightness", "brightness_strength", True, 1.0, max_v=2.0)
    add_filter_row("Contrast", "enable_contrast", "contrast_strength", True, 1.0, max_v=2.0)
    add_filter_row("Rotation", "enable_rotation", "rotation_strength", True, 1.0, max_v=2.0, enable_var=popup.var_rotation)

    add_section("━━━ High Priority ━━━")
    add_filter_row("Perspective Transform", "enable_perspective", "perspective_strength", False, 1.0, max_v=2.0)
    add_filter_row("Motion Blur", "enable_motion_blur", "motion_blur_strength", True, 1.0, max_v=3.0)
    add_filter_row("Saturation", "enable_saturation", "saturation_factor", True, 1.0, min_v=0.5, max_v=1.5)
    add_filter_row("Hue Shift", "enable_hue_shift", "hue_shift_deg", True, 0.0, min_v=-30.0, max_v=30.0)
    add_filter_row("Shadow", "enable_shadow", "shadow_strength", True, 0.3, min_v=0.0, max_v=0.8)
    add_filter_row("Reflection/Glare", "enable_reflection", "reflection_strength", False, 0.5, min_v=0.0, max_v=1.0)

    add_section("━━━ Medium Priority (disabled) ━━━")
    add_filter_row("Vignetting", "enable_vignetting", "vignetting_strength", False, 1.0, max_v=2.0)
    add_filter_row("Chromatic Aberration", "enable_chromatic_aberration", "chromatic_strength", False, 1.0, max_v=2.0)
    add_filter_row("JPEG Quality", "enable_jpeg_compression", "jpeg_quality", False, 85.0, min_v=50.0, max_v=95.0)
    add_filter_row("Color Temperature", "enable_color_temperature", "color_temperature_kelvin", False, 5500.0, min_v=2500.0, max_v=7500.0)

    add_section("━━━ Low Priority (disabled) ━━━")
    add_filter_row("Lens Distortion K1", "enable_lens_distortion", "distortion_k1", False, 0.0, min_v=-0.3, max_v=0.3)
    add_filter_row("Dust Particles", "enable_dust", "dust_density", False, 0.3, min_v=0.0, max_v=1.0)
    add_filter_row("Sharpen", "enable_sharpen", "sharpen_strength", False, 1.0, max_v=2.0)
