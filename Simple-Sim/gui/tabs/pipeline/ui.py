"""UI for the Pipeline Control Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import os

from PIL import Image, ImageTk
import cv2

from gui.utils.tooltip import ToolTip
from simple_sim.schema import MetaRow


class PipelineUI:
    def __init__(self, tab: Any, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)

        # Stats bar state
        self.var_cpu: tk.StringVar = tk.StringVar(value="CPU: -")
        self.var_gpu: tk.StringVar = tk.StringVar(value="GPU: -")
        self.var_ram: tk.StringVar = tk.StringVar(value="RAM: -")
        self.var_ds_size: tk.StringVar = tk.StringVar(value="DS: -")
        self.var_ds_samples: tk.StringVar = tk.StringVar(value="Samples: -")
        self.pb_cpu: ttk.Progressbar
        self.pb_gpu: ttk.Progressbar
        self.pb_ram: ttk.Progressbar
        self.lbl_ds_samples: ttk.Label
        
        # UI components
        self.btn_start: ttk.Button
        self.btn_start_generate: ttk.Button
        self.btn_stop: ttk.Button
        self.btn_filters: ttk.Button
        self.var_run_mode: tk.StringVar = tk.StringVar(value="single")
        self.var_run_count: tk.StringVar = tk.StringVar(value="10")
        self.run_count_entry: ttk.Entry
        self.var_dataset_mode: tk.StringVar = tk.StringVar(value="new")
        self.var_config: tk.StringVar = tk.StringVar(value="configs/run_0001.yaml")
        self.var_out: tk.StringVar = tk.StringVar(value="outputs/sim_data/runs/run_0001")
        self.var_model: tk.StringVar = tk.StringVar(value="outputs/models/run_0001.pt")
        self.var_autosnap: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_snap_every: tk.StringVar = tk.StringVar(value="10")
        self.var_snap_keep: tk.StringVar = tk.StringVar(value="5")
        self.var_wheelhouse: tk.StringVar = tk.StringVar(value="")
        self.var_eventlog: tk.StringVar = tk.StringVar(value="")
        self.var_phase: tk.StringVar = tk.StringVar(value="phase: idle")
        self.var_epoch: tk.StringVar = tk.StringVar(value="epoch: -")
        self.var_ips: tk.StringVar = tk.StringVar(value="img/s: -")
        self.var_last: tk.StringVar = tk.StringVar(value="last: -")
        self.var_dataset: tk.StringVar = tk.StringVar(value="")
        self.var_dataset_multi: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_dataset_multi_paths: tk.StringVar = tk.StringVar(value="[]")
        self.var_dataset_multi_summary: tk.StringVar = tk.StringVar(value="")
        self.dataset_combo: ttk.Combobox
        self.dataset_multi_entry: ttk.Entry
        self.btn_dataset_multi_pick: ttk.Button
        self.chk_dataset_multi: ttk.Checkbutton
        self.model_combo: ttk.Combobox
        self.img_labels: list[ttk.Label] = []
        self.txt: tk.Text
        self.var_continue_epochs: tk.StringVar = tk.StringVar(value="10")
        self.var_continue_out_mode: tk.StringVar = tk.StringVar(value="last")
        self.var_name: tk.StringVar = tk.StringVar(value="pcb_0603_resistor_v1")
        self.var_name_ts: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_model_bundle: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_profile_model: tk.StringVar = tk.StringVar(value="")
        self.var_profile_model_lock: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_profile_build_preset: tk.StringVar = tk.StringVar(value="quick")
        self.combo_profile_model: ttk.Combobox
        self.var_profile: tk.StringVar = tk.StringVar(value="chip_0603_resistor@1")
        self.profile_combo: ttk.Combobox
        self.var_render_backend: tk.StringVar = tk.StringVar(value="opencv_2d")
        self.render_combo: ttk.Combobox
        self.var_profiles_multi: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_profiles_multi_mode: tk.StringVar = tk.StringVar(value="separate")
        self.var_profiles_multi_json: tk.StringVar = tk.StringVar(value="[]")
        self.var_profiles_multi_summary: tk.StringVar = tk.StringVar(value="")
        # Existing filter variables
        self.var_filter_cardinal_rotation_90: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_enable_rotation: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_enable_blur: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_enable_grain: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_enable_brightness: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_enable_contrast: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_rotation_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_blur_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_grain_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_brightness_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_contrast_strength: tk.StringVar = tk.StringVar(value="1.0")
        # High priority new filters (enabled by default)
        self.var_filter_enable_perspective: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_perspective_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_perspective_angle_x: tk.StringVar = tk.StringVar(value="0.0")
        self.var_filter_perspective_angle_y: tk.StringVar = tk.StringVar(value="0.0")

        self.var_filter_enable_motion_blur: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_motion_blur_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_motion_blur_angle: tk.StringVar = tk.StringVar(value="0.0")

        self.var_filter_enable_saturation: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_saturation_factor: tk.StringVar = tk.StringVar(value="1.0")

        self.var_filter_enable_hue_shift: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_hue_shift_deg: tk.StringVar = tk.StringVar(value="0.0")

        self.var_filter_enable_shadow: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_shadow_strength: tk.StringVar = tk.StringVar(value="0.3")
        self.var_filter_shadow_size: tk.StringVar = tk.StringVar(value="0.2")

        self.var_filter_enable_reflection: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_reflection_strength: tk.StringVar = tk.StringVar(value="0.5")
        self.var_filter_reflection_size: tk.StringVar = tk.StringVar(value="0.15")

        # Medium priority new filters (disabled by default)
        self.var_filter_enable_vignetting: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_vignetting_strength: tk.StringVar = tk.StringVar(value="1.0")

        self.var_filter_enable_chromatic_aberration: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_chromatic_strength: tk.StringVar = tk.StringVar(value="1.0")

        self.var_filter_enable_jpeg_compression: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_jpeg_quality: tk.StringVar = tk.StringVar(value="85")

        self.var_filter_enable_color_temperature: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_color_temperature_kelvin: tk.StringVar = tk.StringVar(value="5500")

        # Low priority new filters (disabled by default)
        self.var_filter_enable_lens_distortion: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_distortion_k1: tk.StringVar = tk.StringVar(value="0.0")
        self.var_filter_distortion_k2: tk.StringVar = tk.StringVar(value="0.0")

        self.var_filter_enable_dust: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_dust_density: tk.StringVar = tk.StringVar(value="0.3")
        self.var_filter_dust_size: tk.StringVar = tk.StringVar(value="2.0")

        self.var_filter_enable_sharpen: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_sharpen_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.entry_profiles_multi: ttk.Entry
        self.btn_profiles_multi_pick: ttk.Button
        self.var_dataset_profile: tk.StringVar = tk.StringVar(value="Profile: -")
        self.var_model_profile: tk.StringVar = tk.StringVar(value="")
        self.var_profile_compat: tk.StringVar = tk.StringVar(value="")
        self.lbl_profile_compat: ttk.Label
        self.var_run_progress: tk.StringVar = tk.StringVar(value="run: -")
        
        self._thumb_refs: List[ImageTk.PhotoImage] = []

        # Precise mode
        self.btn_precise_settings: ttk.Button

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 5))

        stats = ttk.Frame(top)
        stats.pack(side="right", padx=(10, 0))
        stats.columnconfigure(1, weight=1)

        perf_frame = ttk.Frame(stats)
        perf_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        perf_frame.columnconfigure(0, weight=1)
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(2, weight=1)

        ttk.Label(perf_frame, textvariable=self.var_cpu, width=12).grid(row=0, column=0, sticky="w")
        self.pb_cpu = ttk.Progressbar(perf_frame, orient="horizontal", mode="determinate", maximum=100, length=100)
        self.pb_cpu.grid(row=1, column=0, sticky="ew")

        ttk.Label(perf_frame, textvariable=self.var_gpu, width=12).grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.pb_gpu = ttk.Progressbar(perf_frame, orient="horizontal", mode="determinate", maximum=100, length=100)
        self.pb_gpu.grid(row=1, column=1, sticky="ew", padx=(6, 0))

        ttk.Label(perf_frame, textvariable=self.var_ram, width=12).grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.pb_ram = ttk.Progressbar(perf_frame, orient="horizontal", mode="determinate", maximum=100, length=100)
        self.pb_ram.grid(row=1, column=2, sticky="ew", padx=(6, 0))

        ds_frame = ttk.Frame(stats)
        ds_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ds_frame.columnconfigure(0, weight=1)
        ds_frame.columnconfigure(1, weight=0)

        ttk.Label(ds_frame, textvariable=self.var_ds_size).grid(row=0, column=0, sticky="w")
        self.lbl_ds_samples = ttk.Label(ds_frame, textvariable=self.var_ds_samples)
        self.lbl_ds_samples.grid(row=1, column=0, sticky="w", pady=(1, 0))
        ttk.Button(ds_frame, text="↻", width=3, command=self.tab._refresh_dataset_stats).grid(
            row=0, column=1, sticky="e", padx=(6, 0)
        )

        self.btn_start = ttk.Button(top, text="▶ Start Pipeline", command=self.tab.start_pipeline, width=15)
        self.btn_start.pack(side="left")

        self.btn_start_generate = ttk.Button(top, text="▶ Generate Only", command=self.tab.start_pipeline_generate_only, width=15)
        self.btn_start_generate.pack(side="left", padx=(6, 0))
        ToolTip(self.btn_start_generate, text_func=lambda: "Generate + validate dataset only (no training/eval)")

        self.btn_stop = ttk.Button(top, text="⏹ Stop", command=self.tab.stop_pipeline, state="disabled", width=10)
        self.btn_stop.pack(side="left", padx=(8, 0))
        self.btn_filters = ttk.Button(top, text="Image Filters", command=self.tab._open_image_filters_popup, width=12)
        self.btn_filters.pack(side="left", padx=(6, 0))
        ToolTip(
            self.btn_filters,
            text_func=lambda: "Configure active image filters (90° rotation, grain, blur, brightness, contrast)",
        )

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top, text="Run Mode:").pack(side="left", padx=(0, 6))
        run_mode_frame = ttk.Frame(top)
        run_mode_frame.pack(side="left")

        ttk.Radiobutton(run_mode_frame, text="Single", variable=self.var_run_mode, value="single").pack(side="left")
        ttk.Radiobutton(run_mode_frame, text="Multiple", variable=self.var_run_mode, value="multiple").pack(side="left", padx=(5, 0))
        ttk.Radiobutton(run_mode_frame, text="Continuous", variable=self.var_run_mode, value="continuous").pack(side="left", padx=(5, 0))
        ttk.Radiobutton(run_mode_frame, text="Precise", variable=self.var_run_mode, value="precise").pack(side="left", padx=(5, 0))

        ttk.Label(top, text="Count:").pack(side="left", padx=(10, 6))
        self.run_count_entry = ttk.Entry(top, textvariable=self.var_run_count, width=5)
        self.run_count_entry.pack(side="left")

        self.btn_precise_settings = ttk.Button(
            top, text="Precise Settings...", width=16,
            command=self.tab._open_precise_popup, state="disabled",
        )
        self.btn_precise_settings.pack(side="left", padx=(8, 0))
        ToolTip(self.btn_precise_settings, text_func=lambda: "Configure exact sample counts per class / per profile")

        def on_run_mode_change(*args):
            mode = self.var_run_mode.get()
            if mode == "multiple":
                self.run_count_entry.configure(state="normal")
            else:
                self.run_count_entry.configure(state="disabled")
            self.btn_precise_settings.configure(state="normal" if mode == "precise" else "disabled")
        self.var_run_mode.trace("w", on_run_mode_change)
        on_run_mode_change()

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top, text="Dataset:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(top, text="Create New", variable=self.var_dataset_mode, value="new").pack(side="left")
        ttk.Radiobutton(top, text="Extend Existing", variable=self.var_dataset_mode, value="extend").pack(side="left", padx=(5, 0))

        top2 = ttk.Frame(self.frame)
        top2.pack(fill="x", pady=(0, 5))

        ttk.Label(top2, text="Config:").pack(side="left", padx=(0, 6))
        ttk.Entry(top2, textvariable=self.var_config, width=24).pack(side="left")

        ttk.Separator(top2, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top2, text="Out:").pack(side="left", padx=(0, 6))
        ttk.Entry(top2, textvariable=self.var_out, width=32).pack(side="left")

        top2b = ttk.Frame(self.frame)
        top2b.pack(fill="x", pady=(0, 5))

        ttk.Label(top2b, text="Model:").pack(side="left", padx=(0, 6))
        self.model_combo = ttk.Combobox(top2b, textvariable=self.var_model, state="readonly", width=36)
        self.model_combo.pack(side="left")
        ToolTip(self.model_combo, text_func=lambda: self.var_model.get())
        ttk.Button(top2b, text="↻", width=3, command=self.tab._refresh_models).pack(side="left", padx=(6, 0))
        ttk.Button(top2b, text="+", width=3, command=self.tab._add_new_model).pack(side="left", padx=(2, 0))

        ttk.Separator(top2b, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Checkbutton(top2b, text="Auto-snap", variable=self.var_autosnap).pack(side="left")
        ttk.Label(top2b, text="Every:").pack(side="left", padx=(8, 4))
        ttk.Combobox(top2b, textvariable=self.var_snap_every, values=["1", "5", "10", "30", "50"], state="readonly", width=5).pack(
            side="left"
        )
        ttk.Label(top2b, text="Keep:").pack(side="left", padx=(8, 4))
        ttk.Entry(top2b, textvariable=self.var_snap_keep, width=4).pack(side="left")

        top2a = ttk.Frame(self.frame)
        top2a.pack(fill="x", pady=(0, 5))

        ttk.Label(top2a, text="Name:").pack(side="left", padx=(0, 6))
        ttk.Entry(top2a, textvariable=self.var_name, width=20).pack(side="left")
        ttk.Checkbutton(top2a, text="Timestamp", variable=self.var_name_ts).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(top2a, text="Bundle", variable=self.var_model_bundle).pack(side="left", padx=(8, 0))
        ttk.Button(top2a, text="Apply", command=self.tab._apply_name_to_out_and_model, width=8).pack(side="left", padx=(8, 0))

        top3 = ttk.Frame(self.frame)
        top3.pack(fill="x", pady=(0, 5))

        ttk.Label(top3, text="Profile:").pack(side="left", padx=(0, 6))
        self.profile_combo = ttk.Combobox(top3, textvariable=self.var_profile, state="readonly", width=16)
        self.profile_combo.pack(side="left")
        ToolTip(self.profile_combo, text_func=lambda: self.var_profile.get())
        self.profile_combo.bind("<<ComboboxSelected>>", self.tab._on_profile_selected)
        ttk.Button(top3, text="↻", width=3, command=self.tab._refresh_profiles).pack(side="left", padx=(6, 0))
        ttk.Button(top3, text="ⓘ", width=3, command=self.tab._show_profile_info).pack(side="left", padx=(3, 0))

        ttk.Separator(top3, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top3, text="Render:").pack(side="left", padx=(0, 6))
        self.render_combo = ttk.Combobox(
            top3, textvariable=self.var_render_backend, values=["opencv_2d", "blender_3d"], state="readonly", width=10,
        )
        self.render_combo.pack(side="left")
        ToolTip(self.render_combo, text_func=lambda: self.var_render_backend.get())
        self.render_combo.bind("<<ComboboxSelected>>", self.tab._on_render_backend_changed)

        ttk.Separator(top3, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top3, text="Multi-Profile:").pack(side="left", padx=(0, 6))
        ttk.Checkbutton(top3, text="Enable", variable=self.var_profiles_multi, command=self.tab._on_profiles_multi_toggle).pack(side="left", padx=(0, 0))
        ttk.Label(top3, text="Mode:").pack(side="left", padx=(8, 4))
        ttk.Combobox(top3, textvariable=self.var_profiles_multi_mode, values=["separate", "mixed"], state="readonly", width=9).pack(side="left")
        self.btn_profiles_multi_pick = ttk.Button(top3, text="Select", width=6, command=self.tab._pick_profiles_multi, state="disabled")
        self.btn_profiles_multi_pick.pack(side="left", padx=(6, 0))
        ToolTip(self.btn_profiles_multi_pick, text_func=lambda: "Select multiple profiles to generate")
        self.entry_profiles_multi = ttk.Entry(top3, textvariable=self.var_profiles_multi_summary, width=18, state="disabled")
        self.entry_profiles_multi.pack(side="left", padx=(6, 0))

        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(10, 0))

        ttk.Label(status, textvariable=self.var_phase).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_run_progress).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_epoch).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_ips).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_last).pack(side="left", padx=(0, 14))

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(10, 0))

        left = ttk.Frame(main, padding=8)
        main.add(left, weight=1)
        ttk.Label(left, text="Datasets").pack(anchor="w")

        dsbar = ttk.Frame(left)
        dsbar.pack(fill="x", pady=(6, 0))

        ds_sel = ttk.Frame(dsbar)
        ds_sel.pack(side="left", fill="x", expand=True)

        self.dataset_combo = ttk.Combobox(ds_sel, textvariable=self.var_dataset, state="readonly", width=38)
        self.dataset_combo.pack(side="left", fill="x", expand=True)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.tab._on_dataset_selected)
        ToolTip(self.dataset_combo, text_func=self.tab._dataset_multi_tooltip_text)

        self.dataset_multi_entry = ttk.Entry(ds_sel, textvariable=self.var_dataset_multi_summary, state="readonly", width=38)
        ToolTip(self.dataset_multi_entry, text_func=self.tab._dataset_multi_tooltip_text)
        self.btn_dataset_multi_pick = ttk.Button(ds_sel, text="Select...", command=self.tab._open_dataset_multi_picker)
        self.chk_dataset_multi = ttk.Checkbutton(
            dsbar, text="Multi", variable=self.var_dataset_multi, command=self.tab._on_dataset_multi_toggle
        )
        self.chk_dataset_multi.pack(side="left", padx=(8, 0))

        ttk.Button(dsbar, text="Refresh", command=self.tab._refresh_datasets).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Snapshot", command=self.tab._snapshot_dataset_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Rename", command=self.tab._rename_dataset_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Delete", command=self.tab._delete_dataset).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Delete Images...", command=self.tab._delete_images_from_dataset).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Move Images...", command=self.tab._move_images_between_datasets).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Create New", command=self.tab._create_new_dataset).pack(side="left", padx=(8, 0))

        dsbtns = ttk.Frame(left)
        dsbtns.pack(fill="x", pady=(6, 0))
        ttk.Button(dsbtns, text="Validate", command=self.tab._validate_selected).pack(side="left")
        ttk.Button(dsbtns, text="Train", command=self.tab._train_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbtns, text="Eval", command=self.tab._eval_selected).pack(side="left", padx=(8, 0))

        ttk.Separator(dsbtns, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(dsbtns, text="+epochs:").pack(side="left")
        ttk.Entry(dsbtns, textvariable=self.var_continue_epochs, width=5).pack(side="left", padx=(6, 10))
        ttk.Label(dsbtns, text="out:").pack(side="left")
        ttk.Combobox(dsbtns, textvariable=self.var_continue_out_mode, values=["last", "best"], state="readonly", width=6).pack(
            side="left", padx=(6, 10)
        )
        ttk.Button(dsbtns, text="Continue Train", command=self.tab._continue_train_selected).pack(side="left")

        profmod = ttk.Frame(left)
        profmod.pack(fill="x", pady=(6, 0))
        ttk.Label(profmod, text="Profile model:").pack(side="left")
        self.combo_profile_model = ttk.Combobox(profmod, textvariable=self.var_profile_model, state="readonly", width=42)
        self.combo_profile_model.pack(side="left", padx=(6, 6), fill="x", expand=True)
        ToolTip(self.combo_profile_model, text_func=lambda: self.var_profile_model.get())
        ttk.Button(profmod, text="↻", width=3, command=self.tab._refresh_profile_models).pack(side="left", padx=(0, 6))

        ttk.Checkbutton(profmod, text="Lock", variable=self.var_profile_model_lock, command=self.tab._apply_profile_model_lock).pack(side="left")

        ttk.Separator(profmod, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(profmod, text="Build:").pack(side="left")
        ttk.Combobox(profmod, textvariable=self.var_profile_build_preset, values=["quick", "full"], state="readonly", width=7).pack(
            side="left", padx=(6, 6)
        )
        ttk.Button(profmod, text="Build Profile Model", command=self.tab._build_profile_model).pack(side="left")

        dsprofile = ttk.Frame(left)
        dsprofile.pack(fill="x", pady=(6, 0))
        ttk.Label(dsprofile, textvariable=self.var_dataset_profile, font=("TkDefaultFont", 9)).pack(side="left")
        ttk.Button(dsprofile, text="Profile Info", command=self.tab._show_dataset_profile_info).pack(side="left", padx=(10, 0))

        self.lbl_profile_compat = ttk.Label(dsprofile, textvariable=self.var_profile_compat, font=("TkDefaultFont", 9, "bold"), foreground="orange")
        self.lbl_profile_compat.pack(side="left", padx=(10, 0))

        ttk.Label(left, text="Latest Images (4x3 grid)").pack(anchor="w", pady=(10, 0))

        img_grid = ttk.Frame(left)
        img_grid.pack(fill="both", expand=True, pady=(6, 0))

        rows, cols = 4, 3
        for r in range(rows):
            for c in range(cols):
                frm = ttk.Frame(img_grid, relief="groove", borderwidth=1)
                frm.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                img_grid.grid_rowconfigure(r, weight=1)
                img_grid.grid_columnconfigure(c, weight=1)

                lbl = ttk.Label(frm, text="(no image)", anchor="center", compound="top")
                lbl.pack(fill="both", expand=True)
                self.img_labels.append(lbl)

        right = ttk.Frame(main, padding=8)
        main.add(right, weight=2)
        ttk.Label(right, text="Live Logs").pack(anchor="w")

        self.txt = tk.Text(right, wrap="none", height=10)
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

        yscroll = ttk.Scrollbar(right, command=self.txt.yview)
        yscroll.place(in_=self.txt, relx=1.0, rely=0, relheight=1.0, anchor="ne")
        self.txt["yscrollcommand"] = yscroll.set

    def append_log(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def set_run_buttons_state(self, running: bool) -> None:
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_start_generate.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")
        self.btn_filters.configure(state="disabled" if running else "normal")

    def update_stats_bar(self, cpu_pct: Optional[float], ram_info: Optional[Tuple[float, int, int]], gpu_pct: Optional[float]):
        if cpu_pct is not None:
            self.var_cpu.set(f"CPU: {cpu_pct:3.0f}%")
            self.pb_cpu["value"] = max(0.0, min(100.0, cpu_pct))

        if ram_info is None:
            self.var_ram.set("RAM: n/a")
            self.pb_ram["value"] = 0
        else:
            ram_pct, used_b, total_b = ram_info
            self.var_ram.set(f"RAM: {ram_pct:3.0f}% ({self.fmt_bytes(used_b)}/{self.fmt_bytes(total_b)})")
            self.pb_ram["value"] = max(0.0, min(100.0, ram_pct))

        if gpu_pct is not None:
            self.var_gpu.set(f"GPU: {gpu_pct:3.0f}%")
            self.pb_gpu["value"] = max(0.0, min(100.0, gpu_pct))

    def fmt_bytes(self, n: int) -> str:
        if n < 0: return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        v, u = float(n), 0
        while v >= 1024.0 and u < len(units) - 1: v, u = v / 1024.0, u + 1
        return f"{int(v)} {units[u]}" if u == 0 else f"{v:.1f} {units[u]}"

    def refresh_thumbnails(self, recent_imgs: List[str], dataset_dir: Optional[Path], label_dict: Dict[str, str], image_to_sample: Dict[str, str], meta_by_sample: Dict[str, MetaRow], meta_by_image_rel: Dict[str, MetaRow], draw_defect_overlay_func: Callable[..., Any]) -> None:
        self._thumb_refs.clear()
        for i, lbl in enumerate(self.img_labels):
            if i >= len(recent_imgs):
                lbl.configure(image="", text="(no image)", compound="top")
                continue

            path = Path(recent_imgs[-1 - i])
            try:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None: raise ValueError("Failed to read image")

                class_label, meta_row = "", None
                if dataset_dir:
                    try:
                        rel_path_str = str(path.relative_to(dataset_dir)).replace('\\', '/')
                        if sample_id := image_to_sample.get(rel_path_str):
                            if class_name := label_dict.get(sample_id): class_label = class_name
                            meta_row = meta_by_sample.get(sample_id)
                        if meta_row is None: meta_row = meta_by_image_rel.get(rel_path_str)
                    except ValueError: pass
                
                if meta_row is not None:
                    try: img_bgr = draw_defect_overlay_func(img_bgr, meta_row.defect, meta_row.nominal)
                    except Exception: pass

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((120, 90))
                tkimg = ImageTk.PhotoImage(img_pil)
                self._thumb_refs.append(tkimg)

                display_text = path.name if not class_label else f"{path.name}\n[{class_label}]"
                lbl.configure(image=tkimg, text=display_text, compound="top")
                lbl.image = tkimg
            except Exception:
                lbl.configure(image="", text=f"(failed) {path.name}", compound="top")

    def set_dataset_samples_info(self, text: str, color: str) -> None:
        self.var_ds_samples.set(text)
        self.lbl_ds_samples.configure(foreground=color)

    def set_profile_model_combo_values(self, values: List[str]) -> None:
        self.combo_profile_model["values"] = values

    def set_profile_model_combo_state(self, state: str) -> None:
        self.combo_profile_model.configure(state=state)

    def set_profiles_multi_state(self, enabled: bool) -> None:
        self.profile_combo.configure(state="disabled" if enabled else "readonly")
        self.btn_profiles_multi_pick.configure(state="normal" if enabled else "disabled")
        self.entry_profiles_multi.configure(state="normal" if enabled else "disabled")

    def set_profile_combo_values(self, values: List[str]) -> None:
        self.profile_combo["values"] = values

    def set_model_combo_values(self, values: List[str]) -> None:
        self.model_combo["values"] = values

    def show_messagebox(self, type: str, title: str, message: str) -> None:
        if type == "error": messagebox.showerror(title, message)
        elif type == "warning": messagebox.showwarning(title, message)
        else: messagebox.showinfo(title, message)
    
    def ask_yes_no(self, title: str, message: str) -> bool:
        return messagebox.askyesno(title, message)

    def ask_string(self, title: str, prompt: str, initialvalue: str = "") -> Optional[str]:
        return simpledialog.askstring(title, prompt, initialvalue=initialvalue, parent=self.frame)

    def open_file_dialog(self, title: str, filetypes: List[Tuple[str, str]], initialdir: str) -> Optional[str]:
        return filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initialdir)

    def open_dir_dialog(self, title: str, initialdir: str) -> Optional[str]:
        return filedialog.askdirectory(title=title, initialdir=initialdir)
    
    def set_dataset_combo_values(self, values: List[str]) -> None:
        self.dataset_combo["values"] = values

    def toggle_dataset_multi_widgets(self, enabled: bool, tooltip_text_func: Callable[[], str]) -> None:
        if enabled:
            self.dataset_combo.pack_forget()
            self.dataset_multi_entry.pack(side="left", fill="x", expand=True)
            self.btn_dataset_multi_pick.pack(side="left", padx=(8, 0))
        else:
            self.dataset_multi_entry.pack_forget()
            self.btn_dataset_multi_pick.pack_forget()
            self.dataset_combo.pack(side="left", fill="x", expand=True)
        ToolTip(self.dataset_multi_entry, text_func=tooltip_text_func)
