"""Pipeline Control Tab - Enhanced version of the original monitor."""

from __future__ import annotations
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from typing import Any, Dict, Optional
import shutil
from datetime import datetime
import re
import shlex

from PIL import Image, ImageTk
import cv2

from .base_tab import BaseTab
from gui.state import UiState
from gui.utils.tooltip import ToolTip
from gui.components.overlay_renderer import draw_defect_overlay
from gui.utils.settings_store import SettingsStore
from simple_sim.profile_hash import hash_profile
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_sim.schema import read_jsonl, LabelRow, MetaRow
from simple_sim.model_bundle import bundle_checkpoint_path


class PipelineControlTab(BaseTab):
    """Tab 1: Pipeline control with enhanced monitoring."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        # Process management
        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()

        # Queues for thread communication
        self.log_q: queue.Queue[str] = queue.Queue()
        self.event_q: queue.Queue[Dict[str, Any]] = queue.Queue()

        # Event log handling
        self.event_log_path = self.sim_root / "outputs" / "live" / "events.jsonl"
        self._event_fp = None
        self._event_pos = 0

        # Image display
        self._thumb_refs = []  # keep PhotoImage references
        self._recent_imgs: list[str] = []
        self._dataset_dirs: list[Path] = []
        self._dataset_labels: list[str] = []
        self._dataset_by_label: dict[str, Path] = {}
        self._label_dict: dict[str, str] = {}  # Map sample_id to class label
        self._image_to_sample: dict[str, str] = {}  # Map image_path to sample_id
        self._meta_by_sample: dict[str, MetaRow] = {}
        self._meta_by_image_rel: dict[str, MetaRow] = {}

        # Profile handling helpers
        self._last_profile_id: str = "chip_0603_resistor@1"
        self._profile_config_cache: Dict[str, Dict[str, Any]] = {}
        self._suspend_profile_event: bool = False

        # Stats bar state
        self.var_cpu: tk.StringVar
        self.var_gpu: tk.StringVar
        self.var_ram: tk.StringVar
        self.var_ds_size: tk.StringVar
        self.var_ds_samples: tk.StringVar
        self.pb_cpu: ttk.Progressbar
        self.pb_gpu: ttk.Progressbar
        self.pb_ram: ttk.Progressbar
        self._cpu_prev_total: Optional[int] = None
        self._cpu_prev_idle: Optional[int] = None
        self._last_stats_ts: float = 0.0
        self._last_gpu_ts: float = 0.0
        self._last_ds_size_ts: float = 0.0
        self._prev_proc_running: bool = False
        self._dataset_size_job_id: int = 0
        self._milestone_thresholds = [1000, 5000, 10000]

        # Model snapshot/versioning controls
        self.var_autosnap: tk.BooleanVar
        self.var_snap_keep: tk.StringVar
        self.var_snap_every: tk.StringVar
        self._last_snap_sig: Optional[tuple[int, int]] = None  # (mtime_ns, size)

        # UI components (will be created in build_ui)
        self.btn_start: ttk.Button
        self.btn_stop: ttk.Button
        self.var_config: tk.StringVar
        self.var_out: tk.StringVar
        self.var_model: tk.StringVar
        self.var_wheelhouse: tk.StringVar
        self.var_eventlog: tk.StringVar
        self.var_phase: tk.StringVar
        self.var_epoch: tk.StringVar
        self.var_ips: tk.StringVar
        self.var_last: tk.StringVar
        self.var_dataset: tk.StringVar
        self.var_dataset_multi: tk.BooleanVar
        self.var_dataset_multi_paths: tk.StringVar  # JSON list[str] of dataset paths (rel to sim_root when possible)
        self.var_dataset_multi_summary: tk.StringVar
        self.dataset_combo: ttk.Combobox
        self.dataset_multi_entry: ttk.Entry
        self.btn_dataset_multi_pick: ttk.Button
        self.chk_dataset_multi: ttk.Checkbutton
        self.model_combo: ttk.Combobox
        self._model_paths: list[Path] = []
        self._all_model_combo_values: list[str] = []
        self._model_profile_cache: dict[str, tuple[Optional[str], Optional[str]]] = {}
        self.img_labels: list[ttk.Label]
        self.txt: tk.Text
        self.var_continue_epochs: tk.StringVar
        self.var_continue_out_mode: tk.StringVar
        self.var_name: tk.StringVar
        self.var_name_ts: tk.BooleanVar
        self.var_model_bundle: tk.BooleanVar
        self.var_profile_model: tk.StringVar
        self.var_profile_model_lock: tk.BooleanVar
        self.var_profile_build_preset: tk.StringVar
        self.combo_profile_model: ttk.Combobox

        # Profile management
        self.var_profile: tk.StringVar
        self.profile_combo: ttk.Combobox
        self._profile_paths: list[Path] = []
        self.var_render_backend: tk.StringVar
        self.var_dataset_profile: tk.StringVar
        self.var_model_profile: tk.StringVar
        self._dataset_milestones: dict[str, int] = {}

    def build_ui(self) -> None:
        """Build the pipeline control UI."""
        # Note: Don't pack self.frame - it's managed by the notebook

        # Top controls - Row 1: Start/Stop and Pipeline Configuration
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 5))

        # Top-right: compact statistics bar (CPU/GPU/RAM + dataset size)
        stats = ttk.Frame(top)
        stats.pack(side="right", padx=(10, 0))
        stats.columnconfigure(1, weight=1)

        self.var_cpu = tk.StringVar(value="CPU: -")
        self.var_gpu = tk.StringVar(value="GPU: -")
        self.var_ram = tk.StringVar(value="RAM: -")
        self.var_ds_size = tk.StringVar(value="DS: -")
        self.var_ds_samples = tk.StringVar(value="Samples: -")

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
        ttk.Button(ds_frame, text="↻", width=3, command=self._refresh_dataset_stats).grid(
            row=0, column=1, sticky="e", padx=(6, 0)
        )

        self.btn_start = ttk.Button(top, text="▶ Start Pipeline", command=self.start_pipeline, width=15)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(top, text="⏹ Stop", command=self.stop_pipeline, state="disabled", width=10)
        self.btn_stop.pack(side="left", padx=(8, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        # Pipeline run mode
        ttk.Label(top, text="Run Mode:").pack(side="left", padx=(0, 6))
        self.var_run_mode = tk.StringVar(value="single")
        run_mode_frame = ttk.Frame(top)
        run_mode_frame.pack(side="left")

        ttk.Radiobutton(run_mode_frame, text="Single", variable=self.var_run_mode,
                       value="single").pack(side="left")
        ttk.Radiobutton(run_mode_frame, text="Multiple", variable=self.var_run_mode,
                       value="multiple").pack(side="left", padx=(5, 0))
        ttk.Radiobutton(run_mode_frame, text="Continuous", variable=self.var_run_mode,
                       value="continuous").pack(side="left", padx=(5, 0))

        ttk.Label(top, text="Count:").pack(side="left", padx=(10, 6))
        self.var_run_count = tk.StringVar(value="10")
        self.run_count_entry = ttk.Entry(top, textvariable=self.var_run_count, width=5)
        self.run_count_entry.pack(side="left")

        # Enable/disable count based on mode
        def on_run_mode_change(*args):
            if self.var_run_mode.get() == "multiple":
                self.run_count_entry.configure(state="normal")
            else:
                self.run_count_entry.configure(state="disabled")
        self.var_run_mode.trace("w", on_run_mode_change)
        on_run_mode_change()

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        # Dataset mode
        ttk.Label(top, text="Dataset:").pack(side="left", padx=(0, 6))
        self.var_dataset_mode = tk.StringVar(value="new")
        ttk.Radiobutton(top, text="Create New", variable=self.var_dataset_mode,
                       value="new").pack(side="left")
        ttk.Radiobutton(top, text="Extend Existing", variable=self.var_dataset_mode,
                       value="extend").pack(side="left", padx=(5, 0))

        # Top controls - Row 2: Config paths
        top2 = ttk.Frame(self.frame)
        top2.pack(fill="x", pady=(0, 5))

        ttk.Label(top2, text="Config:").pack(side="left", padx=(0, 6))
        self.var_config = tk.StringVar(value="configs/run_0001.yaml")
        ttk.Entry(top2, textvariable=self.var_config, width=30).pack(side="left")

        ttk.Label(top2, text="Profile:").pack(side="left", padx=(10, 6))
        self.var_profile = tk.StringVar(value="chip_0603_resistor@1")
        self.profile_combo = ttk.Combobox(top2, textvariable=self.var_profile, state="readonly", width=20)
        self.profile_combo.pack(side="left")
        ToolTip(self.profile_combo, text_func=lambda: self.var_profile.get())
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)
        ttk.Button(top2, text="↻", width=3, command=self._refresh_profiles).pack(side="left", padx=(6, 0))
        ttk.Button(top2, text="ⓘ", width=3, command=self._show_profile_info).pack(side="left", padx=(3, 0))

        ttk.Separator(top2, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top2, text="Render:").pack(side="left", padx=(0, 6))
        self.var_render_backend = tk.StringVar(value="opencv_2d")
        self.render_combo = ttk.Combobox(
            top2,
            textvariable=self.var_render_backend,
            values=["opencv_2d", "blender_3d"],
            state="readonly",
            width=12,
        )
        self.render_combo.pack(side="left")
        ToolTip(self.render_combo, text_func=lambda: self.var_render_backend.get())
        self.render_combo.bind("<<ComboboxSelected>>", self._on_render_backend_changed)

        ttk.Separator(top2, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top2, text="Out:").pack(side="left", padx=(0, 6))
        self.var_out = tk.StringVar(value="outputs/sim_data/runs/run_0001")
        ttk.Entry(top2, textvariable=self.var_out, width=35).pack(side="left")

        ttk.Label(top2, text="Model:").pack(side="left", padx=(10, 6))
        self.var_model = tk.StringVar(value="outputs/models/run_0001.pt")
        self.model_combo = ttk.Combobox(top2, textvariable=self.var_model, state="readonly", width=42)
        self.model_combo.pack(side="left")
        ToolTip(self.model_combo, text_func=lambda: self.var_model.get())
        ttk.Button(top2, text="↻", width=3, command=self._refresh_models).pack(side="left", padx=(6, 0))
        ttk.Button(top2, text="+", width=3, command=self._add_new_model).pack(side="left", padx=(2, 0))

        ttk.Separator(top2, orient="vertical").pack(side="left", fill="y", padx=10)

        self.var_autosnap = tk.BooleanVar(value=True)
        ttk.Checkbutton(top2, text="Auto-snapshot model", variable=self.var_autosnap).pack(side="left")
        ttk.Label(top2, text="Every:").pack(side="left", padx=(10, 6))
        self.var_snap_every = tk.StringVar(value="5")
        ttk.Combobox(top2, textvariable=self.var_snap_every, values=["1", "5", "10", "30", "50"], state="readonly", width=5).pack(
            side="left"
        )
        ttk.Label(top2, text="Keep:").pack(side="left", padx=(10, 6))
        self.var_snap_keep = tk.StringVar(value="30")
        ttk.Entry(top2, textvariable=self.var_snap_keep, width=5).pack(side="left")

        # Top controls - Row 3: Naming helpers (professional naming for datasets/models)
        top3 = ttk.Frame(self.frame)
        top3.pack(fill="x", pady=(0, 5))

        ttk.Label(top3, text="Name:").pack(side="left", padx=(0, 6))
        self.var_name = tk.StringVar(value="pcb_0603_resistor_v1")
        ttk.Entry(top3, textvariable=self.var_name, width=36).pack(side="left")
        self.var_name_ts = tk.BooleanVar(value=True)
        ttk.Checkbutton(top3, text="Timestamp", variable=self.var_name_ts).pack(side="left", padx=(10, 0))
        self.var_model_bundle = tk.BooleanVar(value=False)
        ttk.Checkbutton(top3, text="Multi-Model (.bundle)", variable=self.var_model_bundle).pack(side="left", padx=(10, 0))
        ttk.Button(top3, text="Apply to Out+Model", command=self._apply_name_to_out_and_model).pack(side="left", padx=(10, 0))

        # Status line
        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(10, 0))

        self.var_phase = tk.StringVar(value="phase: idle")
        self.var_run_progress = tk.StringVar(value="run: -")
        self.var_epoch = tk.StringVar(value="epoch: -")
        self.var_ips = tk.StringVar(value="img/s: -")
        self.var_last = tk.StringVar(value="last: -")

        ttk.Label(status, textvariable=self.var_phase).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_run_progress).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_epoch).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_ips).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_last).pack(side="left", padx=(0, 14))

        # Track current run iteration
        self.current_run_iteration = 0
        self.total_run_count = 0

        # Main split
        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(10, 0))

        # Left: datasets + image grid (4x3 = 12 thumbnails)
        left = ttk.Frame(main, padding=8)
        main.add(left, weight=1)
        ttk.Label(left, text="Datasets").pack(anchor="w")

        dsbar = ttk.Frame(left)
        dsbar.pack(fill="x", pady=(6, 0))

        self.var_dataset = tk.StringVar(value="")
        self.var_dataset_multi = tk.BooleanVar(value=False)
        self.var_dataset_multi_paths = tk.StringVar(value="[]")
        self.var_dataset_multi_summary = tk.StringVar(value="")

        # Keep the selection widgets in their own frame so toggling between
        # single/multi does not reorder the surrounding buttons.
        ds_sel = ttk.Frame(dsbar)
        ds_sel.pack(side="left", fill="x", expand=True)

        self.dataset_combo = ttk.Combobox(ds_sel, textvariable=self.var_dataset, state="readonly", width=38)
        self.dataset_combo.pack(side="left", fill="x", expand=True)
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)
        ToolTip(self.dataset_combo, text_func=lambda: self.var_dataset.get())

        # Multi-select: off by default (keeps current UX). When enabled we swap the combobox
        # for a readonly summary + picker dialog.
        self.dataset_multi_entry = ttk.Entry(ds_sel, textvariable=self.var_dataset_multi_summary, state="readonly", width=38)
        ToolTip(self.dataset_multi_entry, text_func=self._dataset_multi_tooltip_text)
        self.btn_dataset_multi_pick = ttk.Button(ds_sel, text="Select...", command=self._open_dataset_multi_picker)
        self.chk_dataset_multi = ttk.Checkbutton(
            dsbar, text="Multi", variable=self.var_dataset_multi, command=self._on_dataset_multi_toggle
        )
        self.chk_dataset_multi.pack(side="left", padx=(8, 0))

        ttk.Button(dsbar, text="Refresh", command=self._refresh_datasets).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Snapshot", command=self._snapshot_dataset_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Rename", command=self._rename_dataset_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Delete", command=self._delete_dataset).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Create New", command=self._create_new_dataset).pack(side="left", padx=(8, 0))

        dsbtns = ttk.Frame(left)
        dsbtns.pack(fill="x", pady=(6, 0))
        ttk.Button(dsbtns, text="Validate", command=self._validate_selected).pack(side="left")
        ttk.Button(dsbtns, text="Train", command=self._train_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbtns, text="Eval", command=self._eval_selected).pack(side="left", padx=(8, 0))

        ttk.Separator(dsbtns, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(dsbtns, text="+epochs:").pack(side="left")
        self.var_continue_epochs = tk.StringVar(value="10")
        ttk.Entry(dsbtns, textvariable=self.var_continue_epochs, width=5).pack(side="left", padx=(6, 10))
        ttk.Label(dsbtns, text="out:").pack(side="left")
        self.var_continue_out_mode = tk.StringVar(value="last")
        ttk.Combobox(dsbtns, textvariable=self.var_continue_out_mode, values=["last", "best"], state="readonly", width=6).pack(
            side="left", padx=(6, 10)
        )
        ttk.Button(dsbtns, text="Continue Train", command=self._continue_train_selected).pack(side="left")

        # Profile classifier model (optional, used by Predictions/Eval as a separate stage)
        profmod = ttk.Frame(left)
        profmod.pack(fill="x", pady=(6, 0))
        ttk.Label(profmod, text="Profile model:").pack(side="left")
        self.var_profile_model = tk.StringVar(value="")
        self.combo_profile_model = ttk.Combobox(profmod, textvariable=self.var_profile_model, state="readonly", width=42)
        self.combo_profile_model.pack(side="left", padx=(6, 6), fill="x", expand=True)
        ToolTip(self.combo_profile_model, text_func=lambda: self.var_profile_model.get())
        ttk.Button(profmod, text="↻", width=3, command=self._refresh_profile_models).pack(side="left", padx=(0, 6))

        self.var_profile_model_lock = tk.BooleanVar(value=False)
        ttk.Checkbutton(profmod, text="Lock", variable=self.var_profile_model_lock, command=self._apply_profile_model_lock).pack(side="left")

        ttk.Separator(profmod, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(profmod, text="Build:").pack(side="left")
        self.var_profile_build_preset = tk.StringVar(value="quick")
        ttk.Combobox(profmod, textvariable=self.var_profile_build_preset, values=["quick", "full"], state="readonly", width=7).pack(
            side="left", padx=(6, 6)
        )
        ttk.Button(profmod, text="Build Profile Model", command=self._build_profile_model).pack(side="left")

        # Dataset profile info
        dsprofile = ttk.Frame(left)
        dsprofile.pack(fill="x", pady=(6, 0))
        self.var_dataset_profile = tk.StringVar(value="Profile: -")
        ttk.Label(dsprofile, textvariable=self.var_dataset_profile, font=("TkDefaultFont", 9)).pack(side="left")
        ttk.Button(dsprofile, text="Profile Info", command=self._show_dataset_profile_info).pack(side="left", padx=(10, 0))

        # Profile compatibility warning
        self.var_profile_compat = tk.StringVar(value="")
        self.lbl_profile_compat = ttk.Label(dsprofile, textvariable=self.var_profile_compat, font=("TkDefaultFont", 9, "bold"), foreground="orange")
        self.lbl_profile_compat.pack(side="left", padx=(10, 0))

        ttk.Label(left, text="Latest Images (4x3 grid)").pack(anchor="w", pady=(10, 0))

        img_grid = ttk.Frame(left)
        img_grid.pack(fill="both", expand=True, pady=(6, 0))

        self.img_labels = []
        rows, cols = 4, 3  # Enhanced: 12 thumbnails instead of 9
        for r in range(rows):
            for c in range(cols):
                frm = ttk.Frame(img_grid, relief="groove", borderwidth=1)
                frm.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                img_grid.grid_rowconfigure(r, weight=1)
                img_grid.grid_columnconfigure(c, weight=1)

                # Use compound so text (filename/class tag) is visible with the thumbnail.
                lbl = ttk.Label(frm, text="(no image)", anchor="center", compound="top")
                lbl.pack(fill="both", expand=True)
                self.img_labels.append(lbl)

        # Right: logs
        right = ttk.Frame(main, padding=8)
        main.add(right, weight=2)
        ttk.Label(right, text="Live Logs").pack(anchor="w")

        self.txt = tk.Text(right, wrap="none", height=10)
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

        yscroll = ttk.Scrollbar(right, command=self.txt.yview)
        yscroll.place(in_=self.txt, relx=1.0, rely=0, relheight=1.0, anchor="ne")
        self.txt["yscrollcommand"] = yscroll.set

        # Initialize datasets and start UI ticker
        self._refresh_datasets()
        self._refresh_models()
        self._refresh_profile_models()
        self._refresh_profiles()
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self._apply_profile_model_lock()

        # Add trace to model selection to check compatibility
        self.var_model.trace("w", lambda *args: self._check_profile_compatibility())

        self._tick_ui()

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None:
            return

        def set_if(var, key: str) -> None:
            v = st.get(key)
            if v is None:
                return
            try:
                var.set(v)
            except Exception:
                pass

        set_if(self.var_run_mode, "pipeline.run_mode")
        set_if(self.var_run_count, "pipeline.run_count")
        set_if(self.var_dataset_mode, "pipeline.dataset_mode")
        set_if(self.var_config, "pipeline.config")
        set_if(self.var_out, "pipeline.out")
        set_if(self.var_model, "pipeline.model")
        set_if(self.var_autosnap, "pipeline.autosnap")
        set_if(self.var_snap_every, "pipeline.snap_every")
        set_if(self.var_snap_keep, "pipeline.snap_keep")
        set_if(self.var_continue_epochs, "pipeline.continue_epochs")
        set_if(self.var_continue_out_mode, "pipeline.continue_out_mode")
        set_if(self.var_name, "pipeline.name")
        set_if(self.var_name_ts, "pipeline.name_ts")
        set_if(self.var_dataset, "pipeline.dataset_selection")
        set_if(self.var_dataset_multi, "pipeline.dataset_multi_enabled")
        set_if(self.var_dataset_multi_paths, "pipeline.dataset_multi_paths")
        set_if(self.var_profile_model, "pipeline.profile_model")
        set_if(self.var_profile_model_lock, "pipeline.profile_model_locked")
        set_if(self.var_profile_build_preset, "pipeline.profile_build_preset")
        set_if(self.var_render_backend, "pipeline.render_backend")

        # Ensure dropdowns reflect loaded values.
        try:
            self._refresh_datasets()
            self._on_dataset_multi_toggle()
            self._on_dataset_selected()
        except Exception:
            pass
        try:
            self._refresh_models()
        except Exception:
            pass

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None:
            return

        def bind(var, key: str) -> None:
            def cb(*_a) -> None:
                try:
                    st.set(key, var.get())
                    st.schedule_save(self.frame)
                except Exception:
                    pass

            try:
                var.trace_add("write", cb)
            except Exception:
                try:
                    var.trace("w", cb)
                except Exception:
                    pass

        bind(self.var_run_mode, "pipeline.run_mode")
        bind(self.var_run_count, "pipeline.run_count")
        bind(self.var_dataset_mode, "pipeline.dataset_mode")
        bind(self.var_config, "pipeline.config")
        bind(self.var_out, "pipeline.out")
        bind(self.var_model, "pipeline.model")
        bind(self.var_autosnap, "pipeline.autosnap")
        bind(self.var_snap_every, "pipeline.snap_every")
        bind(self.var_snap_keep, "pipeline.snap_keep")
        bind(self.var_continue_epochs, "pipeline.continue_epochs")
        bind(self.var_continue_out_mode, "pipeline.continue_out_mode")
        bind(self.var_name, "pipeline.name")
        bind(self.var_name_ts, "pipeline.name_ts")
        bind(self.var_dataset, "pipeline.dataset_selection")
        bind(self.var_dataset_multi, "pipeline.dataset_multi_enabled")
        bind(self.var_dataset_multi_paths, "pipeline.dataset_multi_paths")
        bind(self.var_profile_model, "pipeline.profile_model")
        bind(self.var_profile_model_lock, "pipeline.profile_model_locked")
        bind(self.var_profile_build_preset, "pipeline.profile_build_preset")
        bind(self.var_render_backend, "pipeline.render_backend")

    def _refresh_profile_models(self) -> None:
        root = self.sim_root / "outputs" / "models"
        cand: list[Path] = []
        if root.exists():
            cand.extend(sorted(root.glob("profile_classifier_*.pt")))
            cand.extend(sorted(root.glob("**/profile_classifier_*.pt")))
        # De-dup + sort by mtime desc
        seen: dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file():
                    seen[str(p.resolve())] = p
            except Exception:
                continue
        paths = list(seen.values())
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

        values = [""]
        for p in paths:
            try:
                values.append(str(p.resolve().relative_to(self.sim_root.resolve())))
            except Exception:
                values.append(str(p))
        try:
            self.combo_profile_model["values"] = values
        except Exception:
            pass

        cur = self.var_profile_model.get().strip()
        if cur and cur in values:
            return
        if not cur:
            self.var_profile_model.set("")

    def _apply_profile_model_lock(self) -> None:
        locked = bool(self.var_profile_model_lock.get())
        if locked:
            if not self.var_profile_model.get().strip():
                # Don't allow locking an empty selection.
                try:
                    self.var_profile_model_lock.set(False)
                except Exception:
                    pass
                return
        try:
            self.combo_profile_model.configure(state="disabled" if locked else "readonly")
        except Exception:
            pass

    def _build_profile_model(self) -> None:
        """Generate a mixed-profile dataset and train a profile classifier checkpoint."""
        if self.proc is not None:
            messagebox.showwarning("Busy", "A process is already running. Stop it first.")
            return

        preset = (self.var_profile_build_preset.get().strip() or "quick").lower()
        if preset not in ("quick", "full"):
            preset = "quick"

        if preset == "full":
            cfg = self.sim_root / "configs" / "run_profile_cls.yaml"
            out_ds = self.sim_root / "outputs" / "sim_data" / "runs" / "run_profile_cls"
            out_model = self.sim_root / "outputs" / "models" / "profile_classifier_v1.pt"
        else:
            cfg = self.sim_root / "configs" / "run_profile_cls_quick.yaml"
            out_ds = self.sim_root / "outputs" / "sim_data" / "runs" / "run_profile_cls_quick"
            out_model = self.sim_root / "outputs" / "models" / "profile_classifier_quick.pt"

        if not cfg.exists():
            messagebox.showerror("Missing config", f"Config not found:\n{cfg}")
            return

        # If targets exist, create timestamped variants to avoid clobbering.
        tag = time.strftime("%Y%m%d_%H%M%S")
        if out_ds.exists():
            out_ds = out_ds.parent / f"{out_ds.name}_{tag}"
        if out_model.exists():
            out_model = out_model.with_name(f"{out_model.stem}_{tag}{out_model.suffix}")

        # Pre-select and lock (optional) so users don't accidentally switch mid-build.
        try:
            rel_model = str(out_model.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            rel_model = str(out_model)
        self.var_profile_model.set(rel_model)
        self.var_profile_model_lock.set(True)
        self._apply_profile_model_lock()

        # Build as a single shell command so it runs sequentially.
        cmd = [
            "./.venv/bin/python", "scripts/generate_profile_dataset.py",
            "--config", str(cfg.resolve().relative_to(self.sim_root.resolve()) if str(cfg).startswith(str(self.sim_root)) else str(cfg)),
            "--out", str(out_ds.resolve().relative_to(self.sim_root.resolve()) if str(out_ds).startswith(str(self.sim_root)) else str(out_ds)),
            "&&",
            "./.venv/bin/python", "scripts/train_profile.py",
            "--data", str(out_ds.resolve().relative_to(self.sim_root.resolve()) if str(out_ds).startswith(str(self.sim_root)) else str(out_ds)),
            "--out", rel_model,
        ]

        # Quote args for bash -lc.
        cmd_q = " ".join(shlex.quote(x) for x in cmd)
        self._append_log(f"\n[build profile model] preset={preset}\n")
        self._append_log(f"config: {cfg}\n")
        self._append_log(f"data:   {out_ds}\n")
        self._append_log(f"out:    {out_model}\n")
        self._append_log(f"[cmd] {cmd_q}\n\n")
        # _run_simple_cmd already wraps args into bash -lc, so pass the shell string only.
        self._run_simple_cmd([cmd_q])

    def start_pipeline(self) -> None:
        """Start the pipeline process with configured run mode."""
        if self.proc is not None:
            return

        # Get run configuration
        run_mode = self.var_run_mode.get()
        dataset_mode = self.var_dataset_mode.get()

        # Determine output directory
        if dataset_mode == "extend":
            # Use selected dataset
            selected_ds = self._selected_dataset_dir()
            if not selected_ds:
                messagebox.showerror("Error", "No dataset selected to extend.\n\nSelect a dataset or use 'Create New' mode.")
                return
            # Prevent overwriting immutable dataset snapshots.
            try:
                versions_root = (self.sim_root / "outputs" / "sim_data" / "versions").resolve()
                if str(selected_ds.resolve()).startswith(str(versions_root) + os.sep):
                    messagebox.showerror(
                        "Error",
                        "Cannot extend/overwrite a dataset snapshot under outputs/sim_data/versions.\n\n"
                        "Select a dataset under outputs/sim_data/runs instead."
                    )
                    return
            except Exception:
                pass
            if not messagebox.askyesno(
                "Extend dataset?",
                "Extend Existing will APPEND new samples into the selected dataset folder.\n\n"
                f"Dataset:\n{selected_ds}\n\nContinue?"
            ):
                return
            out_dir = str(selected_ds)
            self._append_log(f"\n=== Extending selected dataset: {out_dir} ===\n")
        else:
            # Create new dataset
            out_dir = self.var_out.get().strip()
            self._append_log(f"\n=== Creating new dataset: {out_dir} ===\n")

        # Get run count
        if run_mode == "single":
            run_count = 1
        elif run_mode == "multiple":
            try:
                run_count = int(self.var_run_count.get())
                if run_count < 1:
                    raise ValueError()
            except:
                messagebox.showerror("Error", "Invalid run count. Please enter a positive integer.")
                return
        else:  # continuous
            run_count = -1  # Infinite

        self._append_log(f"Run mode: {run_mode}" + (f" ({run_count}x)" if run_count > 0 else " (continuous)") + "\n\n")

        # Start pipeline runner in background thread
        threading.Thread(
            target=self._run_pipeline_loop,
            args=(out_dir, run_count, dataset_mode),
            daemon=True
        ).start()

    def _run_pipeline_loop(self, out_dir: str, run_count: int, dataset_mode: str) -> None:
        """Run pipeline in a loop (runs in background thread).

        Args:
            out_dir: Output directory
            run_count: Number of runs (-1 for infinite)
        """
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_evt.clear()

        self.current_run_iteration = 0
        self.total_run_count = run_count

        while run_count == -1 or self.current_run_iteration < run_count:
            if self.stop_evt.is_set():
                self._append_log(f"\n=== Pipeline stopped by user after {self.current_run_iteration} runs ===\n")
                break

            self.current_run_iteration += 1

            # Update status display
            if run_count > 0:
                self.var_run_progress.set(f"run: {self.current_run_iteration}/{run_count}")
            else:
                self.var_run_progress.set(f"run: {self.current_run_iteration} (∞)")

            self._append_log(f"\n{'='*60}\n")
            self._append_log(f"Pipeline Run {self.current_run_iteration}" + (f"/{run_count}" if run_count > 0 else " (continuous)") + "\n")
            self._append_log(f"{'='*60}\n\n")

            # Run single pipeline iteration
            success = self._run_single_pipeline(out_dir)

            if not success and run_count > 1:
                self._append_log(f"\n⚠ Run {self.current_run_iteration} failed, but continuing...\n")

            # Snapshot the produced model checkpoint for later comparisons (versioning).
            try:
                mp = self._resolve_model_path(self.var_model.get().strip())
                if mp.exists():
                    self._write_model_meta(
                        mp,
                        run_i=self.current_run_iteration,
                        dataset_dir=self.state.dataset_dir,
                        dataset_mode=dataset_mode,
                        out_dir=self._resolve_out_dir(out_dir),
                    )
                snap_every = self._snap_every_n()
                if self.var_autosnap.get() and mp.exists() and mp.is_file() and self.current_run_iteration % snap_every == 0:
                    snap = self._snapshot_model_checkpoint(mp, run_i=self.current_run_iteration)
                    if snap:
                        self._append_log(f"[model snapshot] {snap}\n")
            except Exception:
                pass

            # After each run, refresh dataset list and re-compute selected dataset size.
            def after_run_ui_update() -> None:
                self._refresh_datasets()
                if dataset_mode != "extend":
                    out_path = self._resolve_out_dir(out_dir)
                    # Auto-select the output folder if it lives in the runs directory.
                    runs_base = (self.sim_root / "outputs" / "sim_data" / "runs").resolve()
                    try:
                        out_path.resolve().relative_to(runs_base)
                    except Exception:
                        return
                    self.var_dataset.set(out_path.name)
                    self._on_dataset_selected()
                else:
                    if self.state.dataset_dir:
                        self._start_dataset_size_calc(self.state.dataset_dir)

            try:
                self.frame.after(0, after_run_ui_update)
            except Exception:
                pass

            # Small delay between runs
            if run_count != 1 and not self.stop_evt.is_set():
                time.sleep(2)

        self._append_log(f"\n{'='*60}\n")
        self._append_log(f"All pipeline runs complete! Total: {self.current_run_iteration}\n")
        self._append_log(f"{'='*60}\n")

        # Reset buttons and status
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.state.phase = "idle"
        self.var_run_progress.set("run: -")
        self.current_run_iteration = 0
        self.total_run_count = 0

    def _run_single_pipeline(self, out_dir: str) -> bool:
        """Run a single pipeline iteration.

        Args:
            out_dir: Output directory

        Returns:
            True if successful, False otherwise
        """
        eventlog = self.event_log_path
        cfg = self.var_config.get().strip()
        model_path = self.var_model.get().strip()

        env = os.environ.copy()
        env["WHEELHOUSE"] = "wheelhouse"
        env["EVENT_LOG"] = str(eventlog)
        env["CONFIG"] = cfg
        env["DATA_DIR"] = out_dir
        env["MODEL_PATH"] = model_path
        env["DATASET_MODE"] = str(self.var_dataset_mode.get()).strip()

        cmd = ["bash", "-lc", "./run_pipeline.sh"]

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.sim_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            # Start monitoring threads
            t = threading.Thread(target=self._read_process_output, daemon=True)
            t.start()

            te = threading.Thread(target=self._tail_events, daemon=True)
            te.start()

            self.state.phase = "running"

            # Wait for process to complete
            self.proc.wait()
            rc = self.proc.returncode
            self.proc = None

            return rc == 0

        except Exception as e:
            self._append_log(f"\nError running pipeline: {e}\n")
            self.proc = None
            return False

    def stop_pipeline(self) -> None:
        """Stop the pipeline process."""
        self.stop_evt.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _run_simple_cmd(self, args: list[str], extra_env: Optional[Dict[str, str]] = None) -> None:
        """Run a Simple-Sim script and stream its output into the log pane."""
        if self.proc is not None:
            messagebox.showwarning("Busy", "A process is already running. Stop it first.")
            return

        env = os.environ.copy()
        env["EVENT_LOG"] = str(self.event_log_path)
        env["SIMPLE_SIM_EVENT_LOG"] = str(self.event_log_path)
        if extra_env:
            env.update(extra_env)

        cmd = ["bash", "-lc", " ".join(args)]
        self.stop_evt.clear()

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.sim_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        t = threading.Thread(target=self._read_process_output, daemon=True)
        t.start()

        te = threading.Thread(target=self._tail_events, daemon=True)
        te.start()

    def _read_process_output(self) -> None:
        """Read process output in background thread."""
        assert self.proc is not None
        fp = self.proc.stdout
        assert fp is not None
        for line in fp:
            self.log_q.put(line)
            if self.stop_evt.is_set():
                break

        try:
            rc = self.proc.wait(timeout=1.0)
        except Exception:
            rc = None

        self.log_q.put(f"\n[process exited rc={rc}]\n")
        self.proc = None

    def _tail_events(self) -> None:
        """Tail the event log file in background thread."""
        path = self.event_log_path
        for _ in range(200):
            if self.stop_evt.is_set():
                return
            if path.exists():
                break
            time.sleep(0.05)

        pos = 0
        while not self.stop_evt.is_set():
            try:
                if not path.exists():
                    time.sleep(0.1)
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except Exception:
                            continue
                        self.event_q.put(evt)
                    pos = f.tell()
            except Exception:
                time.sleep(0.1)
            time.sleep(0.1)

    def _handle_event(self, evt: Dict[str, Any]) -> None:
        """Handle telemetry events."""
        et = evt.get("event")
        if et == "gen_start":
            self.state.phase = "generating"
            out_dir = evt.get("output_dir")
            if out_dir:
                self.state.dataset_dir = Path(out_dir)
        elif et == "gen_progress":
            self.state.phase = "generating"
            self.state.img_per_s = f"{evt.get('samp_per_s', '-')}"
            last_img = evt.get("last_image_path")
            if last_img:
                self._push_image(last_img)
        elif et == "gen_done":
            self.state.phase = "generated"
            out_dir = evt.get("output_dir")
            if out_dir:
                self.state.dataset_dir = Path(out_dir)
            last_img = evt.get("last_image_path")
            if last_img:
                self._push_image(last_img)
        elif et == "train_start":
            self.state.phase = "training"
            ds = evt.get("dataset_dir")
            if ds:
                self.state.dataset_dir = Path(ds)
        elif et == "epoch_start":
            self.state.phase = "training"
            self.state.epoch = f"{evt.get('epoch', '-')}/{evt.get('epochs', '-')}"
        elif et == "train_batch":
            self.state.phase = "training"
            ips = evt.get("img_per_s")
            if ips is not None:
                self.state.img_per_s = f"{ips:.1f}"
            lid = evt.get("last_id")
            if lid:
                self.state.last_id = str(lid)
            limg = evt.get("last_image_path")
            if limg:
                self.state.last_img = str(limg)
                self._push_image(str(limg))
        elif et == "eval_start":
            self.state.phase = "evaluating"
            ds = evt.get("dataset_dir")
            if ds:
                self.state.dataset_dir = Path(ds)
        elif et == "eval_batch":
            self.state.phase = "evaluating"
            ips = evt.get("img_per_s")
            if ips is not None:
                self.state.img_per_s = f"{ips:.1f}"
            limg = evt.get("last_image_path")
            if limg:
                self.state.last_img = str(limg)
                self._push_image(str(limg))
        elif et == "eval_done":
            self.state.phase = "done"
        elif et == "validate_start":
            self.state.phase = "validating"
        elif et == "validate_done":
            ok = evt.get("ok")
            self.state.phase = "validated_ok" if ok else "validated_fail"

        self.var_phase.set(f"phase: {self.state.phase}")
        self.var_epoch.set(f"epoch: {self.state.epoch}")
        self.var_ips.set(f"img/s: {self.state.img_per_s}")
        self.var_last.set(f"last: {Path(self.state.last_img).name if self.state.last_img != '-' else '-'}")

    def _push_image(self, rel_path: str) -> None:
        """Add image to recent images list."""
        if not self.state.dataset_dir:
            return
        img_path = self.state.dataset_dir / rel_path
        if not img_path.exists():
            return
        name = str(img_path)
        if self._recent_imgs and self._recent_imgs[-1] == name:
            return
        self._recent_imgs.append(name)
        self._recent_imgs = self._recent_imgs[-12:]  # Keep last 12
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        """Refresh thumbnail display."""
        self._thumb_refs.clear()
        for i, lbl in enumerate(self.img_labels):
            if i >= len(self._recent_imgs):
                lbl.configure(image="", text="(no image)", compound="top")
                continue

            path = Path(self._recent_imgs[-1 - i])
            try:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None:
                    raise ValueError("Failed to read image")

                # Get class label for this image
                class_label = ""
                class_name = ""
                meta_row: Optional[MetaRow] = None
                if self.state.dataset_dir:
                    # Get relative path from dataset root
                    try:
                        rel_path = path.relative_to(self.state.dataset_dir)
                        rel_path_str = str(rel_path).replace('\\', '/')  # Normalize path separators

                        # Look up sample_id from image path
                        sample_id = self._image_to_sample.get(rel_path_str)
                        if sample_id:
                            # Get class label from sample_id
                            class_name = self._label_dict.get(sample_id, "") or ""
                            if class_name:
                                class_label = class_name
                            meta_row = self._meta_by_sample.get(sample_id)
                        if meta_row is None:
                            meta_row = self._meta_by_image_rel.get(rel_path_str)
                    except ValueError:
                        pass  # Path is not relative to dataset_dir

                # Render defect overlay on the thumbnail itself.
                if meta_row is not None:
                    try:
                        img_bgr = draw_defect_overlay(img_bgr, meta_row.defect, meta_row.nominal)
                    except Exception:
                        pass

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((280, 180))
                tkimg = ImageTk.PhotoImage(img_pil)
                self._thumb_refs.append(tkimg)

                # Two-line caption keeps the class tag readable.
                display_text = path.name if not class_label else f"{path.name}\n[{class_label}]"
                lbl.configure(image=tkimg, text=display_text, compound="top")
                lbl.image = tkimg
            except Exception:
                lbl.configure(image="", text=f"(failed) {path.name}", compound="top")

    def _tick_ui(self) -> None:
        """UI update ticker."""
        # Drain queues
        try:
            while True:
                line = self.log_q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass

        try:
            while True:
                evt = self.event_q.get_nowait()
                self._handle_event(evt)
        except queue.Empty:
            pass

        # Reset buttons when process finishes
        if self.proc is None and self.btn_start["state"] == "disabled":
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")

        self._maybe_update_stats()
        self.frame.after(120, self._tick_ui)

    def _snapshot_model_checkpoint(self, model_path: Path, *, run_i: int) -> Optional[Path]:
        """Copy a checkpoint to outputs/models/versions/<stem>/ with timestamp for versioning."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return None

            try:
                st = model_path.stat()
                sig = (int(st.st_mtime_ns), int(st.st_size))
                # Avoid duplicating the same file repeatedly (e.g. if a run failed before training wrote new weights).
                if self._last_snap_sig == sig:
                    return None
                self._last_snap_sig = sig
            except Exception:
                pass

            dst_root = self.sim_root / "outputs" / "models" / "versions" / model_path.stem
            dst_root.mkdir(parents=True, exist_ok=True)
            tag = time.strftime("%Y%m%d_%H%M%S")
            base = dst_root / f"{model_path.stem}_{tag}_run{int(run_i):03d}.pt"
            dst = base
            if dst.exists():
                for i in range(1, 1000):
                    cand = dst_root / f"{model_path.stem}_{tag}_run{int(run_i):03d}_{i:03d}.pt"
                    if not cand.exists():
                        dst = cand
                        break
            shutil.copy2(model_path, dst)
            self._write_snapshot_meta(dst, src=model_path, run_i=run_i)
            self._prune_model_snapshots(dst_root)
            return dst
        except Exception:
            return None

    def _write_model_meta(
        self,
        model_path: Path,
        *,
        run_i: int,
        dataset_dir: Optional[Path],
        dataset_mode: str,
        out_dir: Path,
    ) -> None:
        """Write lightweight metadata next to the active model so the GUI can show 'how many runs' it represents."""
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_iteration": int(run_i),
            "dataset_mode": str(dataset_mode),
            "dataset_dir": str(dataset_dir) if dataset_dir else None,
            "out_dir": str(out_dir),
            "model_path": str(model_path),
        }
        meta_path = Path(str(model_path) + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _write_snapshot_meta(self, snapshot_path: Path, *, src: Path, run_i: int) -> None:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_iteration": int(run_i),
            "snapshot_path": str(snapshot_path),
            "source_model_path": str(src),
        }
        meta_path = Path(str(snapshot_path) + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _snap_every_n(self) -> int:
        """Return the snapshot frequency (every N runs)."""
        try:
            s = self.var_snap_every.get().strip()
            n = int(s) if s else 1
        except Exception:
            n = 5
        return max(1, n)

    def _prune_model_snapshots(self, dst_root: Path) -> None:
        """Keep only the newest N snapshots for the model."""
        try:
            keep_s = self.var_snap_keep.get().strip()
            keep = int(keep_s) if keep_s else 0
        except Exception:
            keep = 30
        keep = max(0, keep)
        if keep == 0:
            return

        try:
            snaps = [p for p in dst_root.glob("*.pt") if p.is_file()]
            snaps.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
            for p in snaps[keep:]:
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_update_stats(self) -> None:
        """Update CPU/GPU/RAM stats at a lower frequency than the UI tick."""
        now = time.time()
        if now - self._last_stats_ts < 1.0:
            return
        self._last_stats_ts = now

        cpu_pct = self._read_cpu_percent()
        if cpu_pct is not None:
            self.var_cpu.set(f"CPU: {cpu_pct:3.0f}%")
            self.pb_cpu["value"] = max(0.0, min(100.0, cpu_pct))

        ram = self._read_ram_percent()
        if ram is None:
            self.var_ram.set("RAM: n/a")
            self.pb_ram["value"] = 0
        else:
            ram_pct, used_b, total_b = ram
            self.var_ram.set(f"RAM: {ram_pct:3.0f}% ({self._fmt_bytes(used_b)}/{self._fmt_bytes(total_b)})")
            self.pb_ram["value"] = max(0.0, min(100.0, ram_pct))

        # GPU is optional; update less frequently to avoid UI hiccups.
        if now - self._last_gpu_ts >= 2.0:
            self._last_gpu_ts = now
            gpu_pct = self._read_gpu_percent()
            if gpu_pct is not None:
                self.var_gpu.set(f"GPU: {gpu_pct:3.0f}%")
                self.pb_gpu["value"] = max(0.0, min(100.0, gpu_pct))

        # Dataset size: refresh periodically while running, and once when the proc ends.
        proc_running = self.proc is not None
        if self._prev_proc_running and not proc_running:
            # Just ended.
            if self.state.dataset_dir:
                self._start_dataset_size_calc(self.state.dataset_dir)
            self._last_ds_size_ts = now
        self._prev_proc_running = proc_running

        if proc_running and (now - self._last_ds_size_ts) >= 5.0:
            if self.state.dataset_dir:
                self._start_dataset_size_calc(self.state.dataset_dir)
            self._last_ds_size_ts = now

    def _resolve_out_dir(self, out_dir: str) -> Path:
        p = Path(out_dir)
        if p.is_absolute():
            return p
        return (self.sim_root / p).resolve()

    def _resolve_model_path(self, model_path: str) -> Path:
        p = Path(model_path)
        if p.is_absolute():
            return p
        return (self.sim_root / p).resolve()

    def _read_cpu_percent(self) -> Optional[float]:
        """Linux /proc-based CPU utilization percentage."""
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline().strip()
            parts = line.split()
            if not parts or parts[0] != "cpu" or len(parts) < 5:
                return None
            nums = [int(x) for x in parts[1:]]
            total = sum(nums)
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait

            if self._cpu_prev_total is None or self._cpu_prev_idle is None:
                self._cpu_prev_total = total
                self._cpu_prev_idle = idle
                return None

            dt = total - self._cpu_prev_total
            di = idle - self._cpu_prev_idle
            self._cpu_prev_total = total
            self._cpu_prev_idle = idle
            if dt <= 0:
                return None
            return 100.0 * (1.0 - (di / dt))
        except Exception:
            return None

    def _read_ram_percent(self) -> Optional[tuple[float, int, int]]:
        """Linux /proc-based RAM usage: percent, used bytes, total bytes."""
        try:
            total_kb = None
            avail_kb = None
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                    if total_kb is not None and avail_kb is not None:
                        break
            if total_kb is None or avail_kb is None or total_kb <= 0:
                return None
            used_kb = total_kb - avail_kb
            pct = 100.0 * (used_kb / total_kb)
            return pct, used_kb * 1024, total_kb * 1024
        except Exception:
            return None

    def _read_gpu_percent(self) -> Optional[float]:
        """Try to read NVIDIA GPU utilization via nvidia-smi, return None if unavailable."""
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if not out:
                return None
            # In multi-GPU setups, pick the first line.
            first = out.splitlines()[0].strip()
            return float(first)
        except Exception:
            return None

    def _fmt_bytes(self, n: int) -> str:
        if n < 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        v = float(n)
        u = 0
        while v >= 1024.0 and u < len(units) - 1:
            v /= 1024.0
            u += 1
        if u == 0:
            return f"{int(v)} {units[u]}"
        return f"{v:.1f} {units[u]}"

    def _compute_dir_size_bytes(self, root: Path) -> int:
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    try:
                        fp = os.path.join(dirpath, fn)
                        st = os.stat(fp, follow_symlinks=False)
                        total += int(st.st_size)
                    except Exception:
                        pass
        except Exception:
            pass
        return total

    def _start_dataset_size_calc(self, ds: Path) -> None:
        """Compute dataset folder size in background and update the stats bar."""
        self._dataset_size_job_id += 1
        job_id = self._dataset_size_job_id
        self.var_ds_size.set("DS: calculating...")

        def worker() -> None:
            size_b = self._compute_dir_size_bytes(ds)

            def apply() -> None:
                if job_id != self._dataset_size_job_id:
                    return
                self.var_ds_size.set(f"DS: {self._fmt_bytes(size_b)} ({ds.name})")

            try:
                self.frame.after(0, apply)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_dataset_stats(self) -> None:
        """Re-read manifest stats and re-trigger dataset size calculation."""
        ds = self.state.dataset_dir
        if not ds:
            self.var_ds_samples.set("Samples: -")
            self.var_ds_size.set("DS: -")
            return

        manifest_path = ds / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                stats = manifest.get("dataset_stats", {})
                splits = stats.get("splits", {})
                self._set_dataset_samples_from_manifest(
                    ds,
                    stats.get("total_samples"),
                    splits.get("train"),
                    splits.get("val"),
                    splits.get("test"),
                )
            except Exception as exc:
                print(f"Failed to refresh dataset stats: {exc}")
                self._set_dataset_samples_info("Samples: manifest error", "")
        else:
            self._set_dataset_samples_info("Samples: manifest missing", "")

        self._start_dataset_size_calc(ds)

    def _apply_dataset_settings(self, ds: Path) -> None:
        """Update config/profile/out/model fields when selecting a dataset."""
        if not ds:
            return
        config_path = ds / "config.yaml"
        if config_path.exists():
            self.var_config.set(str(config_path))

        profile_id = self._dataset_profile_id(ds)
        if profile_id:
            self._set_profile_value(profile_id)

        self.var_name.set(ds.name)
        self.var_out.set(str(ds.resolve()))
        model_path = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        try:
            self.var_model.set(str(model_path.resolve()))
        except Exception:
            self.var_model.set(str(model_path))
        self.var_dataset_mode.set("extend")
        self._update_model_dropdown_for_dataset()

    def _set_profile_value(self, profile_id: str) -> None:
        """Set the profile combobox without triggering prompts."""
        self._suspend_profile_event = True
        self.var_profile.set(profile_id)
        self._last_profile_id = profile_id

    def _dataset_profile_hash(self, ds: Optional[Path]) -> Optional[str]:
        if not ds:
            return None
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return manifest.get("component_profile", {}).get("profile_hash")
        except Exception:
            return None

    def _load_model_profile(self, model_path: Path) -> tuple[Optional[str], Optional[str]]:
        # Multi-model bundles are directories; treat them as compatible with any dataset profile.
        try:
            if model_path.is_dir():
                return "multi", None
        except Exception:
            pass
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            comp = checkpoint.get("component_profile")
            if isinstance(comp, dict):
                return comp.get("profile_id"), comp.get("profile_hash")
        except Exception:
            pass
        return None, None

    def _update_model_dropdown_for_dataset(self) -> None:
        """Show only models matching the selected dataset profile (plus multi-model bundles).

        Filtering priority:
        1. Exact profile_id + profile_hash match
        2. Same profile_id (hash mismatch / different version)
        3. Multi-model bundles (compatible with any profile)
        4. Legacy models (no profile metadata) shown with warning marker
        Models with a *different* profile_id are hidden entirely.
        """
        ds = self.state.dataset_dir
        values = list(self._all_model_combo_values)
        if ds:
            profile_id = self._dataset_profile_id(ds)
            profile_hash = self._dataset_profile_hash(ds)
            # If the dataset has no profile metadata, do not filter (legacy).
            if profile_id:
                exact: list[str] = []
                id_only: list[str] = []
                multi: list[str] = []
                legacy: list[str] = []

                for p in self._model_paths:
                    rel_path = str(p.resolve())
                    pid, phash = self._model_profile_cache.get(rel_path, (None, None))

                    # Multi-model bundle (directory) or explicitly tagged profile_id.
                    if pid == "multi" or (isinstance(pid, str) and pid.lower().startswith("multi")):
                        try:
                            multi.append(str(p.resolve().relative_to(self.sim_root.resolve())))
                        except Exception:
                            multi.append(str(p))
                        continue

                    # Legacy model (no profile metadata) - show with marker.
                    if pid is None:
                        try:
                            disp = str(p.resolve().relative_to(self.sim_root.resolve()))
                        except Exception:
                            disp = str(p)
                        legacy.append(disp)
                        continue

                    # Different profile - hide entirely.
                    if pid != profile_id:
                        continue

                    # Prefer exact hash matches, but allow same ID with hash mismatch.
                    try:
                        disp = str(p.resolve().relative_to(self.sim_root.resolve()))
                    except Exception:
                        disp = str(p)
                    if profile_hash and phash and phash == profile_hash:
                        exact.append(disp)
                    else:
                        id_only.append(disp)

                # Order: exact matches, same-ID matches, multi bundles, legacy at end.
                values = exact + id_only + multi + legacy
        try:
            self.model_combo["values"] = values
        except Exception:
            return
        if values:
            current = self.var_model.get().strip()
            if current not in values:
                self.var_model.set(values[0])
        else:
            self.var_model.set("")

    def _ensure_dataset_skeleton(self, profile_id: str, config_path: Path, out_dir: Path) -> bool:
        """Create dataset folder/manifest so it appears in the UI before generation."""
        if not config_path.exists():
            return False

        out_dir = out_dir.resolve()
        manifest_path = out_dir / "dataset_manifest.json"
        if manifest_path.exists():
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / "splits").mkdir(exist_ok=True)

        for split in ["train", "val", "test"]:
            (out_dir / "splits" / f"{split}.txt").write_text("", encoding="utf-8")

        meta_path = out_dir / "meta.jsonl"
        labels_path = out_dir / "labels.jsonl"
        meta_path.write_text("", encoding="utf-8")
        labels_path.write_text("", encoding="utf-8")

        # Copy config into dataset directory for future reference
        try:
            shutil.copy2(config_path, out_dir / "config.yaml")
        except Exception:
            pass

        # Prepare manifest data
        try:
            cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}
        run_block = cfg_data.get("run") or {}
        run_id = str(run_block.get("run_id") or out_dir.name)
        classes = cfg_data.get("classes", {})

        profiles_dir = self.sim_root / "configs" / "profiles"
        profile_path = profiles_dir / f"{profile_id}.yaml"
        profile_hash = hash_profile(profile_path) if profile_path.exists() else ""

        manifest = {
            "manifest_version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "component_profile": {
                "profile_id": profile_id,
                "profile_hash": profile_hash,
                "profile_path": str(profile_path) if profile_path.exists() else "",
            },
            "generator": {
                "version": "1.0.1",
                "git_commit": None,
                "script": "gui.profile_placeholder",
            },
            "dataset_stats": {
                "total_samples": 0,
                "splits": {"train": 0, "val": 0, "test": 0},
                "classes": {str(k): 0 for k in classes.keys()},
            },
            "extend_history": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._append_log(f"[profile] created placeholder dataset manifest for {out_dir.name}\n")
        return True

    def _set_dataset_samples_from_manifest(
        self,
        ds: Path,
        total: Optional[int],
        train: Optional[int],
        val: Optional[int],
        test: Optional[int],
    ) -> None:
        if total is None:
            self._set_dataset_samples_info("Samples: manifest missing", "")
            return
        parts = [f"Samples: {total}"]
        if train is not None and val is not None and test is not None:
            parts.append(f"(t{train}/v{val}/s{test})")
        desc = " ".join(parts)
        color = self._sample_color_from_total(total)
        self._set_dataset_samples_info(desc, color)
        self._check_milestone(ds, total)

    def _set_dataset_samples_info(self, text: str, color: str) -> None:
        self.var_ds_samples.set(text)
        if color:
            self.lbl_ds_samples.configure(foreground=color)
        else:
            self.lbl_ds_samples.configure(foreground="black")

    def _sample_color_from_total(self, total: int) -> str:
        if total >= 10000:
            return "green"
        if total >= 5000:
            return "blue"
        if total >= 1000:
            return "orange"
        return "black"

    def _check_milestone(self, ds: Path, total: int) -> None:
        name = ds.name
        current = self._dataset_milestones.get(name, 0)
        for threshold in self._milestone_thresholds:
            if total >= threshold and threshold > current:
                self._dataset_milestones[name] = threshold
                self._log_dataset_milestone(ds, threshold, total)
                break

    def _log_dataset_milestone(self, ds: Path, threshold: int, total: int) -> None:
        event = {
            "ts": time.time(),
            "event": "dataset_milestone",
            "dataset": ds.name,
            "threshold": threshold,
            "samples": total,
        }
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Failed to log milestone: {e}")

    def _append_log(self, s: str) -> None:
        """Append text to log widget."""
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _refresh_datasets(self) -> None:
        """Refresh dataset dropdown list."""
        sim_data = self.sim_root / "outputs" / "sim_data"
        runs = sim_data / "runs"
        versions = sim_data / "versions"
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)

        cand: list[Path] = []
        cand.extend([p for p in runs.iterdir() if p.is_dir()])
        for p in versions.glob("*/*"):
            if p.is_dir():
                cand.append(p)

        def is_version(p: Path) -> bool:
            try:
                return str(p.resolve()).startswith(str(versions.resolve()) + os.sep)
            except Exception:
                return False

        # Sort runs by name, snapshots by mtime desc.
        def sort_key(p: Path) -> tuple:
            if is_version(p):
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    mt = 0.0
                return (1, -mt, p.name)
            return (0, p.name)

        cand.sort(key=sort_key)

        self._dataset_dirs = cand
        self._dataset_labels = []
        self._dataset_by_label = {}

        seen: dict[str, int] = {}
        for p in cand:
            base_label = self._display_for_dataset(p, runs=runs, versions=versions)
            n = seen.get(base_label, 0) + 1
            seen[base_label] = n
            label = base_label if n == 1 else f"{base_label} ({n})"
            self._dataset_labels.append(label)
            self._dataset_by_label[label] = p

        self.dataset_combo["values"] = self._dataset_labels
        self._sync_multi_selection_after_dataset_refresh()

        # Keep current selection if it still maps.
        cur = self.var_dataset.get().strip()
        if cur and cur in self._dataset_by_label:
            return

        # Prefer state.dataset_dir.
        if self.state.dataset_dir:
            want = self._display_for_dataset(self.state.dataset_dir, runs=runs, versions=versions)
            for label, p in self._dataset_by_label.items():
                if p == self.state.dataset_dir:
                    self.var_dataset.set(label)
                    return
                if label == want:
                    self.var_dataset.set(label)
                    return

        if self._dataset_labels:
            self.var_dataset.set(self._dataset_labels[0])
            self._on_dataset_selected()

    def _dataset_multi_tooltip_text(self) -> str:
        labels = self._selected_dataset_labels()
        if not labels:
            return "(no datasets selected)"
        if len(labels) <= 12:
            return "\n".join(labels)
        head = labels[:12]
        return "\n".join(head) + f"\n... (+{len(labels) - len(head)} more)"

    def _on_dataset_multi_toggle(self) -> None:
        """Switch between single dataset combobox and multi-selection picker."""
        enabled = bool(self.var_dataset_multi.get()) if hasattr(self, "var_dataset_multi") else False

        # Seed multi-selection from current single selection when turning on.
        if enabled:
            paths = self._decode_dataset_paths_json(self.var_dataset_multi_paths.get())
            if not paths:
                ds = self._selected_dataset_dir_single()
                if ds is not None:
                    self.var_dataset_multi_paths.set(self._encode_dataset_paths_json([ds]))

        # Swap widgets.
        try:
            if enabled:
                try:
                    self.dataset_combo.pack_forget()
                except Exception:
                    pass
                self.dataset_multi_entry.pack(side="left", fill="x", expand=True)
                self.btn_dataset_multi_pick.pack(side="left", padx=(8, 0))
            else:
                try:
                    self.dataset_multi_entry.pack_forget()
                    self.btn_dataset_multi_pick.pack_forget()
                except Exception:
                    pass
                self.dataset_combo.pack(side="left", fill="x", expand=True)
        except Exception:
            pass

        self._sync_multi_summary()

        # Keep primary dataset selection consistent for previews/stats.
        if enabled:
            ds0 = self._selected_dataset_dir()
            if ds0 is not None:
                label0 = self._dataset_label_for_path(ds0)
                if label0:
                    try:
                        self.var_dataset.set(label0)
                    except Exception:
                        pass
        try:
            self._on_dataset_selected()
        except Exception:
            pass

    def _open_dataset_multi_picker(self) -> None:
        if not self._dataset_labels:
            messagebox.showinfo("Datasets", "No datasets found. Click Refresh after generating datasets.")
            return

        top = tk.Toplevel(self.frame.winfo_toplevel())
        top.title("Select datasets")
        top.transient(self.frame.winfo_toplevel())
        top.grab_set()

        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select one or more datasets (Ctrl/Shift for multi-select):").pack(anchor="w")

        lb = tk.Listbox(frm, selectmode="extended", height=min(18, max(6, len(self._dataset_labels))))
        lb.pack(fill="both", expand=True, pady=(6, 8))
        for s in self._dataset_labels:
            lb.insert("end", s)

        selected_paths = set()
        for p in self._decode_dataset_paths_json(self.var_dataset_multi_paths.get()):
            try:
                selected_paths.add(str(p.resolve()))
            except Exception:
                selected_paths.add(str(p))

        # Pre-select current multi selection (or fall back to current single selection).
        if not selected_paths:
            ds = self._selected_dataset_dir_single()
            if ds is not None:
                try:
                    selected_paths.add(str(ds.resolve()))
                except Exception:
                    selected_paths.add(str(ds))

        for i, label in enumerate(self._dataset_labels):
            p = self._dataset_by_label.get(label)
            if p is None:
                continue
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in selected_paths:
                lb.selection_set(i)

        btns = ttk.Frame(frm)
        btns.pack(fill="x")

        def clear_sel() -> None:
            lb.selection_clear(0, "end")

        def select_all() -> None:
            lb.selection_set(0, "end")

        def on_ok() -> None:
            idxs = list(lb.curselection())
            paths: list[Path] = []
            for i in idxs:
                try:
                    label = self._dataset_labels[int(i)]
                except Exception:
                    continue
                p = self._dataset_by_label.get(label)
                if p is not None:
                    paths.append(p)

            if not paths:
                ds = self._selected_dataset_dir_single()
                if ds is not None:
                    paths = [ds]

            self.var_dataset_multi_paths.set(self._encode_dataset_paths_json(paths))
            self._sync_multi_summary()

            if paths:
                label0 = self._dataset_label_for_path(paths[0])
                if label0:
                    self.var_dataset.set(label0)
            try:
                self._on_dataset_selected()
            except Exception:
                pass

            try:
                top.grab_release()
            except Exception:
                pass
            top.destroy()

        def on_cancel() -> None:
            try:
                top.grab_release()
            except Exception:
                pass
            top.destroy()

        ttk.Button(btns, text="Select all", command=select_all).pack(side="left")
        ttk.Button(btns, text="Clear", command=clear_sel).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btns, text="OK", command=on_ok).pack(side="right", padx=(0, 8))

        lb.bind("<Double-Button-1>", lambda _e: on_ok())

        try:
            top.minsize(640, 320)
        except Exception:
            pass

    def _decode_dataset_paths_json(self, s: str) -> list[Path]:
        try:
            arr = json.loads(s or "[]")
        except Exception:
            return []
        if not isinstance(arr, list):
            return []
        out: list[Path] = []
        for it in arr:
            if not isinstance(it, str):
                continue
            p = Path(it)
            if not p.is_absolute():
                p = (self.sim_root / p).resolve()
            out.append(p)
        # Dedup while preserving order.
        seen: set[str] = set()
        uniq: list[Path] = []
        for p in out:
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq

    def _encode_dataset_paths_json(self, paths: list[Path]) -> str:
        vals: list[str] = []
        for p in paths:
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            try:
                vals.append(str(rp.relative_to(self.sim_root.resolve())))
            except Exception:
                vals.append(str(rp))
        return json.dumps(vals)

    def _dataset_label_for_path(self, p: Path) -> Optional[str]:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        for label, pp in self._dataset_by_label.items():
            try:
                if pp.resolve() == rp:
                    return label
            except Exception:
                if pp == p:
                    return label
        for label, pp in self._dataset_by_label.items():
            if pp.name == p.name:
                return label
        return None

    def _selected_dataset_labels(self) -> list[str]:
        labels: list[str] = []
        for ds in self._selected_dataset_dirs():
            label = self._dataset_label_for_path(ds)
            if label:
                labels.append(label)
        return labels

    def _sync_multi_summary(self) -> None:
        labels = self._selected_dataset_labels()
        if not labels:
            self.var_dataset_multi_summary.set("(none)")
            return
        if len(labels) == 1:
            self.var_dataset_multi_summary.set(labels[0])
            return
        self.var_dataset_multi_summary.set(f"{labels[0]} (+{len(labels) - 1})")

    def _sync_multi_selection_after_dataset_refresh(self) -> None:
        # Drop any stored paths that no longer exist / are no longer in dropdown candidates.
        if not hasattr(self, "var_dataset_multi_paths"):
            return
        paths = self._decode_dataset_paths_json(self.var_dataset_multi_paths.get())
        if not paths:
            self._sync_multi_summary()
            return

        valid: set[str] = set()
        for p in self._dataset_dirs:
            try:
                valid.add(str(p.resolve()))
            except Exception:
                valid.add(str(p))

        filtered: list[Path] = []
        for p in paths:
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in valid:
                filtered.append(p)

        if filtered != paths:
            try:
                self.var_dataset_multi_paths.set(self._encode_dataset_paths_json(filtered))
            except Exception:
                pass

        if hasattr(self, "var_dataset_multi") and bool(self.var_dataset_multi.get()) and filtered:
            label0 = self._dataset_label_for_path(filtered[0])
            if label0:
                try:
                    self.var_dataset.set(label0)
                except Exception:
                    pass

        self._sync_multi_summary()

    def _display_for_dataset(self, p: Path, *, runs: Path, versions: Path) -> str:
        """Compact label for datasets in dropdown."""
        try:
            rp = p.resolve()
            if str(rp).startswith(str(runs.resolve()) + os.sep):
                return rp.name
            if str(rp).startswith(str(versions.resolve()) + os.sep):
                return f"{rp.parent.name}:{rp.name}"
        except Exception:
            pass
        return p.name

    def _refresh_models(self) -> None:
        """Refresh model dropdown list (includes snapshots)."""
        root = self.sim_root / "outputs" / "models"
        root.mkdir(parents=True, exist_ok=True)

        cand: list[Path] = []
        cand.extend(sorted(root.glob("*.pt")))
        cand.extend([p for p in sorted(root.glob("*.bundle")) if p.is_dir()])
        cand.extend([p for p in sorted((root / "bundles").glob("*")) if p.is_dir()])
        cand.extend(sorted((root / "imports").glob("*.pt")))
        cand.extend(sorted((root / "versions").glob("**/*.pt")))

        uniq: dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file() or p.is_dir():
                    uniq[str(p.resolve())] = p
            except Exception:
                continue
        paths = list(uniq.values())
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        self._model_paths = paths

        self._model_profile_cache.clear()
        for p in self._model_paths:
            rel = str(p.resolve())
            self._model_profile_cache[rel] = self._load_model_profile(p)

        values: list[str] = []
        for p in paths:
            try:
                values.append(str(p.resolve().relative_to(self.sim_root.resolve())))
            except Exception:
                values.append(str(p))

        self._all_model_combo_values = values
        self._update_model_dropdown_for_dataset()

    def _add_new_model(self) -> None:
        """Create a new model entry so training can write to it."""
        ds = self._selected_dataset_dir()
        default_name = ds.name if ds else ""
        name = simpledialog.askstring(
            "New Model",
            "Model name (without .pt):",
            initialvalue=default_name,
            parent=self.frame,
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return

        model_path = self.sim_root / "outputs" / "models" / f"{name}.pt"
        rel = str(model_path.resolve().relative_to(self.sim_root.resolve()))

        # Add to dropdown values if not already present
        if rel not in self._all_model_combo_values:
            self._all_model_combo_values.insert(0, rel)
        current_values = list(self.model_combo["values"])
        if rel not in current_values:
            current_values.insert(0, rel)
            self.model_combo["values"] = current_values

        self.var_model.set(rel)
        self._append_log(f"[model] New model target: {rel}\n")

    def _refresh_profiles(self) -> None:
        """Refresh profile dropdown list."""
        profiles_dir = self.sim_root / "configs" / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        # Find all profile YAML files
        profile_files = sorted(profiles_dir.glob("*.yaml"))
        self._profile_paths = profile_files

        want_backend = (getattr(self, "var_render_backend", None).get() if hasattr(self, "var_render_backend") else "opencv_2d") or "opencv_2d"

        # Extract profile IDs (filename without .yaml), filtered by supported_render_backends if present.
        values: list[str] = []
        for p in profile_files:
            profile_id = p.stem
            try:
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                supported = (((data or {}).get("profile") or {}).get("supported_render_backends") or [])
                if not supported:
                    supported = ["opencv_2d"]
                if want_backend not in supported:
                    continue
            except Exception:
                # If parsing fails, keep it visible in 2D mode only.
                if want_backend != "opencv_2d":
                    continue
            values.append(profile_id)

        try:
            self.profile_combo["values"] = values
        except Exception:
            return

        # Set default if not already set
        cur = self.var_profile.get().strip()
        if not cur or cur not in values:
            if "chip_0603_resistor@1" in values:
                self.var_profile.set("chip_0603_resistor@1")
            elif values:
                self.var_profile.set(values[0])
        self._on_profile_selected()

    def _on_render_backend_changed(self, _event: Optional[object] = None) -> None:
        """Render backend selection affects which profiles are shown."""
        try:
            self._refresh_profiles()
            pid = self.var_profile.get().strip()
            if pid:
                self._autoselect_config_for_profile(pid, prefer_quiet=True)
        except Exception:
            pass

    def _on_profile_selected(self, _event: Optional[object] = None) -> None:
        if self._suspend_profile_event:
            self._suspend_profile_event = False
            return

        profile_id = self.var_profile.get().strip()
        if not profile_id:
            return

        # Keep config selection aligned with Profile + Render mode without prompting.
        try:
            self._autoselect_config_for_profile(profile_id, prefer_quiet=True)
        except Exception:
            pass

        ds = self._selected_dataset_dir()
        current_profile = self._dataset_profile_id(ds)
        if current_profile == profile_id:
            self._last_profile_id = profile_id
            return

        candidate = self._find_dataset_for_profile(profile_id)
        if candidate:
            if messagebox.askyesno(
                "Switch dataset",
                f"A dataset for profile '{profile_id}' already exists:\n{candidate}\n\n"
                "Switch to it so profiles stay separated?",
            ):
                self._select_dataset(candidate)
                self._last_profile_id = profile_id
                return
            self._revert_profile_selection()
            return

        cfg_info = self._config_for_profile(profile_id, want_backend=(self.var_render_backend.get().strip() if hasattr(self, "var_render_backend") else None))
        if messagebox.askyesno(
            "Create dataset",
            f"No dataset currently matches profile '{profile_id}'.\n"
            "Would you like to prepare a new out path for this profile?",
        ):
            self._prepare_dataset_for_profile(profile_id, cfg_info)
            self._last_profile_id = profile_id
            return

        self._revert_profile_selection()

    def _dataset_profile_id(self, ds: Optional[Path]) -> Optional[str]:
        if not ds:
            return None
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return manifest.get("component_profile", {}).get("profile_id")
        except Exception:
            return None

    def _find_dataset_for_profile(self, profile_id: str) -> Optional[Path]:
        runs_root = self.sim_root / "outputs" / "sim_data" / "runs"
        if not runs_root.exists():
            return None
        for ds in sorted(runs_root.iterdir()):
            if not ds.is_dir():
                continue
            if self._dataset_profile_id(ds) == profile_id:
                return ds
        return None

    def _select_dataset(self, ds: Path) -> None:
        self._refresh_datasets()
        runs = self.sim_root / "outputs" / "sim_data" / "runs"
        versions = self.sim_root / "outputs" / "sim_data" / "versions"
        label = self._display_for_dataset(ds, runs=runs, versions=versions)
        for key, value in self._dataset_by_label.items():
            if value == ds or key == label:
                self.var_dataset.set(key)
                self._on_dataset_selected()
                return
        self.state.dataset_dir = ds
        self.var_dataset.set(label)
        self._on_dataset_selected()

    def _revert_profile_selection(self) -> None:
        self._suspend_profile_event = True
        self.var_profile.set(self._last_profile_id or "chip_0603_resistor@1")

    def _prepare_dataset_for_profile(self, profile_id: str, cfg_info: Optional[Dict[str, Any]]) -> None:
        cfg_path: Optional[Path] = None
        if cfg_info:
            cfg_path = cfg_info["path"]
            self.var_config.set(str(cfg_path))
            suggested_run = cfg_info.get("run_id")
        else:
            suggested_run = self._slugify_name(profile_id)
        self.var_dataset_mode.set("new")
        if suggested_run:
            suggested_out = self.sim_root / "outputs" / "sim_data" / "runs" / suggested_run
            self.var_out.set(str(suggested_out))
            self.var_name.set(suggested_run)
            suggested_model = self.sim_root / "outputs" / "models" / f"{suggested_run}.pt"
            self.var_model.set(str(suggested_model))
        self._append_log(f"[profile] prepared dataset for profile {profile_id}\n")
        if cfg_info:
            self._safe_messagebox_info(
                "Ready",
                f"Config '{cfg_path.name}' assigned.\n"
                f"Out directory: {self.var_out.get()}\n"
                "Run the pipeline to generate the matching dataset for this profile.",
            )
        else:
            self._safe_messagebox_info(
                "Ready",
                "No config automatically detected for this profile.\n"
                "Pick or create a config that sets `run.component_profile` to "
                f"'{profile_id}', then run the pipeline to create the dataset.",
            )

        config_candidate = cfg_path if cfg_path else Path(self.var_config.get())
        out_dir = Path(self.var_out.get())
        try:
            created = self._ensure_dataset_skeleton(profile_id, config_candidate, out_dir)
            if created:
                self._refresh_datasets()
                self._select_dataset(out_dir)
        except Exception as exc:
            print(f"Failed to create dataset placeholder: {exc}")

    def _safe_messagebox_info(self, title: str, message: str) -> None:
        root = self.frame.winfo_toplevel()
        try:
            if not root.winfo_exists():
                return
        except Exception:
            return
        messagebox.showinfo(title, message)

    def _config_for_profile(self, profile_id: str, *, want_backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
        cache_key = f"{profile_id}|{want_backend or ''}"
        if cache_key in self._profile_config_cache:
            return self._profile_config_cache[cache_key]

        configs_root = self.sim_root / "configs"
        best_entry = None
        for cfg in sorted(configs_root.glob("*.yaml")):
            try:
                data = yaml.safe_load(cfg)
            except Exception:
                continue
            run_block = data.get("run") or {}
            if run_block.get("component_profile") != profile_id:
                continue
            if want_backend:
                backend = ((data.get("render") or {}).get("backend") or "").strip()
                if backend and backend != want_backend:
                    continue
            if run_block.get("component_profile") == profile_id:
                best_entry = {
                    "path": cfg,
                    "run_id": str(run_block.get("run_id") or cfg.stem),
                }
                break

        if best_entry is None:
            # Fallback: scan text for a matching component_profile line
            needle = f"component_profile: {profile_id}"
            needle_quoted = f'component_profile: "{profile_id}"'
            for cfg in sorted(configs_root.glob("*.yaml")):
                try:
                    text = cfg.read_text(encoding="utf-8")
                except Exception:
                    continue
                if needle in text or needle_quoted in text:
                    if want_backend:
                        # Avoid selecting a mismatched backend config in fallback mode.
                        try:
                            data = yaml.safe_load(text)
                            backend = ((data.get("render") or {}).get("backend") or "").strip()
                            if backend and backend != want_backend:
                                continue
                        except Exception:
                            continue
                    best_entry = {"path": cfg, "run_id": cfg.stem}
                    break

        self._profile_config_cache[cache_key] = best_entry
        return best_entry

    def _autoselect_config_for_profile(self, profile_id: str, *, prefer_quiet: bool) -> None:
        """Auto-pick a config that matches current Profile + Render backend.

        prefer_quiet=True: do not overwrite a matching config silently.
        """
        want_backend = None
        if hasattr(self, "var_render_backend"):
            want_backend = self.var_render_backend.get().strip() or None

        cfg_info = self._config_for_profile(profile_id, want_backend=want_backend)
        if not cfg_info and want_backend:
            # If no backend-specific config exists, fall back to any config for the profile.
            cfg_info = self._config_for_profile(profile_id, want_backend=None)
        if not cfg_info:
            return

        cfg_path: Path = cfg_info["path"]
        try:
            rel = str(cfg_path.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            rel = str(cfg_path)

        cur = self.var_config.get().strip()
        if cur == rel:
            return

        if prefer_quiet and cur and want_backend:
            # If current config already matches want_backend, don't override silently.
            try:
                cur_path = (self.sim_root / cur) if not Path(cur).is_absolute() else Path(cur)
                if cur_path.exists():
                    data = yaml.safe_load(cur_path.read_text(encoding="utf-8"))
                    cur_backend = ((data.get("render") or {}).get("backend") or "").strip()
                    if cur_backend == want_backend:
                        return
            except Exception:
                return

        self.var_config.set(rel)

    def _show_profile_info(self) -> None:
        """Show detailed information about the selected profile."""
        import yaml

        profile_id = self.var_profile.get()
        if not profile_id:
            messagebox.showinfo("Profile Info", "No profile selected")
            return

        profiles_dir = self.sim_root / "configs" / "profiles"
        profile_path = profiles_dir / f"{profile_id}.yaml"

        if not profile_path.exists():
            messagebox.showerror("Error", f"Profile file not found:\n{profile_path}")
            return

        try:
            with open(profile_path, 'r') as f:
                profile_data = yaml.safe_load(f)

            # Create info dialog
            dialog = tk.Toplevel(self.frame)
            dialog.title(f"Profile Info: {profile_id}")
            dialog.geometry("600x500")

            # Create text widget with scrollbar
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)

            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side="right", fill="y")

            text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
            text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)

            # Format and display profile data
            info_text = f"Profile ID: {profile_id}\n"
            info_text += f"Path: {profile_path}\n"
            info_text += "\n" + "="*60 + "\n\n"
            info_text += yaml.dump(profile_data, default_flow_style=False, sort_keys=False)

            text.insert("1.0", info_text)
            text.config(state="disabled")

            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load profile:\n{str(e)}")

    def _check_profile_compatibility(self) -> None:
        """Check if dataset and model profiles are compatible."""
        try:
            ds = self._selected_dataset_dir()
            if not ds:
                self.var_profile_compat.set("")
                return

            # Load dataset profile
            manifest_path = ds / "dataset_manifest.json"
            if not manifest_path.exists():
                self.var_profile_compat.set("")
                return

            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            ds_profile_id = manifest.get('component_profile', {}).get('profile_id')
            ds_profile_hash = manifest.get('component_profile', {}).get('profile_hash')

            if not ds_profile_id:
                self.var_profile_compat.set("")
                return

            # Load model profile if a model is selected
            model_path_str = self.var_model.get().strip()
            if not model_path_str:
                self.var_profile_compat.set("")
                return

            model_path = self._resolve_model_path(model_path_str)
            if not model_path or not model_path.exists():
                self.var_profile_compat.set("")
                return
            # Multi-model bundle (directory): resolve per-profile checkpoint.
            if model_path.is_dir():
                cand = bundle_checkpoint_path(model_path, ds_profile_id, kind="best")
                if not cand.exists():
                    self.var_profile_compat.set(f"⚠ Multi-model: no weights for {ds_profile_id}")
                    self.lbl_profile_compat.configure(foreground="orange")
                    return
                model_path = cand

            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            model_profile = checkpoint.get('component_profile')

            if not model_profile:
                self.var_profile_compat.set("⚠ Model has no profile (legacy)")
                self.lbl_profile_compat.configure(foreground="orange")
                return

            model_profile_id = model_profile.get('profile_id')
            model_profile_hash = model_profile.get('profile_hash')

            # Check compatibility
            if isinstance(model_profile_id, str) and model_profile_id.lower().startswith("multi"):
                self.var_profile_compat.set("✓ Compatible (multi-model)")
                self.lbl_profile_compat.configure(foreground="green")
            elif model_profile_id != ds_profile_id:
                self.var_profile_compat.set(f"✗ INCOMPATIBLE: Model={model_profile_id}, Dataset={ds_profile_id}")
                self.lbl_profile_compat.configure(foreground="red")
            elif model_profile_hash != ds_profile_hash:
                self.var_profile_compat.set(f"⚠ Profile hash mismatch (same ID, different version)")
                self.lbl_profile_compat.configure(foreground="orange")
            else:
                self.var_profile_compat.set(f"✓ Compatible: {ds_profile_id}")
                self.lbl_profile_compat.configure(foreground="green")

        except Exception as e:
            self.var_profile_compat.set("")
            print(f"Profile compatibility check failed: {e}")

    def _show_dataset_profile_info(self) -> None:
        """Show detailed profile information for the selected dataset."""
        import yaml

        ds = self._selected_dataset_dir()
        if not ds:
            messagebox.showinfo("Dataset Profile", "No dataset selected")
            return

        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists():
            messagebox.showerror(
                "No Manifest",
                f"Dataset has no manifest (legacy dataset).\n\n"
                f"Use the backfill tool to add a manifest:\n"
                f"  .venv/bin/python tools/backfill_manifest.py --data {ds}"
            )
            return

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            profile_id = manifest.get('component_profile', {}).get('profile_id', 'unknown')
            profile_hash = manifest.get('component_profile', {}).get('profile_hash', '')
            profile_path_str = manifest.get('component_profile', {}).get('profile_path', '')

            # Try to load the actual profile
            profiles_dir = self.sim_root / "configs" / "profiles"
            profile_path = profiles_dir / f"{profile_id}.yaml"

            # Create info dialog
            dialog = tk.Toplevel(self.frame)
            dialog.title(f"Dataset Profile: {ds.name}")
            dialog.geometry("650x550")

            # Create text widget with scrollbar
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)

            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side="right", fill="y")

            text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
            text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)

            # Format and display info
            info_text = f"Dataset: {ds.name}\n"
            info_text += f"Profile ID: {profile_id}\n"
            info_text += f"Profile Hash: {profile_hash}\n"
            info_text += f"Profile Path: {profile_path_str}\n"
            info_text += "\n" + "="*60 + "\n"
            info_text += "MANIFEST METADATA\n"
            info_text += "="*60 + "\n\n"
            info_text += json.dumps(manifest, indent=2)

            if profile_path.exists():
                info_text += "\n\n" + "="*60 + "\n"
                info_text += "PROFILE DETAILS\n"
                info_text += "="*60 + "\n\n"
                with open(profile_path, 'r') as f:
                    profile_data = yaml.safe_load(f)
                info_text += yaml.dump(profile_data, default_flow_style=False, sort_keys=False)
            else:
                info_text += f"\n\n⚠ Profile file not found at: {profile_path}"

            text.insert("1.0", info_text)
            text.config(state="disabled")

            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset profile info:\n{str(e)}")

    def _slugify_name(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unnamed"

    def _apply_name_to_out_and_model(self) -> None:
        """Set Out/Model fields to a clean, sortable naming scheme."""
        base = self._slugify_name(self.var_name.get())
        if self.var_name_ts.get():
            tag = time.strftime("%Y%m%d_%H%M%S")
            name = f"{base}_{tag}"
        else:
            name = base

        out_rel = f"outputs/sim_data/runs/{name}"
        if hasattr(self, "var_model_bundle") and self.var_model_bundle.get():
            model_rel = f"outputs/models/{name}.bundle"
        else:
            model_rel = f"outputs/models/{name}.pt"

        self.var_out.set(out_rel)
        self.var_model.set(model_rel)
        try:
            self._refresh_models()
        except Exception:
            pass
        self._append_log(f"[naming] out={out_rel} model={model_rel}\n")

    def _selected_dataset_dir(self) -> Optional[Path]:
        """Get currently selected dataset directory."""
        dss = self._selected_dataset_dirs()
        return dss[0] if dss else None

    def _selected_dataset_dir_single(self) -> Optional[Path]:
        """Get selected dataset directory from the single combobox selection."""
        label = self.var_dataset.get().strip()
        if not label:
            return None
        if label in self._dataset_by_label:
            return self._dataset_by_label[label]
        # Back-compat: if the combobox was ever set to a bare name.
        for p in self._dataset_dirs:
            if p.name == label:
                return p
        return None

    def _selected_dataset_dirs(self) -> list[Path]:
        """Get selected dataset directories (multi-select aware)."""
        if hasattr(self, "var_dataset_multi") and bool(self.var_dataset_multi.get()):
            paths = self._decode_dataset_paths_json(self.var_dataset_multi_paths.get())
            out: list[Path] = []
            for p in paths:
                try:
                    if p.exists() and p.is_dir():
                        out.append(p)
                except Exception:
                    continue
            if out:
                return out
        ds = self._selected_dataset_dir_single()
        return [ds] if ds is not None else []

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        """Handle dataset selection change."""
        ds = self._selected_dataset_dir()
        if not ds:
            self.var_dataset_profile.set("Profile: -")
            self.var_ds_samples.set("Samples: -")
            return
        prev_ds = self.state.dataset_dir
        self.state.dataset_dir = ds
        self._recent_imgs.clear()
        self._label_dict.clear()
        self._image_to_sample.clear()
        self._meta_by_sample.clear()
        self._meta_by_image_rel.clear()
        self._start_dataset_size_calc(ds)

        # Load metadata and labels
        try:
            meta_path = ds / "meta.jsonl"
            if meta_path.exists():
                meta_rows = read_jsonl(meta_path, MetaRow)
                self._image_to_sample = {row.image_path: row.id for row in meta_rows}
                self._meta_by_sample = {row.id: row for row in meta_rows}
                self._meta_by_image_rel = {row.image_path: row for row in meta_rows}

            labels_path = ds / "labels.jsonl"
            if labels_path.exists():
                label_rows = read_jsonl(labels_path, LabelRow)
                self._label_dict = {row.id: row.class_name for row in label_rows}
        except Exception as e:
            print(f"Failed to load metadata/labels: {e}")

        # Load dataset profile from manifest
        try:
            manifest_path = ds / "dataset_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                profile_id = manifest.get('component_profile', {}).get('profile_id', 'unknown')
                profile_hash = manifest.get('component_profile', {}).get('profile_hash', '')
                stats = manifest.get("dataset_stats", {})
                total_samples = stats.get("total_samples")
                splits = stats.get("splits", {})
                train = splits.get("train")
                val = splits.get("val")
                test = splits.get("test")
                hash_short = profile_hash.split(':')[1][:12] if ':' in profile_hash else profile_hash[:12]
                self.var_dataset_profile.set(f"Profile: {profile_id} ({hash_short}...)")
                self._set_dataset_samples_from_manifest(ds, total_samples, train, val, test)
            else:
                self.var_dataset_profile.set("Profile: ⚠ No manifest (legacy dataset)")
                self._set_dataset_samples_info("Samples: legacy dataset", "black")
        except Exception as e:
            self.var_dataset_profile.set(f"Profile: ⚠ Error loading manifest")
            self._set_dataset_samples_info("Samples: error", "")
            print(f"Failed to load dataset manifest: {e}")

        # Check profile compatibility
        self._check_profile_compatibility()
        self._apply_dataset_settings(ds)

        img_dir = ds / "images"
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.png"))
            for p in imgs[-12:]:
                self._recent_imgs.append(str(p))
            self._refresh_thumbnails()

        # Notify other tabs of dataset change (handled by MonitorAppTabbed).
        try:
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception:
            pass

        # Update model dropdown default for this dataset, if available.
        try:
            # If the user currently points at the dataset-default model for the previous dataset,
            # keep it tracking the selected dataset.
            if prev_ds is not None:
                cur_model = self.var_model.get().strip()
                prev_default = str((self.sim_root / "outputs" / "models" / f"{prev_ds.name}.pt").resolve())
                cur_resolved = str(self._resolve_model_path(cur_model))
                if cur_resolved == prev_default:
                    self.var_model.set(str(self.sim_root / "outputs" / "models" / f"{ds.name}.pt"))
            self._refresh_models()
        except Exception:
            pass

    def _delete_dataset(self) -> None:
        """Delete selected dataset."""
        ds = self._selected_dataset_dir()
        if not ds:
            return
        if self.proc is not None:
            messagebox.showwarning("Busy", "Stop the running process before deleting datasets.")
            return
        if not messagebox.askyesno("Delete dataset", f"Delete dataset folder?\n\n{ds}"):
            return
        import shutil
        try:
            shutil.rmtree(ds)
            self._append_log(f"\n[deleted dataset {ds}]\n")
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
        self._refresh_datasets()

    def _create_new_dataset(self) -> None:
        """Create a new dataset folder under outputs/sim_data/runs with a placeholder manifest."""
        if self.proc is not None:
            messagebox.showwarning("Busy", "Stop the running process before creating datasets.")
            return

        initial = self._slugify_name(getattr(self, "var_name", tk.StringVar(value="")).get())
        new_name = simpledialog.askstring(
            "Create dataset",
            "New dataset folder name (will be created under outputs/sim_data/runs):",
            initialvalue=initial,
            parent=self.frame.winfo_toplevel(),
        )
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if "/" in new_name or "\\" in new_name:
            messagebox.showerror("Error", "Name must not contain path separators.")
            return

        out_dir = (self.sim_root / "outputs" / "sim_data" / "runs" / new_name).resolve()
        if out_dir.exists():
            if messagebox.askyesno("Dataset exists", f"Dataset already exists:\n\n{out_dir}\n\nSelect it?"):
                self._select_dataset(out_dir)
            return

        config_candidate = Path(self.var_config.get().strip()) if hasattr(self, "var_config") else None
        if not config_candidate or not config_candidate.exists():
            messagebox.showerror(
                "Missing config",
                f"Config not found:\n\n{config_candidate}\n\nPick a valid Config first, then retry.",
            )
            return

        profile_id = self.var_profile.get().strip() if hasattr(self, "var_profile") else ""
        if not profile_id:
            messagebox.showerror("Missing profile", "No profile selected.")
            return

        try:
            self._ensure_dataset_skeleton(profile_id, config_candidate, out_dir)
        except Exception as e:
            messagebox.showerror("Create failed", str(e))
            return

        # Prepare pipeline fields to generate into this folder.
        try:
            self.var_dataset_mode.set("new")
        except Exception:
            pass
        try:
            self.var_out.set(str(out_dir))
        except Exception:
            pass
        try:
            self.var_name.set(new_name)
        except Exception:
            pass
        try:
            model_path = (self.sim_root / "outputs" / "models" / f"{new_name}.pt").resolve()
            self.var_model.set(str(model_path))
            self._refresh_models()
        except Exception:
            pass

        self._refresh_datasets()
        self._select_dataset(out_dir)
        self._append_log(f"[create dataset] {out_dir}\n")

    def _snapshot_dataset_selected(self) -> None:
        """Create a dataset snapshot under outputs/sim_data/versions/ for later training/testing.

        Uses hardlinks when possible (fast + space efficient), falls back to copy.
        """
        ds = self._selected_dataset_dir()
        if not ds:
            return
        if self.proc is not None:
            messagebox.showwarning("Busy", "Stop the running process before snapshotting datasets.")
            return

        sim_data = self.sim_root / "outputs" / "sim_data"
        versions_root = sim_data / "versions"
        versions_root.mkdir(parents=True, exist_ok=True)

        group = ds.name
        tag = time.strftime("%Y%m%d_%H%M%S")
        dst = versions_root / group / f"{group}_{tag}"

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            messagebox.showerror("Error", f"Snapshot destination already exists:\n{dst}")
            return

        if not messagebox.askyesno("Snapshot dataset", f"Create snapshot?\n\nFrom:\n{ds}\n\nTo:\n{dst}"):
            return

        self._append_log(f"\n[snapshot dataset] {ds} -> {dst}\n")

        def link_or_copy_file(src: Path, dstp: Path) -> None:
            dstp.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dstp)
            except Exception:
                shutil.copy2(src, dstp)

        def worker() -> None:
            try:
                for dirpath, dirnames, filenames in os.walk(ds):
                    rel = Path(dirpath).relative_to(ds)
                    for fn in filenames:
                        s = Path(dirpath) / fn
                        d = dst / rel / fn
                        try:
                            link_or_copy_file(s, d)
                        except Exception:
                            pass
                self.log_q.put(f"[snapshot dataset] done: {dst}\n")
            except Exception as e:
                self.log_q.put(f"[snapshot dataset] failed: {e}\n")
                try:
                    shutil.rmtree(dst)
                except Exception:
                    pass
            finally:
                try:
                    self.frame.after(0, self._refresh_datasets)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _rename_dataset_selected(self) -> None:
        """Rename a dataset folder under outputs/sim_data/runs or outputs/sim_data/versions."""
        ds = self._selected_dataset_dir()
        if not ds:
            return
        if self.proc is not None:
            messagebox.showwarning("Busy", "Stop the running process before renaming datasets.")
            return

        sim_data = (self.sim_root / "outputs" / "sim_data").resolve()
        runs = (sim_data / "runs").resolve()
        versions = (sim_data / "versions").resolve()

        try:
            rp = ds.resolve()
        except Exception:
            rp = ds

        if str(rp).startswith(str(runs) + os.sep):
            kind = "run"
            old_name = rp.name
            parent = rp.parent
        elif str(rp).startswith(str(versions) + os.sep):
            kind = "snapshot"
            old_name = rp.name
            parent = rp.parent
        else:
            messagebox.showerror("Error", f"Can only rename datasets under:\n{runs}\n{versions}\n\nSelected:\n{ds}")
            return

        new_name = simpledialog.askstring(
            "Rename dataset",
            f"New folder name for this {kind} dataset:",
            initialvalue=old_name,
            parent=self.frame.winfo_toplevel(),
        )
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if "/" in new_name or "\\" in new_name:
            messagebox.showerror("Error", "Name must not contain path separators.")
            return

        dst = parent / new_name
        if dst.exists():
            messagebox.showerror("Error", f"Target already exists:\n{dst}")
            return

        if not messagebox.askyesno("Rename dataset", f"Rename?\n\nFrom:\n{rp}\n\nTo:\n{dst}"):
            return

        try:
            rp.rename(dst)
        except Exception as e:
            messagebox.showerror("Error", f"Rename failed:\n{e}")
            return

        # Update state + dropdown.
        self.state.dataset_dir = dst
        self._refresh_datasets()
        # Set the combobox selection to the new label.
        try:
            label = self._display_for_dataset(dst, runs=runs, versions=versions)
            for k, v in self._dataset_by_label.items():
                if v == dst or k == label:
                    self.var_dataset.set(k)
                    break
        except Exception:
            pass

        # Refresh the UI for the new selection immediately
        self._on_dataset_selected()

        # If Model: was the old dataset-default path, update it to the new default.
        try:
            cur_model = self.var_model.get().strip()
            cur_resolved = str(self._resolve_model_path(cur_model))
            old_default = str((self.sim_root / "outputs" / "models" / f"{old_name}.pt").resolve())
            if cur_resolved == old_default:
                self.var_model.set(str(self.sim_root / "outputs" / "models" / f"{new_name}.pt"))
            cur_out = self.var_out.get().strip()
            try:
                cur_out_res = str(self._resolve_out_dir(cur_out))
                old_out = str((self.sim_root / "outputs" / "sim_data" / "runs" / old_name).resolve())
                if cur_out_res == old_out:
                    self.var_out.set(str(self.sim_root / "outputs" / "sim_data" / "runs" / new_name))
            except Exception:
                pass
        except Exception:
            pass

        # Mirror rename in outputs/models
        self._rename_models_for_dataset(old_name, new_name)

        self._append_log(f"[rename dataset] {rp} -> {dst}\n")
        try:
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception:
            pass

    def _rename_models_for_dataset(self, old_name: str, new_name: str) -> None:
        models_root = self.sim_root / "outputs" / "models"
        for suffix in ["", "_last"]:
            old = models_root / f"{old_name}{suffix}.pt"
            new = models_root / f"{new_name}{suffix}.pt"
            if old.exists():
                try:
                    old.rename(new)
                except Exception as e:
                    print(f"Failed to rename model {old} -> {new}: {e}")
            old_meta = models_root / f"{old_name}{suffix}.pt.meta.json"
            new_meta = models_root / f"{new_name}{suffix}.pt.meta.json"
            if old_meta.exists():
                try:
                    old_meta.rename(new_meta)
                except Exception as e:
                    print(f"Failed to rename meta {old_meta} -> {new_meta}: {e}")

        versions_root = models_root / "versions"
        old_dir = versions_root / old_name
        new_dir = versions_root / new_name
        if old_dir.exists():
            try:
                old_dir.rename(new_dir)
            except Exception as e:
                print(f"Failed to rename versioned models {old_dir} -> {new_dir}: {e}")

    def _is_model_bundle_target(self, model_path: Path) -> bool:
        """Return True if train/eval should treat this as a multi-model bundle target."""
        try:
            if model_path.exists() and model_path.is_dir():
                return True
        except Exception:
            pass
        return str(model_path).endswith(".bundle")

    def _datasets_require_bundle(self, dss: list[Path]) -> bool:
        """Return True when multiple selected datasets span multiple component profiles."""
        pids: set[str] = set()
        for ds in dss:
            pid = self._dataset_profile_id(ds)
            if pid:
                pids.add(pid)
        return len(pids) > 1

    def _validate_selected(self) -> None:
        """Validate selected dataset."""
        dss = self._selected_dataset_dirs()
        if not dss:
            return
        if len(dss) == 1:
            self._run_simple_cmd(["./.venv/bin/python", "tools/validate_dataset.py", "--data", str(dss[0])])
            return

        parts: list[str] = []
        for ds in dss:
            argv = ["./.venv/bin/python", "tools/validate_dataset.py", "--data", str(ds)]
            parts.append(" ".join(shlex.quote(x) for x in argv))
        self._append_log(f"\n=== Validate {len(dss)} datasets ===\n")
        self._run_simple_cmd([" && ".join(parts)])

    def _train_selected(self) -> None:
        """Train model on selected dataset."""
        dss = self._selected_dataset_dirs()
        if not dss:
            return
        # Respect the Model: field when present; fall back to outputs/models/<dataset>.pt
        model_s = self.var_model.get().strip() if hasattr(self, "var_model") else ""
        if model_s:
            model_out = self._resolve_model_path(model_s)
        else:
            model_out = (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()

        if len(dss) == 1:
            self._run_simple_cmd(["./.venv/bin/python", "scripts/train.py", "--data", str(dss[0]), "--out", str(model_out)])
            return

        # Multi-dataset training = sequential train runs into the same output checkpoint.
        # First dataset trains from scratch; subsequent datasets resume for +epochs.
        if self._datasets_require_bundle(dss) and not self._is_model_bundle_target(model_out):
            messagebox.showerror(
                "Multi-dataset training requires bundle",
                "You selected datasets with different component profiles.\n\n"
                "Train output must be a multi-model bundle directory (ends with .bundle).\n\n"
                "Tip: enable 'Multi-Model (.bundle)' and click 'Apply to Out+Model', or set Model to outputs/models/<name>.bundle."
            )
            return

        extra_s = self.var_continue_epochs.get().strip() if hasattr(self, "var_continue_epochs") else "10"
        try:
            extra = int(extra_s)
        except Exception:
            messagebox.showerror("Error", "+epochs must be an integer.")
            return
        if extra < 0:
            messagebox.showerror("Error", "+epochs must be >= 0.")
            return

        out_mode = (self.var_continue_out_mode.get().strip() if hasattr(self, "var_continue_out_mode") else "best") or "best"
        if out_mode not in ("last", "best"):
            out_mode = "best"

        parts: list[str] = []
        is_bundle = self._is_model_bundle_target(model_out)
        seen_pids: set[str] = set()
        for i, ds in enumerate(dss):
            pid = self._dataset_profile_id(ds) or f"__no_profile__:{ds}"
            if is_bundle:
                # Bundle semantics:
                # - For a profile we haven't trained yet: run from scratch (no --resume).
                # - For repeats of the same profile: resume + extra epochs.
                if pid in seen_pids:
                    argv = [
                        "./.venv/bin/python", "scripts/train.py",
                        "--data", str(ds),
                        "--out", str(model_out),
                        "--resume", str(model_out),
                        "--extra-epochs", str(extra),
                        "--out-mode", out_mode,
                    ]
                else:
                    argv = ["./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_out)]
                    seen_pids.add(pid)
            else:
                # Single-checkpoint semantics: always resume after the first dataset.
                if i == 0:
                    argv = ["./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_out)]
                else:
                    argv = [
                        "./.venv/bin/python", "scripts/train.py",
                        "--data", str(ds),
                        "--out", str(model_out),
                        "--resume", str(model_out),
                        "--extra-epochs", str(extra),
                        "--out-mode", out_mode,
                    ]
            parts.append(" ".join(shlex.quote(x) for x in argv))

        self._append_log(f"\n=== Multi-dataset train: {len(dss)} datasets ===\n")
        for ds in dss:
            self._append_log(f"  - {ds}\n")
        self._append_log(f"out: {model_out}\n")
        self._append_log(f"+epochs per subsequent dataset: {extra} (out-mode={out_mode})\n\n")

        self._run_simple_cmd([" && ".join(parts)])

    def _eval_selected(self) -> None:
        """Evaluate model on selected dataset."""
        dss = self._selected_dataset_dirs()
        if not dss:
            return
        model_s = self.var_model.get().strip() if hasattr(self, "var_model") else ""
        if model_s:
            model_in = self._resolve_model_path(model_s)
        else:
            model_in = (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_in.exists():
            messagebox.showerror("Missing model", f"Model not found:\n{model_in}\n\nRun Train first.")
            return
        if len(dss) == 1:
            self._run_simple_cmd(["./.venv/bin/python", "scripts/eval.py", "--data", str(dss[0]), "--model", str(model_in)])
            return

        parts: list[str] = []
        for ds in dss:
            argv = ["./.venv/bin/python", "scripts/eval.py", "--data", str(ds), "--model", str(model_in)]
            parts.append(" ".join(shlex.quote(x) for x in argv))
        self._append_log(f"\n=== Eval on {len(dss)} datasets ===\n")
        self._run_simple_cmd([" && ".join(parts)])

    def _continue_train_selected(self) -> None:
        """Continue training (resume) on a fixed dataset using an existing checkpoint."""
        dss = self._selected_dataset_dirs()
        if not dss:
            return

        model_s = self.var_model.get().strip() if hasattr(self, "var_model") else ""
        if model_s:
            model_path = self._resolve_model_path(model_s)
        else:
            model_path = (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()

        if not model_path.exists():
            messagebox.showerror("Missing model", f"Model not found to resume from:\n{model_path}\n\nRun Train first.")
            return

        extra_s = self.var_continue_epochs.get().strip() if hasattr(self, "var_continue_epochs") else "10"
        try:
            extra = int(extra_s)
        except Exception:
            messagebox.showerror("Error", "Extra epochs must be an integer.")
            return
        if extra < 1:
            messagebox.showerror("Error", "Extra epochs must be >= 1.")
            return

        out_mode = (self.var_continue_out_mode.get().strip() if hasattr(self, "var_continue_out_mode") else "last") or "last"
        if out_mode not in ("last", "best"):
            out_mode = "last"

        if len(dss) > 1 and self._datasets_require_bundle(dss) and not self._is_model_bundle_target(model_path):
            messagebox.showerror(
                "Multi-dataset resume requires bundle",
                "You selected datasets with different component profiles.\n\n"
                "Resume target must be a multi-model bundle directory (ends with .bundle)."
            )
            return

        # If resuming into a bundle, ensure each selected dataset's profile has a checkpoint present.
        if len(dss) > 1 and self._is_model_bundle_target(model_path):
            missing: list[str] = []
            for ds in dss:
                pid = self._dataset_profile_id(ds)
                if not pid:
                    missing.append(f"{ds} (missing profile metadata)")
                    continue
                try:
                    ckpt = bundle_checkpoint_path(model_path, pid, kind="best")
                    if not ckpt.exists():
                        missing.append(f"{pid} (no checkpoint in bundle)")
                except Exception:
                    missing.append(f"{pid} (bundle lookup failed)")
            if missing:
                messagebox.showerror(
                    "Missing bundle checkpoints",
                    "The selected model bundle does not have checkpoints for all selected datasets.\n\n"
                    + "\n".join(missing)
                )
                return

        if len(dss) == 1:
            ds = dss[0]
            self._append_log(
                f"\n=== Continue training: +{extra} epochs (out={out_mode}) ===\n"
                f"data:  {ds}\n"
                f"model: {model_path}\n\n"
            )
            self._run_simple_cmd([
                "./.venv/bin/python",
                "scripts/train.py",
                "--data", str(ds),
                "--out", str(model_path),
                "--resume", str(model_path),
                "--extra-epochs", str(extra),
                "--out-mode", out_mode,
            ])
            return

        parts: list[str] = []
        for ds in dss:
            argv = [
                "./.venv/bin/python",
                "scripts/train.py",
                "--data", str(ds),
                "--out", str(model_path),
                "--resume", str(model_path),
                "--extra-epochs", str(extra),
                "--out-mode", out_mode,
            ]
            parts.append(" ".join(shlex.quote(x) for x in argv))

        self._append_log(
            f"\n=== Continue training on {len(dss)} datasets: +{extra} epochs each (out={out_mode}) ===\n"
            f"model: {model_path}\n"
        )
        for ds in dss:
            self._append_log(f"data:  {ds}\n")
        self._append_log("\n")

        self._run_simple_cmd([" && ".join(parts)])
