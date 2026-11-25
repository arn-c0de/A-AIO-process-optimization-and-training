"""Widgets of the Profile Preview tab (mixed into :class:`ProfilePreviewTab`).

Builds the settings column and the gallery; the tab module holds the behaviour. The widgets and Tk
variables stay attributes of the tab (``self.var_profile``, ``self.cb_backend`` ...).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from gui.utils.pipeline_helpers import BOARD_DRC_CHOICES, BOARD_ROUTING_CHOICES, BOARD_RULES_CHOICES, LOOK_CHOICES
from gui.widgets import LogView
from simple_sim.generators.opencv2d.photoreal import SUPERSAMPLING_CHOICES

from .logic import CAMERA_CHOICES, CONFIG_AUTO, QUALITY_PRESETS, STYLE_CHOICES

QUALITY_CUSTOM = "Custom"
FROM_CONFIG = "from config"
ENGINE_CHOICES = [FROM_CONFIG, "CYCLES", "BLENDER_EEVEE"]
DEVICE_CHOICES = [FROM_CONFIG, "GPU", "CPU"]
THUMB_SIZES = [128, 192, 256, 384, 512]
MAX_PER_STATE = 50


class ProfilePreviewUIMixin:
    """Tk variables and widget construction; callbacks (``self._start_render`` ...) live in the tab."""

    frame: ttk.Frame
    _start_render: Any
    _request_stop: Any
    _open_folder: Any
    _relayout: Any
    _load_profiles: Any
    _on_profile_changed: Any
    _on_backend_changed: Any
    _update_total: Any
    _new_seed: Any
    _on_canvas_resize: Any
    _on_wheel: Any

    def _create_vars(self) -> None:
        self.var_profile = tk.StringVar()
        self.var_backend = tk.StringVar()
        self.var_config = tk.StringVar(value=CONFIG_AUTO)
        # "Standard" by default: "Ultra" means 256 Cycles samples + 4x boards in 3D (very slow first run).
        self.var_quality = tk.StringVar(value="Standard")
        self.var_ss = tk.IntVar(value=QUALITY_PRESETS["Standard"]["supersampling"])
        self.var_style = tk.StringVar(value="photoreal")
        self.var_camera = tk.StringVar(value=CAMERA_CHOICES[0])
        self.var_samples = tk.IntVar(value=QUALITY_PRESETS["Standard"]["samples"])
        self.var_engine = tk.StringVar(value=FROM_CONFIG)
        self.var_device = tk.StringVar(value=FROM_CONFIG)
        # render.look / render.board on top of the run config (both backends); LOOK_CHOICES[0] is "from config".
        self.var_look = tk.StringVar(value=LOOK_CHOICES[0])
        self.var_board_routing = tk.StringVar(value=BOARD_ROUTING_CHOICES[0])
        self.var_board_drc = tk.StringVar(value=BOARD_DRC_CHOICES[0])
        self.var_board_rules = tk.StringVar(value=BOARD_RULES_CHOICES[0])
        self.var_variants = tk.IntVar(value=2)
        self.var_seed = tk.StringVar(value="2026")
        self.var_cardinal = tk.BooleanVar(value=True)
        self.var_filters = tk.BooleanVar(value=False)
        self.var_thumb = tk.IntVar(value=256)
        self.var_overlay = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Ready.")


    def _build_settings(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Profile", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Profile").grid(row=0, column=0, sticky="w")
        self.cb_profile = ttk.Combobox(box, textvariable=self.var_profile, state="readonly", width=28)
        self.cb_profile.grid(row=0, column=1, sticky="ew", pady=2)
        self.cb_profile.bind("<<ComboboxSelected>>", lambda _e: self._on_profile_changed())
        # Re-scan configs/profiles/ whenever the list opens: profiles added while the GUI runs show up.
        self.cb_profile.configure(postcommand=self._load_profiles)
        ttk.Label(box, text="Backend").grid(row=1, column=0, sticky="w")
        self.cb_backend = ttk.Combobox(box, textvariable=self.var_backend, state="readonly", width=28)
        self.cb_backend.grid(row=1, column=1, sticky="ew", pady=2)
        self.cb_backend.bind("<<ComboboxSelected>>", lambda _e: self._on_backend_changed())
        ttk.Label(box, text="Run config").grid(row=2, column=0, sticky="w")
        self.cb_config = ttk.Combobox(box, textvariable=self.var_config, state="readonly", width=28)
        self.cb_config.grid(row=2, column=1, sticky="ew", pady=2)

        box = ttk.LabelFrame(parent, text="Quality", padding=8)
        box.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(box)
        row.grid(row=0, column=0, columnspan=2, sticky="w")
        for name in [*QUALITY_PRESETS, QUALITY_CUSTOM]:
            ttk.Radiobutton(row, text=name, value=name, variable=self.var_quality).pack(side="left", padx=(0, 6))
        ttk.Label(box, text="Supersampling").grid(row=1, column=0, sticky="w")
        ss_row = ttk.Frame(box)
        ss_row.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Spinbox(ss_row, from_=min(SUPERSAMPLING_CHOICES), to=max(SUPERSAMPLING_CHOICES), textvariable=self.var_ss,
                    width=4, state="readonly").pack(side="left")
        ttk.Label(ss_row, text="x  (2D render / 3D board)").pack(side="left", padx=(4, 0))
        ttk.Label(box, text="Style (2D)").grid(row=2, column=0, sticky="w")
        self.cb_style = ttk.Combobox(box, textvariable=self.var_style, values=STYLE_CHOICES, state="readonly", width=14)
        self.cb_style.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Camera").grid(row=3, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.var_camera, values=CAMERA_CHOICES, state="readonly",
                     width=14).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Cycles samples (3D)").grid(row=4, column=0, sticky="w")
        self.sp_samples = ttk.Spinbox(box, from_=1, to=4096, increment=16, textvariable=self.var_samples, width=6)
        self.sp_samples.grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Engine (3D)").grid(row=5, column=0, sticky="w")
        self.cb_engine = ttk.Combobox(box, textvariable=self.var_engine, values=ENGINE_CHOICES, state="readonly", width=14)
        self.cb_engine.grid(row=5, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Device (3D)").grid(row=6, column=0, sticky="w")
        self.cb_device = ttk.Combobox(box, textvariable=self.var_device, values=DEVICE_CHOICES, state="readonly", width=14)
        self.cb_device.grid(row=6, column=1, sticky="w", pady=2)
        # render.look and render.board overrides (both backends: the 3D bake draws the same board).
        ttk.Label(box, text="Look").grid(row=7, column=0, sticky="w")
        self.cb_look = ttk.Combobox(box, textvariable=self.var_look, values=list(LOOK_CHOICES), state="readonly", width=14)
        self.cb_look.grid(row=7, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Copper (board)").grid(row=8, column=0, sticky="w")
        self.cb_board_routing = ttk.Combobox(box, textvariable=self.var_board_routing, values=list(BOARD_ROUTING_CHOICES),
                                             state="readonly", width=14)
        self.cb_board_routing.grid(row=8, column=1, sticky="w", pady=2)
        ttk.Label(box, text="DRC (board)").grid(row=9, column=0, sticky="w")
        self.cb_board_drc = ttk.Combobox(box, textvariable=self.var_board_drc, values=list(BOARD_DRC_CHOICES),
                                         state="readonly", width=14)
        self.cb_board_drc.grid(row=9, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Design rules").grid(row=10, column=0, sticky="w")
        self.cb_board_rules = ttk.Combobox(box, textvariable=self.var_board_rules, values=list(BOARD_RULES_CHOICES),
                                           state="readonly", width=14)
        self.cb_board_rules.grid(row=10, column=1, sticky="w", pady=2)

        box = ttk.LabelFrame(parent, text="Samples", padding=8)
        box.pack(fill="both", expand=True, pady=(8, 0))
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Defect states").grid(row=0, column=0, sticky="nw")
        self.lb_states = tk.Listbox(box, selectmode="extended", height=7, exportselection=False)
        self.lb_states.grid(row=0, column=1, sticky="nsew", pady=2)
        box.rowconfigure(0, weight=1)
        ttk.Label(box, text="Images per state").grid(row=1, column=0, sticky="w")
        count_row = ttk.Frame(box)
        count_row.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Spinbox(count_row, from_=1, to=MAX_PER_STATE, textvariable=self.var_variants, width=4).pack(side="left")
        self.var_total = tk.StringVar()
        ttk.Label(count_row, textvariable=self.var_total).pack(side="left", padx=(6, 0))
        self.lb_states.bind("<<ListboxSelect>>", lambda _e: self._update_total())
        self.var_variants.trace_add("write", lambda *_: self._update_total())
        ttk.Label(box, text="Seed").grid(row=2, column=0, sticky="w")
        seed_row = ttk.Frame(box)
        seed_row.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Entry(seed_row, textvariable=self.var_seed, width=10).pack(side="left")
        ttk.Button(seed_row, text="New seed", command=self._new_seed).pack(side="left", padx=(4, 0))
        ttk.Checkbutton(box, text="Random 90° orientation", variable=self.var_cardinal).grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(box, text="Apply GUI image filters (blur, grain, ...)", variable=self.var_filters).grid(
            row=4, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(8, 0))
        self.btn_render = ttk.Button(btns, text="Render preview", command=self._start_render)
        self.btn_render.pack(side="left")
        self.btn_stop = ttk.Button(btns, text="Stop", command=self._request_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Open folder", command=self._open_folder).pack(side="left", padx=(4, 0))
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 0))
        ttk.Label(parent, textvariable=self.var_status, wraplength=320, justify="left").pack(fill="x", pady=(4, 0))

    def _build_gallery(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill="x")
        ttk.Label(top, text="Thumbnail size").pack(side="left")
        cb = ttk.Combobox(top, textvariable=self.var_thumb, values=THUMB_SIZES, state="readonly", width=5)
        cb.pack(side="left", padx=(4, 0))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._relayout())
        ttk.Checkbutton(top, text="Defect overlay (like the sample gallery)", variable=self.var_overlay,
                        command=self._relayout).pack(side="left", padx=(12, 0))
        ttk.Label(top, text="Click an image for full size.").pack(side="left", padx=(12, 0))

        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas = tk.Canvas(holder, highlightthickness=0)
        vsb = ttk.Scrollbar(holder, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self._grid_win = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda _e: self._on_canvas_resize())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, self._on_wheel, add="+")

        self.log = LogView(parent, height=5, wrap="word")
        self.log.pack(fill="x", pady=(6, 0))
