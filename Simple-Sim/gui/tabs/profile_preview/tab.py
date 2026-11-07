"""Profile Preview tab: render a few images per defect state of one profile with chosen settings.

Lets you check how a profile looks (style, supersampling up to "Ultra", camera preset,
Blender quality) before generating a dataset. Rendering runs in a worker thread.
"""

from __future__ import annotations

import queue
import random
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from gui.widgets.layout import keep_pane_visible
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui.components.overlay_renderer import draw_defect_overlay
from gui.tabs.core.base import BaseTab
from gui.utils.pipeline_helpers import BOARD_DRC_CHOICES, BOARD_ROUTING_CHOICES, BOARD_RULES_CHOICES, LOOK_CHOICES
from gui.utils.settings_store import bind_persisted_vars
from simple_sim.generators.opencv2d.photoreal import SUPERSAMPLING_CHOICES
from simple_sim.preview.render import BACKEND_2D, BACKEND_3D, resolve_defect_types

from .logic import (
    CAMERA_CHOICES,
    CONFIG_AUTO,
    QUALITY_PRESETS,
    STYLE_CHOICES,
    PreviewCancelled,
    PreviewImage,
    PreviewRequest,
    ProfilePreviewLogic,
    read_sidecar,
)

if TYPE_CHECKING:
    from gui.state import UiState

QUALITY_CUSTOM = "Custom"
FROM_CONFIG = "from config"
ENGINE_CHOICES = [FROM_CONFIG, "CYCLES", "BLENDER_EEVEE"]
DEVICE_CHOICES = [FROM_CONFIG, "GPU", "CPU"]
THUMB_SIZES = [128, 192, 256, 384, 512]
MAX_PER_STATE = 50


class ProfilePreviewTab(BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.logic = ProfilePreviewLogic(sim_root)
        self._profiles: dict[str, set[str]] = {}
        self._profile_data: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._images: list[PreviewImage] = []
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._cells: list[ttk.Frame] = []
        # Decoded preview images (path, mtime, overlay) -> PIL, so relayouts (resize, thumbnail size,
        # overlay toggle) do not re-read and re-decode every PNG.
        self._pil_cache: dict[tuple[str, int, bool], Image.Image] = {}
        self._poll_id: str | None = None
        self._applying_preset = False
        self._t_start = 0.0
        self._expected = 0
        self._layout_cols = 0
        self._last_image_s = 0.0

    # ------------------------------------------------------------------ UI

    def build_ui(self) -> None:
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

        paned = ttk.PanedWindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        keep_pane_visible(paned, 0, left, min_width=300)  # settings column must never collapse

        self._build_settings(left)
        self._build_gallery(right)

        self._load_profiles()
        bind_persisted_vars(
            self.state.settings_store,
            [
                ("profile_preview.profile", self.var_profile),
                ("profile_preview.quality", self.var_quality),
                ("profile_preview.supersampling", self.var_ss),
                ("profile_preview.style", self.var_style),
                ("profile_preview.camera", self.var_camera),
                ("profile_preview.samples", self.var_samples),
                ("profile_preview.engine", self.var_engine),
                ("profile_preview.device", self.var_device),
                ("profile_preview.look", self.var_look),
                ("profile_preview.board_routing", self.var_board_routing),
                ("profile_preview.board_drc", self.var_board_drc),
                ("profile_preview.board_rules", self.var_board_rules),
                ("profile_preview.variants", self.var_variants),
                ("profile_preview.seed", self.var_seed),
                ("profile_preview.cardinal", self.var_cardinal),
                ("profile_preview.filters", self.var_filters),
                ("profile_preview.thumb", self.var_thumb),
                ("profile_preview.overlay", self.var_overlay),
            ],
            self.frame,
            skip_empty_strings=True,
        )
        if self.var_profile.get() not in self._profiles and self._profiles:
            self.var_profile.set(next(iter(self._profiles)))
        self._on_profile_changed()
        self.var_quality.trace_add("write", lambda *_: self._apply_quality())
        for var in (self.var_ss, self.var_samples):
            var.trace_add("write", lambda *_: self._mark_custom())
        self._show_existing()

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

        self.log = tk.Text(parent, height=5, wrap="word", state="disabled")
        self.log.pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------------ settings logic

    def _load_profiles(self) -> None:
        self._profiles.clear()
        self._profile_data.clear()
        for pid, prof, backends in self.logic.profiles():
            self._profiles[pid] = backends
            self._profile_data[pid] = prof
        self.cb_profile.configure(values=list(self._profiles))

    def _on_profile_changed(self) -> None:
        pid = self.var_profile.get()
        backends = [b for b in (BACKEND_2D, BACKEND_3D) if b in self._profiles.get(pid, set())]
        self.cb_backend.configure(values=backends)
        if self.var_backend.get() not in backends:
            self.var_backend.set(backends[0] if backends else "")
        self.lb_states.delete(0, "end")
        for s in resolve_defect_types(self._profile_data.get(pid, {}), ""):
            self.lb_states.insert("end", s)
        self.lb_states.selection_set(0, "end")
        self._update_total()
        self._on_backend_changed()

    def _on_backend_changed(self) -> None:
        names = self.logic.configs_for(self.var_profile.get(), self.var_backend.get()) if self.var_backend.get() else []
        self.cb_config.configure(values=[CONFIG_AUTO, *names])
        if self.var_config.get() not in (CONFIG_AUTO, *names):
            self.var_config.set(CONFIG_AUTO)
        is_3d = self.var_backend.get() == BACKEND_3D
        self.cb_style.configure(state="disabled" if is_3d else "readonly")
        for w in (self.cb_engine, self.cb_device):
            w.configure(state="readonly" if is_3d else "disabled")
        self.sp_samples.configure(state="normal" if is_3d else "disabled")
        self._show_existing()

    def _update_total(self) -> None:
        """'= N images total' next to the per-state count."""
        try:
            per_state = max(1, min(MAX_PER_STATE, int(self.var_variants.get())))
        except (tk.TclError, ValueError):
            self.var_total.set("")
            return
        n_states = len(self.lb_states.curselection())
        self.var_total.set(f"x {n_states} states = {per_state * n_states} images")

    def _apply_quality(self) -> None:
        preset = QUALITY_PRESETS.get(self.var_quality.get())
        if preset is None:
            return
        self._applying_preset = True
        try:
            self.var_ss.set(preset["supersampling"])
            self.var_samples.set(preset["samples"])
        finally:
            self._applying_preset = False

    def _mark_custom(self) -> None:
        if self._applying_preset:
            return
        preset = QUALITY_PRESETS.get(self.var_quality.get())
        try:
            current = {"supersampling": int(self.var_ss.get()), "samples": int(self.var_samples.get())}
        except (tk.TclError, ValueError):
            return
        if preset is not None and preset != current:
            match = next((k for k, v in QUALITY_PRESETS.items() if v == current), QUALITY_CUSTOM)
            self.var_quality.set(match)

    def _new_seed(self) -> None:
        self.var_seed.set(str(random.randint(0, 999_999)))

    def _request(self) -> PreviewRequest:
        states = [self.lb_states.get(i) for i in self.lb_states.curselection()]
        if not states:
            raise ValueError("Select at least one defect state.")
        try:
            seed = int(self.var_seed.get().strip())
            variants = max(1, min(MAX_PER_STATE, int(self.var_variants.get())))
            samples = max(1, int(self.var_samples.get()))
        except (ValueError, tk.TclError) as e:
            raise ValueError("Seed, variants and samples must be integers.") from e
        filters = None
        if self.var_filters.get():
            from simple_sim.filter_profiles import load_shared_filter_profiles, shared_settings_path

            if self.state.settings_store is not None:
                self.state.settings_store.save()
            _p, _a, filters = load_shared_filter_profiles(shared_settings_path(self.sim_root))
        return PreviewRequest(
            profile_id=self.var_profile.get(),
            backend=self.var_backend.get(),
            config_name=self.var_config.get(),
            states=states,
            variants=variants,
            seed=seed,
            style=self.var_style.get(),
            supersampling=int(self.var_ss.get()),
            camera=self.var_camera.get(),
            use_filters=bool(self.var_filters.get()),
            image_filters=filters,
            cardinal_rotation=bool(self.var_cardinal.get()),
            blender_samples=samples,
            blender_engine="" if self.var_engine.get() == FROM_CONFIG else self.var_engine.get(),
            blender_device="" if self.var_device.get() == FROM_CONFIG else self.var_device.get(),
            look="" if self.var_look.get() == LOOK_CHOICES[0] else self.var_look.get(),
            board_routing="" if self.var_board_routing.get() == BOARD_ROUTING_CHOICES[0] else self.var_board_routing.get(),
            board_drc="" if self.var_board_drc.get() == BOARD_DRC_CHOICES[0] else self.var_board_drc.get(),
            board_rules="" if self.var_board_rules.get() == BOARD_RULES_CHOICES[0] else self.var_board_rules.get(),
        )

    # ------------------------------------------------------------------ rendering

    def _start_render(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            req = self._request()
        except ValueError as e:
            messagebox.showerror("Profile Preview", str(e), parent=self.frame)
            return
        self._stop.clear()
        self._images = []
        self._relayout()
        self._expected = len(req.states) * req.variants
        self.progress.configure(maximum=self._expected, value=0)
        self.btn_render.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._t_start = time.perf_counter()
        self._last_image_s = 0.0
        self.var_status.set(f"Rendering {self._expected} images...")
        self._log_clear()

        def run() -> None:
            try:
                self.logic.render(
                    req,
                    on_image=lambda im: self._queue.put(("image", im)),
                    on_log=lambda s: self._queue.put(("log", s)),
                    should_stop=self._stop.is_set,
                )
                self._queue.put(("done", None))
            except PreviewCancelled:
                self._queue.put(("cancelled", None))
            except Exception as e:  # noqa: BLE001 - shown in the GUI
                self._queue.put(("error", f"{type(e).__name__}: {e}"))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()
        self._poll()

    def _request_stop(self) -> None:
        """Stop takes effect between images; a running Blender render/board bake finishes first."""
        self._stop.set()
        self.btn_stop.configure(state="disabled")
        self.var_status.set("Stopping ... finishing the image/bake in progress")

    def _poll(self) -> None:
        finished = False
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "image":
                    self._images.append(payload)
                    n_cells = len(self._cells)
                    self._add_thumb(len(self._images) - 1, payload)
                    if len(self._cells) == n_cells:
                        self._log(f"[warn] {payload.state} #{payload.variant}: image missing or unreadable "
                                  f"({payload.path})")
                    self.progress.configure(value=len(self._images))
                    self._last_image_s = payload.seconds
                elif kind == "log":
                    self._log(payload)
                else:
                    finished = True
                    self._finish(kind, payload)
        except queue.Empty:
            pass
        if not finished:
            # Live elapsed time, so a slow first image (3D: board bake + Blender start) does not look frozen.
            elapsed = time.perf_counter() - self._t_start
            last = f" | last image {self._last_image_s:.1f}s" if self._last_image_s else ""
            if self._stop.is_set():
                self.var_status.set(f"Stopping ... finishing the image/bake in progress ({elapsed:.0f}s elapsed)")
            else:
                self.var_status.set(f"Rendering image {min(len(self._images) + 1, self._expected)}/{self._expected} "
                                    f"... {elapsed:.0f}s elapsed{last}")
            self._poll_id = self.frame.after(100, self._poll)

    def _finish(self, kind: str, payload: Any) -> None:
        self._poll_id = None
        self.btn_render.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        elapsed = time.perf_counter() - self._t_start
        if kind == "done":
            missing = len(self._images) - len(self._cells)
            note = f" ({missing} missing/unreadable, see log)" if missing > 0 else ""
            self.var_status.set(f"Done: {len(self._cells)} images in {elapsed:.1f}s.{note}")
        elif kind == "cancelled":
            self.var_status.set(f"Stopped after {len(self._images)} images.")
        else:
            self.var_status.set("Render failed, see log.")
            self._log(str(payload))
            messagebox.showerror("Profile Preview", str(payload), parent=self.frame)

    # ------------------------------------------------------------------ gallery

    def _show_existing(self) -> None:
        """Show images of the last render of the selected profile/backend (if any)."""
        if self._worker is not None and self._worker.is_alive():
            return
        out_dir = self.logic.output_dir(self.var_profile.get(), self.var_backend.get())
        order = {s: i for i, s in enumerate(self.lb_states.get(0, "end"))}
        images = []
        for p in out_dir.glob("*_v*.png"):
            state, _, v = p.stem.rpartition("_v")
            if v.isdigit():
                side = read_sidecar(p)
                images.append(PreviewImage(state, int(v), p, 0.0, str(side.get("info") or ""),
                                           side.get("defect"), side.get("nominal")))
        images.sort(key=lambda im: (order.get(im.state, len(order)), im.state, im.variant))
        self._images = images
        self._relayout()

    def _columns(self) -> int:
        cell = int(self.var_thumb.get()) + 16
        return max(1, self.canvas.winfo_width() // cell)

    def _on_canvas_resize(self) -> None:
        cols = self._columns()
        if cols != self._layout_cols:
            # Only the column count changed: move the existing cells instead of rebuilding them.
            self._layout_cols = cols
            for i, cell in enumerate(self._cells):
                cell.grid_configure(row=i // cols, column=i % cols)

    def _relayout(self) -> None:
        self._layout_cols = self._columns()
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._thumb_refs.clear()
        self._cells.clear()
        for i, im in enumerate(self._images):
            self._add_thumb(i, im)
        self.canvas.yview_moveto(0.0)

    def _load(self, im: PreviewImage) -> Image.Image | None:
        """Preview image as PIL RGB, with the defect overlay if enabled (cached per file version)."""
        overlay = bool(self.var_overlay.get() and im.defect and im.nominal)
        try:
            key = (str(im.path), im.path.stat().st_mtime_ns, overlay)
        except OSError:
            return None
        pil = self._pil_cache.get(key)
        if pil is not None:
            return pil
        img = cv2.imread(str(im.path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if overlay:
            try:
                img = draw_defect_overlay(img, im.defect, im.nominal)
            except Exception:  # noqa: BLE001 - overlay is cosmetic
                pass
        pil = Image.fromarray(np.ascontiguousarray(img[..., ::-1]))
        if len(self._pil_cache) >= 4 * MAX_PER_STATE:
            self._pil_cache.clear()
        self._pil_cache[key] = pil
        return pil

    def _add_thumb(self, index: int, im: PreviewImage) -> None:
        size = int(self.var_thumb.get())
        pil = self._load(im)
        if pil is None:  # file removed meanwhile
            return
        resample = Image.Resampling.NEAREST if size >= pil.width * 2 else Image.Resampling.LANCZOS
        thumb = pil.resize((size, round(size * pil.height / pil.width)), resample)
        photo = ImageTk.PhotoImage(thumb)
        self._thumb_refs.append(photo)
        cols = self._layout_cols or self._columns()
        cell = ttk.Frame(self.grid_frame, padding=4)
        pos = len(self._cells)  # not *index*: skipped (unreadable) images leave no gap
        cell.grid(row=pos // cols, column=pos % cols, sticky="n")
        self._cells.append(cell)
        lbl = ttk.Label(cell, image=photo, cursor="hand2")
        lbl.pack()
        lbl.bind("<Button-1>", lambda _e, i=im: self._open_full(i))
        caption = f"{im.state}  #{im.variant}"
        if im.info:
            caption += f"\n{im.info}"
        ttk.Label(cell, text=caption, justify="center", wraplength=size).pack()

    def _open_full(self, im: PreviewImage) -> None:
        path = im.path
        pil = self._load(im)
        if pil is None:
            messagebox.showerror("Profile Preview", f"Cannot read {path}", parent=self.frame)
            return
        top = tk.Toplevel(self.frame)
        top.title(path.name)
        var_zoom = tk.IntVar(value=3 if pil.width <= 300 else 1)
        bar = ttk.Frame(top, padding=4)
        bar.pack(fill="x")
        ttk.Label(bar, text=f"{path.name}  ({pil.width}x{pil.height})  Zoom").pack(side="left")
        label = ttk.Label(top)
        label.pack()

        def show() -> None:
            z = int(var_zoom.get())
            photo = ImageTk.PhotoImage(pil.resize((pil.width * z, pil.height * z), Image.Resampling.NEAREST))
            label.configure(image=photo)
            label.image = photo  # keep a reference

        for z in (1, 2, 3, 4):
            ttk.Radiobutton(bar, text=f"{z}x", value=z, variable=var_zoom, command=show).pack(side="left", padx=2)
        show()

    def _on_wheel(self, event: tk.Event) -> None:
        # Bound application-wide: scroll only when the pointer is over the gallery
        # canvas or one of its descendants (not the log, lists or settings).
        if not self.initialized:
            return
        try:
            w = self.canvas.winfo_containing(event.x_root, event.y_root)
        except Exception:  # noqa: BLE001 - e.g. pointer over a destroyed/foreign window
            return
        while w is not None and w is not self.canvas:
            w = getattr(w, "master", None)
        if w is None:
            return
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")

    def _open_folder(self) -> None:
        out_dir = self.logic.output_dir(self.var_profile.get(), self.var_backend.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener, str(out_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Profile Preview", f"{e}\n{out_dir}", parent=self.frame)

    # ------------------------------------------------------------------ log

    def _log(self, s: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", s.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ lifecycle

    def on_close(self) -> None:
        self._stop.set()
        if self._poll_id is not None:
            try:
                self.frame.after_cancel(self._poll_id)
            except Exception:
                pass
