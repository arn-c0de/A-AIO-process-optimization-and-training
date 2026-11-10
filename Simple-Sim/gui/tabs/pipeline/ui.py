"""UI for the Pipeline Control Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
from pathlib import Path
from typing import Any
from collections.abc import Callable

from PIL import Image, ImageTk
import cv2

from gui import dialogs
from gui.theme.style import font
from gui.theme.tokens import SPACE
from gui.utils.pipeline_helpers import (
    BOARD_DRC_CHOICES,
    BOARD_ROUTING_CHOICES,
    BOARD_RULES_CHOICES,
    FROM_CONFIG,
    LOOK_CHOICES,
)
from gui.utils.tooltip import ToolTip
from gui.widgets import (
    CollapsibleSection,
    FormGrid,
    LogView,
    PathPicker,
    ScrollableFrame,
    Section,
    Toolbar,
    icon_button,
    keep_pane_visible,
    menu_button,
)
from simple_sim.config import DEFAULT_COMPONENT_PROFILE
from simple_sim.schema import MetaRow
from gui.utils.formatting import fmt_bytes
import contextlib


# Legacy colour names returned by gui.utils.pipeline_helpers -> themed label styles.
_COLOR_TO_STYLE = {
    "green": "CardSuccess.TLabel",
    "blue": "CardAccent.TLabel",
    "orange": "CardWarning.TLabel",
    "red": "CardDanger.TLabel",
    "gray": "CardMuted.TLabel",
    "grey": "CardMuted.TLabel",
    "black": "Card.TLabel",
    "": "Card.TLabel",
}


class PipelineUI:
    def __init__(self, tab: Any, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=0)

        # Stats bar state
        self.var_ds_size: tk.StringVar = tk.StringVar(value="DS: -")
        self.var_ds_samples: tk.StringVar = tk.StringVar(value="Samples: -")
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
        self.var_profile: tk.StringVar = tk.StringVar(value=DEFAULT_COMPONENT_PROFILE)
        self.profile_combo: ttk.Combobox
        self.var_render_backend: tk.StringVar = tk.StringVar(value="opencv_2d")
        self.render_combo: ttk.Combobox
        # render.look / render.board overrides on top of the run config (FROM_CONFIG = keep the config's).
        self.var_look: tk.StringVar = tk.StringVar(value=FROM_CONFIG)
        self.var_board_routing: tk.StringVar = tk.StringVar(value=FROM_CONFIG)
        self.var_board_drc: tk.StringVar = tk.StringVar(value=FROM_CONFIG)
        self.var_board_rules: tk.StringVar = tk.StringVar(value=FROM_CONFIG)
        self.look_combo: ttk.Combobox
        self.board_routing_combo: ttk.Combobox
        self.board_drc_combo: ttk.Combobox
        self.board_rules_combo: ttk.Combobox
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

        self.var_filter_enable_motion_blur: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_filter_motion_blur_strength: tk.StringVar = tk.StringVar(value="1.0")
        self.var_filter_motion_blur_angle: tk.StringVar = tk.StringVar(value="0.0")

        self.var_filter_enable_saturation: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_saturation_factor: tk.StringVar = tk.StringVar(value="1.0")

        self.var_filter_enable_hue_shift: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_filter_hue_shift_deg: tk.StringVar = tk.StringVar(value="0.0")

        self.var_filter_enable_shadow: tk.BooleanVar = tk.BooleanVar(value=False)
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
        
        self._thumb_refs: list[ImageTk.PhotoImage] = []

        # Precise mode
        self.btn_precise_settings: ttk.Button

    def build_ui(self) -> None:
        store = getattr(self.tab.state, "settings_store", None)
        sim_root = getattr(self.tab, "sim_root", None)

        # ------------------------------------------------------------ action bar
        actions = Toolbar(self.frame)
        actions.pack(fill="x")
        self.btn_start = actions.button("Start Pipeline", self.tab.start_pipeline,
                                        style="Accent.TButton", help_key="pipeline.start")
        self.btn_start_generate = actions.button("Generate Only", self.tab.start_pipeline_generate_only,
                                                 help_key="pipeline.generate_only")
        self.btn_stop = actions.button("Stop", self.tab.stop_pipeline, style="Danger.TButton",
                                       help_key="pipeline.stop", state="disabled")
        actions.separator()
        self.btn_filters = actions.button("Image Filters…", self.tab._open_image_filters_popup,
                                          help_key="pipeline.filters")

        stats = ttk.Frame(actions)
        actions.add_right(stats)
        icon_button(stats, "refresh", self.tab._refresh_dataset_stats, help_key="common.refresh").pack(
            side="right", padx=(SPACE.S, 0))
        self.lbl_ds_samples = ttk.Label(stats, textvariable=self.var_ds_samples, style="BodyBold.TLabel")
        self.lbl_ds_samples.pack(side="right", padx=(SPACE.M, 0))
        ttk.Label(stats, textvariable=self.var_ds_size, style="Muted.TLabel").pack(side="right")
        ToolTip(self.lbl_ds_samples, help_key="pipeline.ds_stats")

        # ----------------------------------------------------------- status strip
        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(SPACE.M, 0))
        for var in (self.var_phase, self.var_run_progress, self.var_epoch, self.var_ips):
            ttk.Label(status, textvariable=var, style="Badge.TLabel").pack(side="left", padx=(0, SPACE.S))
        ttk.Label(status, textvariable=self.var_last, style="Small.TLabel").pack(side="left", padx=(SPACE.S, 0))

        # ---------------------------------------------------------------- panes
        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(SPACE.M, 0))

        settings_col = ScrollableFrame(main)
        main.add(settings_col, weight=0)
        keep_pane_visible(main, 0, settings_col.inner, min_width=320)
        col = settings_col.inner
        self._build_source_section(col, sim_root, store)
        self._build_output_section(col, sim_root)
        self._build_run_section(col)
        self._build_model_section(col, store)

        right = ttk.Frame(main)
        main.add(right, weight=1)
        self._build_datasets_section(right, store)

        work = ttk.Panedwindow(right, orient="horizontal")
        work.pack(fill="both", expand=True, pady=(SPACE.M, 0))

        images = Section(work, "Latest Images")
        work.add(images, weight=1)
        img_grid = ttk.Frame(images, style="Card.TFrame")
        img_grid.pack(fill="both", expand=True)
        rows, cols = 4, 3
        for r in range(rows):
            img_grid.grid_rowconfigure(r, weight=1)
            for c in range(cols):
                img_grid.grid_columnconfigure(c, weight=1)
                lbl = ttk.Label(img_grid, text="(no image)", anchor="center", compound="top",
                                style="CardMuted.TLabel", font=font("Small"))
                lbl.grid(row=r, column=c, padx=SPACE.XS, pady=SPACE.XS, sticky="nsew")
                self.img_labels.append(lbl)

        logs = Section(work, "Live Log")
        work.add(logs, weight=2)
        log_actions = logs.actions
        self.log_view = LogView(logs, height=10)
        self.log_view.pack(fill="both", expand=True)
        self.txt = self.log_view.text
        b_clear = ttk.Button(log_actions, text="Clear", style="Icon.TButton", command=self.log_view.clear)
        b_clear.pack(side="right")
        ToolTip(b_clear, help_key="common.clear_log")
        b_copy = ttk.Button(log_actions, text="Copy", style="Icon.TButton", command=self.log_view.copy_all)
        b_copy.pack(side="right", padx=(0, SPACE.S))
        ToolTip(b_copy, help_key="common.copy_log")

    # ------------------------------------------------------------------ sections
    def _build_source_section(self, parent: tk.Misc, sim_root: Path | None, store: Any) -> None:
        sec = Section(parent, "Source", help_key="pipeline.source")
        sec.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(sec, style="Card.TFrame")
        form.pack(fill="x")

        prof = form.inline()
        self.profile_combo = ttk.Combobox(prof, textvariable=self.var_profile, state="readonly", width=22)
        self.profile_combo.pack(side="left", fill="x", expand=True)
        self.profile_combo.bind("<<ComboboxSelected>>", self.tab._on_profile_selected)
        ToolTip(self.profile_combo, help_key="pipeline.profile", text_func=lambda: self.var_profile.get())
        icon_button(prof, "refresh", self.tab._refresh_profiles, help_key="common.refresh").pack(side="left", padx=(SPACE.S, 0))
        icon_button(prof, "info", self.tab._show_profile_info, help_key="pipeline.profile_info").pack(side="left", padx=(SPACE.XS, 0))
        form.row("Profile", prof, help_key="pipeline.profile")

        self.render_combo = ttk.Combobox(form, textvariable=self.var_render_backend,
                                         values=["opencv_2d", "blender_3d"], state="readonly", width=14)
        self.render_combo.bind("<<ComboboxSelected>>", self.tab._on_render_backend_changed)
        form.row("Render", self.render_combo, help_key="pipeline.render", sticky="w")

        self.look_combo = ttk.Combobox(form, textvariable=self.var_look, values=list(LOOK_CHOICES),
                                       state="readonly", width=14)
        form.row("Look", self.look_combo, help_key="pipeline.look", sticky="w")

        board = form.inline()
        self.board_routing_combo = ttk.Combobox(board, textvariable=self.var_board_routing,
                                                values=list(BOARD_ROUTING_CHOICES), state="readonly", width=12)
        self.board_routing_combo.pack(side="left")
        ToolTip(self.board_routing_combo, help_key="pipeline.board_routing")
        ttk.Label(board, text="DRC", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.M, SPACE.XS))
        self.board_drc_combo = ttk.Combobox(board, textvariable=self.var_board_drc, values=list(BOARD_DRC_CHOICES),
                                            state="readonly", width=11)
        self.board_drc_combo.pack(side="left")
        ToolTip(self.board_drc_combo, help_key="pipeline.board_drc")
        ttk.Label(board, text="Rules", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.M, SPACE.XS))
        self.board_rules_combo = ttk.Combobox(board, textvariable=self.var_board_rules, values=list(BOARD_RULES_CHOICES),
                                              state="readonly", width=14)
        self.board_rules_combo.pack(side="left")
        ToolTip(self.board_rules_combo, help_key="pipeline.board_rules")
        form.row("Copper", board, help_key="pipeline.board_routing", sticky="w")

        form.row("Config", PathPicker(form, self.var_config, title="Select run config",
                                      filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*")],
                                      relative_to=sim_root, width=22, style="Card.TFrame"),
                 help_key="pipeline.config")

        multi = CollapsibleSection(sec, "Multi-Profile", store=store, state_key="pipeline.ui.multi_open",
                                   style="Card.TFrame", opened=bool(self.var_profiles_multi.get()))
        multi.pack(fill="x", pady=(SPACE.S, 0))
        mrow = ttk.Frame(multi.body, style="Card.TFrame")
        mrow.pack(fill="x")
        chk = ttk.Checkbutton(mrow, text="Enable", variable=self.var_profiles_multi, style="Card.TCheckbutton",
                              command=self.tab._on_profiles_multi_toggle)
        chk.pack(side="left")
        ToolTip(chk, help_key="pipeline.multi_profile")
        ttk.Label(mrow, text="Mode", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.L, SPACE.S))
        mode = ttk.Combobox(mrow, textvariable=self.var_profiles_multi_mode, values=["separate", "mixed"],
                            state="readonly", width=9)
        mode.pack(side="left")
        ToolTip(mode, help_key="pipeline.multi_mode")
        mrow2 = ttk.Frame(multi.body, style="Card.TFrame")
        mrow2.pack(fill="x", pady=(SPACE.S, 0))
        self.btn_profiles_multi_pick = ttk.Button(mrow2, text="Select…", command=self.tab._pick_profiles_multi,
                                                  state="disabled")
        self.btn_profiles_multi_pick.pack(side="left")
        ToolTip(self.btn_profiles_multi_pick, help_key="pipeline.multi_pick")
        self.entry_profiles_multi = ttk.Entry(mrow2, textvariable=self.var_profiles_multi_summary, state="disabled")
        self.entry_profiles_multi.pack(side="left", fill="x", expand=True, padx=(SPACE.S, 0))

    def _build_output_section(self, parent: tk.Misc, sim_root: Path | None) -> None:
        sec = Section(parent, "Output", help_key="pipeline.output")
        sec.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(sec, style="Card.TFrame")
        form.pack(fill="x")

        mode = form.inline()
        r1 = ttk.Radiobutton(mode, text="Create New", variable=self.var_dataset_mode, value="new", style="Card.TRadiobutton")
        r1.pack(side="left")
        r2 = ttk.Radiobutton(mode, text="Extend Existing", variable=self.var_dataset_mode, value="extend",
                             style="Card.TRadiobutton")
        r2.pack(side="left", padx=(SPACE.M, 0))
        for w in (r1, r2):
            ToolTip(w, help_key="pipeline.dataset_mode")
        form.row("Dataset", mode, help_key="pipeline.dataset_mode", sticky="w")

        form.row("Output", PathPicker(form, self.var_out, mode="dir", title="Select dataset output folder",
                                      relative_to=sim_root, width=22, style="Card.TFrame"),
                 help_key="pipeline.out")
        form.row("Name", ttk.Entry(form, textvariable=self.var_name, width=22), help_key="pipeline.name")

        opts = form.inline()
        c1 = ttk.Checkbutton(opts, text="Timestamp", variable=self.var_name_ts, style="Card.TCheckbutton")
        c1.pack(side="left")
        ToolTip(c1, help_key="pipeline.name_ts")
        c2 = ttk.Checkbutton(opts, text="Bundle", variable=self.var_model_bundle, style="Card.TCheckbutton")
        c2.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(c2, help_key="pipeline.bundle")
        apply_btn = ttk.Button(opts, text="Apply Name", command=self.tab._apply_name_to_out_and_model)
        apply_btn.pack(side="right")
        ToolTip(apply_btn, help_key="pipeline.apply_name")
        form.row("", opts)

    def _build_run_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Run", help_key="pipeline.run")
        sec.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(sec, style="Card.TFrame")
        form.pack(fill="x")

        modes = form.inline()
        for i, (text, value) in enumerate((("Single", "single"), ("Multiple", "multiple"),
                                           ("Continuous", "continuous"), ("Precise", "precise"))):
            rb = ttk.Radiobutton(modes, text=text, variable=self.var_run_mode, value=value, style="Card.TRadiobutton")
            rb.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, SPACE.L), pady=(0, SPACE.XS))
            ToolTip(rb, help_key="pipeline.run_mode")
        form.row("Mode", modes, help_key="pipeline.run_mode", sticky="w")

        cnt = form.inline()
        self.run_count_entry = ttk.Spinbox(cnt, textvariable=self.var_run_count, from_=1, to=10000, width=6)
        self.run_count_entry.pack(side="left")
        self.btn_precise_settings = ttk.Button(cnt, text="Precise Settings…", command=self.tab._open_precise_popup,
                                               state="disabled")
        self.btn_precise_settings.pack(side="right")
        ToolTip(self.btn_precise_settings, help_key="pipeline.precise")
        form.row("Count", cnt, help_key="pipeline.run_count")

        def on_run_mode_change(*_args: Any) -> None:
            mode = self.var_run_mode.get()
            self.run_count_entry.configure(state="normal" if mode == "multiple" else "disabled")
            self.btn_precise_settings.configure(state="normal" if mode == "precise" else "disabled")

        self.var_run_mode.trace_add("write", on_run_mode_change)
        on_run_mode_change()

    def _build_model_section(self, parent: tk.Misc, store: Any) -> None:
        sec = Section(parent, "Model", help_key="pipeline.model")
        sec.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(sec, style="Card.TFrame")
        form.pack(fill="x")
        mrow = form.inline()
        self.model_combo = ttk.Combobox(mrow, textvariable=self.var_model, state="readonly", width=22)
        self.model_combo.pack(side="left", fill="x", expand=True)
        ToolTip(self.model_combo, help_key="pipeline.model", text_func=lambda: self.var_model.get())
        icon_button(mrow, "refresh", self.tab._refresh_models, help_key="common.refresh").pack(side="left", padx=(SPACE.S, 0))
        icon_button(mrow, "add", self.tab._add_new_model, help_key="pipeline.model_add").pack(side="left", padx=(SPACE.XS, 0))
        form.row("Checkpoint", mrow, help_key="pipeline.model")

        snap = CollapsibleSection(sec, "Auto-Snapshot", store=store, state_key="pipeline.ui.snap_open",
                                  style="Card.TFrame")
        snap.pack(fill="x", pady=(SPACE.S, 0))
        srow = ttk.Frame(snap.body, style="Card.TFrame")
        srow.pack(fill="x")
        c = ttk.Checkbutton(srow, text="Enable", variable=self.var_autosnap, style="Card.TCheckbutton")
        c.pack(side="left")
        ToolTip(c, help_key="pipeline.autosnap")
        ttk.Label(srow, text="Every", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.L, SPACE.S))
        every = ttk.Combobox(srow, textvariable=self.var_snap_every, values=["1", "5", "10", "30", "50"],
                             state="readonly", width=4)
        every.pack(side="left")
        ToolTip(every, help_key="pipeline.snap_every")
        ttk.Label(srow, text="runs, keep", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.S, SPACE.S))
        keep = ttk.Spinbox(srow, textvariable=self.var_snap_keep, from_=1, to=999, width=4)
        keep.pack(side="left")
        ToolTip(keep, help_key="pipeline.snap_keep")

    def _build_datasets_section(self, parent: tk.Misc, store: Any) -> None:
        sec = Section(parent, "Datasets", help_key="pipeline.datasets")
        sec.pack(fill="x")

        sel = ttk.Frame(sec, style="Card.TFrame")
        sel.pack(fill="x")
        self._ds_sel = ttk.Frame(sel, style="Card.TFrame")
        self._ds_sel.pack(side="left", fill="x", expand=True)
        self.dataset_combo = ttk.Combobox(self._ds_sel, textvariable=self.var_dataset, state="readonly", width=38)
        self.dataset_combo.pack(side="left", fill="x", expand=True)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.tab._on_dataset_selected)
        ToolTip(self.dataset_combo, help_key="common.dataset", text_func=self.tab._dataset_multi_tooltip_text)
        self.dataset_multi_entry = ttk.Entry(self._ds_sel, textvariable=self.var_dataset_multi_summary,
                                             state="readonly", width=38)
        ToolTip(self.dataset_multi_entry, text_func=self.tab._dataset_multi_tooltip_text)
        self.btn_dataset_multi_pick = ttk.Button(self._ds_sel, text="Select…",
                                                 command=self.tab._open_dataset_multi_picker)
        ToolTip(self.btn_dataset_multi_pick, help_key="pipeline.dataset_multi_pick")
        self.chk_dataset_multi = ttk.Checkbutton(sel, text="Multi", variable=self.var_dataset_multi,
                                                 command=self.tab._on_dataset_multi_toggle, style="Card.TCheckbutton")
        self.chk_dataset_multi.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(self.chk_dataset_multi, help_key="pipeline.dataset_multi")
        icon_button(sel, "refresh", self.tab._refresh_datasets, help_key="common.refresh").pack(side="left", padx=(SPACE.S, 0))

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(SPACE.M, 0))
        bar.button("Validate", self.tab._validate_selected, help_key="pipeline.validate")
        bar.button("Train", self.tab._train_selected, help_key="pipeline.train")
        bar.button("Evaluate", self.tab._eval_selected, help_key="pipeline.eval")
        bar.add_right(menu_button(bar, "Manage", [
            ("Snapshot", self.tab._snapshot_dataset_selected),
            ("Rename…", self.tab._rename_dataset_selected),
            ("Create New…", self.tab._create_new_dataset),
            None,
            ("Delete Images…", self.tab._delete_images_from_dataset),
            ("Move Images…", self.tab._move_images_between_datasets),
            None,
            ("Delete Dataset…", self.tab._delete_dataset),
        ], help_key="pipeline.datasets_manage"))

        info = ttk.Frame(sec, style="Card.TFrame")
        info.pack(fill="x", pady=(SPACE.M, 0))
        ttk.Label(info, textvariable=self.var_dataset_profile, style="CardMuted.TLabel").pack(side="left")
        b = ttk.Button(info, text="Profile Info…", command=self.tab._show_dataset_profile_info)
        b.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(b, help_key="pipeline.dataset_profile_info")
        self.lbl_profile_compat = ttk.Label(info, textvariable=self.var_profile_compat, style="CardWarning.TLabel")
        self.lbl_profile_compat.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(self.lbl_profile_compat, help_key="pipeline.profile_compat")

        cont = CollapsibleSection(sec, "Continue Training", store=store, state_key="pipeline.ui.continue_open",
                                  style="Card.TFrame")
        cont.pack(fill="x", pady=(SPACE.S, 0))
        crow = ttk.Frame(cont.body, style="Card.TFrame")
        crow.pack(fill="x")
        ttk.Label(crow, text="+Epochs", style="CardMuted.TLabel").pack(side="left", padx=(0, SPACE.S))
        ep = ttk.Spinbox(crow, textvariable=self.var_continue_epochs, from_=0, to=10000, width=5)
        ep.pack(side="left")
        ToolTip(ep, help_key="pipeline.continue_epochs")
        ttk.Label(crow, text="From", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.L, SPACE.S))
        frm = ttk.Combobox(crow, textvariable=self.var_continue_out_mode, values=["last", "best"], state="readonly", width=6)
        frm.pack(side="left")
        ToolTip(frm, help_key="pipeline.continue_out")
        btn = ttk.Button(crow, text="Continue Training", command=self.tab._continue_train_selected)
        btn.pack(side="left", padx=(SPACE.L, 0))
        ToolTip(btn, help_key="pipeline.continue_train")

        pm = CollapsibleSection(sec, "Profile Model", store=store, state_key="pipeline.ui.profile_model_open",
                                style="Card.TFrame")
        pm.pack(fill="x")
        prow = ttk.Frame(pm.body, style="Card.TFrame")
        prow.pack(fill="x")
        self.combo_profile_model = ttk.Combobox(prow, textvariable=self.var_profile_model, state="readonly", width=36)
        self.combo_profile_model.pack(side="left", fill="x", expand=True)
        ToolTip(self.combo_profile_model, help_key="pipeline.profile_model", text_func=lambda: self.var_profile_model.get())
        icon_button(prow, "refresh", self.tab._refresh_profile_models, help_key="common.refresh").pack(
            side="left", padx=(SPACE.S, 0))
        lock = ttk.Checkbutton(prow, text="Lock", variable=self.var_profile_model_lock,
                               command=self.tab._apply_profile_model_lock, style="Card.TCheckbutton")
        lock.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(lock, help_key="pipeline.profile_model_lock")
        prow2 = ttk.Frame(pm.body, style="Card.TFrame")
        prow2.pack(fill="x", pady=(SPACE.S, 0))
        ttk.Label(prow2, text="Preset", style="CardMuted.TLabel").pack(side="left", padx=(0, SPACE.S))
        pre = ttk.Combobox(prow2, textvariable=self.var_profile_build_preset, values=["quick", "full"],
                           state="readonly", width=7)
        pre.pack(side="left")
        ToolTip(pre, help_key="pipeline.build_preset")
        bb = ttk.Button(prow2, text="Build Profile Model", command=self.tab._build_profile_model)
        bb.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(bb, help_key="pipeline.build_profile_model")

    def append_log(self, s: str) -> None:
        self.log_view.append(s)

    def set_compat_color(self, color: str) -> None:
        """Map legacy colour names from the helpers to theme styles."""
        self.lbl_profile_compat.configure(style=_COLOR_TO_STYLE.get(str(color).lower(), "CardMuted.TLabel"))

    def set_run_buttons_state(self, running: bool) -> None:
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_start_generate.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")
        self.btn_filters.configure(state="disabled" if running else "normal")

    fmt_bytes = staticmethod(fmt_bytes)

    def refresh_thumbnails(self, recent_imgs: list[str], dataset_dir: Path | None, label_dict: dict[str, str], image_to_sample: dict[str, str], meta_by_sample: dict[str, MetaRow], meta_by_image_rel: dict[str, MetaRow], draw_defect_overlay_func: Callable[..., Any]) -> None:
        """Synchronous decode + show (the tab uses ``prepare_thumbnails`` in a worker instead)."""
        self.show_thumbnails(self.prepare_thumbnails(recent_imgs, dataset_dir, label_dict, image_to_sample,
                                                     meta_by_sample, meta_by_image_rel, draw_defect_overlay_func))

    def prepare_thumbnails(self, recent_imgs: list[str], dataset_dir: Path | None, label_dict: dict[str, str], image_to_sample: dict[str, str], meta_by_sample: dict[str, MetaRow], meta_by_image_rel: dict[str, MetaRow], draw_defect_overlay_func: Callable[..., Any]) -> list[tuple[Image.Image | None, str]]:
        """Tk-free part (safe on a worker thread): one ``(PIL thumbnail or None, caption)`` per slot."""
        return prepare_thumbnail_cells(len(self.img_labels), recent_imgs, dataset_dir, label_dict, image_to_sample,
                                       meta_by_sample, meta_by_image_rel, draw_defect_overlay_func)

    def show_thumbnails(self, prepared: list[tuple[Image.Image | None, str]]) -> None:
        """Tk part: PhotoImages must be created on the Tk thread."""
        self._thumb_refs.clear()
        for lbl, (img_pil, text) in zip(self.img_labels, prepared):
            if img_pil is None:
                lbl.configure(image="", text=text, compound="top")
                continue
            try:
                tkimg = ImageTk.PhotoImage(img_pil)
            except Exception:
                lbl.configure(image="", text=f"(failed) {text.splitlines()[0] if text else ''}", compound="top")
                continue
            self._thumb_refs.append(tkimg)
            lbl.configure(image=tkimg, text=text, compound="top")
            lbl.image = tkimg

    def set_dataset_samples_info(self, text: str, color: str) -> None:
        self.var_ds_samples.set(text)
        style = _COLOR_TO_STYLE.get(str(color).lower(), "Card.TLabel").replace("Card", "", 1) or "TLabel"
        self.lbl_ds_samples.configure(style="BodyBold.TLabel" if style in ("TLabel", ".TLabel") else style)

    def set_profile_model_combo_values(self, values: list[str]) -> None:
        self.combo_profile_model["values"] = values

    def set_profile_model_combo_state(self, state: str) -> None:
        self.combo_profile_model.configure(state=state)

    def set_profiles_multi_state(self, enabled: bool) -> None:
        self.profile_combo.configure(state="disabled" if enabled else "readonly")
        self.btn_profiles_multi_pick.configure(state="normal" if enabled else "disabled")
        self.entry_profiles_multi.configure(state="normal" if enabled else "disabled")

    def set_profile_combo_values(self, values: list[str]) -> None:
        self.profile_combo["values"] = values

    def set_model_combo_values(self, values: list[str]) -> None:
        self.model_combo["values"] = values

    def show_messagebox(self, type: str, title: str, message: str) -> None:
        if type == "error":
            with contextlib.suppress(Exception):
                self.tab._append_log(f"[popup error] {title}\n{message}\n")
            dialogs.show_error(self.frame, title, str(message))
        elif type == "warning":
            dialogs.show_warning(self.frame, title, message)
        else:
            dialogs.show_info(self.frame, title, message)

    def ask_yes_no(self, title: str, message: str) -> bool:
        return dialogs.confirm(self.frame, title, message)

    def ask_string(self, title: str, prompt: str, initialvalue: str = "") -> str | None:
        return simpledialog.askstring(title, prompt, initialvalue=initialvalue, parent=self.frame)

    def open_file_dialog(self, title: str, filetypes: list[tuple[str, str]], initialdir: str) -> str | None:
        return filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initialdir)

    def open_dir_dialog(self, title: str, initialdir: str) -> str | None:
        return filedialog.askdirectory(title=title, initialdir=initialdir)
    
    def set_dataset_combo_values(self, values: list[str]) -> None:
        self.dataset_combo["values"] = values

    def toggle_dataset_multi_widgets(self, enabled: bool) -> None:
        if enabled:
            self.dataset_combo.pack_forget()
            self.dataset_multi_entry.pack(side="left", fill="x", expand=True)
            self.btn_dataset_multi_pick.pack(side="left", padx=(8, 0))
        else:
            self.dataset_multi_entry.pack_forget()
            self.btn_dataset_multi_pick.pack_forget()
            self.dataset_combo.pack(side="left", fill="x", expand=True)


def prepare_thumbnail_cells(n_slots: int, recent_imgs: list[str], dataset_dir: Path | None, label_dict: dict[str, str],
                            image_to_sample: dict[str, str], meta_by_sample: dict[str, MetaRow],
                            meta_by_image_rel: dict[str, MetaRow],
                            draw_defect_overlay_func: Callable[..., Any]) -> list[tuple[Image.Image | None, str]]:
    """Decode the newest ``n_slots`` images (newest first), draw the defect overlay, shrink to 120x90.

    Tk-free: runs on a worker thread. A slot without image yields ``(None, "(no image)")``, an
    unreadable one ``(None, "(failed) <name>")`` - the captions the label showed before.
    """
    out: list[tuple[Image.Image | None, str]] = []
    for i in range(n_slots):
        if i >= len(recent_imgs):
            out.append((None, "(no image)"))
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
                with contextlib.suppress(Exception): img_bgr = draw_defect_overlay_func(img_bgr, meta_row.defect, meta_row.nominal)

            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            img_pil.thumbnail((120, 90))
            out.append((img_pil, path.name if not class_label else f"{path.name}\n[{class_label}]"))
        except Exception:
            out.append((None, f"(failed) {path.name}"))
    return out
