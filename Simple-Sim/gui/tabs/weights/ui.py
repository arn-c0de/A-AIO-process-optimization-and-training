"""UI for the Weights Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from gui.theme.style import font
from gui.theme.tokens import SPACE
from gui.utils.formatting import fmt_bytes
from gui.utils.tooltip import ToolTip
from gui.widgets import FormGrid, LogView, ScrolledTree, Section, Toolbar

if TYPE_CHECKING:
    from .tab import WeightsTab


class WeightsUI:
    def __init__(self, tab: WeightsTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=0)

        # UI components
        self.var_dataset: tk.StringVar
        self.var_active_model: tk.StringVar
        self.var_selected_model: tk.StringVar
        self.var_dataset_samples: tk.StringVar

        self.var_split: tk.StringVar
        self.var_device: tk.StringVar
        self.var_max_samples: tk.StringVar
        self.chk_save_preds: tk.BooleanVar

        self.var_cmp_a: tk.StringVar
        self.var_cmp_b: tk.StringVar
        self.var_report_scope: tk.StringVar
        self.var_report_split: tk.StringVar
        self.var_report_sort: tk.StringVar

        self.list_cmp_datasets: tk.Listbox

        self.tree: ttk.Treeview
        self.tree_reports: ttk.Treeview
        self.models_view: ScrolledTree
        self.reports_view: ScrolledTree
        self.lbl_reports_empty: ttk.Label
        self.log_view: LogView
        self.txt: tk.Text
        self.var_status: tk.StringVar

        self.combo_a: ttk.Combobox
        self.combo_b: ttk.Combobox

        self._models_menu: tk.Menu | None = None
        self._reports_menu: tk.Menu | None = None
        self._groups_menu: tk.Menu | None = None

    def build_ui(self) -> None:
        self._build_toolbar()
        self._build_context()

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(SPACE.M, 0))

        left = ttk.Frame(main)
        main.add(left, weight=2)
        self._build_models_section(left)

        right = ttk.Frame(main)
        main.add(right, weight=3)
        self._build_evaluate_section(right)
        self._build_compare_section(right)
        self._build_reports_section(right)
        self._build_log_section(right)

        tl = self.frame.winfo_toplevel()
        tl.bind("<Button-1>", self.tab._hide_models_menu, add="+")
        tl.bind("<Escape>", self.tab._hide_models_menu, add="+")

    # ------------------------------------------------------------------ sections
    def _build_toolbar(self) -> None:
        bar = Toolbar(self.frame)
        bar.pack(fill="x")
        bar.icon("refresh", self.tab._refresh_models, help_key="weights.refresh")
        bar.button("Snapshot Active", self.tab._snapshot_active, help_key="weights.snapshot")
        bar.button("Import…", self.tab._import_model, help_key="weights.import")
        bar.button("Export Selected…", self.tab._export_selected, help_key="weights.export")
        bar.button("Activate Selected", self.tab._activate_selected, help_key="weights.activate")
        bar.button("Profile Info…", self.tab._show_model_profile_info, help_key="weights.profile_info")
        bar.separator()
        bar.button("Delete Selected", self.tab._delete_selected, style="Danger.TButton", help_key="weights.delete")

        self.var_status = tk.StringVar(value="status: idle")
        status = ttk.Label(bar, textvariable=self.var_status, style="Badge.TLabel")
        bar.add_right(status)
        ToolTip(status, help_key="weights.status")

    def _build_context(self) -> None:
        form = FormGrid(self.frame)
        form.pack(fill="x", pady=(SPACE.M, 0))

        self.var_dataset = tk.StringVar(value="")
        self.var_dataset_samples = tk.StringVar(value="Samples: -")
        ds = form.inline()
        ds_entry = ttk.Entry(ds, textvariable=self.var_dataset, width=60, state="readonly")
        ds_entry.pack(side="left", fill="x", expand=True)
        ToolTip(ds_entry, help_key="weights.dataset", text_func=lambda: self.var_dataset.get())
        ttk.Label(ds, textvariable=self.var_dataset_samples, style="Muted.TLabel").pack(side="left", padx=(SPACE.M, 0))
        form.row("Dataset", ds, help_key="weights.dataset")

        self.var_active_model = tk.StringVar(value="")
        active = ttk.Entry(form, textvariable=self.var_active_model, width=60)
        form.row("Active Model", active, help_key="weights.active_model")
        ToolTip(active, text_func=lambda: self.var_active_model.get())

    def _build_models_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Available Checkpoints", help_key="weights.checkpoints")
        sec.pack(fill="both", expand=True)

        self.var_selected_model = tk.StringVar(value="")
        ttk.Label(sec, textvariable=self.var_selected_model, style="CardMuted.TLabel", wraplength=420,
                  justify="left").pack(anchor="w", pady=(0, SPACE.S))

        self.models_view = ScrolledTree(
            sec,
            columns=[
                ("Fav", "Fav", 50, "center"),
                ("Runs", "Runs", 60, "e"),
                ("Size", "Size", 80, "e"),
                ("MTime", "Modified", 140, "w"),
                ("Path", "Path", 400, "w"),
                ("Group", "Group", 120, "w"),
            ],
            show_tree=True,
            tree_heading="Model",
            tree_width=200,
            selectmode="extended",
            hscroll=True,
            sortable=False,  # rows are grouped; order is favorites first, newest first
            striped=False,
        )
        self.models_view.pack(fill="both", expand=True)
        self.tree = self.models_view.tree
        self.tree.tag_configure("group_header", font=font("BodyBold"))
        ToolTip(self.tree, help_key="weights.tree")

        self.tree.bind("<<TreeviewSelect>>", self.tab._on_tree_select)
        self.tree.bind("<Button-3>", self.tab._on_models_right_click)
        self.tree.bind("<Button-2>", self.tab._on_models_right_click)
        self.tree.bind("<ButtonPress-1>", self.tab._on_tree_mouse_down, add="+")
        self.tree.bind("<ButtonRelease-1>", self.tab._on_tree_mouse_up, add="+")

        # Entry labels are looked up by tab._on_models_right_click (entryconfigure) - keep them in sync.
        self._models_menu = tk.Menu(self.frame, tearoff=0)
        self._models_menu.add_command(label="Run Selected", command=self.tab._run_selected)
        self._models_menu.add_command(label="Activate Selected", command=self.tab._activate_selected)
        self._models_menu.add_command(label="Export Selected…", command=self.tab._export_selected)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Add to Favorites", command=self.tab._add_selected_to_favorites)
        self._models_menu.add_command(label="Remove from Favorites", command=self.tab._remove_selected_from_favorites)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Add to Arena", command=self.tab._add_selected_to_arena)
        self._models_menu.add_command(label="Remove from Arena", command=self.tab._remove_selected_from_arena)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Generate Arena Report", command=self.tab._generate_arena_report)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Set as Compare A", command=self.tab._set_selected_as_compare_a)
        self._models_menu.add_command(label="Set as Compare B", command=self.tab._set_selected_as_compare_b)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Rename Selected…", command=self.tab._rename_selected)
        self._models_menu.add_command(label="Duplicate Selected…", command=self.tab._duplicate_selected)
        self._models_menu.add_command(label="Delete Selected", command=self.tab._delete_selected)
        self._models_menu.add_separator()
        self._groups_menu = tk.Menu(self._models_menu, tearoff=0)
        self._models_menu.add_cascade(label="Move to Group", menu=self._groups_menu)
        self._models_menu.add_command(label="New Group…", command=self.tab._prompt_new_group)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Refresh List", command=self.tab._refresh_models)

        ttk.Label(sec, text="Right-click for more actions; drag a checkpoint onto a group to move it.",
                  style="CardMuted.TLabel").pack(anchor="w", pady=(SPACE.S, 0))

    def _build_evaluate_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Evaluate (predict.sh)", help_key="weights.evaluate_section")
        sec.pack(fill="x", pady=(0, SPACE.M))

        opts = ttk.Frame(sec, style="Card.TFrame")
        opts.pack(fill="x")
        self.var_split = tk.StringVar(value="test")
        self.var_device = tk.StringVar(value="auto")
        self.var_max_samples = tk.StringVar(value="")
        self.chk_save_preds = tk.BooleanVar(value=True)

        ttk.Label(opts, text="Split", style="CardMuted.TLabel").pack(side="left", padx=(0, SPACE.S))
        split = ttk.Combobox(opts, textvariable=self.var_split, values=["train", "val", "test", "all"],
                             state="readonly", width=6)
        split.pack(side="left")
        ToolTip(split, help_key="common.split")

        ttk.Label(opts, text="Device", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.L, SPACE.S))
        device = ttk.Combobox(opts, textvariable=self.var_device, values=["auto", "cpu", "cuda"],
                              state="readonly", width=6)
        device.pack(side="left")
        ToolTip(device, help_key="common.device")

        ttk.Label(opts, text="Max", style="CardMuted.TLabel").pack(side="left", padx=(SPACE.L, SPACE.S))
        max_entry = ttk.Entry(opts, textvariable=self.var_max_samples, width=7)
        max_entry.pack(side="left")
        ToolTip(max_entry, help_key="weights.max_samples")

        chk = ttk.Checkbutton(opts, text="Save per-sample predictions", variable=self.chk_save_preds,
                              style="Card.TCheckbutton")
        chk.pack(side="left", padx=(SPACE.L, 0))
        ToolTip(chk, help_key="weights.save_preds")

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(SPACE.M, 0))
        bar.button("Run Selected", self.tab._run_selected, style="Accent.TButton", help_key="weights.evaluate")
        bar.button("Stop", self.tab._stop, help_key="weights.stop")

    def _build_compare_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Compare Two Models", help_key="weights.compare_section")
        sec.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(sec, style="Card.TFrame")
        form.pack(fill="x")

        self.var_cmp_a = tk.StringVar(value="")
        self.combo_a = ttk.Combobox(form, textvariable=self.var_cmp_a, values=[], state="readonly")
        form.row("Model A", self.combo_a, help_key="weights.compare_a")
        ToolTip(self.combo_a, text_func=lambda: self.var_cmp_a.get())

        self.var_cmp_b = tk.StringVar(value="")
        self.combo_b = ttk.Combobox(form, textvariable=self.var_cmp_b, values=[], state="readonly")
        form.row("Model B", self.combo_b, help_key="weights.compare_b")
        ToolTip(self.combo_b, text_func=lambda: self.var_cmp_b.get())

        ds_frame = form.inline("Card.TFrame")
        ds_frame.columnconfigure(0, weight=1)
        self.list_cmp_datasets = tk.Listbox(ds_frame, height=4, selectmode="extended", exportselection=False)
        ds_scroll = ttk.Scrollbar(ds_frame, orient="vertical", command=self.list_cmp_datasets.yview)
        self.list_cmp_datasets.configure(yscrollcommand=ds_scroll.set)
        self.list_cmp_datasets.grid(row=0, column=0, sticky="ew")
        ds_scroll.grid(row=0, column=1, sticky="ns")
        ToolTip(self.list_cmp_datasets, help_key="weights.compare_datasets")

        ds_btns = ttk.Frame(ds_frame, style="Card.TFrame")
        ds_btns.grid(row=0, column=2, sticky="ns", padx=(SPACE.M, 0))
        for i, (text, cmd, key) in enumerate((
            ("Refresh", self.tab._refresh_cmp_datasets, "weights.compare_refresh"),
            ("Use Current", self.tab._select_current_dataset_for_compare, "weights.compare_use_current"),
            ("Select All", self.tab._select_all_datasets_for_compare, "weights.compare_select_all"),
        )):
            b = ttk.Button(ds_btns, text=text, command=cmd)
            b.pack(fill="x", pady=(SPACE.S if i else 0, 0))
            ToolTip(b, help_key=key)
        form.row("Datasets", ds_frame, help_key="weights.compare_datasets")

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(SPACE.M, 0))
        bar.button("Run A Then B", self.tab._run_compare, help_key="weights.compare")

    def _build_reports_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "History / Reports", help_key="weights.reports_section")
        sec.pack(fill="both", expand=True, pady=(0, SPACE.M))

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(0, SPACE.S))
        bar.icon("refresh", self.tab._refresh_reports, help_key="weights.reports_refresh")

        self.var_report_scope = tk.StringVar(value="selected_model")
        self.var_report_split = tk.StringVar(value="any")
        self.var_report_sort = tk.StringVar(value="newest")
        for label, var, values, width, key in (
            ("Scope", self.var_report_scope, ["selected_model", "dataset", "all"], 14, "weights.report_scope"),
            ("Split", self.var_report_split, ["any", "train", "val", "test", "all"], 6, "weights.report_split"),
            ("Sort", self.var_report_sort, ["newest", "accuracy", "macro_f1"], 8, "weights.report_sort"),
        ):
            bar.add(ttk.Label(bar, text=label, style="CardMuted.TLabel"))
            combo = ttk.Combobox(bar, textvariable=var, values=values, state="readonly", width=width)
            bar.add(combo)
            combo.bind("<<ComboboxSelected>>", lambda _e: self.tab._refresh_reports())
            ToolTip(combo, help_key=key)
        bar.button("Show Summary", self.tab._show_selected_report_summary, help_key="weights.report_summary",
                   right=True)

        self.lbl_reports_empty = ttk.Label(
            sec, text="No reports match this filter. Run an evaluation or change Scope / Split.",
            style="CardMuted.TLabel")

        self.reports_view = ScrolledTree(
            sec,
            columns=[
                ("When", "When", 150, "w"),
                ("Split", "Split", 55, "center"),
                ("Seen", "Seen", 65, "e"),
                ("Acc", "Acc", 70, "e"),
                ("F1", "MacroF1", 80, "e"),
                ("Model", "Model", 120, "w"),
                ("Dataset", "Dataset", 120, "w"),
                ("Path", "Report path", 380, "w"),
            ],
            hscroll=True,
            height=8,
        )
        self.reports_view.pack(fill="both", expand=True)
        self.tree_reports = self.reports_view.tree

        self._reports_menu = tk.Menu(self.frame, tearoff=0)
        self._reports_menu.add_command(label="Delete Report File", command=self.tab._delete_selected_report)
        self.tree_reports.bind("<Button-3>", self.tab._on_reports_right_click)
        self.tree_reports.bind("<Button-2>", self.tab._on_reports_right_click)
        self.tree_reports.bind("<Double-1>", lambda _e: self.tab._show_selected_report_summary())

    def _build_log_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Logs / Output")
        sec.pack(fill="both", expand=True)
        self.log_view = LogView(sec, height=14)
        self.log_view.pack(fill="both", expand=True)
        self.txt = self.log_view.text
        b_clear = ttk.Button(sec.actions, text="Clear", style="Icon.TButton", command=self.log_view.clear)
        b_clear.pack(side="right")
        ToolTip(b_clear, help_key="common.clear_log")
        b_copy = ttk.Button(sec.actions, text="Copy", style="Icon.TButton", command=self.log_view.copy_all)
        b_copy.pack(side="right", padx=(0, SPACE.S))
        ToolTip(b_copy, help_key="common.copy_log")

    fmt_bytes = staticmethod(fmt_bytes)

    def append_log(self, s: str) -> None:
        self.log_view.append(s)

    def set_compare_combobox_values(self, values: list[str]) -> None:
        self.combo_a.configure(values=values)
        self.combo_b.configure(values=values)

    def populate_reports_tree(self, reports_data: list[tuple[str, ...]]) -> None:
        self.reports_view.clear()
        for r_values in reports_data:
            self.reports_view.insert(r_values[1:], iid=r_values[0])
        if reports_data:
            self.lbl_reports_empty.pack_forget()
        elif not self.lbl_reports_empty.winfo_manager():
            self.lbl_reports_empty.pack(anchor="w", pady=(0, SPACE.S), before=self.reports_view)
