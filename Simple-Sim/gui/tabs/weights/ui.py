"""UI for the Weights Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import os

from gui.utils.tooltip import ToolTip

if TYPE_CHECKING:
    from .tab import WeightsTab


class WeightsUI:
    def __init__(self, tab: WeightsTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)

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
        self.txt: tk.Text
        self.var_status: tk.StringVar

        self.combo_a: ttk.Combobox
        self.combo_b: ttk.Combobox

        self._models_menu: Optional[tk.Menu] = None
        self._reports_menu: Optional[tk.Menu] = None
        self._groups_menu: Optional[tk.Menu] = None

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_dataset, width=60, state="readonly").pack(side="left", padx=(5, 10))
        self.var_dataset_samples = tk.StringVar(value="Samples: -")
        ttk.Label(top, textvariable=self.var_dataset_samples).pack(side="left", padx=(0, 10))

        ttk.Label(top, text="Active model:").pack(side="left")
        self.var_active_model = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_active_model, width=45).pack(side="left", padx=(5, 10))

        ttk.Button(top, text="Refresh", command=self.tab._refresh_models).pack(side="left")
        ttk.Button(top, text="Snapshot active", command=self.tab._snapshot_active).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Import...", command=self.tab._import_model).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Export selected...", command=self.tab._export_selected).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Activate selected", command=self.tab._activate_selected).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Profile Info", command=self.tab._show_model_profile_info).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Delete selected", command=self.tab._delete_selected).pack(side="left", padx=(8, 0))

        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(0, 8))
        self.var_status = tk.StringVar(value="status: idle")
        ttk.Label(status, textvariable=self.var_status).pack(side="left")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=5)
        main.add(left, weight=2)
        ttk.Label(left, text="Available checkpoints").pack(anchor="w")

        self.var_selected_model = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.var_selected_model).pack(anchor="w", pady=(4, 6))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Fav", "Runs", "Size", "MTime", "Path", "Group"),
            show="tree headings",
            selectmode="extended",
        )
        self.tree.heading("#0", text="Model")
        self.tree.column("#0", width=200, anchor="w")
        self.tree.heading("Fav", text="Fav")
        self.tree.heading("Runs", text="Runs")
        self.tree.heading("Size", text="Size")
        self.tree.heading("MTime", text="Modified")
        self.tree.heading("Path", text="Path")
        self.tree.heading("Group", text="Group")
        self.tree.column("Fav", width=50, anchor="center")
        self.tree.column("Runs", width=60, anchor="e")
        self.tree.column("Size", width=80, anchor="e")
        self.tree.column("MTime", width=140, anchor="w")
        self.tree.column("Path", width=400, anchor="w")
        self.tree.column("Group", width=120, anchor="w")
        self.tree.tag_configure("group_header", font=("TkDefaultFont", 10, "bold"))

        scroll_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self.tab._on_tree_select)
        self.tree.bind("<Button-3>", self.tab._on_models_right_click)
        self.tree.bind("<Button-2>", self.tab._on_models_right_click)
        self.tree.bind("<ButtonPress-1>", self.tab._on_tree_mouse_down, add="+")
        self.tree.bind("<ButtonRelease-1>", self.tab._on_tree_mouse_up, add="+")

        self._models_menu = tk.Menu(self.frame, tearoff=0)
        self._models_menu.add_command(label="Run selected", command=self.tab._run_selected)
        self._models_menu.add_command(label="Activate selected", command=self.tab._activate_selected)
        self._models_menu.add_command(label="Export selected...", command=self.tab._export_selected)
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
        self._models_menu.add_command(label="Rename selected...", command=self.tab._rename_selected)
        self._models_menu.add_command(label="Duplicate selected...", command=self.tab._duplicate_selected)
        self._models_menu.add_command(label="Delete selected", command=self.tab._delete_selected)
        self._models_menu.add_separator()
        self._groups_menu = tk.Menu(self._models_menu, tearoff=0)
        self._models_menu.add_cascade(label="Move to group...", menu=self._groups_menu)
        self._models_menu.add_command(label="New group...", command=self.tab._prompt_new_group)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Refresh list", command=self.tab._refresh_models)

        tl = self.frame.winfo_toplevel()
        tl.bind("<Button-1>", self.tab._hide_models_menu, add="+")
        tl.bind("<Escape>", self.tab._hide_models_menu, add="+")

        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)

        runbox = ttk.LabelFrame(right, text="Evaluate (predict.sh)", padding=8)
        runbox.pack(fill="x", pady=(0, 8))

        ttk.Label(runbox, text="Split").grid(row=0, column=0, sticky="w")
        self.var_split = tk.StringVar(value="test")
        ttk.Combobox(runbox, textvariable=self.var_split, values=["train", "val", "test", "all"], state="readonly", width=6).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )

        ttk.Label(runbox, text="Device").grid(row=0, column=2, sticky="w")
        self.var_device = tk.StringVar(value="auto")
        ttk.Combobox(runbox, textvariable=self.var_device, values=["auto", "cpu", "cuda"], state="readonly", width=6).grid(
            row=0, column=3, sticky="w", padx=(6, 12)
        )

        ttk.Label(runbox, text="Max").grid(row=0, column=4, sticky="w")
        self.var_max_samples = tk.StringVar(value="")
        ttk.Entry(runbox, textvariable=self.var_max_samples, width=7).grid(row=0, column=5, sticky="w", padx=(6, 12))

        self.chk_save_preds = tk.BooleanVar(value=True)
        ttk.Checkbutton(runbox, text="Save per-sample preds", variable=self.chk_save_preds).grid(row=0, column=6, sticky="w")

        ttk.Button(runbox, text="Run selected", command=self.tab._run_selected).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(runbox, text="Stop", command=self.tab._stop).grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(6, 0))

        cmpbox = ttk.LabelFrame(right, text="Compare two models", padding=8)
        cmpbox.pack(fill="x", pady=(0, 8))
        cmpbox.columnconfigure(1, weight=1)
        cmpbox.columnconfigure(3, weight=1)

        ttk.Label(cmpbox, text="A").grid(row=0, column=0, sticky="w")
        self.var_cmp_a = tk.StringVar(value="")
        self.combo_a = ttk.Combobox(cmpbox, textvariable=self.var_cmp_a, values=[], state="readonly")
        self.combo_a.grid(row=0, column=1, sticky="ew", padx=(6, 12))

        ttk.Label(cmpbox, text="B").grid(row=0, column=2, sticky="w")
        self.var_cmp_b = tk.StringVar(value="")
        self.combo_b = ttk.Combobox(cmpbox, textvariable=self.var_cmp_b, values=[], state="readonly")
        self.combo_b.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        ttk.Button(cmpbox, text="Run A then B", command=self.tab._run_compare).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        ttk.Label(cmpbox, text="Datasets").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        ds_frame = ttk.Frame(cmpbox)
        ds_frame.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        ds_frame.columnconfigure(0, weight=1)

        self.list_cmp_datasets = tk.Listbox(
            ds_frame,
            height=4,
            selectmode="extended",
            exportselection=False,
        )
        ds_scroll = ttk.Scrollbar(ds_frame, orient="vertical", command=self.list_cmp_datasets.yview)
        self.list_cmp_datasets.configure(yscrollcommand=ds_scroll.set)
        self.list_cmp_datasets.grid(row=0, column=0, sticky="ew")
        ds_scroll.grid(row=0, column=1, sticky="ns")

        ds_btns = ttk.Frame(ds_frame)
        ds_btns.grid(row=0, column=2, sticky="ns", padx=(10, 0))
        ttk.Button(ds_btns, text="Refresh", command=self.tab._refresh_cmp_datasets).pack(fill="x")
        ttk.Button(ds_btns, text="Use current", command=self.tab._select_current_dataset_for_compare).pack(fill="x", pady=(6, 0))
        ttk.Button(ds_btns, text="Select all", command=self.tab._select_all_datasets_for_compare).pack(fill="x", pady=(6, 0))

        repbox = ttk.LabelFrame(right, text="History / Reports", padding=8)
        repbox.pack(fill="both", expand=True, pady=(0, 8))

        repctl = ttk.Frame(repbox)
        repctl.pack(fill="x", pady=(0, 6))

        ttk.Button(repctl, text="Refresh reports", command=self.tab._refresh_reports).pack(side="left")

        ttk.Label(repctl, text="Scope:").pack(side="left", padx=(12, 4))
        self.var_report_scope = tk.StringVar(value="selected_model")
        ttk.Combobox(
            repctl,
            textvariable=self.var_report_scope,
            values=["selected_model", "dataset", "all"],
            state="readonly",
            width=14,
        ).pack(side="left")

        ttk.Label(repctl, text="Split:").pack(side="left", padx=(12, 4))
        self.var_report_split = tk.StringVar(value="any")
        ttk.Combobox(
            repctl,
            textvariable=self.var_report_split,
            values=["any", "train", "val", "test", "all"],
            state="readonly",
            width=6,
        ).pack(side="left")

        ttk.Label(repctl, text="Sort:").pack(side="left", padx=(12, 4))
        self.var_report_sort = tk.StringVar(value="newest")
        ttk.Combobox(
            repctl,
            textvariable=self.var_report_sort,
            values=["newest", "accuracy", "macro_f1"],
            state="readonly",
            width=8,
        ).pack(side="left")

        ttk.Button(repctl, text="Show summary", command=self.tab._show_selected_report_summary).pack(side="left", padx=(12, 0))

        rep_tree_frame = ttk.Frame(repbox)
        rep_tree_frame.pack(fill="both", expand=True)
        rep_tree_frame.columnconfigure(0, weight=1)
        rep_tree_frame.rowconfigure(0, weight=1)

        self.tree_reports = ttk.Treeview(
            rep_tree_frame,
            columns=("When", "Split", "Seen", "Acc", "F1", "Model", "Dataset", "Path"),
            show="headings",
        )
        self.tree_reports.heading("When", text="When")
        self.tree_reports.heading("Split", text="Split")
        self.tree_reports.heading("Seen", text="Seen")
        self.tree_reports.heading("Acc", text="Acc")
        self.tree_reports.heading("F1", text="MacroF1")
        self.tree_reports.heading("Model", text="Model")
        self.tree_reports.heading("Dataset", text="Dataset")
        self.tree_reports.heading("Path", text="Report path")

        self.tree_reports.column("When", width=150, anchor="w")
        self.tree_reports.column("Split", width=55, anchor="center")
        self.tree_reports.column("Seen", width=65, anchor="e")
        self.tree_reports.column("Acc", width=70, anchor="e")
        self.tree_reports.column("F1", width=80, anchor="e")
        self.tree_reports.column("Model", width=120, anchor="w")
        self.tree_reports.column("Dataset", width=120, anchor="w")
        self.tree_reports.column("Path", width=380, anchor="w")

        rep_scroll_y = ttk.Scrollbar(rep_tree_frame, orient="vertical", command=self.tree_reports.yview)
        rep_scroll_x = ttk.Scrollbar(rep_tree_frame, orient="horizontal", command=self.tree_reports.xview)
        self.tree_reports.configure(yscrollcommand=rep_scroll_y.set, xscrollcommand=rep_scroll_x.set)
        self.tree_reports.grid(row=0, column=0, sticky="nsew")
        rep_scroll_y.grid(row=0, column=1, sticky="ns")
        rep_scroll_x.grid(row=1, column=0, sticky="ew")

        self._reports_menu = tk.Menu(self.frame, tearoff=0)
        self._reports_menu.add_command(label="Delete report file", command=self.tab._delete_selected_report)
        self.tree_reports.bind("<Button-3>", self.tab._on_reports_right_click)
        self.tree_reports.bind("<Button-2>", self.tab._on_reports_right_click)

        ttk.Label(right, text="Logs / Output").pack(anchor="w")
        self.txt = tk.Text(right, height=14, wrap="none")
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

    def fmt_bytes(self, n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        v = float(max(0, n))
        u = 0
        while v >= 1024.0 and u < len(units) - 1:
            v /= 1024.0
            u += 1
        if u == 0:
            return f"{int(v)} {units[u]}"
        return f"{v:.1f} {units[u]}"

    def append_log(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def populate_models_tree(self, models: List[Tuple[str, str, str, str, str, str, str]], group_iids: Dict[str, str], groups_to_open: set) -> None:
        self.tree.delete(*self.tree.get_children())
        for group, group_iid in group_iids.items():
            self.tree.insert(
                "",
                "end",
                iid=group_iid,
                text=group,
                values=("", "", "", "", "", ""),
                tags=("group_header",),
                open=group_iid in groups_to_open,
            )
        for parent_iid, iid, display_name, fav_s, runs_s, size_s, mt, rel, group in models:
            self.tree.insert(
                parent_iid,
                "end",
                iid=iid,
                text=display_name,
                values=(fav_s, runs_s, size_s, mt, rel, group),
                tags=("model_entry",),
            )
    
    def set_compare_combobox_values(self, values: List[str]) -> None:
        self.combo_a.configure(values=values)
        self.combo_b.configure(values=values)
        
    def populate_reports_tree(self, reports_data: List[Tuple[str, ...]]) -> None:
        self.tree_reports.delete(*self.tree_reports.get_children())
        for r_values in reports_data:
            self.tree_reports.insert("", "end", iid=r_values[0], values=r_values[1:])
