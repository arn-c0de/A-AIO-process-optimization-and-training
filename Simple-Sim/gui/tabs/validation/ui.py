"""UI for the Validation Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip
from gui.widgets import LogView, ScrolledTree, Section, Toolbar

if TYPE_CHECKING:
    from .tab import ValidationTab


# (check id, label, help key) - ids are the keys ValidationLogic.run_validation expects.
VALIDATION_CHECKS = [
    ("schema", "Schema Validation", "validation.check_schema"),
    ("outliers", "Image Outliers", "validation.check_outliers"),
    ("duplicates", "Duplicate Detection", "validation.check_duplicates"),
    ("edge_cases", "Edge Cases", "validation.check_edge_cases"),
]

_RESULTS_PLACEHOLDER = "No results yet. Choose the checks on the left and press Run Validation."


class ValidationUI:
    def __init__(self, tab: ValidationTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=0)
        self.check_vars: dict[str, tk.BooleanVar] = {}
        self.results_text: tk.Text
        self.results_view: LogView
        self.flag_tree: ttk.Treeview
        self.flag_view: ScrolledTree
        self.lbl_flags_empty: ttk.Label
        self.var_flag_count = tk.StringVar(value="")
        self.var_dataset: tk.StringVar
        self.dataset_combo: ttk.Combobox
        self.btn_run: ttk.Button

    def build_ui(self) -> None:
        tab = self.tab

        # --- Dataset selection + primary action.
        top = Toolbar(self.frame)
        top.pack(fill="x", pady=(0, SPACE.M))
        top.add(ttk.Label(top, text="Dataset", style="Field.TLabel"))
        self.var_dataset = tk.StringVar()
        self.dataset_combo = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=55)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _e: tab._on_dataset_selected())
        ToolTip(self.dataset_combo, help_key="validation.dataset")
        top.add(self.dataset_combo)
        top.icon("refresh", tab._refresh_datasets, help_key="validation.refresh")
        top.separator()
        self.btn_run = top.button("Run Validation", tab._run_validation, help_key="validation.run",
                                  style="Accent.TButton")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        # --- Left: which checks to run, export.
        left = ttk.Frame(main, padding=(0, 0, SPACE.M, 0))
        main.add(left, weight=1)
        checks = Section(left, "Checks", help_key="validation.checks")
        checks.pack(fill="x")
        for test_id, test_name, help_key in VALIDATION_CHECKS:
            var = tk.BooleanVar(value=True)
            self.check_vars[test_id] = var
            cb = ttk.Checkbutton(checks, text=test_name, variable=var, style="Card.TCheckbutton")
            cb.pack(anchor="w", pady=SPACE.XS)
            ToolTip(cb, help_key=help_key)
        ttk.Separator(checks, orient="horizontal").pack(fill="x", pady=SPACE.M)
        btn_export = ttk.Button(checks, text="Export Report", command=tab._export_report)
        btn_export.pack(fill="x")
        ToolTip(btn_export, help_key="validation.export")

        # --- Right: results text + flagged samples.
        right = ttk.Panedwindow(main, orient="vertical")
        main.add(right, weight=3)

        res = Section(right, "Results")
        right.add(res, weight=1)
        self.results_view = LogView(res, height=12, wrap="word")
        self.results_view.pack(fill="both", expand=True)
        self.results_text = self.results_view.text
        self.set_results_text(_RESULTS_PLACEHOLDER)

        flags = Section(right, "Flagged Samples", help_key="validation.flags")
        right.add(flags, weight=1)
        ttk.Label(flags.actions, textvariable=self.var_flag_count, style="CardMuted.TLabel").pack(side="right")
        self.lbl_flags_empty = ttk.Label(
            flags, style="CardMuted.TLabel", wraplength=560, justify="left",
            text="No flagged samples. Image Outliers, Duplicate Detection and Edge Cases add flags here.")
        self.flag_view = ScrolledTree(
            flags, [("Flag", "Flag", 100, "w"), ("Reason", "Reason", 300, "w")],
            show_tree=True, tree_heading="Sample ID", tree_width=200, height=10,
        )
        self.flag_view.pack(fill="both", expand=True)
        self.flag_tree = self.flag_view.tree

        flag_btns = Toolbar(flags, style="Card.TFrame")
        flag_btns.pack(fill="x", pady=(SPACE.S, 0))
        flag_btns.button("Remove Flag", tab._remove_flag, help_key="validation.remove_flag")
        flag_btns.button("Clear All Flags…", tab._clear_all_flags, help_key="validation.clear_flags",
                         style="Danger.TButton")
        self.refresh_flag_tree([])

    def set_results_text(self, text: str) -> None:
        self.results_view.clear()
        self.results_view.append(text)

    def set_running(self, running: bool) -> None:
        try:
            self.btn_run.configure(state="disabled" if running else "normal")
        except Exception:
            pass

    def refresh_flag_tree(self, flagged_samples: list) -> None:
        self.flag_view.clear()
        n = 0
        for sample_id, flags in flagged_samples:
            for flag_dict in flags:
                self.flag_view.insert((flag_dict['flag'], flag_dict['reason']), text=sample_id)
                n += 1
        self.var_flag_count.set(f"{n} flag(s)" if n else "")
        if n:
            self.lbl_flags_empty.pack_forget()
        elif not self.lbl_flags_empty.winfo_manager():
            self.lbl_flags_empty.pack(fill="x", pady=(0, SPACE.S), before=self.flag_view)

    def display_results(self, results: dict) -> None:
        output = "=" * 60 + "\nVALIDATION RESULTS\n" + "=" * 60 + "\n\n"
        if results.get('schema_validation'):
            schema = results['schema_validation']
            status = "✓ PASSED" if schema['passed'] else "✗ FAILED"
            output += f"Schema Validation: {status}\n  {schema['message']}\n\n"

        outliers = results.get('outliers', [])
        output += f"Image Outliers: {len(outliers)} found\n"
        if outliers:
            for outlier in outliers[:10]:
                output += f"  - {outlier['sample_id']}: {outlier['reason']}\n"
            if len(outliers) > 10:
                output += f"  ... and {len(outliers) - 10} more\n"
        output += "\n"

        duplicates = results.get('duplicates', [])
        output += f"Duplicate Images: {len(duplicates)} pairs found\n"
        if duplicates:
            for id1, id2, dist in duplicates[:5]:
                output += f"  - {id1} ↔ {id2} (distance={dist})\n"
            if len(duplicates) > 5:
                output += f"  ... and {len(duplicates) - 5} more\n"
        output += "\n"

        edge_cases = results.get('edge_cases', [])
        output += f"Edge Cases: {len(edge_cases)} found\n"
        if edge_cases:
            for case in edge_cases[:10]:
                output += f"  - {case['sample_id']}: {case['reason']}\n"
            if len(edge_cases) > 10:
                output += f"  ... and {len(edge_cases) - 10} more\n"

        output += "\n" + "=" * 60
        self.set_results_text(output)
