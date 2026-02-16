"""UI for the Validation Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .tab import ValidationTab


class ValidationUI:
    def __init__(self, tab: ValidationTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)
        self.check_vars: Dict[str, tk.BooleanVar] = {}
        self.results_text: tk.Text
        self.flag_tree: ttk.Treeview
        self.var_dataset: tk.StringVar
        self.dataset_combo: ttk.Combobox

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar()
        self.dataset_combo = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=55)
        self.dataset_combo.pack(side="left", padx=(5, 8))
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self.tab._on_dataset_selected())
        ttk.Button(top, text="Refresh", command=self.tab._refresh_datasets).pack(side="left", padx=(0, 12))

        ttk.Button(top, text="Run Validation Suite", command=self.tab._run_validation).pack(side="left", padx=15)

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=5)
        main.add(left, weight=1)

        ttk.Label(left, text="Validation Tests").pack(anchor="w")

        tests_frame = ttk.Frame(left)
        tests_frame.pack(fill="x", pady=(5, 10))

        tests = [
            ("schema", "Schema Validation"),
            ("outliers", "Image Outliers"),
            ("duplicates", "Duplicate Detection"),
            ("edge_cases", "Edge Cases")
        ]

        for test_id, test_name in tests:
            var = tk.BooleanVar(value=True)
            self.check_vars[test_id] = var
            ttk.Checkbutton(tests_frame, text=test_name, variable=var).pack(anchor="w", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Button(left, text="Export Report", command=self.tab._export_report).pack(fill="x", pady=5)

        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)

        results_label = ttk.Label(right, text="Validation Results")
        results_label.pack(anchor="w")

        results_frame = ttk.Frame(right, relief="sunken", borderwidth=1)
        results_frame.pack(fill="both", expand=True, pady=(5, 10))

        self.results_text = tk.Text(results_frame, height=12, wrap="word")
        results_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scroll.pack(side="right", fill="y")
        self.results_text.configure(state="disabled")

        flags_label = ttk.Label(right, text="Flagged Samples")
        flags_label.pack(anchor="w", pady=(10, 0))

        flags_frame = ttk.Frame(right)
        flags_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.flag_tree = ttk.Treeview(flags_frame, columns=("Flag", "Reason"), show="tree headings", height=10)
        self.flag_tree.heading("#0", text="Sample ID")
        self.flag_tree.column("#0", width=200)
        self.flag_tree.heading("Flag", text="Flag")
        self.flag_tree.column("Flag", width=100)
        self.flag_tree.heading("Reason", text="Reason")
        self.flag_tree.column("Reason", width=300)
        flag_scroll = ttk.Scrollbar(flags_frame, orient="vertical", command=self.flag_tree.yview)
        self.flag_tree.configure(yscrollcommand=flag_scroll.set)
        self.flag_tree.pack(side="left", fill="both", expand=True)
        flag_scroll.pack(side="right", fill="y")

        flag_btns = ttk.Frame(right)
        flag_btns.pack(fill="x", pady=(5, 0))
        ttk.Button(flag_btns, text="Remove Flag", command=self.tab._remove_flag).pack(side="left", padx=2)
        ttk.Button(flag_btns, text="Clear All Flags", command=self.tab._clear_all_flags).pack(side="left", padx=2)
    
    def set_results_text(self, text: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")
        
    def refresh_flag_tree(self, flagged_samples: list) -> None:
        self.flag_tree.delete(*self.flag_tree.get_children())
        for sample_id, flags in flagged_samples:
            for flag_dict in flags:
                self.flag_tree.insert("", "end", text=sample_id, values=(flag_dict['flag'], flag_dict['reason']))

    def display_results(self, results: dict) -> None:
        output = "=" * 60 + "\\nVALIDATION RESULTS\\n" + "=" * 60 + "\\n\\n"
        if results.get('schema_validation'):
            schema = results['schema_validation']
            status = "✓ PASSED" if schema['passed'] else "✗ FAILED"
            output += f"Schema Validation: {status}\\n  {schema['message']}\\n\\n"

        outliers = results.get('outliers', [])
        output += f"Image Outliers: {len(outliers)} found\\n"
        if outliers:
            for outlier in outliers[:10]:
                output += f"  - {outlier['sample_id']}: {outlier['reason']}\\n"
            if len(outliers) > 10:
                output += f"  ... and {len(outliers) - 10} more\\n"
        output += "\\n"

        duplicates = results.get('duplicates', [])
        output += f"Duplicate Images: {len(duplicates)} pairs found\\n"
        if duplicates:
            for id1, id2, dist in duplicates[:5]:
                output += f"  - {id1} ↔ {id2} (distance={dist})\\n"
            if len(duplicates) > 5:
                output += f"  ... and {len(duplicates) - 5} more\\n"
        output += "\\n"

        edge_cases = results.get('edge_cases', [])
        output += f"Edge Cases: {len(edge_cases)} found\\n"
        if edge_cases:
            for case in edge_cases[:10]:
                output += f"  - {case['sample_id']}: {case['reason']}\\n"
            if len(edge_cases) > 10:
                output += f"  ... and {len(edge_cases) - 10} more\\n"

        output += "\\n" + "=" * 60
        self.set_results_text(output)
