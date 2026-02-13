"""Validation Tab - Dataset quality testing and flagging."""

from __future__ import annotations
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import threading
import os

from .base_tab import BaseTab
from gui.state import UiState
from gui.utils.validation_suite import run_validation_suite, detect_image_outliers
from gui.utils.flag_manager import FlagManager
from gui.components.chart_widgets import create_confusion_matrix_widget

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_sim.schema import read_jsonl, MetaRow, LabelRow


class ValidationTab(BaseTab):
    """Tab 3: Advanced dataset validation and flagging."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.flag_manager: Optional[FlagManager] = None
        self.validation_results: Optional[dict] = None

        # UI components
        self.var_dataset: tk.StringVar
        self.dataset_combo: ttk.Combobox
        self._dataset_dirs: list[Path] = []
        self._dataset_labels: list[str] = []
        self._dataset_by_label: dict[str, Path] = {}
        self.check_vars: dict[str, tk.BooleanVar] = {}
        self.results_text: tk.Text
        self.flag_tree: ttk.Treeview

    def build_ui(self) -> None:
        """Build the validation UI."""
        # Note: Don't pack self.frame - it's managed by the notebook

        # Top controls
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar()
        self.dataset_combo = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=55)
        self.dataset_combo.pack(side="left", padx=(5, 8))
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_dataset_selected())
        ttk.Button(top, text="Refresh", command=self._refresh_datasets).pack(side="left", padx=(0, 12))

        ttk.Button(top, text="Run Validation Suite", command=self._run_validation).pack(side="left", padx=15)

        # Main split
        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Left: Test selection
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

        ttk.Button(left, text="Export Report", command=self._export_report).pack(fill="x", pady=5)

        # Right: Results and flags
        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)

        # Results panel
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

        # Flagged samples panel
        flags_label = ttk.Label(right, text="Flagged Samples")
        flags_label.pack(anchor="w", pady=(10, 0))

        flags_frame = ttk.Frame(right)
        flags_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.flag_tree = ttk.Treeview(flags_frame,
                                      columns=("Flag", "Reason"),
                                      show="tree headings",
                                      height=10)
        self.flag_tree.heading("#0", text="Sample ID")
        self.flag_tree.heading("Flag", text="Flag")
        self.flag_tree.heading("Reason", text="Reason")
        self.flag_tree.column("#0", width=200)
        self.flag_tree.column("Flag", width=100)
        self.flag_tree.column("Reason", width=300)

        flag_scroll = ttk.Scrollbar(flags_frame, orient="vertical", command=self.flag_tree.yview)
        self.flag_tree.configure(yscrollcommand=flag_scroll.set)

        self.flag_tree.pack(side="left", fill="both", expand=True)
        flag_scroll.pack(side="right", fill="y")

        # Flag action buttons
        flag_btns = ttk.Frame(right)
        flag_btns.pack(fill="x", pady=(5, 0))

        ttk.Button(flag_btns, text="Remove Flag", command=self._remove_flag).pack(side="left", padx=2)
        ttk.Button(flag_btns, text="Clear All Flags", command=self._clear_all_flags).pack(side="left", padx=2)

        # Load initial dataset
        self._refresh_datasets()
        self.on_dataset_changed()

    def on_dataset_changed(self) -> None:
        """Handle dataset change from other tabs."""
        # Sync dropdown to shared state.
        self._sync_combo_to_state()

        if self.state.dataset_dir:
            self.flag_manager = FlagManager(self.state.dataset_dir)
            self._refresh_flag_tree()

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
        self._sync_combo_to_state()

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

    def _sync_combo_to_state(self) -> None:
        """Set combo to match current state.dataset_dir if possible."""
        if not getattr(self, "dataset_combo", None):
            return

        # If state points to one of our candidates, select its label.
        if self.state.dataset_dir:
            for label, p in self._dataset_by_label.items():
                if p == self.state.dataset_dir:
                    self.var_dataset.set(label)
                    return

        # If nothing selected yet, pick first available.
        cur = (self.var_dataset.get() or "").strip()
        if cur:
            return
        if self._dataset_labels:
            self.var_dataset.set(self._dataset_labels[0])

    def _on_dataset_selected(self) -> None:
        """Update shared dataset selection from this tab."""
        label = (self.var_dataset.get() or "").strip()
        if not label:
            return
        ds = self._dataset_by_label.get(label)
        if not ds:
            return
        self.state.dataset_dir = ds
        try:
            # Notify app (MonitorAppTabbed binds this event on the notebook).
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception:
            pass

    def _run_validation(self) -> None:
        """Run validation suite in background thread."""
        if not self.state.dataset_dir:
            messagebox.showinfo("Info", "No dataset selected")
            return

        # Disable button during validation
        self._set_results("Running validation suite...\n\nThis may take a while for large datasets.")

        def validate():
            try:
                results = run_validation_suite(
                    self.state.dataset_dir,
                    run_schema=self.check_vars["schema"].get(),
                    run_outliers=self.check_vars["outliers"].get(),
                    run_duplicates=self.check_vars["duplicates"].get(),
                    run_edge_cases=self.check_vars["edge_cases"].get()
                )

                self.validation_results = results

                # Update flags
                if self.flag_manager:
                    # Add outlier flags
                    for outlier in results.get('outliers', []):
                        self.flag_manager.add_flag(
                            outlier['sample_id'],
                            outlier['flag'],
                            outlier['reason']
                        )

                    # Add edge case flags
                    for edge_case in results.get('edge_cases', []):
                        self.flag_manager.add_flag(
                            edge_case['sample_id'],
                            edge_case['flag'],
                            edge_case['reason']
                        )

                    # Add duplicate flags
                    for id1, id2, distance in results.get('duplicates', []):
                        reason = f'Duplicate of {id2} (distance={distance})'
                        self.flag_manager.add_flag(id1, 'DUPLICATE', reason)

                # Display results
                self._display_results(results)
                self._refresh_flag_tree()

            except Exception as e:
                messagebox.showerror("Error", f"Validation failed:\n{e}")
                self._set_results(f"Validation failed:\n{e}")

        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _display_results(self, results: dict) -> None:
        """Display validation results."""
        output = "=" * 60 + "\n"
        output += "VALIDATION RESULTS\n"
        output += "=" * 60 + "\n\n"

        # Schema validation
        if results.get('schema_validation'):
            schema = results['schema_validation']
            status = "✓ PASSED" if schema['passed'] else "✗ FAILED"
            output += f"Schema Validation: {status}\n"
            output += f"  {schema['message']}\n\n"

        # Outliers
        outliers = results.get('outliers', [])
        output += f"Image Outliers: {len(outliers)} found\n"
        if outliers:
            for outlier in outliers[:10]:  # Show first 10
                output += f"  - {outlier['sample_id']}: {outlier['reason']}\n"
            if len(outliers) > 10:
                output += f"  ... and {len(outliers) - 10} more\n"
        output += "\n"

        # Duplicates
        duplicates = results.get('duplicates', [])
        output += f"Duplicate Images: {len(duplicates)} pairs found\n"
        if duplicates:
            for id1, id2, dist in duplicates[:5]:  # Show first 5
                output += f"  - {id1} ↔ {id2} (distance={dist})\n"
            if len(duplicates) > 5:
                output += f"  ... and {len(duplicates) - 5} more\n"
        output += "\n"

        # Edge cases
        edge_cases = results.get('edge_cases', [])
        output += f"Edge Cases: {len(edge_cases)} found\n"
        if edge_cases:
            for case in edge_cases[:10]:
                output += f"  - {case['sample_id']}: {case['reason']}\n"
            if len(edge_cases) > 10:
                output += f"  ... and {len(edge_cases) - 10} more\n"

        output += "\n" + "=" * 60

        self._set_results(output)

    def _set_results(self, text: str) -> None:
        """Set results text."""
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    def _refresh_flag_tree(self) -> None:
        """Refresh flag tree view."""
        self.flag_tree.delete(*self.flag_tree.get_children())

        if not self.flag_manager:
            return

        flagged_samples = self.flag_manager.get_all_flagged_samples()

        for sample_id in flagged_samples:
            flags = self.flag_manager.get_flags(sample_id)

            for flag_dict in flags:
                self.flag_tree.insert("", "end",
                                     text=sample_id,
                                     values=(flag_dict['flag'], flag_dict['reason']))

    def _remove_flag(self) -> None:
        """Remove selected flag."""
        selection = self.flag_tree.selection()
        if not selection or not self.flag_manager:
            return

        item = selection[0]
        sample_id = self.flag_tree.item(item, "text")
        flag_type = self.flag_tree.item(item, "values")[0]

        self.flag_manager.remove_flag(sample_id, flag_type)
        self._refresh_flag_tree()

    def _clear_all_flags(self) -> None:
        """Clear all flags."""
        if not self.flag_manager:
            return

        if messagebox.askyesno("Confirm", "Clear all flags?"):
            for sample_id in self.flag_manager.get_all_flagged_samples():
                self.flag_manager.remove_flag(sample_id)
            self._refresh_flag_tree()

    def _export_report(self) -> None:
        """Export validation report."""
        if not self.validation_results:
            messagebox.showinfo("Info", "Run validation first")
            return

        import json
        from datetime import datetime

        output_path = self.state.dataset_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            # Make results JSON-serializable (NumPy scalars/arrays, tuples, Paths, etc.).
            try:
                import numpy as _np  # type: ignore
            except Exception:  # pragma: no cover
                _np = None

            def _jsonify(obj):
                # Primitive types
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                # Path-like
                if isinstance(obj, Path):
                    return str(obj)
                # NumPy scalar / array
                if _np is not None:
                    try:
                        if isinstance(obj, _np.generic):
                            return obj.item()
                        if isinstance(obj, _np.ndarray):
                            return obj.tolist()
                    except Exception:
                        pass
                # Containers
                if isinstance(obj, dict):
                    return {str(k): _jsonify(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [_jsonify(v) for v in obj]
                # Fallback: stringify
                return str(obj)

            with open(output_path, 'w') as f:
                json.dump(_jsonify(self.validation_results), f, indent=2)

            messagebox.showinfo("Success", f"Report exported to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
