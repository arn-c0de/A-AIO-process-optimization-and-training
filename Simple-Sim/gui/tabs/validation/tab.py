"""Validation Tab"""

from __future__ import annotations
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import threading

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.dataset_catalog import dataset_sort_key, sim_data_roots, unique_dataset_labels
from .ui import ValidationUI
from .logic import ValidationLogic


class ValidationTab(BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.ui = ValidationUI(self, self.frame)
        self.logic = ValidationLogic(sim_root)
        self.validation_results: Optional[dict] = None
        self._dataset_by_label: dict[str, Path] = {}

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self._refresh_datasets()
        self.on_dataset_changed()

    def on_dataset_changed(self) -> None:
        self._sync_combo_to_state()
        if self.state.dataset_dir:
            self.logic.set_dataset(self.state.dataset_dir)
            self._refresh_flag_tree()

    def _refresh_datasets(self) -> None:
        cand = self.logic.get_datasets()
        cand.sort(key=lambda p: dataset_sort_key(self.sim_root, p))

        runs, versions = sim_data_roots(self.sim_root)
        self._dataset_by_label = unique_dataset_labels(cand, runs_root=runs, versions_root=versions)
        self.ui.dataset_combo["values"] = list(self._dataset_by_label)
        self._sync_combo_to_state()

    def _sync_combo_to_state(self) -> None:
        if not hasattr(self.ui, "dataset_combo"): return
        if self.state.dataset_dir:
            for label, p in self._dataset_by_label.items():
                if p == self.state.dataset_dir:
                    self.ui.var_dataset.set(label)
                    return
        cur = (self.ui.var_dataset.get() or "").strip()
        if not cur and self.ui.dataset_combo["values"]:
            self.ui.var_dataset.set(self.ui.dataset_combo["values"][0])

    def _on_dataset_selected(self) -> None:
        label = (self.ui.var_dataset.get() or "").strip()
        if not label: return
        ds = self._dataset_by_label.get(label)
        if not ds: return
        self.state.dataset_dir = ds
        try:
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception: pass

    def _run_validation(self) -> None:
        if not self.state.dataset_dir:
            messagebox.showinfo("Info", "No dataset selected")
            return
        self.ui.set_results_text("Running validation suite...\n\nThis may take a while for large datasets.")
        # Read Tk state on the main thread; the worker must not touch widgets/variables.
        checks = {name: var.get() for name, var in self.ui.check_vars.items()}
        dataset_dir = self.state.dataset_dir

        def on_error(e: Exception) -> None:
            messagebox.showerror("Error", f"Validation failed:\n{e}")
            self.ui.set_results_text(f"Validation failed:\n{e}")

        def on_done(results: dict) -> None:
            self.validation_results = results
            self.ui.display_results(results)
            self._refresh_flag_tree()

        def validate():
            try:
                results = self.logic.run_validation(dataset_dir, checks)
                self.frame.after(0, on_done, results)
            except Exception as e:
                self.frame.after(0, on_error, e)

        threading.Thread(target=validate, daemon=True).start()

    def _refresh_flag_tree(self) -> None:
        flagged_samples = self.logic.get_all_flagged_samples()
        tree_data = []
        for sample_id in flagged_samples:
            flags = self.logic.get_flags(sample_id)
            tree_data.append((sample_id, flags))
        self.ui.refresh_flag_tree(tree_data)

    def _remove_flag(self) -> None:
        selection = self.ui.flag_tree.selection()
        if not selection: return
        item = selection[0]
        sample_id = self.ui.flag_tree.item(item, "text")
        flag_type = self.ui.flag_tree.item(item, "values")[0]
        self.logic.remove_flag(sample_id, flag_type)
        self._refresh_flag_tree()

    def _clear_all_flags(self) -> None:
        if messagebox.askyesno("Confirm", "Clear all flags?"):
            self.logic.clear_all_flags()
            self._refresh_flag_tree()

    def _export_report(self) -> None:
        if not self.validation_results or not self.state.dataset_dir:
            messagebox.showinfo("Info", "Run validation first")
            return
        try:
            path = self.logic.export_report(self.validation_results, self.state.dataset_dir)
            messagebox.showinfo("Success", f"Report exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
