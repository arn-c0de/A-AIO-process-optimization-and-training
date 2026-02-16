"""Validation Tab"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import os

from gui.tabs.core.base import BaseTab
from gui.state import UiState
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

        def is_version(p: Path) -> bool:
            try:
                return str(p.resolve()).startswith(str(self.sim_root / "outputs" / "sim_data" / "versions").resolve() + os.sep)
            except Exception: return False

        cand.sort(key=lambda p: (1, -p.stat().st_mtime, p.name) if is_version(p) else (0, p.name))
        
        self._dataset_by_label.clear()
        labels = []
        seen: dict[str, int] = {}
        for p in cand:
            base_label = self._display_for_dataset(p)
            n = seen.get(base_label, 0) + 1
            seen[base_label] = n
            label = base_label if n == 1 else f"{base_label} ({n})"
            labels.append(label)
            self._dataset_by_label[label] = p
            
        self.ui.dataset_combo["values"] = labels
        self._sync_combo_to_state()

    def _display_for_dataset(self, p: Path) -> str:
        try:
            rp = p.resolve()
            runs_base = (self.sim_root / "outputs" / "sim_data" / "runs").resolve()
            versions_base = (self.sim_root / "outputs" / "sim_data" / "versions").resolve()
            if str(rp).startswith(str(runs_base) + os.sep): return rp.name
            if str(rp).startswith(str(versions_base) + os.sep): return f"{rp.parent.name}:{rp.name}"
        except Exception: pass
        return p.name

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
        self.ui.set_results_text("Running validation suite...\\n\\nThis may take a while for large datasets.")

        def validate():
            try:
                checks = {name: var.get() for name, var in self.ui.check_vars.items()}
                results = self.logic.run_validation(self.state.dataset_dir, checks)
                self.validation_results = results
                self.frame.after(0, lambda: self.ui.display_results(results))
                self.frame.after(0, self._refresh_flag_tree)
            except Exception as e:
                messagebox.showerror("Error", f"Validation failed:\\n{e}")
                self.ui.set_results_text(f"Validation failed:\\n{e}")

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
            messagebox.showinfo("Success", f"Report exported to:\\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\\n{e}")
