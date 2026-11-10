"""Validation Tab"""

from __future__ import annotations
from tkinter import ttk
from pathlib import Path

from gui import dialogs
from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.dataset_catalog import dataset_sort_key, sim_data_roots, unique_dataset_labels
from gui.core.tasks import runner_for
from .ui import ValidationUI
from .logic import ValidationLogic
from gui.utils.tk_safe import quiet


class ValidationTab(BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.ui = ValidationUI(self, self.frame)
        self.logic = ValidationLogic(sim_root)
        self.validation_results: dict | None = None
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
        with quiet():
            self.parent.event_generate("<<DatasetChanged>>", when="tail")

    def _run_validation(self) -> None:
        if not dialogs.require(self.state.dataset_dir, self.frame,
                               "Select a dataset first (Dataset field at the top).", title="No Dataset"):
            return
        # Read Tk state on the main thread; the worker must not touch widgets/variables.
        checks = {name: var.get() for name, var in self.ui.check_vars.items()}
        if not dialogs.require(any(checks.values()), self.frame,
                               "Enable at least one check in the Checks section.", title="No Checks Selected"):
            return
        self.ui.set_results_text("Running validation suite...\n\nThis may take a while for large datasets.")
        dataset_dir = self.state.dataset_dir

        def on_error(e: BaseException) -> None:
            self.ui.set_results_text(f"Validation failed:\n{e}")
            dialogs.show_error(self.frame, "Validation Failed", e)

        def on_done(results: dict) -> None:
            self.validation_results = results
            self._results_dataset_dir = dataset_dir
            self.ui.display_results(results)
            # The run may have written through its own manager (dataset switched meanwhile).
            self.logic.reload_flags()
            self._refresh_flag_tree()

        def on_finally() -> None:
            self.ui.set_running(False)

        self.ui.set_running(True)
        runner_for(self.state, self.frame).run(
            lambda h: self.logic.run_validation(dataset_dir, checks, cancelled=h.cancelled),
            name="Validation", on_done=on_done, on_error=on_error, on_finally=on_finally)

    def _refresh_flag_tree(self) -> None:
        flagged_samples = self.logic.get_all_flagged_samples()
        tree_data = []
        for sample_id in flagged_samples:
            flags = self.logic.get_flags(sample_id)
            tree_data.append((sample_id, flags))
        self.ui.refresh_flag_tree(tree_data)

    def _remove_flag(self) -> None:
        selection = self.ui.flag_tree.selection()
        if not dialogs.require(selection, self.frame, "Select a flagged sample in the list first."):
            return
        item = selection[0]
        sample_id = self.ui.flag_tree.item(item, "text")
        flag_type = self.ui.flag_tree.item(item, "values")[0]
        self.logic.remove_flag(sample_id, flag_type)
        self._refresh_flag_tree()
        self._status(f"Flag {flag_type} removed from {sample_id}.")

    def _clear_all_flags(self) -> None:
        samples = self.logic.get_all_flagged_samples()
        if not samples:
            self._status("There are no flags to clear.", "info")
            return
        if dialogs.confirm(self.frame, "Clear All Flags",
                           f"Remove all flags from {len(samples)} sample(s) of this dataset?\n\n"
                           "Images and labels are not changed.",
                           danger=True, ok_text="Clear Flags"):
            self.logic.clear_all_flags()
            self._refresh_flag_tree()
            self._status(f"Flags cleared for {len(samples)} sample(s).")

    def _export_report(self) -> None:
        if not dialogs.require(self.validation_results and self.state.dataset_dir, self.frame,
                               "Run the validation first; the report contains its results.",
                               title="No Results"):
            return
        try:
            # The report goes to the dataset the results came from, not to the current selection.
            path = self.logic.export_report(
                self.validation_results, getattr(self, "_results_dataset_dir", None) or self.state.dataset_dir)
        except Exception as e:
            dialogs.show_error(self.frame, "Export Failed", e)
            return
        if not self._status(f"Validation report exported to {path}"):
            dialogs.show_info(self.frame, "Report Exported", f"Report exported to:\n{path}")
