"""Datasets Tab - dataset catalog with categories and archive state."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from gui import dialogs
from gui.state import UiState
from gui.tabs.core.base import BaseTab
from gui.utils.dataset_catalog import (
    DEFAULT_CATEGORY,
    DatasetCatalog,
    dataset_display_name,
    sim_data_roots,
    sorted_dataset_dirs,
)

from .logic import dataset_created_display, dataset_details_text
from .ui import DatasetsUI

_NEW_CATEGORY = "\x00new-category"  # sentinel item in the category picker


class DatasetsTab(BaseTab):
    """Manage dataset categories/archive state used by other tabs."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.catalog = DatasetCatalog(sim_root, state.settings_store)
        self._dataset_by_iid: dict[str, Path] = {}
        self.var_show_archived = tk.BooleanVar(value=True)
        self.var_category_filter = tk.StringVar(value="All")
        self.ui = DatasetsUI(self, self.frame)
        self.tree: ttk.Treeview
        self.combo_category_filter: ttk.Combobox

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self.tree = self.ui.tree
        self.combo_category_filter = self.ui.combo_category_filter
        self._menu = self.ui.menu
        self.refresh()

    # ------------------------------------------------------------------ helpers
    def _status(self, text: str, level: str = "success") -> bool:
        """Non-blocking status message; returns False when no AppContext is available."""
        ctx = getattr(self.state, "ctx", None)
        if ctx is None:
            return False
        try:
            ctx.status(text, level)
            return True
        except Exception:
            return False

    def refresh(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.catalog.reload()
        datasets = sorted_dataset_dirs(self.sim_root)
        self.catalog.prune_unknown(datasets)
        self.catalog.save(self.frame)

        selected_keys = {self._path_key_for(self._dataset_by_iid[i]) for i in self.tree.selection() if i in self._dataset_by_iid}

        self._dataset_by_iid.clear()
        self.ui.list_view.clear()

        filter_values = ["All", DEFAULT_CATEGORY] + [c for c in self.catalog.all_categories() if c != DEFAULT_CATEGORY]
        self.combo_category_filter["values"] = filter_values
        if self.var_category_filter.get() not in filter_values:
            self.var_category_filter.set("All")
        active_filter = self.var_category_filter.get().strip() or "All"

        runs, versions = sim_data_roots(self.sim_root)
        for idx, ds in enumerate(datasets):
            cat = self.catalog.category_for(ds)
            archived = self.catalog.is_archived(ds)
            if not self.var_show_archived.get() and archived:
                continue
            if active_filter != "All" and cat != active_filter:
                continue
            iid = f"ds::{idx}"
            self._dataset_by_iid[iid] = ds
            self.ui.list_view.insert(
                (
                    dataset_display_name(ds, runs_root=runs, versions_root=versions),
                    cat,
                    self._dataset_created_display(ds),
                    "yes" if archived else "no",
                    str(ds),
                ),
                iid=iid,
                tags=("muted",) if archived else (),
            )

        for iid, ds in self._dataset_by_iid.items():
            if self._path_key_for(ds) in selected_keys:
                self.tree.selection_add(iid)

        shown, total = len(self._dataset_by_iid), len(datasets)
        self.ui.var_count.set(f"{shown} of {total} shown" if shown != total else f"{total} datasets")
        if not datasets:
            self.ui.set_empty_hint("No datasets found under outputs/sim_data. "
                                   "Generate one in the Pipeline tab, then press Refresh.")
        elif not shown:
            self.ui.set_empty_hint("No dataset matches the current filter. "
                                   "Choose Category 'All' or enable Show Archived.")
        else:
            self.ui.set_empty_hint("")

    def _dataset_created_display(self, ds: Path) -> str:
        return dataset_created_display(ds)

    def _path_key_for(self, ds: Path) -> str:
        try:
            return str(ds.resolve())
        except Exception:
            return str(ds)

    def _selected_dataset_paths(self) -> list[Path]:
        return [self._dataset_by_iid[iid] for iid in self.tree.selection() if iid in self._dataset_by_iid]

    def _require_selection(self) -> list[Path]:
        paths = self._selected_dataset_paths()
        dialogs.require(paths, self.frame, "Select one or more datasets in the list first.")
        return paths

    def _custom_categories(self) -> list[str]:
        return [c for c in self.catalog.all_categories() if c != DEFAULT_CATEGORY]

    def _pick_custom_category(self, title: str, prompt: str) -> str | None:
        cats = self._custom_categories()
        if not dialogs.require(cats, self.frame, "There are no custom categories yet. "
                                                 "Create one with Categories > New Category…",
                               title="No Categories"):
            return None
        current = self.var_category_filter.get().strip()
        picked = dialogs.pick_from_list(self.frame, title, cats, prompt=prompt,
                                        selected=[current] if current in cats else [])
        return picked[0] if picked else None

    def _ask_new_category_name(self) -> str | None:
        name = dialogs.ask_string(self.frame, "New Category", "Category name:")
        if name is None:
            return None
        name = name.strip()
        if not name or name == DEFAULT_CATEGORY:
            return None
        return name

    # ------------------------------------------------------------------ actions
    def _set_category_for_selected(self) -> None:
        paths = self._require_selection()
        if not paths:
            return
        category = self._ask_category_with_list(
            title="Set Category",
            initial=self.catalog.category_for(paths[0]),
            count=len(paths),
        )
        if category is None:
            return
        category = category.strip() or DEFAULT_CATEGORY
        self.catalog.set_category(paths, category)
        self.catalog.save(self.frame)
        self.refresh()
        self._notify_catalog_changed()
        self._status(f"Category '{category}' set for {len(paths)} dataset(s).")

    def _new_category(self) -> None:
        name = self._ask_new_category_name()
        if name is None:
            return
        if name in self.catalog.custom_categories:
            self._status(f"Category '{name}' already exists.", "info")
            return
        self.catalog.custom_categories.append(name)
        self.catalog.save(self.frame)
        self.refresh()
        self._notify_catalog_changed()
        self._status(f"Category '{name}' created.")

    def _rename_category(self) -> None:
        old_name = self._pick_custom_category("Rename Category", "Category to rename:")
        if old_name is None:
            return
        new_name = dialogs.ask_string(self.frame, "Rename Category", f"New name for '{old_name}':", initial=old_name)
        if new_name is None:
            return
        if self.catalog.rename_category(old_name, new_name):
            if self.var_category_filter.get() == old_name:
                self.var_category_filter.set(new_name.strip())
            self.catalog.save(self.frame)
            self.refresh()
            self._notify_catalog_changed()
            self._status(f"Category '{old_name}' renamed to '{new_name.strip()}'.")

    def _delete_category(self) -> None:
        name = self._pick_custom_category("Delete Category", "Category to delete:")
        if name is None:
            return
        n = sum(1 for v in self.catalog.category_assignments.values() if v == name)
        if not dialogs.confirm(
            self.frame,
            "Delete Category",
            f"Delete category '{name}'?\n\n{n} dataset(s) in it will move to '{DEFAULT_CATEGORY}'. "
            "No dataset files are deleted.",
            danger=True,
            ok_text="Delete Category",
        ):
            return
        if self.catalog.delete_category(name):
            self.catalog.save(self.frame)
            self.refresh()
            self._notify_catalog_changed()
            self._status(f"Category '{name}' deleted.")

    def _set_archive_for_selected(self, archived: bool) -> None:
        paths = self._require_selection()
        if not paths:
            return
        self.catalog.set_archived(paths, archived=archived)
        self.catalog.save(self.frame)
        self.refresh()
        self._notify_catalog_changed()
        self._status(f"{'Archived' if archived else 'Unarchived'} {len(paths)} dataset(s).")

    def _notify_catalog_changed(self) -> None:
        try:
            self.parent.event_generate("<<DatasetCatalogChanged>>", when="tail")
        except Exception:
            pass

    def _on_right_click(self, event) -> None:
        row = self.tree.identify_row(event.y)
        if row:
            if row not in self.tree.selection():
                self.tree.selection_set(row)
            self._menu.tk_popup(event.x_root, event.y_root)

    def _show_details_selected(self) -> None:
        paths = self._require_selection()
        if not paths:
            return
        if not dialogs.require(len(paths) == 1, self.frame, "Select exactly one dataset to show its details."):
            return

        ds = paths[0]
        try:
            details = self._dataset_details_text(ds)
        except Exception as exc:
            dialogs.show_error(self.frame, "Dataset Details", exc, hint=f"Could not read {ds}.")
            return
        dialogs.show_text(self.frame, f"Dataset Details: {ds.name}", details, size=(760, 560))

    def _dataset_details_text(self, ds: Path) -> str:
        return dataset_details_text(ds)

    def _ask_category_with_list(self, *, title: str, initial: str = "", count: int = 1) -> str | None:
        categories = self.catalog.all_categories()
        items = list(categories) + [_NEW_CATEGORY]
        labels = list(categories) + ["+ New Category…"]
        picked = dialogs.pick_from_list(
            self.frame,
            title,
            items,
            labels=labels,
            selected=[initial] if initial in categories else [],
            prompt=f"Category for {count} selected dataset(s). Double-click to choose.",
        )
        if not picked:
            return None
        if picked[0] == _NEW_CATEGORY:
            return self._ask_new_category_name()
        return picked[0]
