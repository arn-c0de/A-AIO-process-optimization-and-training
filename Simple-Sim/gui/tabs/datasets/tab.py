"""Datasets Tab - dataset catalog with categories and archive state."""

from __future__ import annotations

import json
import os
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Dict, List, Optional

from gui.state import UiState
from gui.tabs.core.base import BaseTab
from gui.utils.dataset_catalog import (
    DEFAULT_CATEGORY,
    DatasetCatalog,
    dataset_display_name,
)


class DatasetsTab(BaseTab):
    """Manage dataset categories/archive state used by other tabs."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.catalog = DatasetCatalog(sim_root, state.settings_store)
        self._dataset_by_iid: Dict[str, Path] = {}
        self.var_show_archived = tk.BooleanVar(value=True)
        self.var_category_filter = tk.StringVar(value="All")
        self.tree: ttk.Treeview

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(top, text="Show Details", command=self._show_details_selected).pack(side="left", padx=(6, 0))
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Button(top, text="Set Category...", command=self._set_category_for_selected).pack(side="left")
        ttk.Button(top, text="New Category...", command=self._new_category).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Rename Category...", command=self._rename_category).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Delete Category...", command=self._delete_category).pack(side="left", padx=(6, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Archive", command=lambda: self._set_archive_for_selected(True)).pack(side="left")
        ttk.Button(top, text="Unarchive", command=lambda: self._set_archive_for_selected(False)).pack(side="left", padx=(6, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Checkbutton(top, text="Show archived", variable=self.var_show_archived, command=self.refresh).pack(side="left")
        ttk.Label(top, text="Category:").pack(side="left", padx=(8, 4))
        self.combo_category_filter = ttk.Combobox(top, textvariable=self.var_category_filter, state="readonly", width=20)
        self.combo_category_filter.pack(side="left")
        self.combo_category_filter.bind("<<ComboboxSelected>>", lambda _e: self.refresh())

        cols = ("dataset", "category", "archived", "location")
        self.tree = ttk.Treeview(self.frame, columns=cols, show="headings", selectmode="extended")
        self.tree.pack(fill="both", expand=True)
        self.tree.heading("dataset", text="Dataset")
        self.tree.heading("category", text="Category")
        self.tree.heading("archived", text="Archived")
        self.tree.heading("location", text="Location")
        self.tree.column("dataset", width=280, anchor="w")
        self.tree.column("category", width=170, anchor="w")
        self.tree.column("archived", width=90, anchor="center")
        self.tree.column("location", width=500, anchor="w")
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Button-2>", self._on_right_click)

        self._menu = tk.Menu(self.frame, tearoff=0)
        self._menu.add_command(label="Show details", command=self._show_details_selected)
        self._menu.add_separator()
        self._menu.add_command(label="Set category...", command=self._set_category_for_selected)
        self._menu.add_separator()
        self._menu.add_command(label="Archive selected", command=lambda: self._set_archive_for_selected(True))
        self._menu.add_command(label="Unarchive selected", command=lambda: self._set_archive_for_selected(False))
        self._menu.add_separator()
        self._menu.add_command(label="Refresh", command=self.refresh)

        self.refresh()

    def refresh(self) -> None:
        self.catalog.reload()
        datasets = self._scan_datasets()
        self.catalog.prune_unknown(datasets)
        self.catalog.save(self.frame)

        selected_keys = {self._path_key_for(self._dataset_by_iid[i]) for i in self.tree.selection() if i in self._dataset_by_iid}

        self._dataset_by_iid.clear()
        self.tree.delete(*self.tree.get_children())

        filter_values = ["All", DEFAULT_CATEGORY] + [c for c in self.catalog.all_categories() if c != DEFAULT_CATEGORY]
        self.combo_category_filter["values"] = filter_values
        if self.var_category_filter.get() not in filter_values:
            self.var_category_filter.set("All")
        active_filter = self.var_category_filter.get().strip() or "All"

        runs, versions = self._sim_data_roots()
        for idx, ds in enumerate(datasets):
            cat = self.catalog.category_for(ds)
            archived = self.catalog.is_archived(ds)
            if not self.var_show_archived.get() and archived:
                continue
            if active_filter != "All" and cat != active_filter:
                continue
            iid = f"ds::{idx}"
            self._dataset_by_iid[iid] = ds
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    dataset_display_name(ds, runs_root=runs, versions_root=versions),
                    cat,
                    "yes" if archived else "no",
                    str(ds),
                ),
            )

        for iid, ds in self._dataset_by_iid.items():
            if self._path_key_for(ds) in selected_keys:
                self.tree.selection_add(iid)

    def _scan_datasets(self) -> List[Path]:
        runs, versions = self._sim_data_roots()
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)
        cand: List[Path] = [p for p in runs.iterdir() if p.is_dir()]
        cand.extend([p for p in versions.glob("*/*") if p.is_dir()])
        cand.sort(
            key=lambda p: (1, -p.stat().st_mtime, p.name)
            if str(p.resolve()).startswith(str(versions.resolve()) + os.sep)
            else (0, p.name)
        )
        return cand

    def _sim_data_roots(self) -> tuple[Path, Path]:
        base = self.sim_root / "outputs" / "sim_data"
        return base / "runs", base / "versions"

    def _path_key_for(self, ds: Path) -> str:
        try:
            return str(ds.resolve())
        except Exception:
            return str(ds)

    def _selected_dataset_paths(self) -> List[Path]:
        return [self._dataset_by_iid[iid] for iid in self.tree.selection() if iid in self._dataset_by_iid]

    def _set_category_for_selected(self) -> None:
        paths = self._selected_dataset_paths()
        if not paths:
            messagebox.showinfo("Datasets", "No dataset selected.")
            return
        category = self._ask_category_with_list(
            title="Set category",
            initial=self.catalog.category_for(paths[0]),
        )
        if category is None:
            return
        category = category.strip() or DEFAULT_CATEGORY
        self.catalog.set_category(paths, category)
        self.catalog.save(self.frame)
        self.refresh()
        self._notify_catalog_changed()

    def _new_category(self) -> None:
        name = simpledialog.askstring("New category", "Category name:", parent=self.frame)
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if name == DEFAULT_CATEGORY:
            return
        if name not in self.catalog.custom_categories:
            self.catalog.custom_categories.append(name)
            self.catalog.save(self.frame)
            self.refresh()
            self._notify_catalog_changed()

    def _rename_category(self) -> None:
        current = self.var_category_filter.get().strip()
        initial = "" if current in {"", "All", DEFAULT_CATEGORY} else current
        old_name = simpledialog.askstring("Rename category", "Current category name:", initialvalue=initial, parent=self.frame)
        if old_name is None:
            return
        new_name = simpledialog.askstring("Rename category", "New category name:", initialvalue=old_name, parent=self.frame)
        if new_name is None:
            return
        if self.catalog.rename_category(old_name, new_name):
            self.catalog.save(self.frame)
            self.refresh()
            self._notify_catalog_changed()

    def _delete_category(self) -> None:
        current = self.var_category_filter.get().strip()
        initial = "" if current in {"", "All", DEFAULT_CATEGORY} else current
        name = simpledialog.askstring("Delete category", "Category to delete:", initialvalue=initial, parent=self.frame)
        if name is None:
            return
        name = name.strip()
        if not name or name == DEFAULT_CATEGORY:
            return
        if not messagebox.askyesno(
            "Delete category",
            f"Delete category '{name}'?\nDatasets in this category will move to '{DEFAULT_CATEGORY}'.",
            parent=self.frame,
        ):
            return
        if self.catalog.delete_category(name):
            self.catalog.save(self.frame)
            self.refresh()
            self._notify_catalog_changed()

    def _set_archive_for_selected(self, archived: bool) -> None:
        paths = self._selected_dataset_paths()
        if not paths:
            messagebox.showinfo("Datasets", "No dataset selected.")
            return
        self.catalog.set_archived(paths, archived=archived)
        self.catalog.save(self.frame)
        self.refresh()
        self._notify_catalog_changed()

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
        paths = self._selected_dataset_paths()
        if not paths:
            messagebox.showinfo("Datasets", "No dataset selected.")
            return
        if len(paths) > 1:
            messagebox.showinfo("Datasets", "Please select exactly one dataset.")
            return

        ds = paths[0]
        try:
            details = self._dataset_details_text(ds)
        except Exception as exc:
            messagebox.showerror("Dataset Details", f"Failed to load details:\n{exc}")
            return

        dialog = tk.Toplevel(self.frame.winfo_toplevel())
        dialog.title(f"Dataset Details: {ds.name}")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        dialog.geometry("760x560")

        frm = ttk.Frame(dialog, padding=10)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", details)
        txt.configure(state="disabled")
        ttk.Button(frm, text="Close", command=dialog.destroy).pack(anchor="e", pady=(8, 0))

    def _dataset_details_text(self, ds: Path) -> str:
        lines: List[str] = []
        lines.append(f"Dataset: {ds.name}")
        lines.append(f"Path: {ds}")
        lines.append("")

        manifest = {}
        manifest_path = ds / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        total_samples = None
        split_stats = {}
        if isinstance(manifest, dict):
            ds_stats = manifest.get("dataset_stats") or {}
            if isinstance(ds_stats, dict):
                total_samples = ds_stats.get("total_samples")
                split_stats = ds_stats.get("splits") or {}

        if total_samples is None:
            labels_path = ds / "labels.jsonl"
            if labels_path.exists():
                try:
                    total_samples = sum(1 for _ in labels_path.open("r", encoding="utf-8") if _.strip())
                except Exception:
                    total_samples = None

        lines.append(f"Total samples: {total_samples if total_samples is not None else 'unknown'}")

        if isinstance(split_stats, dict) and split_stats:
            tr = int(split_stats.get("train", 0) or 0)
            va = int(split_stats.get("val", 0) or 0)
            te = int(split_stats.get("test", 0) or 0)
            lines.append(f"Splits: train={tr}, val={va}, test={te}")
        lines.append("")

        profile_counts = Counter()
        labels_path = ds / "labels.jsonl"
        if labels_path.exists():
            with labels_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    pid = str(row.get("profile_id") or "").strip()
                    if not pid:
                        pid = "unknown_or_legacy"
                    profile_counts[pid] += 1

        if not profile_counts:
            comp = manifest.get("component_profile") if isinstance(manifest, dict) else None
            if isinstance(comp, dict):
                pid = str(comp.get("profile_id") or "").strip()
                if pid:
                    if isinstance(total_samples, int) and total_samples >= 0:
                        profile_counts[pid] = total_samples
                    else:
                        profile_counts[pid] = 0
            profiles = manifest.get("profiles") if isinstance(manifest, dict) else None
            if isinstance(profiles, list):
                for p in profiles:
                    if isinstance(p, dict):
                        pid = str(p.get("profile_id") or "").strip()
                        if pid and pid not in profile_counts:
                            profile_counts[pid] = 0

        lines.append("Samples per profile:")
        if profile_counts:
            for pid, cnt in sorted(profile_counts.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"- {pid}: {cnt}")
        else:
            lines.append("- (no profile data found)")

        return "\n".join(lines)

    def _ask_category_with_list(self, *, title: str, initial: str = "") -> Optional[str]:
        categories = self.catalog.all_categories()
        dialog = tk.Toplevel(self.frame.winfo_toplevel())
        dialog.title(title)
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        dialog.minsize(440, 320)

        chosen: List[str] = []
        frm = ttk.Frame(dialog, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Available categories (click to use):").pack(anchor="w")
        lb = tk.Listbox(frm, exportselection=False, height=min(12, max(5, len(categories))))
        lb.pack(fill="both", expand=True, pady=(6, 8))
        for cat in categories:
            lb.insert("end", cat)

        entry_var = tk.StringVar(value=initial)
        ttk.Label(frm, text="Category name:").pack(anchor="w")
        ent = ttk.Entry(frm, textvariable=entry_var)
        ent.pack(fill="x", pady=(4, 8))

        if initial:
            try:
                idx = categories.index(initial)
                lb.selection_set(idx)
                lb.see(idx)
            except Exception:
                pass

        def _pick_selected(_evt=None) -> None:
            idxs = lb.curselection()
            if not idxs:
                return
            entry_var.set(str(lb.get(int(idxs[0]))))

        lb.bind("<<ListboxSelect>>", _pick_selected)
        lb.bind("<Double-Button-1>", lambda _e: _confirm())

        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="right")

        def _confirm() -> None:
            value = entry_var.get().strip()
            if not value:
                messagebox.showinfo("Category", "Please enter or select a category.", parent=dialog)
                return
            chosen.append(value)
            dialog.destroy()

        ttk.Button(btns, text="Set", command=_confirm).pack(side="right", padx=(0, 8))

        ent.focus_set()
        dialog.wait_window()
        return chosen[0] if chosen else None
