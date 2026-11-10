"""Weights Tab - Model checkpoint versioning/backup/export + comparisons."""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Any

from gui import dialogs
from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.checkpoint_meta import profile_label_suffix
from gui.utils.dataset_catalog import sim_data_roots, unique_dataset_labels
from gui.utils.settings_store import restore_persisted_vars
from gui.core.tasks import DispatchQueue, as_task_fn, runner_for
from simple_sim.dataset_files import DATASET_MANIFEST, LABELS_JSONL
from simple_sim.file_cache import FileCache
from simple_sim.io_utils import now_iso, path_size_bytes
from simple_sim.models import load_checkpoint
from simple_sim.reports import metric_value, report_timestamp
from .ui import WeightsUI
from .logic import WeightsLogic
from datetime import datetime
import contextlib


class WeightsTab(BaseTab):
    """Tab: manage model weights (snapshots/import/export) and run evaluations."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        # WeightsLogic only put()s; DispatchQueue drains on the Tk thread (no after() poll).
        self.log_q: DispatchQueue = DispatchQueue()
        self.ui = WeightsUI(self, self.frame)
        self.logic = WeightsLogic(sim_root, self.log_q)

        self._models: list[Path] = []
        self._model_by_iid: dict[str, Path] = {}
        self._favorites: dict[str, dict[str, Any]] = {}
        self._arena: dict[str, dict[str, Any]] = {}

        self._group_assignments: dict[str, str] = {}
        self._custom_groups: list[str] = []
        self._group_iids: dict[str, str] = {}
        self._drag_src_iid: str | None = None

        self._reports: list[dict[str, Any]] = []
        self._report_by_iid: dict[str, dict[str, Any]] = {}

        self._cmp_dataset_dirs: list[Path] = []
        self._cmp_dataset_labels: list[str] = []
        self._cmp_dataset_by_label: dict[str, Path] = {}

        # Stat-first caches: manifest/labels are re-read only when the files change (S6).
        self._ds_file_cache = FileCache(max_entries=128)
        self._profile_info_cache: dict[tuple[str, float], str] = {}

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self.on_dataset_changed()
        self._refresh_cmp_datasets()
        self._favorites = self.logic.load_favorites()
        self._arena = self.logic.load_arena()
        self._refresh_models()
        self._refresh_reports()
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self.bind_queue(self.log_q, self._on_log_lines)

    def on_activate(self) -> None:
        """Build lazily once, then auto-refresh model categorization on each open."""
        super().on_activate()
        # Keep group placement up-to-date without requiring a manual Refresh click.
        self._favorites = self.logic.load_favorites()
        self._arena = self.logic.load_arena()
        self._refresh_models()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir: return
        self.ui.var_dataset.set(str(self.state.dataset_dir))
        active = self.logic._model_root() / f"{self.state.dataset_dir.name}.pt"
        self.ui.var_active_model.set(str(active))
        self.ui.var_dataset_samples.set(self._describe_dataset_samples(self.state.dataset_dir))
        if hasattr(self.ui, "list_cmp_datasets") and self.ui.list_cmp_datasets.size() > 0:
            if not self.ui.list_cmp_datasets.curselection(): self._select_current_dataset_for_compare()

    @staticmethod
    def _fmt_list(items: list[str], limit: int = 8) -> str:
        """Newline-joined list, truncated to ``limit`` entries with a "+N more" line."""
        shown = os.linesep.join(items[:limit])
        return f"{shown}\n... (+{len(items) - limit} more)" if len(items) > limit else shown

    def _run_background(self, fn: Any, *, name: str, on_done: Any, on_error: Any, on_finally: Any = None) -> None:
        """Run ``fn()`` / ``fn(task)`` off the UI thread via ctx.tasks (status bar progress); callbacks run on the UI thread."""
        call = as_task_fn(fn)  # fn() or fn(task) for cancel/progress
        runner_for(self.state, self.frame).run(call, name=name, on_done=on_done, on_error=on_error, on_finally=on_finally)

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None: return
        # Explicit None check inside the helper: False/"" are valid persisted values (e.g. unchecked "save preds").
        restore_persisted_vars(st, self._persisted_var_bindings())

        self._refresh_reports()
        custom_raw = st.get("weights.custom_groups")
        if isinstance(custom_raw, str):
            try:
                parsed = json.loads(custom_raw)
                if isinstance(parsed, list): self._custom_groups = [str(k) for k in parsed if isinstance(k, str) and k]
            except Exception: pass
        assignments_raw = st.get("weights.group_assignments")
        if isinstance(assignments_raw, str):
            try:
                parsed = json.loads(assignments_raw)
                if isinstance(parsed, dict): self._group_assignments = {str(k): str(v) for k, v in parsed.items() if k and v}
            except Exception: pass

    def _persisted_var_bindings(self) -> list[tuple[str, Any]]:
        return [("weights.split", self.ui.var_split), ("weights.device", self.ui.var_device),
                ("weights.max_samples", self.ui.var_max_samples), ("weights.save_preds", self.ui.chk_save_preds),
                ("weights.report_scope", self.ui.var_report_scope), ("weights.report_split", self.ui.var_report_split),
                ("weights.report_sort", self.ui.var_report_sort), ("weights.compare_a", self.ui.var_cmp_a),
                ("weights.compare_b", self.ui.var_cmp_b)]

    def on_close(self) -> None:
        self.logic.stop()

    def _on_log_lines(self, lines: list[str]) -> None:
        for line in lines: self.ui.append_log(line)

    def _describe_dataset_samples(self, ds: Path) -> str:
        # The text depends on the manifest and (fallback) labels.jsonl only.
        return self._ds_file_cache.cached(("samples", str(ds)), (ds / DATASET_MANIFEST, ds / LABELS_JSONL),
                                          lambda: self.logic.describe_dataset_samples(ds))

    def _dataset_classes(self, ds: Path) -> list[str]:
        # Worker thread: FileCache is thread-safe; a copy keeps the cached list immutable.
        return list(self._ds_file_cache.cached(("classes", str(ds)), (ds / LABELS_JSONL,),
                                               lambda: tuple(self.logic.get_dataset_classes(ds))))

    def _refresh_cmp_datasets(self) -> None:
        cand = self.logic.get_cmp_datasets()
        self._cmp_dataset_dirs = cand
        
        old_sel = {str(self.ui.list_cmp_datasets.get(i)) for i in self.ui.list_cmp_datasets.curselection()}

        runs, versions = sim_data_roots(self.sim_root)
        self._cmp_dataset_by_label = unique_dataset_labels(cand, runs_root=runs, versions_root=versions)
        self._cmp_dataset_labels = list(self._cmp_dataset_by_label)

        self.ui.list_cmp_datasets.delete(0, "end")
        for label in self._cmp_dataset_labels:
            self.ui.list_cmp_datasets.insert("end", label)

        if old_sel:
            for idx, label in enumerate(self._cmp_dataset_labels):
                if label in old_sel: self.ui.list_cmp_datasets.selection_set(idx)
        else: self._select_current_dataset_for_compare()

    def _select_current_dataset_for_compare(self) -> None:
        try: self.ui.list_cmp_datasets.selection_clear(0, "end")
        except Exception: return
        if not self.state.dataset_dir: return
        runs, versions = sim_data_roots(self.sim_root)
        want = self.logic._display_for_dataset(self.state.dataset_dir, runs=runs, versions=versions)
        for idx, label in enumerate(self._cmp_dataset_labels):
            if label == want or self._cmp_dataset_by_label.get(label) == self.state.dataset_dir:
                self.ui.list_cmp_datasets.selection_set(idx)
                self.ui.list_cmp_datasets.see(idx)
                return

    def _select_all_datasets_for_compare(self) -> None:
        with contextlib.suppress(Exception): self.ui.list_cmp_datasets.selection_set(0, "end")

    def _selected_compare_datasets(self) -> list[Path]:
        ds: list[Path] = []
        try:
            for i in self.ui.list_cmp_datasets.curselection():
                label = str(self.ui.list_cmp_datasets.get(i))
                if p := self._cmp_dataset_by_label.get(label): ds.append(p)
        except Exception: pass
        if not ds and self.state.dataset_dir: ds = [self.state.dataset_dir]
        return [p for p in ds if p.exists() and p.is_dir()]

    def _selected_model_path(self) -> Path | None:
        sel = self.ui.tree.selection()
        if not sel: return None
        return self._model_by_iid.get(sel[0])

    def _selected_model_paths(self) -> list[Path]:
        return [p for iid in self.ui.tree.selection() if (p := self._model_by_iid.get(iid))]

    def _is_favorited(self, p: Path) -> bool:
        return self.logic.rel_path(p) in self._favorites

    def _is_arena_tracked(self, p: Path) -> bool:
        return self.logic.rel_path(p) in self._arena

    def _add_selected_to_favorites(self) -> None:
        paths = self._selected_model_paths()
        if not paths: return
        self._assign_selected_to_group(paths, "Favorites")

    def _remove_selected_from_favorites(self) -> None:
        paths = [p for p in self._selected_model_paths() if self._is_favorited(p)]
        if not paths: return
        if not dialogs.confirm(self.frame, "Remove from Favorites", f"Remove {len(paths)} checkpoint(s) from favorites?\n\n{self._fmt_list([self.logic.rel_path(p) for p in paths])}"): return
        self._assign_selected_to_group(paths, "Uncategorized")

    def _add_selected_to_arena(self) -> None:
        paths = self._selected_model_paths()
        if not paths: return
        self.logic.add_to_arena(paths, self._arena)
        self.logic.save_arena(self._arena)
        self.ui.append_log(f"[arena] added {len(paths)} model(s)\n")
        self._status(f"Added {len(paths)} model(s) to the Arena")
        self._refresh_models()

    def _remove_selected_from_arena(self) -> None:
        paths = [p for p in self._selected_model_paths() if self._is_arena_tracked(p)]
        if not paths: return
        if not dialogs.confirm(self.frame, "Remove from Arena", f"Remove {len(paths)} checkpoint(s) from Arena tracking?\n\n{self._fmt_list([self.logic.rel_path(p) for p in paths])}"): return
        self.logic.remove_from_arena(paths, self._arena)
        self.logic.save_arena(self._arena)
        self.ui.append_log(f"[arena] removed {len(paths)} model(s)\n")
        self._refresh_models()

    def _generate_arena_report(self) -> None:
        try: self.logic.generate_arena_report()
        except FileNotFoundError as e: dialogs.show_error(self.frame, "Missing Script", str(e))
        except Exception as e: dialogs.show_error(self.frame, "Arena Report Failed", e)

    def _refresh_models(self) -> None:
        self._models = self.logic.get_models()
        self._models.sort(key=lambda p: (0 if self._is_favorited(p) else 1, -(p.stat().st_mtime if p.exists() else 0.0)))

        prev_open_groups = {child for child in self.ui.tree.get_children() if child.startswith("group::") and self.ui.tree.item(child, "open")}
        prev_selected_paths = {str(self._model_by_iid[iid].resolve()) for iid in self.ui.tree.selection() if iid in self._model_by_iid}
        
        self.ui.tree.delete(*self.ui.tree.get_children())
        self._model_by_iid.clear()
        self._group_iids.clear()

        group_names = self._available_groups()
        group_members: dict[str, list[Path]] = {gn: [] for gn in group_names}
        for p in self._models:
            group = self._group_for_path(p)
            group_members.setdefault(group, []).append(p)
        
        display_values: list[str] = []
        restore_selection: list[str] = []
        item_counter = 0

        for group in group_names:
            group_iid = f"group::{group}"
            self.ui.tree.insert(
                "",
                "end",
                iid=group_iid,
                text=f"{group} ({len(group_members.get(group, []))})",
                values=("",) * 5 + (f"{len(group_members.get(group, []))} models",),
                tags=("group_header",),
                open=(group_iid in prev_open_groups) or (group == "Favorites"),
            )
            self._group_iids[group] = group_iid
            
            for p in group_members.get(group, []):
                st = p.stat()
                is_bundle = p.is_dir()
                size_s = self.ui.fmt_bytes(path_size_bytes(p) if is_bundle else st.st_size)
                display_name = f"[Bundle] {p.name}" if is_bundle else p.name
                if self._is_arena_tracked(p): display_name = f"[A] {display_name}"
                mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                rel = self.logic.rel_path(p)
                runs_s = self.logic.runs_for_model(p)
                fav_s = "★" if self._is_favorited(p) else ""
                iid = f"model::{item_counter}"; item_counter += 1
                self.ui.tree.insert(group_iid, "end", iid=iid, text=display_name, values=(fav_s, runs_s, size_s, mt, rel, group), tags=("model_entry",))
                self._model_by_iid[iid] = p
                if str(p.resolve()) in prev_selected_paths: restore_selection.append(iid)
                display_values.append(rel)
        
        if restore_selection: self.ui.tree.selection_set(restore_selection)
        self.ui.set_compare_combobox_values(display_values)
        if display_values and not self.ui.var_cmp_a.get(): self.ui.var_cmp_a.set(display_values[0])
        if len(display_values) > 1 and not self.ui.var_cmp_b.get(): self.ui.var_cmp_b.set(display_values[1])
        
        self._refresh_reports()
        self._refresh_groups_menu()

    def _available_groups(self) -> list[str]:
        names: list[str] = ["Favorites", "Snapshots", "Imports"]
        names.extend([g for g in self._custom_groups if g and g not in names])
        names.extend([v for v in sorted(set(self._group_assignments.values())) if v and v not in names])
        if "Uncategorized" not in names: names.append("Uncategorized")
        return names

    def _group_for_path(self, p: Path) -> str:
        rel = self.logic.rel_path(p)
        if rel in self._group_assignments and self._group_assignments[rel] in self._available_groups(): return self._group_assignments[rel]
        if self._is_favorited(p): return "Favorites"
        rp = p.resolve()
        root = self.logic._model_root().resolve()
        versions, imports = (root / "versions").resolve(), (root / "imports").resolve()
        if str(rp).startswith(str(versions) + os.sep): return "Snapshots"
        if str(rp).startswith(str(imports) + os.sep): return "Imports"
        return "Uncategorized"

    def _is_group_header(self, iid: str) -> bool: return bool(iid and iid.startswith("group::"))

    def _group_from_iid(self, iid: str) -> str | None:
        if not iid or "::" not in iid: return None
        return iid.split("::", 1)[1]

    def _refresh_groups_menu(self) -> None:
        if not self.ui._groups_menu: return
        self.ui._groups_menu.delete(0, "end")
        for group in self._available_groups():
            self.ui._groups_menu.add_command(label=group, command=lambda g=group: self._assign_selected_to_group(self._selected_model_paths(), g))

    def _assign_selected_to_group(self, paths: list[Path], group: str | None) -> None:
        if not paths or not group: return
        fav_changed = False
        for p in paths:
            rel = self.logic.rel_path(p)
            if group == "Favorites":
                if rel not in self._favorites:
                    self._favorites[rel] = {"path": rel, "added": now_iso(timespec="seconds")}
                    fav_changed = True
            else:
                if rel in self._favorites:
                    self._favorites.pop(rel, None)
                    fav_changed = True
            self._group_assignments[rel] = group
        if fav_changed: self.logic.save_favorites(self._favorites)
        self._save_group_state()
        self.ui.append_log(f"[group] assigned {len(paths)} model(s) -> {group}\n")
        self._refresh_models()

    def _save_group_state(self) -> None:
        st = self._store()
        if st is None: return
        st.set("weights.group_assignments", json.dumps(self._group_assignments, ensure_ascii=True))
        st.set("weights.custom_groups", json.dumps(self._custom_groups, ensure_ascii=True))
        st.schedule_save(self.frame)

    def _prompt_new_group(self) -> None:
        name = dialogs.ask_string(self.frame, "New Group", "Group name:")
        if name is None: return  # cancelled
        if not (name := name.strip()) or "::" in name:
            dialogs.show_error(self.frame, "Invalid Group Name", "Group name must be non-empty and not contain '::'.")
            return
        if name in self._available_groups():
            dialogs.show_info(self.frame, "Group Exists", f"Group already exists: {name}")
            return
        self._custom_groups.append(name)
        self._save_group_state()
        self._refresh_models()
        self._status(f"Group created: {name}")

    def _duplicate_selected(self) -> None:
        paths = self._selected_model_paths()
        if not dialogs.require(len(paths) == 1, self.frame, "Select exactly one checkpoint to duplicate."): return
        src = paths[0]
        if not (new_name := dialogs.ask_string(self.frame, "Duplicate Checkpoint", "New checkpoint name (without extension):", initial=src.stem)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name:
            dialogs.show_error(self.frame, "Invalid Name", "Name must be non-empty and not contain path separators.")
            return
        dst = src.with_name(new_name + src.suffix)
        if dst.exists():
            dialogs.show_error(self.frame, "Target Exists", f"Target already exists:\n{dst}")
            return
        try:
            self.logic.duplicate_model(src, dst)
        except Exception as e:
            dialogs.show_error(self.frame, "Duplicate Failed", e)
            return
        
        rel_src, rel_dst = self.logic.rel_path(src), self.logic.rel_path(dst)
        if rel_src in self._group_assignments: self._group_assignments[rel_dst] = self._group_assignments[rel_src]
        if self._is_favorited(src):
            self._favorites[rel_dst] = {"path": rel_dst, "added": now_iso(timespec="seconds")}
            self.logic.save_favorites(self._favorites)
        self._save_group_state()
        self.ui.append_log(f"[duplicate] {rel_src} -> {rel_dst}\n")
        self._status(f"Duplicated: {rel_dst}")
        self._refresh_models()

    def _on_tree_mouse_down(self, event: tk.Event) -> None:
        iid = self.ui.tree.identify_row(event.y)
        if iid and not self._is_group_header(iid):
            self._drag_src_iid = iid
        else:
            self._drag_src_iid = None

    def _on_tree_mouse_up(self, event: tk.Event) -> None:
        if not self._drag_src_iid: return
        target = self.ui.tree.identify_row(event.y)
        if not target or not self._is_group_header(target):
            self._drag_src_iid = None
            return
        if group := self._group_from_iid(target): self._assign_selected_to_group(self._selected_model_paths(), group)
        self._drag_src_iid = None

    def _on_tree_select(self, _evt: object | None = None) -> None:
        if not (sel := self.ui.tree.selection()): return
        if not (p := self._model_by_iid.get(sel[0])): return

        profile_info = self._profile_info_for(p)
        self.ui.var_selected_model.set(f"selected: {self.logic.rel_path(p)}{profile_info}")
        if self.ui.var_report_scope.get() == "selected_model": self._refresh_reports()

    def _profile_info_for(self, p: Path) -> str:
        """Profile suffix for the selection label; torch.load results are cached by (path, mtime)."""
        try: key: tuple[str, float] = (str(p.resolve()), p.stat().st_mtime)
        except Exception: key = (str(p), -1.0)
        if (cached := self._profile_info_cache.get(key)) is not None: return cached
        try: profile_info = profile_label_suffix(load_checkpoint(p))
        except Exception: profile_info = " | Profile: ⚠ Error loading"
        self._profile_info_cache[key] = profile_info
        return profile_info

    def _on_models_right_click(self, event: tk.Event) -> None:
        self._refresh_groups_menu()
        if not self.ui._models_menu: return
        
        iid = self.ui.tree.identify_row(event.y)
        if iid and iid not in self.ui.tree.selection(): self.ui.tree.selection_set(iid)
        
        paths = self._selected_model_paths()
        can_delete = any(self.logic.is_deletable_checkpoint(p) and not self._is_favorited(p) and not self._is_arena_tracked(p) for p in paths)
        can_run = bool(paths) and bool(self.state.dataset_dir)
        
        self.ui._models_menu.entryconfigure("Delete Selected", state="normal" if can_delete else "disabled")
        self.ui._models_menu.entryconfigure("Run Selected", state="normal" if can_run else "disabled")
        self.ui._models_menu.entryconfigure("Activate Selected", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Export Selected…", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Set as Compare A", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Set as Compare B", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Rename Selected…", state="normal" if any(self.logic.is_deletable_checkpoint(p) for p in paths) else "disabled")
        
        if paths:
            is_fav = self._is_favorited(paths[0])
            self.ui._models_menu.entryconfigure("Add to Favorites", state="disabled" if is_fav else "normal")
            self.ui._models_menu.entryconfigure("Remove from Favorites", state="normal" if is_fav else "disabled")
            is_arena = self._is_arena_tracked(paths[0])
            self.ui._models_menu.entryconfigure("Add to Arena", state="disabled" if is_arena else "normal")
            self.ui._models_menu.entryconfigure("Remove from Arena", state="normal" if is_arena else "disabled")
            self.ui._models_menu.entryconfigure("Generate Arena Report", state="normal")
        else:
            self.ui._models_menu.entryconfigure("Add to Favorites", state="disabled")
            self.ui._models_menu.entryconfigure("Remove from Favorites", state="disabled")
            self.ui._models_menu.entryconfigure("Add to Arena", state="disabled")
            self.ui._models_menu.entryconfigure("Remove from Arena", state="disabled")
            self.ui._models_menu.entryconfigure("Generate Arena Report", state="normal")
            
        try: self.ui._models_menu.tk_popup(event.x_root, event.y_root)
        finally: self.ui._models_menu.grab_release()

    def _hide_models_menu(self, _event: tk.Event | None = None) -> None:
        with contextlib.suppress(Exception): self.ui._models_menu.unpost()
        with contextlib.suppress(Exception): self.ui._reports_menu.unpost()

    def _on_reports_right_click(self, event: tk.Event) -> None:
        if not self.ui._reports_menu: return
        iid = self.ui.tree_reports.identify_row(event.y)
        if iid and iid not in self.ui.tree_reports.selection(): self.ui.tree_reports.selection_set(iid)
        try: self.ui._reports_menu.tk_popup(event.x_root, event.y_root)
        finally: self.ui._reports_menu.grab_release()

    def _delete_selected_report(self) -> None:
        if not dialogs.require(sel := self.ui.tree_reports.selection(), self.frame, "Select a report first."): return None
        if not (r := self._report_by_iid.get(sel[0])) or not (p := r.get("_path")): return dialogs.show_error(self.frame, "Delete Report", "Report path missing.")
        rp = Path(str(p))
        if not rp.exists(): return dialogs.show_warning(self.frame, "Report Not Found", f"Report file not found:\n{rp}")

        try: rp.resolve().relative_to(self.logic.sim_root.resolve())  # Check if rp is within sim_root
        except ValueError: return dialogs.show_error(self.frame, "Delete Blocked", f"Refusing to delete file outside repo:\n{rp}")

        if not dialogs.confirm(self.frame, "Delete Report", f"Delete report file {rp.name}?\n\n{rp}", danger=True): return None
        try: rp.unlink()
        except Exception as e:
            dialogs.show_error(self.frame, "Delete Failed", f"Failed to delete:\n{rp}\n\n{e}")
        else:
            self._status(f"Report deleted: {rp.name}")

        self.ui.append_log(f"[delete] report: {rp}\n")
        self._refresh_reports()

    def _set_selected_as_compare_a(self) -> None:
        if p := self._selected_model_path(): self.ui.var_cmp_a.set(self.logic.rel_path(p))

    def _set_selected_as_compare_b(self) -> None:
        if p := self._selected_model_path(): self.ui.var_cmp_b.set(self.logic.rel_path(p))

    def _active_model_path(self) -> Path | None:
        if not (s := self.ui.var_active_model.get().strip()): return None
        return Path(s)

    def _snapshot_active(self) -> None:
        if not (active := self._active_model_path()) or not active.exists():
            dialogs.show_error(self.frame, "Active Model Not Found", f"Active model not found:\n{active}",
                               hint="Train a model for this dataset or enter the checkpoint path under Active Model.")
            return
        try:
            dst = self.logic.snapshot_active_model(active)
            self.ui.append_log(f"[snapshot] {active} -> {dst}\n")
            self._refresh_models()
            self._status(f"Snapshot saved: {self.logic.rel_path(dst)}")
        except Exception as e:
            dialogs.show_error(self.frame, "Snapshot Failed", e)

    def _import_model(self) -> None:
        src = filedialog.askopenfilename(parent=self.frame.winfo_toplevel(), title="Import model (.pt) or bundle (.zip)", filetypes=[("PyTorch checkpoint", "*.pt"), ("Bundle archive", "*.zip"), ("All files", "*.*")])
        if not src or not (src_p := Path(src)).exists(): return
        try:
            dst = self.logic.import_model(src_p)
            self.ui.append_log(f"[import] {src_p} -> {dst}\n")
            self._refresh_models()
            self._status(f"Imported: {Path(dst).name}")
        except Exception as e:
            dialogs.show_error(self.frame, "Import Failed", e)

    def _export_selected(self) -> None:
        if not dialogs.require(p := self._selected_model_path(), self.frame, "Select a model first."): return
        try:
            if p.is_dir():
                dst = filedialog.asksaveasfilename(parent=self.frame.winfo_toplevel(), title="Export selected bundle (zip)", initialfile=f"{p.name}.zip" if not p.name.endswith(".zip") else p.name, defaultextension=".zip", filetypes=[("Bundle archive", "*.zip"), ("All files", "*.*")])
                if not dst: return
                self.logic.export_model(p, Path(dst))
                self.ui.append_log(f"[export] bundle {p} -> {Path(dst).with_suffix('.zip')}\n")
                self._status(f"Exported bundle: {Path(dst).with_suffix('.zip').name}")
            else:
                dst = filedialog.asksaveasfilename(parent=self.frame.winfo_toplevel(), title="Export selected model", initialfile=p.name, defaultextension=".pt", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
                if not dst: return
                self.logic.export_model(p, Path(dst))
                self.ui.append_log(f"[export] {p} -> {dst}\n")
                self._status(f"Exported: {Path(dst).name}")
        except Exception as e:
            dialogs.show_error(self.frame, "Export Failed", e)

    def _activate_selected(self) -> None:
        if not dialogs.require(src := self._selected_model_path(), self.frame, "Select a model first."): return
        if not (dst := self._active_model_path()): return
        if dst.exists() and not dialogs.confirm(self.frame, "Overwrite Active Model", f"Overwrite the active model {dst.name}?\n\n{dst}\n\nWith:\n{src}", danger=True, ok_text="Overwrite"): return
        try:
            self.logic.activate_model(src, dst)
            self.ui.append_log(f"[activate] {src} -> {dst}\n")
            self.state.current_model_path = dst
            self._refresh_models()
            self._status(f"Activated: {src.name} -> {dst.name}")
        except Exception as e:
            dialogs.show_error(self.frame, "Activate Failed", e)

    def _delete_selected(self) -> None:
        paths = self._selected_model_paths()
        if not dialogs.require(paths, self.frame, "Select one or more snapshots to delete."): return

        fav, arena, deletable, blocked = [], [], [], []
        for p in paths:
            if self._is_favorited(p): fav.append(p)
            elif self._is_arena_tracked(p): arena.append(p)
            elif self.logic.is_deletable_checkpoint(p): deletable.append(p)
            else: blocked.append(p)

        if fav:
            dialogs.show_warning(self.frame, "Favorited", f"These selected checkpoint(s) are favorited and cannot be deleted from here.\n\n{self._fmt_list([self.logic.rel_path(p) for p in fav])}\n\nRemove from favorites first, then delete.")
        if arena:
            dialogs.show_warning(self.frame, "Arena-Tracked", f"These selected checkpoint(s) are tracked in the Arena and cannot be deleted from here.\n\n{self._fmt_list([self.logic.rel_path(p) for p in arena])}\n\nRemove from Arena first, then delete.")
        if blocked:
            dialogs.show_warning(self.frame, "Not Deletable", f"Some selected files are not deletable here (only versions/imports are allowed):\n\n{self._fmt_list([str(p) for p in blocked])}")
        if not deletable: return

        if not dialogs.confirm(self.frame, "Delete Checkpoints", f"Delete {len(deletable)} checkpoint(s)/bundle(s)?\n\n{self._fmt_list([self.logic.rel_path(p) for p in deletable])}\n\nThis will also delete any adjacent *.meta.json file (for .pt checkpoints).", danger=True): return

        deleted = 0
        for p in deletable:
            try:
                self.logic.delete_model(p)
                deleted += 1
            except Exception as e: self.ui.append_log(f"[delete] failed: {p} ({e})\n")
        self.ui.append_log(f"[delete] deleted {deleted} item(s)\n")
        self._status(f"Deleted {deleted} of {len(deletable)} checkpoint(s)", "success" if deleted == len(deletable) else "warning")
        self._refresh_models()

    def _show_model_profile_info(self) -> None:
        if not dialogs.require(p := self._selected_model_path(), self.frame, "Select a model first.", title="Model Profile"): return
        info_text, err = self.logic.show_model_profile_info(p)
        if err: dialogs.show_error(self.frame, "Model Profile Failed", f"Failed to load model profile info:\n{err}")
        else: dialogs.show_text(self.frame, f"Model Profile: {p.name}", info_text, size=(650, 550))

    def _rename_selected(self) -> None:
        paths = self._selected_model_paths()
        if not dialogs.require(len(paths) == 1, self.frame, "Select exactly one checkpoint to rename."): return None
        src = paths[0]
        if not self.logic.is_deletable_checkpoint(src): return dialogs.show_warning(self.frame, "Rename Not Allowed", "Only snapshots/imports can be renamed from here.")

        old_rel, old_name = self.logic.rel_path(src), src.stem
        new_stem = dialogs.ask_string(self.frame, "Rename Checkpoint", "New name (no path, no extension):", initial=old_name)
        if new_stem is None: return None  # cancelled
        if not (new_stem := new_stem.strip()) or "/" in new_stem or "\\" in new_stem: return dialogs.show_error(self.frame, "Invalid Name", "Name must be non-empty and not contain path separators.")

        dst = src.with_name(new_stem + src.suffix)
        if dst.exists(): return dialogs.show_error(self.frame, "Target Exists", f"Target already exists:\n{dst}")

        was_fav = old_rel in self._favorites; fav_entry = self._favorites.get(old_rel)
        was_arena = old_rel in self._arena; arena_entry = self._arena.get(old_rel)

        try:
            self.logic.rename_model(src, dst)
        except Exception as e: return dialogs.show_error(self.frame, "Rename Failed", e)

        if was_fav and fav_entry:
            self._favorites.pop(old_rel, None)
            fav_entry = dict(fav_entry)
            fav_entry["path"], fav_entry["renamed"] = self.logic.rel_path(dst), now_iso(timespec="seconds")
            self._favorites[fav_entry["path"]] = fav_entry
            self.logic.save_favorites(self._favorites)
        if was_arena and arena_entry:
            self._arena.pop(old_rel, None)
            arena_entry = dict(arena_entry)
            arena_entry["path"], arena_entry["renamed"] = self.logic.rel_path(dst), now_iso(timespec="seconds")
            self._arena[arena_entry["path"]] = arena_entry
            self.logic.save_arena(self._arena)
        if old_rel in self._group_assignments:
            self._group_assignments[self.logic.rel_path(dst)] = self._group_assignments.pop(old_rel)
            self._save_group_state()

        self.ui.append_log(f"[rename] {old_rel} -> {self.logic.rel_path(dst)}\n")
        self._status(f"Renamed: {src.name} -> {dst.name}")
        self._refresh_models()

    def _stop(self) -> None:
        self.logic.stop()
        self.ui.var_status.set("status: stopping...")

    def _path_from_combo(self, rel: str) -> Path | None:
        if not rel: return None
        p = (self.logic.sim_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if p.exists(): return p
        for mp in self._models:
            if self.logic.rel_path(mp) == rel: return mp
        return None

    def _run_selected(self) -> None:
        if not dialogs.require(p := self._selected_model_path(), self.frame, "Select a model first."): return None
        if not dialogs.require(self.state.dataset_dir, self.frame, "No dataset selected. Choose one in the Pipeline tab first."): return None
        if self._run_in_progress(): return dialogs.show_info(self.frame, "Run In Progress", "A run is already in progress.")

        try:
            self.ui.var_status.set(f"status: running (selected({p.name}))")
            self._busy = True
            self.logic.run_predict(p, self.state.dataset_dir, f"selected({p.name})", self.ui.var_split.get().strip(), self.ui.var_device.get().strip(), self.ui.var_max_samples.get().strip(), self.ui.chk_save_preds.get(), lambda *res: self.post(lambda: self._on_predict_complete(*res)))
        except Exception as e:
            self._busy = False
            self.ui.var_status.set("status: failed")
            dialogs.show_error(self.frame, "Evaluation Failed", e)

    def _resolved(self, p: Any) -> str:
        p = Path(str(p))
        try: return str((p if p.is_absolute() else self.logic.sim_root / p).resolve())
        except OSError: return str(p)

    def _run_in_progress(self) -> bool:
        """A run was started from this tab and has not finished.

        The child process alone is no guard: it only exists after the worker's Popen, and a compare
        has gaps between its runs, so a second click could start a concurrent run.
        """
        return bool(getattr(self, "_busy", False)) or bool(self.logic.proc and self.logic.proc.poll() is None)

    def _on_predict_complete(self, report_path: Path | None, was_stopped: bool) -> None:
        self._busy = False
        if was_stopped: self.ui.var_status.set("status: stopped")
        # No report without a stop: batch-predict failed (class mismatch, missing checkpoint, ...); the log has the error.
        elif not (report_path and report_path.exists()): self.ui.var_status.set("status: failed (see log)")
        else: self.ui.var_status.set("status: done")
        if report_path and report_path.exists(): self.ui.append_log(self.logic.short_report(report_path))
        self._refresh_reports()

    def _run_compare(self) -> None:
        a = self._path_from_combo(self.ui.var_cmp_a.get().strip())
        b = self._path_from_combo(self.ui.var_cmp_b.get().strip())
        if not dialogs.require(a and b, self.frame, "Choose two models to compare (Model A and Model B)."): return None
        if self._run_in_progress(): return dialogs.show_info(self.frame, "Run In Progress", "A run is already in progress.")

        # Read Tk state on the main thread; the worker must not touch widgets/variables.
        ds_dirs = self._selected_compare_datasets()
        if not dialogs.require(ds_dirs, self.frame, "Select at least one dataset to compare the models on."):
            return None
        split = self.ui.var_split.get().strip()
        device = self.ui.var_device.get().strip()
        max_samples_s = self.ui.var_max_samples.get().strip()
        save_preds = self.ui.chk_save_preds.get()

        def worker(task: Any) -> None:
            paired: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
            cls_a, cls_b = self.logic.get_model_classes(a), self.logic.get_model_classes(b)

            for ds in ds_dirs:
                if self.logic.stop_evt.is_set() or task.cancelled(): break
                if (ds_classes := self._dataset_classes(ds)):
                    if cls_a and sorted(ds_classes) != sorted(cls_a):
                        self.log_q.put(f"\n[skip] A({a.name}) incompatible with dataset {ds.name}: model classes={cls_a} dataset classes={ds_classes}\n")
                        continue
                    if cls_b and sorted(ds_classes) != sorted(cls_b):
                        self.log_q.put(f"\n[skip] B({b.name}) incompatible with dataset {ds.name}: model classes={cls_b} dataset classes={ds_classes}\n")
                        continue

                ra = self.logic.run_predict_blocking(a, ds, f"A({a.name})", split, device, max_samples_s, save_preds)
                rb = self.logic.run_predict_blocking(b, ds, f"B({b.name})", split, device, max_samples_s, save_preds)
                
                if ra and rb:
                    paired.append((ds, ra, rb))
                    self.log_q.put(self.logic.compare_summary(ra, rb, model_a=a, model_b=b, split=split))
                else: self.log_q.put(f"\n[skip] compare incomplete for dataset {ds.name} (see logs above)\n")

            if len(paired) > 1: self.log_q.put(self.logic.compare_summary_multi(paired, model_a=a, model_b=b, split=split))

        def on_done(_result: Any) -> None:
            stopped = self.logic.stop_evt.is_set()
            self.ui.var_status.set("status: stopped" if stopped else "status: done")
            if not stopped: self._status(f"Compare finished: {a.name} vs {b.name}")
            self._refresh_reports()

        def on_error(exc: BaseException) -> None:
            self.log_q.put(f"\n[error] compare failed: {exc}\n")
            self.ui.var_status.set("status: failed")
            self._status(f"Compare failed: {exc}", "error")

        def on_finally() -> None:
            self._busy = False

        self.logic.stop_evt.clear()
        self.ui.var_status.set("status: running compare")
        self._busy = True
        try:
            self._run_background(worker, name="Compare models", on_done=on_done, on_error=on_error, on_finally=on_finally)
        except BaseException:
            self._busy = False
            raise

    def _refresh_reports(self) -> None:
        self._reports = self.logic.load_reports()
        
        scope = self.ui.var_report_scope.get().strip()
        split_filter = self.ui.var_report_split.get().strip()
        sort_key = self.ui.var_report_sort.get().strip()

        selected_model = self._selected_model_path()
        ds_dir = self.state.dataset_dir

        filtered: list[dict[str, Any]] = []
        for r in self._reports:
            r_split = str(r.get("split") or "-")
            if split_filter != "any" and r_split != split_filter: continue
            
            if scope == "selected_model":
                if not selected_model: continue
                rp = str(r.get("model_path") or "")
                matched = False
                if Path(rp).name == selected_model.name: matched = True
                elif str(r.get("model_stem") or "") == selected_model.stem: matched = True
                elif selected_model.is_dir():
                    try:
                        report_model, bundle_dir = Path(rp).resolve(), selected_model.resolve()
                        if report_model.parent == bundle_dir: matched = True
                    except Exception: pass
                if not matched: continue
            elif scope == "dataset":
                # Resolved: reports store the path as given (relative from the CLI, absolute from the GUI).
                if not ds_dir or not r.get("dataset_path") or self._resolved(r["dataset_path"]) != self._resolved(ds_dir): continue
            filtered.append(r)
        
        if sort_key in ("accuracy", "macro_f1"): filtered.sort(key=lambda rr: metric_value(rr, sort_key) or 0.0, reverse=True)
        else: filtered.sort(key=report_timestamp, reverse=True)
        
        reports_data = []
        self._report_by_iid.clear()
        for i, r in enumerate(filtered):
            iid = f"r{i}"
            when = (r.get("timestamp") or "-").replace("T", " ")[:19]
            split = str(r.get("split") or "-")
            seen_v = r.get("seen_samples") or r.get("test_size")
            seen = str(int(seen_v)) if seen_v is not None else "-"
            acc, f1 = r.get("metrics", {}).get("accuracy"), r.get("metrics", {}).get("macro_f1")
            acc_s = f"{float(acc):.4f}" if acc is not None else "-"
            f1_s = f"{float(f1):.4f}" if f1 is not None else "-"
            model_name = Path(str(r.get("model_path") or "-")).name
            dataset_name = Path(str(r.get("dataset_path") or "-")).name
            path_disp = self.logic.rel_path(Path(str(r.get("_path") or "-"))) if r.get("_path") != "-" else "-"
            reports_data.append((iid, when, split, seen, acc_s, f1_s, model_name, dataset_name, path_disp))
            self._report_by_iid[iid] = r
        self.ui.populate_reports_tree(reports_data)

    def _show_selected_report_summary(self) -> None:
        if not dialogs.require(sel := self.ui.tree_reports.selection(), self.frame, "Select a report first."): return
        if not (r := self._report_by_iid.get(sel[0])): return
        if (p := r.get("_path")) and Path(str(p)).exists(): self.ui.append_log(self.logic.short_report(Path(str(p))))
        else:
            try:
                m = r.get("metrics", {})
                acc, f1 = float(m.get("accuracy", 0.0)), float(m.get("macro_f1", 0.0))
                self.ui.append_log(f"\n[report] ts={r.get('timestamp')} split={r.get('split')} acc={acc:.4f} f1={f1:.4f} path={p}\n")
            except Exception: pass
