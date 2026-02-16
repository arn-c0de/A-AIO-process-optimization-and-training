"""Weights Tab - Model checkpoint versioning/backup/export + comparisons."""

from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore
from .ui import WeightsUI
from .logic import WeightsLogic
from datetime import datetime


class WeightsTab(BaseTab):
    """Tab: manage model weights (snapshots/import/export) and run evaluations."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.log_q: queue.Queue[str] = queue.Queue()
        self.ui = WeightsUI(self, self.frame)
        self.logic = WeightsLogic(sim_root, self.log_q)

        self._models: List[Path] = []
        self._model_by_iid: Dict[str, Path] = {}
        self._favorites: Dict[str, Dict[str, Any]] = {}
        self._arena: Dict[str, Dict[str, Any]] = {}

        self._group_assignments: Dict[str, str] = {}
        self._custom_groups: List[str] = []
        self._group_iids: Dict[str, str] = {}
        self._drag_src_iid: Optional[str] = None

        self._reports: List[Dict[str, Any]] = []
        self._report_by_iid: Dict[str, Dict[str, Any]] = {}

        self._cmp_dataset_dirs: List[Path] = []
        self._cmp_dataset_labels: List[str] = []
        self._cmp_dataset_by_label: Dict[str, Path] = {}

        self._ui_tick_id: Optional[str] = None

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
        self._tick_ui()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir: return
        self.ui.var_dataset.set(str(self.state.dataset_dir))
        active = self.logic._model_root() / f"{self.state.dataset_dir.name}.pt"
        self.ui.var_active_model.set(str(active))
        self.ui.var_dataset_samples.set(self.logic.describe_dataset_samples(self.state.dataset_dir))
        if hasattr(self.ui, "list_cmp_datasets") and self.ui.list_cmp_datasets.size() > 0:
            if not self.ui.list_cmp_datasets.curselection(): self._select_current_dataset_for_compare()

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None: return
        for key, var in [("weights.split", self.ui.var_split), ("weights.device", self.ui.var_device),
                         ("weights.max_samples", self.ui.var_max_samples), ("weights.save_preds", self.ui.chk_save_preds),
                         ("weights.report_scope", self.ui.var_report_scope), ("weights.report_split", self.ui.var_report_split),
                         ("weights.report_sort", self.ui.var_report_sort), ("weights.compare_a", self.ui.var_cmp_a),
                         ("weights.compare_b", self.ui.var_cmp_b)]:
            if v := st.get(key):
                try: var.set(v)
                except Exception: pass
        
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

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None: return
        for var, key in [(self.ui.var_split, "weights.split"), (self.ui.var_device, "weights.device"),
                         (self.ui.var_max_samples, "weights.max_samples"), (self.ui.chk_save_preds, "weights.save_preds"),
                         (self.ui.var_report_scope, "weights.report_scope"), (self.ui.var_report_split, "weights.report_split"),
                         (self.ui.var_report_sort, "weights.report_sort"), (self.ui.var_cmp_a, "weights.compare_a"),
                         (self.ui.var_cmp_b, "weights.compare_b")]:
            var.trace_add("write", lambda *a, v=var, k=key: (st.set(k, v.get()), st.schedule_save(self.frame)))

    def _tick_ui(self) -> None:
        while True:
            try: self.ui.append_log(self.log_q.get_nowait())
            except queue.Empty: break
        self._ui_tick_id = self.frame.after(150, self._tick_ui)

    def _refresh_cmp_datasets(self) -> None:
        cand = self.logic.get_cmp_datasets()
        self._cmp_dataset_dirs = cand
        
        old_sel = {str(self.ui.list_cmp_datasets.get(i)) for i in self.ui.list_cmp_datasets.curselection()}

        self._cmp_dataset_labels = []
        self._cmp_dataset_by_label = {}
        seen: dict[str, int] = {}
        for p in cand:
            runs, versions = self.logic._model_root().parent / "sim_data" / "runs", self.logic._model_root().parent / "sim_data" / "versions"
            base = self.logic._display_for_dataset(p, runs=runs, versions=versions)
            n = seen.get(base, 0) + 1
            seen[base] = n
            label = base if n == 1 else f"{base} ({n})"
            self._cmp_dataset_labels.append(label)
            self._cmp_dataset_by_label[label] = p

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
        runs, versions = self.logic._model_root().parent / "sim_data" / "runs", self.logic._model_root().parent / "sim_data" / "versions"
        want = self.logic._display_for_dataset(self.state.dataset_dir, runs=runs, versions=versions)
        for idx, label in enumerate(self._cmp_dataset_labels):
            if label == want or (p := self._cmp_dataset_by_label.get(label)) == self.state.dataset_dir:
                self.ui.list_cmp_datasets.selection_set(idx)
                self.ui.list_cmp_datasets.see(idx)
                return

    def _select_all_datasets_for_compare(self) -> None:
        try: self.ui.list_cmp_datasets.selection_set(0, "end")
        except Exception: pass

    def _selected_compare_datasets(self) -> List[Path]:
        ds: List[Path] = []
        try:
            for i in self.ui.list_cmp_datasets.curselection():
                label = str(self.ui.list_cmp_datasets.get(i))
                if p := self._cmp_dataset_by_label.get(label): ds.append(p)
        except Exception: pass
        if not ds and self.state.dataset_dir: ds = [self.state.dataset_dir]
        return [p for p in ds if p.exists() and p.is_dir()]

    def _selected_model_path(self) -> Optional[Path]:
        sel = self.ui.tree.selection()
        if not sel: return None
        return self._model_by_iid.get(sel[0])

    def _selected_model_paths(self) -> List[Path]:
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
        if not messagebox.askyesno("Remove favorite", f"Remove from favorites?\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in paths[:8])}\\n... (+{len(paths) - 8} more)" if len(paths) > 8 else f"Remove from favorites?\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in paths)}"): return
        self._assign_selected_to_group(paths, "Uncategorized")

    def _add_selected_to_arena(self) -> None:
        paths = self._selected_model_paths()
        if not paths: return
        self.logic.add_to_arena(paths, self._arena)
        self.logic.save_arena(self._arena)
        self.ui.append_log(f"[arena] added {len(paths)} model(s)\\n")
        self._refresh_models()

    def _remove_selected_from_arena(self) -> None:
        paths = [p for p in self._selected_model_paths() if self._is_arena_tracked(p)]
        if not paths: return
        if not messagebox.askyesno("Remove from arena", f"Remove from arena tracking?\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in paths[:8])}\\n... (+{len(paths) - 8} more)" if len(paths) > 8 else f"Remove from arena tracking?\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in paths)}"): return
        self.logic.remove_from_arena(paths, self._arena)
        self.logic.save_arena(self._arena)
        self.ui.append_log(f"[arena] removed {len(paths)} model(s)\\n")
        self._refresh_models()

    def _generate_arena_report(self) -> None:
        try: self.logic.generate_arena_report()
        except FileNotFoundError as e: messagebox.showerror("Missing script", str(e))
        except Exception as e: messagebox.showerror("Error", f"Failed to generate arena report:\\n{e}")

    def _refresh_models(self) -> None:
        self._models = self.logic.get_models()
        self._models.sort(key=lambda p: (0 if self._is_favorited(p) else 1, -(p.stat().st_mtime if p.exists() else 0.0)))

        prev_open_groups = {child for child in self.ui.tree.get_children() if child.startswith("group::") and self.ui.tree.item(child, "open")}
        prev_selected_paths = {str(self._model_by_iid[iid].resolve()) for iid in self.ui.tree.selection() if iid in self._model_by_iid}
        
        self.ui.tree.delete(*self.ui.tree.get_children())
        self._model_by_iid.clear()
        self._group_iids.clear()

        group_names = self._available_groups()
        group_members: Dict[str, List[Path]] = {gn: [] for gn in group_names}
        for p in self._models:
            group = self._group_for_path(p)
            group_members.setdefault(group, []).append(p)
        
        display_values: List[str] = []
        restore_selection: List[str] = []
        item_counter = 0

        for group in group_names:
            group_iid = f"group::{group}"
            self.ui.tree.insert("", "end", iid=group_iid, text=f"{group} ({len(group_members.get(group, []))})", values=("",)*5 + (f"{len(group_members.get(group, []))} models",), tags=("group_header",), open=group_iid in prev_open_groups)
            self._group_iids[group] = group_iid
            
            for p in group_members.get(group, []):
                st = p.stat()
                is_bundle = p.is_dir()
                size_s = self.ui.fmt_bytes(int(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if is_bundle else st.st_size))
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

    def _available_groups(self) -> List[str]:
        names: List[str] = ["Favorites", "Snapshots", "Imports"]
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

    def _group_from_iid(self, iid: str) -> Optional[str]:
        if not iid or "::" not in iid: return None
        return iid.split("::", 1)[1]

    def _refresh_groups_menu(self) -> None:
        if not self.ui._groups_menu: return
        self.ui._groups_menu.delete(0, "end")
        for group in self._available_groups():
            self.ui._groups_menu.add_command(label=group, command=lambda g=group: self._assign_selected_to_group(self._selected_model_paths(), g))

    def _assign_selected_to_group(self, paths: List[Path], group: Optional[str]) -> None:
        if not paths or not group: return
        fav_changed = False
        for p in paths:
            rel = self.logic.rel_path(p)
            if group == "Favorites":
                if rel not in self._favorites:
                    self._favorites[rel] = {"path": rel, "added": datetime.now().isoformat(timespec="seconds")}
                    fav_changed = True
            else:
                if rel in self._favorites:
                    self._favorites.pop(rel, None)
                    fav_changed = True
            self._group_assignments[rel] = group
        if fav_changed: self.logic.save_favorites(self._favorites)
        self._save_group_state()
        self.ui.append_log(f"[group] assigned {len(paths)} model(s) -> {group}\\n")
        self._refresh_models()

    def _save_group_state(self) -> None:
        st = self._store()
        if st is None: return
        st.set("weights.group_assignments", json.dumps(self._group_assignments, ensure_ascii=True))
        st.set("weights.custom_groups", json.dumps(self._custom_groups, ensure_ascii=True))
        st.schedule_save(self.frame)

    def _prompt_new_group(self) -> None:
        name = simpledialog.askstring("New group", "Group name:", parent=self.frame.winfo_toplevel())
        if not (name := (name or "").strip()) or "::" in name:
            messagebox.showerror("Error", "Group name must be non-empty and not contain '::'.")
            return
        if name in self._available_groups():
            messagebox.showinfo("Info", f"Group already exists: {name}")
            return
        self._custom_groups.append(name)
        self._save_group_state()
        self._refresh_models()

    def _duplicate_selected(self) -> None:
        if not (paths := self._selected_model_paths()) or len(paths) != 1:
            messagebox.showinfo("Info", "Select exactly one checkpoint to duplicate.")
            return
        src = paths[0]
        if not (new_name := simpledialog.askstring("Duplicate checkpoint", "New checkpoint name (without extension):", initialvalue=src.stem, parent=self.frame.winfo_toplevel())): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name:
            messagebox.showerror("Error", "Name must be non-empty and not contain path separators.")
            return
        dst = src.with_name(new_name + src.suffix)
        if dst.exists():
            messagebox.showerror("Error", f"Target already exists:\\n{dst}")
            return
        try:
            self.logic.duplicate_model(src, dst)
        except Exception as e:
            messagebox.showerror("Error", f"Duplicate failed:\\n{e}")
            return
        
        rel_src, rel_dst = self.logic.rel_path(src), self.logic.rel_path(dst)
        if rel_src in self._group_assignments: self._group_assignments[rel_dst] = self._group_assignments[rel_src]
        if self._is_favorited(src):
            self._favorites[rel_dst] = {"path": rel_dst, "added": datetime.now().isoformat(timespec="seconds")}
            self.logic.save_favorites(self._favorites)
        self._save_group_state()
        self.ui.append_log(f"[duplicate] {rel_src} -> {rel_dst}\\n")
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

    def _on_tree_select(self, _evt: Optional[object] = None) -> None:
        if not (sel := self.ui.tree.selection()): return
        if not (p := self._model_by_iid.get(sel[0])): return

        profile_info = "";
        try:
            checkpoint = torch.load(p, map_location='cpu')
            profile_data = checkpoint.get('component_profile')
            if profile_data:
                profile_id = profile_data.get('profile_id', 'unknown')
                profile_hash = profile_data.get('profile_hash', '')
                hash_short = profile_hash.split(':')[1][:12] if ':' in profile_hash else profile_hash[:12]
                profile_info = f" | Profile: {profile_id} ({hash_short}...)"
            else: profile_info = " | Profile: ⚠ No profile (legacy)"
        except Exception: profile_info = " | Profile: ⚠ Error loading"
        self.ui.var_selected_model.set(f"selected: {self.logic.rel_path(p)}{profile_info}")
        if self.ui.var_report_scope.get() == "selected_model": self._refresh_reports()

    def _on_models_right_click(self, event: tk.Event) -> None:
        self._refresh_groups_menu()
        if not self.ui._models_menu: return
        
        iid = self.ui.tree.identify_row(event.y)
        if iid and iid not in self.ui.tree.selection(): self.ui.tree.selection_set(iid)
        
        paths = self._selected_model_paths()
        can_delete = any(self.logic.is_deletable_checkpoint(p) and not self._is_favorited(p) and not self._is_arena_tracked(p) for p in paths)
        can_run = bool(paths) and bool(self.state.dataset_dir)
        
        self.ui._models_menu.entryconfigure("Delete selected", state="normal" if can_delete else "disabled")
        self.ui._models_menu.entryconfigure("Run selected", state="normal" if can_run else "disabled")
        self.ui._models_menu.entryconfigure("Activate selected", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Export selected...", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Set as Compare A", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Set as Compare B", state="normal" if bool(paths) else "disabled")
        self.ui._models_menu.entryconfigure("Rename selected...", state="normal" if any(self.logic.is_deletable_checkpoint(p) for p in paths) else "disabled")
        
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

    def _hide_models_menu(self, _event: Optional[tk.Event] = None) -> None:
        try: self.ui._models_menu.unpost()
        except Exception: pass
        try: self.ui._reports_menu.unpost()
        except Exception: pass

    def _on_reports_right_click(self, event: tk.Event) -> None:
        if not self.ui._reports_menu: return
        iid = self.ui.tree_reports.identify_row(event.y)
        if iid and iid not in self.ui.tree_reports.selection(): self.ui.tree_reports.selection_set(iid)
        try: self.ui._reports_menu.tk_popup(event.x_root, event.y_root)
        finally: self.ui._reports_menu.grab_release()

    def _delete_selected_report(self) -> None:
        if not (sel := self.ui.tree_reports.selection()): return messagebox.showinfo("Info", "Select a report first.")
        if not (r := self._report_by_iid.get(sel[0])) or not (p := r.get("_path")): return messagebox.showerror("Error", "Report path missing.")
        rp = Path(str(p))
        if not rp.exists(): return messagebox.showwarning("Not found", f"Report file not found:\\n{rp}")
        
        try: (self.logic.sim_root.resolve()).relative_to(rp.resolve()) # Check if rp is within sim_root
        except ValueError: return messagebox.showerror("Blocked", f"Refusing to delete file outside repo:\\n{rp}")

        if not messagebox.askyesno("Delete report", f"Delete this report file?\\n\\n{rp}"): return
        try: rp.unlink()
        except Exception as e: messagebox.showerror("Error", f"Failed to delete:\\n{rp}\\n\\n{e}")
        
        self.ui.append_log(f"[delete] report: {rp}\\n")
        self._refresh_reports()

    def _set_selected_as_compare_a(self) -> None:
        if p := self._selected_model_path(): self.ui.var_cmp_a.set(self.logic.rel_path(p))

    def _set_selected_as_compare_b(self) -> None:
        if p := self._selected_model_path(): self.ui.var_cmp_b.set(self.logic.rel_path(p))

    def _active_model_path(self) -> Optional[Path]:
        if not (s := self.ui.var_active_model.get().strip()): return None
        return Path(s)

    def _snapshot_active(self) -> None:
        if not (active := self._active_model_path()) or not active.exists():
            messagebox.showerror("Error", f"Active model not found:\\n{active}")
            return
        try:
            dst = self.logic.snapshot_active_model(active)
            self.ui.append_log(f"[snapshot] {active} -> {dst}\\n")
            self._refresh_models()
        except Exception as e:
            messagebox.showerror("Error", f"Snapshot failed:\\n{e}")

    def _import_model(self) -> None:
        src = filedialog.askopenfilename(title="Import model (.pt) or bundle (.zip)", filetypes=[("PyTorch checkpoint", "*.pt"), ("Bundle archive", "*.zip"), ("All files", "*.*")])
        if not src or not (src_p := Path(src)).exists(): return
        try:
            dst = self.logic.import_model(src_p)
            self.ui.append_log(f"[import] {src_p} -> {dst}\\n")
            self._refresh_models()
        except Exception as e:
            messagebox.showerror("Error", f"Import failed:\\n{e}")

    def _export_selected(self) -> None:
        if not (p := self._selected_model_path()):
            messagebox.showinfo("Info", "Select a model first.")
            return
        try:
            if p.is_dir():
                dst = filedialog.asksaveasfilename(title="Export selected bundle (zip)", initialfile=f"{p.name}.zip" if not p.name.endswith(".zip") else p.name, defaultextension=".zip", filetypes=[("Bundle archive", "*.zip"), ("All files", "*.*")])
                if not dst: return
                self.logic.export_model(p, Path(dst))
                self.ui.append_log(f"[export] bundle {p} -> {Path(dst).with_suffix('.zip')}\\n")
            else:
                dst = filedialog.asksaveasfilename(title="Export selected model", initialfile=p.name, defaultextension=".pt", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
                if not dst: return
                self.logic.export_model(p, Path(dst))
                self.ui.append_log(f"[export] {p} -> {dst}\\n")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\\n{e}")

    def _activate_selected(self) -> None:
        if not (src := self._selected_model_path()): return messagebox.showinfo("Info", "Select a model first.")
        if not (dst := self._active_model_path()): return
        if dst.exists() and not messagebox.askyesno("Overwrite active model?", f"Overwrite active model?\\n\\n{dst}\\n\\nWith:\\n{src}"): return
        try:
            self.logic.activate_model(src, dst)
            self.ui.append_log(f"[activate] {src} -> {dst}\\n")
            self.state.current_model_path = dst
            self._refresh_models()
        except Exception as e:
            messagebox.showerror("Error", f"Activate failed:\\n{e}")

    def _delete_selected(self) -> None:
        paths = self._selected_model_paths()
        if not paths: return messagebox.showinfo("Info", "Select one or more snapshots to delete.")
        
        fav, arena, deletable, blocked = [], [], [], []
        for p in paths:
            if self._is_favorited(p): fav.append(p)
            elif self._is_arena_tracked(p): arena.append(p)
            elif self.logic.is_deletable_checkpoint(p): deletable.append(p)
            else: blocked.append(p)

        if fav:
            messagebox.showwarning("Favorited", f"These selected checkpoint(s) are favorited and cannot be deleted from here.\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in fav[:8])}\\n... (+{len(fav) - 8} more)\\n\\nRemove from favorites first, then delete." if len(fav) > 8 else f"These selected checkpoint(s) are favorited and cannot be deleted from here.\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in fav)}\\n\\nRemove from favorites first, then delete.")
        if arena:
            messagebox.showwarning("Arena-tracked", f"These selected checkpoint(s) are tracked in the Arena and cannot be deleted from here.\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in arena[:8])}\\n... (+{len(arena) - 8} more)\\n\\nRemove from Arena first, then delete." if len(arena) > 8 else f"These selected checkpoint(s) are tracked in the Arena and cannot be deleted from here.\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in arena)}\\n\\nRemove from Arena first, then delete.")
        if blocked:
            messagebox.showwarning("Not deletable", f"Some selected files are not deletable here (only versions/imports are allowed):\\n\\n{os.linesep.join(str(p) for p in blocked[:8])}\\n... (+{len(blocked) - 8} more)" if len(blocked) > 8 else f"Some selected files are not deletable here (only versions/imports are allowed):\\n\\n{os.linesep.join(str(p) for p in blocked)}")
        if not deletable: return

        if not messagebox.askyesno("Delete checkpoints", f"Delete selected checkpoint(s)/bundle(s)?\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in deletable[:8])}\\n... (+{len(deletable) - 8} more)\\n\\nThis will also delete any adjacent *.meta.json file (for .pt checkpoints)." if len(deletable) > 8 else f"Delete selected checkpoint(s)/bundle(s)?\\n\\n{os.linesep.join(self.logic.rel_path(p) for p in deletable)}\\n\\nThis will also delete any adjacent *.meta.json file (for .pt checkpoints)."): return

        deleted = 0
        for p in deletable:
            try:
                self.logic.delete_model(p)
                deleted += 1
            except Exception as e: self.ui.append_log(f"[delete] failed: {p} ({e})\\n")
        self.ui.append_log(f"[delete] deleted {deleted} item(s)\\n")
        self._refresh_models()

    def _show_model_profile_info(self) -> None:
        if not (p := self._selected_model_path()): return messagebox.showinfo("Model Profile", "No model selected")
        info_text, err = self.logic.show_model_profile_info(p)
        if err: messagebox.showerror("Error", f"Failed to load model profile info:\\n{err}")
        else:
            dialog = tk.Toplevel(self.frame)
            dialog.title(f"Model Profile: {p.name}")
            dialog.geometry("650x550")
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)
            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side="right", fill="y")
            text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
            text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)
            text.insert("1.0", info_text)
            text.config(state="disabled")
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

    def _rename_selected(self) -> None:
        if not (paths := self._selected_model_paths()) or len(paths) != 1: return messagebox.showinfo("Info", "Select exactly one checkpoint to rename.")
        src = paths[0]
        if not self.logic.is_deletable_checkpoint(src): return messagebox.showwarning("Not allowed", "Only snapshots/imports can be renamed from here.")
        
        old_rel, old_name = self.logic.rel_path(src), src.stem
        new_stem = simpledialog.askstring("Rename checkpoint", "New name (no path, no extension):", initialvalue=old_name, parent=self.frame.winfo_toplevel())
        if not (new_stem := (new_stem or "").strip()) or "/" in new_stem or "\\" in new_stem: return messagebox.showerror("Error", "Name must be non-empty and not contain path separators.")
        
        dst = src.with_name(new_stem + src.suffix)
        if dst.exists(): return messagebox.showerror("Error", f"Target already exists:\\n{dst}")
        
        was_fav = old_rel in self._favorites; fav_entry = self._favorites.get(old_rel)
        was_arena = old_rel in self._arena; arena_entry = self._arena.get(old_rel)

        try:
            self.logic.rename_model(src, dst)
        except Exception as e: return messagebox.showerror("Error", f"Rename failed:\\n{e}")

        if was_fav and fav_entry:
            self._favorites.pop(old_rel, None)
            fav_entry = dict(fav_entry)
            fav_entry["path"], fav_entry["renamed"] = self.logic.rel_path(dst), datetime.now().isoformat(timespec="seconds")
            self._favorites[fav_entry["path"]] = fav_entry
            self.logic.save_favorites(self._favorites)
        if was_arena and arena_entry:
            self._arena.pop(old_rel, None)
            arena_entry = dict(arena_entry)
            arena_entry["path"], arena_entry["renamed"] = self.logic.rel_path(dst), datetime.now().isoformat(timespec="seconds")
            self._arena[arena_entry["path"]] = arena_entry
            self.logic.save_arena(self._arena)
        if old_rel in self._group_assignments:
            self._group_assignments[self.logic.rel_path(dst)] = self._group_assignments.pop(old_rel)
            self._save_group_state()

        self.ui.append_log(f"[rename] {old_rel} -> {self.logic.rel_path(dst)}\\n")
        self._refresh_models()

    def _stop(self) -> None:
        self.logic.stop_prediction()
        self.ui.var_status.set("status: stopping...")

    def _path_from_combo(self, rel: str) -> Optional[Path]:
        if not rel: return None
        p = (self.logic.sim_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if p.exists(): return p
        for mp in self._models:
            if self.logic.rel_path(mp) == rel: return mp
        return None

    def _run_selected(self) -> None:
        if not (p := self._selected_model_path()): return messagebox.showinfo("Info", "Select a model first.")
        if not self.state.dataset_dir: return messagebox.showinfo("Info", "No dataset selected.")
        if self.logic.proc and self.logic.proc.poll() is None: return messagebox.showinfo("Info", "A run is already in progress.")

        try:
            self.ui.var_status.set(f"status: running (selected({p.name}))")
            self.logic.run_predict(p, self.state.dataset_dir, f"selected({p.name})", self.ui.var_split.get().strip(), self.ui.var_device.get().strip(), self.ui.var_max_samples.get().strip(), self.ui.chk_save_preds.get(), self._on_predict_complete)
        except Exception as e: messagebox.showerror("Error", str(e))

    def _on_predict_complete(self, report_path: Optional[Path], was_stopped: bool) -> None:
        if was_stopped: self.ui.var_status.set("status: stopped")
        else: self.ui.var_status.set("status: done")
        if report_path and report_path.exists(): self.ui.append_log(self.logic.short_report(report_path))
        self._refresh_reports()

    def _run_compare(self) -> None:
        if not (a := self._path_from_combo(self.ui.var_cmp_a.get().strip())) or not (b := self._path_from_combo(self.ui.var_cmp_b.get().strip())): return messagebox.showinfo("Info", "Choose two models to compare.")
        if self.logic.proc and self.logic.proc.poll() is None: return messagebox.showinfo("Info", "A run is already in progress.")

        def worker() -> None:
            ds_dirs = self._selected_compare_datasets()
            if not ds_dirs:
                self.log_q.put("[error] no dataset selected\\n")
                return

            paired: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
            split = self.ui.var_split.get().strip()
            device = self.ui.var_device.get().strip()
            max_samples_s = self.ui.var_max_samples.get().strip()
            save_preds = self.ui.chk_save_preds.get()

            for ds in ds_dirs:
                if self.logic.stop_evt.is_set(): break
                if (ds_classes := self.logic.get_dataset_classes(ds)):
                    if (cls_a := self.logic.get_model_classes(a)) and sorted(ds_classes) != sorted(cls_a):
                        self.log_q.put(f"\\n[skip] A({a.name}) incompatible with dataset {ds.name}: model classes={cls_a} dataset classes={ds_classes}\\n")
                        continue
                    if (cls_b := self.logic.get_model_classes(b)) and sorted(ds_classes) != sorted(cls_b):
                        self.log_q.put(f"\\n[skip] B({b.name}) incompatible with dataset {ds.name}: model classes={cls_b} dataset classes={ds_classes}\\n")
                        continue

                ra = self.logic.run_predict_blocking(a, ds, f"A({a.name})", split, device, max_samples_s, save_preds)
                rb = self.logic.run_predict_blocking(b, ds, f"B({b.name})", split, device, max_samples_s, save_preds)
                
                if ra and rb:
                    paired.append((ds, ra, rb))
                    self.log_q.put(self.logic.compare_summary(ra, rb, model_a=a, model_b=b, split=split))
                else: self.log_q.put(f"\\n[skip] compare incomplete for dataset {ds.name} (see logs above)\\n")

            if len(paired) > 1: self.log_q.put(self.logic.compare_summary_multi(paired, model_a=a, model_b=b, split=split))
            self.frame.after(0, lambda: self.ui.var_status.set("status: done" if not self.logic.stop_evt.is_set() else "status: stopped"))

        self.logic.stop_evt.clear()
        self.ui.var_status.set("status: running compare")
        threading.Thread(target=worker, daemon=True).start()

    def _refresh_reports(self) -> None:
        self._reports = self.logic.load_reports()
        
        scope = self.ui.var_report_scope.get().strip()
        split_filter = self.ui.var_report_split.get().strip()
        sort_key = self.ui.var_report_sort.get().strip()

        selected_model = self._selected_model_path()
        ds_dir = self.state.dataset_dir

        filtered: List[Dict[str, Any]] = []
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
                if not ds_dir or str(r.get("dataset_path") or "") != str(ds_dir): continue
            filtered.append(r)
        
        def ts_of(rr: Dict[str, Any]) -> float:
            s = rr.get("timestamp") or ""
            try: return datetime.fromisoformat(str(s)).timestamp()
            except Exception: return os.path.getmtime(str(rr.get("_path") or ""))
        def metric(rr: Dict[str, Any], k: str) -> float:
            try: return float((rr.get("metrics") or {}).get(k) or 0.0)
            except Exception: return 0.0

        if sort_key == "accuracy": filtered.sort(key=lambda rr: metric(rr, "accuracy"), reverse=True)
        elif sort_key == "macro_f1": filtered.sort(key=lambda rr: metric(rr, "macro_f1"), reverse=True)
        else: filtered.sort(key=ts_of, reverse=True)
        
        reports_data = []
        self._report_by_iid.clear()
        for i, r in enumerate(filtered):
            iid = f"r{i}"
            when = (r.get("timestamp") or "-").replace("T", " ")[:19]
            split = str(r.get("split") or "-")
            seen = str(int(r.get("seen_samples") or r.get("test_size") or "-"))
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
        if not (sel := self.ui.tree_reports.selection()): return messagebox.showinfo("Info", "Select a report first.")
        if not (r := self._report_by_iid.get(sel[0])): return
        if (p := r.get("_path")) and Path(str(p)).exists(): self.ui.append_log(self.logic.short_report(Path(str(p))))
        else:
            try:
                m = r.get("metrics", {})
                acc, f1 = float(m.get("accuracy", 0.0)), float(m.get("macro_f1", 0.0))
                self.ui.append_log(f"\\n[report] ts={r.get('timestamp')} split={r.get('split')} acc={acc:.4f} f1={f1:.4f} path={p}\\n")
            except Exception: pass
