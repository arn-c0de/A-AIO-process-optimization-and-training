"""Merge Tab - Merge profile-specific weights into a multi-profile bundle model."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, simpledialog
from typing import Any, Dict, List, Optional, Tuple

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore
from simple_sim.model_bundle import read_bundle_meta
from .ui import MergeUI
from .logic import MergeLogic


class MergeTab(BaseTab):
    """Tab: Merge multiple profile-specific weights into a multi-profile bundle."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        self.ui = MergeUI(self, self.frame)
        self.logic = MergeLogic(sim_root)

        # profile_id -> list of (display_name, Path)
        self._profile_models: Dict[str, List[Tuple[str, Path]]] = {}
        # profile_id -> selected Path (one per profile)
        self._selected_weights: Dict[str, Path] = {}
        # Treeview item tracking
        self._iid_to_model: Dict[str, Tuple[str, Path]] = {}  # iid -> (profile_id, path)
        # Cache: resolved_path -> (profile_id, profile_hash, val_accuracy, val_f1)
        self._model_meta_cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]] = {}
        # Group data (shared with weights tab via settings store + favorites.json)
        self._favorites: Dict[str, Dict[str, Any]] = {}
        self._group_assignments: Dict[str, str] = {}
        self._custom_groups: List[str] = []

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self._load_group_data()
        self._scan_models()
        self._refresh_bundles()

    def _append_log(self, s: str) -> None:
        self.ui.append_log(s)

    def _load_group_data(self) -> None:
        self._favorites.clear()
        fav_path = self.logic._model_root() / "favorites.json"
        if fav_path.exists():
            try:
                obj = json.loads(fav_path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    for it in (obj.get("favorites") or []):
                        if isinstance(it, dict):
                            rp = str(it.get("path") or "").strip()
                            if rp:
                                self._favorites[rp] = it
            except Exception:
                pass

        self._group_assignments.clear()
        self._custom_groups.clear()
        try:
            st = SettingsStore(self.sim_root)
            custom_raw = st.get("weights.custom_groups")
            if isinstance(custom_raw, str):
                parsed = json.loads(custom_raw)
                if isinstance(parsed, list):
                    self._custom_groups = [str(k) for k in parsed if isinstance(k, str) and k]
            assign_raw = st.get("weights.group_assignments")
            if isinstance(assign_raw, str):
                parsed = json.loads(assign_raw)
                if isinstance(parsed, dict):
                    self._group_assignments = {str(k): str(v) for k, v in parsed.items() if k and v}
        except Exception:
            pass

    def _available_groups(self) -> List[str]:
        names: List[str] = ["Favorites", "Snapshots", "Imports", "Bundles"]
        for custom in self._custom_groups:
            if custom and custom not in names:
                names.append(custom)
        for val in sorted(set(self._group_assignments.values())):
            if val and val not in names:
                names.append(val)
        if "Uncategorized" not in names:
            names.append("Uncategorized")
        return names

    def _group_for_path(self, p: Path) -> str:
        rel = self.logic._rel(p)
        if rel in self._group_assignments:
            candidate = self._group_assignments[rel]
            if candidate in self._available_groups():
                return candidate
        if rel in self._favorites:
            return "Favorites"
        if self.logic._bundle_root_for_path(p) is not None:
            return "Bundles"
        rp = p.resolve()
        root = self.logic._model_root().resolve()
        versions = (root / "versions").resolve()
        imports = (root / "imports").resolve()
        rp_str = str(rp)
        if rp_str.startswith(str(versions) + os.sep):
            return "Snapshots"
        if rp_str.startswith(str(imports) + os.sep):
            return "Imports"
        return "Uncategorized"

    def _scan_models(self) -> None:
        self._load_group_data()
        self._profile_models = self.logic.scan_models()

        self.ui.rebuild_tree(self._profile_models, self._iid_to_model, self._model_meta_cache,
                             self._selected_weights, self._favorites, self._group_for_path, self._available_groups())
        self._refresh_selection_summary()
        self._refresh_bundles()
        self._append_log(f"[scan] Found {sum(len(v) for v in self._profile_models.values())} "
                         f"models across {len(self._profile_models)} profile(s)\n")

    def _refresh_selection_summary(self) -> None:
        self.ui.refresh_selection_summary(self._selected_weights)

    def _on_tree_double_click(self, event: tk.Event) -> None:
        self._select_weight()

    def _select_weight(self) -> None:
        sel = self.ui.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a model from the list.")
            return

        iid = sel[0]
        entry = self._iid_to_model.get(iid)
        if not entry:
            messagebox.showinfo("Info", "Select a model, not a profile header.")
            return

        profile_id, model_path = entry
        self._selected_weights[profile_id] = model_path
        self.ui.rebuild_tree(self._profile_models, self._iid_to_model, self._model_meta_cache,
                             self._selected_weights, self._favorites, self._group_for_path, self._available_groups())
        self._append_log(f"[select] {profile_id} -> {model_path.name}\n")

    def _deselect_weight(self) -> None:
        sel = self.ui.tree.selection()
        if not sel:
            return

        iid = sel[0]
        if iid.startswith("profile::"):
            profile_id = iid.split("::", 1)[1]
        else:
            entry = self._iid_to_model.get(iid)
            if not entry:
                return
            profile_id = entry[0]

        if profile_id in self._selected_weights:
            del self._selected_weights[profile_id]
            self.ui.rebuild_tree(self._profile_models, self._iid_to_model, self._model_meta_cache,
                                 self._selected_weights, self._favorites, self._group_for_path, self._available_groups())
            self._append_log(f"[deselect] {profile_id}\n")

    def _do_merge(self) -> None:
        if not self._selected_weights:
            messagebox.showinfo("Info", "Select at least one weight per profile first.")
            return

        if len(self._selected_weights) < 2:
            if not messagebox.askyesno(
                "Only 1 profile",
                f"Only {len(self._selected_weights)} profile selected.\n"
                "A merged bundle is more useful with multiple profiles.\n\n"
                "Create anyway?",
            ):
                return

        name = self.ui.var_bundle_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Bundle name must not be empty.")
            return

        if self.ui.var_add_timestamp.get():
            tag = time.strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{tag}"

        bundle_dir = self.logic._model_root() / f"{name}.bundle"
        if bundle_dir.exists():
            messagebox.showerror("Error", f"Bundle already exists:\n{bundle_dir}")
            return

        self.ui.btn_merge.configure(state="disabled")
        self.ui.var_status.set("Status: merging...")

        def worker() -> None:
            try:
                profiles_merged, model_details, sources = self.logic.do_merge(bundle_dir, self._selected_weights, self._model_meta_cache)
                self.logic.write_bundle_details(bundle_dir, model_details, sources=sources)

                self._log_threadsafe(
                    f"\n[merge] Bundle created: {bundle_dir}\n"
                    f"[merge] Contains {len(profiles_merged)} profiles: {', '.join(profiles_merged)}\n"
                )

                def on_done() -> None:
                    self.ui.btn_merge.configure(state="normal")
                    self.ui.var_status.set(f"Status: bundle created ({name})")
                    self._refresh_bundles()

                self.frame.after(0, on_done)

            except Exception as e:
                self._log_threadsafe(f"\n[error] Merge failed: {e}\n")
                try:
                    if bundle_dir.exists():
                        shutil.rmtree(bundle_dir)
                except Exception:
                    pass

                def on_error() -> None:
                    self.ui.btn_merge.configure(state="normal")
                    self.ui.var_status.set("Status: merge failed")

                try:
                    self.frame.after(0, on_error)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _do_merge_ensemble(self) -> None:
        if not self._selected_weights:
            messagebox.showinfo("Info", "Select at least one weight per profile first.")
            return

        name = self.ui.var_bundle_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Name must not be empty.")
            return

        if self.ui.var_add_timestamp.get():
            tag = time.strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{tag}"

        out_path = self.logic._model_root() / f"{name}_ensemble.pt"
        if out_path.exists():
            messagebox.showerror("Error", f"Target already exists:\n{out_path}")
            return

        self.ui.btn_merge.configure(state="disabled")
        self.ui.btn_merge_ensemble.configure(state="disabled")
        self.ui.var_status.set("Status: building ensemble...")
        log_q = queue.Queue() # temp queue for ensemble logging

        def worker() -> None:
            try:
                self.logic.do_merge_ensemble(out_path, self._selected_weights, log_q)

                def on_done() -> None:
                    self.ui.btn_merge.configure(state="normal")
                    self.ui.btn_merge_ensemble.configure(state="normal")
                    self.ui.var_status.set(f"Status: ensemble created ({out_path.name})")
                    self._scan_models()
                    self._refresh_bundles()

                self.frame.after(0, on_done)

            except Exception as e:
                self._log_threadsafe(f"\n[error] Ensemble build failed: {e}\n")
                try:
                    if out_path.exists():
                        out_path.unlink()
                except Exception:
                    pass

                def on_error() -> None:
                    self.ui.btn_merge.configure(state="normal")
                    self.ui.btn_merge_ensemble.configure(state="normal")
                    self.ui.var_status.set("Status: ensemble failed")

                self.frame.after(0, on_error)
            finally:
                while True: # drain logs to main UI log
                    try: self._log_threadsafe(log_q.get_nowait())
                    except queue.Empty: break

        threading.Thread(target=worker, daemon=True).start()

    def _log_threadsafe(self, msg: str) -> None:
        self.frame.after(0, lambda: self._append_log(msg))

    def _refresh_bundles(self) -> None:
        bundle_dirs = self.logic.get_existing_bundles()
        self.ui.refresh_bundles(bundle_dirs, read_bundle_meta, self.ui.fmt_bytes)

    def _show_bundle_info(self) -> None:
        sel = self.ui.bundle_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a bundle first.")
            return

        bd = Path(sel[0])
        if not bd.exists():
            messagebox.showerror("Error", f"Bundle not found:\n{bd}")
            return

        meta = read_bundle_meta(bd)
        details = self.logic._read_bundle_details(bd)

        dialog = tk.Toplevel(self.frame)
        dialog.title(f"Bundle: {bd.name}")
        dialog.geometry("550x400")

        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
        text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text.yview)

        info = f"Bundle: {bd.name}\n"
        info += f"Path: {bd}\n\n"

        if meta:
            info += f"Version: {meta.bundle_version}\n"
            info += f"Created: {meta.created_at}\n"
            info += f"Updated: {meta.updated_at}\n\n"
            info += "Checkpoints:\n"
            for pid, fname in sorted(meta.checkpoints.items()):
                cp = bd / fname
                exists = "exists" if cp.exists() else "MISSING"
                size = ""
                if cp.exists():
                    try:
                        size = f" ({self.ui.fmt_bytes(int(cp.stat().st_size))})"
                    except Exception:
                        pass
                info += f"  {pid} -> {fname} [{exists}]{size}\n"

            if details and isinstance(details.get("models"), dict):
                info += "\nModel Details:\n"
                info += "-" * 50 + "\n"
                for pid, minfo in sorted(details["models"].items()):
                    info += f"\n  Profile: {pid}\n"
                    info += f"    Source:   {minfo.get('source_name', '?')}\n"
                    info += f"    Path:     {minfo.get('source_path', '?')}\n"
                    acc = minfo.get('val_accuracy')
                    f1 = minfo.get('val_f1')
                    info += f"    Accuracy: {acc:.4f}\n" if acc is not None else "    Accuracy: -\n"
                    info += f"    F1:       {f1:.4f}\n" if f1 is not None else "    F1:       -\n"
                    phash = minfo.get('profile_hash') or ""
                    if phash:
                        info += f"    Hash:     {phash[:40]}...\n"
                    info += f"    Modified: {minfo.get('source_modified', '?')}\n"
                    src_size = minfo.get('source_size_bytes', 0)
                    if src_size:
                        info += f"    Size:     {self.ui.fmt_bytes(int(src_size))}\n"
        else:
            info += "(No bundle.json metadata)\n\n"
            info += "Files:\n"
            for f in sorted(bd.iterdir()):
                info += f"  {f.name}\n"

        text.insert("1.0", info)
        text.config(state="disabled")
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

    def _load_bundle_into_selection(self) -> None:
        sel = self.ui.bundle_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a bundle first.")
            return

        bd = Path(sel[0])
        if not bd.exists():
            messagebox.showerror("Error", f"Bundle not found:\n{bd}")
            return

        meta = read_bundle_meta(bd)
        loaded = 0
        self._selected_weights.clear() # Clear existing selection when loading bundle

        if meta and meta.checkpoints:
            for pid, fname in sorted(meta.checkpoints.items()):
                p = bd / fname
                if p.exists():
                    self._selected_weights[str(pid)] = p
                    loaded += 1
        else:
            for p in sorted(bd.glob("*.pt")):
                pid, _phash, _acc, _f1 = self.logic.load_model_meta(p, self._model_meta_cache)
                if pid and pid != "multi":
                    self._selected_weights[str(pid)] = p
                    loaded += 1

        if loaded <= 0:
            messagebox.showwarning("No checkpoints", f"No usable checkpoints found in:\n{bd}")
            return

        self.ui.rebuild_tree(self._profile_models, self._iid_to_model, self._model_meta_cache,
                             self._selected_weights, self._favorites, self._group_for_path, self._available_groups())
        self._append_log(f"[bundle] loaded {loaded} checkpoint(s) into selection from {bd.name}\n")

    def _delete_bundle(self) -> None:
        sel = self.ui.bundle_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a bundle first.")
            return

        bd = Path(sel[0])
        if not bd.exists():
            return

        if not messagebox.askyesno("Delete Bundle", f"Delete this bundle?\\n\\n{bd}"):
            return

        try:
            self.logic.delete_bundle(bd)
            self._append_log(f"[delete] Bundle deleted: {bd}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Delete failed:\\n{e}")
            return

        self._refresh_bundles()

    def on_dataset_changed(self) -> None:
        if self.initialized:
            self._scan_models()
