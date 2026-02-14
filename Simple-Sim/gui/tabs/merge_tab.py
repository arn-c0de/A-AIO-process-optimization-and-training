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

from .base_tab import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore
from simple_sim.model_bundle import (
    bundle_checkpoint_path,
    upsert_bundle_meta,
    is_bundle_dir,
    read_bundle_meta,
)


class MergeTab(BaseTab):
    """Tab: Merge multiple profile-specific weights into a multi-profile bundle."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

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
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Merge", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Refresh", command=self._scan_models).pack(side="right")

        # Main split
        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(0, 8))

        # Left: profile tree with available weights
        left = ttk.Frame(main, padding=5)
        main.add(left, weight=3)

        ttk.Label(left, text="Available weights by profile type").pack(anchor="w")

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True, pady=(6, 0))

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Acc", "F1", "Size", "Modified", "Hash"),
            show="tree headings",
            selectmode="browse",
        )
        self.tree.heading("#0", text="Profile / Model")
        self.tree.column("#0", width=260, anchor="w")
        self.tree.heading("Acc", text="Accuracy")
        self.tree.column("Acc", width=75, anchor="e")
        self.tree.heading("F1", text="F1")
        self.tree.column("F1", width=70, anchor="e")
        self.tree.heading("Size", text="Size")
        self.tree.column("Size", width=75, anchor="e")
        self.tree.heading("Modified", text="Modified")
        self.tree.column("Modified", width=130, anchor="w")
        self.tree.heading("Hash", text="Profile Hash")
        self.tree.column("Hash", width=110, anchor="w")
        self.tree.tag_configure("profile_header", font=("TkDefaultFont", 10, "bold"))
        self.tree.tag_configure("group_header", font=("TkDefaultFont", 9, "italic"))
        self.tree.tag_configure("selected_weight", foreground="green")

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.tree.bind("<Double-1>", self._on_tree_double_click)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_frame, text="Select weight for profile", command=self._select_weight).pack(side="left")
        ttk.Button(btn_frame, text="Deselect", command=self._deselect_weight).pack(side="left", padx=(8, 0))

        # Right: selection summary + merge controls
        right = ttk.Frame(main, padding=5)
        main.add(right, weight=2)

        ttk.Label(right, text="Selected weights for merge").pack(anchor="w")

        self.selection_tree = ttk.Treeview(
            right,
            columns=("Model",),
            show="headings",
            selectmode="browse",
            height=8,
        )
        self.selection_tree.heading("Model", text="Profile -> Model")
        self.selection_tree.column("Model", width=350, anchor="w")
        self.selection_tree.pack(fill="x", pady=(6, 0))

        # Merge controls
        merge_frame = ttk.LabelFrame(right, text="Merge", padding=8)
        merge_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(merge_frame, text="Bundle Name:").grid(row=0, column=0, sticky="w")
        self.var_bundle_name = tk.StringVar(value="merged_v1")
        ttk.Entry(merge_frame, textvariable=self.var_bundle_name, width=30).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        merge_frame.columnconfigure(1, weight=1)

        self.var_add_timestamp = tk.BooleanVar(value=True)
        ttk.Checkbutton(merge_frame, text="Append timestamp", variable=self.var_add_timestamp).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self.btn_merge = ttk.Button(merge_frame, text="Merge -> Bundle", command=self._do_merge)
        self.btn_merge.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.var_status = tk.StringVar(value="Status: idle")
        ttk.Label(merge_frame, textvariable=self.var_status).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        # Existing bundles
        bundles_frame = ttk.LabelFrame(right, text="Existing Bundles", padding=8)
        bundles_frame.pack(fill="x", pady=(10, 0))

        self.bundle_tree = ttk.Treeview(
            bundles_frame,
            columns=("Profiles", "Created"),
            show="headings",
            selectmode="browse",
            height=5,
        )
        self.bundle_tree.heading("Profiles", text="Included Profiles")
        self.bundle_tree.column("Profiles", width=250, anchor="w")
        self.bundle_tree.heading("Created", text="Created")
        self.bundle_tree.column("Created", width=140, anchor="w")
        self.bundle_tree.pack(fill="x")

        bundle_btns = ttk.Frame(bundles_frame)
        bundle_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(bundle_btns, text="Bundle Info", command=self._show_bundle_info).pack(side="left")
        ttk.Button(bundle_btns, text="Load Into Selection", command=self._load_bundle_into_selection).pack(side="left", padx=(8, 0))
        ttk.Button(bundle_btns, text="Delete Bundle", command=self._delete_bundle).pack(side="left", padx=(8, 0))

        # Logs
        ttk.Label(right, text="Log").pack(anchor="w", pady=(10, 0))
        self.txt = tk.Text(right, height=8, wrap="none")
        self.txt.pack(fill="both", expand=True, pady=(4, 0))
        self.txt.configure(state="disabled")

        # Initial scan
        self._scan_models()
        self._refresh_bundles()

    def _append_log(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _model_root(self) -> Path:
        return self.sim_root / "outputs" / "models"

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            return str(p)

    def _load_group_data(self) -> None:
        """Load favorites and group assignments (same data as weights tab)."""
        # Favorites from favorites.json
        self._favorites.clear()
        fav_path = self._model_root() / "favorites.json"
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

        # Group assignments and custom groups from settings store
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
        names: List[str] = ["Favorites", "Snapshots", "Imports"]
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
        rel = self._rel(p)
        if rel in self._group_assignments:
            candidate = self._group_assignments[rel]
            if candidate in self._available_groups():
                return candidate
        if self._rel(p) in self._favorites:
            return "Favorites"
        rp = p.resolve()
        root = self._model_root().resolve()
        versions = (root / "versions").resolve()
        imports = (root / "imports").resolve()
        rp_str = str(rp)
        if rp_str.startswith(str(versions) + os.sep):
            return "Snapshots"
        if rp_str.startswith(str(imports) + os.sep):
            return "Imports"
        return "Uncategorized"

    def _scan_models(self) -> None:
        """Scan all model checkpoints and group by profile_id."""
        self._load_group_data()
        root = self._model_root()
        root.mkdir(parents=True, exist_ok=True)

        cand: List[Path] = []
        cand.extend(sorted(root.glob("*.pt")))
        cand.extend(sorted((root / "versions").glob("**/*.pt")))
        cand.extend(sorted((root / "imports").glob("*.pt")))
        # Include checkpoints inside bundles so they can be selected/re-merged.
        # Bundles are directories like <name>.bundle or outputs/models/bundles/<name>/.
        bundle_dirs: List[Path] = []
        for p in sorted(root.glob("*.bundle")):
            if is_bundle_dir(p):
                bundle_dirs.append(p)
        bundles_subdir = root / "bundles"
        if bundles_subdir.exists():
            for p in sorted(bundles_subdir.iterdir()):
                if is_bundle_dir(p):
                    bundle_dirs.append(p)
        for bd in bundle_dirs:
            cand.extend(sorted(bd.glob("*.pt")))

        # Deduplicate
        uniq: Dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file():
                    uniq[str(p.resolve())] = p
            except Exception:
                continue

        self._profile_models.clear()
        self._iid_to_model.clear()
        self._model_meta_cache.clear()

        for p in uniq.values():
            profile_id, profile_hash, _, _ = self._load_model_meta(p)
            if profile_id is None or profile_id == "multi":
                continue
            if profile_id not in self._profile_models:
                self._profile_models[profile_id] = []
            try:
                disp = p.name
            except Exception:
                disp = str(p)
            self._profile_models[profile_id].append((disp, p))

        # Sort models within each profile by mtime (newest first)
        for pid in self._profile_models:
            self._profile_models[pid].sort(
                key=lambda t: t[1].stat().st_mtime if t[1].exists() else 0.0,
                reverse=True,
            )

        self._rebuild_tree()
        self._refresh_bundles()
        self._append_log(f"[scan] Found {sum(len(v) for v in self._profile_models.values())} "
                         f"models across {len(self._profile_models)} profile(s)\n")

    def _load_model_meta(self, model_path: Path) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
        """Load profile_id, profile_hash, val_accuracy and val_f1 from a checkpoint.

        Returns:
            (profile_id, profile_hash, val_accuracy, val_f1)
        """
        key = str(model_path.resolve())
        cached = self._model_meta_cache.get(key)
        if cached is not None:
            return cached

        result: Tuple[Optional[str], Optional[str], Optional[float], Optional[float]] = (None, None, None, None)
        try:
            import torch
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            comp = checkpoint.get("component_profile")
            pid = None
            phash = None
            if isinstance(comp, dict):
                pid = comp.get("profile_id")
                phash = comp.get("profile_hash")

            val_acc = None
            val_f1 = None
            try:
                v = checkpoint.get("val_accuracy")
                if v is not None:
                    val_acc = float(v)
            except Exception:
                pass
            try:
                v = checkpoint.get("val_f1")
                if v is not None:
                    val_f1 = float(v)
            except Exception:
                pass

            result = (pid, phash, val_acc, val_f1)
        except Exception:
            pass

        self._model_meta_cache[key] = result
        return result

    def _rebuild_tree(self) -> None:
        """Rebuild the profile/model tree with group sub-nodes (same categories as weights tab)."""
        # Preserve expanded state
        prev_open: set = set()
        for child in self.tree.get_children():
            if self.tree.item(child, "open"):
                prev_open.add(child)
            for sub in self.tree.get_children(child):
                if self.tree.item(sub, "open"):
                    prev_open.add(sub)

        self.tree.delete(*self.tree.get_children())
        self._iid_to_model.clear()

        group_names = self._available_groups()
        counter = 0
        for profile_id in sorted(self._profile_models.keys()):
            models = self._profile_models[profile_id]
            selected = self._selected_weights.get(profile_id)

            # Profile header
            count = len(models)
            sel_marker = " [SELECTED]" if selected else ""
            profile_iid = f"profile::{profile_id}"
            was_open = profile_iid in prev_open if prev_open else True
            self.tree.insert(
                "",
                "end",
                iid=profile_iid,
                text=f"{profile_id} ({count} weights){sel_marker}",
                values=("", "", "", "", ""),
                tags=("profile_header",),
                open=was_open,
            )

            # Group models within this profile
            grouped: Dict[str, List[Tuple[str, Path]]] = {g: [] for g in group_names}
            for disp, p in models:
                g = self._group_for_path(p)
                if g not in grouped:
                    grouped[g] = []
                grouped[g].append((disp, p))

            for group in group_names:
                members = grouped.get(group, [])
                if not members:
                    continue

                group_iid = f"group::{profile_id}::{group}"
                was_group_open = group_iid in prev_open if prev_open else True
                self.tree.insert(
                    profile_iid,
                    "end",
                    iid=group_iid,
                    text=f"{group} ({len(members)})",
                    values=("", "", "", "", ""),
                    tags=("group_header",),
                    open=was_group_open,
                )

                for disp, p in members:
                    try:
                        st = p.stat()
                        size_s = self._fmt_bytes(int(st.st_size))
                        mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        size_s = "?"
                        mt = "?"

                    _, phash, val_acc, val_f1 = self._load_model_meta(p)
                    acc_s = f"{val_acc:.4f}" if val_acc is not None else "-"
                    f1_s = f"{val_f1:.4f}" if val_f1 is not None else "-"
                    hash_short = ""
                    if phash:
                        hash_short = phash.split(":")[1][:12] if ":" in phash else phash[:12]

                    iid = f"model::{counter}"
                    counter += 1

                    is_selected = selected is not None and selected.resolve() == p.resolve()
                    tags = ("selected_weight",) if is_selected else ()
                    marker = " *" if is_selected else ""

                    fav_s = "★" if self._rel(p) in self._favorites else ""
                    display_name = f"{fav_s} {disp}{marker}".strip()

                    self.tree.insert(
                        group_iid,
                        "end",
                        iid=iid,
                        text=display_name,
                        values=(acc_s, f1_s, size_s, mt, hash_short),
                        tags=tags,
                    )
                    self._iid_to_model[iid] = (profile_id, p)

        self._refresh_selection_summary()

    def _refresh_selection_summary(self) -> None:
        """Update the selection summary tree."""
        self.selection_tree.delete(*self.selection_tree.get_children())
        for profile_id in sorted(self._selected_weights.keys()):
            p = self._selected_weights[profile_id]
            self.selection_tree.insert(
                "",
                "end",
                values=(f"{profile_id}  ->  {p.name}",),
            )

    def _on_tree_double_click(self, event: tk.Event) -> None:
        """Double-click to select a weight."""
        self._select_weight()

    def _select_weight(self) -> None:
        """Mark the selected model as the chosen weight for its profile."""
        sel = self.tree.selection()
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
        self._rebuild_tree()
        self._append_log(f"[select] {profile_id} -> {model_path.name}\n")

    def _deselect_weight(self) -> None:
        """Remove the selection for the currently highlighted profile."""
        sel = self.tree.selection()
        if not sel:
            return

        iid = sel[0]
        # Check if it's a profile header
        if iid.startswith("profile::"):
            profile_id = iid.split("::", 1)[1]
        else:
            entry = self._iid_to_model.get(iid)
            if not entry:
                return
            profile_id = entry[0]

        if profile_id in self._selected_weights:
            del self._selected_weights[profile_id]
            self._rebuild_tree()
            self._append_log(f"[deselect] {profile_id}\n")

    def _do_merge(self) -> None:
        """Create a merged bundle from selected weights."""
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

        name = self.var_bundle_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Bundle name must not be empty.")
            return

        if self.var_add_timestamp.get():
            tag = time.strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{tag}"

        bundle_dir = self._model_root() / f"{name}.bundle"
        if bundle_dir.exists():
            messagebox.showerror("Error", f"Bundle already exists:\n{bundle_dir}")
            return

        self.btn_merge.configure(state="disabled")
        self.var_status.set("Status: merging...")

        def worker() -> None:
            try:
                bundle_dir.mkdir(parents=True, exist_ok=True)
                profiles_merged = []
                model_details: Dict[str, Any] = {}

                for profile_id, src_path in sorted(self._selected_weights.items()):
                    dst = bundle_checkpoint_path(bundle_dir, profile_id, kind="best")
                    self._log_threadsafe(f"[merge] {profile_id}: {src_path.name} -> {dst.name}\n")
                    shutil.copy2(src_path, dst)
                    upsert_bundle_meta(bundle_dir, profile_id, dst)
                    profiles_merged.append(profile_id)

                    # Collect detailed model info
                    _, phash, val_acc, val_f1 = self._load_model_meta(src_path)
                    try:
                        src_size = src_path.stat().st_size
                        src_mtime = datetime.fromtimestamp(src_path.stat().st_mtime).isoformat(timespec="seconds")
                    except Exception:
                        src_size = 0
                        src_mtime = ""
                    model_details[profile_id] = {
                        "source_path": str(src_path),
                        "source_name": src_path.name,
                        "profile_hash": phash,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                        "source_size_bytes": src_size,
                        "source_modified": src_mtime,
                    }

                # Write extended bundle info with model details
                self._write_bundle_details(bundle_dir, model_details)

                self._log_threadsafe(
                    f"\n[merge] Bundle created: {bundle_dir}\n"
                    f"[merge] Contains {len(profiles_merged)} profiles: {', '.join(profiles_merged)}\n"
                )

                def on_done() -> None:
                    self.btn_merge.configure(state="normal")
                    self.var_status.set(f"Status: bundle created ({name})")
                    self._refresh_bundles()

                self.frame.after(0, on_done)

            except Exception as e:
                self._log_threadsafe(f"\n[error] Merge failed: {e}\n")
                # Clean up on failure
                try:
                    if bundle_dir.exists():
                        shutil.rmtree(bundle_dir)
                except Exception:
                    pass

                def on_error() -> None:
                    self.btn_merge.configure(state="normal")
                    self.var_status.set("Status: merge failed")

                try:
                    self.frame.after(0, on_error)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _write_bundle_details(self, bundle_dir: Path, model_details: Dict[str, Any]) -> None:
        """Write extended model details into bundle_details.json."""
        details_path = bundle_dir / "bundle_details.json"
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        obj = {
            "created_at": now,
            "models": model_details,
        }
        details_path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def _read_bundle_details(self, bundle_dir: Path) -> Optional[Dict[str, Any]]:
        """Read bundle_details.json if present."""
        details_path = bundle_dir / "bundle_details.json"
        if not details_path.exists():
            return None
        try:
            return json.loads(details_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _log_threadsafe(self, msg: str) -> None:
        """Append log from any thread."""
        try:
            self.frame.after(0, lambda: self._append_log(msg))
        except Exception:
            pass

    def _refresh_bundles(self) -> None:
        """Refresh the existing bundles list."""
        self.bundle_tree.delete(*self.bundle_tree.get_children())
        root = self._model_root()

        bundle_dirs: List[Path] = []
        for p in sorted(root.glob("*.bundle")):
            if is_bundle_dir(p):
                bundle_dirs.append(p)
        bundles_subdir = root / "bundles"
        if bundles_subdir.exists():
            for p in sorted(bundles_subdir.iterdir()):
                if is_bundle_dir(p):
                    bundle_dirs.append(p)

        for bd in bundle_dirs:
            meta = read_bundle_meta(bd)
            if meta:
                profiles = ", ".join(sorted(meta.checkpoints.keys()))
                created = meta.created_at[:19] if meta.created_at else "?"
            else:
                # Scan .pt files inside
                pts = list(bd.glob("*.pt"))
                profiles = ", ".join(p.stem for p in pts) if pts else "(empty)"
                try:
                    created = datetime.fromtimestamp(bd.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    created = "?"

            self.bundle_tree.insert(
                "",
                "end",
                iid=str(bd),
                values=(profiles, created),
            )

    def _show_bundle_info(self) -> None:
        """Show detailed info for the selected bundle."""
        sel = self.bundle_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a bundle first.")
            return

        bd = Path(sel[0])
        if not bd.exists():
            messagebox.showerror("Error", f"Bundle not found:\n{bd}")
            return

        meta = read_bundle_meta(bd)

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
                        size = f" ({self._fmt_bytes(int(cp.stat().st_size))})"
                    except Exception:
                        pass
                info += f"  {pid} -> {fname} [{exists}]{size}\n"

            # Show detailed model info if available
            details = self._read_bundle_details(bd)
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
                        info += f"    Size:     {self._fmt_bytes(int(src_size))}\n"
        else:
            info += "(No bundle.json metadata)\n\n"
            info += "Files:\n"
            for f in sorted(bd.iterdir()):
                info += f"  {f.name}\n"

        text.insert("1.0", info)
        text.config(state="disabled")
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

    def _load_bundle_into_selection(self) -> None:
        """Load per-profile checkpoints from the selected bundle into the merge selection.

        This allows re-bundling an existing bundle (or using it as a starting point).
        """
        sel = self.bundle_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a bundle first.")
            return

        bd = Path(sel[0])
        if not bd.exists():
            messagebox.showerror("Error", f"Bundle not found:\n{bd}")
            return

        meta = read_bundle_meta(bd)
        loaded = 0
        if meta and meta.checkpoints:
            for pid, fname in sorted(meta.checkpoints.items()):
                p = bd / fname
                if p.exists():
                    self._selected_weights[str(pid)] = p
                    loaded += 1
        else:
            # Fallback: infer profile_id from checkpoint metadata.
            for p in sorted(bd.glob("*.pt")):
                pid, _phash, _acc, _f1 = self._load_model_meta(p)
                if pid and pid != "multi":
                    self._selected_weights[str(pid)] = p
                    loaded += 1

        if loaded <= 0:
            messagebox.showwarning("No checkpoints", f"No usable checkpoints found in:\n{bd}")
            return

        self._rebuild_tree()
        self._append_log(f"[bundle] loaded {loaded} checkpoint(s) into selection from {bd.name}\n")

    def _delete_bundle(self) -> None:
        """Delete the selected bundle."""
        sel = self.bundle_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a bundle first.")
            return

        bd = Path(sel[0])
        if not bd.exists():
            return

        if not messagebox.askyesno("Delete Bundle", f"Delete this bundle?\n\n{bd}"):
            return

        try:
            shutil.rmtree(bd)
            self._append_log(f"[delete] Bundle deleted: {bd}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Delete failed:\n{e}")
            return

        self._refresh_bundles()

    def _fmt_bytes(self, n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        v = float(max(0, n))
        u = 0
        while v >= 1024.0 and u < len(units) - 1:
            v /= 1024.0
            u += 1
        if u == 0:
            return f"{int(v)} {units[u]}"
        return f"{v:.1f} {units[u]}"

    def on_dataset_changed(self) -> None:
        """React to dataset change - rescan models."""
        if self.initialized:
            self._scan_models()
