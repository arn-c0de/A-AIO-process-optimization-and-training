"""Weights Tab - Model checkpoint versioning/backup/export + comparisons."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
import re
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, List, Optional, Tuple

from .base_tab import BaseTab
from gui.state import UiState


class WeightsTab(BaseTab):
    """Tab: manage model weights (snapshots/import/export) and run evaluations."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()
        self.log_q: queue.Queue[str] = queue.Queue()

        self._models: List[Path] = []
        self._model_by_iid: Dict[str, Path] = {}
        self._favorites: Dict[str, Dict[str, Any]] = {}  # rel_path -> metadata

        self._reports: List[Dict[str, Any]] = []
        self._report_by_iid: Dict[str, Dict[str, Any]] = {}

        # UI components
        self.var_dataset: tk.StringVar
        self.var_active_model: tk.StringVar
        self.var_selected_model: tk.StringVar

        self.var_split: tk.StringVar
        self.var_device: tk.StringVar
        self.var_max_samples: tk.StringVar
        self.chk_save_preds: tk.BooleanVar

        self.var_cmp_a: tk.StringVar
        self.var_cmp_b: tk.StringVar
        self.var_report_scope: tk.StringVar
        self.var_report_split: tk.StringVar
        self.var_report_sort: tk.StringVar

        self.tree: ttk.Treeview
        self.tree_reports: ttk.Treeview
        self.txt: tk.Text
        self.var_status: tk.StringVar

        self._ui_tick_id: Optional[str] = None
        self._models_menu: Optional[tk.Menu] = None

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_dataset, width=60, state="readonly").pack(side="left", padx=(5, 10))

        ttk.Label(top, text="Active model:").pack(side="left")
        self.var_active_model = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_active_model, width=45).pack(side="left", padx=(5, 10))

        ttk.Button(top, text="Refresh", command=self._refresh_models).pack(side="left")
        ttk.Button(top, text="Snapshot active", command=self._snapshot_active).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Import...", command=self._import_model).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Export selected...", command=self._export_selected).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Activate selected", command=self._activate_selected).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Delete selected", command=self._delete_selected).pack(side="left", padx=(8, 0))

        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(0, 8))
        self.var_status = tk.StringVar(value="status: idle")
        ttk.Label(status, textvariable=self.var_status).pack(side="left")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Left: model list
        left = ttk.Frame(main, padding=5)
        main.add(left, weight=2)
        ttk.Label(left, text="Available checkpoints").pack(anchor="w")

        self.var_selected_model = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.var_selected_model).pack(anchor="w", pady=(4, 6))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("Fav", "Runs", "Size", "MTime", "Path"), show="headings")
        self.tree.heading("Fav", text="Fav")
        self.tree.heading("Runs", text="Runs")
        self.tree.heading("Size", text="Size")
        self.tree.heading("MTime", text="Modified")
        self.tree.heading("Path", text="Path")
        self.tree.column("Fav", width=50, anchor="center")
        self.tree.column("Runs", width=60, anchor="e")
        self.tree.column("Size", width=80, anchor="e")
        self.tree.column("MTime", width=140, anchor="w")
        self.tree.column("Path", width=520, anchor="w")

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        # Right-click context menu for quick actions (delete/export/activate).
        self.tree.bind("<Button-3>", self._on_models_right_click)
        self.tree.bind("<Button-2>", self._on_models_right_click)  # macOS

        self._models_menu = tk.Menu(self.frame, tearoff=0)
        self._models_menu.add_command(label="Run selected", command=self._run_selected)
        self._models_menu.add_command(label="Activate selected", command=self._activate_selected)
        self._models_menu.add_command(label="Export selected...", command=self._export_selected)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Add to Favorites", command=self._add_selected_to_favorites)
        self._models_menu.add_command(label="Remove from Favorites", command=self._remove_selected_from_favorites)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Set as Compare A", command=self._set_selected_as_compare_a)
        self._models_menu.add_command(label="Set as Compare B", command=self._set_selected_as_compare_b)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Delete selected", command=self._delete_selected)
        self._models_menu.add_separator()
        self._models_menu.add_command(label="Refresh list", command=self._refresh_models)

        # Ensure context menu closes when clicking elsewhere / pressing Esc.
        tl = self.frame.winfo_toplevel()
        tl.bind("<Button-1>", self._hide_models_menu, add="+")
        tl.bind("<Escape>", self._hide_models_menu, add="+")

        # Right: evaluate/compare + logs
        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)

        runbox = ttk.LabelFrame(right, text="Evaluate (predict.sh)", padding=8)
        runbox.pack(fill="x", pady=(0, 8))

        ttk.Label(runbox, text="Split").grid(row=0, column=0, sticky="w")
        self.var_split = tk.StringVar(value="test")
        ttk.Combobox(runbox, textvariable=self.var_split, values=["train", "val", "test", "all"], state="readonly", width=6).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )

        ttk.Label(runbox, text="Device").grid(row=0, column=2, sticky="w")
        self.var_device = tk.StringVar(value="auto")
        ttk.Combobox(runbox, textvariable=self.var_device, values=["auto", "cpu", "cuda"], state="readonly", width=6).grid(
            row=0, column=3, sticky="w", padx=(6, 12)
        )

        ttk.Label(runbox, text="Max").grid(row=0, column=4, sticky="w")
        self.var_max_samples = tk.StringVar(value="")
        ttk.Entry(runbox, textvariable=self.var_max_samples, width=7).grid(row=0, column=5, sticky="w", padx=(6, 12))

        self.chk_save_preds = tk.BooleanVar(value=True)
        ttk.Checkbutton(runbox, text="Save per-sample preds", variable=self.chk_save_preds).grid(row=0, column=6, sticky="w")

        ttk.Button(runbox, text="Run selected", command=self._run_selected).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(runbox, text="Stop", command=self._stop).grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(6, 0))

        cmpbox = ttk.LabelFrame(right, text="Compare two models", padding=8)
        cmpbox.pack(fill="x", pady=(0, 8))
        cmpbox.columnconfigure(1, weight=1)
        cmpbox.columnconfigure(3, weight=1)

        ttk.Label(cmpbox, text="A").grid(row=0, column=0, sticky="w")
        self.var_cmp_a = tk.StringVar(value="")
        self.combo_a = ttk.Combobox(cmpbox, textvariable=self.var_cmp_a, values=[], state="readonly")
        self.combo_a.grid(row=0, column=1, sticky="ew", padx=(6, 12))

        ttk.Label(cmpbox, text="B").grid(row=0, column=2, sticky="w")
        self.var_cmp_b = tk.StringVar(value="")
        self.combo_b = ttk.Combobox(cmpbox, textvariable=self.var_cmp_b, values=[], state="readonly")
        self.combo_b.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        ttk.Button(cmpbox, text="Run A then B", command=self._run_compare).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        repbox = ttk.LabelFrame(right, text="History / Reports", padding=8)
        repbox.pack(fill="both", expand=True, pady=(0, 8))

        repctl = ttk.Frame(repbox)
        repctl.pack(fill="x", pady=(0, 6))

        ttk.Button(repctl, text="Refresh reports", command=self._refresh_reports).pack(side="left")

        ttk.Label(repctl, text="Scope:").pack(side="left", padx=(12, 4))
        self.var_report_scope = tk.StringVar(value="selected_model")
        ttk.Combobox(
            repctl,
            textvariable=self.var_report_scope,
            values=["selected_model", "dataset", "all"],
            state="readonly",
            width=14,
        ).pack(side="left")

        ttk.Label(repctl, text="Split:").pack(side="left", padx=(12, 4))
        self.var_report_split = tk.StringVar(value="any")
        ttk.Combobox(
            repctl,
            textvariable=self.var_report_split,
            values=["any", "train", "val", "test", "all"],
            state="readonly",
            width=6,
        ).pack(side="left")

        ttk.Label(repctl, text="Sort:").pack(side="left", padx=(12, 4))
        self.var_report_sort = tk.StringVar(value="newest")
        ttk.Combobox(
            repctl,
            textvariable=self.var_report_sort,
            values=["newest", "accuracy", "macro_f1"],
            state="readonly",
            width=8,
        ).pack(side="left")

        ttk.Button(repctl, text="Show summary", command=self._show_selected_report_summary).pack(side="left", padx=(12, 0))

        rep_tree_frame = ttk.Frame(repbox)
        rep_tree_frame.pack(fill="both", expand=True)

        self.tree_reports = ttk.Treeview(
            rep_tree_frame,
            columns=("When", "Split", "Seen", "Acc", "F1", "Model", "Dataset", "Path"),
            show="headings",
        )
        self.tree_reports.heading("When", text="When")
        self.tree_reports.heading("Split", text="Split")
        self.tree_reports.heading("Seen", text="Seen")
        self.tree_reports.heading("Acc", text="Acc")
        self.tree_reports.heading("F1", text="MacroF1")
        self.tree_reports.heading("Model", text="Model")
        self.tree_reports.heading("Dataset", text="Dataset")
        self.tree_reports.heading("Path", text="Report path")

        self.tree_reports.column("When", width=150, anchor="w")
        self.tree_reports.column("Split", width=55, anchor="center")
        self.tree_reports.column("Seen", width=65, anchor="e")
        self.tree_reports.column("Acc", width=70, anchor="e")
        self.tree_reports.column("F1", width=80, anchor="e")
        self.tree_reports.column("Model", width=120, anchor="w")
        self.tree_reports.column("Dataset", width=120, anchor="w")
        self.tree_reports.column("Path", width=380, anchor="w")

        rep_scroll = ttk.Scrollbar(rep_tree_frame, orient="vertical", command=self.tree_reports.yview)
        self.tree_reports.configure(yscrollcommand=rep_scroll.set)
        self.tree_reports.pack(side="left", fill="both", expand=True)
        rep_scroll.pack(side="right", fill="y")

        ttk.Label(right, text="Logs / Output").pack(anchor="w")
        self.txt = tk.Text(right, height=14, wrap="none")
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

        self.on_dataset_changed()
        self._load_favorites()
        self._refresh_models()
        self._refresh_reports()
        self._tick_ui()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir:
            return
        self.var_dataset.set(str(self.state.dataset_dir))

        active = self.sim_root / "outputs" / "models" / f"{self.state.dataset_dir.name}.pt"
        self.var_active_model.set(str(active))

    def _append_log(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _tick_ui(self) -> None:
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self._ui_tick_id = self.frame.after(150, self._tick_ui)

    def _model_root(self) -> Path:
        return self.sim_root / "outputs" / "models"

    def _refresh_models(self) -> None:
        root = self._model_root()
        root.mkdir(parents=True, exist_ok=True)

        cand: List[Path] = []
        cand.extend(sorted(root.glob("*.pt")))
        cand.extend(sorted((root / "versions").glob("**/*.pt")))
        cand.extend(sorted((root / "imports").glob("*.pt")))

        # Deduplicate + sort by (favorite first, then mtime desc)
        uniq: Dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file():
                    uniq[str(p.resolve())] = p
            except Exception:
                continue
        paths = list(uniq.values())
        paths.sort(
            key=lambda p: (
                0 if self._is_favorited(p) else 1,
                -(p.stat().st_mtime if p.exists() else 0.0),
            )
        )
        self._models = paths

        self.tree.delete(*self.tree.get_children())
        self._model_by_iid.clear()

        display_values: List[str] = []
        for i, p in enumerate(self._models):
            st = p.stat()
            size_s = self._fmt_bytes(int(st.st_size))
            mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            rel = self._rel(p)
            runs_s = self._runs_for_model(p)
            fav_s = "YES" if self._is_favorited(p) else ""
            iid = f"m{i}"
            self.tree.insert("", "end", iid=iid, values=(fav_s, runs_s, size_s, mt, rel))
            self._model_by_iid[iid] = p
            display_values.append(rel)

        # Populate compare selectors
        self.combo_a.configure(values=display_values)
        self.combo_b.configure(values=display_values)

        if display_values and not self.var_cmp_a.get():
            self.var_cmp_a.set(display_values[0])
        if len(display_values) > 1 and not self.var_cmp_b.get():
            self.var_cmp_b.set(display_values[1])

        # Also refresh reports since new versions may have been created.
        try:
            self._refresh_reports()
        except Exception:
            pass

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            return str(p)

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

    def _on_tree_select(self, _evt: Optional[object] = None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        p = self._model_by_iid.get(sel[0])
        if not p:
            return
        self.var_selected_model.set(f"selected: {self._rel(p)}")
        # Keep the report table in sync with the current selection.
        try:
            if self.var_report_scope.get() == "selected_model":
                self._refresh_reports()
        except Exception:
            pass

    def _selected_model_path(self) -> Optional[Path]:
        sel = self.tree.selection()
        if not sel:
            return None
        return self._model_by_iid.get(sel[0])

    def _selected_model_paths(self) -> List[Path]:
        out: List[Path] = []
        for iid in self.tree.selection():
            p = self._model_by_iid.get(iid)
            if p:
                out.append(p)
        return out

    def _on_models_right_click(self, event: tk.Event) -> None:
        """Show context menu for model list."""
        if not self._models_menu:
            return
        try:
            iid = self.tree.identify_row(event.y)
        except Exception:
            iid = ""
        if iid:
            # Ensure the clicked row is selected before acting.
            cur = set(self.tree.selection())
            if iid not in cur:
                self.tree.selection_set(iid)
                self.tree.focus(iid)

        # Enable/disable "Delete selected" based on whether selection is deletable.
        try:
            paths = self._selected_model_paths()
            can_delete = any(self._is_deletable_checkpoint(p) and not self._is_favorited(p) for p in paths)
            self._models_menu.entryconfigure("Delete selected", state=("normal" if can_delete else "disabled"))
            can_run = bool(paths) and bool(self.state.dataset_dir)
            self._models_menu.entryconfigure("Run selected", state=("normal" if can_run else "disabled"))
            self._models_menu.entryconfigure("Activate selected", state=("normal" if bool(paths) else "disabled"))
            self._models_menu.entryconfigure("Export selected...", state=("normal" if bool(paths) else "disabled"))
            self._models_menu.entryconfigure("Set as Compare A", state=("normal" if bool(paths) else "disabled"))
            self._models_menu.entryconfigure("Set as Compare B", state=("normal" if bool(paths) else "disabled"))
            # Favorites actions depend on whether the first selected path is already favorited.
            if paths:
                is_fav = self._is_favorited(paths[0])
                self._models_menu.entryconfigure("Add to Favorites", state=("disabled" if is_fav else "normal"))
                self._models_menu.entryconfigure("Remove from Favorites", state=("normal" if is_fav else "disabled"))
            else:
                self._models_menu.entryconfigure("Add to Favorites", state="disabled")
                self._models_menu.entryconfigure("Remove from Favorites", state="disabled")
        except Exception:
            pass

        try:
            self._models_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self._models_menu.grab_release()
            except Exception:
                pass

    def _hide_models_menu(self, _event: Optional[tk.Event] = None) -> None:
        if not self._models_menu:
            return
        try:
            self._models_menu.unpost()
        except Exception:
            pass

    def _set_selected_as_compare_a(self) -> None:
        p = self._selected_model_path()
        if not p:
            return
        self.var_cmp_a.set(self._rel(p))

    def _set_selected_as_compare_b(self) -> None:
        p = self._selected_model_path()
        if not p:
            return
        self.var_cmp_b.set(self._rel(p))

    def _favorites_path(self) -> Path:
        return self.sim_root / "outputs" / "models" / "favorites.json"

    def _load_favorites(self) -> None:
        p = self._favorites_path()
        self._favorites = {}
        if not p.exists():
            return
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        favs = obj.get("favorites")
        if isinstance(favs, list):
            for it in favs:
                if not isinstance(it, dict):
                    continue
                rp = str(it.get("path") or "").strip()
                if not rp:
                    continue
                self._favorites[rp] = it

    def _save_favorites(self) -> None:
        p = self._favorites_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        favs = list(self._favorites.values())
        # stable sort by path
        favs.sort(key=lambda it: str(it.get("path") or ""))
        obj = {"favorites": favs}
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _is_favorited(self, p: Path) -> bool:
        rp = self._rel(p)
        return rp in self._favorites

    def _add_selected_to_favorites(self) -> None:
        p = self._selected_model_path()
        if not p:
            return
        rp = self._rel(p)
        if rp in self._favorites:
            return
        self._favorites[rp] = {
            "path": rp,
            "added": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_favorites()
        self._append_log(f"[favorite] added {rp}\n")
        self._refresh_models()

    def _remove_selected_from_favorites(self) -> None:
        p = self._selected_model_path()
        if not p:
            return
        rp = self._rel(p)
        if rp not in self._favorites:
            return
        if not messagebox.askyesno("Remove favorite", f"Remove from favorites?\n\n{rp}"):
            return
        self._favorites.pop(rp, None)
        self._save_favorites()
        self._append_log(f"[favorite] removed {rp}\n")
        self._refresh_models()

    def _active_model_path(self) -> Optional[Path]:
        s = self.var_active_model.get().strip()
        if not s:
            return None
        return Path(s)

    def _snapshot_active(self) -> None:
        active = self._active_model_path()
        if not active or not active.exists():
            messagebox.showerror("Error", f"Active model not found:\n{active}")
            return

        dst = self._snapshot_path_for(active)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(active, dst)
        except Exception as e:
            messagebox.showerror("Error", f"Snapshot failed:\n{e}")
            return

        self._append_log(f"[snapshot] {active} -> {dst}\n")
        self._refresh_models()

    def _snapshot_path_for(self, model_path: Path) -> Path:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = self._model_root() / "versions" / model_path.stem
        base = root / f"{model_path.stem}_{tag}.pt"
        if not base.exists():
            return base
        for i in range(1, 1000):
            cand = root / f"{model_path.stem}_{tag}_{i:03d}.pt"
            if not cand.exists():
                return cand
        return base

    def _import_model(self) -> None:
        src = filedialog.askopenfilename(
            title="Import model checkpoint (.pt)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if not src:
            return
        src_p = Path(src)
        if not src_p.exists():
            return

        dst_dir = self._model_root() / "imports"
        dst_dir.mkdir(parents=True, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = dst_dir / f"{tag}_{src_p.name}"
        try:
            shutil.copy2(src_p, dst)
        except Exception as e:
            messagebox.showerror("Error", f"Import failed:\n{e}")
            return

        self._append_log(f"[import] {src_p} -> {dst}\n")
        self._refresh_models()

    def _export_selected(self) -> None:
        p = self._selected_model_path()
        if not p:
            messagebox.showinfo("Info", "Select a model first.")
            return
        dst = filedialog.asksaveasfilename(
            title="Export selected model",
            initialfile=p.name,
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if not dst:
            return
        try:
            shutil.copy2(p, Path(dst))
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
            return
        self._append_log(f"[export] {p} -> {dst}\n")

    def _activate_selected(self) -> None:
        src = self._selected_model_path()
        if not src:
            messagebox.showinfo("Info", "Select a model first.")
            return
        dst = self._active_model_path()
        if not dst:
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not messagebox.askyesno("Overwrite active model?", f"Overwrite active model?\n\n{dst}\n\nWith:\n{src}"):
                return
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            messagebox.showerror("Error", f"Activate failed:\n{e}")
            return

        self._append_log(f"[activate] {src} -> {dst}\n")
        self.state.current_model_path = dst
        self._refresh_models()

    def _is_deletable_checkpoint(self, p: Path) -> bool:
        """Allow deleting only snapshots/imports inside this repo (avoid accidental deletion elsewhere)."""
        try:
            rp = p.resolve()
            root = (self.sim_root / "outputs" / "models").resolve()
            versions = (root / "versions").resolve()
            imports = (root / "imports").resolve()
            # Explicitly do NOT allow deleting the active models directly under outputs/models/*.pt
            # (those are the main entrypoints for training/eval).
            return str(rp).startswith(str(versions) + os.sep) or str(rp).startswith(str(imports) + os.sep)
        except Exception:
            return False

    def _delete_selected(self) -> None:
        paths = self._selected_model_paths()
        if not paths:
            messagebox.showinfo("Info", "Select one or more snapshots to delete.")
            return

        fav = [p for p in paths if self._is_favorited(p)]
        if fav:
            msg = "These selected checkpoint(s) are favorited and cannot be deleted from here.\n\n"
            msg += "\n".join(self._rel(p) for p in fav[:8])
            if len(fav) > 8:
                msg += f"\n... (+{len(fav) - 8} more)"
            msg += "\n\nRemove from favorites first, then delete."
            messagebox.showwarning("Favorited", msg)
            return

        deletable = [p for p in paths if self._is_deletable_checkpoint(p)]
        blocked = [p for p in paths if p not in deletable]

        if blocked:
            msg = "Some selected files are not deletable here (only versions/imports are allowed):\n\n"
            msg += "\n".join(str(p) for p in blocked[:8])
            if len(blocked) > 8:
                msg += f"\n... (+{len(blocked) - 8} more)"
            messagebox.showwarning("Not deletable", msg)

        if not deletable:
            return

        msg = "Delete selected checkpoint file(s)?\n\n"
        msg += "\n".join(self._rel(p) for p in deletable[:8])
        if len(deletable) > 8:
            msg += f"\n... (+{len(deletable) - 8} more)"
        msg += "\n\nThis will also delete any adjacent *.meta.json file."

        if not messagebox.askyesno("Delete checkpoints", msg):
            return

        deleted = 0
        for p in deletable:
            try:
                meta = Path(str(p) + ".meta.json")
                if meta.exists():
                    try:
                        meta.unlink()
                    except Exception:
                        pass
                p.unlink()
                deleted += 1
            except Exception as e:
                self._append_log(f"[delete] failed: {p} ({e})\n")

        self._append_log(f"[delete] deleted {deleted} file(s)\n")
        self._refresh_models()

    def _stop(self) -> None:
        self.stop_evt.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.var_status.set("status: stopping...")

    def _run_selected(self) -> None:
        p = self._selected_model_path()
        if not p:
            messagebox.showinfo("Info", "Select a model first.")
            return
        self._run_predict(model_path=p, label=f"selected({p.name})")

    def _run_compare(self) -> None:
        a = self._path_from_combo(self.var_cmp_a.get().strip())
        b = self._path_from_combo(self.var_cmp_b.get().strip())
        if not a or not b:
            messagebox.showinfo("Info", "Choose two models to compare.")
            return

        def worker() -> None:
            ra = self._run_predict_blocking(a, label=f"A({a.name})")
            rb = self._run_predict_blocking(b, label=f"B({b.name})")
            if ra and rb:
                self.log_q.put(self._compare_summary(ra, rb, model_a=a, model_b=b))

        threading.Thread(target=worker, daemon=True).start()

    def _compare_summary(self, ra: Dict[str, Any], rb: Dict[str, Any], *, model_a: Path, model_b: Path) -> str:
        """Return a human-readable compare summary including a winner line."""
        def fmetric(r: Dict[str, Any], k: str) -> Optional[float]:
            try:
                return float((r.get("metrics") or {}).get(k))
            except Exception:
                return None

        def fn_rate(r: Dict[str, Any], cls: str) -> Optional[float]:
            try:
                d = (r.get("metrics") or {}).get("critical_fn_rates") or {}
                return float(d.get(cls))
            except Exception:
                return None

        split = str(self.var_split.get().strip() or "test")
        dsname = Path(str(ra.get("dataset_path") or rb.get("dataset_path") or "-")).name
        runs_a = self._runs_for_model(model_a)
        runs_b = self._runs_for_model(model_b)

        acc_a = fmetric(ra, "accuracy")
        acc_b = fmetric(rb, "accuracy")
        f1_a = fmetric(ra, "macro_f1")
        f1_b = fmetric(rb, "macro_f1")
        cfn_a = fn_rate(ra, "MISALIGNED")
        cfn_b = fn_rate(rb, "MISALIGNED")

        # Winner: prefer macro_f1, then accuracy; ties allowed.
        winner = "TIE"
        try:
            eps = 1e-12
            if f1_a is not None and f1_b is not None and abs(f1_a - f1_b) > eps:
                winner = "A" if f1_a > f1_b else "B"
            elif acc_a is not None and acc_b is not None and abs(acc_a - acc_b) > eps:
                winner = "A" if acc_a > acc_b else "B"
        except Exception:
            pass

        lines: List[str] = []
        lines.append("\n[compare]")
        lines.append(f"  dataset: {dsname}   split: {split}")
        lines.append(f"  A: {model_a.name}  (runs={runs_a})")
        lines.append(f"  B: {model_b.name}  (runs={runs_b})")
        if acc_a is not None and acc_b is not None:
            lines.append(f"  accuracy: A={acc_a:.4f}  B={acc_b:.4f}  delta(B-A)={acc_b-acc_a:+.4f}")
        if f1_a is not None and f1_b is not None:
            lines.append(f"  macro_f1:  A={f1_a:.4f}  B={f1_b:.4f}  delta(B-A)={f1_b-f1_a:+.4f}")
        if cfn_a is not None and cfn_b is not None:
            # Lower is better for FN rate.
            lines.append(f"  MISALIGNED FN rate: A={cfn_a:.4f}  B={cfn_b:.4f}  delta(B-A)={cfn_b-cfn_a:+.4f}")
        if winner == "TIE":
            lines.append("  winner: TIE (no measurable difference with current metrics)")
        else:
            lines.append(f"  winner: {winner}")
        lines.append("")
        return "\n".join(lines)

    def _path_from_combo(self, rel: str) -> Optional[Path]:
        if not rel:
            return None
        p = (self.sim_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if p.exists():
            return p
        # Fallback: search by the rendered relative path.
        for mp in self._models:
            if self._rel(mp) == rel:
                return mp
        return None

    def _predict_cmd(self, model_path: Path) -> Tuple[List[str], Path]:
        if not self.state.dataset_dir:
            raise ValueError("No dataset selected")
        data_dir = self.state.dataset_dir
        split = self.var_split.get().strip() or "test"
        device = self.var_device.get().strip() or "auto"
        max_samples_s = self.var_max_samples.get().strip()
        save_preds = bool(self.chk_save_preds.get())

        out_dir = data_dir / "predictions" / "weights_tab"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            "bash",
            str(self.sim_root / "predict.sh"),
            "--model",
            str(model_path),
            "--data",
            str(data_dir),
            "--split",
            split,
            "--out-dir",
            str(out_dir),
        ]
        if device != "auto":
            cmd.extend(["--device", device])
        if max_samples_s:
            int(max_samples_s)  # validate
            cmd.extend(["--max-samples", max_samples_s])
        if save_preds:
            cmd.append("--save-preds")
        return cmd, out_dir

    def _run_predict(self, *, model_path: Path, label: str) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Info", "A run is already in progress.")
            return
        if not self.state.dataset_dir:
            messagebox.showinfo("Info", "No dataset selected.")
            return
        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found:\n{model_path}")
            return

        try:
            cmd, out_dir = self._predict_cmd(model_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.stop_evt.clear()
        self.var_status.set(f"status: running ({label})")
        self._append_log(f"\n$ {' '.join(cmd)}\n")

        def worker() -> None:
            report_path: Optional[Path] = None
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.sim_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    if self.stop_evt.is_set():
                        break
                    self.log_q.put(line)
                    if "Report saved to:" in line:
                        try:
                            report_path = Path(line.split("Report saved to:", 1)[-1].strip())
                        except Exception:
                            report_path = None

                rc = self.proc.wait()
                if self.stop_evt.is_set():
                    self.log_q.put("\n[stopped]\n")
                elif rc != 0:
                    self.log_q.put(f"\n[error] predict.sh exited with code {rc}\n")
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run predict.sh: {e}\n")
            finally:
                try:
                    if report_path is None:
                        cand = sorted(out_dir.glob("batch_report_*.json"), key=lambda p: p.stat().st_mtime)
                        report_path = cand[-1] if cand else None
                except Exception:
                    pass

                def apply() -> None:
                    if self.stop_evt.is_set():
                        self.var_status.set("status: stopped")
                    else:
                        self.var_status.set("status: done")
                    if report_path and report_path.exists():
                        self._append_log(self._short_report(report_path))

                try:
                    self.frame.after(0, apply)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _run_predict_blocking(self, model_path: Path, *, label: str) -> Optional[Dict[str, Any]]:
        """Run predict.sh sequentially from a background thread; returns report json if found."""
        if not self.state.dataset_dir:
            self.log_q.put("[error] no dataset selected\n")
            return None
        try:
            cmd, out_dir = self._predict_cmd(model_path)
        except Exception as e:
            self.log_q.put(f"[error] {e}\n")
            return None

        self.stop_evt.clear()
        self.log_q.put(f"\n$ {' '.join(cmd)}\n")
        self.log_q.put(f"[run] {label}\n")
        report_path: Optional[Path] = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.sim_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if self.stop_evt.is_set():
                    break
                self.log_q.put(line)
                if "Report saved to:" in line:
                    try:
                        report_path = Path(line.split("Report saved to:", 1)[-1].strip())
                    except Exception:
                        report_path = None
            rc = proc.wait()
            if self.stop_evt.is_set():
                self.log_q.put("\n[stopped]\n")
                return None
            if rc != 0:
                self.log_q.put(f"\n[error] predict.sh exited with code {rc}\n")
                return None
        except Exception as e:
            self.log_q.put(f"\n[error] Failed to run predict.sh: {e}\n")
            return None

        if report_path is None:
            try:
                cand = sorted(out_dir.glob("batch_report_*.json"), key=lambda p: p.stat().st_mtime)
                report_path = cand[-1] if cand else None
            except Exception:
                report_path = None
        if not report_path or not report_path.exists():
            return None

        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _short_report(self, report_path: Path) -> str:
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            return f"[report] failed to load {report_path}: {e}\n"

        metrics = obj.get("metrics") or {}
        acc = float(metrics.get("accuracy") or 0.0) if "accuracy" in metrics else None
        f1 = float(metrics.get("macro_f1") or 0.0) if "macro_f1" in metrics else None
        seen = obj.get("seen_samples", "-")
        split = obj.get("split", "-")
        model = Path(obj.get("model_path", "-")).name
        ds = Path(obj.get("dataset_path", "-")).name

        lines = []
        lines.append("\n" + "-" * 60)
        lines.append(f"[report] {report_path.name}")
        lines.append(f"  model: {model}")
        lines.append(f"  data:  {ds}")
        lines.append(f"  split: {split}  seen: {seen}")
        if acc is not None:
            lines.append(f"  accuracy: {acc:.4f}")
        if f1 is not None:
            lines.append(f"  macro_f1:  {f1:.4f}")
        lines.append("-" * 60 + "\n")
        return "\n".join(lines)

    _RE_SNAP_RUN = re.compile(r"_run(?P<run>\d+)(?:_|\\.|$)")

    def _runs_for_model(self, p: Path) -> str:
        """Return a string representing how many pipeline runs this checkpoint corresponds to (best-effort)."""
        # Prefer meta json written by PipelineControlTab.
        meta = self._read_model_meta(p)
        if meta is not None:
            try:
                ri = int(meta.get("run_iteration"))
                return str(ri)
            except Exception:
                pass

        # Fallback: parse from snapshot filename.
        m = self._RE_SNAP_RUN.search(p.name)
        if m:
            try:
                return str(int(m.group("run")))
            except Exception:
                pass

        # If this is an "active" model (outputs/models/<stem>.pt), try to read its sibling meta file.
        sib = Path(str(p) + ".meta.json")
        if sib.exists():
            try:
                obj = json.loads(sib.read_text(encoding="utf-8"))
                return str(int(obj.get("run_iteration")))
            except Exception:
                pass

        return "-"

    def _read_model_meta(self, p: Path) -> Optional[Dict[str, Any]]:
        mp = Path(str(p) + ".meta.json")
        if not mp.exists():
            return None
        try:
            obj = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _report_search_dirs(self) -> List[Path]:
        dirs: List[Path] = []
        # Common default from scripts/batch_predict.py
        dirs.append(self.sim_root / "outputs" / "models")
        dirs.append(self.sim_root / "outputs" / "models" / "history")
        # PredictionsTab default out-dir: <dataset>/predictions
        if self.state.dataset_dir:
            dirs.append(self.state.dataset_dir / "predictions")
            dirs.append(self.state.dataset_dir / "predictions" / "weights_tab")
        # Dedup
        out: List[Path] = []
        seen = set()
        for d in dirs:
            try:
                rp = str(d.resolve())
            except Exception:
                rp = str(d)
            if rp in seen:
                continue
            seen.add(rp)
            out.append(d)
        return out

    def _load_reports_from_dirs(self, dirs: List[Path]) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        seen_paths = set()
        for d in dirs:
            if not d.exists():
                continue
            cand = []
            cand.extend(d.glob("report_*.json"))
            cand.extend(d.glob("batch_report_*.json"))
            for p in sorted(cand, key=lambda x: x.stat().st_mtime if x.exists() else 0.0, reverse=True):
                sp = str(p)
                if sp in seen_paths:
                    continue
                seen_paths.add(sp)
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict) or "metrics" not in obj:
                    continue
                obj["_path"] = sp
                reports.append(obj)
        return reports

    def _refresh_reports(self) -> None:
        dirs = self._report_search_dirs()
        self._reports = self._load_reports_from_dirs(dirs)

        scope = self.var_report_scope.get().strip() if hasattr(self, "var_report_scope") else "selected_model"
        split_filter = self.var_report_split.get().strip() if hasattr(self, "var_report_split") else "any"
        sort_key = self.var_report_sort.get().strip() if hasattr(self, "var_report_sort") else "newest"

        selected_model = self._selected_model_path()
        ds_dir = self.state.dataset_dir

        filtered: List[Dict[str, Any]] = []
        for r in self._reports:
            r_split = str(r.get("split") or "-")
            if split_filter != "any" and r_split != split_filter:
                continue

            if scope == "selected_model":
                if not selected_model:
                    continue
                rp = str(r.get("model_path") or "")
                # Match by basename; allows comparing versions anywhere.
                if Path(rp).name != selected_model.name:
                    # Some reports store "model_stem" only; fallback
                    if str(r.get("model_stem") or "") != selected_model.stem:
                        continue
            elif scope == "dataset":
                if not ds_dir:
                    continue
                if str(r.get("dataset_path") or "") != str(ds_dir):
                    continue

            filtered.append(r)

        def ts_of(rr: Dict[str, Any]) -> float:
            s = rr.get("timestamp") or ""
            try:
                return datetime.fromisoformat(str(s)).timestamp()
            except Exception:
                try:
                    return os.path.getmtime(str(rr.get("_path") or ""))
                except Exception:
                    return 0.0

        def metric(rr: Dict[str, Any], k: str) -> float:
            try:
                return float((rr.get("metrics") or {}).get(k) or 0.0)
            except Exception:
                return 0.0

        if sort_key == "accuracy":
            filtered.sort(key=lambda rr: metric(rr, "accuracy"), reverse=True)
        elif sort_key == "macro_f1":
            filtered.sort(key=lambda rr: metric(rr, "macro_f1"), reverse=True)
        else:
            filtered.sort(key=ts_of, reverse=True)

        # Render
        self.tree_reports.delete(*self.tree_reports.get_children())
        self._report_by_iid.clear()

        for i, r in enumerate(filtered):
            iid = f"r{i}"
            when = str(r.get("timestamp") or "-")
            when = when.replace("T", " ")[:19]
            split = str(r.get("split") or "-")
            seen = r.get("seen_samples") or r.get("test_size") or "-"
            try:
                seen_s = str(int(seen))
            except Exception:
                seen_s = str(seen)

            m = r.get("metrics") or {}
            acc = m.get("accuracy")
            f1 = m.get("macro_f1")
            acc_s = f"{float(acc):.4f}" if acc is not None else "-"
            f1_s = f"{float(f1):.4f}" if f1 is not None else "-"

            model_name = Path(str(r.get("model_path") or "-")).name
            dataset_name = Path(str(r.get("dataset_path") or "-")).name
            path = str(r.get("_path") or "-")
            path_disp = self._rel(Path(path)) if path != "-" else "-"

            self.tree_reports.insert(
                "",
                "end",
                iid=iid,
                values=(when, split, seen_s, acc_s, f1_s, model_name, dataset_name, path_disp),
            )
            self._report_by_iid[iid] = r

    def _show_selected_report_summary(self) -> None:
        sel = self.tree_reports.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a report first.")
            return
        r = self._report_by_iid.get(sel[0])
        if not r:
            return
        p = r.get("_path")
        if p and Path(str(p)).exists():
            self._append_log(self._short_report(Path(str(p))))
        else:
            # Fallback: print a minimal inline summary
            try:
                m = r.get("metrics") or {}
                acc = float(m.get("accuracy") or 0.0)
                f1 = float(m.get("macro_f1") or 0.0)
                self._append_log(
                    f"\n[report] ts={r.get('timestamp')} split={r.get('split')} acc={acc:.4f} f1={f1:.4f} path={p}\n"
                )
            except Exception:
                pass
