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
import zipfile
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from tkinter import simpledialog
from typing import Any, Dict, List, Optional, Tuple

from .base_tab import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore


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

        self._group_assignments: Dict[str, str] = {}
        self._custom_groups: List[str] = []
        self._group_iids: Dict[str, str] = {}
        self._drag_src_iid: Optional[str] = None

        self._reports: List[Dict[str, Any]] = []
        self._report_by_iid: Dict[str, Dict[str, Any]] = {}

        # Compare dataset selection (multi)
        self._cmp_dataset_dirs: List[Path] = []
        self._cmp_dataset_labels: List[str] = []
        self._cmp_dataset_by_label: Dict[str, Path] = {}

        # UI components
        self.var_dataset: tk.StringVar
        self.var_active_model: tk.StringVar
        self.var_selected_model: tk.StringVar
        self.var_dataset_samples: tk.StringVar

        self.var_split: tk.StringVar
        self.var_device: tk.StringVar
        self.var_max_samples: tk.StringVar
        self.chk_save_preds: tk.BooleanVar

        self.var_cmp_a: tk.StringVar
        self.var_cmp_b: tk.StringVar
        self.var_report_scope: tk.StringVar
        self.var_report_split: tk.StringVar
        self.var_report_sort: tk.StringVar

        self.list_cmp_datasets: tk.Listbox

        self.tree: ttk.Treeview
        self.tree_reports: ttk.Treeview
        self.txt: tk.Text
        self.var_status: tk.StringVar

        self._ui_tick_id: Optional[str] = None
        self._models_menu: Optional[tk.Menu] = None
        self._reports_menu: Optional[tk.Menu] = None

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_dataset, width=60, state="readonly").pack(side="left", padx=(5, 10))
        self.var_dataset_samples = tk.StringVar(value="Samples: -")
        ttk.Label(top, textvariable=self.var_dataset_samples).pack(side="left", padx=(0, 10))

        ttk.Label(top, text="Active model:").pack(side="left")
        self.var_active_model = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_active_model, width=45).pack(side="left", padx=(5, 10))

        ttk.Button(top, text="Refresh", command=self._refresh_models).pack(side="left")
        ttk.Button(top, text="Snapshot active", command=self._snapshot_active).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Import...", command=self._import_model).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Export selected...", command=self._export_selected).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Activate selected", command=self._activate_selected).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Profile Info", command=self._show_model_profile_info).pack(side="left", padx=(8, 0))
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
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Fav", "Runs", "Size", "MTime", "Path", "Group"),
            show="tree headings",
            selectmode="extended",
        )
        self.tree.heading("#0", text="Model")
        self.tree.column("#0", width=200, anchor="w")
        self.tree.heading("Fav", text="Fav")
        self.tree.heading("Runs", text="Runs")
        self.tree.heading("Size", text="Size")
        self.tree.heading("MTime", text="Modified")
        self.tree.heading("Path", text="Path")
        self.tree.heading("Group", text="Group")
        self.tree.column("Fav", width=50, anchor="center")
        self.tree.column("Runs", width=60, anchor="e")
        self.tree.column("Size", width=80, anchor="e")
        self.tree.column("MTime", width=140, anchor="w")
        self.tree.column("Path", width=400, anchor="w")
        self.tree.column("Group", width=120, anchor="w")
        self.tree.tag_configure("group_header", font=("TkDefaultFont", 10, "bold"))

        scroll_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        # Right-click context menu for quick actions (delete/export/activate).
        self.tree.bind("<Button-3>", self._on_models_right_click)
        self.tree.bind("<Button-2>", self._on_models_right_click)  # macOS
        self.tree.bind("<ButtonPress-1>", self._on_tree_mouse_down, add="+")
        self.tree.bind("<ButtonRelease-1>", self._on_tree_mouse_up, add="+")

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
        self._models_menu.add_command(label="Rename selected...", command=self._rename_selected)
        self._models_menu.add_command(label="Duplicate selected...", command=self._duplicate_selected)
        self._models_menu.add_command(label="Delete selected", command=self._delete_selected)
        self._models_menu.add_separator()
        self._groups_menu = tk.Menu(self._models_menu, tearoff=0)
        self._models_menu.add_cascade(label="Move to group...", menu=self._groups_menu)
        self._models_menu.add_command(label="New group...", command=self._prompt_new_group)
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

        ttk.Button(cmpbox, text="Run A then B", command=self._run_compare).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        # Dataset selection for compare (multi-select)
        ttk.Label(cmpbox, text="Datasets").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        ds_frame = ttk.Frame(cmpbox)
        ds_frame.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        ds_frame.columnconfigure(0, weight=1)

        self.list_cmp_datasets = tk.Listbox(
            ds_frame,
            height=4,
            selectmode="extended",
            exportselection=False,
        )
        ds_scroll = ttk.Scrollbar(ds_frame, orient="vertical", command=self.list_cmp_datasets.yview)
        self.list_cmp_datasets.configure(yscrollcommand=ds_scroll.set)
        self.list_cmp_datasets.grid(row=0, column=0, sticky="ew")
        ds_scroll.grid(row=0, column=1, sticky="ns")

        ds_btns = ttk.Frame(ds_frame)
        ds_btns.grid(row=0, column=2, sticky="ns", padx=(10, 0))
        ttk.Button(ds_btns, text="Refresh", command=self._refresh_cmp_datasets).pack(fill="x")
        ttk.Button(ds_btns, text="Use current", command=self._select_current_dataset_for_compare).pack(fill="x", pady=(6, 0))
        ttk.Button(ds_btns, text="Select all", command=self._select_all_datasets_for_compare).pack(fill="x", pady=(6, 0))

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
        rep_tree_frame.columnconfigure(0, weight=1)
        rep_tree_frame.rowconfigure(0, weight=1)

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

        rep_scroll_y = ttk.Scrollbar(rep_tree_frame, orient="vertical", command=self.tree_reports.yview)
        rep_scroll_x = ttk.Scrollbar(rep_tree_frame, orient="horizontal", command=self.tree_reports.xview)
        self.tree_reports.configure(yscrollcommand=rep_scroll_y.set, xscrollcommand=rep_scroll_x.set)
        self.tree_reports.grid(row=0, column=0, sticky="nsew")
        rep_scroll_y.grid(row=0, column=1, sticky="ns")
        rep_scroll_x.grid(row=1, column=0, sticky="ew")

        # Right-click context menu for report rows.
        self._reports_menu = tk.Menu(self.frame, tearoff=0)
        self._reports_menu.add_command(label="Delete report file", command=self._delete_selected_report)
        self.tree_reports.bind("<Button-3>", self._on_reports_right_click)
        self.tree_reports.bind("<Button-2>", self._on_reports_right_click)  # macOS

        ttk.Label(right, text="Logs / Output").pack(anchor="w")
        self.txt = tk.Text(right, height=14, wrap="none")
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

        self.on_dataset_changed()
        self._refresh_cmp_datasets()
        self._load_favorites()
        self._refresh_models()
        self._refresh_reports()
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self._tick_ui()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir:
            return
        self.var_dataset.set(str(self.state.dataset_dir))

        active = self.sim_root / "outputs" / "models" / f"{self.state.dataset_dir.name}.pt"
        self.var_active_model.set(str(active))
        self.var_dataset_samples.set(self._describe_dataset_samples(self.state.dataset_dir))
        # If user hasn't selected compare datasets yet, default to current dataset.
        try:
            if hasattr(self, "list_cmp_datasets") and self.list_cmp_datasets.size() > 0:
                if not self.list_cmp_datasets.curselection():
                    self._select_current_dataset_for_compare()
        except Exception:
            pass

    def _sim_data_roots(self) -> tuple[Path, Path]:
        sim_data = self.sim_root / "outputs" / "sim_data"
        return sim_data / "runs", sim_data / "versions"

    def _display_for_dataset(self, p: Path, *, runs: Path, versions: Path) -> str:
        try:
            rp = p.resolve()
            if str(rp).startswith(str(runs.resolve()) + os.sep):
                return rp.name
            if str(rp).startswith(str(versions.resolve()) + os.sep):
                return f"{rp.parent.name}:{rp.name}"
        except Exception:
            pass
        return p.name

    def _refresh_cmp_datasets(self) -> None:
        """Refresh dataset list used for compare runs (multi-select)."""
        runs, versions = self._sim_data_roots()
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)

        cand: list[Path] = []
        cand.extend([p for p in runs.iterdir() if p.is_dir()])
        for p in versions.glob("*/*"):
            if p.is_dir():
                cand.append(p)

        def is_version(p: Path) -> bool:
            try:
                return str(p.resolve()).startswith(str(versions.resolve()) + os.sep)
            except Exception:
                return False

        def sort_key(p: Path) -> tuple:
            if is_version(p):
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    mt = 0.0
                return (1, -mt, p.name)
            return (0, p.name)

        cand.sort(key=sort_key)
        self._cmp_dataset_dirs = cand

        # Preserve existing selections by label.
        old_sel: set[str] = set()
        try:
            for i in self.list_cmp_datasets.curselection():
                old_sel.add(str(self.list_cmp_datasets.get(i)))
        except Exception:
            pass

        self._cmp_dataset_labels = []
        self._cmp_dataset_by_label = {}
        seen: dict[str, int] = {}
        for p in cand:
            base = self._display_for_dataset(p, runs=runs, versions=versions)
            n = seen.get(base, 0) + 1
            seen[base] = n
            label = base if n == 1 else f"{base} ({n})"
            self._cmp_dataset_labels.append(label)
            self._cmp_dataset_by_label[label] = p

        self.list_cmp_datasets.delete(0, "end")
        for label in self._cmp_dataset_labels:
            self.list_cmp_datasets.insert("end", label)

        # Restore old selection where possible.
        if old_sel:
            for idx, label in enumerate(self._cmp_dataset_labels):
                if label in old_sel:
                    try:
                        self.list_cmp_datasets.selection_set(idx)
                    except Exception:
                        pass
        else:
            self._select_current_dataset_for_compare()

    def _select_current_dataset_for_compare(self) -> None:
        """Select only the current dataset in the compare dataset list (if present)."""
        try:
            self.list_cmp_datasets.selection_clear(0, "end")
        except Exception:
            return
        if not self.state.dataset_dir:
            return
        runs, versions = self._sim_data_roots()
        want = self._display_for_dataset(self.state.dataset_dir, runs=runs, versions=versions)
        # Might have been disambiguated.
        for idx, label in enumerate(self._cmp_dataset_labels):
            if label == want:
                self.list_cmp_datasets.selection_set(idx)
                self.list_cmp_datasets.see(idx)
                return
            p = self._cmp_dataset_by_label.get(label)
            if p and p == self.state.dataset_dir:
                self.list_cmp_datasets.selection_set(idx)
                self.list_cmp_datasets.see(idx)
                return

    def _select_all_datasets_for_compare(self) -> None:
        try:
            self.list_cmp_datasets.selection_set(0, "end")
        except Exception:
            pass

    def _selected_compare_datasets(self) -> List[Path]:
        ds: List[Path] = []
        try:
            for i in self.list_cmp_datasets.curselection():
                label = str(self.list_cmp_datasets.get(i))
                p = self._cmp_dataset_by_label.get(label)
                if p:
                    ds.append(p)
        except Exception:
            pass
        # Fallback: current dataset.
        if not ds and self.state.dataset_dir:
            ds = [self.state.dataset_dir]
        # Filter out non-existent dirs (stale list).
        out: List[Path] = []
        for p in ds:
            try:
                if p.exists() and p.is_dir():
                    out.append(p)
            except Exception:
                continue
        return out

    def _dataset_classes(self, ds_dir: Path) -> Optional[List[str]]:
        """Best-effort dataset class list (from manifest if present, else from labels.jsonl)."""
        try:
            mp = ds_dir / "dataset_manifest.json"
            if mp.exists():
                obj = json.loads(mp.read_text(encoding="utf-8"))
                classes = (obj.get("dataset_stats") or {}).get("classes") or {}
                if isinstance(classes, dict):
                    keys = [str(k) for k in classes.keys() if k]
                    if keys:
                        return sorted(set(keys))
        except Exception:
            pass
        try:
            lp = ds_dir / "labels.jsonl"
            if not lp.exists():
                return None
            seen: set[str] = set()
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    c = obj.get("class_name")
                    if isinstance(c, str) and c:
                        seen.add(c)
            return sorted(seen) if seen else None
        except Exception:
            return None

    def _model_classes(self, model_path: Path) -> Optional[List[str]]:
        """Best-effort model class list from checkpoint metadata."""
        try:
            import torch  # local import to keep module import light
            ckpt = torch.load(model_path, map_location="cpu")
            cls = ckpt.get("class_names")
            if isinstance(cls, list) and cls:
                out = [str(x) for x in cls if isinstance(x, (str, int, float))]
                out = [s for s in out if s]
                return sorted(set(out)) if out else None
        except Exception:
            pass
        return None

    def _describe_dataset_samples(self, ds: Optional[Path]) -> str:
        if ds is None:
            return "Samples: -"
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists():
            return "Samples: legacy dataset"
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            stats = manifest.get("dataset_stats", {})
            total = stats.get("total_samples")
            splits = stats.get("splits", {})
            train = splits.get("train")
            val = splits.get("val")
            test = splits.get("test")
            if total is None:
                return "Samples: metadata missing"
            desc = f"Samples: {total}"
            if train is not None and val is not None and test is not None:
                desc += f" (t{train}/v{val}/s{test})"
            return desc
        except Exception:
            return "Samples: error"

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None:
            return
        for key, var in [
            ("weights.split", self.var_split),
            ("weights.device", self.var_device),
            ("weights.max_samples", self.var_max_samples),
            ("weights.save_preds", self.chk_save_preds),
            ("weights.report_scope", self.var_report_scope),
            ("weights.report_split", self.var_report_split),
            ("weights.report_sort", self.var_report_sort),
            ("weights.compare_a", self.var_cmp_a),
            ("weights.compare_b", self.var_cmp_b),
        ]:
            v = st.get(key)
            if v is None:
                continue
            try:
                var.set(v)
            except Exception:
                pass
        try:
            self._refresh_reports()
        except Exception:
            pass
        # Load custom group data
        custom_raw = st.get("weights.custom_groups")
        if isinstance(custom_raw, str):
            try:
                parsed = json.loads(custom_raw)
                if isinstance(parsed, list):
                    self._custom_groups = [str(k) for k in parsed if isinstance(k, str) and k]
            except Exception:
                pass
        assignments_raw = st.get("weights.group_assignments")
        if isinstance(assignments_raw, str):
            try:
                parsed = json.loads(assignments_raw)
                if isinstance(parsed, dict):
                    self._group_assignments = {
                        str(k): str(v) for k, v in parsed.items() if k and v
                    }
            except Exception:
                pass

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None:
            return

        def bind(var, key: str) -> None:
            def cb(*_a) -> None:
                try:
                    st.set(key, var.get())
                    st.schedule_save(self.frame)
                except Exception:
                    pass

            try:
                var.trace_add("write", cb)
            except Exception:
                try:
                    var.trace("w", cb)
                except Exception:
                    pass

        bind(self.var_split, "weights.split")
        bind(self.var_device, "weights.device")
        bind(self.var_max_samples, "weights.max_samples")
        bind(self.chk_save_preds, "weights.save_preds")
        bind(self.var_report_scope, "weights.report_scope")
        bind(self.var_report_split, "weights.report_split")
        bind(self.var_report_sort, "weights.report_sort")
        bind(self.var_cmp_a, "weights.compare_a")
        bind(self.var_cmp_b, "weights.compare_b")

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
        # Include bundle directories
        cand.extend([p for p in sorted(root.glob("*.bundle")) if p.is_dir()])
        bundles_subdir = root / "bundles"
        if bundles_subdir.exists():
            cand.extend([p for p in sorted(bundles_subdir.iterdir()) if p.is_dir()])

        # Deduplicate + sort by (favorite first, then mtime desc)
        uniq: Dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file() or p.is_dir():
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

        # Preserve expanded groups and selection before rebuilding
        prev_open_groups: set = set()
        for child in self.tree.get_children():
            if child.startswith("group::") and self.tree.item(child, "open"):
                prev_open_groups.add(child)

        prev_selected_paths: set = set()
        for iid in self.tree.selection():
            if iid in self._model_by_iid:
                prev_selected_paths.add(str(self._model_by_iid[iid].resolve()))

        self.tree.delete(*self.tree.get_children())
        self._model_by_iid.clear()
        self._group_iids.clear()

        display_values: List[str] = [self._rel(p) for p in self._models]

        group_names = self._available_groups()
        group_members: Dict[str, List[Path]] = {gn: [] for gn in group_names}
        for p in self._models:
            group = self._group_for_path(p)
            if group not in group_members:
                group_members[group] = []
            group_members[group].append(p)

        restore_selection: List[str] = []
        item_counter = 0
        for group in group_names:
            group_iid = f"group::{group}"
            count = len(group_members.get(group, []))
            label = f"{group} ({count})" if count else group
            self.tree.insert(
                "",
                "end",
                iid=group_iid,
                text=label,
                values=("", "", "", "", "", f"{count} models"),
                tags=("group_header",),
                open=group_iid in prev_open_groups,
            )
            self._group_iids[group] = group_iid
            for p in group_members.get(group, []):
                st = p.stat()
                is_bundle = p.is_dir()
                if is_bundle:
                    # Sum size of all files in bundle directory
                    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    size_s = self._fmt_bytes(int(total))
                    display_name = f"[Bundle] {p.name}"
                else:
                    size_s = self._fmt_bytes(int(st.st_size))
                    display_name = p.name
                mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                rel = self._rel(p)
                runs_s = self._runs_for_model(p)
                fav_s = "★" if self._is_favorited(p) else ""
                iid = f"model::{item_counter}"
                item_counter += 1
                self.tree.insert(
                    group_iid,
                    "end",
                    iid=iid,
                    text=display_name,
                    values=(fav_s, runs_s, size_s, mt, rel, group),
                    tags=("model_entry",),
                )
                self._model_by_iid[iid] = p
                if str(p.resolve()) in prev_selected_paths:
                    restore_selection.append(iid)

        # Restore previous selection
        if restore_selection:
            self.tree.selection_set(restore_selection)

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

        # Keep group menu entries in sync with the available groups.
        self._refresh_groups_menu()

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            return str(p)

    def _available_groups(self) -> List[str]:
        """Return the list of groups to show in the UI."""
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
        if self._is_favorited(p):
            return "Favorites"
        rp = p.resolve()
        root = self._model_root().resolve()
        versions = (root / "versions").resolve()
        imports = (root / "imports").resolve()
        try:
            rp_str = str(rp)
        except Exception:
            rp_str = ""
        if versions and rp_str.startswith(str(versions) + os.sep):
            return "Snapshots"
        if imports and rp_str.startswith(str(imports) + os.sep):
            return "Imports"
        return "Uncategorized"

    def _is_group_header(self, iid: str) -> bool:
        return bool(iid and iid.startswith("group::"))

    def _group_from_iid(self, iid: str) -> Optional[str]:
        if not iid or "::" not in iid:
            return None
        return iid.split("::", 1)[1]

    def _refresh_groups_menu(self) -> None:
        if not self._groups_menu:
            return
        self._groups_menu.delete(0, "end")
        for group in self._available_groups():
            self._groups_menu.add_command(
                label=group,
                command=lambda g=group: self._assign_selected_to_group(self._selected_model_paths(), g),
            )

    def _assign_selected_to_group(self, paths: List[Path], group: Optional[str]) -> None:
        if not paths or not group:
            return
        fav_changed = False
        for p in paths:
            rel = self._rel(p)
            if group == "Favorites":
                if rel not in self._favorites:
                    self._favorites[rel] = {
                        "path": rel,
                        "added": datetime.now().isoformat(timespec="seconds"),
                    }
                    fav_changed = True
            else:
                if rel in self._favorites:
                    self._favorites.pop(rel, None)
                    fav_changed = True
            self._group_assignments[rel] = group
        if fav_changed:
            self._save_favorites()
        self._save_group_state()
        self._append_log(f"[group] assigned {len(paths)} model(s) -> {group}\n")
        self._refresh_models()

    def _save_group_state(self) -> None:
        st = self._store()
        if st is None:
            return
        try:
            st.set("weights.group_assignments", json.dumps(self._group_assignments, ensure_ascii=True))
            st.set("weights.custom_groups", json.dumps(self._custom_groups, ensure_ascii=True))
            st.schedule_save(self.frame)
        except Exception:
            pass

    def _prompt_new_group(self) -> None:
        name = simpledialog.askstring(
            "New group",
            "Group name:",
            parent=self.frame.winfo_toplevel(),
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if "::" in name:
            messagebox.showerror("Error", "Group name must not contain '::'.")
            return
        if name in self._available_groups():
            messagebox.showinfo("Info", f"Group already exists: {name}")
            return
        self._custom_groups.append(name)
        self._save_group_state()
        self._refresh_models()

    def _duplicate_selected(self) -> None:
        paths = self._selected_model_paths()
        if not paths:
            messagebox.showinfo("Info", "Select a checkpoint first.")
            return
        if len(paths) != 1:
            messagebox.showinfo("Info", "Select exactly one checkpoint to duplicate.")
            return
        src = paths[0]
        new_name = simpledialog.askstring(
            "Duplicate checkpoint",
            "New checkpoint name (without extension):",
            initialvalue=src.stem,
            parent=self.frame.winfo_toplevel(),
        )
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if "/" in new_name or "\\" in new_name:
            messagebox.showerror("Error", "Name must not contain path separators.")
            return
        dst = src.with_name(new_name + src.suffix)
        if dst.exists():
            messagebox.showerror("Error", f"Target already exists:\n{dst}")
            return
        try:
            shutil.copy2(src, dst)
            meta_src = Path(str(src) + ".meta.json")
            meta_dst = Path(str(dst) + ".meta.json")
            if meta_src.exists():
                shutil.copy2(meta_src, meta_dst)
        except Exception as e:
            messagebox.showerror("Error", f"Duplicate failed:\n{e}")
            return
        rel_src = self._rel(src)
        rel_dst = self._rel(dst)
        if rel_src in self._group_assignments:
            self._group_assignments[rel_dst] = self._group_assignments[rel_src]
            self._save_group_state()
        if self._is_favorited(src):
            self._favorites[rel_dst] = {
                "path": rel_dst,
                "added": datetime.now().isoformat(timespec="seconds"),
            }
            self._save_favorites()
        self._append_log(f"[duplicate] {self._rel(src)} -> {self._rel(dst)}\n")
        self._refresh_models()

    def _on_tree_mouse_down(self, event: tk.Event) -> None:
        iid = self.tree.identify_row(event.y)
        if iid and not self._is_group_header(iid):
            self._drag_src_iid = iid
        else:
            self._drag_src_iid = None

    def _on_tree_mouse_up(self, event: tk.Event) -> None:
        if not self._drag_src_iid:
            return
        target = self.tree.identify_row(event.y)
        if not target or not self._is_group_header(target):
            self._drag_src_iid = None
            return
        group = self._group_from_iid(target)
        if group:
            self._assign_selected_to_group(self._selected_model_paths(), group)
        self._drag_src_iid = None

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

        # Load profile info from checkpoint
        profile_info = ""
        try:
            import torch
            checkpoint = torch.load(p, map_location='cpu')
            profile_data = checkpoint.get('component_profile')
            if profile_data:
                profile_id = profile_data.get('profile_id', 'unknown')
                profile_hash = profile_data.get('profile_hash', '')
                hash_short = profile_hash.split(':')[1][:12] if ':' in profile_hash else profile_hash[:12]
                profile_info = f" | Profile: {profile_id} ({hash_short}...)"
            else:
                profile_info = " | Profile: ⚠ No profile (legacy)"
        except Exception as e:
            profile_info = f" | Profile: ⚠ Error loading"

        self.var_selected_model.set(f"selected: {self._rel(p)}{profile_info}")

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
        self._refresh_groups_menu()
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
            can_rename = any(self._is_deletable_checkpoint(p) for p in paths)
            self._models_menu.entryconfigure("Rename selected...", state=("normal" if can_rename else "disabled"))
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
        try:
            if self._models_menu:
                self._models_menu.unpost()
        except Exception:
            pass
        try:
            if self._reports_menu:
                self._reports_menu.unpost()
        except Exception:
            pass

    def _on_reports_right_click(self, event: tk.Event) -> None:
        if not self._reports_menu:
            return
        try:
            iid = self.tree_reports.identify_row(event.y)
        except Exception:
            iid = ""
        if not iid:
            return
        try:
            self.tree_reports.selection_set(iid)
            self.tree_reports.focus(iid)
        except Exception:
            pass
        try:
            self._reports_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self._reports_menu.grab_release()
            except Exception:
                pass

    def _delete_selected_report(self) -> None:
        sel = self.tree_reports.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a report first.")
            return
        r = self._report_by_iid.get(sel[0])
        if not r:
            return
        p = r.get("_path")
        if not p:
            messagebox.showerror("Error", "Report path missing.")
            return
        rp = Path(str(p))
        if not rp.exists():
            messagebox.showwarning("Not found", f"Report file not found:\n{rp}")
            return

        # Safety: only allow deleting files under this repo root.
        try:
            root = self.sim_root.resolve()
            rr = rp.resolve()
            rr.relative_to(root)
        except Exception:
            messagebox.showerror("Blocked", f"Refusing to delete file outside repo:\n{rp}")
            return

        if not messagebox.askyesno("Delete report", f"Delete this report file?\n\n{rp}"):
            return
        try:
            rp.unlink()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete:\n{rp}\n\n{e}")
            return

        self._append_log(f"[delete] report: {rp}\n")
        try:
            self._refresh_reports()
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
        paths = self._selected_model_paths()
        if not paths:
            return
        self._assign_selected_to_group(paths, "Favorites")

    def _remove_selected_from_favorites(self) -> None:
        paths = [p for p in self._selected_model_paths() if self._is_favorited(p)]
        if not paths:
            return
        names = "\n".join(self._rel(p) for p in paths[:8])
        if len(paths) > 8:
            names += f"\n... (+{len(paths) - 8} more)"
        if not messagebox.askyesno("Remove favorite", f"Remove from favorites?\n\n{names}"):
            return
        self._assign_selected_to_group(paths, "Uncategorized")

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
            title="Import model (.pt) or bundle (.zip)",
            filetypes=[
                ("PyTorch checkpoint", "*.pt"),
                ("Bundle archive", "*.zip"),
                ("All files", "*.*"),
            ],
        )
        if not src:
            return
        src_p = Path(src)
        if not src_p.exists():
            return

        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Bundle import (.zip): extract into outputs/models/bundles/<name>.bundle/
        if src_p.suffix.lower() == ".zip":
            bundles_dir = self._model_root() / "bundles"
            bundles_dir.mkdir(parents=True, exist_ok=True)

            # Prefer keeping a ".bundle" suffix in the extracted folder name.
            stem = src_p.stem  # "foo.bundle" if file is foo.bundle.zip
            if not stem.endswith(".bundle"):
                stem = f"{stem}.bundle"
            dst = bundles_dir / f"{tag}_{stem}"

            # Avoid collisions.
            if dst.exists():
                for i in range(1, 1000):
                    cand = bundles_dir / f"{tag}_{src_p.stem}_{i:03d}.bundle"
                    if not cand.exists():
                        dst = cand
                        break

            tmp = bundles_dir / f".tmp_import_{tag}"
            try:
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(src_p, "r") as zf:
                    members = zf.namelist()
                    if not members:
                        raise ValueError("Empty zip archive")

                    # Safe extract: prevent absolute paths and path traversal.
                    for m in members:
                        mp = Path(m)
                        if mp.is_absolute() or ".." in mp.parts:
                            raise ValueError(f"Unsafe path in zip: {m}")
                    zf.extractall(tmp)

                # If zip contains a single top-level dir, use it; else use tmp itself.
                kids = [p for p in tmp.iterdir()]
                top_dirs = [p for p in kids if p.is_dir()]
                extracted_root = tmp
                if len(kids) == 1 and kids[0].is_dir():
                    extracted_root = kids[0]

                # If extracted_root isn't bundle-like, but contains exactly one *.bundle dir, use that.
                if not extracted_root.name.endswith(".bundle"):
                    bundle_dirs = [p for p in extracted_root.iterdir() if p.is_dir() and p.name.endswith(".bundle")]
                    if len(bundle_dirs) == 1:
                        extracted_root = bundle_dirs[0]

                pts = list(extracted_root.glob("*.pt"))
                if not pts:
                    raise ValueError("Bundle archive contains no .pt checkpoints at the expected level")

                # Move into place.
                shutil.move(str(extracted_root), str(dst))
            except Exception as e:
                try:
                    if dst.exists():
                        shutil.rmtree(dst)
                except Exception:
                    pass
                messagebox.showerror("Error", f"Bundle import failed:\n{e}")
                return
            finally:
                try:
                    if tmp.exists():
                        shutil.rmtree(tmp)
                except Exception:
                    pass

            self._append_log(f"[import] bundle {src_p} -> {dst}\n")
            self._refresh_models()
            return

        # Default: single checkpoint import (.pt)
        dst_dir = self._model_root() / "imports"
        dst_dir.mkdir(parents=True, exist_ok=True)
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
        # Bundles are directories; export them as a .zip archive by default.
        if p.is_dir():
            dst = filedialog.asksaveasfilename(
                title="Export selected bundle (zip)",
                initialfile=f"{p.name}.zip" if not p.name.endswith(".zip") else p.name,
                defaultextension=".zip",
                filetypes=[("Bundle archive", "*.zip"), ("All files", "*.*")],
            )
            if not dst:
                return
            dst_p = Path(dst)
            try:
                # shutil.make_archive wants a base name without extension.
                base = dst_p
                if base.suffix.lower() == ".zip":
                    base = base.with_suffix("")
                # Include the bundle directory itself in the archive.
                shutil.make_archive(str(base), "zip", root_dir=str(p.parent), base_dir=p.name)
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{e}")
                return
            self._append_log(f"[export] bundle {p} -> {dst_p.with_suffix('.zip')}\n")
            return

        dst = filedialog.asksaveasfilename(
            title="Export selected model",
            initialfile=p.name,
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
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
        """Allow deleting any checkpoint inside outputs/models/ (avoid accidental deletion elsewhere)."""
        try:
            rp = p.resolve()
            root = (self.sim_root / "outputs" / "models").resolve()
            return str(rp).startswith(str(root) + os.sep)
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

    def _show_model_profile_info(self) -> None:
        """Show detailed profile information for the selected model."""
        import yaml

        p = self._selected_model_path()
        if not p:
            messagebox.showinfo("Model Profile", "No model selected")
            return

        try:
            import torch
            checkpoint = torch.load(p, map_location='cpu')

            profile_data = checkpoint.get('component_profile')
            if not profile_data:
                messagebox.showinfo(
                    "No Profile",
                    "This model checkpoint has no component profile metadata.\n\n"
                    "It was likely trained before the profile system was implemented."
                )
                return

            profile_id = profile_data.get('profile_id', 'unknown')
            profile_hash = profile_data.get('profile_hash', '')
            dataset_path = checkpoint.get('trained_on_dataset', 'unknown')
            manifest_hash = checkpoint.get('dataset_manifest_hash', '')

            # Try to load the actual profile
            profiles_dir = self.sim_root / "configs" / "profiles"
            profile_path = profiles_dir / f"{profile_id}.yaml"

            # Create info dialog
            dialog = tk.Toplevel(self.frame)
            dialog.title(f"Model Profile: {p.name}")
            dialog.geometry("650x550")

            # Create text widget with scrollbar
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)

            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side="right", fill="y")

            text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10))
            text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)

            # Format and display info
            info_text = f"Model: {p.name}\n"
            info_text += f"Path: {p}\n\n"
            info_text += "="*60 + "\n"
            info_text += "COMPONENT PROFILE\n"
            info_text += "="*60 + "\n"
            info_text += f"Profile ID: {profile_id}\n"
            info_text += f"Profile Hash: {profile_hash}\n"
            info_text += f"Trained on dataset: {dataset_path}\n"
            info_text += f"Dataset manifest hash: {manifest_hash}\n"

            if profile_path.exists():
                info_text += "\n" + "="*60 + "\n"
                info_text += "PROFILE DETAILS\n"
                info_text += "="*60 + "\n\n"
                with open(profile_path, 'r') as f:
                    profile_yaml = yaml.safe_load(f)
                info_text += yaml.dump(profile_yaml, default_flow_style=False, sort_keys=False)
            else:
                info_text += f"\n\n⚠ Profile file not found at: {profile_path}"

            # Also show some checkpoint metadata
            info_text += "\n" + "="*60 + "\n"
            info_text += "CHECKPOINT METADATA\n"
            info_text += "="*60 + "\n"
            info_text += f"Epoch: {checkpoint.get('epoch', 'unknown')}\n"
            info_text += f"Val F1: {checkpoint.get('val_f1', 'unknown')}\n"
            info_text += f"Val Accuracy: {checkpoint.get('val_accuracy', 'unknown')}\n"
            info_text += f"Classes: {checkpoint.get('class_names', 'unknown')}\n"

            text.insert("1.0", info_text)
            text.config(state="disabled")

            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model profile info:\n{str(e)}")

    def _rename_selected(self) -> None:
        paths = self._selected_model_paths()
        if not paths:
            messagebox.showinfo("Info", "Select a snapshot to rename.")
            return
        if len(paths) != 1:
            messagebox.showinfo("Info", "Select exactly one checkpoint to rename.")
            return

        src = paths[0]
        if not self._is_deletable_checkpoint(src):
            messagebox.showwarning("Not allowed", "Only snapshots/imports can be renamed from here.")
            return

        old_rel = self._rel(src)
        old_name = src.stem
        new_stem = simpledialog.askstring(
            "Rename checkpoint",
            "New name (no path, no extension):",
            initialvalue=old_name,
            parent=self.frame.winfo_toplevel(),
        )
        if not new_stem:
            return
        new_stem = new_stem.strip()
        if not new_stem:
            return
        # Basic safety: disallow path separators.
        if "/" in new_stem or "\\" in new_stem:
            messagebox.showerror("Error", "Name must not contain path separators.")
            return

        dst = src.with_name(new_stem + src.suffix)
        if dst.exists():
            messagebox.showerror("Error", f"Target already exists:\n{dst}")
            return

        # If favorited, we update the favorites entry to the new path.
        was_fav = old_rel in self._favorites
        fav_entry = self._favorites.get(old_rel) if was_fav else None

        try:
            src.rename(dst)
            # Rename adjacent meta file if present.
            src_meta = Path(str(src) + ".meta.json")
            dst_meta = Path(str(dst) + ".meta.json")
            if src_meta.exists() and not dst_meta.exists():
                try:
                    src_meta.rename(dst_meta)
                except Exception:
                    pass
        except Exception as e:
            messagebox.showerror("Error", f"Rename failed:\n{e}")
            return

        if was_fav and fav_entry is not None:
            try:
                self._favorites.pop(old_rel, None)
                fav_entry = dict(fav_entry)
                fav_entry["path"] = self._rel(dst)
                fav_entry["renamed"] = datetime.now().isoformat(timespec="seconds")
                self._favorites[fav_entry["path"]] = fav_entry
                self._save_favorites()
            except Exception:
                pass
        if old_rel in self._group_assignments:
            self._group_assignments[self._rel(dst)] = self._group_assignments.pop(old_rel)
            self._save_group_state()

        self._append_log(f"[rename] {old_rel} -> {self._rel(dst)}\n")
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
            ds_dirs = self._selected_compare_datasets()
            if not ds_dirs:
                self.log_q.put("[error] no dataset selected\n")
                return

            # Pre-load model classes once; if missing, still allow running predict.sh.
            cls_a = self._model_classes(a)
            cls_b = self._model_classes(b)

            paired: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []

            for ds in ds_dirs:
                ds_classes = self._dataset_classes(ds)

                if ds_classes and cls_a and sorted(ds_classes) != sorted(cls_a):
                    self.log_q.put(
                        f"\n[skip] A({a.name}) incompatible with dataset {ds.name}: "
                        f"model classes={cls_a} dataset classes={ds_classes}\n"
                    )
                    continue
                if ds_classes and cls_b and sorted(ds_classes) != sorted(cls_b):
                    self.log_q.put(
                        f"\n[skip] B({b.name}) incompatible with dataset {ds.name}: "
                        f"model classes={cls_b} dataset classes={ds_classes}\n"
                    )
                    continue

                ra = self._run_predict_blocking(a, data_dir=ds, label=f"A({a.name})")
                rb = self._run_predict_blocking(b, data_dir=ds, label=f"B({b.name})")
                if ra and rb:
                    paired.append((ds, ra, rb))
                    self.log_q.put(self._compare_summary(ra, rb, model_a=a, model_b=b))
                else:
                    self.log_q.put(f"\n[skip] compare incomplete for dataset {ds.name} (see logs above)\n")

            if len(paired) > 1:
                self.log_q.put(self._compare_summary_multi(paired, model_a=a, model_b=b))

        threading.Thread(target=worker, daemon=True).start()

    def _compare_summary_multi(
        self,
        paired: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]],
        *,
        model_a: Path,
        model_b: Path,
    ) -> str:
        def metric(r: Dict[str, Any], k: str) -> Optional[float]:
            try:
                return float((r.get("metrics") or {}).get(k))
            except Exception:
                return None

        def seen(r: Dict[str, Any]) -> int:
            try:
                return int(r.get("seen_samples") or 0)
            except Exception:
                return 0

        def wavg(which: str, k: str) -> Optional[float]:
            num = 0.0
            den = 0.0
            for _ds, ra, rb in paired:
                r = ra if which == "A" else rb
                s = seen(r)
                v = metric(r, k)
                if s <= 0 or v is None:
                    continue
                num += v * float(s)
                den += float(s)
            return (num / den) if den > 0 else None

        split = str(self.var_split.get().strip() or "test")
        names = [ds.name for ds, _ra, _rb in paired]
        acc_a = wavg("A", "accuracy")
        acc_b = wavg("B", "accuracy")
        f1_a = wavg("A", "macro_f1")
        f1_b = wavg("B", "macro_f1")

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
        lines.append("\n[compare multi]")
        lines.append(f"  split: {split}")
        lines.append(f"  datasets ({len(names)}): {', '.join(names)}")
        if acc_a is not None and acc_b is not None:
            lines.append(f"  accuracy (wavg): A={acc_a:.4f}  B={acc_b:.4f}  delta(B-A)={acc_b-acc_a:+.4f}")
        if f1_a is not None and f1_b is not None:
            lines.append(f"  macro_f1 (wavg):  A={f1_a:.4f}  B={f1_b:.4f}  delta(B-A)={f1_b-f1_a:+.4f}")
        if winner == "TIE":
            lines.append("  winner: TIE (no measurable difference with current metrics)")
        else:
            lines.append(f"  winner: {winner}")
        lines.append(f"  A: {model_a.name}")
        lines.append(f"  B: {model_b.name}")
        lines.append("")
        return "\n".join(lines)

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

    def _predict_cmd(self, model_path: Path, *, data_dir: Path) -> Tuple[List[str], Path]:
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
            cmd, out_dir = self._predict_cmd(model_path, data_dir=self.state.dataset_dir)
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

    def _run_predict_blocking(self, model_path: Path, *, data_dir: Path, label: str) -> Optional[Dict[str, Any]]:
        """Run predict.sh sequentially from a background thread; returns report json if found."""
        try:
            cmd, out_dir = self._predict_cmd(model_path, data_dir=data_dir)
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
            class_mismatch = False
            for line in proc.stdout:
                if self.stop_evt.is_set():
                    break
                if "do not match dataset classes" in line:
                    class_mismatch = True
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
                if class_mismatch:
                    self.log_q.put(f"[skip] incompatible model/dataset (class mismatch)\n")
                    return None
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
        def add_dataset_prediction_dirs(ds: Path) -> None:
            # PredictionsTab default out-dir: <dataset>/predictions
            dirs.append(ds / "predictions")
            # WeightsTab default out-dir: <dataset>/predictions/weights_tab
            dirs.append(ds / "predictions" / "weights_tab")

        dirs: List[Path] = []
        # Common default from scripts/batch_predict.py
        dirs.append(self.sim_root / "outputs" / "models")
        dirs.append(self.sim_root / "outputs" / "models" / "history")
        # Also scan prediction report dirs across all datasets, so history does not "disappear"
        # when switching datasets in the UI.
        sim_data = self.sim_root / "outputs" / "sim_data"
        runs = sim_data / "runs"
        versions = sim_data / "versions"
        try:
            if runs.exists():
                for ds in runs.iterdir():
                    if ds.is_dir():
                        add_dataset_prediction_dirs(ds)
        except Exception:
            pass
        try:
            if versions.exists():
                for ds in versions.glob("*/*"):
                    if ds.is_dir():
                        add_dataset_prediction_dirs(ds)
        except Exception:
            pass
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
