"""Predictions Tab - Run batch predictions (predict.sh) and inspect results."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from .base_tab import BaseTab
from gui.state import UiState
from gui.components.chart_widgets import create_confusion_matrix_widget
from gui.components.overlay_renderer import draw_prediction_overlay, draw_two_stage_overlay
from gui.utils.tooltip import ToolTip
from gui.utils.settings_store import SettingsStore


class PredictionsTab(BaseTab):
    """Tab: Batch predictions + dataset-level evaluation, powered by predict.sh."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()
        self.log_q: queue.Queue[str] = queue.Queue()

        self._pred_rows: List[Dict[str, Any]] = []
        self._pred_row_by_id: Dict[str, Dict[str, Any]] = {}
        self._active_dataset_dir: Optional[Path] = None

        # Multi-dataset run support
        self.var_multi_datasets: tk.BooleanVar
        self._multi_dataset_paths: List[str] = []  # dataset dirs (string paths, typically sim_root-relative)
        self._multi_results: List[Dict[str, Any]] = []
        self._multi_result_by_key: Dict[str, Dict[str, Any]] = {}

        self._cm_canvas = None
        self._photo: Optional[ImageTk.PhotoImage] = None

        # UI components
        self.var_dataset: tk.StringVar
        self.var_model: tk.StringVar
        self.var_split: tk.StringVar
        self.var_device: tk.StringVar
        self.var_max_samples: tk.StringVar
        self.var_filter: tk.StringVar
        self.var_status: tk.StringVar
        self.var_report_path: tk.StringVar
        self.var_preds_path: tk.StringVar
        self.chk_save_preds: tk.BooleanVar

        self.var_profile_model: tk.StringVar
        self.var_multi_sel: tk.StringVar

        self.tree: ttk.Treeview
        self.tree_ds: ttk.Treeview
        self.ds_summary_frame: ttk.Frame
        self.canvas: tk.Canvas
        self.txt_metrics: tk.Text
        self.txt_logs: tk.Text
        self.btn_run: ttk.Button
        self.btn_stop: ttk.Button
        self.combo_dataset: ttk.Combobox
        self.combo_model: ttk.Combobox
        self.combo_profile_model: ttk.Combobox

        self._dataset_dirs: List[Path] = []
        self._dataset_by_label: Dict[str, Path] = {}

        self._ui_tick_id: Optional[str] = None

    def build_ui(self) -> None:
        # Use a grid so the start/stop buttons remain visible even on narrower windows.
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))
        top.columnconfigure(1, weight=1)  # dataset combo
        top.columnconfigure(4, weight=1)  # model combo

        ttk.Label(top, text="Dataset:").grid(row=0, column=0, sticky="w")
        self.var_dataset = tk.StringVar(value="")
        self.combo_dataset = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly")
        self.combo_dataset.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.combo_dataset.bind("<<ComboboxSelected>>", self._on_dataset_selected)
        ToolTip(self.combo_dataset, text_func=lambda: self.var_dataset.get())
        ttk.Button(top, text="↻", width=3, command=self._refresh_datasets).grid(row=0, column=2, sticky="w", padx=(0, 12))

        ttk.Label(top, text="Model:").grid(row=0, column=3, sticky="w")
        self.var_model = tk.StringVar(value="")
        self.combo_model = ttk.Combobox(top, textvariable=self.var_model, state="readonly")
        self.combo_model.grid(row=0, column=4, sticky="ew", padx=(6, 6))
        ttk.Button(top, text="↻", width=3, command=self._refresh_models).grid(row=0, column=5, sticky="w", padx=(0, 12))

        self.btn_run = ttk.Button(top, text="Start Predictions", command=self._run_predictions, width=16)
        self.btn_run.grid(row=0, column=6, sticky="e")
        self.btn_stop = ttk.Button(top, text="Stop", command=self._stop_predictions, state="disabled", width=8)
        self.btn_stop.grid(row=0, column=7, sticky="e", padx=(8, 0))

        opts = ttk.Frame(self.frame)
        opts.pack(fill="x", pady=(0, 8))

        ttk.Label(opts, text="Split:").pack(side="left")
        self.var_split = tk.StringVar(value="test")
        ttk.Combobox(opts, textvariable=self.var_split, values=["train", "val", "test", "all"], state="readonly", width=6).pack(
            side="left", padx=(6, 12)
        )

        ttk.Label(opts, text="Device:").pack(side="left")
        # "auto" lets batch_predict decide (cuda if available else cpu)
        self.var_device = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.var_device, values=["auto", "cpu", "cuda"], state="readonly", width=6).pack(
            side="left", padx=(6, 12)
        )

        ttk.Label(opts, text="Max:").pack(side="left")
        self.var_max_samples = tk.StringVar(value="")
        ttk.Entry(opts, textvariable=self.var_max_samples, width=6).pack(side="left", padx=(6, 12))

        self.chk_save_preds = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Save per-sample preds", variable=self.chk_save_preds).pack(side="left")

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=(12, 12), pady=2)
        ttk.Label(opts, text="Profile Model:").pack(side="left")
        self.var_profile_model = tk.StringVar(value="")
        self.combo_profile_model = ttk.Combobox(opts, textvariable=self.var_profile_model, state="readonly", width=30)
        self.combo_profile_model.pack(side="left", padx=(6, 4))
        ttk.Button(opts, text="↻", width=3, command=self._refresh_profile_models).pack(side="left")

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=(12, 12), pady=2)
        self.var_multi_datasets = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Multi datasets", variable=self.var_multi_datasets, command=self._on_multi_toggle).pack(side="left")
        ttk.Button(opts, text="Select…", command=self._open_multi_dataset_dialog).pack(side="left", padx=(8, 0))
        self.var_multi_sel = tk.StringVar(value="selected: 0")
        ttk.Label(opts, textvariable=self.var_multi_sel).pack(side="left", padx=(8, 0))

        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(0, 8))
        self.var_status = tk.StringVar(value="status: idle")
        ttk.Label(status, textvariable=self.var_status).pack(side="left")

        paths = ttk.Frame(self.frame)
        paths.pack(fill="x", pady=(0, 10))
        self.var_report_path = tk.StringVar(value="report: -")
        self.var_preds_path = tk.StringVar(value="preds: -")
        ttk.Label(paths, textvariable=self.var_report_path).pack(side="left", padx=(0, 14))
        ttk.Label(paths, textvariable=self.var_preds_path).pack(side="left")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Left: prediction table
        left = ttk.Frame(main, padding=5)
        main.add(left, weight=2)

        # Multi-run dataset summary (hidden in single-dataset mode)
        self.ds_summary_frame = ttk.Frame(left)
        ttk.Label(self.ds_summary_frame, text="Dataset Results").pack(anchor="w")
        ds_tree_frame = ttk.Frame(self.ds_summary_frame)
        ds_tree_frame.pack(fill="x", expand=False, pady=(4, 0))
        self.tree_ds = ttk.Treeview(
            ds_tree_frame,
            columns=("Dataset", "Acc", "F1", "Seen", "ProfAcc"),
            show="headings",
            height=6,
        )
        self.tree_ds.heading("Dataset", text="Dataset")
        self.tree_ds.heading("Acc", text="Acc")
        self.tree_ds.heading("F1", text="Macro F1")
        self.tree_ds.heading("Seen", text="Seen")
        self.tree_ds.heading("ProfAcc", text="Prof Acc")
        self.tree_ds.column("Dataset", width=210, anchor="w")
        self.tree_ds.column("Acc", width=70, anchor="e")
        self.tree_ds.column("F1", width=80, anchor="e")
        self.tree_ds.column("Seen", width=60, anchor="e")
        self.tree_ds.column("ProfAcc", width=80, anchor="e")
        ds_scroll = ttk.Scrollbar(ds_tree_frame, orient="vertical", command=self.tree_ds.yview)
        self.tree_ds.configure(yscrollcommand=ds_scroll.set)
        self.tree_ds.pack(side="left", fill="x", expand=True)
        ds_scroll.pack(side="right", fill="y")
        self.tree_ds.bind("<<TreeviewSelect>>", self._on_tree_ds_select)
        self.ds_summary_frame.pack_forget()

        filters = ttk.Frame(left)
        filters.pack(fill="x", pady=(0, 6))
        ttk.Label(filters, text="Show:").pack(side="left")
        self.var_filter = tk.StringVar(value="all")
        ttk.Radiobutton(filters, text="All", variable=self.var_filter, value="all", command=self._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Wrong", variable=self.var_filter, value="wrong", command=self._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Correct", variable=self.var_filter, value="correct", command=self._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Prof Wrong", variable=self.var_filter, value="prof_wrong", command=self._refresh_tree).pack(side="left", padx=(6, 0))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("GT", "Pred", "Conf", "OK", "GT Prof", "Pred Prof", "Prof"),
            show="tree headings",
        )
        self.tree.heading("#0", text="Sample ID")
        self.tree.heading("GT", text="GT")
        self.tree.heading("Pred", text="Pred")
        self.tree.heading("Conf", text="Conf")
        self.tree.heading("OK", text="OK")
        self.tree.heading("GT Prof", text="GT Prof")
        self.tree.heading("Pred Prof", text="Pred Prof")
        self.tree.heading("Prof", text="Prof")
        self.tree.column("#0", width=200)
        self.tree.column("GT", width=80, anchor="center")
        self.tree.column("Pred", width=90, anchor="center")
        self.tree.column("Conf", width=60, anchor="e")
        self.tree.column("OK", width=40, anchor="center")
        self.tree.column("GT Prof", width=120, anchor="center")
        self.tree.column("Pred Prof", width=120, anchor="center")
        self.tree.column("Prof", width=40, anchor="center")

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right: metrics + confusion matrix + image viewer + logs
        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)

        top_right = ttk.Panedwindow(right, orient="vertical")
        top_right.pack(fill="both", expand=True)

        metrics_box = ttk.Frame(top_right, padding=5)
        top_right.add(metrics_box, weight=2)

        ttk.Label(metrics_box, text="Metrics / Report").pack(anchor="w")
        self.txt_metrics = tk.Text(metrics_box, height=10, wrap="word")
        self.txt_metrics.pack(fill="x", expand=False, pady=(6, 6))
        self.txt_metrics.configure(state="disabled")

        cm_box = ttk.Frame(metrics_box)
        cm_box.pack(fill="both", expand=True)
        ttk.Label(cm_box, text="Confusion Matrix").pack(anchor="w")
        self._cm_parent = ttk.Frame(cm_box)
        self._cm_parent.pack(fill="both", expand=True, pady=(6, 0))

        viewer_box = ttk.Frame(top_right, padding=5)
        top_right.add(viewer_box, weight=2)
        ttk.Label(viewer_box, text="Image Viewer").pack(anchor="w")
        self.canvas = tk.Canvas(viewer_box, bg="gray20", width=520, height=520)
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))

        logs_box = ttk.Frame(top_right, padding=5)
        top_right.add(logs_box, weight=1)
        ttk.Label(logs_box, text="Logs").pack(anchor="w")
        self.txt_logs = tk.Text(logs_box, height=10, wrap="none")
        self.txt_logs.pack(fill="both", expand=True, pady=(6, 0))
        self.txt_logs.configure(state="disabled")

        # Load initial dataset if already selected
        self.on_dataset_changed()
        self._refresh_datasets()
        self._refresh_models()
        self._refresh_profile_models()
        self._load_persisted_settings()
        # Persisted dataset selection is loaded after the initial refresh; sync state now.
        try:
            self._on_dataset_selected()
        except Exception:
            pass
        self._wire_settings_autosave()
        self._update_multi_sel_label()
        self._apply_multi_mode_ui()
        self._tick_ui()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir:
            return
        # Keep combobox showing the dataset name, not the full path.
        # Prefer matching the current dropdown label if we already have a mapping.
        try:
            if self._dataset_by_label:
                for label, p in self._dataset_by_label.items():
                    if p == self.state.dataset_dir:
                        self.var_dataset.set(label)
                        break
                else:
                    self.var_dataset.set(self.state.dataset_dir.name)
            else:
                self.var_dataset.set(self.state.dataset_dir.name)
        except Exception:
            self.var_dataset.set(self.state.dataset_dir.name)
        # Refresh dropdowns to include the current dataset/model.
        try:
            self._refresh_datasets()
            self._refresh_models()
        except Exception:
            pass

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None:
            return
        for key, var in [
            ("pred.dataset_selection", self.var_dataset),
            ("pred.model_selection", self.var_model),
            ("pred.split", self.var_split),
            ("pred.device", self.var_device),
            ("pred.max_samples", self.var_max_samples),
            ("pred.profile_model", self.var_profile_model),
        ]:
            v = st.get(key)
            if v is None:
                continue
            try:
                var.set(v)
            except Exception:
                pass
        try:
            self.chk_save_preds.set(bool(st.get("pred.save_preds", True)))
        except Exception:
            pass
        try:
            self.var_multi_datasets.set(bool(st.get("pred.multi_datasets", False)))
        except Exception:
            pass
        try:
            raw = st.get("pred.multi_dataset_paths_json")
            if isinstance(raw, str) and raw.strip():
                obj = json.loads(raw)
                if isinstance(obj, list):
                    self._multi_dataset_paths = [str(x) for x in obj if str(x).strip()]
        except Exception:
            self._multi_dataset_paths = []

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

        bind(self.var_dataset, "pred.dataset_selection")
        bind(self.var_model, "pred.model_selection")
        bind(self.var_split, "pred.split")
        bind(self.var_device, "pred.device")
        bind(self.var_max_samples, "pred.max_samples")
        bind(self.var_profile_model, "pred.profile_model")
        bind(self.var_multi_datasets, "pred.multi_datasets")

        def on_chk() -> None:
            try:
                st.set("pred.save_preds", bool(self.chk_save_preds.get()))
                st.schedule_save(self.frame)
            except Exception:
                pass

        try:
            self.chk_save_preds.trace_add("write", lambda *_a: on_chk())
        except Exception:
            pass

    def _persist_multi_dataset_paths(self) -> None:
        st = self._store()
        if st is None:
            return
        try:
            st.set("pred.multi_dataset_paths_json", json.dumps(self._multi_dataset_paths, ensure_ascii=True))
            st.schedule_save(self.frame)
        except Exception:
            pass

    def _update_multi_sel_label(self) -> None:
        # Keep it stable even if dataset list hasn't been refreshed yet.
        n = len([p for p in self._multi_dataset_paths if str(p).strip()])
        try:
            self.var_multi_sel.set(f"selected: {n}")
        except Exception:
            pass

    def _on_multi_toggle(self) -> None:
        self._apply_multi_mode_ui()
        self._update_multi_sel_label()

    def _apply_multi_mode_ui(self) -> None:
        try:
            if bool(self.var_multi_datasets.get()):
                if self.ds_summary_frame.winfo_ismapped() == 0:
                    self.ds_summary_frame.pack(fill="x", pady=(0, 8))
            else:
                if self.ds_summary_frame.winfo_ismapped() != 0:
                    self.ds_summary_frame.pack_forget()
                # Clear any stale summary.
                try:
                    self.tree_ds.delete(*self.tree_ds.get_children())
                except Exception:
                    pass
        except Exception:
            pass

    def _open_multi_dataset_dialog(self) -> None:
        # Ensure we have up-to-date dataset list.
        try:
            self._refresh_datasets()
        except Exception:
            pass

        dlg = tk.Toplevel(self.frame)
        dlg.title("Select Datasets")
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select one or more datasets (Ctrl/Shift):").pack(anchor="w")

        list_frame = ttk.Frame(frm)
        list_frame.pack(fill="both", expand=True, pady=(8, 8))
        lb = tk.Listbox(list_frame, selectmode="extended", height=16)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        labels = list(self._dataset_by_label.keys())
        for x in labels:
            lb.insert("end", x)

        # Preselect currently stored paths.
        want = set()
        for p in self._multi_dataset_paths:
            rp = str(p).strip()
            if not rp:
                continue
            want.add(rp)
        for idx, lab in enumerate(labels):
            p = self._dataset_by_label.get(lab)
            if not p:
                continue
            try:
                rel = str(p.resolve().relative_to(self.sim_root.resolve()))
            except Exception:
                rel = str(p.resolve())
            if rel in want:
                lb.selection_set(idx)

        btns = ttk.Frame(frm)
        btns.pack(fill="x")

        def on_ok() -> None:
            sel = list(lb.curselection())
            picked: List[str] = []
            for i in sel:
                try:
                    lab = labels[int(i)]
                except Exception:
                    continue
                p = self._dataset_by_label.get(lab)
                if not p:
                    continue
                try:
                    picked.append(str(p.resolve().relative_to(self.sim_root.resolve())))
                except Exception:
                    picked.append(str(p.resolve()))
            self._multi_dataset_paths = picked
            self._persist_multi_dataset_paths()
            self._update_multi_sel_label()
            try:
                dlg.destroy()
            except Exception:
                pass

        ttk.Button(btns, text="OK", command=on_ok).pack(side="right")
        ttk.Button(btns, text="Cancel", command=lambda: dlg.destroy()).pack(side="right", padx=(0, 8))

        try:
            dlg.geometry("560x420")
        except Exception:
            pass

    def _active_data_dir(self) -> Optional[Path]:
        return self._active_dataset_dir or self.state.dataset_dir

    def _append_log(self, s: str) -> None:
        self.txt_logs.configure(state="normal")
        self.txt_logs.insert("end", s)
        self.txt_logs.see("end")
        self.txt_logs.configure(state="disabled")

    def _tick_ui(self) -> None:
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)

            # Parse out report/preds paths from batch_predict output for convenience.
            if "Report saved to:" in line:
                p = line.split("Report saved to:", 1)[-1].strip()
                self.var_report_path.set(f"report: {p}")
            if "Predictions saved to:" in line:
                p = line.split("Predictions saved to:", 1)[-1].strip()
                self.var_preds_path.set(f"preds: {p}")

        self._ui_tick_id = self.frame.after(150, self._tick_ui)

    def _stop_predictions(self) -> None:
        self.stop_evt.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.var_status.set("status: stopping...")

    def _resolve_dataset_path(self, s: str) -> Path:
        p = Path(s)
        if p.is_absolute():
            return p
        return (self.sim_root / p).resolve()

    def _selected_dataset_paths(self) -> List[Path]:
        """Return dataset dirs to run on (single or multi)."""
        if hasattr(self, "var_multi_datasets") and bool(self.var_multi_datasets.get()):
            out: List[Path] = []
            for s in self._multi_dataset_paths:
                s = str(s).strip()
                if not s:
                    continue
                out.append(self._resolve_dataset_path(s))
            # De-dup while preserving order
            seen = set()
            uniq: List[Path] = []
            for p in out:
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(p)
            return uniq

        ds = self.state.dataset_dir
        return [ds] if ds else []

    def _run_predictions(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Info", "Predictions already running.")
            return

        model_s = self.var_model.get().strip()
        model_path = self._resolve_model_path(model_s)
        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found:\n{model_path}")
            return

        dataset_dirs = self._selected_dataset_paths()
        if not dataset_dirs:
            messagebox.showinfo("Info", "No dataset selected.")
            return

        split = self.var_split.get().strip() or "test"
        device = self.var_device.get().strip() or "auto"
        max_samples_s = self.var_max_samples.get().strip()
        save_preds = bool(self.chk_save_preds.get())
        started_s = time.time()
        if max_samples_s:
            try:
                int(max_samples_s)
            except Exception:
                messagebox.showerror("Error", "Max must be empty or an integer.")
                return

        profile_model_s = self.var_profile_model.get().strip()
        pm_path: Optional[Path] = None
        if profile_model_s:
            pm_path = self._resolve_model_path(profile_model_s)
            if not pm_path.exists():
                messagebox.showerror("Error", f"Profile model not found:\n{pm_path}")
                return

        self.var_status.set("status: running...")
        self.var_report_path.set("report: -")
        self.var_preds_path.set("preds: -")
        self._set_metrics_text("Running batch prediction...\n")
        self._clear_confusion_matrix()
        self._pred_rows = []
        self._pred_row_by_id = {}
        self._refresh_tree()
        self._multi_results = []
        self._multi_result_by_key = {}
        try:
            self.tree_ds.delete(*self.tree_ds.get_children())
        except Exception:
            pass

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_evt.clear()

        def worker() -> None:
            last_report_path: Optional[Path] = None
            last_preds_path: Optional[Path] = None
            last_dataset_dir: Optional[Path] = None
            try:
                for idx, data_dir in enumerate(dataset_dirs, 1):
                    if self.stop_evt.is_set():
                        break

                    if not (data_dir / "meta.jsonl").exists() or not (data_dir / "labels.jsonl").exists():
                        self.log_q.put(f"[error] dataset missing meta.jsonl/labels.jsonl: {data_dir}\n")
                        continue

                    out_dir = data_dir / "predictions"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    cmd: List[str] = ["bash", str(self.sim_root / "predict.sh"),
                                      "--model", str(model_path),
                                      "--data", str(data_dir),
                                      "--split", split,
                                      "--out-dir", str(out_dir)]
                    if device != "auto":
                        cmd.extend(["--device", device])
                    if max_samples_s:
                        cmd.extend(["--max-samples", max_samples_s])
                    if save_preds:
                        cmd.append("--save-preds")
                    if pm_path is not None:
                        cmd.extend(["--profile-model", str(pm_path)])

                    self.log_q.put(f"\n[dataset {idx}/{len(dataset_dirs)}] {data_dir}\n")
                    self.log_q.put("[cmd] " + " ".join(cmd) + "\n")

                    report_path: Optional[Path] = None
                    preds_path: Optional[Path] = None
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
                        if "Predictions saved to:" in line:
                            try:
                                preds_path = Path(line.split("Predictions saved to:", 1)[-1].strip())
                            except Exception:
                                preds_path = None

                    rc = self.proc.wait()
                    if self.stop_evt.is_set():
                        self.log_q.put("\n[stopped]\n")
                        break
                    if rc != 0:
                        self.log_q.put(f"\n[error] predict.sh exited with code {rc}\n")
                        continue

                    # Best-effort: if we didn't parse paths, pick newest from out_dir, but only
                    # from this run (mtime >= started_s) to avoid accidentally loading stale results.
                    try:
                        if report_path is None:
                            cand = [p for p in out_dir.glob("batch_report_*.json") if p.stat().st_mtime >= started_s - 0.5]
                            cand = sorted(cand, key=lambda p: p.stat().st_mtime)
                            report_path = cand[-1] if cand else None
                        if save_preds and preds_path is None:
                            cand = [p for p in out_dir.glob("batch_preds_*.jsonl") if p.stat().st_mtime >= started_s - 0.5]
                            cand = sorted(cand, key=lambda p: p.stat().st_mtime)
                            preds_path = cand[-1] if cand else None
                    except Exception:
                        pass

                    # Collect multi-run results
                    try:
                        key = str(data_dir.resolve())
                        label = data_dir.name
                        result = {
                            "key": key,
                            "label": label,
                            "dataset_dir": data_dir,
                            "report_path": report_path,
                            "preds_path": preds_path,
                        }
                        self._multi_results.append(result)
                        self._multi_result_by_key[key] = result
                    except Exception:
                        pass

                    last_report_path = report_path
                    last_preds_path = preds_path
                    last_dataset_dir = data_dir
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run predictions: {e}\n")
            finally:
                def apply() -> None:
                    self.btn_run.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    if self.stop_evt.is_set():
                        self.var_status.set("status: stopped")
                    else:
                        self.var_status.set("status: done")
                    if bool(self.var_multi_datasets.get()):
                        if last_dataset_dir:
                            self._active_dataset_dir = last_dataset_dir
                        self._render_multi_summary()
                    else:
                        if dataset_dirs:
                            self._active_dataset_dir = dataset_dirs[0]
                        if last_report_path and last_report_path.exists():
                            self._load_report(last_report_path)
                        if last_preds_path and last_preds_path.exists():
                            self._load_preds(last_preds_path)

                try:
                    self.frame.after(0, apply)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _render_multi_summary(self) -> None:
        """Populate the dataset-results table from collected report JSONs."""
        try:
            self._apply_multi_mode_ui()
        except Exception:
            pass

        try:
            self.tree_ds.delete(*self.tree_ds.get_children())
        except Exception:
            return

        rows: List[Tuple[str, str, str, str, str, str]] = []
        for res in self._multi_results:
            key = str(res.get("key") or "")
            label = str(res.get("label") or "")
            report_path = res.get("report_path")
            preds_path = res.get("preds_path")

            acc_s = ""
            f1_s = ""
            seen_s = ""
            prof_acc_s = ""

            if isinstance(report_path, Path) and report_path.exists():
                try:
                    obj = json.loads(report_path.read_text(encoding="utf-8"))
                    metrics = obj.get("metrics") or {}
                    acc = metrics.get("accuracy")
                    f1 = metrics.get("macro_f1")
                    seen = obj.get("seen_samples")
                    prof_acc = obj.get("profile_accuracy")
                    if acc is not None:
                        acc_s = f"{float(acc):.4f}"
                    if f1 is not None:
                        f1_s = f"{float(f1):.4f}"
                    if seen is not None:
                        seen_s = str(int(seen))
                    if prof_acc is not None:
                        prof_acc_s = f"{float(prof_acc):.4f}"
                except Exception:
                    pass

            # Keep paths around for selection handler.
            res["report_path"] = report_path
            res["preds_path"] = preds_path

            if not key:
                # fallback iid must be unique; use label + index
                key = f"{label}:{len(rows)}"
                res["key"] = key
                self._multi_result_by_key[key] = res
            rows.append((key, label, acc_s, f1_s, seen_s, prof_acc_s))

        for key, label, acc_s, f1_s, seen_s, prof_acc_s in rows:
            try:
                self.tree_ds.insert("", "end", iid=key, values=(label, acc_s, f1_s, seen_s, prof_acc_s))
            except Exception:
                # iid collisions should be rare; fall back to auto iid
                self.tree_ds.insert("", "end", values=(label, acc_s, f1_s, seen_s, prof_acc_s))

        # Autoselect first dataset to show details.
        try:
            kids = list(self.tree_ds.get_children())
            if kids:
                self.tree_ds.selection_set(kids[0])
                self.tree_ds.see(kids[0])
                self._on_tree_ds_select()
        except Exception:
            pass

    def _on_tree_ds_select(self, _evt: Optional[object] = None) -> None:
        sel = self.tree_ds.selection()
        if not sel:
            return
        key = sel[0]
        res = self._multi_result_by_key.get(key)
        if not res:
            return
        data_dir = res.get("dataset_dir")
        if isinstance(data_dir, Path):
            self._active_dataset_dir = data_dir

        report_path = res.get("report_path")
        preds_path = res.get("preds_path")
        if isinstance(report_path, Path) and report_path.exists():
            self._load_report(report_path)
        if isinstance(preds_path, Path) and preds_path.exists():
            self._load_preds(preds_path)
        else:
            # No per-sample preds saved; clear table.
            try:
                self._pred_rows = []
                self._pred_row_by_id = {}
                self._refresh_tree()
            except Exception:
                pass

    def _datasets_base(self) -> Path:
        return self.sim_root / "outputs" / "sim_data"

    def _refresh_datasets(self) -> None:
        sim_data = self._datasets_base()
        runs = sim_data / "runs"
        versions = sim_data / "versions"
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)

        cand: List[Path] = []
        cand.extend([p for p in runs.iterdir() if p.is_dir()])
        cand.extend([p for p in versions.glob("*/*") if p.is_dir()])

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
        self._dataset_dirs = cand

        def display(p: Path) -> str:
            try:
                rp = p.resolve()
                if str(rp).startswith(str(runs.resolve()) + os.sep):
                    return rp.name
                if str(rp).startswith(str(versions.resolve()) + os.sep):
                    return f"{rp.parent.name}:{rp.name}"
            except Exception:
                pass
            return p.name

        self._dataset_by_label = {}
        labels: List[str] = []
        seen: Dict[str, int] = {}
        for p in cand:
            base = display(p)
            n = seen.get(base, 0) + 1
            seen[base] = n
            label = base if n == 1 else f"{base} ({n})"
            labels.append(label)
            self._dataset_by_label[label] = p

        self.combo_dataset["values"] = labels

        # Keep current selection if still valid.
        cur = self.var_dataset.get().strip()
        if cur and cur in self._dataset_by_label:
            # Ensure state follows the UI selection (important after loading persisted settings).
            self.state.dataset_dir = self._dataset_by_label[cur]
            return

        # Prefer current state.dataset_dir
        if self.state.dataset_dir:
            want = display(self.state.dataset_dir)
            # Might have been disambiguated.
            for label, p in self._dataset_by_label.items():
                if p == self.state.dataset_dir:
                    self.var_dataset.set(label)
                    return
                if label == want:
                    self.var_dataset.set(label)
                    return

        if labels:
            self.var_dataset.set(labels[0])
            self._on_dataset_selected()

    def _selected_dataset_dir(self) -> Optional[Path]:
        disp = self.var_dataset.get().strip()
        if not disp:
            return None
        p = self._dataset_by_label.get(disp)
        return p if p and p.exists() else None

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        self.state.dataset_dir = ds
        self._active_dataset_dir = ds
        # Notify other tabs (MonitorAppTabbed listens to this).
        try:
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception:
            pass
        # Update model choices based on dataset.
        self._refresh_models()

    def _model_search_paths(self) -> List[Path]:
        root = self.sim_root / "outputs" / "models"
        return [
            root,
            root / "versions",
            root / "imports",
        ]

    def _refresh_models(self) -> None:
        # List available checkpoints including snapshots and bundle directories.
        cand: List[Path] = []
        for root in self._model_search_paths():
            if not root.exists():
                continue
            if root.name == "versions":
                cand.extend(sorted(root.glob("**/*.pt")))
            else:
                cand.extend(sorted(root.glob("*.pt")))
                # Bundles can exist as <name>.bundle dirs or under outputs/models/bundles/<name>/.
                cand.extend([p for p in sorted(root.glob("*.bundle")) if p.is_dir()])
                bundles_subdir = root / "bundles"
                if bundles_subdir.exists():
                    cand.extend([p for p in sorted(bundles_subdir.iterdir()) if p.is_dir()])

        # Dedup + sort
        uniq: Dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file() or p.is_dir():
                    uniq[str(p.resolve())] = p
            except Exception:
                continue
        paths = list(uniq.values())
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

        values: List[str] = []
        for p in paths:
            try:
                values.append(str(p.resolve().relative_to(self.sim_root.resolve())))
            except Exception:
                values.append(str(p))

        self.combo_model["values"] = values

        # Prefer dataset-specific active model if present.
        default_model: Optional[Path] = None
        if self.state.dataset_dir:
            dm = self.sim_root / "outputs" / "models" / f"{self.state.dataset_dir.name}.pt"
            if dm.exists():
                default_model = dm

        cur = self.var_model.get().strip()
        if cur and cur in values:
            return
        if default_model is not None:
            try:
                rel = str(default_model.resolve().relative_to(self.sim_root.resolve()))
            except Exception:
                rel = str(default_model)
            if rel in values:
                self.var_model.set(rel)
                return
        if values:
            self.var_model.set(values[0])

    def _refresh_profile_models(self) -> None:
        """Populate the profile model combobox with available profile classifier checkpoints."""
        root = self.sim_root / "outputs" / "models"
        cand: List[Path] = []
        if root.exists():
            cand.extend(sorted(root.glob("profile_classifier_*.pt")))
            # Also check sub-directories
            cand.extend(sorted(root.glob("**/profile_classifier_*.pt")))
        # Dedup
        seen: Dict[str, Path] = {}
        for p in cand:
            try:
                seen[str(p.resolve())] = p
            except Exception:
                continue
        paths = list(seen.values())
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

        values = [""]  # empty = no profile model (default)
        for p in paths:
            try:
                values.append(str(p.resolve().relative_to(self.sim_root.resolve())))
            except Exception:
                values.append(str(p))
        self.combo_profile_model["values"] = values

        cur = self.var_profile_model.get().strip()
        if cur and cur in values:
            return
        # Keep empty (disabled) by default
        if not cur:
            self.var_profile_model.set("")

    def _resolve_model_path(self, s: str) -> Path:
        p = Path(s)
        if p.is_absolute():
            return p
        # Treat as workspace-relative under sim_root.
        return (self.sim_root / p).resolve()

    def _set_metrics_text(self, s: str) -> None:
        self.txt_metrics.configure(state="normal")
        self.txt_metrics.delete("1.0", "end")
        self.txt_metrics.insert("1.0", s)
        self.txt_metrics.configure(state="disabled")

    def _clear_confusion_matrix(self) -> None:
        for w in list(self._cm_parent.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._cm_canvas = None

    def _load_report(self, report_path: Path) -> None:
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            self._set_metrics_text(f"Failed to load report:\n{e}")
            return

        metrics = obj.get("metrics") or {}
        cm = metrics.get("confusion_matrix")
        class_names = list((metrics.get("per_class") or {}).keys())

        # Summary text (keep it short; details in report JSON on disk).
        lines = []
        lines.append(f"Report: {report_path.name}")
        lines.append(f"Model:  {Path(obj.get('model_path', '-')).name}")
        lines.append(f"Split:  {obj.get('split', '-')}")
        lines.append(f"Seen:   {obj.get('seen_samples', '-')}")
        if "img_per_s" in obj:
            lines.append(f"Speed:  {float(obj.get('img_per_s') or 0.0):.1f} img/s")
        if "accuracy" in metrics:
            lines.append(f"\nAccuracy: {float(metrics.get('accuracy') or 0.0):.4f}")
        if "macro_f1" in metrics:
            lines.append(f"Macro F1:  {float(metrics.get('macro_f1') or 0.0):.4f}")
        cfn = metrics.get("critical_fn_rates") or {}
        if cfn:
            lines.append("\nCritical FN rates:")
            for k in sorted(cfn.keys()):
                lines.append(f"  {k}: {float(cfn.get(k) or 0.0):.4f}")

        self._set_metrics_text("\n".join(lines) + "\n")

        self._clear_confusion_matrix()
        try:
            if isinstance(cm, list) and cm and class_names:
                cm_np = np.array(cm, dtype=np.int64)
                self._cm_canvas = create_confusion_matrix_widget(self._cm_parent, cm_np, class_names)
                self._cm_canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            self._append_log(f"[warn] failed to render confusion matrix: {e}\n")

    def _load_preds(self, preds_path: Path) -> None:
        rows: List[Dict[str, Any]] = []
        try:
            with open(preds_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load predictions:\n{e}")
            return

        # Backfill GT profile from dataset labels.jsonl (so the table can show GT Prof even when
        # batch_predict wasn't run with --profile-model and thus didn't emit gt_profile fields).
        try:
            self._enrich_rows_with_gt_profile(rows)
        except Exception as e:
            # Non-fatal: keep rendering the table even if enrichment fails.
            self._append_log(f"[warn] failed to enrich rows with GT profile: {e}\n")

        self._pred_rows = rows
        self._pred_row_by_id = {str(r.get("id")): r for r in rows if r.get("id")}
        self._refresh_tree()

    def _load_gt_profile_map_from_labels(self) -> Dict[str, str]:
        """Load sample-id -> profile_id map from the current dataset's labels.jsonl (v2)."""
        data_dir = self._active_data_dir()
        if not data_dir:
            return {}
        labels_path = data_dir / "labels.jsonl"
        if not labels_path.exists():
            return {}

        gt_profile_map: Dict[str, str] = {}
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = str(obj.get("id") or "")
                pid = str(obj.get("profile_id") or "")
                if sid and pid:
                    gt_profile_map[sid] = pid
        return gt_profile_map

    def _enrich_rows_with_gt_profile(self, rows: List[Dict[str, Any]]) -> None:
        gt_profile_map = self._load_gt_profile_map_from_labels()
        if not gt_profile_map:
            return

        for r in rows:
            sid = str(r.get("id") or "")
            if not sid:
                continue

            if not r.get("gt_profile"):
                pid = gt_profile_map.get(sid, "")
                if pid:
                    r["gt_profile"] = pid

            # If a profile prediction exists, compute correctness if absent.
            if r.get("pred_profile") and r.get("gt_profile") and r.get("profile_correct") is None:
                r["profile_correct"] = bool(str(r.get("pred_profile")) == str(r.get("gt_profile")))

    def _refresh_tree(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())

        mode = self.var_filter.get().strip()
        for r in self._pred_rows:
            sid = str(r.get("id") or "")
            if not sid:
                continue
            gt = str(r.get("gt") or "?")
            pred = str(r.get("pred") or "?")
            correct = bool(r.get("correct"))
            if mode == "wrong" and correct:
                continue
            if mode == "correct" and not correct:
                continue

            # Profile data (may be absent)
            gt_prof = str(r.get("gt_profile") or "")
            pred_prof = str(r.get("pred_profile") or "")
            profile_correct = r.get("profile_correct")

            if mode == "prof_wrong":
                # Only show rows where profile was wrong (skip rows without profile data)
                if profile_correct is None or profile_correct:
                    continue

            conf = self._top1_confidence(r)
            ok = "✓" if correct else "✗"
            prof_mark = ""
            if profile_correct is not None:
                prof_mark = "✓" if profile_correct else "✗"
            self.tree.insert("", "end", iid=sid, text=sid,
                             values=(gt, pred, f"{conf:.1%}", ok, gt_prof, pred_prof, prof_mark))

    def _top1_confidence(self, row: Dict[str, Any]) -> float:
        try:
            topk = row.get("topk") or []
            if not topk:
                return 0.0
            # Prefer the probability of the predicted class if present.
            pred = row.get("pred")
            for t in topk:
                if t.get("class") == pred:
                    return float(t.get("prob") or 0.0)
            return float(topk[0].get("prob") or 0.0)
        except Exception:
            return 0.0

    def _on_tree_select(self, _evt: Optional[object] = None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        sid = sel[0]
        row = self._pred_row_by_id.get(sid)
        data_dir = self._active_data_dir()
        if not row or not data_dir:
            return

        image_rel = row.get("image_path")
        if not image_rel:
            return
        image_path = data_dir / str(image_rel)
        if not image_path.exists():
            messagebox.showerror("Error", f"Image not found:\n{image_path}")
            return

        gt = str(row.get("gt") or "?")
        pred = str(row.get("pred") or "?")
        conf = self._top1_confidence(row)

        gt_profile = str(row.get("gt_profile") or "")
        pred_profile = str(row.get("pred_profile") or "")
        profile_conf = float(row.get("profile_confidence") or 0.0)

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("failed to read image")
            if gt_profile and pred_profile:
                img_overlay = draw_two_stage_overlay(
                    img, gt, pred, conf, gt_profile, pred_profile, profile_conf
                )
            else:
                img_overlay = draw_prediction_overlay(img, gt, pred, conf)
            img_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            pil.thumbnail((520, 520))
            self._photo = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            cx = max(1, int(self.canvas.winfo_width() / 2))
            cy = max(1, int(self.canvas.winfo_height() / 2))
            self.canvas.create_image(cx, cy, image=self._photo, anchor="center")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\n{e}")
