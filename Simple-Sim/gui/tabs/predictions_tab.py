"""Predictions Tab - Run batch predictions (predict.sh) and inspect results."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
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
from gui.components.overlay_renderer import draw_prediction_overlay
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

        self.tree: ttk.Treeview
        self.canvas: tk.Canvas
        self.txt_metrics: tk.Text
        self.txt_logs: tk.Text
        self.btn_run: ttk.Button
        self.btn_stop: ttk.Button
        self.combo_dataset: ttk.Combobox
        self.combo_model: ttk.Combobox

        self._dataset_dirs: List[Path] = []

        self._ui_tick_id: Optional[str] = None

    def build_ui(self) -> None:
        # Use a grid so the start/stop buttons remain visible even on narrower windows.
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="Dataset:").grid(row=0, column=0, sticky="w")
        self.var_dataset = tk.StringVar(value="")
        self.combo_dataset = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly")
        self.combo_dataset.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.combo_dataset.bind("<<ComboboxSelected>>", self._on_dataset_selected)
        ToolTip(self.combo_dataset, text_func=lambda: self.var_dataset.get())
        ttk.Button(top, text="↻", width=3, command=self._refresh_datasets).grid(row=0, column=2, sticky="w", padx=(0, 12))

        ttk.Label(top, text="Model:").grid(row=0, column=2, sticky="w")
        self.var_model = tk.StringVar(value="")
        self.combo_model = ttk.Combobox(top, textvariable=self.var_model, state="readonly")
        self.combo_model.grid(row=0, column=3, sticky="ew", padx=(6, 6))
        ttk.Button(top, text="↻", width=3, command=self._refresh_models).grid(row=0, column=4, sticky="w", padx=(0, 12))

        self.btn_run = ttk.Button(top, text="Start Predictions", command=self._run_predictions, width=16)
        self.btn_run.grid(row=0, column=5, sticky="e")
        self.btn_stop = ttk.Button(top, text="Stop", command=self._stop_predictions, state="disabled", width=8)
        self.btn_stop.grid(row=0, column=6, sticky="e", padx=(8, 0))

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

        filters = ttk.Frame(left)
        filters.pack(fill="x", pady=(0, 6))
        ttk.Label(filters, text="Show:").pack(side="left")
        self.var_filter = tk.StringVar(value="all")
        ttk.Radiobutton(filters, text="All", variable=self.var_filter, value="all", command=self._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Wrong", variable=self.var_filter, value="wrong", command=self._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Correct", variable=self.var_filter, value="correct", command=self._refresh_tree).pack(side="left", padx=(6, 0))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("GT", "Pred", "Conf", "OK"),
            show="tree headings",
        )
        self.tree.heading("#0", text="Sample ID")
        self.tree.heading("GT", text="GT")
        self.tree.heading("Pred", text="Pred")
        self.tree.heading("Conf", text="Conf")
        self.tree.heading("OK", text="OK")
        self.tree.column("#0", width=260)
        self.tree.column("GT", width=90, anchor="center")
        self.tree.column("Pred", width=100, anchor="center")
        self.tree.column("Conf", width=70, anchor="e")
        self.tree.column("OK", width=50, anchor="center")

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
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self._tick_ui()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir:
            return
        # Keep combobox showing the dataset name, not the full path.
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

    def _run_predictions(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Info", "Predictions already running.")
            return
        if not self.state.dataset_dir:
            messagebox.showinfo("Info", "No dataset selected.")
            return

        model_s = self.var_model.get().strip()
        model_path = self._resolve_model_path(model_s)
        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found:\n{model_path}")
            return

        data_dir = self.state.dataset_dir
        if not (data_dir / "meta.jsonl").exists() or not (data_dir / "labels.jsonl").exists():
            messagebox.showerror("Error", f"Dataset missing meta.jsonl/labels.jsonl:\n{data_dir}")
            return

        out_dir = data_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        split = self.var_split.get().strip() or "test"
        device = self.var_device.get().strip() or "auto"
        max_samples_s = self.var_max_samples.get().strip()
        save_preds = bool(self.chk_save_preds.get())

        cmd: List[str] = ["bash", str(self.sim_root / "predict.sh"),
                          "--model", str(model_path),
                          "--data", str(data_dir),
                          "--split", split,
                          "--out-dir", str(out_dir)]
        if device != "auto":
            cmd.extend(["--device", device])
        if max_samples_s:
            try:
                int(max_samples_s)
            except Exception:
                messagebox.showerror("Error", "Max must be empty or an integer.")
                return
            cmd.extend(["--max-samples", max_samples_s])
        if save_preds:
            cmd.append("--save-preds")

        self.var_status.set("status: running...")
        self.var_report_path.set("report: -")
        self.var_preds_path.set("preds: -")
        self._set_metrics_text("Running batch prediction...\n")
        self._clear_confusion_matrix()
        self._pred_rows = []
        self._pred_row_by_id = {}
        self._refresh_tree()

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_evt.clear()

        def worker() -> None:
            report_path: Optional[Path] = None
            preds_path: Optional[Path] = None
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
                    if "Predictions saved to:" in line:
                        try:
                            preds_path = Path(line.split("Predictions saved to:", 1)[-1].strip())
                        except Exception:
                            preds_path = None

                rc = self.proc.wait()
                if self.stop_evt.is_set():
                    self.log_q.put("\n[stopped]\n")
                elif rc != 0:
                    self.log_q.put(f"\n[error] predict.sh exited with code {rc}\n")
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run predictions: {e}\n")
            finally:
                # Best-effort: if we didn't parse paths, pick newest from out_dir.
                try:
                    if report_path is None:
                        cand = sorted(out_dir.glob("batch_report_*.json"), key=lambda p: p.stat().st_mtime)
                        report_path = cand[-1] if cand else None
                    if preds_path is None:
                        cand = sorted(out_dir.glob("batch_preds_*.jsonl"), key=lambda p: p.stat().st_mtime)
                        preds_path = cand[-1] if cand else None
                except Exception:
                    pass

                def apply() -> None:
                    self.btn_run.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    if self.stop_evt.is_set():
                        self.var_status.set("status: stopped")
                    else:
                        self.var_status.set("status: done")
                    if report_path and report_path.exists():
                        self._load_report(report_path)
                    if preds_path and preds_path.exists():
                        self._load_preds(preds_path)

                try:
                    self.frame.after(0, apply)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

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

        names = [display(p) for p in cand]
        self.combo_dataset["values"] = names

        # Keep current selection if still valid.
        if self.var_dataset.get() in names:
            return

        # Prefer current state.dataset_dir
        if self.state.dataset_dir:
            want = display(self.state.dataset_dir)
            if want in names:
                self.var_dataset.set(want)
                return

        if names:
            self.var_dataset.set(names[0])
            self._on_dataset_selected()

    def _selected_dataset_dir(self) -> Optional[Path]:
        disp = self.var_dataset.get().strip()
        if not disp:
            return None
        for p in self._dataset_dirs:
            if p.name == disp:
                return p
            # versions display: group:snap
            try:
                sim_data = self._datasets_base()
                versions = (sim_data / "versions").resolve()
                rp = p.resolve()
                if str(rp).startswith(str(versions) + os.sep):
                    if disp == f"{rp.parent.name}:{rp.name}":
                        return p
            except Exception:
                pass
        return None

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        self.state.dataset_dir = ds
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
        # List available checkpoints including snapshots.
        cand: List[Path] = []
        for root in self._model_search_paths():
            if not root.exists():
                continue
            if root.name == "versions":
                cand.extend(sorted(root.glob("**/*.pt")))
            else:
                cand.extend(sorted(root.glob("*.pt")))

        # Dedup + sort
        uniq: Dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file():
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

        self._pred_rows = rows
        self._pred_row_by_id = {str(r.get("id")): r for r in rows if r.get("id")}
        self._refresh_tree()

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

            conf = self._top1_confidence(r)
            ok = "✓" if correct else "✗"
            self.tree.insert("", "end", iid=sid, text=sid, values=(gt, pred, f"{conf:.1%}", ok))

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
        if not row or not self.state.dataset_dir:
            return

        image_rel = row.get("image_path")
        if not image_rel:
            return
        image_path = self.state.dataset_dir / str(image_rel)
        if not image_path.exists():
            messagebox.showerror("Error", f"Image not found:\n{image_path}")
            return

        gt = str(row.get("gt") or "?")
        pred = str(row.get("pred") or "?")
        conf = self._top1_confidence(row)

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("failed to read image")
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
