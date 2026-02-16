"""UI for the Predictions Tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from gui.components.chart_widgets import create_confusion_matrix_widget
from gui.components.overlay_renderer import draw_prediction_overlay, draw_two_stage_overlay
from gui.utils.tooltip import ToolTip
import numpy as np
from PIL import Image, ImageTk
import cv2


if TYPE_CHECKING:
    from .tab import PredictionsTab


class PredictionsUI:
    def __init__(self, tab: PredictionsTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)

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
        self.chk_auto_train_bundle: tk.BooleanVar
        self.var_profile_model: tk.StringVar
        self.var_multi_sel: tk.StringVar
        self.var_multi_datasets: tk.BooleanVar

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
        self._cm_parent: ttk.Frame

    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        top.columnconfigure(4, weight=1)

        ttk.Label(top, text="Dataset:").grid(row=0, column=0, sticky="w")
        self.var_dataset = tk.StringVar(value="")
        self.combo_dataset = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly")
        self.combo_dataset.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.combo_dataset.bind("<<ComboboxSelected>>", self.tab._on_dataset_selected)
        ToolTip(self.combo_dataset, text_func=lambda: self.var_dataset.get())
        ds_tools = ttk.Frame(top)
        ds_tools.grid(row=0, column=2, sticky="w", padx=(0, 12))
        ttk.Button(ds_tools, text="↻", width=3, command=self.tab._refresh_datasets).pack(side="left")
        self.var_multi_datasets = tk.BooleanVar(value=False)
        ttk.Checkbutton(ds_tools, text="Multi", variable=self.var_multi_datasets, command=self.tab._on_multi_toggle).pack(side="left", padx=(8, 0))
        ttk.Button(ds_tools, text="Select…", command=self.tab._open_multi_dataset_dialog).pack(side="left", padx=(6, 0))
        self.var_multi_sel = tk.StringVar(value="selected: 0")
        ttk.Label(ds_tools, textvariable=self.var_multi_sel).pack(side="left", padx=(8, 0))

        ttk.Label(top, text="Model:").grid(row=0, column=3, sticky="w")
        self.var_model = tk.StringVar(value="")
        self.combo_model = ttk.Combobox(top, textvariable=self.var_model, state="readonly")
        self.combo_model.grid(row=0, column=4, sticky="ew", padx=(6, 6))
        ttk.Button(top, text="↻", width=3, command=self.tab._refresh_models).grid(row=0, column=5, sticky="w", padx=(0, 12))

        self.btn_run = ttk.Button(top, text="Start Predictions", command=self.tab._run_predictions, width=16)
        self.btn_run.grid(row=0, column=6, sticky="e")
        self.btn_stop = ttk.Button(top, text="Stop", command=self.tab._stop_predictions, state="disabled", width=8)
        self.btn_stop.grid(row=0, column=7, sticky="e", padx=(8, 0))

        opts = ttk.Frame(self.frame)
        opts.pack(fill="x", pady=(0, 8))

        ttk.Label(opts, text="Split:").pack(side="left")
        self.var_split = tk.StringVar(value="test")
        ttk.Combobox(opts, textvariable=self.var_split, values=["train", "val", "test", "all"], state="readonly", width=6).pack(side="left", padx=(6, 12))

        ttk.Label(opts, text="Device:").pack(side="left")
        self.var_device = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.var_device, values=["auto", "cpu", "cuda"], state="readonly", width=6).pack(side="left", padx=(6, 12))

        ttk.Label(opts, text="Max:").pack(side="left")
        self.var_max_samples = tk.StringVar(value="")
        ttk.Entry(opts, textvariable=self.var_max_samples, width=6).pack(side="left", padx=(6, 12))

        self.chk_save_preds = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Save per-sample preds", variable=self.chk_save_preds).pack(side="left")

        self.chk_auto_train_bundle = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Auto-train missing bundle ckpt", variable=self.chk_auto_train_bundle).pack(side="left", padx=(10, 0))

        ttk.Separator(opts, orient="vertical").pack(side="left", fill="y", padx=(12, 12), pady=2)
        ttk.Label(opts, text="Profile Model:").pack(side="left")
        self.var_profile_model = tk.StringVar(value="")
        self.combo_profile_model = ttk.Combobox(opts, textvariable=self.var_profile_model, state="readonly", width=30)
        self.combo_profile_model.pack(side="left", padx=(6, 4))
        ttk.Button(opts, text="↻", width=3, command=self.tab._refresh_profile_models).pack(side="left")

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

        left = ttk.Frame(main, padding=5)
        main.add(left, weight=2)

        self.ds_summary_frame = ttk.Frame(left)
        ttk.Label(self.ds_summary_frame, text="Dataset Results").pack(anchor="w")
        ds_tree_frame = ttk.Frame(self.ds_summary_frame)
        ds_tree_frame.pack(fill="x", expand=False, pady=(4, 0))
        self.tree_ds = ttk.Treeview(ds_tree_frame, columns=("Dataset", "Acc", "F1", "Seen", "ProfAcc"), show="headings", height=6)
        self.tree_ds.heading("Dataset", text="Dataset")
        self.tree_ds.column("Dataset", width=210, anchor="w")
        self.tree_ds.heading("Acc", text="Acc")
        self.tree_ds.column("Acc", width=70, anchor="e")
        self.tree_ds.heading("F1", text="Macro F1")
        self.tree_ds.column("F1", width=80, anchor="e")
        self.tree_ds.heading("Seen", text="Seen")
        self.tree_ds.column("Seen", width=60, anchor="e")
        self.tree_ds.heading("ProfAcc", text="Prof Acc")
        self.tree_ds.column("ProfAcc", width=80, anchor="e")
        ds_scroll = ttk.Scrollbar(ds_tree_frame, orient="vertical", command=self.tree_ds.yview)
        self.tree_ds.configure(yscrollcommand=ds_scroll.set)
        self.tree_ds.pack(side="left", fill="x", expand=True)
        ds_scroll.pack(side="right", fill="y")
        self.tree_ds.bind("<<TreeviewSelect>>", self.tab._on_tree_ds_select)
        self.ds_summary_frame.pack_forget()

        filters = ttk.Frame(left)
        filters.pack(fill="x", pady=(0, 6))
        ttk.Label(filters, text="Show:").pack(side="left")
        self.var_filter = tk.StringVar(value="all")
        ttk.Radiobutton(filters, text="All", variable=self.var_filter, value="all", command=self.tab._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Wrong", variable=self.var_filter, value="wrong", command=self.tab._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Correct", variable=self.var_filter, value="correct", command=self.tab._refresh_tree).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(filters, text="Prof Wrong", variable=self.var_filter, value="prof_wrong", command=self.tab._refresh_tree).pack(side="left", padx=(6, 0))
        
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("GT", "Pred", "Conf", "OK", "GT Prof", "Pred Prof", "Prof"), show="tree headings")
        self.tree.heading("#0", text="Sample ID")
        self.tree.column("#0", width=200)
        self.tree.heading("GT", text="GT")
        self.tree.column("GT", width=80, anchor="center")
        self.tree.heading("Pred", text="Pred")
        self.tree.column("Pred", width=90, anchor="center")
        self.tree.heading("Conf", text="Conf")
        self.tree.column("Conf", width=60, anchor="e")
        self.tree.heading("OK", text="OK")
        self.tree.column("OK", width=40, anchor="center")
        self.tree.heading("GT Prof", text="GT Prof")
        self.tree.column("GT Prof", width=120, anchor="center")
        self.tree.heading("Pred Prof", text="Pred Prof")
        self.tree.column("Pred Prof", width=120, anchor="center")
        self.tree.heading("Prof", text="Prof")
        self.tree.column("Prof", width=40, anchor="center")
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.tab._on_tree_select)

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

    def set_metrics_text(self, s: str) -> None:
        self.txt_metrics.configure(state="normal")
        self.txt_metrics.delete("1.0", "end")
        self.txt_metrics.insert("1.0", s)
        self.txt_metrics.configure(state="disabled")

    def append_log(self, s: str) -> None:
        self.txt_logs.configure(state="normal")
        self.txt_logs.insert("end", s)
        self.txt_logs.see("end")
        self.txt_logs.configure(state="disabled")

    def clear_confusion_matrix(self) -> None:
        for w in list(self._cm_parent.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._cm_canvas = None

    def render_confusion_matrix(self, cm: List[List[int]], class_names: List[str]) -> None:
        self.clear_confusion_matrix()
        try:
            if isinstance(cm, list) and cm and class_names:
                cm_np = np.array(cm, dtype=np.int64)
                self._cm_canvas = create_confusion_matrix_widget(self._cm_parent, cm_np, class_names)
                self._cm_canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            self.append_log(f"[warn] failed to render confusion matrix: {e}\\n")

    def refresh_tree(self, pred_rows: List[Dict[str, Any]], filter_mode: str) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())

        for r in pred_rows:
            sid = str(r.get("id") or "")
            if not sid:
                continue
            correct = bool(r.get("correct"))
            if filter_mode == "wrong" and correct:
                continue
            if filter_mode == "correct" and not correct:
                continue
            
            profile_correct = r.get("profile_correct")
            if filter_mode == "prof_wrong":
                if profile_correct is None or profile_correct:
                    continue

            gt = str(r.get("gt") or "?")
            pred = str(r.get("pred") or "?")
            conf = self._top1_confidence(r)
            ok = "✓" if correct else "✗"
            prof_mark = ""
            if profile_correct is not None:
                prof_mark = "✓" if profile_correct else "✗"
            
            self.tree.insert("", "end", iid=sid, text=sid, values=(gt, pred, f"{conf:.1%}", ok, str(r.get("gt_profile") or ""), str(r.get("pred_profile") or ""), prof_mark))

    def _top1_confidence(self, row: Dict[str, Any]) -> float:
        try:
            topk = row.get("topk") or []
            if not topk:
                return 0.0
            pred = row.get("pred")
            for t in topk:
                if t.get("class") == pred:
                    return float(t.get("prob") or 0.0)
            return float(topk[0].get("prob") or 0.0)
        except Exception:
            return 0.0

    def display_image(self, image_path: str, row: Dict[str, Any]) -> None:
        gt = str(row.get("gt") or "?")
        pred = str(row.get("pred") or "?")
        conf = self._top1_confidence(row)
        gt_profile = str(row.get("gt_profile") or "")
        pred_profile = str(row.get("pred_profile") or "")
        profile_conf = float(row.get("profile_confidence") or 0.0)

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("failed to read image")
            if gt_profile and pred_profile:
                img_overlay = draw_two_stage_overlay(img, gt, pred, conf, gt_profile, pred_profile, profile_conf)
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
            # Should be handled in the tab with a messagebox
            raise e
