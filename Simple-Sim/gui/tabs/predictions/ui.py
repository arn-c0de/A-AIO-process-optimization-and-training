"""UI for the Predictions Tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from gui.components.chart_widgets import create_confusion_matrix_widget
from gui.components.overlay_renderer import draw_prediction_overlay, draw_two_stage_overlay
from gui.theme.style import on_theme_changed, palette
from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip
from gui.widgets import FormGrid, LogView, ScrolledTree, Section, Toolbar, icon_button
import numpy as np
from PIL import Image, ImageTk
import cv2
import contextlib


if TYPE_CHECKING:
    from .tab import PredictionsTab


class PredictionsUI:
    def __init__(self, tab: PredictionsTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=SPACE.L)

        self._cm_canvas = None
        self._photo: ImageTk.PhotoImage | None = None
        self._right_paned: ttk.Panedwindow | None = None
        self._min_logs_pane_px: int = 180
        self._pane_adjust_scheduled: bool = False

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
        self.tree_view: ScrolledTree
        self.lbl_tree_empty: ttk.Label
        self.metrics_view: LogView
        self.log_view: LogView
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
        # ------------------------------------------------------------ inputs
        top = ttk.Frame(self.frame)
        top.pack(fill="x")
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)

        src = Section(top, "Source", help_key="predictions.source")
        src.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE.M))
        form = FormGrid(src, style="Card.TFrame")
        form.pack(fill="x")

        ds_row = form.inline("Card.TFrame")
        self.var_dataset = tk.StringVar(value="")
        self.combo_dataset = ttk.Combobox(ds_row, textvariable=self.var_dataset, state="readonly")
        self.combo_dataset.pack(side="left", fill="x", expand=True)
        self.combo_dataset.bind("<<ComboboxSelected>>", self.tab._on_dataset_selected)
        ToolTip(self.combo_dataset, help_key="common.dataset", text_func=lambda: self.var_dataset.get())
        icon_button(ds_row, "refresh", self.tab._refresh_datasets, help_key="common.refresh").pack(
            side="left", padx=(SPACE.S, 0))
        self.var_multi_datasets = tk.BooleanVar(value=False)
        chk_multi = ttk.Checkbutton(ds_row, text="Multi", variable=self.var_multi_datasets,
                                    command=self.tab._on_multi_toggle, style="Card.TCheckbutton")
        chk_multi.pack(side="left", padx=(SPACE.M, 0))
        ToolTip(chk_multi, help_key="predictions.dataset_multi")
        b_sel = ttk.Button(ds_row, text="Select…", command=self.tab._open_multi_dataset_dialog)
        b_sel.pack(side="left", padx=(SPACE.S, 0))
        ToolTip(b_sel, help_key="predictions.dataset_multi_pick")
        self.var_multi_sel = tk.StringVar(value="selected: 0")
        ttk.Label(ds_row, textvariable=self.var_multi_sel, style="CardMuted.TLabel").pack(
            side="left", padx=(SPACE.S, 0))
        form.row("Dataset", ds_row)

        model_row = form.inline("Card.TFrame")
        self.var_model = tk.StringVar(value="")
        self.combo_model = ttk.Combobox(model_row, textvariable=self.var_model, state="readonly")
        self.combo_model.pack(side="left", fill="x", expand=True)
        ToolTip(self.combo_model, help_key="common.model", text_func=lambda: self.var_model.get())
        icon_button(model_row, "refresh", self.tab._refresh_models, help_key="common.refresh").pack(
            side="left", padx=(SPACE.S, 0))
        form.row("Model", model_row)

        pm_row = form.inline("Card.TFrame")
        self.var_profile_model = tk.StringVar(value="")
        self.combo_profile_model = ttk.Combobox(pm_row, textvariable=self.var_profile_model, state="readonly", width=30)
        self.combo_profile_model.pack(side="left", fill="x", expand=True)
        ToolTip(self.combo_profile_model, help_key="predictions.profile_model")
        icon_button(pm_row, "refresh", self.tab._refresh_profile_models, help_key="common.refresh").pack(
            side="left", padx=(SPACE.S, 0))
        form.row("Profile Model", pm_row)

        opts = Section(top, "Options")
        opts.grid(row=0, column=1, sticky="nsew")
        oform = FormGrid(opts, style="Card.TFrame")
        oform.pack(fill="x")
        self.var_split = tk.StringVar(value="test")
        oform.row("Split", ttk.Combobox(oform, textvariable=self.var_split, values=["train", "val", "test", "all"],
                                        state="readonly", width=8), help_key="common.split", sticky="w")
        self.var_device = tk.StringVar(value="auto")
        oform.row("Device", ttk.Combobox(oform, textvariable=self.var_device, values=["auto", "cpu", "cuda"],
                                         state="readonly", width=8), help_key="common.device", sticky="w")
        self.var_max_samples = tk.StringVar(value="")
        oform.row("Max Samples", ttk.Entry(oform, textvariable=self.var_max_samples, width=8),
                  help_key="predictions.max_samples", sticky="w")
        self.chk_save_preds = tk.BooleanVar(value=True)
        cb_save = oform.full(ttk.Checkbutton(oform, text="Save per-sample predictions", variable=self.chk_save_preds,
                                             style="Card.TCheckbutton"))
        ToolTip(cb_save, help_key="predictions.save_preds")
        self.chk_auto_train_bundle = tk.BooleanVar(value=False)
        cb_auto = oform.full(ttk.Checkbutton(oform, text="Auto-train missing bundle checkpoints",
                                             variable=self.chk_auto_train_bundle, style="Card.TCheckbutton"))
        ToolTip(cb_auto, help_key="predictions.auto_train_bundle")

        # ------------------------------------------------------------ actions + status
        bar = Toolbar(self.frame)
        bar.pack(fill="x", pady=(SPACE.M, 0))
        self.btn_run = bar.button("Start Predictions", self.tab._run_predictions,
                                  style="Accent.TButton", help_key="predictions.run")
        self.btn_stop = bar.button("Stop", self.tab._stop_predictions, style="Danger.TButton",
                                   help_key="predictions.stop", state="disabled")
        bar.separator()
        self.var_status = tk.StringVar(value="status: idle")
        bar.add(ttk.Label(bar, textvariable=self.var_status, style="Badge.TLabel"))

        paths = ttk.Frame(self.frame)
        paths.pack(fill="x", pady=(SPACE.S, 0))
        self.var_report_path = tk.StringVar(value="report: -")
        self.var_preds_path = tk.StringVar(value="preds: -")
        ttk.Label(paths, textvariable=self.var_report_path, style="Small.TLabel").pack(side="left", padx=(0, SPACE.L))
        ttk.Label(paths, textvariable=self.var_preds_path, style="Small.TLabel").pack(side="left")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(SPACE.M, 0))

        # ------------------------------------------------------------ results list
        left = Section(main, "Results", help_key="predictions.results")
        main.add(left, weight=2)

        # Holder keeps the multi-dataset table at the top when the tab (re)packs it.
        ds_holder = ttk.Frame(left, style="Card.TFrame")
        ds_holder.pack(fill="x")
        self.ds_summary_frame = ttk.Frame(ds_holder, style="Card.TFrame")
        ttk.Label(self.ds_summary_frame, text="Dataset Results", style="CardMuted.TLabel").pack(anchor="w")
        ds_view = ScrolledTree(
            self.ds_summary_frame,
            [("Dataset", "Dataset", 210, "w"), ("Acc", "Acc", 70, "e"), ("F1", "Macro F1", 80, "e"),
             ("Seen", "Seen", 60, "e"), ("ProfAcc", "Prof Acc", 80, "e")],
            height=6,
        )
        ds_view.pack(fill="x", expand=False, pady=(SPACE.S, 0))
        self.tree_ds = ds_view.tree
        ToolTip(self.tree_ds, help_key="predictions.ds_results")
        self.tree_ds.bind("<<TreeviewSelect>>", self.tab._on_tree_ds_select)

        filters = Toolbar(left, style="Card.TFrame")
        filters.pack(fill="x", pady=(0, SPACE.S))
        filters.add(ttk.Label(filters, text="Show", style="CardMuted.TLabel"))
        self.var_filter = tk.StringVar(value="all")
        for text, value in (("All", "all"), ("Wrong", "wrong"), ("Correct", "correct"), ("Prof Wrong", "prof_wrong")):
            rb = ttk.Radiobutton(filters, text=text, variable=self.var_filter, value=value,
                                 command=self.tab._refresh_tree, style="Card.TRadiobutton")
            filters.add(rb)
            ToolTip(rb, help_key="predictions.show_filter")
        filters.add_right(icon_button(filters, "next", self.tab._next_sample, help_key="common.next"))
        filters.add_right(icon_button(filters, "prev", self.tab._prev_sample, help_key="common.prev"))

        self.tree_view = ScrolledTree(
            left,
            [("GT", "GT", 80, "center"), ("Pred", "Pred", 90, "center"), ("Conf", "Conf", 60, "e"),
             ("OK", "OK", 40, "center"), ("GT Prof", "GT Prof", 120, "center"),
             ("Pred Prof", "Pred Prof", 120, "center"), ("Prof", "Prof", 40, "center")],
            show_tree=True,
            tree_heading="Sample ID",
            tree_width=200,
        )
        self.tree_view.pack(fill="both", expand=True)
        self.tree = self.tree_view.tree
        self.lbl_tree_empty = ttk.Label(self.tree_view, text="No predictions yet. Press Start Predictions.",
                                        style="CardMuted.TLabel")
        self.lbl_tree_empty.place(relx=0.5, rely=0.4, anchor="center")
        self.tree.bind("<<TreeviewSelect>>", self.tab._on_tree_select)
        self.tree.bind("<Left>", lambda _e: (self.tab._prev_sample(), "break")[1])
        self.tree.bind("<Right>", lambda _e: (self.tab._next_sample(), "break")[1])

        # ------------------------------------------------------------ report / viewer / logs
        right = ttk.Frame(main)
        main.add(right, weight=3)

        top_right = ttk.Panedwindow(right, orient="vertical")
        top_right.pack(fill="both", expand=True)
        self._right_paned = top_right
        top_right.bind("<Configure>", lambda _e: self._schedule_enforce_log_pane_min_height())

        metrics_box = Section(top_right, "Metrics / Report")
        top_right.add(metrics_box, weight=2)
        self.metrics_view = LogView(metrics_box, height=10, wrap="word")
        self.metrics_view.pack(fill="x", expand=False, pady=(0, SPACE.M))
        self.txt_metrics = self.metrics_view.text

        cm_box = ttk.Frame(metrics_box, style="Card.TFrame")
        cm_box.pack(fill="both", expand=True)
        ttk.Label(cm_box, text="Confusion Matrix", style="CardMuted.TLabel").pack(anchor="w")
        self._cm_parent = ttk.Frame(cm_box, style="Card.TFrame")
        self._cm_parent.pack(fill="both", expand=True, pady=(SPACE.S, 0))

        viewer_box = Section(top_right, "Image Viewer")
        top_right.add(viewer_box, weight=2)
        self.canvas = tk.Canvas(viewer_box, background=palette(self.frame).canvas_bg, highlightthickness=0,
                                width=520, height=520, takefocus=1)
        self.canvas._simsim_role = "image"  # type: ignore[attr-defined]
        self.canvas.pack(fill="both", expand=True)
        on_theme_changed(self.canvas, lambda pal: self.canvas.configure(background=pal.canvas_bg))
        self.canvas.bind("<Button-1>", lambda _e: self.canvas.focus_set())
        self.canvas.bind("<Left>", lambda _e: (self.tab._prev_sample(), "break")[1])
        self.canvas.bind("<Right>", lambda _e: (self.tab._next_sample(), "break")[1])

        logs_box = Section(top_right, "Logs", height=220)
        logs_box.pack_propagate(False)
        top_right.add(logs_box, weight=2)
        self.log_view = LogView(logs_box, height=10)
        self.log_view.pack(fill="both", expand=True)
        self.txt_logs = self.log_view.text
        b_clear = ttk.Button(logs_box.actions, text="Clear", style="Icon.TButton", command=self.log_view.clear)
        b_clear.pack(side="right")
        ToolTip(b_clear, help_key="common.clear_log")
        b_copy = ttk.Button(logs_box.actions, text="Copy", style="Icon.TButton", command=self.log_view.copy_all)
        b_copy.pack(side="right", padx=(0, SPACE.S))
        ToolTip(b_copy, help_key="common.copy_log")
        self._schedule_enforce_log_pane_min_height()

    def set_metrics_text(self, s: str) -> None:
        self.txt_metrics.configure(state="normal")
        self.txt_metrics.delete("1.0", "end")
        self.txt_metrics.insert("1.0", s)
        self.txt_metrics.configure(state="disabled")

    def append_log(self, s: str) -> None:
        self.log_view.append(s)

    def clear_confusion_matrix(self) -> None:
        for w in list(self._cm_parent.winfo_children()):
            with contextlib.suppress(Exception):
                w.destroy()
        self._cm_canvas = None
        self._schedule_enforce_log_pane_min_height()

    def render_confusion_matrix(self, cm: list[list[int]], class_names: list[str]) -> None:
        self.clear_confusion_matrix()
        try:
            if isinstance(cm, list) and cm and class_names:
                cm_np = np.array(cm, dtype=np.int64)
                self._cm_canvas = create_confusion_matrix_widget(self._cm_parent, cm_np, class_names)
                self._cm_canvas.get_tk_widget().pack(fill="both", expand=True)
                self._schedule_enforce_log_pane_min_height()
        except Exception as e:
            self.append_log(f"[warn] failed to render confusion matrix: {e}\n")

    def _schedule_enforce_log_pane_min_height(self) -> None:
        if self._pane_adjust_scheduled:
            return
        self._pane_adjust_scheduled = True
        try:
            self.frame.after_idle(self._enforce_log_pane_min_height)
        except Exception:
            self._pane_adjust_scheduled = False

    def _enforce_log_pane_min_height(self) -> None:
        self._pane_adjust_scheduled = False
        if self._right_paned is None:
            return
        try:
            pane_count = len(self._right_paned.panes())
            if pane_count < 2:
                return
            total_h = self._right_paned.winfo_height()
            if total_h <= 0:
                return
            sash_idx = pane_count - 2
            max_sash_pos = max(0, total_h - self._min_logs_pane_px)
            cur = self._right_paned.sashpos(sash_idx)
            if cur > max_sash_pos:
                self._right_paned.sashpos(sash_idx, max_sash_pos)
        except Exception:
            pass

    def refresh_tree(self, pred_rows: list[dict[str, Any]], filter_mode: str) -> None:
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
            
            self.tree_view.insert(
                (gt, pred, f"{conf:.1%}", ok, str(r.get("gt_profile") or ""), str(r.get("pred_profile") or ""), prof_mark),
                iid=sid,
                text=sid,
                tags=() if correct else ("danger",),
            )
        if self.tree.get_children(""):
            self.lbl_tree_empty.place_forget()
        else:
            self.lbl_tree_empty.configure(
                text="No predictions yet. Press Start Predictions." if not pred_rows else "No samples match this filter."
            )
            self.lbl_tree_empty.place(relx=0.5, rely=0.4, anchor="center")

    def _top1_confidence(self, row: dict[str, Any]) -> float:
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

    def display_image(self, image_path: str, row: dict[str, Any]) -> None:
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
