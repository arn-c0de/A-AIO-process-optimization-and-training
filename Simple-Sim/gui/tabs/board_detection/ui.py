"""UI for the Board Detection Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING, Optional

from simple_sim.two_stage import TwoStageResult

if TYPE_CHECKING:
    from .tab import BoardDetectionTab


class BoardDetectionUI:
    def __init__(self, tab: BoardDetectionTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)
        
        self._profile_model_var = tk.StringVar()
        self._defect_bundle_var = tk.StringVar()
        self._image_path_var = tk.StringVar()
        self._device_var = tk.StringVar(value="cpu")
        self._conf_var = tk.StringVar(value="0.5")
        self._status_var = tk.StringVar(value="Ready")
        self._review_var = tk.StringVar(value="")
        
        self._predict_btn: ttk.Button
        self._results_text: tk.Text
        self._review_label: ttk.Label
        
    def build_ui(self) -> None:
        main = ttk.PanedWindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=5)
        main.add(left, weight=1)

        right = ttk.Frame(main, padding=5)
        main.add(right, weight=1)

        model_frame = ttk.LabelFrame(left, text="Models", padding=5)
        model_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(model_frame, text="Profile Classifier (.pt):").pack(anchor="w")
        profile_row = ttk.Frame(model_frame)
        profile_row.pack(fill="x")
        ttk.Entry(profile_row, textvariable=self._profile_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(profile_row, text="Browse", command=self.tab._browse_profile_model).pack(side="right", padx=(4, 0))

        ttk.Label(model_frame, text="Defect Bundle (.bundle dir):").pack(anchor="w", pady=(4, 0))
        bundle_row = ttk.Frame(model_frame)
        bundle_row.pack(fill="x")
        ttk.Entry(bundle_row, textvariable=self._defect_bundle_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bundle_row, text="Browse", command=self.tab._browse_defect_bundle).pack(side="right", padx=(4, 0))

        image_frame = ttk.LabelFrame(left, text="Image Input", padding=5)
        image_frame.pack(fill="x", pady=(0, 5))

        img_row = ttk.Frame(image_frame)
        img_row.pack(fill="x")
        ttk.Entry(img_row, textvariable=self._image_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(img_row, text="Browse", command=self.tab._browse_image).pack(side="right", padx=(4, 0))

        ctrl_frame = ttk.Frame(left)
        ctrl_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(ctrl_frame, text="Device:").pack(side="left")
        device_combo = ttk.Combobox(ctrl_frame, textvariable=self._device_var, values=["cpu", "cuda"], width=6, state="readonly")
        device_combo.pack(side="left", padx=(4, 10))

        ttk.Label(ctrl_frame, text="Min confidence:").pack(side="left")
        ttk.Entry(ctrl_frame, textvariable=self._conf_var, width=6).pack(side="left", padx=(4, 10))

        self._predict_btn = ttk.Button(ctrl_frame, text="Predict", command=self.tab._on_predict)
        self._predict_btn.pack(side="right")

        ttk.Label(left, textvariable=self._status_var, foreground="gray").pack(anchor="w")

        self._results_frame = ttk.LabelFrame(right, text="Results", padding=5)
        self._results_frame.pack(fill="both", expand=True)

        self._results_text = tk.Text(self._results_frame, wrap="word", state="disabled", font=("Courier", 10))
        self._results_text.pack(fill="both", expand=True)

        self._review_label = ttk.Label(right, textvariable=self._review_var, foreground="orange", font=("", 11, "bold"))
        self._review_label.pack(anchor="w", pady=(4, 0))

    def set_predict_button_state(self, state: str) -> None:
        self._predict_btn.configure(state=state)

    def set_status(self, status: str) -> None:
        self._status_var.set(status)

    def set_review_flag(self, flag: str) -> None:
        self._review_var.set(flag)

    def show_result(self, result: TwoStageResult) -> None:
        lines = []
        lines.append("PROFILE IDENTIFICATION")
        lines.append(f"  Predicted: {result.profile_id}")
        lines.append(f"  Confidence: {result.profile_confidence:.4f}")
        lines.append("  Probabilities:")
        for name, prob in sorted(result.profile_probabilities.items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            lines.append(f"    {name:<30} {prob:.4f} {bar}")

        lines.append("")
        lines.append("DEFECT CLASSIFICATION")
        lines.append(f"  Predicted: {result.defect_class}")
        lines.append(f"  Confidence: {result.defect_confidence:.4f}")
        lines.append("  Probabilities:")
        for name, prob in sorted(result.defect_probabilities.items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            lines.append(f"    {name:<15} {prob:.4f} {bar}")

        self._results_text.configure(state="normal")
        self._results_text.delete("1.0", "end")
        self._results_text.insert("1.0", "\n".join(lines))
        self._results_text.configure(state="disabled")

        if result.review_flag:
            self._review_var.set("REVIEW FLAG: Low profile confidence - manual verification recommended")
        else:
            self._review_var.set("")

    def show_error(self, msg: str) -> None:
        self.set_predict_button_state("normal")
        self.set_status("Error")
        messagebox.showerror("Prediction Error", msg)
