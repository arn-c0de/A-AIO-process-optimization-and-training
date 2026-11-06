"""UI for the Board Detection Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from gui import dialogs
from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip
from gui.widgets import FormGrid, LogView, PathPicker, Section
from simple_sim.two_stage import TwoStageResult

if TYPE_CHECKING:
    from .tab import BoardDetectionTab

_RESULTS_PLACEHOLDER = (
    "No prediction yet.\n\n"
    "1. Choose a profile classifier (.pt) and a defect bundle (.bundle folder).\n"
    "2. Choose an image.\n"
    "3. Press Predict."
)


class BoardDetectionUI:
    def __init__(self, tab: BoardDetectionTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=0)

        self._profile_model_var = tk.StringVar()
        self._defect_bundle_var = tk.StringVar()
        self._image_path_var = tk.StringVar()
        self._device_var = tk.StringVar(value="cpu")
        self._conf_var = tk.StringVar(value="0.5")
        self._status_var = tk.StringVar(value="Ready")
        self._review_var = tk.StringVar(value="")

        self._predict_btn: ttk.Button
        self._results_text: tk.Text
        self._results_view: LogView
        self._review_label: ttk.Label

    def build_ui(self) -> None:
        tab = self.tab
        main = ttk.PanedWindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=(0, 0, SPACE.M, 0))
        main.add(left, weight=1)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        # --- Models
        models = Section(left, "Models", help_key="board_detection.models")
        models.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(models, style="Card.TFrame")
        form.pack(fill="x")
        pp_profile = PathPicker(form, self._profile_model_var, style="Card.TFrame")
        pp_profile.button.configure(command=tab._browse_profile_model)
        form.row("Profile Classifier", pp_profile, help_key="board_detection.profile_model",
                 hint="Component-type classifier checkpoint (.pt)")
        pp_bundle = PathPicker(form, self._defect_bundle_var, style="Card.TFrame")
        pp_bundle.button.configure(command=tab._browse_defect_bundle)
        form.row("Defect Bundle", pp_bundle, help_key="board_detection.bundle",
                 hint="Multi-profile model folder (*.bundle)")

        # --- Image + options
        inp = Section(left, "Input", help_key="board_detection.input")
        inp.pack(fill="x", pady=(0, SPACE.M))
        form2 = FormGrid(inp, style="Card.TFrame")
        form2.pack(fill="x")
        pp_image = PathPicker(form2, self._image_path_var, style="Card.TFrame")
        pp_image.button.configure(command=tab._browse_image)
        form2.row("Image", pp_image, help_key="board_detection.image")
        form2.row("Device", lambda p: ttk.Combobox(p, textvariable=self._device_var, values=["cpu", "cuda"],
                                                   width=8, state="readonly"),
                  help_key="board_detection.device", sticky="w")
        form2.row("Min Confidence", lambda p: ttk.Entry(p, textvariable=self._conf_var, width=8),
                  help_key="board_detection.confidence", sticky="w")

        actions = ttk.Frame(inp, style="Card.TFrame")
        actions.pack(fill="x", pady=(SPACE.M, 0))
        self._predict_btn = ttk.Button(actions, text="Predict", command=tab._on_predict, style="Accent.TButton")
        self._predict_btn.pack(side="right")
        ToolTip(self._predict_btn, help_key="board_detection.predict")
        ttk.Label(actions, textvariable=self._status_var, style="CardMuted.TLabel").pack(side="left")

        # --- Results
        self._results_frame = Section(right, "Results", help_key="board_detection.results")
        self._results_frame.pack(fill="both", expand=True)
        self._review_label = ttk.Label(self._results_frame, textvariable=self._review_var,
                                       style="CardWarning.TLabel", wraplength=480, justify="left")
        self._review_label.pack(anchor="w", pady=(0, SPACE.S))
        self._results_view = LogView(self._results_frame, height=20, wrap="none")
        self._results_view.pack(fill="both", expand=True)
        self._results_text = self._results_view.text
        self._set_results(_RESULTS_PLACEHOLDER)

    def _set_results(self, text: str) -> None:
        self._results_view.clear()
        self._results_view.append(text, level="")

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

        self._set_results("\n".join(lines))

        if result.review_flag:
            self._review_var.set("REVIEW FLAG: Low profile confidence - manual verification recommended")
        else:
            self._review_var.set("")

    def show_error(self, msg: str) -> None:
        self.set_predict_button_state("normal")
        self.set_status("Error")
        dialogs.show_error(self.frame, "Prediction Failed", msg,
                           hint="Check the model paths, the device and that the image can be read.")
