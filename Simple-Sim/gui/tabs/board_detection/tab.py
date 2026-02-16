"""Board Detection tab - Two-stage inference (profile + defect classification)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gui.state import UiState

from gui.tabs.core.base import BaseTab
from .ui import BoardDetectionUI
from .logic import BoardDetectionLogic
from simple_sim.two_stage import TwoStageResult


class BoardDetectionTab(BaseTab):
    """Board Detection tab for two-stage inference."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.ui = BoardDetectionUI(self, self.frame)
        self.logic = BoardDetectionLogic()
        self._result: Optional[TwoStageResult] = None

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)

    def _browse_profile_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Profile Classifier Model",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir=str(self.sim_root / "outputs" / "models"),
        )
        if path:
            self.ui._profile_model_var.set(path)

    def _browse_defect_bundle(self) -> None:
        path = filedialog.askdirectory(
            title="Select Defect Model Bundle Directory",
            initialdir=str(self.sim_root / "outputs" / "models"),
        )
        if path:
            self.ui._defect_bundle_var.set(path)

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
            initialdir=str(self.sim_root / "outputs" / "sim_data"),
        )
        if path:
            self.ui._image_path_var.set(path)

    def _on_predict(self) -> None:
        profile_model = self.ui._profile_model_var.get().strip()
        defect_bundle = self.ui._defect_bundle_var.get().strip()
        image_path = self.ui._image_path_var.get().strip()

        if not profile_model or not defect_bundle or not image_path:
            messagebox.showwarning("Missing Input", "Please set profile model, defect bundle, and image path.")
            return

        if not Path(profile_model).exists():
            messagebox.showerror("Error", f"Profile model not found: {profile_model}")
            return
        if not Path(defect_bundle).exists():
            messagebox.showerror("Error", f"Defect bundle not found: {defect_bundle}")
            return
        if not Path(image_path).exists():
            messagebox.showerror("Error", f"Image not found: {image_path}")
            return

        try:
            min_conf = float(self.ui._conf_var.get())
        except ValueError:
            min_conf = 0.5

        self.ui.set_predict_button_state("disabled")
        self.ui.set_status("Loading models and predicting...")
        self.ui.set_review_flag("")

        self.logic.predict(
            profile_model_path=Path(profile_model),
            defect_bundle_path=Path(defect_bundle),
            image_path=Path(image_path),
            device=self.ui._device_var.get(),
            min_profile_confidence=min_conf,
            on_complete=self._on_predict_complete,
            on_error=self.ui.show_error,
        )

    def _on_predict_complete(self, result: TwoStageResult) -> None:
        self.ui.set_predict_button_state("normal")
        self.ui.set_status("Prediction complete")
        self._result = result
        self.ui.show_result(result)
