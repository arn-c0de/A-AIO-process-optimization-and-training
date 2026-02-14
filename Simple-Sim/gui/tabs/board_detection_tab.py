"""Board Detection tab - Two-stage inference (profile + defect classification)."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gui.state import UiState

from gui.tabs.base_tab import BaseTab


class BoardDetectionTab(BaseTab):
    """Board Detection tab for two-stage inference."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self._classifier = None
        self._result = None

    def build_ui(self) -> None:
        """Build the Board Detection tab UI."""
        # Main paned layout
        main = ttk.PanedWindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Left panel: controls
        left = ttk.Frame(main, padding=5)
        main.add(left, weight=1)

        # Right panel: results
        right = ttk.Frame(main, padding=5)
        main.add(right, weight=1)

        # --- Left: Model selection ---
        model_frame = ttk.LabelFrame(left, text="Models", padding=5)
        model_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(model_frame, text="Profile Classifier (.pt):").pack(anchor="w")
        profile_row = ttk.Frame(model_frame)
        profile_row.pack(fill="x")
        self._profile_model_var = tk.StringVar()
        ttk.Entry(profile_row, textvariable=self._profile_model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(profile_row, text="Browse", command=self._browse_profile_model).pack(side="right", padx=(4, 0))

        ttk.Label(model_frame, text="Defect Bundle (.bundle dir):").pack(anchor="w", pady=(4, 0))
        bundle_row = ttk.Frame(model_frame)
        bundle_row.pack(fill="x")
        self._defect_bundle_var = tk.StringVar()
        ttk.Entry(bundle_row, textvariable=self._defect_bundle_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bundle_row, text="Browse", command=self._browse_defect_bundle).pack(side="right", padx=(4, 0))

        # --- Left: Image input ---
        image_frame = ttk.LabelFrame(left, text="Image Input", padding=5)
        image_frame.pack(fill="x", pady=(0, 5))

        img_row = ttk.Frame(image_frame)
        img_row.pack(fill="x")
        self._image_path_var = tk.StringVar()
        ttk.Entry(img_row, textvariable=self._image_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(img_row, text="Browse", command=self._browse_image).pack(side="right", padx=(4, 0))

        # --- Left: Device + Predict ---
        ctrl_frame = ttk.Frame(left)
        ctrl_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(ctrl_frame, text="Device:").pack(side="left")
        self._device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(ctrl_frame, textvariable=self._device_var, values=["cpu", "cuda"], width=6, state="readonly")
        device_combo.pack(side="left", padx=(4, 10))

        ttk.Label(ctrl_frame, text="Min confidence:").pack(side="left")
        self._conf_var = tk.StringVar(value="0.5")
        ttk.Entry(ctrl_frame, textvariable=self._conf_var, width=6).pack(side="left", padx=(4, 10))

        self._predict_btn = ttk.Button(ctrl_frame, text="Predict", command=self._on_predict)
        self._predict_btn.pack(side="right")

        # --- Left: Status ---
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self._status_var, foreground="gray").pack(anchor="w")

        # --- Right: Results ---
        self._results_frame = ttk.LabelFrame(right, text="Results", padding=5)
        self._results_frame.pack(fill="both", expand=True)

        self._results_text = tk.Text(self._results_frame, wrap="word", state="disabled", font=("Courier", 10))
        self._results_text.pack(fill="both", expand=True)

        # Review flag indicator
        self._review_var = tk.StringVar(value="")
        self._review_label = ttk.Label(right, textvariable=self._review_var, foreground="orange", font=("", 11, "bold"))
        self._review_label.pack(anchor="w", pady=(4, 0))

    def _browse_profile_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Profile Classifier Model",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir=str(self.sim_root / "outputs" / "models"),
        )
        if path:
            self._profile_model_var.set(path)

    def _browse_defect_bundle(self) -> None:
        path = filedialog.askdirectory(
            title="Select Defect Model Bundle Directory",
            initialdir=str(self.sim_root / "outputs" / "models"),
        )
        if path:
            self._defect_bundle_var.set(path)

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
            initialdir=str(self.sim_root / "outputs" / "sim_data"),
        )
        if path:
            self._image_path_var.set(path)

    def _on_predict(self) -> None:
        profile_model = self._profile_model_var.get().strip()
        defect_bundle = self._defect_bundle_var.get().strip()
        image_path = self._image_path_var.get().strip()

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
            min_conf = float(self._conf_var.get())
        except ValueError:
            min_conf = 0.5

        self._predict_btn.configure(state="disabled")
        self._status_var.set("Loading models and predicting...")
        self._review_var.set("")

        def _run():
            try:
                from simple_sim.two_stage import TwoStageClassifier

                # Re-create classifier if settings changed
                if (self._classifier is None
                        or getattr(self._classifier, '_init_profile_path', None) != profile_model
                        or getattr(self._classifier, '_init_bundle_path', None) != defect_bundle):
                    self._classifier = TwoStageClassifier(
                        profile_model_path=Path(profile_model),
                        defect_bundle_path=Path(defect_bundle),
                        device=self._device_var.get(),
                        min_profile_confidence=min_conf,
                    )
                    self._classifier._init_profile_path = profile_model
                    self._classifier._init_bundle_path = defect_bundle
                else:
                    self._classifier.min_profile_confidence = min_conf

                result = self._classifier.predict(Path(image_path))
                self.frame.after(0, lambda: self._show_result(result))
            except Exception as e:
                self.frame.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, result) -> None:
        self._predict_btn.configure(state="normal")
        self._status_var.set("Prediction complete")
        self._result = result

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

    def _show_error(self, msg: str) -> None:
        self._predict_btn.configure(state="normal")
        self._status_var.set("Error")
        messagebox.showerror("Prediction Error", msg)
