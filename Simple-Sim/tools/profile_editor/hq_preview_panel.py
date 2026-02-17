from __future__ import annotations

from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk


class HQPreviewPanel(ttk.Frame):
    """Displays Blender HQ preview image and status."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.status_var = tk.StringVar(value="HQ preview idle")
        self.path_var = tk.StringVar(value="")

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(top, textvariable=self.status_var).pack(side="left")
        ttk.Label(top, textvariable=self.path_var, foreground="#5f6a6a").pack(side="right")

        self.canvas = tk.Canvas(self, bg="#0d0f14", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._img = None
        self._img_path: Optional[Path] = None
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

    def set_status(self, status: str) -> None:
        self.status_var.set(status)

    def set_image(self, path: Optional[Path]) -> None:
        self._img_path = Path(path) if path else None
        self.path_var.set(str(self._img_path) if self._img_path else "")
        self._redraw()

    def apply_theme(self, *, dark: bool) -> None:
        self.canvas.configure(bg="#0d0f14" if dark else "#f5f7fa")
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        if self._img_path is None or not self._img_path.exists():
            self.canvas.create_text(w * 0.5, h * 0.5, text="No HQ image", fill="#cfd8dc")
            return

        try:
            from PIL import Image, ImageTk  # type: ignore

            img = Image.open(self._img_path)
            img.thumbnail((w - 16, h - 16), Image.Resampling.LANCZOS)
            self._img = ImageTk.PhotoImage(img)
            self.canvas.create_image(w * 0.5, h * 0.5, image=self._img)
        except Exception as exc:
            self.canvas.create_text(w * 0.5, h * 0.5, text=f"Preview load error: {exc}", fill="#f1948a")
