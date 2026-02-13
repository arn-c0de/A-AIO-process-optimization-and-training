"""Tiny Tkinter tooltip helper (no external deps)."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional


class ToolTip:
    """A small hover tooltip for a widget.

    Usage:
      ToolTip(widget, text_func=lambda: "hello")
    """

    def __init__(self, widget: tk.Widget, *, text_func: Callable[[], str], delay_ms: int = 450) -> None:
        self.widget = widget
        self.text_func = text_func
        self.delay_ms = int(delay_ms)

        self._after_id: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None
        self._label: Optional[tk.Label] = None

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")

    def _on_enter(self, _evt: tk.Event) -> None:
        self._schedule()

    def _on_leave(self, _evt: tk.Event) -> None:
        self._cancel()
        self._hide()

    def _on_motion(self, _evt: tk.Event) -> None:
        # Keep tooltip anchored near the cursor even when widget moves.
        if self._tip is not None:
            try:
                x, y = self.widget.winfo_pointerxy()
                self._tip.geometry(f"+{x + 12}+{y + 16}")
            except Exception:
                pass

    def _schedule(self) -> None:
        self._cancel()
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except Exception:
            self._after_id = None

    def _cancel(self) -> None:
        if not self._after_id:
            return
        try:
            self.widget.after_cancel(self._after_id)
        except Exception:
            pass
        self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        text = ""
        try:
            text = str(self.text_func() or "").strip()
        except Exception:
            text = ""
        if not text:
            return

        # Reuse existing tooltip if it exists.
        if self._tip is None or not self._tip.winfo_exists():
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.attributes("-topmost", True)
            self._label = tk.Label(
                self._tip,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("TkDefaultFont", 9),
            )
            self._label.pack(ipadx=6, ipady=3)
        else:
            assert self._label is not None
            self._label.configure(text=text)

        try:
            x, y = self.widget.winfo_pointerxy()
            self._tip.geometry(f"+{x + 12}+{y + 16}")
            self._tip.deiconify()
        except Exception:
            self._hide()

    def _hide(self) -> None:
        if self._tip is None:
            return
        try:
            self._tip.withdraw()
        except Exception:
            pass

