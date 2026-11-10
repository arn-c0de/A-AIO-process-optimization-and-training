"""Theme-aware hover tooltip (no external deps).

Usage::

    ToolTip(widget, text="Static help")
    ToolTip(widget, text_func=lambda: var.get())          # dynamic text
    ToolTip(widget, help_key="pipeline.run_mode")         # text from gui/help/texts.py
    ToolTip(widget, help_key="x", text_func=lambda: v())  # help + current value

``tooltip(widget, "text")`` is a short alias. A tooltip also publishes its text as
a hover hint (``<<HoverHint>>`` virtual event data via ``set_hover_hint``) so the
status bar can show it.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
import contextlib

_WRAP_PX = 360
_hint_sink: Callable[[str], None] | None = None


def set_hover_hint_sink(sink: Callable[[str], None] | None) -> None:
    """Register a callback (e.g. the status bar) that shows hover hints."""
    global _hint_sink
    _hint_sink = sink


def _emit_hint(text: str) -> None:
    if _hint_sink is not None:
        with contextlib.suppress(Exception):
            _hint_sink(text)


class ToolTip:
    """A small hover tooltip for a widget."""

    def __init__(
        self,
        widget: tk.Widget,
        *,
        text: str | None = None,
        text_func: Callable[[], str] | None = None,
        help_key: str | None = None,
        delay_ms: int = 450,
    ) -> None:
        self.widget = widget
        self.text = text
        self.text_func = text_func
        self.help_key = help_key
        self.delay_ms = int(delay_ms)

        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        self._label: tk.Label | None = None

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")
        # Allow later lookup (e.g. tests: every button has help).
        try:
            widget._simsim_tooltip = self  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------ text
    def current_text(self) -> str:
        parts: list[str] = []
        if self.help_key:
            from gui.help.texts import help_text

            ht = help_text(self.help_key)
            if ht:
                parts.append(ht)
        if self.text:
            parts.append(str(self.text))
        if self.text_func is not None:
            try:
                dyn = str(self.text_func() or "").strip()
            except Exception:
                dyn = ""
            if dyn and dyn not in parts:
                parts.append(dyn if not parts else f"Current: {dyn}")
        return "\n\n".join(p.strip() for p in parts if p and p.strip())

    # ---------------------------------------------------------------- events
    def _on_enter(self, _evt: tk.Event) -> None:
        self._schedule()
        text = self.current_text()
        if text:
            _emit_hint(text.split("\n", 1)[0])

    def _on_leave(self, _evt: tk.Event) -> None:
        self._cancel()
        self._hide()
        _emit_hint("")

    def _on_motion(self, _evt: tk.Event) -> None:
        if self._tip is not None:
            try:
                x, y = self.widget.winfo_pointerxy()
                self._tip.geometry(f"+{x + 14}+{y + 18}")
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
        with contextlib.suppress(Exception):
            self.widget.after_cancel(self._after_id)
        self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        text = self.current_text()
        if not text:
            return
        from gui.theme.style import font, palette

        pal = palette(self.widget)
        if self._tip is None or not self._tip.winfo_exists():
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            with contextlib.suppress(Exception):
                self._tip.attributes("-topmost", True)
            self._label = tk.Label(self._tip, justify="left", wraplength=_WRAP_PX, borderwidth=0)
            self._label.pack(ipadx=8, ipady=5)
        assert self._label is not None
        self._tip.configure(background=pal.tooltip_bg)
        self._label.configure(text=text, background=pal.tooltip_bg, foreground=pal.tooltip_text, font=font("Small"))
        try:
            x, y = self.widget.winfo_pointerxy()
            self._tip.geometry(f"+{x + 14}+{y + 18}")
            self._tip.deiconify()
        except Exception:
            self._hide()

    def _hide(self) -> None:
        if self._tip is None:
            return
        with contextlib.suppress(Exception):
            self._tip.withdraw()


def tooltip(widget: tk.Widget, text: str | Callable[[], str] | None = None, *, help_key: str | None = None) -> ToolTip:
    """Attach a tooltip: ``text`` may be a string or a zero-arg callable."""
    if callable(text):
        return ToolTip(widget, text_func=text, help_key=help_key)
    return ToolTip(widget, text=text, help_key=help_key)
