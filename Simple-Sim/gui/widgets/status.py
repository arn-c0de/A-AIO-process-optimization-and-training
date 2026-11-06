"""Global status bar: message (with level + auto-clear), hover hints, task progress, system stats."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from gui.theme.tokens import SPACE

_LEVEL_STYLE = {"info": "Status.TLabel", "success": "StatusSuccess.TLabel",
                "warning": "StatusWarning.TLabel", "error": "StatusDanger.TLabel"}


class StatusBar(ttk.Frame):
    def __init__(self, parent: tk.Misc, **kw: Any) -> None:
        super().__init__(parent, style="Status.TFrame", padding=(SPACE.M, SPACE.XS + 1), **kw)
        self._define_styles()
        self.var_msg = tk.StringVar(value="Ready")
        self.var_hint = tk.StringVar(value="")
        self.var_task = tk.StringVar(value="")
        self.var_stats = tk.StringVar(value="")
        self._msg = ttk.Label(self, textvariable=self.var_msg, style="Status.TLabel")
        self._msg.pack(side="left")
        self._hint = ttk.Label(self, textvariable=self.var_hint, style="Status.TLabel")
        ttk.Label(self, textvariable=self.var_stats, style="Status.TLabel").pack(side="right")
        self._task_box = ttk.Frame(self, style="Status.TFrame")
        self._task_box.pack(side="right", padx=(0, SPACE.L))
        self._task_lbl = ttk.Label(self._task_box, textvariable=self.var_task, style="Status.TLabel")
        self._task_lbl.pack(side="left", padx=(0, SPACE.S))
        self._bar = ttk.Progressbar(self._task_box, length=90, mode="indeterminate")
        self._clear_id: str | None = None
        self._last_msg = ("Ready", "info")
        self._tasks: dict[int, Any] = {}

    def _define_styles(self) -> None:
        from gui.theme.style import palette

        pal = palette(self)
        st = ttk.Style(self)
        st.configure("StatusSuccess.TLabel", background=pal.surface_alt, foreground=pal.success)
        st.configure("StatusWarning.TLabel", background=pal.surface_alt, foreground=pal.warning)
        st.configure("StatusDanger.TLabel", background=pal.surface_alt, foreground=pal.danger)

    def retheme(self) -> None:
        self._define_styles()

    # ------------------------------------------------------------------ messages
    def show(self, text: str, level: str = "info", *, timeout_ms: int | None = None) -> None:
        """Show a message. ``success``/``info`` clear after 6 s, warnings/errors stay until replaced."""
        self.var_msg.set(text or "Ready")
        self._msg.configure(style=_LEVEL_STYLE.get(level, "Status.TLabel"))
        self._last_msg = (text, level)
        if self._clear_id is not None:
            try:
                self.after_cancel(self._clear_id)
            except Exception:
                pass
            self._clear_id = None
        if timeout_ms is None:
            timeout_ms = 6000 if level in ("info", "success") else 0
        if timeout_ms:
            self._clear_id = self.after(timeout_ms, lambda: self.show("Ready", "info", timeout_ms=0))

    def hint(self, text: str) -> None:
        """Hover hint: temporarily replaces the message while the pointer is over a control."""
        if text:
            self._msg.pack_forget()
            self.var_hint.set(text)
            self._hint.pack(side="left")
        else:
            self._hint.pack_forget()
            self.var_hint.set("")
            self._msg.pack(side="left")

    # --------------------------------------------------------------------- tasks
    def task_started(self, task: Any) -> None:
        self._tasks[task.id] = task
        self._render_tasks()

    def task_progress(self, task: Any) -> None:
        self._tasks[task.id] = task
        self._render_tasks()

    def task_finished(self, task: Any) -> None:
        self._tasks.pop(task.id, None)
        self._render_tasks()
        if task.state == "error":
            self.show(f"{task.name} failed: {task.error}", "error")
        elif task.state == "cancelled":
            self.show(f"{task.name} cancelled", "warning", timeout_ms=6000)
        else:
            self.show(f"{task.name} finished ({task.elapsed_s:.1f} s)", "success")

    def _render_tasks(self) -> None:
        if not self._tasks:
            self.var_task.set("")
            self._bar.stop()
            self._bar.pack_forget()
            return
        tasks = list(self._tasks.values())
        t = tasks[-1]
        more = f" (+{len(tasks) - 1})" if len(tasks) > 1 else ""
        msg = f" – {t.message}" if t.message else ""
        self.var_task.set(f"{t.name}{msg}{more}")
        if not self._bar.winfo_ismapped():
            self._bar.pack(side="left")
        if t.fraction is None:
            if str(self._bar.cget("mode")) != "indeterminate":
                self._bar.configure(mode="indeterminate")
            self._bar.start(15)
        else:
            self._bar.stop()
            self._bar.configure(mode="determinate", maximum=1.0, value=t.fraction)

    def set_stats(self, text: str) -> None:
        self.var_stats.set(text)
