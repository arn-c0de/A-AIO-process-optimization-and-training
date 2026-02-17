from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

ApplyCallback = Callable[[str, str], None]


class YamlEditorPanel(ttk.Frame):
    """Right-side YAML editors for profile + run with auto-apply."""

    def __init__(self, master: tk.Misc, on_apply: ApplyCallback):
        super().__init__(master)
        self._on_apply = on_apply
        self._suppress = False
        self._pending_after: Optional[str] = None

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)

        self.profile_tab = ttk.Frame(self.tabs)
        self.run_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.profile_tab, text="Profile YAML")
        self.tabs.add(self.run_tab, text="Run YAML")

        self.profile_error = tk.StringVar(value="")
        self.run_error = tk.StringVar(value="")

        self.profile_text = self._build_editor(self.profile_tab, "profile", self.profile_error)
        self.run_text = self._build_editor(self.run_tab, "run", self.run_error)

    def _build_editor(self, parent: ttk.Frame, target: str, err_var: tk.StringVar) -> tk.Text:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(controls, text="Apply", command=lambda: self._apply_now(target)).pack(side="left")
        ttk.Label(controls, textvariable=err_var, foreground="#b00020").pack(side="left", padx=8)

        wrap = ttk.Frame(parent)
        wrap.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        y_scroll = ttk.Scrollbar(wrap, orient="vertical")
        text = tk.Text(wrap, wrap="none", undo=True, yscrollcommand=y_scroll.set)
        y_scroll.configure(command=text.yview)

        text.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")

        text.bind("<KeyRelease>", lambda _e, t=target: self._schedule_apply(t))
        return text

    def _schedule_apply(self, target: str) -> None:
        if self._suppress:
            return
        if self._pending_after:
            self.after_cancel(self._pending_after)
        self._pending_after = self.after(350, lambda: self._apply_now(target))

    def _apply_now(self, target: str) -> None:
        if self._suppress:
            return
        text = self.get_text(target)
        self._on_apply(target, text)

    def set_text(self, target: str, text: str) -> None:
        widget = self.profile_text if target == "profile" else self.run_text
        self._suppress = True
        try:
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
        finally:
            self._suppress = False

    def set_error(self, target: str, error: str) -> None:
        if target == "profile":
            self.profile_error.set(error)
        else:
            self.run_error.set(error)

    def clear_error(self, target: str) -> None:
        self.set_error(target, "")

    def get_text(self, target: str) -> str:
        widget = self.profile_text if target == "profile" else self.run_text
        return widget.get("1.0", "end-1c")

    def apply_theme(self, *, dark: bool) -> None:
        if dark:
            bg = "#11161c"
            fg = "#f5f7fa"
            insert_bg = "#f5f7fa"
        else:
            bg = "#ffffff"
            fg = "#000000"
            insert_bg = "#000000"
        for w in (self.profile_text, self.run_text):
            w.configure(bg=bg, fg=fg, insertbackground=insert_bg)
