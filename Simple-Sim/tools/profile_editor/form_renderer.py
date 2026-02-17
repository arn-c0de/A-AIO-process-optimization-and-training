from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import tkinter as tk
from tkinter import ttk

from .state_store import flatten_leaf_paths

FieldCallback = Callable[[str, Tuple[str, ...], str], None]


class FormRenderer(ttk.Frame):
    """Scrollable dynamic form generated from YAML leaves."""

    def __init__(self, master: tk.Misc, target: str, on_field_commit: FieldCallback):
        super().__init__(master)
        self.target = target
        self._on_field_commit = on_field_commit
        self._entries: Dict[Tuple[str, ...], ttk.Entry] = {}

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.info_label = ttk.Label(self.inner, text="No data loaded", foreground="#666")
        self.info_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self.canvas.itemconfig(self.window_id, width=event.width)

    def set_data(self, data: Dict[str, Any]) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self._entries.clear()

        leaves = flatten_leaf_paths(data or {})
        if not leaves:
            self.info_label = ttk.Label(self.inner, text="No editable fields", foreground="#666")
            self.info_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)
            return

        leaves.sort(key=lambda item: item[0])

        row = 0
        last_group = None
        for path, value in leaves:
            group = path[0] if path else "root"
            if group != last_group:
                ttk.Separator(self.inner, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 6))
                row += 1
                ttk.Label(self.inner, text=group, font=("TkDefaultFont", 10, "bold")).grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4)
                )
                row += 1
                last_group = group

            label = ".".join(path[1:]) if len(path) > 1 else path[0]
            ttk.Label(self.inner, text=label).grid(row=row, column=0, sticky="w", padx=(8, 6), pady=2)
            entry = ttk.Entry(self.inner)
            entry.insert(0, self._to_form_text(value))
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=2)
            entry.bind("<Return>", lambda _e, p=path, w=entry: self._commit_entry(p, w))
            entry.bind("<FocusOut>", lambda _e, p=path, w=entry: self._commit_entry(p, w))
            self._entries[path] = entry
            row += 1

        self.inner.columnconfigure(1, weight=1)

    def _commit_entry(self, path: Tuple[str, ...], widget: ttk.Entry) -> None:
        self._on_field_commit(self.target, path, widget.get())

    def apply_theme(self, *, dark: bool) -> None:
        bg = "#1f2329" if dark else "#ffffff"
        self.canvas.configure(bg=bg)

    @staticmethod
    def _to_form_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        return str(value)
