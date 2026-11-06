"""Form helpers: aligned label/field grids, path pickers, inline validation."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from collections.abc import Callable, Sequence
from typing import Any

from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip


def set_invalid(widget: ttk.Widget, invalid: bool) -> None:
    """Toggle the red ``Invalid.*`` outline on a ttk Entry/Combobox."""
    try:
        base = widget.winfo_class()  # "TEntry" / "TCombobox"
        widget.configure(style=f"Invalid.{base}" if invalid else base)
    except Exception:
        pass


class FormGrid(ttk.Frame):
    """Two-column form: right-aligned muted labels, stretching fields.

    ``row(label, widget_or_factory, help_key=...)`` returns the field widget.
    A factory is called with the grid as parent, so fields can be created inline::

        form.row("Config", lambda p: ttk.Entry(p, textvariable=var))
    """

    def __init__(self, parent: tk.Misc, *, style: str = "TFrame", label_width: int = 0, **kw: Any) -> None:
        super().__init__(parent, style=style, **kw)
        self.columnconfigure(1, weight=1)
        self._r = 0
        self._label_style = "CardMuted.TLabel" if style == "Card.TFrame" else "Field.TLabel"
        self._label_width = label_width
        self.labels: dict[str, ttk.Label] = {}
        self.hints: dict[str, ttk.Label] = {}

    def row(
        self,
        label: str,
        field: tk.Widget | Callable[[tk.Misc], tk.Widget],
        *,
        help_key: str | None = None,
        tooltip: str | None = None,
        hint: str | None = None,
        sticky: str = "ew",
    ) -> tk.Widget:
        widget = field(self) if callable(field) and not isinstance(field, tk.Misc) else field
        lbl = ttk.Label(self, text=label, style=self._label_style, anchor="e",
                        **({"width": self._label_width} if self._label_width else {}))
        lbl.grid(row=self._r, column=0, sticky="e", padx=(0, SPACE.M), pady=SPACE.XS + 1)
        widget.grid(row=self._r, column=1, sticky=sticky, pady=SPACE.XS + 1)  # type: ignore[union-attr]
        if help_key or tooltip:
            ToolTip(lbl, text=tooltip, help_key=help_key)
            ToolTip(widget, text=tooltip, help_key=help_key)  # type: ignore[arg-type]
        self.labels[label] = lbl
        self._r += 1
        if hint is not None:
            h = ttk.Label(self, text=hint, style="Small.TLabel" if self._label_style == "Field.TLabel" else "CardMuted.TLabel")
            h.grid(row=self._r, column=1, sticky="w")
            self.hints[label] = h
            self._r += 1
        return widget  # type: ignore[return-value]

    def full(self, widget: tk.Widget | Callable[[tk.Misc], tk.Widget], *, pady: int = SPACE.XS + 1) -> tk.Widget:
        """A widget spanning both columns (e.g. a checkbox row)."""
        w = widget(self) if callable(widget) and not isinstance(widget, tk.Misc) else widget
        w.grid(row=self._r, column=0, columnspan=2, sticky="ew", pady=pady)  # type: ignore[union-attr]
        self._r += 1
        return w  # type: ignore[return-value]

    def inline(self, parent_style: str | None = None) -> ttk.Frame:
        """A horizontal frame to put several small widgets into one field cell."""
        return ttk.Frame(self, style=parent_style or self.cget("style") or "TFrame")


class PathPicker(ttk.Frame):
    """Entry + Browse button for a file or directory.

    ``relative_to``: chosen paths inside this root are stored relative (as in configs).
    ``must_exist``: entry turns red while the path does not exist.
    """

    def __init__(
        self,
        parent: tk.Misc,
        variable: tk.StringVar,
        *,
        mode: str = "file",  # "file" | "dir" | "save"
        title: str = "Choose path",
        filetypes: Sequence[tuple[str, str]] = (("All files", "*"),),
        relative_to: Path | None = None,
        must_exist: bool = False,
        width: int = 30,
        on_change: Callable[[str], None] | None = None,
        style: str = "TFrame",
        **kw: Any,
    ) -> None:
        super().__init__(parent, style=style, **kw)
        self.var = variable
        self.mode = mode
        self.title = title
        self.filetypes = list(filetypes)
        self.relative_to = Path(relative_to) if relative_to else None
        self.must_exist = must_exist
        self.on_change = on_change
        self.entry = ttk.Entry(self, textvariable=variable, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.button = ttk.Button(self, text="Browse…", command=self.browse)
        self.button.pack(side="left", padx=(SPACE.S, 0))
        ToolTip(self.button, help_key="common.browse")
        if must_exist:
            variable.trace_add("write", lambda *_: self.validate())
            self.validate()

    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute() and self.relative_to is not None:
            p = self.relative_to / p
        return p

    def validate(self) -> bool:
        value = self.var.get().strip()
        ok = bool(value) and self._resolve(value).exists()
        set_invalid(self.entry, not ok)
        return ok

    def browse(self) -> None:
        cur = self.var.get().strip()
        start = self._resolve(cur) if cur else (self.relative_to or Path.cwd())
        initialdir = str(start if start.is_dir() else start.parent) if start else None
        if self.mode == "dir":
            chosen = filedialog.askdirectory(parent=self, title=self.title, initialdir=initialdir)
        elif self.mode == "save":
            chosen = filedialog.asksaveasfilename(parent=self, title=self.title, initialdir=initialdir,
                                                  filetypes=self.filetypes)
        else:
            chosen = filedialog.askopenfilename(parent=self, title=self.title, initialdir=initialdir,
                                                filetypes=self.filetypes)
        if not chosen:
            return
        p = Path(chosen)
        if self.relative_to is not None:
            try:
                p = p.resolve().relative_to(self.relative_to.resolve())
            except ValueError:
                pass
        self.var.set(str(p))
        if self.on_change is not None:
            self.on_change(str(p))
