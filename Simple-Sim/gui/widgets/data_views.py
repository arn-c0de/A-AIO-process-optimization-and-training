"""Data views: Treeview with scrollbars, log view."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from collections.abc import Sequence
from typing import Any

from gui.theme.style import font, on_theme_changed, palette
from gui.theme.tokens import Palette

# Column spec: (id, heading, width, anchor) – width/anchor optional.
ColumnSpec = tuple


class ScrolledTree(ttk.Frame):
    """``ttk.Treeview`` with vertical (+ optional horizontal) scrollbar.

    ``columns``: ``[("name", "Name", 200, "w"), ("acc", "Accuracy", 80, "e"), ...]``.
    With ``show_tree=True`` the #0 tree column is shown (``tree_heading`` sets its title).
    ``sortable=True`` sorts by column on heading click (numeric when possible).
    Tags ``group``, ``muted``, ``success``, ``warning``, ``danger`` and ``odd`` are themed.
    """

    def __init__(
        self,
        parent: tk.Misc,
        columns: Sequence[ColumnSpec] = (),
        *,
        show_tree: bool = False,
        tree_heading: str = "",
        tree_width: int = 220,
        height: int = 10,
        selectmode: str = "browse",
        hscroll: bool = False,
        sortable: bool = True,
        striped: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(parent, **kw)
        ids = [c[0] for c in columns]
        show = "tree headings" if show_tree else "headings"
        self.tree = ttk.Treeview(self, columns=ids, show=show, height=height, selectmode=selectmode)
        self._striped = striped
        if show_tree:
            self.tree.heading("#0", text=tree_heading, anchor="w")
            self.tree.column("#0", width=tree_width, stretch=True)
        for spec in columns:
            cid, heading = spec[0], spec[1]
            width = spec[2] if len(spec) > 2 else 120
            anchor = spec[3] if len(spec) > 3 else "w"
            cmd = (lambda c=cid: self.sort_by(c)) if sortable else ""
            self.tree.heading(cid, text=heading, anchor=anchor, command=cmd)
            self.tree.column(cid, width=width, anchor=anchor, stretch=True)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        if hscroll:
            hbar = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
            self.tree.configure(xscrollcommand=hbar.set)
            hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._sort_state: dict[str, bool] = {}
        self._last_top: str | None = None  # last inserted top-level row (O(1) striping)
        self._apply_tags(palette(self))
        on_theme_changed(self, self._apply_tags)

    def _apply_tags(self, pal: Palette) -> None:
        t = self.tree
        t.tag_configure("group", font=font("BodyBold"))
        t.tag_configure("muted", foreground=pal.text_muted)
        t.tag_configure("success", foreground=pal.success)
        t.tag_configure("warning", foreground=pal.warning)
        t.tag_configure("danger", foreground=pal.danger)
        t.tag_configure("odd", background=pal.bg if pal.name == "light" else pal.surface_alt)

    # Convenience passthroughs ----------------------------------------------------------
    def clear(self) -> None:
        self.tree.delete(*self.tree.get_children(""))
        self._last_top = None
        # Rows come back in default order: drop the stale sort marker.
        if self._sort_state:
            self._sort_state = {}
            self._set_sort_marker(None, False)

    def _next_is_odd(self) -> bool:
        last = self._last_top
        if last is not None and self.tree.exists(last) and not self.tree.parent(last):
            # Next row alternates with the last one; O(1) instead of counting children.
            return "odd" not in self.tree.item(last, "tags")
        return len(self.tree.get_children("")) % 2 == 1

    def insert(self, values: Sequence[Any] = (), *, parent: str = "", text: str = "", iid: str | None = None,
               tags: Sequence[str] = (), open: bool = False) -> str:
        tags = list(tags)
        if self._striped and not parent and self._next_is_odd():
            tags.append("odd")
        kw: dict[str, Any] = {"values": list(values), "text": text, "tags": tags, "open": open}
        if iid is not None:
            kw["iid"] = iid
        new = self.tree.insert(parent, "end", **kw)
        if not parent:
            self._last_top = new
        return new

    def selected(self) -> list[str]:
        return list(self.tree.selection())

    def sort_by(self, col: str) -> None:
        desc = not self._sort_state.get(col, False)
        self._sort_state = {col: desc}

        def key(iid: str) -> Any:
            v = self.tree.set(iid, col)
            try:
                return (0, float(str(v).rstrip("%")))
            except ValueError:
                return (1, str(v).lower())

        for parent in ("", *tuple(self.tree.get_children(""))):
            kids = list(self.tree.get_children(parent))
            kids.sort(key=key, reverse=desc)
            for i, iid in enumerate(kids):
                self.tree.move(iid, parent, i)
        top = self.tree.get_children("")
        if self._striped:
            for i, iid in enumerate(top):
                tags = [t for t in self.tree.item(iid, "tags") if t != "odd"]
                if i % 2 == 1:
                    tags.append("odd")
                self.tree.item(iid, tags=tags)
        self._last_top = top[-1] if top else None
        self._set_sort_marker(col, desc)

    def _set_sort_marker(self, col: str | None, desc: bool) -> None:
        for c in self.tree["columns"]:
            text = str(self.tree.heading(c, "text")).rstrip(" ▲▼")
            if c == col:
                text += " ▼" if desc else " ▲"
            self.tree.heading(c, text=text)


class LogView(ttk.Frame):
    """Read-only monospace log with scrollbar, level colouring and a line limit.

    ``append(text)`` auto-detects levels from markers like ``[error]``/``Traceback``,
    ``warning``/``[warn]`` and ``done``/``OK``; ``append(text, level="error")`` forces one.
    """

    def __init__(self, parent: tk.Misc, *, height: int = 10, max_lines: int = 5000, wrap: str = "none", **kw: Any) -> None:
        super().__init__(parent, **kw)
        pal = palette(self)
        self.text = tk.Text(self, wrap=wrap, height=height, font=font("Mono"), borderwidth=0, highlightthickness=0,
                            background=pal.log_bg, foreground=pal.log_text, padx=4, pady=2)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=vbar.set, state="disabled")
        self.text.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        if wrap == "none":
            hbar = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
            self.text.configure(xscrollcommand=hbar.set)
            hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.max_lines = int(max_lines)
        # Own variable per view: a menu checkbutton without one shares a global
        # Tcl variable named after its label across all LogViews.
        self._var_autoscroll = tk.BooleanVar(self, value=True)
        self._apply_tags(pal)
        on_theme_changed(self, self._apply_tags)
        self.text.bind("<Button-3>", self._context_menu)

    @property
    def autoscroll(self) -> bool:
        return bool(self._var_autoscroll.get())

    @autoscroll.setter
    def autoscroll(self, value: bool) -> None:
        self._var_autoscroll.set(bool(value))

    def _apply_tags(self, pal: Palette) -> None:
        self.text.configure(background=pal.log_bg, foreground=pal.log_text, insertbackground=pal.text,
                            selectbackground=pal.select_bg)
        self.text.tag_configure("error", foreground=pal.danger)
        self.text.tag_configure("warning", foreground=pal.warning)
        self.text.tag_configure("success", foreground=pal.success)
        self.text.tag_configure("muted", foreground=pal.text_muted)
        self.text.tag_configure("heading", foreground=pal.accent, font=font("Mono"))

    @staticmethod
    def detect_level(line: str) -> str | None:
        low = line.lower()
        if "traceback" in low or "[error" in low or "error:" in low or "failed" in low or "[popup error]" in low:
            return "error"
        if "warn" in low:
            return "warning"
        if line.lstrip().startswith("===") or line.lstrip().startswith("---"):
            return "heading"
        if "[done]" in low or " done" in low[:40] or low.strip().startswith("ok ") or "success" in low:
            return "success"
        return None

    def append(self, text: str, level: str | None = None) -> None:
        if not text:
            return
        t = self.text
        at_end = t.yview()[1] >= 0.999
        t.configure(state="normal")
        if level is not None:  # "" = plain text, no auto-detection
            t.insert("end", text, (level,) if level else ())
        else:
            for line in text.splitlines(keepends=True):
                lvl = self.detect_level(line)
                t.insert("end", line, (lvl,) if lvl else ())
        # Trim to max_lines.
        try:
            n = int(t.index("end-1c").split(".")[0])
            if n > self.max_lines:
                t.delete("1.0", f"{n - self.max_lines}.0")
        except Exception:
            pass
        t.configure(state="disabled")
        if self.autoscroll and at_end:
            t.see("end")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def get(self) -> str:
        return self.text.get("1.0", "end-1c")

    def copy_all(self) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(self.get())
        except Exception:
            pass

    def _context_menu(self, event: tk.Event) -> None:
        m = tk.Menu(self, tearoff=False)
        m.add_command(label="Copy Selection", command=lambda: self.text.event_generate("<<Copy>>"))
        m.add_command(label="Copy All", command=self.copy_all)
        m.add_separator()
        m.add_checkbutton(label="Auto-Scroll", variable=self._var_autoscroll, onvalue=True, offvalue=False)
        m.add_command(label="Clear", command=self.clear)
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()
class LogViewHost:
    """Mixin for tab UIs that own a ``log_view`` (:class:`LogView`): the tab's ``append_log`` entry point."""

    log_view: LogView

    def append_log(self, s: str) -> None:
        self.log_view.append(s)


