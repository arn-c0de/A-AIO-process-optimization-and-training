"""Shared dialogs: errors with details, confirmations, text viewer, list picker.

Use these instead of calling ``messagebox`` / building ``Toplevel`` dialogs per tab::

    from gui import dialogs
    if not dialogs.require(ds, parent, "Select a dataset first."): return
    if dialogs.confirm(parent, "Delete dataset", f"Delete {n} images from {ds.name}?", danger=True): ...
    dialogs.show_error(parent, "Training failed", exc)
    picked = dialogs.pick_from_list(parent, "Select datasets", items, multi=True)
"""

from __future__ import annotations

import tkinter as tk
import traceback
from tkinter import messagebox, simpledialog, ttk
from collections.abc import Sequence
from typing import Any

from gui.theme.tokens import SPACE
from gui.utils.tk_safe import quiet


def _toplevel(parent: tk.Misc | None) -> tk.Misc | None:
    try:
        return parent.winfo_toplevel() if parent is not None else None
    except Exception:
        return parent


class _Dialog(tk.Toplevel):
    """Modal, centered, Esc closes, themed background."""

    def __init__(self, parent: tk.Misc | None, title: str, size: tuple[int, int] | None = None) -> None:
        master = _toplevel(parent)
        super().__init__(master)
        from gui.theme.style import palette

        self.configure(background=palette(master).bg)
        self.title(title)
        self.result: Any = None
        if master is not None:
            with quiet():
                self.transient(master)
        self.bind("<Escape>", lambda _e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.body = ttk.Frame(self, padding=SPACE.XL)
        self.body.pack(fill="both", expand=True)
        self._size = size

    def show(self) -> Any:
        self.update_idletasks()
        w, h = self._size or (self.winfo_reqwidth(), self.winfo_reqheight())
        try:
            m = self.master
            x = m.winfo_rootx() + max(0, (m.winfo_width() - w) // 2)
            y = m.winfo_rooty() + max(0, (m.winfo_height() - h) // 3)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            self.geometry(f"{w}x{h}")
        with quiet():
            self.grab_set()
        self.focus_set()
        self.wait_window()
        return self.result

    def buttons(self, specs: Sequence[tuple[str, Any, str | None]], *,
                default: int | None = 0) -> ttk.Frame:
        """Bottom-right button row: ``[(text, command, style), ...]`` (first = rightmost).

        Enter invokes the focused button; when no button has focus it invokes
        ``specs[default]`` (``None`` = Enter does nothing). Destructive dialogs
        pass the index of "Cancel" so Enter never deletes anything by accident.
        """
        bar = ttk.Frame(self.body)
        bar.pack(side="bottom", fill="x", pady=(SPACE.L, 0))
        btns: list[ttk.Button] = []
        for i, (text, cmd, style) in enumerate(specs):  # first = rightmost (primary)
            b = ttk.Button(bar, text=text, command=cmd, **({"style": style} if style else {}))
            b.pack(side="right", padx=(0, SPACE.S if i else 0))
            btns.append(b)

        def on_return(_e: Any = None) -> None:
            focus = self.focus_get()
            if focus in btns:
                focus.invoke()
            elif default is not None and 0 <= default < len(specs):
                specs[default][1]()

        if specs:
            self.bind("<Return>", on_return)
        return bar

    def ok(self, result: Any = True) -> None:
        self.result = result
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()


# ---------------------------------------------------------------- simple messages
def show_info(parent: tk.Misc | None, title: str, message: str) -> None:
    messagebox.showinfo(title, message, parent=_toplevel(parent))


def show_warning(parent: tk.Misc | None, title: str, message: str) -> None:
    messagebox.showwarning(title, message, parent=_toplevel(parent))


def require(condition: Any, parent: tk.Misc | None, message: str, title: str = "Selection Needed") -> bool:
    """Return ``bool(condition)``; when falsy, tell the user what is missing."""
    if condition:
        return True
    messagebox.showinfo(title, message, parent=_toplevel(parent))
    return False


def confirm(parent: tk.Misc | None, title: str, message: str, *, danger: bool = False,
            ok_text: str | None = None) -> bool:
    """Yes/No confirmation. ``danger=True`` uses a red confirm button."""
    if not danger and ok_text is None:
        return bool(messagebox.askyesno(title, message, parent=_toplevel(parent)))
    dlg = _Dialog(parent, title)
    ttk.Label(dlg.body, text=title, style="Heading.TLabel").pack(anchor="w")
    ttk.Label(dlg.body, text=message, wraplength=440, justify="left").pack(anchor="w", pady=(SPACE.M, 0))
    dlg.buttons([
        (ok_text or ("Delete" if danger else "OK"), lambda: dlg.ok(True), "Danger.TButton" if danger else "Accent.TButton"),
        ("Cancel", dlg.cancel, None),
    ], default=1 if danger else 0)  # danger: Enter = Cancel
    return bool(dlg.show())


def ask_string(parent: tk.Misc | None, title: str, prompt: str, initial: str = "") -> str | None:
    return simpledialog.askstring(title, prompt, initialvalue=initial, parent=_toplevel(parent))


def show_error(parent: tk.Misc | None, title: str, error: BaseException | str, *, hint: str | None = None) -> None:
    """Readable error: short message (+ hint), expandable details for exceptions."""
    if isinstance(error, BaseException):
        short = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    else:
        short, details = str(error), ""
    if not details:
        msg = short + (f"\n\n{hint}" if hint else "")
        messagebox.showerror(title, msg, parent=_toplevel(parent))
        return
    dlg = _Dialog(parent, title)
    ttk.Label(dlg.body, text=title, style="Heading.TLabel").pack(anchor="w")
    ttk.Label(dlg.body, text=short, style="Danger.TLabel", wraplength=520, justify="left").pack(anchor="w", pady=(SPACE.M, 0))
    if hint:
        ttk.Label(dlg.body, text=hint, style="Muted.TLabel", wraplength=520, justify="left").pack(anchor="w", pady=(SPACE.S, 0))
    box = ttk.Frame(dlg.body)
    shown = tk.BooleanVar(value=False)

    def toggle() -> None:
        if shown.get():
            box.pack(fill="both", expand=True, pady=(SPACE.M, 0))
        else:
            box.pack_forget()
        # show() fixed the geometry to the collapsed size; drop the explicit
        # size so the window grows/shrinks with the details box (position kept).
        try:
            dlg.update_idletasks()
            w = max(dlg.winfo_width(), dlg.winfo_reqwidth())
            dlg.geometry(f"{w}x{dlg.winfo_reqheight()}")
        except Exception:
            pass

    ttk.Checkbutton(dlg.body, text="Show Details", variable=shown, command=toggle, style="Toolbutton").pack(
        anchor="w", pady=(SPACE.M, 0))
    from gui.widgets.data_views import LogView

    log = LogView(box, height=12)
    log.pack(fill="both", expand=True)
    log.append(details)

    def copy() -> None:
        dlg.clipboard_clear()
        dlg.clipboard_append(f"{title}\n{short}\n\n{details}")

    dlg.buttons([("Close", dlg.cancel, "Accent.TButton"), ("Copy Details", copy, None)])
    dlg.show()


# --------------------------------------------------------------------- text viewer
def show_text(parent: tk.Misc | None, title: str, text: str, *, mono: bool = True,
              size: tuple[int, int] = (680, 520), modal: bool = False) -> tk.Toplevel:
    """Read-only text window (reports, info dumps, help)."""
    dlg = _Dialog(parent, title, size=size)
    if mono:
        from gui.widgets.data_views import LogView

        view = LogView(dlg.body, height=20, wrap="none")
        view.pack(fill="both", expand=True)
        view.append(text, level="")
    else:
        lbl = ttk.Label(dlg.body, text=text, wraplength=size[0] - 60, justify="left")
        lbl.pack(fill="both", expand=True, anchor="nw")

    def copy() -> None:
        dlg.clipboard_clear()
        dlg.clipboard_append(text)

    dlg.buttons([("Close", dlg.cancel, "Accent.TButton"), ("Copy", copy, None)])
    if modal:
        dlg.show()
    else:
        dlg.update_idletasks()
        try:
            m = dlg.master
            x = m.winfo_rootx() + max(0, (m.winfo_width() - size[0]) // 2)
            y = m.winfo_rooty() + max(0, (m.winfo_height() - size[1]) // 3)
            dlg.geometry(f"{size[0]}x{size[1]}+{x}+{y}")
        except Exception:
            dlg.geometry(f"{size[0]}x{size[1]}")
    return dlg


# --------------------------------------------------------------------- list picker
def pick_from_list(
    parent: tk.Misc | None,
    title: str,
    items: Sequence[str],
    *,
    multi: bool = False,
    selected: Sequence[str] = (),
    prompt: str = "",
    labels: Sequence[str] | None = None,
    size: tuple[int, int] = (520, 460),
    ok_text: str = "OK",
) -> list[str] | None:
    """Pick one or several entries with a search filter. Returns ``None`` on cancel."""
    items = list(items)
    shown_labels = list(labels) if labels is not None else items
    dlg = _Dialog(parent, title, size=size)
    if prompt:
        ttk.Label(dlg.body, text=prompt, style="Muted.TLabel", wraplength=size[0] - 60).pack(anchor="w", pady=(0, SPACE.M))
    var_q = tk.StringVar()
    search = ttk.Entry(dlg.body, textvariable=var_q)
    search.pack(fill="x")
    ttk.Label(dlg.body, text="Type to filter", style="Small.TLabel").pack(anchor="w", pady=(SPACE.XS, SPACE.S))
    box = ttk.Frame(dlg.body)
    box.pack(fill="both", expand=True)
    lb = tk.Listbox(box, selectmode="extended" if multi else "browse", exportselection=False, activestyle="none")
    sb = ttk.Scrollbar(box, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    lb.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    chosen: set[str] = set(selected)
    visible: list[int] = []
    var_count = tk.StringVar()

    def sync_from_listbox() -> None:
        sel = {visible[i] for i in lb.curselection()}
        if not multi:
            # Single mode: the clicked entry replaces the choice, including a
            # preselected entry that the filter currently hides.
            if sel:
                chosen.clear()
                chosen.update(items[i] for i in sel)
            return
        for idx in visible:
            if idx in sel:
                chosen.add(items[idx])
            else:
                chosen.discard(items[idx])
        var_count.set(f"{len(chosen)} selected" if multi else "")

    def render(*_a: Any) -> None:
        q = var_q.get().strip().lower()
        lb.delete(0, "end")
        visible.clear()
        for i, lab in enumerate(shown_labels):
            if q and q not in lab.lower():
                continue
            visible.append(i)
            lb.insert("end", lab)
            if items[i] in chosen:
                lb.selection_set("end")
        var_count.set(f"{len(chosen)} selected" if multi else "")

    lb.bind("<<ListboxSelect>>", lambda _e: sync_from_listbox())
    var_q.trace_add("write", render)
    render()

    def accept() -> None:
        sync_from_listbox()
        if multi:
            dlg.ok([it for it in items if it in chosen])
        else:
            sel = lb.curselection()
            if sel:
                dlg.ok([items[visible[sel[0]]]])
            else:  # nothing visible selected: keep the (possibly hidden) choice
                dlg.ok([next(it for it in items if it in chosen)] if chosen else None)

    lb.bind("<Double-Button-1>", lambda _e: accept())
    bar = dlg.buttons([(ok_text, accept, "Accent.TButton"), ("Cancel", dlg.cancel, None)])
    if multi:
        ttk.Button(bar, text="All", command=lambda: (lb.selection_set(0, "end"), sync_from_listbox())).pack(side="left")
        ttk.Button(bar, text="None", command=lambda: (lb.selection_clear(0, "end"), sync_from_listbox())).pack(
            side="left", padx=(SPACE.S, 0))
        ttk.Label(bar, textvariable=var_count, style="Muted.TLabel").pack(side="left", padx=(SPACE.M, 0))
    search.focus_set()
    return dlg.show()


# Public base for custom dialogs (modal via ``.show()``; ``.body`` frame, ``.buttons([...])``).
Dialog = _Dialog
