"""Layout building blocks: cards, collapsible sections, toolbars, button bars."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from collections.abc import Callable, Sequence
from typing import Any

from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip
import contextlib

# Unicode glyphs used for icon buttons. One place, consistent everywhere.
ICONS = {
    "refresh": "↻",   # ↻
    "add": "+",
    "info": "i",
    "help": "?",
    "prev": "\u2039",      # single left angle quote
    "next": "\u203a",      # single right angle quote
    "browse": "…",    # …
    "play": "▶",      # ▶
    "stop": "■",      # ■
    "menu": "▾",      # ▾
    "expanded": "▾",  # ▾
    "collapsed": "▸", # ▸
    "close": "✕",     # ✕
}


def icon_button(
    parent: tk.Misc,
    icon: str,
    command: Callable[[], Any] | None = None,
    *,
    help_key: str | None = None,
    tooltip: str | None = None,
    **kw: Any,
) -> ttk.Button:
    """Compact square button showing an icon glyph; always gets a tooltip."""
    btn = ttk.Button(parent, text=ICONS.get(icon, icon), command=command, style="Icon.TButton", width=2, **kw)
    if help_key or tooltip:
        ToolTip(btn, text=tooltip, help_key=help_key)
    return btn


def _help_popup(parent: tk.Misc, title: str, text: str) -> None:
    from gui.dialogs import show_text

    show_text(parent, title, text, mono=False, size=(460, 260))


class Section(ttk.Frame):
    """Bordered card with a heading row; content is packed into the section itself.

    ``help_key`` adds a "?" button that shows ``SECTION_HELP[help_key]``.
    ``section.actions`` is a frame on the right side of the heading for extra buttons.
    Children must use ``pack`` (the heading row is packed first).
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        help_key: str | None = None,
        padding: int = SPACE.L,
        **kw: Any,
    ) -> None:
        super().__init__(parent, style="Section.TFrame", padding=padding, **kw)
        self.header = ttk.Frame(self, style="Card.TFrame")
        self.header.pack(fill="x", pady=(0, SPACE.S))
        self.title_label = ttk.Label(self.header, text=title, style="CardHeading.TLabel")
        self.title_label.pack(side="left")
        self.actions = ttk.Frame(self.header, style="Card.TFrame")
        self.actions.pack(side="right")
        if help_key:
            from gui.help.texts import section_help

            text = section_help(help_key)
            if text:
                b = icon_button(self.actions, "help", lambda: _help_popup(self, title, text), tooltip="What is this?")
                b.pack(side="right", padx=(SPACE.S, 0))
        self.body = self


class CollapsibleSection(ttk.Frame):
    """Header with ▸/▾ toggle; the ``body`` frame is shown/hidden.

    ``state_key`` + ``store`` (SettingsStore) persist the open/closed state.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        opened: bool = False,
        store: Any = None,
        state_key: str | None = None,
        style: str = "TFrame",
        **kw: Any,
    ) -> None:
        super().__init__(parent, style=style, **kw)
        self._store = store
        self._key = state_key
        if store is not None and state_key:
            saved = store.get(state_key)
            if isinstance(saved, bool):
                opened = saved
        self._open = tk.BooleanVar(value=opened)
        self._title = title
        self._btn = ttk.Checkbutton(self, text=self._label(), variable=self._open, command=self._apply,
                                    style="Card.Toolbutton" if style == "Card.TFrame" else "Toolbutton",
                                    takefocus=True)
        self._btn.pack(fill="x", anchor="w")
        self.body = ttk.Frame(self, style=style, padding=(SPACE.L, SPACE.XS, 0, SPACE.S))
        self._apply(save=False)

    def _label(self) -> str:
        return f"{ICONS['expanded'] if self._open.get() else ICONS['collapsed']}  {self._title}"

    def _apply(self, save: bool = True) -> None:
        self._btn.configure(text=self._label())
        if self._open.get():
            self.body.pack(fill="x", expand=False)
        else:
            self.body.pack_forget()
        if save and self._store is not None and self._key:
            self._store.set(self._key, bool(self._open.get()))
            with contextlib.suppress(Exception):
                self._store.schedule_save(self)

    def set_open(self, opened: bool) -> None:
        self._open.set(bool(opened))
        self._apply()


class Toolbar(ttk.Frame):
    """Horizontal bar of controls with uniform spacing (``add`` packs left, ``add_right`` right)."""

    def __init__(self, parent: tk.Misc, *, style: str = "TFrame", gap: int = SPACE.S, **kw: Any) -> None:
        super().__init__(parent, style=style, **kw)
        self._gap = gap
        self._n_left = 0
        self._n_right = 0

    def add(self, widget: tk.Widget, *, expand: bool = False) -> tk.Widget:
        widget.pack(side="left", padx=(self._gap if self._n_left else 0, 0), fill="x" if expand else None, expand=expand)
        self._n_left += 1
        return widget

    def add_right(self, widget: tk.Widget) -> tk.Widget:
        widget.pack(side="right", padx=(0, self._gap if self._n_right else 0))
        self._n_right += 1
        return widget

    def separator(self) -> None:
        sep = ttk.Separator(self, orient="vertical")
        sep.pack(side="left", fill="y", padx=SPACE.M, pady=SPACE.XS)

    def button(self, text: str, command: Callable[[], Any], *, help_key: str | None = None,
               tooltip: str | None = None, style: str | None = None, right: bool = False, **kw: Any) -> ttk.Button:
        btn = ttk.Button(self, text=text, command=command, **({"style": style} if style else {}), **kw)
        if help_key or tooltip:
            ToolTip(btn, text=tooltip, help_key=help_key)
        return self.add_right(btn) if right else self.add(btn)  # type: ignore[return-value]

    def icon(self, icon: str, command: Callable[[], Any], *, help_key: str | None = None,
             tooltip: str | None = None, right: bool = False) -> ttk.Button:
        btn = icon_button(self, icon, command, help_key=help_key, tooltip=tooltip)
        return self.add_right(btn) if right else self.add(btn)  # type: ignore[return-value]


class ButtonBar(Toolbar):
    """Row of buttons from a spec list: ``[(text, command, {help_key/style/...}), ...]``.

    Buttons are reachable as ``bar.buttons[text]``.
    """

    def __init__(self, parent: tk.Misc, specs: Sequence[tuple] = (), **kw: Any) -> None:
        super().__init__(parent, **kw)
        self.buttons: dict[str, ttk.Button] = {}
        for spec in specs:
            if spec is None or spec == "|":
                self.separator()
                continue
            text, cmd, *rest = spec
            opts = dict(rest[0]) if rest else {}
            self.buttons[text] = self.button(text, cmd, **opts)


def menu_button(parent: tk.Misc, text: str, items: Sequence[tuple | None], *, help_key: str | None = None) -> ttk.Menubutton:
    """Drop-down button; ``items`` = ``[(label, command), None (separator), ...]``."""
    mb = ttk.Menubutton(parent, text=text, direction="below")  # the theme draws the arrow
    menu = tk.Menu(mb, tearoff=False)
    for item in items:
        if item is None:
            menu.add_separator()
        else:
            label, cmd = item[0], item[1]
            menu.add_command(label=label, command=cmd)
    mb["menu"] = menu
    if help_key:
        ToolTip(mb, help_key=help_key)
    return mb


class EmptyState(ttk.Frame):
    """Centered hint for empty views: title, text and an optional action button."""

    def __init__(self, parent: tk.Misc, title: str, text: str = "", *, action: str | None = None,
                 command: Callable[[], Any] | None = None, style: str = "TFrame", **kw: Any) -> None:
        super().__init__(parent, style=style, padding=SPACE.XXL, **kw)
        inner = ttk.Frame(self, style=style)
        inner.place(relx=0.5, rely=0.45, anchor="center")
        lbl_style = "CardHeading.TLabel" if style == "Card.TFrame" else "Heading.TLabel"
        ttk.Label(inner, text=title, style=lbl_style, anchor="center").pack()
        self._text = ttk.Label(inner, text=text, style="CardMuted.TLabel" if style == "Card.TFrame" else "Muted.TLabel",
                               anchor="center", justify="center", wraplength=360)
        self._text.pack(pady=(SPACE.S, 0))
        if action and command:
            ttk.Button(inner, text=action, command=command, style="Accent.TButton").pack(pady=(SPACE.L, 0))

    def set_text(self, text: str) -> None:
        self._text.configure(text=text)


class ScrollableFrame(ttk.Frame):
    """Vertically scrollable container; put children into ``.inner``."""

    def __init__(self, parent: tk.Misc, *, style: str = "TFrame", fit_width: bool = True, **kw: Any) -> None:
        super().__init__(parent, style=style, **kw)
        from gui.theme.style import on_theme_changed, palette

        self._fit_width = fit_width
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=palette(self).bg)
        self.canvas._simsim_role = "bg"  # type: ignore[attr-defined]
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style=style)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.inner.bind_all(seq, self._on_wheel, add="+")
        on_theme_changed(self, lambda pal: self.canvas.configure(background=pal.bg))

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self._fit_width:
            req = self.inner.winfo_reqwidth()
            if req and int(self.canvas.cget("width")) != req:
                self.canvas.configure(width=req)

    def _on_wheel(self, event: tk.Event) -> None:
        # Only scroll when the pointer is over this frame.
        try:
            w = self.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return
        while w is not None and w is not self:
            w = getattr(w, "master", None)
        if w is None:
            return
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta * 2, "units")


def keep_pane_visible(paned: ttk.Panedwindow, index: int = 0, pane: tk.Misc | None = None, *,
                      min_width: int = 220) -> None:
    """Make sure a weight-0 pane of a ttk.Panedwindow is not collapsed to width 0.

    ttk places sashes when the paned window is first laid out; if that happens while it is
    still tiny (lazy tab build, hidden page), a weight-0 pane can end up 0 px wide and never
    grow back. This sets the sash to the pane's requested width once the window is mapped
    (and again on the first real resize).
    """
    state = {"done": False}

    def fix(_event: object = None) -> None:
        if state["done"]:
            return
        try:
            total = paned.winfo_width()
            if total <= 1:
                return
            target = pane.winfo_reqwidth() if pane is not None else min_width
            target = max(min_width, min(int(target), total - 200))
            if paned.sashpos(index) < target - 4:
                paned.sashpos(index, target)
            state["done"] = True
        except Exception:
            pass

    paned.bind("<Map>", lambda _e: paned.after_idle(fix), add="+")
    paned.bind("<Configure>", fix, add="+")
    paned.after_idle(fix)
