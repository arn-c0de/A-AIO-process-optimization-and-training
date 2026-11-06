"""Apply the design tokens to a Tk root (ttk styles, named fonts, classic widgets).

Usage::

    from gui.theme.style import apply_theme, palette, on_theme_changed
    apply_theme(root, "dark")
    pal = palette(any_widget)          # active Palette
    on_theme_changed(widget, callback) # re-color custom canvases on toggle

Custom ttk styles provided (use via ``style=...``):

- Frames:   ``Card.TFrame`` (surface), ``Toolbar.TFrame`` (surface_alt), ``Status.TFrame``
- Labels:   ``Title.TLabel``, ``Heading.TLabel``, ``Muted.TLabel``, ``Small.TLabel``,
            ``Success.TLabel``, ``Warning.TLabel``, ``Danger.TLabel``, ``Card.TLabel``,
            ``CardMuted.TLabel``, ``CardHeading.TLabel``, ``Status.TLabel``,
            ``Badge.TLabel``, ``Field.TLabel`` (form labels)
- Buttons:  ``Accent.TButton`` (primary), ``Danger.TButton``, ``Icon.TButton`` (compact)
- Groups:   ``Card.TLabelframe`` (default look of all LabelFrames as well)
- Entries:  ``Invalid.TEntry`` (red outline for inline validation)
"""

from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from collections.abc import Callable

from gui.theme.tokens import FONT_ROLES, SPACE, Palette, font_name, palette_for

_ATTR_PALETTE = "_simsim_palette"
_ATTR_LISTENERS = "_simsim_theme_listeners"

# Old hardcoded canvas backgrounds that get re-colored to ``canvas_bg``.
_LEGACY_CANVAS_BG = {"gray20", "grey20", "#333333", "black", "gray15", "gray25"}


def _toplevel_root(widget: tk.Misc) -> tk.Misc:
    try:
        return widget._root()  # type: ignore[attr-defined]
    except Exception:
        return widget


def palette(widget: tk.Misc | None = None) -> Palette:
    """Active palette for ``widget``'s root (LIGHT before ``apply_theme`` ran)."""
    if widget is None:
        widget = tk._default_root  # type: ignore[attr-defined]
    if widget is None:
        return palette_for("light")
    return getattr(_toplevel_root(widget), _ATTR_PALETTE, None) or palette_for("light")


def is_dark(widget: tk.Misc | None = None) -> bool:
    return palette(widget).name == "dark"


def on_theme_changed(widget: tk.Misc, callback: Callable[[Palette], None]) -> None:
    """Call ``callback(palette)`` after every ``apply_theme`` (until ``widget`` is destroyed)."""
    root = _toplevel_root(widget)
    listeners: list = getattr(root, _ATTR_LISTENERS, None) or []
    listeners.append((widget, callback))
    setattr(root, _ATTR_LISTENERS, listeners)


def font(role: str) -> str:
    """Name of the Tk named font for ``role`` (``"Heading"``, ``"Mono"``, ...)."""
    return font_name(role)


def ui_scale_from_env() -> float | None:
    raw = os.environ.get("SIMPLE_SIM_UI_SCALE", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if 0.5 <= val <= 4.0 else None


def apply_scaling(root: tk.Misc, factor: float | None) -> None:
    """Scale Tk points->pixels by ``factor`` (1.0 = 96 dpi). ``None`` keeps Tk's default."""
    if factor is None:
        return
    try:
        root.tk.call("tk", "scaling", 96.0 / 72.0 * float(factor))
    except Exception:
        pass


def _ensure_fonts(root: tk.Misc) -> None:
    try:
        base = tkfont.nametofont("TkDefaultFont", root=root).actual()
        mono = tkfont.nametofont("TkFixedFont", root=root).actual()
    except Exception:
        base = {"family": "Helvetica"}
        mono = {"family": "Courier"}
    for role, (size, weight, slant, fam) in FONT_ROLES.items():
        family = mono["family"] if fam == "mono" else base["family"]
        name = font_name(role)
        try:
            f = tkfont.nametofont(name, root=root)
            f.configure(family=family, size=size, weight=weight, slant=slant)
        except Exception:
            tkfont.Font(root=root, name=name, family=family, size=size, weight=weight, slant=slant)
    # Body text for the classic default font, so un-styled widgets match.
    try:
        tkfont.nametofont("TkDefaultFont", root=root).configure(size=FONT_ROLES["Body"][0])
        tkfont.nametofont("TkTextFont", root=root).configure(size=FONT_ROLES["Body"][0])
    except Exception:
        pass


def ensure_basic_styles(widget: tk.Misc) -> None:
    """Minimal fallback for windows opened outside the themed app (e.g. the
    standalone preview viewer): create the named fonts and the label styles
    used by shared components, *without* restyling anything else.

    No-op once ``apply_theme`` ran on ``widget``'s root.
    """
    root = _toplevel_root(widget)
    if getattr(root, _ATTR_PALETTE, None) is not None:
        return
    try:
        base = tkfont.nametofont("TkDefaultFont", root=root).actual()
        mono = tkfont.nametofont("TkFixedFont", root=root).actual()
    except Exception:
        base, mono = {"family": "Helvetica"}, {"family": "Courier"}
    existing = set(tkfont.names(root=root))
    for role, (size, weight, slant, fam) in FONT_ROLES.items():
        name = font_name(role)
        if name in existing:
            continue
        family = mono["family"] if fam == "mono" else base["family"]
        try:
            tkfont.Font(root=root, name=name, family=family, size=size, weight=weight, slant=slant)
        except Exception:
            pass
    p = palette_for("light")
    style = ttk.Style(root)
    for name, opts in (
        ("Heading.TLabel", {"font": font_name("Heading")}),
        ("Muted.TLabel", {"foreground": p.text_muted}),
        ("Small.TLabel", {"foreground": p.text_muted, "font": font_name("Small")}),
        ("Success.TLabel", {"foreground": p.success}),
        ("Warning.TLabel", {"foreground": p.warning}),
        ("Danger.TLabel", {"foreground": p.danger}),
    ):
        try:
            if not style.lookup(name, "foreground") and not style.lookup(name, "font"):
                style.configure(name, **opts)
        except Exception:
            pass


def _row_height(root: tk.Misc) -> int:
    try:
        return int(tkfont.nametofont(font_name("Body"), root=root).metrics("linespace") * 1.35)
    except Exception:
        return 20


def apply_theme(root: tk.Misc, mode: str | bool | None = "light") -> Palette:
    """Configure ttk + classic widgets for ``mode`` ("light"/"dark") and notify listeners."""
    pal = palette_for(mode)
    root = _toplevel_root(root)
    setattr(root, _ATTR_PALETTE, pal)
    _ensure_fonts(root)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    body, small, heading, title = font("Body"), font("Small"), font("Heading"), font("Title")
    bold = font("BodyBold")
    p = pal

    style.configure(
        ".",
        background=p.bg,
        foreground=p.text,
        fieldbackground=p.surface,
        bordercolor=p.border,
        darkcolor=p.bg,
        lightcolor=p.bg,
        troughcolor=p.surface_alt,
        focuscolor=p.accent,
        selectbackground=p.select_bg,
        selectforeground=p.text,
        insertcolor=p.text,
        font=body,
    )
    style.map(".", foreground=[("disabled", p.text_muted)])

    # Frames
    style.configure("TFrame", background=p.bg)
    style.configure("Card.TFrame", background=p.surface)
    style.configure("Section.TFrame", background=p.surface, bordercolor=p.border, lightcolor=p.surface,
                    darkcolor=p.surface, relief="solid", borderwidth=1)
    style.configure("Toolbar.TFrame", background=p.surface_alt)
    style.configure("Status.TFrame", background=p.surface_alt)

    # Labels
    style.configure("TLabel", background=p.bg, foreground=p.text)
    style.configure("Title.TLabel", font=title)
    style.configure("Heading.TLabel", font=heading)
    style.configure("Muted.TLabel", foreground=p.text_muted)
    style.configure("Small.TLabel", foreground=p.text_muted, font=small)
    style.configure("Field.TLabel", foreground=p.text_muted)
    style.configure("Success.TLabel", foreground=p.success)
    style.configure("Warning.TLabel", foreground=p.warning)
    style.configure("Danger.TLabel", foreground=p.danger)
    style.configure("BodyBold.TLabel", font=bold)
    style.configure("Accent.TLabel", foreground=p.accent)
    style.configure("Card.TLabel", background=p.surface)
    for _n, _c in (("Success", p.success), ("Warning", p.warning), ("Danger", p.danger), ("Accent", p.accent)):
        style.configure(f"Card{_n}.TLabel", background=p.surface, foreground=_c)
    style.configure("CardMuted.TLabel", background=p.surface, foreground=p.text_muted)
    style.configure("CardHeading.TLabel", background=p.surface, foreground=p.text_muted, font=heading)
    style.configure("Status.TLabel", background=p.surface_alt, foreground=p.text_muted, font=small)
    style.configure("Toolbar.TLabel", background=p.surface_alt, foreground=p.text_muted)
    style.configure("Badge.TLabel", foreground=p.text_muted, font=small, padding=(0, 0, SPACE.M, 0))

    # Buttons: plain and compact. Accent/Danger only tint text + border (no filled blocks).
    # One control height everywhere: buttons, menubuttons, entries, comboboxes, spinboxes.
    pad = (SPACE.L, 1)
    field_pad = (SPACE.S + 1, 1)
    style.configure("TButton", background=p.surface, foreground=p.text, bordercolor=p.border,
                    lightcolor=p.surface, darkcolor=p.surface, padding=pad, relief="flat", font=body,
                    width=0, anchor="center")
    style.map(
        "TButton",
        background=[("disabled", p.bg), ("pressed", p.border), ("active", p.surface_alt)],
        foreground=[("disabled", p.text_muted)],
        bordercolor=[("focus", p.accent), ("active", p.text_muted)],
        lightcolor=[("active", p.surface_alt)],
        darkcolor=[("active", p.surface_alt)],
    )
    for name, fg in (("Accent", p.accent), ("Danger", p.danger)):
        style.configure(f"{name}.TButton", foreground=fg, bordercolor=fg, padding=pad, font=body)
        style.map(
            f"{name}.TButton",
            background=[("disabled", p.bg), ("pressed", p.border), ("active", p.surface_alt)],
            foreground=[("disabled", p.text_muted)],
            bordercolor=[("disabled", p.border), ("focus", fg), ("active", fg)],
            lightcolor=[("active", p.surface_alt)],
            darkcolor=[("active", p.surface_alt)],
        )
    style.configure("Icon.TButton", padding=(SPACE.S + 1, 1), width=0)
    style.configure("TMenubutton", background=p.surface, foreground=p.text, bordercolor=p.border,
                    lightcolor=p.surface, darkcolor=p.surface, arrowcolor=p.text_muted, padding=pad,
                    relief="flat", font=body, arrowsize=8)
    style.map("TMenubutton",
              background=[("disabled", p.bg), ("pressed", p.border), ("active", p.surface_alt)],
              foreground=[("disabled", p.text_muted)],
              bordercolor=[("focus", p.accent), ("active", p.text_muted)],
              lightcolor=[("active", p.surface_alt)], darkcolor=[("active", p.surface_alt)])
    style.configure("Toolbutton", background=p.bg, padding=(SPACE.M, SPACE.XS))
    style.map("Toolbutton", background=[("selected", p.bg), ("active", p.surface_alt)])
    style.configure("Card.Toolbutton", background=p.surface, foreground=p.text_muted, padding=(0, SPACE.XS),
                    font=body, anchor="w")
    style.map("Card.Toolbutton", background=[("selected", p.surface), ("active", p.surface_alt)])

    # Toggles: small square/round indicator, accent mark, same baseline as other controls.
    for cls in ("TCheckbutton", "TRadiobutton"):
        style.configure(cls, background=p.bg, foreground=p.text, font=body, padding=(0, 1),
                        indicatorsize=12, indicatormargin=(0, 0, SPACE.S + 1, 0),
                        indicatorbackground=p.surface, indicatorforeground=p.accent,
                        upperbordercolor=p.text_muted, lowerbordercolor=p.text_muted)
        style.map(cls,
                  background=[("active", p.bg)],
                  foreground=[("disabled", p.text_muted)],
                  indicatorbackground=[("disabled", p.surface_alt), ("pressed", p.surface_alt)],
                  upperbordercolor=[("focus", p.accent), ("active", p.text)],
                  lowerbordercolor=[("focus", p.accent), ("active", p.text)])
        style.configure(f"Card.{cls}", background=p.surface)
        style.map(f"Card.{cls}", background=[("active", p.surface)])

    # Inputs
    for cls in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(cls, fieldbackground=p.surface, foreground=p.text, bordercolor=p.border,
                        lightcolor=p.surface, darkcolor=p.surface, arrowcolor=p.text_muted,
                        padding=field_pad, insertcolor=p.text, arrowsize=11)
        style.map(cls, bordercolor=[("focus", p.accent)], lightcolor=[("focus", p.accent)],
                  fieldbackground=[("readonly", p.surface), ("disabled", p.surface_alt)],
                  foreground=[("disabled", p.text_muted)],
                  selectbackground=[("readonly", p.surface)], selectforeground=[("readonly", p.text)])
    style.configure("Invalid.TEntry", bordercolor=p.danger, lightcolor=p.danger)
    style.configure("Invalid.TCombobox", bordercolor=p.danger, lightcolor=p.danger)

    # Notebook
    style.configure("TNotebook", background=p.bg, bordercolor=p.border, tabmargins=(SPACE.S, SPACE.S, 0, 0))
    style.configure("TNotebook.Tab", background=p.bg, foreground=p.text_muted, bordercolor=p.border,
                    lightcolor=p.bg, padding=(SPACE.L, SPACE.XS))
    style.map("TNotebook.Tab",
              background=[("selected", p.surface), ("active", p.surface_alt)],
              foreground=[("selected", p.text), ("active", p.text)],
              lightcolor=[("selected", p.accent)],
              expand=[("selected", (0, 1, 0, 0))])

    # Sidebar navigation + page notebook without visible tabs.
    style.configure("Sidebar.TFrame", background=p.surface_alt)
    style.configure("Brand.TLabel", background=p.surface_alt, foreground=p.text, font=heading)
    style.configure("BrandSub.TLabel", background=p.surface_alt, foreground=p.text_muted, font=small)
    style.configure("SidebarHeading.TLabel", background=p.surface_alt, foreground=p.text_muted,
                    font=small, padding=(SPACE.M, SPACE.L, SPACE.M, SPACE.XS))
    style.configure("Nav.Toolbutton", background=p.surface_alt, foreground=p.text, anchor="w",
                    padding=(SPACE.M, SPACE.XS + 1), font=body, relief="flat", borderwidth=0)
    style.map("Nav.Toolbutton",
              background=[("selected", p.surface), ("active", p.border)],
              foreground=[("selected", p.text)])
    style.configure("Pages.TNotebook", background=p.bg, borderwidth=0, tabmargins=0)
    style.layout("Pages.TNotebook.Tab", [])
    style.configure("PageDesc.TLabel", foreground=p.text_muted)

    # Group boxes look like cards with a heading.
    style.configure("TLabelframe", background=p.surface, bordercolor=p.border, lightcolor=p.surface,
                    darkcolor=p.surface, relief="solid", borderwidth=1, padding=SPACE.M)
    style.configure("TLabelframe.Label", background=p.surface, foreground=p.text_muted, font=heading)
    style.configure("Card.TLabelframe", background=p.surface)
    style.configure("Card.TLabelframe.Label", background=p.surface, font=heading)

    # Tables
    style.configure("Treeview", background=p.surface, fieldbackground=p.surface, foreground=p.text,
                    bordercolor=p.border, lightcolor=p.surface, darkcolor=p.surface, rowheight=_row_height(root))
    style.map("Treeview", background=[("selected", p.select_bg)], foreground=[("selected", p.text)])
    style.configure("Treeview.Heading", background=p.surface_alt, foreground=p.text_muted, font=font("SmallBold"),
                    bordercolor=p.border, lightcolor=p.surface_alt, darkcolor=p.surface_alt, relief="flat",
                    padding=(SPACE.M, SPACE.XS))
    style.map("Treeview.Heading", background=[("active", p.border)])

    # Misc
    style.configure("TPanedwindow", background=p.bg)
    style.configure("Sash", sashthickness=SPACE.M, gripcount=0, background=p.bg)
    style.configure("TSeparator", background=p.border)
    style.configure("TScrollbar", background=p.surface_alt, troughcolor=p.bg, bordercolor=p.bg,
                    lightcolor=p.surface_alt, darkcolor=p.surface_alt, arrowcolor=p.text_muted, gripcount=0,
                    arrowsize=11)
    style.map("TScrollbar", background=[("active", p.border)])
    style.configure("TProgressbar", background=p.accent, troughcolor=p.surface_alt, bordercolor=p.surface_alt,
                    lightcolor=p.accent, darkcolor=p.accent, thickness=4)
    style.configure("TScale", background=p.surface, troughcolor=p.surface_alt, bordercolor=p.border,
                    lightcolor=p.surface, darkcolor=p.surface, sliderlength=14, sliderthickness=12,
                    troughrelief="flat", gripcount=0)
    style.map("TScale", background=[("active", p.surface_alt)])

    # Classic Tk widgets created later pick these up via the option database.
    root.option_add("*TCombobox*Listbox.background", p.surface)
    root.option_add("*TCombobox*Listbox.foreground", p.text)
    root.option_add("*TCombobox*Listbox.selectBackground", p.select_bg)
    root.option_add("*TCombobox*Listbox.selectForeground", p.text)
    root.option_add("*TCombobox*Listbox.font", body)
    for cls in ("Text", "Listbox"):
        root.option_add(f"*{cls}.background", p.log_bg if cls == "Text" else p.surface)
        root.option_add(f"*{cls}.foreground", p.log_text if cls == "Text" else p.text)
        root.option_add(f"*{cls}.selectBackground", p.select_bg)
        root.option_add(f"*{cls}.selectForeground", p.text)
        root.option_add(f"*{cls}.highlightThickness", 0)
        root.option_add(f"*{cls}.borderWidth", 0)
    root.option_add("*Text.insertBackground", p.text)
    root.option_add("*Text.font", font("Mono"))
    root.option_add("*Menu.background", p.surface)
    root.option_add("*Menu.foreground", p.text)
    root.option_add("*Menu.activeBackground", p.select_bg)
    root.option_add("*Menu.activeForeground", p.text)
    root.option_add("*Toplevel.background", p.bg)

    try:
        root.configure(bg=p.bg)  # type: ignore[call-arg]
    except Exception:
        pass
    _retheme_classic(root, pal)
    _notify(root, pal)
    return pal


def _retheme_classic(widget: tk.Misc, pal: Palette) -> None:
    """Re-color already created classic widgets (Text, Listbox, Canvas, Toplevel, Menu)."""
    for child in widget.winfo_children():
        try:
            if isinstance(child, tk.Text):
                child.configure(background=pal.log_bg, foreground=pal.log_text, insertbackground=pal.text,
                                selectbackground=pal.select_bg, highlightthickness=0)
            elif isinstance(child, tk.Listbox):
                child.configure(background=pal.surface, foreground=pal.text,
                                selectbackground=pal.select_bg, selectforeground=pal.text, highlightthickness=0)
            elif isinstance(child, tk.Canvas):
                role = getattr(child, "_simsim_role", None)
                cur = str(child.cget("background"))
                if role == "image" or cur in _LEGACY_CANVAS_BG:
                    child.configure(background=pal.canvas_bg, highlightthickness=0)
                    child._simsim_role = "image"  # type: ignore[attr-defined]
                elif role == "surface" or cur in {"white", "#ffffff", "SystemButtonFace", "#d9d9d9"}:
                    child.configure(background=pal.surface, highlightthickness=0)
                    child._simsim_role = "surface"  # type: ignore[attr-defined]
            elif isinstance(child, tk.Toplevel):
                child.configure(background=pal.bg)
            elif isinstance(child, tk.Menu):
                child.configure(background=pal.surface, foreground=pal.text,
                                activebackground=pal.select_bg, activeforeground=pal.text)
        except Exception:
            pass
        _retheme_classic(child, pal)


def _notify(root: tk.Misc, pal: Palette) -> None:
    listeners = getattr(root, _ATTR_LISTENERS, None) or []
    alive = []
    for widget, cb in listeners:
        try:
            if not widget.winfo_exists():
                continue
        except Exception:
            continue
        alive.append((widget, cb))
        try:
            cb(pal)
        except Exception:
            pass
    setattr(root, _ATTR_LISTENERS, alive)
