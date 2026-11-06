"""Design tokens (tk-free): colors, spacing and font roles.

Every color, spacing and font in the GUI comes from here. Widgets never hardcode
hex values; they ask ``gui.theme.style.palette(widget)`` for the active palette.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str             # window background
    surface: str        # cards / sections / input fields
    surface_alt: str    # toolbars, table headings, hover
    border: str         # subtle separators and outlines
    text: str
    text_muted: str
    accent: str         # primary action, selection, focus
    accent_hover: str
    accent_text: str    # text on accent background
    select_bg: str      # selection in lists/tables
    success: str
    warning: str
    danger: str
    danger_hover: str
    canvas_bg: str      # image viewers
    tooltip_bg: str
    tooltip_text: str
    log_bg: str
    log_text: str

    def as_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


LIGHT = Palette(
    name="light",
    bg="#f5f6f8",
    surface="#ffffff",
    surface_alt="#eef0f3",
    border="#d9dde3",
    text="#1b1f24",
    text_muted="#6b7280",
    accent="#2f6fed",
    accent_hover="#2559c4",
    accent_text="#ffffff",
    select_bg="#dbe7ff",
    success="#1f9d55",
    warning="#b7791f",
    danger="#d64545",
    danger_hover="#b83232",
    canvas_bg="#2b2f36",
    tooltip_bg="#1f2329",
    tooltip_text="#f5f7fa",
    log_bg="#fbfbfc",
    log_text="#1b1f24",
)

DARK = Palette(
    name="dark",
    bg="#16191d",
    surface="#1f2329",
    surface_alt="#262b32",
    border="#323840",
    text="#e8ebef",
    text_muted="#9aa3ad",
    accent="#4c8dff",
    accent_hover="#6aa0ff",
    accent_text="#0d1117",
    select_bg="#2a3d63",
    success="#3ecf7a",
    warning="#e0a43a",
    danger="#f06262",
    danger_hover="#ff7a7a",
    canvas_bg="#0f1216",
    tooltip_bg="#e8ebef",
    tooltip_text="#16191d",
    log_bg="#12151a",
    log_text="#d7dde4",
)

PALETTES = {"light": LIGHT, "dark": DARK}


def palette_for(mode: str | bool | None) -> Palette:
    """``"dark"``/``True`` -> DARK, anything else -> LIGHT."""
    if mode is True or (isinstance(mode, str) and mode.lower() == "dark"):
        return DARK
    return LIGHT


class SPACE:
    """Compact spacing scale (px)."""

    XS = 1
    S = 3
    M = 5
    L = 8
    XL = 10
    XXL = 16


# Font roles -> (size, weight, slant, family role). Family role "ui" maps to the
# platform UI font (TkDefaultFont), "mono" to TkFixedFont.
FONT_ROLES: dict[str, tuple[int, str, str, str]] = {
    "Title": (11, "bold", "roman", "ui"),
    "Heading": (9, "bold", "roman", "ui"),
    "Body": (9, "normal", "roman", "ui"),
    "BodyBold": (9, "bold", "roman", "ui"),
    "Small": (8, "normal", "roman", "ui"),
    "SmallBold": (8, "bold", "roman", "ui"),
    "SmallItalic": (8, "normal", "italic", "ui"),
    "Mono": (9, "normal", "roman", "mono"),
}

# Tk named fonts are registered as "SimSim<Role>" (e.g. "SimSimHeading").
FONT_PREFIX = "SimSim"


def font_name(role: str) -> str:
    if role not in FONT_ROLES:
        raise KeyError(f"unknown font role {role!r}; known: {sorted(FONT_ROLES)}")
    return FONT_PREFIX + role


# Fixed colors for defect classes (badges, overlays, charts); same in both themes.
CLASS_COLORS: dict[str, str] = {
    "OK": "#1f9d55",
    "MISSING": "#d64545",
    "MISALIGNED": "#d69e2e",
    "TOMBSTONE": "#805ad5",
    "SOLDER_BRIDGE": "#dd6b20",
    "CORNER_LIFT": "#3182ce",
    "SOLDER_BALL": "#d53f8c",
    "EXCESS_SOLDER": "#319795",
}


def class_color(name: str, default: str = "#718096") -> str:
    return CLASS_COLORS.get(str(name).upper(), default)
