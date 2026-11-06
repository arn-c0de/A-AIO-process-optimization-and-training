"""Shared design system for the Simple-Sim Tk apps (monitor GUI + profile editor).

``tokens`` is tk-free (palettes, spacing, font specs); ``style`` applies them to a
Tk root via ``apply_theme``. Widgets read the active palette with ``palette()``.
"""

from gui.theme.tokens import DARK, LIGHT, SPACE, Palette, palette_for

__all__ = ["DARK", "LIGHT", "SPACE", "Palette", "palette_for"]
