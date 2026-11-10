"""Reusable, theme-aware Tk building blocks.

Presentational only: widgets never touch datasets/models themselves; tabs wire
callbacks. Import from here::

    from gui.widgets import Section, CollapsibleSection, FormGrid, PathPicker, ButtonBar, \
        ScrolledTree, LogView, StatusBar, EmptyState, ScrollableFrame, icon_button
"""

from gui.widgets.data_views import LogView, LogViewHost, ScrolledTree
from gui.widgets.form import FormGrid, PathPicker, set_invalid
from gui.widgets.layout import (
    ICONS,
    ButtonBar,
    CollapsibleSection,
    EmptyState,
    ScrollableFrame,
    Section,
    Toolbar,
    icon_button,
    keep_pane_visible,
    menu_button,
)
from gui.widgets.status import StatusBar

__all__ = [
    "ICONS",
    "ButtonBar",
    "CollapsibleSection",
    "EmptyState",
    "FormGrid",
    "LogView",
    "LogViewHost",
    "PathPicker",
    "ScrollableFrame",
    "ScrolledTree",
    "Section",
    "StatusBar",
    "Toolbar",
    "icon_button",
    "keep_pane_visible",
    "menu_button",
    "set_invalid",
]
