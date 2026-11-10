"""Reusable GUI components.

Exports are resolved lazily so tk-free helpers (e.g. ``overlay_renderer`` used
by ``tools/update_sample_gallery.py``) import without tkinter.
"""

from gui.lazy import lazy_exports

_EXPORTS = {
    'draw_defect_overlay': '.overlay_renderer',
    'create_confusion_matrix_widget': '.chart_widgets',
    'create_class_distribution_widget': '.chart_widgets',
    'FilterPopup': '.filter_popup',
    'open_filter_popup': '.filter_popup',
}

__all__ = list(_EXPORTS)


__getattr__ = lazy_exports(__name__, _EXPORTS, globals())
