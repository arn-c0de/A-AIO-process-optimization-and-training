"""Reusable GUI components.

Exports are resolved lazily so tk-free helpers (e.g. ``overlay_renderer`` used
by ``tools/update_sample_gallery.py``) import without tkinter.
"""

from importlib import import_module

_EXPORTS = {
    'ImageCache': '.image_cache',
    'draw_defect_overlay': '.overlay_renderer',
    'create_confusion_matrix_widget': '.chart_widgets',
    'create_class_distribution_widget': '.chart_widgets',
    'FilterPopup': '.filter_popup',
    'open_filter_popup': '.filter_popup',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
