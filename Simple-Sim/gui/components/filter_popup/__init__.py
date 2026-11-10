"""Filter popup package.

``FilterPopup`` and ``open_filter_popup`` are resolved lazily so that the tk-free
submodules (``preview_render``, ...) can be imported without tkinter.
"""

from gui.lazy import lazy_exports

_EXPORTS = {
    "FilterPopup": ".popup",
    "open_filter_popup": ".popup",
}

__all__ = list(_EXPORTS)


__getattr__ = lazy_exports(__name__, _EXPORTS, globals())
