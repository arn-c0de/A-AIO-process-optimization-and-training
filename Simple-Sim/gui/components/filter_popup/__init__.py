"""Filter popup package.

``FilterPopup`` and ``open_filter_popup`` are resolved lazily so that the tk-free
submodules (``preview_render``, ...) can be imported without tkinter.
"""

from importlib import import_module

_EXPORTS = {
    "FilterPopup": ".popup",
    "open_filter_popup": ".popup",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
