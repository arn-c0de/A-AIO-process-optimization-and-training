"""Utility modules for GUI functionality.

Exports are resolved lazily so that tk-free helpers (e.g. ``dataset_ops``)
can be imported in environments without tkinter (tests, CLI tools).
"""

from importlib import import_module

_EXPORTS = {
    'detect_image_outliers': '.validation_suite',
    'detect_duplicate_images': '.validation_suite',
    'run_validation_suite': '.validation_suite',
    'load_model': '.model_inference',
    'predict_image': '.model_inference',
    'FlagManager': '.flag_manager',
    'SettingsStore': '.settings_store',
    'ToolTip': '.tooltip',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
