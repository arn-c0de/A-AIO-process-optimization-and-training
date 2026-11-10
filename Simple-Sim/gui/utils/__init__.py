"""Utility modules for GUI functionality.

Exports are resolved lazily so that tk-free helpers (e.g. ``dataset_ops``)
can be imported in environments without tkinter (tests, CLI tools).
"""

from gui.lazy import lazy_exports

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


__getattr__ = lazy_exports(__name__, _EXPORTS, globals())
