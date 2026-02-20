# Refactoring Summary — `gui/utils/`

**Date:** 2026-02-18
**Branch:** `1.0.2-refactor`
**Scope:** `Simple-Sim/gui/utils/` — 4 files modified, 2 unchanged

---

## Changes by File

### `validation_suite.py`

- **GU01** Removed the `sys.path.insert(0, ...)` hack at module level and the now-redundant
  `import sys` statement. The launcher already configures `sys.path`; all other GUI modules
  resolve `simple_sim` and `tools` without this workaround.
- **GU02** Extracted `_detect_metric_outliers(sample_ids, scores, metric, threshold_sigma,
  reason_template)` helper. The two nearly-identical z-score loops in `detect_image_outliers`
  (blur and brightness) are replaced by two calls to this helper.  The unused `i` variable
  from the old `enumerate(zip(...))` pattern is gone — the helper iterates with plain `zip`.
- **GU03** Moved the three inline tolerance literals in `detect_edge_cases` to module-level
  private constants: `_OK_SHIFT_PX = 3.0`, `_OK_ROTATION_DEG = 5.0`,
  `_TOMBSTONE_TILT_DEG = 75.0`. All references inside the function now use the named constants.
- **GU04** Added `_LOG = logging.getLogger(__name__)`. Replaced all four `print()` calls
  — three failure messages in `run_validation_suite` and the `imagehash`-missing warning in
  `detect_duplicate_images` — with `_LOG.warning(...)`.

---

### `model_inference.py`

- **GU05** Removed the inline ImageNet normalisation literals
  `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` from `_load_model`.
  The module now imports `IMAGENET_MEAN` and `IMAGENET_STD` from `simple_sim.data_loader`
  (the single source of truth established by R08 in the prior refactoring).
- **GU06** Moved the `transforms.Compose([...])` construction out of `_load_model` into a
  module-level constant `_INFERENCE_TRANSFORM`. `_load_model` assigns
  `self.transform = _INFERENCE_TRANSFORM` instead of rebuilding the pipeline on every call.
  This matches the pattern used by `_PREPROCESS_TRANSFORM` in `two_stage.py` (R09).
- **GU07** Added `_LOG = logging.getLogger(__name__)`. Replaced the two bare `print()` calls
  in `load_model` and `predict_image` with `_LOG.warning(...)`.

---

### `flag_manager.py`

- **GU08** Added `_LOG = logging.getLogger(__name__)`. Replaced `print(f"Failed to load
  flags: {e}")` and `print(f"Failed to save flags: {e}")` in `_load_flags` / `_save_flags`
  with `_LOG.warning("Failed to load flags: %s", e)` and
  `_LOG.warning("Failed to save flags: %s", e)`.

---

### `__init__.py`

- **GU09** Added imports of `SettingsStore` (from `.settings_store`) and `ToolTip`
  (from `.tooltip`) and included both in `__all__`. Callers that previously used
  `from gui.utils.settings_store import SettingsStore` can now use
  `from gui.utils import SettingsStore`.

---

## What Was Not Changed

| File | Reason |
|------|--------|
| `gui/utils/tooltip.py` | Already minimal and clean; no anti-patterns present |
| `gui/utils/settings_store.py` | Already minimal and clean; no anti-patterns present |

---

## No Behaviour Changes

All changes are purely structural:

- `sys.path` is not modified by this module any more, but the path was already set correctly
  by the launcher before import.
- The z-score computation in `_detect_metric_outliers` is mathematically identical to the
  two inline loops it replaced.
- `_INFERENCE_TRANSFORM` is created once at import time; the `Compose` object is stateless
  and reusable, so assigning it to `self.transform` is equivalent to constructing a fresh
  one on each call.
- `_LOG.warning(...)` records the same message that `print()` wrote; it is suppressed by
  default unless the caller configures a handler, which is the standard Python logging
  contract.
