# Refactoring Plan — `simple_sim`

**Date:** 2026-02-18
**Scope:** `simple_sim/` package
**Goal:** Improve readability, maintainability, and correctness without changing behaviour.

---

## Items

### [R01] `config.py` — Split `validate_config()` god-function
**Priority:** Critical
`validate_config()` is 225 lines long and validates nine distinct sections inline.
Each section becomes its own private helper: `_validate_run`, `_validate_roi`,
`_validate_classes`, `_validate_domains`, `_validate_render`, `_validate_augment`,
`_validate_train`, `_validate_eval`. The public function becomes an orchestrator.

---

### [R02] `config.py` — Remove redundant `valid_classes` local assignments
**Priority:** Critical
`valid_classes = VALID_CLASSES` appears twice (lines 100, 252) as local shadowing of
the module-level constant. Use `VALID_CLASSES` directly.

---

### [R03] `schema.py` — Use dataclass introspection for field validation
**Priority:** Medium
`read_jsonl` maintains manual `expected_fields` sets per schema class. These drift
silently when fields are added. Replace with `{f.name for f in dataclasses.fields(schema_cls)}`.

---

### [R04] `schema.py` — Remove external `__post_init__()` call
**Priority:** Medium
`write_jsonl` calls `row.__post_init__()` directly to re-validate. This is an
anti-pattern; `__post_init__` is an internal dataclass hook. Remove the explicit call —
the dataclass is already validated at construction time.

---

### [R05] `defects.py` — Name magic numbers, split into per-type helpers
**Priority:** Medium
`sample_defect_params` is 110 lines with unnamed float literals (`0.7`, `5.0`, `8.0`,
`1e-6`, etc.). Extract module-level constants and private helpers
`_sample_ok`, `_sample_misaligned`, `_sample_tombstone`, `_sample_solder_bridge`,
`_sample_corner_lift`.

---

### [R06] `data_loader.py` — Extract shared base class
**Priority:** Critical
`ROIDataset` and `ProfileDataset` share ~60 % of code: init structure, image loading,
ImageNet normalization, `get_class_distribution`. Extract `_BaseROIDataset` with
shared logic; subclasses override only label resolution.

---

### [R07] `data_loader.py` — Replace `print()` warnings with `warnings.warn()`
**Priority:** Medium
`print("WARNING: ...")` cannot be filtered, suppressed, or redirected by callers.
Use `warnings.warn(..., stacklevel=2)`.

---

### [R08] `data_loader.py` + `two_stage.py` — Deduplicate ImageNet normalization constants
**Priority:** Critical
`mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]` are hardcoded in both
files. Define `IMAGENET_MEAN` / `IMAGENET_STD` once in `data_loader.py` and import
in `two_stage.py`.

---

### [R09] `two_stage.py` — Module-level preprocessing transform
**Priority:** Critical
`_preprocess()` constructs a new `transforms.Compose` on every call. Move to a
module-level constant `_PREPROCESS_TRANSFORM`.

---

### [R10] `manifest.py` — Use `__version__` instead of hardcoded `'1.0.1'`
**Priority:** Critical
The generator version `'1.0.1'` is hardcoded twice. Import `__version__` from
`simple_sim` and use it consistently.

---

### [R11] `manifest.py` — Extract shared helpers
**Priority:** Medium
`write_dataset_manifest` and `write_multi_profile_manifest` duplicate: git-commit
lookup, class-count computation, and atomic JSON write. Extract
`_compute_class_counts(label_rows)` and `_atomic_json_write(path, obj)`.

---

### [R12] `splits.py` — Generic pairwise overlap check
**Priority:** Low
`assert_no_overlap` hardcodes three named set variables and three pairwise checks.
Replace with `itertools.combinations` to work for any number of splits.

---

### [R13] `model_bundle.py` — Add missing docstrings
**Priority:** Low
`is_bundle_dir`, `read_bundle_meta`, and `upsert_bundle_meta` lack docstrings
inconsistent with the rest of the package.

---

### [R14] `generator_2d.py` — Vectorize `apply_shadow` / `apply_reflection`
**Priority:** Low
Both functions use nested Python loops over pixel coordinates. Replace with numpy
meshgrid/ogrid operations for correct and significantly faster execution.

---

## Excluded

- `generator_2d.py:normalize_image_filters` — large but inherently data-driven;
  restructuring would require a separate config schema change.
- `blender/render_batch.py` — intentionally thin; no changes needed.
- `telemetry.py` — the `or None` pattern is intentional (handles empty-string env var).
