# simple_sim Package Overview

This document describes the architecture and responsibilities of all modules in `Simple-Sim/simple_sim`.

## Package Purpose

`simple_sim` is the core simulation and ML support package for:
- synthetic PCB sample generation (OpenCV 2D and Blender 3D)
- dataset schema/manifest handling
- deterministic split and seed logic
- model bundle and profile-hash support
- training/evaluation data loading and metrics
- two-stage inference (profile model + defect model)

## High-Level Structure

- Core data/config/runtime
  - `config.py`, `config_schema.py`, `schema.py`, `manifest.py`, `dataset_store.py`, `splits.py`, `rng.py`, `telemetry.py`
- Defect and profile domain logic
  - `defects.py`, `profile_hash.py`
- Inference/evaluation helpers
  - `data_loader.py`, `metrics.py`, `model_bundle.py`, `two_stage.py`
- Generator backends
  - `generators/opencv2d/*` + facades
  - `generators/blender3d/*` + Blender batch script
- Compatibility re-export modules
  - `generator_2d.py`, `generator_3d.py`, `generators/opencv_2d.py`, `generators/blender_3d.py`

## Module-by-Module

## `simple_sim/__init__.py`

- Declares package version: `__version__`.

## `simple_sim/config.py`

Purpose: YAML config loading and strict runtime validation.

Key API:
- `load_config(path)`
- `validate_config(cfg)`
- `validate_splits(cfg)`
- `validate_tolerances(cfg)`

Responsibilities:
- Enforces required sections by schema version and mode.
- Validates:
  - run metadata (`run_id`, `seed`, `schema_version`, mode constraints)
  - ROI, classes, domains, split fractions
  - render backend (`opencv_2d` / `blender_3d`)
  - train/eval settings
- Supports profile-classifier mode rules.

## `simple_sim/config_schema.py`

Purpose: typed dataclass view for baseline config parsing.

Key API:
- `RunConfig`, `RoiConfig`, `TrainConfig`, `EvalConfig`, `SimulationConfig`
- `parse_config_typed(cfg)`

Responsibilities:
- Converts untyped dict config into typed dataclasses for strict baseline checks before deeper validation.

## `simple_sim/schema.py`

Purpose: JSONL contracts for dataset metadata and labels.

Key API:
- `MetaRow` dataclass
- `LabelRow` dataclass
- `write_jsonl(path, rows)`
- `read_jsonl(path, schema_cls)`
- `validate_jsonl_pair(meta_path, labels_path)`

Responsibilities:
- Validates row-level constraints in `__post_init__`.
- Supports schema v1/v2 compatibility (`profile_id`, `render_meta`).
- Ensures meta/label consistency by row count and ID set.

## `simple_sim/manifest.py`

Purpose: dataset manifest creation, update, and validation.

Key API:
- `hash_file(file_path)`
- `write_dataset_manifest(...)`
- `read_dataset_manifest(manifest_path)`
- `validate_manifest_profile(...)`
- `write_multi_profile_manifest(...)`

Responsibilities:
- Writes single-profile (manifest v1) and multi-profile (manifest v2) manifests.
- Tracks extend history and class/split/profile statistics.
- Includes generator metadata (`simple_sim` version, git commit, script).
- Performs atomic JSON writes.

## `simple_sim/dataset_store.py`

Purpose: atomic dataset materialization with validation.

Key API:
- `validate_dataset_files(data_dir)`
- `write_dataset(output_dir, images, meta_rows, label_rows, config)`

Responsibilities:
- Writes complete dataset into a temp folder first.
- Validates files and JSONL coherence before final rename.
- Prevents partial/corrupt dataset states.

## `simple_sim/splits.py`

Purpose: stratified train/val/test split generation and validation.

Key API:
- `generate_splits(meta_rows, label_rows, config, run_seed)`
- `assert_no_overlap(splits)`
- `write_splits(output_dir, splits)`
- `read_split(split_file)`
- `check_class_coverage(splits, label_rows)`

Responsibilities:
- Stratifies by `(domain, class)`.
- Guarantees determinism via seed-based shuffling.
- Detects split overlap and missing class coverage.

## `simple_sim/rng.py`

Purpose: deterministic seed and sample-id utilities.

Key API:
- `derive_sample_seed(run_seed, domain, index)`
- `make_sample_id(run_id, domain, split, index)`
- `parse_sample_id(sample_id)`

Responsibilities:
- Stable hash-derived 63-bit seeds.
- Canonical sample ID format round-trip.

## `simple_sim/telemetry.py`

Purpose: best-effort JSONL event emission for live monitoring.

Key API:
- `emit(event, **fields)`

Responsibilities:
- Appends events when `SIMPLE_SIM_EVENT_LOG` is configured.
- Never raises into generation/training flow on telemetry failure.

## `simple_sim/defects.py`

Purpose: defect-label logic and defect-parameter sampling.

Key API:
- `classify_defect(nominal, defect_params, tolerances)`
- `sample_defect_params(defect_type, rng, tolerances=None)`

Responsibilities:
- Deterministic rule-based class assignment.
- Sampling strategies per defect type:
  - `OK`, `MISSING`, `MISALIGNED`, `TOMBSTONE`, `SOLDER_BRIDGE`, `CORNER_LIFT`
- Uses tolerance-aware bounds with safe defaults.

## `simple_sim/profile_hash.py`

Purpose: profile loading and semantic hashing.

Key API:
- `canonicalize_profile(profile_dict)`
- `hash_profile(profile_path)`
- `load_profile(profile_id, profiles_dir)`
- `get_profile_metadata(profile_dict)`

Responsibilities:
- Builds stable hashes independent of formatting/noise fields.
- Validates profile-id/file-name consistency.
- Exposes compact metadata for logs/UI.

## `simple_sim/model_bundle.py`

Purpose: multi-model bundle conventions and metadata.

Key API:
- `is_bundle_dir(p)`
- `bundle_checkpoint_path(bundle_dir, profile_id, kind="best"|"last")`
- `read_bundle_meta(bundle_dir)`
- `upsert_bundle_meta(bundle_dir, profile_id, ckpt_path)`
- `BundleMeta` dataclass

Responsibilities:
- Maps profile IDs to bundle checkpoint filenames.
- Maintains `bundle.json` for discoverability and lifecycle metadata.

## `simple_sim/data_loader.py`

Purpose: PyTorch dataset implementations for generated ROI data.

Key API:
- `ROIDataset`
- `ProfileDataset`
- shared constants: `IMAGENET_MEAN`, `IMAGENET_STD`

Responsibilities:
- Loads split IDs and JSONL schema rows.
- Reads images, applies ImageNet normalization transform.
- Produces class-index labels for:
  - defect classification (`ROIDataset`)
  - profile classification (`ProfileDataset`)
- Warns and excludes invalid sample rows.

## `simple_sim/metrics.py`

Purpose: classification metric computation and formatting.

Key API:
- `compute_metrics(y_true, y_pred, class_names, critical_classes=None)`
- `format_metrics(metrics, class_names)`

Responsibilities:
- Computes accuracy, macro-F1, per-class metrics, confusion matrix.
- Computes critical false-negative rates.
- Produces readable text report output.

## `simple_sim/two_stage.py`

Purpose: two-stage inference pipeline.

Key API:
- `TwoStageResult` dataclass
- `TwoStageClassifier`
- internal helpers: `_load_model`, `_preprocess`

Responsibilities:
- Stage 1: profile classification model.
- Stage 2: profile-specific defect model loaded from bundle.
- Returns profile + defect probabilities/confidences and review flag.
- Lazy-loads per-profile defect models and reuses them.

## Generators Package

## `simple_sim/generators/__init__.py`

- Re-exports OpenCV and Blender generator APIs.
- Re-exports filter-settings helpers.

## `simple_sim/generators/filter_settings.py`

Purpose: centralized filter normalization and preset shaping.

Key API:
- `normalize_image_filters(image_filters)`
- `default_filter_values_for_popup()`
- `clean_filter_values_for_popup()`

Responsibilities:
- Normalizes bool/float/int/string filter payloads with bounds.
- Supports realism policy parameters and randomizable min/max ranges.
- Produces popup-compatible value dictionaries.

## OpenCV 2D Backend (`simple_sim/generators/opencv2d/*`)

## `opencv2d/__init__.py`

- Re-exports rendering, geometry, draw primitives, filter primitives, and augment logic.

## `opencv2d/geometry.py`

- `sample_nominal_geometry(...)` samples footprint geometry ranges.

## `opencv2d/draw.py`

- Draw primitives:
  - substrate
  - pads (footprint-aware layout)
  - solder
  - component body with defect transforms
- Supports footprint-specific pad positioning logic (`chip_2pad`, `sot23`, `qfn*`, `soic_16`).

## `opencv2d/filters.py`

- Image filter primitives:
  - blur/noise/brightness/contrast
  - perspective/motion blur
  - chromatic aberration/vignetting
  - saturation/hue shift
  - sharpen/lens distortion/jpeg compression
  - shadow/reflection/dust/color temperature

## `opencv2d/augment.py`

- `sample_augment_params(...)`: base augmentation sampling per domain/config.
- `apply_image_filter_overrides(...)`: maps normalized filter settings into effective augment values.
- Supports custom mode and realism mode integration.

## `opencv2d/realism.py`

- Realism grouped sampling model (`G1..G8`) with constraint logic.
- K-group selection policy (`k0/k1/k2`) and weighted non-replacement sampling.
- Optional profile preset loading from YAML.
- `apply_realism_groups(...)` applies correlated augment effects.

## `opencv2d/render.py`

- `render_roi(...)` orchestrates full 2D pipeline:
  - draw substrate/pads/solder/component
  - apply ordered filter chain
  - optional global in-plane rotation

## Blender 3D Backend

## `simple_sim/generators/blender3d/io.py`

- `write_jobs_jsonl(path, jobs)`: writes render jobs for batch Blender execution.

## `simple_sim/generators/blender3d/runner.py`

- `render_blender_batch(...)`: executes Blender in background with render script + jobs file.
- Validates script/jobs existence and raises on non-zero return code.

## `simple_sim/blender/render_batch.py`

Purpose: standalone Blender-side batch renderer script.

Responsibilities:
- Parses Blender custom CLI args.
- Builds minimal scene per job from JSONL:
  - substrate, pads, component, solder joints
  - footprint-specific geometry for `chip_2pad`, `sot23`, `qfn*`, `soic_16`
- Configures renderer/view transform and light/camera rig.
- Applies defect transforms and augmentation-driven global rotation.
- Renders each job image and fails fast on error.

## Compatibility Facades

- `simple_sim/generator_2d.py`
- `simple_sim/generator_3d.py`
- `simple_sim/generators/opencv_2d.py`
- `simple_sim/generators/blender_3d.py`

Responsibilities:
- Preserve older import paths while re-exporting split backend modules.

## Data Flow Summary

1. Config + profile are loaded/validated (`config.py`, `profile_hash.py`).
2. Geometry/defect/augment are sampled (`geometry.py`, `defects.py`, `augment.py`).
3. Samples are rendered (`render.py` or Blender batch path).
4. Metadata/labels are written using schema contracts (`schema.py`, `dataset_store.py`).
5. Splits and manifest metadata are written (`splits.py`, `manifest.py`).
6. Training/eval uses datasets + metrics (`data_loader.py`, `metrics.py`).
7. Inference can run two-stage with per-profile bundles (`two_stage.py`, `model_bundle.py`).

## Operational Notes

- Determinism is a first-class concern (`rng.py`, seeded split/sampling logic).
- Many writes are atomic or temp-first (`dataset_store.py`, `manifest.py`, `schema.py`).
- Bundle/profile metadata is used to prevent silent cross-profile model misuse.
