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
  - `classes.py` (class/split/backend vocabularies), `coerce.py` (lenient bool/float/int parsing), `io_utils.py` (JSON/JSONL/YAML I/O, timestamps)
  - `config.py`, `config_schema.py`, `schema.py`, `manifest.py`, `dataset_store.py`, `splits.py`, `rng.py`, `telemetry.py`
- Defect and profile domain logic
  - `defects.py`, `profile_hash.py`
- Inference/evaluation helpers
  - `data_loader.py`, `metrics.py`, `model_bundle.py`, `two_stage.py`
- Generator backends
  - `generators/opencv2d/*`
  - `generators/blender3d/*` + Blender batch script
- Image-filter settings
  - `filter_settings.py` (pure Python, shared by both backends and the GUI)

## Module-by-Module

## `simple_sim/__init__.py`

- Declares package version: `__version__`.
- Adds a `NullHandler` to the `simple_sim` logger.

## Logging and errors

- Library modules never `print` or `sys.exit`: progress goes to `logging.getLogger(__name__).info`, user warnings to `.warning`. Only `cli.py`, `cli_utils.py`, `__main__.py` and the Blender script `blender/render_batch.py` print.
- `cli_utils.configure_cli_logging()` (called by `run_cli`) shows `simple_sim` records on stdout as plain lines, warnings as `WARNING: ...`, so the console output of the scripts is unchanged apart from that prefix.
- Errors are neutral (`manifest.ProfileMismatchError`, `manifest.RendererMismatchError`, both `ValueError`); the scripts add the flag advice with `cli_utils.with_hint(..., PROFILE_MISMATCH_HINT)` / `RENDERER_MISMATCH_HINT`.
- Console text blocks are returned as strings: `metrics.format_success_criteria`, `eval_utils.format_checkpoint_info`, `eval_utils.format_distribution`.

## `simple_sim/classes.py`

- `DefectClass`, `Split`, `Backend` (`StrEnum`, values are the strings stored in JSON/YAML).
- Derived plain-string constants: `VALID_CLASSES`, `STANDARD_CLASSES`, `CRITICAL_CLASSES`, `TYPE_LABELLED_CLASSES`, `PLACEMENT_CLASSES`, `SOLDER_ONLY_CLASSES`, `NEVER_TOMBSTONE`, `SPLIT_NAMES`, `BACKENDS`.
- Stdlib only; `schema.py` takes `VALID_CLASSES` from here, so it no longer needs PyYAML.

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
- `dataset_stats(meta_rows, label_rows, splits)`
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

Responsibilities:
- Builds stable hashes independent of formatting/noise fields.
- Validates profile-id/file-name consistency.

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

Purpose: classification metric computation and formatting (numpy only, values identical to scikit-learn's, checked by `tests/test_metrics.py`).

Key API:
- `compute_metrics(y_true, y_pred, class_names, critical_classes=None)`
- `evaluate_success_criteria(val_accuracy, metrics, entity="class") -> SuccessCheck` and `format_success_criteria(check, ascii_only=False)`
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

## Board Model (`simple_sim/board/*`)

Procedural PCB around the inspected part as plain data in mm (`render.board.routing: escape`;
`legacy` keeps the old copper generator). Shared by the 2D render and the board baked for Blender.

| Module | Responsibility |
|---|---|
| `board/__init__.py` | `build_window_board(...)` (pipeline for one render window), `BoardOptions` / `board_options(render_cfg)` / `validate_board_options`, keyed random streams per stage, `DEFAULT_MM_PER_PX` |
| `board/model.py` | Dataclasses `Pad`, `Part`, `Track`, `Via`, `Zone`, `Port`, `Board` (mm, JSON-serialisable, no OpenCV) |
| `board/rules.py` | `DesignRules` and presets (`consumer_dense`, `industrial`, `hdi`) as ranges; `sample_rules()` draws one set per board |
| `board/footprints.py` | Footprint library: `footprint_pads()` (pixel geometry with pin, role, fan-out direction; the source of `draw._compute_pad_positions`), `chip_pads()`, `part_from_pads()` / `pads_px()` (mm <-> px), `family_of()` |
| `board/placement.py` | `neighbour_layout()` (legacy neighbour chips, pixel dicts) and `place_window()` (target + neighbours as `Part`s) |
| `board/netlist.py` | `assign_nets()`: nets per part family, shared nets between target and neighbours |
| `board/occupancy.py` | `Grid` (mm -> cells, drawing helpers) and `Occupancy` (net id per cell, conflict checks with clearance) |
| `board/routing/` | `Router` protocol, `RouteResult`, `seed_occupancy()`; `escape.py`: fan-out router (stubs, 45° routes to ports or vias, thermal via grids, through-nets, buses) |
| `board/zones.py` | Ground pours: `sample_zones()`, `zone_fill()` (clearance, thermal relief) on any grid, `stitch_vias()` |
| `board/drc.py` | `check(board) -> DrcReport`: clearance, dangling ends, forbidden pads, keep-out, widths |
| `board/raster.py` | `rasterize(board, shape, px_to_mm) -> CopperLayers` (anti-aliased copper mask + via positions for `photoreal._draw_copper`) |

## Generators Package

## `simple_sim/generators/__init__.py`

- Intentionally empty: import `simple_sim.generators.opencv2d` or `simple_sim.generators.blender3d` directly, so loading one backend (or `generators.version`) does not load the other or OpenCV.

## `simple_sim/filter_settings.py`

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

## `simple_sim/generators/blender3d/bake.py`

- `prepare_jobs(jobs, work_dir)`: bakes the photoreal 2D board (mask, traces, vias, silkscreen, pad finish)
  per job to a texture and adds the shared layout (pads, body pose, family, thermal pad).
- `apply_camera(jobs, output_root)`: runs every render through the 2D camera model.
- `profile_job_fields(profile)`: part type + tolerances a job needs for the bake.

## `simple_sim/generators/blender3d/runner.py`

- `render_blender_batch(...)`: bakes boards, executes Blender in background with render script + prepared
  jobs file, removes the textures, then applies the camera model.
- Validates script/jobs existence and raises on non-zero return code.

## `simple_sim/blender/render_batch.py`

Purpose: standalone Blender-side batch renderer script.

Responsibilities:
- Parses Blender custom CLI args.
- Builds the scene per job from the prepared JSONL (layout from the bake, no own pad logic):
  - textured board plane, solder fillet height-field meshes per pad
  - part per family: chip body + end caps, SOT/SOIC/QFN mold body, gull-wing leads, pin-1 mark, marking text
- Configures renderer (sRGB `Standard` view transform), AOI ring light and ROI-framed orthographic camera.
- Applies defect transforms and augmentation-driven global rotation.
- Renders each job image and fails fast on error.


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
