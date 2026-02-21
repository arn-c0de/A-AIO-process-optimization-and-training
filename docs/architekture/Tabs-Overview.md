# Tabs Overview

This document describes all GUI tabs under `Simple-Sim/gui/tabs`, including their purpose, major UI actions, logic responsibilities, and important data flow.

## Tab System Architecture

- `core/base.py` defines `BaseTab` with a shared lifecycle:
  - `build_ui()` for lazy UI creation
  - `on_activate()` for first-time initialization
  - optional hooks: `on_dataset_changed()`, `refresh()`
- `core/registry.py` defines `TabRegistry` and the active tab order:
  - `pipeline` → `datasets` → `analysis` → `validation` → `predictions` → `weights` → `merge` → `board_detection`
- Tabs communicate via shared `UiState` and UI events (especially `<<DatasetChanged>>` and `<<DatasetCatalogChanged>>`).

## Pipeline Control Tab (`pipeline`)

### Purpose
Main orchestration tab for dataset generation, training, evaluation, profile workflows, and live monitoring.

### UI Responsibilities (`pipeline/ui.py`)
- Start/stop controls for:
  - full pipeline
  - generate-only
- Run modes:
  - single, multiple, continuous, precise
- Dataset controls:
  - create new vs extend existing
  - single and multi-dataset selection
  - dataset stats and size indicators
- Model/profile controls:
  - model picker
  - profile picker
  - multi-profile mode (`separate` / `mixed`)
  - profile-classifier model controls
- Image filter controls (opens dedicated filter popup).
- Live status bar, logs, and recent thumbnails.

### Logic Responsibilities (`pipeline/tab.py` + `pipeline/logic.py`)
- Builds and validates run specs per profile and backend.
- Starts/stops subprocess workflows (`run_pipeline.sh`, generate/train/eval helper commands).
- Handles event streaming from JSONL logs and maps events to UI phase/status updates.
- Implements precise mode sample count overrides by writing temporary config files.
- Manages filter profiles and converts UI filter state into payload for generation.
- Supports multi-profile generation:
  - separate outputs
  - mixed generation with merge fallback for cross-backend profiles
- Dataset operations:
  - create, rename, snapshot, delete
  - delete/move selected samples between datasets
- Multi-dataset training/eval/continue-train strategies, including merge-then-train flows.
- Model metadata handling:
  - profile compatibility checks
  - model snapshots and snapshot pruning
  - profile-classifier build runs
- System and dataset monitoring:
  - CPU/GPU/RAM polling
  - asynchronous dataset size calculation
  - milestone logging

### Key Side Effects
- Writes to `outputs/sim_data/*`, `outputs/models/*`, `outputs/live/*`.
- Appends operational events to `outputs/live/events.jsonl`.
- Persists extensive UI state/filter settings in `SettingsStore`.

## Datasets Tab (`datasets`)

### Purpose
Central dataset catalog manager (category + archive status) used by other tabs.

### UI Responsibilities (`datasets/tab.py`)
- Dataset table with columns:
  - dataset name
  - category
  - created
  - archived
  - location
- Actions:
  - refresh
  - show details
  - set/new/rename/delete category
  - archive/unarchive selected
- Filtering:
  - show/hide archived
  - filter by category
- Context menu on dataset rows.

### Logic Responsibilities
- Scans dataset roots (`outputs/sim_data/runs`, `outputs/sim_data/versions`).
- Uses `DatasetCatalog` for category/archive metadata.
- Generates detailed dataset summary text:
  - total samples
  - split stats
  - per-profile sample counts
- Emits `<<DatasetCatalogChanged>>` when catalog metadata changes.

## Analysis Tab (`analysis`)

### Purpose
Explore dataset samples, run model inference/analysis, and manage human feedback corrections.

### UI Responsibilities (`analysis/ui.py`)
- Dataset selection and refresh.
- Hierarchical sample tree (`run/domain/split`) with class/date columns.
- Filters:
  - search
  - class/run/domain/split/profile
  - sorting modes
  - multi-profile filter selection dialog
- Image viewer with overlay and metadata panel.
- Navigation and dataset editing actions:
  - previous/next sample
  - delete image(s)
  - move image(s)
- Feedback panel:
  - thumbs-up/thumbs-down
  - corrected class + optional note
  - feedback history table and dialog
- Analysis actions:
  - baseline analysis
  - recompute with feedback labels

### Logic Responsibilities (`analysis/tab.py` + `analysis/logic.py`)
- Loads `meta.jsonl` and `labels.jsonl` into in-memory maps.
- Attempts model auto-selection/loading based on dataset-manifest hash.
- Applies effective labels from feedback history.
- Filters and renders sample hierarchy.
- Runs dataset analysis in background thread:
  - per-sample predictions
  - aggregate metrics (accuracy, macro F1, per-class, confusion matrix)
- Writes `analysis_results.json` to dataset directory.
- Supports feedback persistence and replay via `FeedbackManager`.

## Validation Tab (`validation`)

### Purpose
Execute validation suite on datasets and manage generated data quality flags.

### UI Responsibilities (`validation/ui.py`)
- Dataset selection.
- Configurable checks:
  - schema validation
  - outlier detection
  - duplicate detection
  - edge-case detection
- Results text output panel.
- Flagged sample tree with actions:
  - remove selected flag
  - clear all flags
- Export validation report.

### Logic Responsibilities (`validation/logic.py`)
- Runs `run_validation_suite(...)` with selected checks.
- Maps findings to `FlagManager` entries.
- Provides flag query/mutation helpers.
- Exports JSON report with safe normalization for Path/numpy values.

## Predictions Tab (`predictions`)

### Purpose
Batch prediction runner for one or many datasets with report browsing and per-sample review.

### UI Responsibilities (`predictions/ui.py`)
- Dataset + model pickers, including multi-dataset mode.
- Prediction options:
  - split
  - device
  - max samples
  - save per-sample predictions
  - auto-train missing bundle checkpoint
  - optional profile model
- Run/stop controls and live logs.
- Metrics panel + confusion matrix widget.
- Per-sample prediction table with filters (`all/wrong/correct/prof_wrong`).
- Multi-dataset summary table (Acc/F1/Seen/Profile-Acc).
- Image preview with prediction overlays.

### Logic Responsibilities (`predictions/tab.py` + `predictions/logic.py`)
- Resolves dataset/model paths and validates existence.
- Executes `predict.sh` per dataset in worker thread.
- Supports bundle models:
  - resolves profile-specific checkpoint from bundle
  - optional auto-train missing checkpoint
- Parses run output for report/preds paths.
- Loads report JSON and prediction JSONL.
- Enriches prediction rows with GT profile from labels.
- Aggregates and renders multi-dataset result summary.

## Weights Tab (`weights`)

### Purpose
Model checkpoint and bundle lifecycle management plus evaluation/compare workflows.

### UI Responsibilities (`weights/ui.py`)
- Model tree with grouped view and metadata columns.
- Actions:
  - refresh
  - snapshot active model
  - import/export
  - activate selected
  - profile info
  - rename/duplicate/delete
- Favorites and Arena workflows (via context menu).
- Group management (custom groups + drag/drop assignment).
- Evaluation controls (`predict.sh` parameters).
- A/B compare controls with multi-dataset selection.
- Reports/history table with filters and deletion.
- Logs panel.

### Logic Responsibilities (`weights/tab.py` + `weights/logic.py`)
- Scans models from root, versions, imports, and bundle directories.
- Stores and loads favorites (`favorites.json`) and arena (`arena.json`).
- Handles checkpoint/bundle import/export/rename/duplicate/delete.
- Executes prediction runs:
  - async single run
  - blocking compare runs
- Reads reports from multiple prediction output locations.
- Generates short summaries and compare summaries (single and weighted multi-dataset).
- Extracts model profile/class metadata and validates dataset/model class compatibility.
- Provides sample-count summary from dataset manifests/labels.

## Merge Tab (`merge`)

### Purpose
Merge selected profile-specific checkpoints into a multi-profile bundle or a single ensemble checkpoint.

### UI Responsibilities (`merge/ui.py`)
- Tree of available weights grouped by profile and model groups.
- Selection controls per profile.
- Merge controls:
  - output name
  - optional timestamp
  - merge to `.bundle`
  - merge to single ensemble `.pt`
- Existing bundles list:
  - inspect info
  - load into current selection
  - delete bundle
- Log output.

### Logic Responsibilities (`merge/tab.py` + `merge/logic.py`)
- Scans candidate checkpoints (including bundle-contained `.pt`).
- Loads model metadata (profile ID/hash, validation metrics).
- Bundle merge:
  - copies checkpoints into target bundle
  - updates bundle metadata
  - writes `bundle_details.json` with provenance/sources
- Ensemble merge:
  - embeds multiple slim checkpoints in one `simple_sim_ensemble_v1` file
  - validates class-name compatibility
- Handles bundle listing and deletion.

## Board Detection Tab (`board_detection`)

### Purpose
Two-stage inference UI (profile classification + defect classification) for single image inputs.

### UI Responsibilities (`board_detection/ui.py`)
- Select:
  - profile classifier checkpoint
  - defect bundle directory
  - input image
- Set device and minimum profile confidence.
- Trigger prediction and show:
  - profile probabilities
  - defect probabilities
  - review flag indicator
- Error/status display.

### Logic Responsibilities (`board_detection/tab.py` + `board_detection/logic.py`)
- Validates selected paths and numeric inputs.
- Runs prediction asynchronously through `TwoStageClassifier`.
- Reuses classifier instance when model/bundle paths stay unchanged.
- Updates UI with final `TwoStageResult`.

## Cross-Tab Coupling Notes

- `state.dataset_dir` is the primary shared context across tabs.
- Tabs react to dataset changes through `on_dataset_changed()` and `<<DatasetChanged>>` events.
- Dataset category/archive metadata from `DatasetCatalog` affects visibility in multiple tabs.
- `SettingsStore` persists tab-specific selections and behavior defaults across sessions.
