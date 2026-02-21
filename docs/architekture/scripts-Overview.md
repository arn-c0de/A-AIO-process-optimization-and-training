# Scripts Overview

This document describes all executable scripts under `Simple-Sim/scripts`, including purpose, key behavior, inputs/outputs, and how each script connects to `simple_sim` modules.

## Directory Purpose

`Simple-Sim/scripts` is the operational CLI layer of the project. It provides commands for:
- dataset generation (single-profile and multi-profile)
- model training/evaluation/inference
- two-stage profile+defect inference
- report/history aggregation
- debug preview rendering for profiles and backends

## Script Groups

- Dataset generation
  - `generate.py`
  - `generate_profile_dataset.py`
- Defect model lifecycle
  - `train.py`
  - `eval.py`
  - `predict.py`
  - `batch_predict.py`
  - `sample_check.py`
- Profile classifier lifecycle
  - `train_profile.py`
  - `eval_profile.py`
  - `predict_two_stage.py`
- Reporting and visualization utilities
  - `arena_report.py`
  - `render_debug_previews.py`

## `generate.py`

Purpose:
- Generates synthetic defect datasets from a run config.
- Supports both OpenCV 2D and Blender 3D backends.
- Supports dataset extension mode (`--extend`) for appending samples.

Key behavior:
- Loads and validates config via `simple_sim.config`.
- Loads component profile and computes semantic profile hash (`simple_sim.profile_hash`).
- Generates deterministic IDs/seeds (`simple_sim.rng`).
- Samples defect parameters and labels (`simple_sim.defects`).
- Applies augmentation + optional filter overrides from `IMAGE_FILTERS` env JSON.
- Writes JSONL/schema rows (`MetaRow`, `LabelRow`), split files, and manifest.
- For Blender backend:
  - writes jobs JSONL
  - runs batch renderer
  - post-processes rendered images with the same filter stack.

Important CLI:
- `--config`, `--out`
- `--extend`
- `--disable-cardinal-rotation-90`

Outputs/artifacts:
- Dataset directory with `images/`, `meta.jsonl`, `labels.jsonl`, `splits/*.txt`, `config.yaml`.
- Manifest:
  - `dataset_manifest.json` v1 (single profile) or v2 (multi-profile extend flows).
- Blender job files (`blender_jobs.jsonl` / `blender_jobs_extend.jsonl`) when backend is 3D.
- Telemetry events (if configured) via `simple_sim.telemetry.emit`.

## `generate_profile_dataset.py`

Purpose:
- Generates a mixed-profile dataset for training a profile classifier.

Key behavior:
- Requires `render.backend=opencv_2d`.
- Iterates configured `run.component_profiles`.
- Generates schema v2 labels with `profile_id`.
- Creates deterministic stratified splits.
- Writes a multi-profile manifest.

Important CLI:
- `--config`, `--out`
- `--disable-cardinal-rotation-90`

Outputs/artifacts:
- Dataset files (`images/`, `meta.jsonl`, `labels.jsonl`, `splits/*.txt`, `config.yaml`).
- `dataset_manifest.json` (manifest v2 with `component_profiles`).

## `train.py`

Purpose:
- Trains a defect classifier (ResNet18) on generated datasets.

Key behavior:
- Uses `ROIDataset` from `simple_sim.data_loader`.
- Uses shared metric computation (`simple_sim.metrics`).
- Validates dataset manifest/profile compatibility.
- Supports resume training and extra epochs.
- Supports AMP on CUDA (`--amp/--no-amp`).
- Supports bundle output mode:
  - if `--out` is a directory or ends with `.bundle`, writes per-profile checkpoint path via `simple_sim.model_bundle.bundle_checkpoint_path`.
- Writes both best and last checkpoints depending on mode.
- Updates bundle metadata (`bundle.json`) via `upsert_bundle_meta`.

Important CLI:
- `--data`, `--out`, `--device`
- `--resume`, `--allow-profile-mismatch`
- `--extra-epochs`
- `--out-mode best|last`
- `--amp` / `--no-amp`

Outputs/artifacts:
- Main checkpoint at `--out` (or bundle-resolved path).
- Last checkpoint at `<stem>_last.pt`.
- Checkpoint metadata includes class list, profile metadata, config, and manifest hash.

## `eval.py`

Purpose:
- Evaluates a trained defect classifier on dataset `test` split.

Key behavior:
- Loads checkpoint and reconstructs ResNet18 head from class list.
- Validates manifest/profile compatibility (unless override flag).
- Supports bundle directory resolution to profile-specific checkpoint.
- Computes and prints metrics + success criteria checks.

Important CLI:
- `--data`, `--model`, `--device`
- `--allow-profile-mismatch`

Outputs/artifacts:
- `report_<model_stem>.json` next to model checkpoint.
- Telemetry events (`eval_start`, `eval_batch`, `eval_done`) when logging is enabled.

## `predict.py`

Purpose:
- Single-image or sample-ID prediction for defect class.

Key behavior:
- Accepts direct image path or dataset lookup by sample ID.
- Can resolve bundle directory model to a profile-specific checkpoint (requires dataset manifest).
- Supports standard checkpoint and ensemble checkpoint format (`simple_sim_ensemble_v1`).
- Optionally prints ground truth if dataset/id provided.

Important CLI:
- `--model`
- `--image` or `--data` + `--id`
- `--device`, `--topk`

Outputs/artifacts:
- Console output only (prediction, top-k probabilities, optional GT/match).

## `batch_predict.py`

Purpose:
- Batch inference and evaluation on a dataset split with history comparison.

Key behavior:
- Runs inference over `train|val|test|all` split.
- Computes full metrics (`accuracy`, `macro_f1`, per-class, confusion data).
- Optional per-sample prediction export (`--save-preds`).
- Optional two-stage-like profile columns using `--profile-model`.
- Handles bundle directory resolution to profile-specific checkpoint.
- Loads prior reports and ranks history by selected metric.

Important CLI:
- `--model`, `--data`, `--split`
- `--batch-size`, `--num-workers`, `--max-samples`
- `--out-dir`, `--save-preds`, `--topk`
- History controls:
  - `--history-scope`, `--history-metric`, `--history-critical-class`
  - `--history-split`, `--history-limit`, `--history-dirs`
- Optional `--profile-model`

Outputs/artifacts:
- `batch_report_<model_stem>_<split>_<timestamp>.json`
- `batch_preds_<model_stem>_<split>_<timestamp>.jsonl` (if `--save-preds`)

## `sample_check.py`

Purpose:
- Quick manual sanity check: sample N examples per class and compare prediction vs GT.

Key behavior:
- Reads `meta.jsonl`/`labels.jsonl`.
- Randomly samples per class for selected split with deterministic seed.
- Runs mini-batch inference and prints row-wise correctness.

Important CLI:
- `--model`, `--data`, `--split`
- `--per-class`, `--seed`, `--batch-size`, `--device`

Outputs/artifacts:
- Console table and summary accuracy only.

## `train_profile.py`

Purpose:
- Trains a profile classifier (component-type classifier) on mixed-profile dataset.

Key behavior:
- Uses `ProfileDataset` from `simple_sim.data_loader`.
- Reuses shared train/validate loops from `scripts/train.py`.
- Requires at least two profiles in data.
- Saves best checkpoint and `_last` checkpoint.
- Emits `train_profile_start` telemetry.

Important CLI:
- `--data`, `--out`, `--device`
- `--amp` / `--no-amp`

Outputs/artifacts:
- Best profile checkpoint at `--out`.
- Last checkpoint at `<stem>_last.pt`.
- Checkpoint tagged with `checkpoint_type=profile_classifier`.

## `eval_profile.py`

Purpose:
- Evaluates profile classifier on test split.

Key behavior:
- Loads profile-classifier checkpoint and profile class names.
- Builds `ProfileDataset` test loader.
- Reuses `evaluate(...)` from `scripts/eval.py`.
- Computes metrics and success criteria.

Important CLI:
- `--data`, `--model`, `--device`

Outputs/artifacts:
- `report_profile_classifier_<model_stem>.json` next to model.

## `predict_two_stage.py`

Purpose:
- End-to-end two-stage prediction:
  1. profile prediction
  2. defect prediction using profile-specific checkpoint from bundle.

Key behavior:
- Wraps `simple_sim.two_stage.TwoStageClassifier`.
- Applies confidence threshold for profile stage (`--min-confidence`) and marks review flag.

Important CLI:
- `--profile-model`
- `--defect-bundle`
- `--image`
- `--device`, `--min-confidence`

Outputs/artifacts:
- Console output for stage-1 and stage-2 probabilities/results.

## `arena_report.py`

Purpose:
- Generates an aggregate markdown report for Arena-tracked models.

Key behavior:
- Reads tracked models from `outputs/models/arena.json`.
- Scans report files (`report_*.json`, `batch_report_*.json`) from model/history and dataset prediction folders.
- Computes best-per-dataset and overall ranking (accuracy/F1/time tie-break).
- Computes model/dataset storage sizes.
- Renders lightweight SVG charts.

Important CLI:
- `--sim-root` (auto-detects repo root if omitted)

Outputs/artifacts:
- `ARENA_REPORT.md` in Simple-Sim root.
- `ARENA_REPORT_assets/*.svg` chart files.

## `render_debug_previews.py`

Purpose:
- Renders per-profile debug preview images for 2D and/or 3D pipelines.

Key behavior:
- Discovers profiles from `configs/profiles`.
- Supports backend modes: `opencv_2d`, `blender_3d`, `both`, `auto`.
- Supports profile selection, auto-selection of missing previews, and interactive/non-interactive workflows.
- Integrates shared filter settings with GUI settings JSON.
- For 3D previews: writes Blender jobs JSONL and runs one batch render.
- Writes/refreshes preview gallery HTML.
- Optional Tkinter viewer with rerun capabilities and optional browser open mode.

Important CLI:
- Discovery/selection:
  - `--profiles-dir`, `--configs-dir`, `--backend`, `--profiles`, `--all`, `--non-interactive`
- Rendering/config:
  - `--config`, `--seed`, `--variable-seeds`, `--states`
  - `--disable-cardinal-rotation-90`
  - `--blender`, `--samples`, `--device`, `--roi`
- Run mode/view:
  - `--out`, `--settings-file`, `--dry-run`, `--view`
  - legacy `--open` / `--no-open`

Outputs/artifacts:
- Preview images under `outputs/debug_previews/previews/<profile_id>/<backend>/<state>.png`.
- Gallery index at `outputs/debug_previews/index.html`.
- 3D job file `outputs/debug_previews/blender_jobs_previews.jsonl`.
- Shared GUI filter settings at `outputs/gui/settings.json` (default path).

## Cross-Script Data Flow

- Dataset generation scripts (`generate.py`, `generate_profile_dataset.py`) produce dataset schema and manifest files consumed by:
  - training/eval/predict scripts
  - GUI tabs
  - history/arena tools
- Training scripts write checkpoints consumed by:
  - `eval.py`, `predict.py`, `batch_predict.py`
  - two-stage pipeline (`predict_two_stage.py`)
- Evaluation scripts write JSON reports consumed by:
  - `batch_predict.py` history comparison
  - `arena_report.py` rankings and charts
- Preview script (`render_debug_previews.py`) is a diagnostic companion for profile/backend validation and GUI filter tuning.

