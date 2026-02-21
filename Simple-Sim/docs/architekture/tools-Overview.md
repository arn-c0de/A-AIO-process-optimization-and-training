# Tools Overview

This document describes all executable tools under `Simple-Sim/tools`, including purpose, inputs/outputs, and integration points.

## Directory Purpose

`Simple-Sim/tools` contains support utilities outside the main train/eval pipeline in `scripts/`.  
Typical responsibilities:
- migration/backfill for legacy datasets
- dataset validation and merge helpers
- documentation and gallery generation
- local environment helper scripts
- profile editor launcher + implementation

## Tool Groups

- Dataset migration and repair
  - `backfill_manifest.py`
  - `backfill_labels_v2.py`
- Dataset quality and composition
  - `validate_dataset.py`
  - `create_multi_dataset.py`
- Documentation and sample assets
  - `update_architektur_md.sh`
  - `update_sample_gallery.py`
  - `update_sample_gallery.sh`
- Environment/tooling helpers
  - `build_wheelhouse.sh`
- Interactive editor
  - `run_profile_editor.sh`
  - `profile_editor/*`

## Dataset Migration and Repair

## `tools/backfill_manifest.py`

Purpose:
- Creates `dataset_manifest.json` for legacy datasets that do not have one yet.

Key behavior:
- Supports single dataset (`--data`) and batch mode (`--data-root`).
- Reads `config.yaml`, `meta.jsonl`, `labels.jsonl`.
- Resolves profile ID (`run.component_profile` or `--profile` fallback).
- Computes profile hash via `simple_sim.profile_hash.hash_profile`.
- Writes manifest v1 with dataset stats and extend history.

Important CLI:
- `--data`
- `--data-root`
- `--profile`
- `--profiles-dir`

Outputs/artifacts:
- `dataset_manifest.json` in each processed dataset folder.
- Console summary with success/skip/fail counts.

## `tools/backfill_labels_v2.py`

Purpose:
- Upgrades `labels.jsonl` rows to schema v2 by adding `profile_id`.

Key behavior:
- Reads `dataset_manifest.json` and extracts `component_profile.profile_id`.
- Rewrites `labels.jsonl` using `LabelRow(schema_version=2, ..., profile_id=...)`.
- No-op if labels are already v2.

Important CLI:
- `--data`

Outputs/artifacts:
- Updated `labels.jsonl` in-place.

## Dataset Quality and Composition

## `tools/validate_dataset.py`

Purpose:
- Performs end-to-end dataset integrity checks.

Key behavior:
- Verifies required files/folders (`meta.jsonl`, `labels.jsonl`, `config.yaml`, `splits/`, `images/`).
- Validates JSONL schemas and ID consistency.
- Checks image existence/readability via OpenCV.
- Ensures no train/val/test split overlap.
- Reports split class coverage warnings.
- Performs deterministic seed spot-checks (with relaxed mode for manifest v2/extended datasets).
- Emits telemetry (`validate_start`, `validate_done`).

Important CLI:
- `--data`
- `--seed-mismatch-log-limit`

Outputs/artifacts:
- Console validation report.
- Exit code `0` on pass, `1` on failure.

## `tools/create_multi_dataset.py`

Purpose:
- Merges multiple existing datasets into one output dataset.

Key behavior:
- Copies source images into unified `images/` folder.
- Rewrites sample IDs and split-local indexing for merged run.
- Writes merged `meta.jsonl`, `labels.jsonl`, `splits/*.txt`, `config.yaml`.
- Writes `dataset_manifest.json` using `write_dataset_manifest`.

Important CLI:
- `--datasets <list...>`
- `--out`
- `--config` (default `configs/run_multi.yaml`)

Outputs/artifacts:
- New dataset directory at `--out`.

## Documentation and Sample Assets

## `tools/update_architektur_md.sh`

Purpose:
- Auto-generates `Simple-Sim/architektur.md` from project tree snapshots.

Key behavior:
- Collects directory/file structure using `find` with practical depth limits.
- Excludes transient noise (`__pycache__`, `.venv`, `.pytest_cache`, etc.).
- Renders markdown with timestamp, conventions, and auto-generated tree.

Outputs/artifacts:
- `architektur.md` (overwritten on each run).

## `tools/update_sample_gallery.py`

Purpose:
- Rebuilds `SAMPLE_GALLERY.md` and profile-class sample images in `images/samples/`.

Key behavior:
- Discovers all profile YAMLs in `configs/profiles`.
- Detects profile backend (`opencv_2d` or `blender_3d`).
- Generates one sample per defect class (typically `OK`, `MISSING`, `MISALIGNED`, `TOMBSTONE`).
- For 2D: renders directly via OpenCV generator.
- For 3D: stages Blender jobs, renders in grouped batches, then draws defect overlays.
- Rewrites gallery markdown with TOC and embedded images.

Important CLI:
- `--profiles-dir`, `--configs-dir`
- `--out-images-dir`, `--out-md`
- `--seed`, `--roi`
- `--blender`, `--samples`, `--device`
- `--skip-2d`, `--skip-3d`, `--dry-run`

Outputs/artifacts:
- `images/samples/<profile_id>/<CLASS>.png`
- `SAMPLE_GALLERY.md`
- temporary jobs in `outputs/sample_gallery_jobs_*.jsonl`

## `tools/update_sample_gallery.sh`

Purpose:
- Wrapper that resolves an available Python interpreter and runs `update_sample_gallery.py`.

Key behavior:
- Prefers `Simple-Sim/.venv/bin/python`, then repo `.venv`, then `python3`.
- Changes into `Simple-Sim/` before execution.

## Environment and Tooling Helpers

## `tools/build_wheelhouse.sh`

Purpose:
- Downloads Python wheels for offline installation.

Key behavior:
- Creates `.venv-wheelhouse`.
- Runs `pip download -r requirements.txt -d <out_dir>`.

Important usage:
- `./tools/build_wheelhouse.sh wheelhouse`

Outputs/artifacts:
- Wheel files in output directory (default `wheelhouse/`).

## Interactive Editor

## `tools/run_profile_editor.sh`

Purpose:
- Launcher for profile editor GUI.

Key behavior:
- Uses `Simple-Sim/.venv/bin/python` when available, otherwise `python3`.
- Executes `tools/profile_editor/app.py --sim-root <Simple-Sim>`.

## `tools/profile_editor/` package

Purpose:
- Interactive profile/run editor with synchronized form, YAML, fast 3D preview, and Blender HQ preview.

Main modules:
- `app.py`
  - Tkinter app shell, layout wiring, actions, session settings, debounce logic.
- `sync_controller.py`
  - Coordinates registry and state operations (load/save/refresh).
- `profile_registry.py`
  - Discovers profile/run YAML files and mapping between profile IDs and run files.
- `state_store.py`
  - Single source-of-truth for profile/run documents, dirty state, YAML apply, field updates.
- `form_renderer.py`
  - Dynamic field form generated from YAML leaf paths.
- `yaml_editor.py`
  - Dual YAML text editors (profile/run) with auto-apply and parse error display.
- `preview_3d.py`
  - Fast interactive 3D preview canvas (orbit/pan/zoom and component visibility).
- `hq_preview_panel.py`
  - Displays rendered HQ preview image + status text.
- `blender_live_preview.py`
  - Background worker that prepares jobs and calls Blender batch rendering.
- `system_monitor.py`
  - CPU/GPU/RAM sampling for live status.
- `yaml_io.py`
  - YAML load/dump/save with atomic write behavior.

Reference:
- Detailed usage and feature guide: `tools/profile_editor/README.md`.
