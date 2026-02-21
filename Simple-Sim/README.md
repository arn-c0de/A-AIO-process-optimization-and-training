# Simple-Sim: Synthetic PCB Defect Detection System

Synthetic AOI-style ROI generation + training/evaluation pipeline for PCB component defect classification.

![Simple-Sim Pipeline Dashboard](images/pipeline-dashboard-simple-sim-v1.0.png)
*Note:* Pipeline overview showing the end-to-end workflow (generate, validate, train, evaluate) and run status.
![Simple-Sim Prediction Dashboard](images/simlesim-prediction-tab.png)
*Note:* Prediction view for scoring an ROI image and inspecting the predicted class and confidence.
![Simple-Sim A/B Testing Dashboard](images/simple-sim-a-b-testing.png)
*Note:* A/B testing view to compare two runs/models side-by-side using the same evaluation data.
![Simple-Sim Weight Merge Dashboard](images/simple-sim-weight-merge.png)
*Note:* Weight merge view to combine per-profile checkpoints into a bundle directory or a single-file ensemble model.

See [`Sample Gallery`](SAMPLE_GALLERY.md) for auto-generated reference images with defect overlays from all available datasets and profiles.
See [`Changelog`](CHANGELOG.md) for a version-by-version overview of changes.

See latest Arena stats: [`ARENA_REPORT.md`](ARENA_REPORT.md)

## Contents

- [Completed Testing Research](#completed-testing-research)
- [Documentation](#documentation)
- [Overview](#overview)
- [Features](#features)
- [Extended Image Filters (Current Research)](#extended-image-filters-current-research)
- [Quick Start](#quick-start)
- [Component Profiles](#component-profiles)
- [Manifests and Pipeline Guards](#manifests-and-pipeline-guards)
- [Bundles vs Ensembles](#bundles-vs-ensembles)
- [Production Deployment: Bundles vs Single Models](#production-deployment-bundles-vs-single-models)
- [Configuration](#configuration)
- [Dataset Schema](#dataset-schema)
- [Project Structure](#project-structure)
- [GUI Tabs](#gui-tabs)
- [Success Criteria](#success-criteria)
- [Troubleshooting](#troubleshooting)
- [Deferred to M1+](#deferred-to-m1)
- [Testing](#testing)
- [License](#license)

## Completed Testing Research
- Completed training restart log (same datasets, with random 90° orientation per image + SOIC16 profile testing): [`docs/training_logs/2026-02-16-Training-Restart-90deg-SOIC16.md`](docs/training_logs/2026-02-16-Training-Restart-90deg-SOIC16.md)
- Current research: renewed training restart with extended image filter options to improve robustness/generalization (living research log): [`docs/training_logs/2026-02-17-Training-Restart-Extended-Image-Filters.md`](docs/training_logs/2026-02-17-Training-Restart-Extended-Image-Filters.md)  
  Filter configuration and implementation reference: [`docs/guides/FILTER_SETTINGS.md`](docs/guides/FILTER_SETTINGS.md)
- Training logs index: [`docs/training_logs/INDEX.md`](docs/training_logs/INDEX.md)

## Documentation

For comprehensive guides and references, see [`Documentation Index`](docs/INDEX.md). Quick links:
- [Command Cheatsheet](docs/guides/CHEATSHEET.md) - Common operations and workflows
- [3D Rendering Quickstart](docs/guides/3D_RENDERING_QUICKSTART.md) - Blender 3D rendering guide
- [Profile System Implementation](PROFILE_SYSTEM_IMPLEMENTATION.md) - Component profile design


## Overview

Simple-Sim generates deterministic synthetic datasets of PCB component defects for training and evaluating machine learning models.

Key idea: the pipeline is **profile-based**. A **component profile** (versioned YAML) defines the component geometry/tolerances/render defaults and the intended defect set for a component type (e.g. 0603 resistor, SOT-23 transistor, QFN-32 IC). Run configs select a profile via `run.component_profile`.

The system supports two rendering backends to create AOI-like ROI images:

- `opencv_2d`: 2D OpenCV renderer (fast, default)
- `blender_3d`: 3D physically-based renderer via Blender/Cycles (batch rendering)

Defect classes are **component-dependent**; common classes include:

- **OK**: Component within tolerance
- **MISSING**: Component not present
- **MISALIGNED**: Excessive shift or rotation
- **TOMBSTONE**: Component tilted beyond profile tolerance

## Features

- **Data reliability**: Deterministic generation from `(run_seed, domain, index)`.
- **Data contracts**: Strict JSONL/YAML validation and pipeline quality gates.
- **Pipeline design**: Generation, training, and evaluation are cleanly separated.
- **Metrics**: Accuracy, precision, recall, F1, confusion matrix, and FN rates.
- **Profiles (v1.0.1)**: Versioned component profiles in `configs/profiles/` (e.g. resistor, SOT-23, QFN-32).
- **Footprint-aware rendering**: 2-pad (chip), 3-pad (SOT-23), and 4-pad (QFN) layouts from profile definitions.
- **Provenance + safety (v1.0.1)**: `dataset_manifest.json`, profile hashing, and mismatch guards.
- **GUI profile awareness (v1.0.1)**: Profile compatibility indicators for datasets/models.
- **Batch scoring history (v1.0.1)**: `scripts/batch_predict.py` with report history and class-mismatch guards.
- **Profile-filtered model selection (v1.0.1)**: Pipeline model list prioritizes compatible checkpoints.
- **Multi-profile bundles**: `.bundle` directories with one checkpoint per profile (`<profile_id>.pt`) and automatic profile dispatch.
- **Single-file ensembles**: One `.pt` with multiple sub-models; inference averages logits across members.
- **Bundle metadata**: `bundle_details.json` with source path, profile hash, size, and key metrics.
- **Extended filter research**: Advanced image filter pipeline with persisted profiles and randomized ranges.

## Extended Image Filters (Current Research)

Simple-Sim now supports an extended filter set for synthetic image robustness testing and training restarts.

New filter families include:
- Perspective transform / skew
- Motion blur
- Saturation / hue shift
- Shadow / occlusion
- Reflection / glare
- Vignetting
- Chromatic aberration
- JPEG compression artifacts
- Color temperature variation
- Lens distortion
- Dust / dirt particles
- Sharpen

Each filter supports:
- Enable/disable toggles
- Strength/value parameters
- Optional randomized min/max ranges per filter key in filter profiles

Reference:
- Filter keys, parameter ranges, priority, and integration notes: [`docs/guides/FILTER_SETTINGS.md`](docs/guides/FILTER_SETTINGS.md)
- Current restart research log using these filters: [`docs/training_logs/2026-02-17-Training-Restart-Extended-Image-Filters.md`](docs/training_logs/2026-02-17-Training-Restart-Extended-Image-Filters.md)

## Quick Start

### 1. Installation

```bash
cd Simple-Sim
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Offline install (no PyPI access):
```bash
cd Simple-Sim
python -m venv .venv
WHEELHOUSE=/path/to/wheelhouse .venv/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.txt
```

To build a wheelhouse on a machine with internet:
```bash
cd Simple-Sim
./tools/build_wheelhouse.sh wheelhouse
```

### 2. Update Architecture Documentation

```bash
cd Simple-Sim
./tools/update_architektur_md.sh
```

### 3. Evaluate a Trained Model

```bash
.venv/bin/python scripts/eval.py \
  --data outputs/sim_data/runs/run_0001 \
  --model outputs/models/run_0001.pt
```

**Outputs:**
- Console: Formatted metrics table
- File: `outputs/models/report_run_0001.json`
- Success criteria check results

## Component Profiles

Profiles live in `configs/profiles/` and are versioned via `profile_id` like `chip_0603_resistor@1`.

Included profiles:
- `chip_0603_resistor@1`: 2-pad 0603 chip resistor (OK/MISSING/MISALIGNED/TOMBSTONE)
- `sot23_transistor@1`: 3-pad SOT-23 transistor (OK/MISSING/MISALIGNED/TOMBSTONE)
- `qfn32_ic@1`: 4-pad QFN-32 IC (OK/MISSING/MISALIGNED/TOMBSTONE)

3D profiles are intentionally **separate profile IDs** to keep datasets distinct in the GUI:
- `chip_0603_resistor_3d@1`
- `sot23_transistor_3d@1`
- `qfn32_ic_3d@1`

Reference run configs you can start from:
- `configs/run_0001.yaml` (0603 resistor, 256×256 ROI)
- `configs/run_sot23.yaml` (SOT-23 transistor, 256×256 ROI)
- `configs/run_qfn32.yaml` (QFN-32 IC, 768×768 ROI)

3D reference configs:
- `configs/run_0001_3d.yaml`
- `configs/run_sot23_3d.yaml`
- `configs/run_qfn32_3d.yaml`

## Manifests and Pipeline Guards

Each generated dataset includes a `dataset_manifest.json` with provenance (profile id/hash/path, generator version, git commit) and extend history.

The CLI scripts include guards to prevent silent corruption, for example:
- `scripts/generate.py --extend ...` refuses to extend datasets if the profile id/hash differs
- `scripts/train.py --resume ...` refuses to resume across profile mismatches
- `scripts/predict.py` / `scripts/eval.py` / `scripts/batch_predict.py` validate dataset/model compatibility

Legacy datasets (without a manifest) can be migrated:
```bash
cd Simple-Sim
.venv/bin/python tools/backfill_manifest.py --data outputs/sim_data/runs/run_0001
```

Implementation notes: `PROFILE_SYSTEM_IMPLEMENTATION.md`.

## Bundles vs Ensembles

Simple-Sim supports two ways to "combine" multiple profile-specific checkpoints:

1. **Bundle directory** (`*.bundle/`)
   - Contains one checkpoint per profile: `chip_0603_resistor@1.pt`, `sot23_transistor@1.pt`, ...
   - Predict/eval resolves the dataset profile from `dataset_manifest.json` and loads the matching file.
   - If the bundle does not include a checkpoint for the dataset profile, you must train that profile into the bundle.

2. **Single-file ensemble** (`*_ensemble.pt`)
   - One `.pt` file that stores multiple sub-model checkpoints.
   - Predict/eval averages logits across sub-models.
   - Useful for experiments and broad scoring, but does not magically learn unseen component types.

Full documentation: [`docs/guides/MODEL_MERGE_BUNDLES_ENSEMBLES.md`](docs/guides/MODEL_MERGE_BUNDLES_ENSEMBLES.md).

## Production Deployment: Bundles vs Single Models

### Bundled Models (Multi-Profile)
- Require an additional **object classification model** upstream to identify component type
- Once component type is known, bundle dispatches to the correct per-profile checkpoint
- Higher accuracy per component type
- Larger model footprint (multiple sub-models)

### Single Models (Cross-Profile)
- **Self-contained**: No external classifier needed
- Trained on **all datasets combined** (all profiles, all defect types)
- **Much smaller model size** than bundles
- Slightly lower per-component accuracy, but **significantly better at randomized mixed-component recognition**
- Ideal for edge deployment, real-world PCB defect detection where component type is unknown

**Recommendation**: For production, use single models (e.g., `random-datacrawler-v1`) trained on all datasets. They sacrifice per-component peak accuracy for robustness and simplicity.

## Configuration

Configuration files (`configs/*.yaml`) define all dataset and training parameters. See `configs/run_0001.yaml` for annotated reference.

### Key Sections

- **run**: Run ID, master seed, schema version
- **run.component_profile (v2)**: Component profile ID from `configs/profiles/`
- **roi**: Image dimensions, mm/pixel scaling
- **classes**: Sample counts per class
- **tolerances**: Classification thresholds (shift, rotation, tilt) (profile-provided in schema v2)
- **domains**: Lighting, blur, noise ranges per domain
- **splits**: Train/val/test fractions and domain assignments
- **render**: Colors, rendering backend
- **augment**: Rotation, brightness, contrast ranges
- **train**: Model, epochs, batch size, optimizer params
- **eval**: Batch size, metrics to compute

## Dataset Schema

### meta.jsonl
One row per sample with complete generation metadata:

```json
{
  "schema_version": 2,
  "id": "run_0001/domain_A/train/000042",
  "run_id": "run_0001",
  "domain": "domain_A",
  "split": "train",
  "seed": 1234567890123456789,
  "image_path": "images/000042.png",
  "render_backend": "opencv_2d",
  "footprint": "0603",
  "nominal": {...},
  "defect": {...},
  "augment": {...},
  "render_meta": {}
}
```

For `render_backend: blender_3d`, `render_meta` stores backend-specific information (e.g. Cycles samples/device and 3D scene defaults from the selected 3D profile).

### labels.jsonl
One row per sample with class label:

```json
{
  "schema_version": 1,
  "id": "run_0001/domain_A/train/000042",
  "class_name": "MISALIGNED"
}
```

### splits/
- `train.txt`: Training sample IDs
- `val.txt`: Validation sample IDs
- `test.txt`: Test sample IDs

## Project Structure

```
Simple-Sim/
├── configs/       # Run-Configs + Profile-YAMLs
├── simple_sim/    # Core package (generation/training/eval helpers)
├── scripts/       # CLI entrypoints (generate/train/eval/predict)
├── tools/         # Validation, migration, profile-editor utilities
├── gui/           # Tkinter app (tabs/components/utils)
├── tests/         # Unit + integration tests
├── docs/          # Guides, logs, technical notes
├── images/        # README/UI screenshots + sample galleries
├── third_party/   # Third-party notices
├── wheelhouse/    # Offline dependency wheels
└── outputs/
    ├── sim_data/runs/  # Generated datasets
    └── models/         # Trained models + .bundle directories
```

Full architecture (detailed folder/file structure + `Last update`): [`architektur.md`](architektur.md)

## GUI Tabs

The GUI is a multi-tab Tkinter application launched via `./gui/run.sh`.

### Pipeline Control

- Starts generation/training/evaluation in single, multi, or continuous mode.
- Model picker is profile-aware; it shows only compatible checkpoints/bundles first.
- Legacy checkpoints (without profile metadata) are listed separately.
- If no compatible model exists, `+` creates a new model entry.
- `Render:` selector (`opencv_2d` / `blender_3d`) filters profile choices and auto-selects matching configs.

### Analysis

- Browse generated images with defect overlays.
- Inspect metadata and per-sample defect parameters.

### Predictions

- Run batch prediction with selected dataset/model.
- Shows per-sample class predictions and confidence values.

### Weights

- Manage checkpoints and bundles (active, snapshots, imports, custom groups).
- Actions: snapshot/import/export/rename/duplicate/delete/favorites/drag-and-drop grouping.
- Includes evaluation runner and side-by-side compare mode.
- Report history is filterable/searchable (scope, split, sort).

### Validation

- Runs structural and semantic dataset checks.
- Reports findings and flags inconsistencies.

### Merge

- Merges profile-specific checkpoints into one `.bundle`.
- Scans `outputs/models/`, groups by `profile_id`, and lets you pick one checkpoint per profile.
- Writes `bundle.json` and `bundle_details.json` (source, metrics, hash, size, date).
- Bundles are selectable in Weights and Pipeline tabs; runtime auto-resolves the matching profile checkpoint.

## Success Criteria

### Must-Have (Blocking)
- [x] 400 images generated in < 5 minutes
- [x] All JSONL rows valid (schema validation passes)
- [x] No split overlap (validation check passes)
- [x] Deterministic: same config → same labels
- [x] Training completes without errors
- [x] Val accuracy > 70% after 10 epochs
- [x] Test metrics saved to report.json with confusion matrix

### Nice-to-Have
- Val accuracy > 90% (indicates good separability)
- TOMBSTONE recall > 0.8 (validates 2D simulation quality)
- Generation speed > 100 samples/sec

## Troubleshooting

### Training accuracy not converging (`val_acc < 70%`)
- Increase dataset size: 200 samples per class (800 total)
- Add dropout or increase L2 regularization
- Check class distribution in splits

### `TOMBSTONE` class not distinguishable
- Increase tilt threshold to 80-85°
- Enhance visual difference (thinner vertical rectangle)
- Add shadows for depth cues

### Determinism verification fails
- Labels must match exactly
- Images only need SSIM > 0.99 (floating-point variations OK)
- Check random seed propagation

### CUDA out of memory
- Reduce `train.batch_size` in config
- Use CPU mode with `--device cpu`
- Reduce ROI image resolution in config

## Deferred to M1+

- Multi-domain generation (MVP: single domain_A)
- Challenge sets (MVP: train/val/test only)
- More advanced 3D physics/geometry (current Blender backend is an early MVP batch renderer)
- Dashboard/TensorBoard (MVP: console + JSON reports)
- Hyperparameter sweeps (MVP: single config)
- Advanced augmentation (MVP: basic blur/noise/brightness)
- More defect types (e.g. opens, insufficient solder, polarity/marking)

## Testing

Run unit tests:

```bash
.venv/bin/python -m pytest tests/
```

Run end-to-end pipeline test:

```bash
.venv/bin/python -m pytest tests/test_pipeline_e2e.py -v
```

## License

This project is **proprietary**. No permission is granted to use, copy, modify,
or distribute this software without prior written permission.

See [`../LICENSE`](../LICENSE).

Third-party dependencies remain under their own licenses; see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
