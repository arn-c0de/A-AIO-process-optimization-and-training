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

See Latest Arena stats : [`Simple-Sim/ARENA_REPORT.md`](Simple-Sim/ARENA_REPORT.md)

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

- **Deterministic generation**: Every sample reproducible from (run_seed, domain, index)
- **Contract-based design**: Strict JSONL/YAML schemas with validation
- **Quality gates**: Validation at each pipeline stage
- **Modular structure**: Training/eval code decoupled from generation
- **Comprehensive metrics**: Accuracy, precision, recall, F1, confusion matrix, FN rates
- **Multi-component profiles (v1.0.1)**: Versioned profiles in `configs/profiles/` (resistor/SOT-23/QFN-32 included)
- **Footprint-aware rendering**: 2-pad (chip), 3-pad (SOT-23), and 4-pad (QFN) layouts driven by the component profile
- **Provenance + safety (v1.0.1)**: Dataset manifests + profile hashing + pipeline guards to prevent profile/class mismatches
- **GUI profile awareness (v1.0.1)**: Profile dropdown, compatibility indicators, dataset/model profile display, info dialogs
- **Batch scoring with history (v1.0.1)**: `scripts/batch_predict.py` with report history compare and class-mismatch guards
- **Profile-filtered model selection (v1.0.1)**: Pipeline model dropdown only shows checkpoints matching the dataset profile
- **Multi-profile bundles**: Merge per-profile checkpoints into a single `.bundle` directory. A bundle is a *container* of per-profile checkpoints named `<profile_id>.pt`. Training into a bundle adds/updates the checkpoint for the dataset profile. Prediction/eval with a bundle resolves the dataset's `component_profile.profile_id` and loads that checkpoint.
- **Single-file ensemble models (new)**: Merge multiple profile checkpoints into one `.pt` that contains an ensemble (multiple sub-models). At inference time, logits are averaged across sub-models. This is useful for quick cross-profile scoring without bundle dispatch, but it is not a replacement for true multi-profile training.
- **Bundle metadata**: Each bundle stores `bundle_details.json` with per-model source info (accuracy, F1, profile hash, source path, size)

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

## Component Profiles (New)

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


```bash
.venv/bin/python scripts/eval.py \
  --data outputs/sim_data/runs/run_0001 \
  --model outputs/models/run_0001.pt
```

**Outputs:**
- Console: Formatted metrics table
- File: `outputs/models/report_run_0001.json`
- Success criteria check results

## Manifests + Pipeline Guards (New)

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

## Bundles vs Ensembles (New)

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
├── README.md
├── requirements.txt
├── configs/
│   ├── profiles/              # Versioned component profiles (YAML)
│   ├── run_0001.yaml          # 0603 resistor reference (schema v2 + profile)
│   ├── run_sot23.yaml         # SOT-23 transistor example
│   └── run_qfn32.yaml         # QFN-32 example (larger ROI + extra defect classes)
├── simple_sim/                # Core package
│   ├── schema.py              # Data contracts
│   ├── config.py              # Config validation
│   ├── rng.py                 # Deterministic seeding
│   ├── dataset_store.py       # Atomic dataset writing
│   ├── defects.py             # Defect classification
│   ├── generator_2d.py        # OpenCV rendering
│   ├── generator_3d.py        # Blender/Cycles batch rendering (invokes Blender once per dataset)
│   ├── blender/               # Blender Python scripts (headless batch renderer)
│   ├── splits.py              # Stratified splitting
│   ├── metrics.py             # Evaluation metrics
│   ├── data_loader.py         # PyTorch Dataset
│   ├── manifest.py            # dataset_manifest.json read/write
│   ├── profile_hash.py        # Deterministic profile hashing + loading
│   └── model_bundle.py         # Multi-model bundle support
├── scripts/
│   ├── generate.py            # Dataset generation
│   ├── train.py               # Model training
│   ├── eval.py                # Test evaluation
│   ├── predict.py             # Single-image prediction
│   └── batch_predict.py       # Batch scoring + report history
├── tools/
│   ├── validate_dataset.py    # Dataset validation
│   ├── backfill_manifest.py   # Add manifest to legacy datasets
│   └── create_multi_dataset.py # Helper for multi-dataset workflows
├── gui/                       # Tkinter multi-tab GUI
│   ├── monitor.py             # Main application (MonitorAppTabbed)
│   ├── state.py               # Shared UI state
│   ├── tabs/
│   │   ├── pipeline_tab.py    # Pipeline control with profile-filtered model selection
│   │   ├── analysis_tab.py    # Image browser with defect overlays
│   │   ├── predictions_tab.py # Batch prediction interface
│   │   ├── weights_tab.py     # Checkpoint management and evaluation
│   │   ├── validation_tab.py  # Automated dataset validation
│   │   └── merge_tab.py       # Multi-profile weight merging into bundles
│   ├── components/            # Reusable UI components
│   └── utils/                 # Settings, tooltips, inference helpers
├── tests/                     # Unit tests
└── outputs/
    ├── sim_data/runs/         # Generated datasets
    └── models/                # Trained models and .bundle directories
```

## GUI Tabs

The GUI is a multi-tab Tkinter application launched via `./gui/run.sh`.

### Pipeline Control

Runs the generation/training/evaluation pipeline. Supports single, multiple, and continuous run modes. Dataset and model selection are profile-aware: the model dropdown only shows checkpoints and bundles whose embedded profile matches the selected dataset. Legacy checkpoints (no profile metadata) are listed at the bottom of the dropdown. Profile compatibility is checked and displayed next to the dataset info. When no compatible model exists (e.g. after switching to a new profile), the "+" button creates a new model entry by name for training.

The tab also exposes a `Render:` selector (`opencv_2d` vs `blender_3d`). The Profile dropdown is filtered based on the selected render backend, and the GUI auto-selects a matching config when possible (based on `run.component_profile` + `render.backend`).

### Analysis

Interactive image browser with defect overlays. Allows browsing generated samples, viewing metadata, and inspecting per-sample defect parameters.

### Predictions

Batch prediction interface. Runs `predict.sh` against a dataset with a selected model and displays per-sample results with confidence scores.

### Weights

Model checkpoint management. Lists all checkpoints (active, snapshots, imports) and multi-profile bundles grouped into categories. Supports snapshot, import, export, rename, duplicate, delete, favorites, and drag-and-drop grouping. Bundles appear with a `[Bundle]` marker and aggregated size. Includes an evaluation runner to score any checkpoint against the current dataset and a compare mode to run two checkpoints side-by-side with a winner summary. Report history is searchable by scope, split, and sort order.

### Validation

Automated dataset validation with flagging. Runs structural and semantic checks on the selected dataset and reports issues.

### Merge

Combines profile-specific weights into a single multi-profile bundle. The tab scans all checkpoints under `outputs/models/`, groups them by their embedded `profile_id`, and organizes them into the same group categories as the Weights tab (Favorites, Snapshots, Imports, custom groups). Each model shows accuracy, F1, size, modification date, and profile hash. For each profile type, exactly one checkpoint can be selected. The merge operation copies the selected checkpoints into a `.bundle` directory and writes both `bundle.json` metadata and `bundle_details.json` with detailed per-model information (source path, accuracy, F1, profile hash, size, modification date). Existing bundles are listed with their included profiles and creation date, and can be inspected or deleted.

A merged bundle appears as a first-class entry in the Weights tab and can be selected as a model in the Pipeline Control tab. When a bundle is used for training, evaluation, or prediction, the pipeline automatically resolves the correct per-profile checkpoint based on the dataset profile.

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

### Training not converging (val acc < 70%)
- Increase dataset size: 200 samples per class (800 total)
- Add dropout or increase L2 regularization
- Check class distribution in splits

### TOMBSTONE not distinguishable
- Increase tilt threshold to 80-85°
- Enhance visual difference (thinner vertical rectangle)
- Add shadows for depth cues

### Determinism verification fails
- Labels must match exactly
- Images only need SSIM > 0.99 (floating-point variations OK)
- Check random seed propagation

### CUDA out of memory
- Reduce batch size in config
- Use `--device cpu` flag
- Reduce image resolution in ROI config

## What's Deferred to M1+

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
