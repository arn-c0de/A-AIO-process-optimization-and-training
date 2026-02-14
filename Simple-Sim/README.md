# Simple-Sim: Synthetic PCB Defect Detection System

Synthetic AOI-style ROI generation + training/evaluation pipeline for PCB component defect classification.

![Simple-Sim Pipeline Dashboard](images/pipeline-dashboard-simple-sim-v1.0.png)
![Simple-Sim Prediction Dashboard](images/simlesim-prediction-tab.png)
![Simple-Sim A/B Testing Dashboard](images/simple-sim-a-b-testing.png)


## Overview

Simple-Sim generates deterministic synthetic datasets of PCB component defects for training and evaluating machine learning models.

Key idea: the pipeline is **profile-based**. A **component profile** (versioned YAML) defines the component geometry/tolerances/render defaults and the intended defect set for a component type (e.g. 0603 resistor, SOT-23 transistor, QFN-32 IC). Run configs select a profile via `run.component_profile`.

The system uses 2D OpenCV-based rendering to create AOI-like ROI images. Defect classes are **component-dependent**; common classes include:

- **OK**: Component within tolerance
- **MISSING**: Component not present
- **MISALIGNED**: Excessive shift or rotation
- **TOMBSTONE**: Component tilted beyond profile tolerance
- **SOLDER_BRIDGE**: Shorts between pads/leads (e.g. QFN)
- **CORNER_LIFT**: Lifted corner / partial non-wet (e.g. QFN)

## Features

- **Deterministic generation**: Every sample reproducible from (run_seed, domain, index)
- **Contract-based design**: Strict JSONL/YAML schemas with validation
- **Quality gates**: Validation at each pipeline stage
- **Modular structure**: Training/eval code decoupled from generation
- **Comprehensive metrics**: Accuracy, precision, recall, F1, confusion matrix, FN rates
- **Multi-component profiles (v1.0.1)**: Versioned profiles in `configs/profiles/` (resistor/SOT-23/QFN-32 included)
- **Provenance + safety (v1.0.1)**: Dataset manifests + profile hashing + pipeline guards to prevent profile/class mismatches
- **GUI profile awareness (v1.0.1)**: Profile dropdown, compatibility indicators, dataset/model profile display, info dialogs
- **Batch scoring with history (v1.0.1)**: `scripts/batch_predict.py` with report history compare and class-mismatch guards

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
- `sot23_transistor@1`: 3-lead SOT-23 transistor (OK/MISSING/MISALIGNED/TOMBSTONE)
- `qfn32_ic@1`: QFN-32 IC (OK/MISSING/MISALIGNED/SOLDER_BRIDGE/CORNER_LIFT)

Reference run configs you can start from:
- `configs/run_0001.yaml` (0603 resistor, 256×256 ROI)
- `configs/run_sot23.yaml` (SOT-23 transistor, 256×256 ROI)
- `configs/run_qfn32.yaml` (QFN-32 IC, 768×768 ROI)

## Licensing / Third-Party Dependencies

- This repository is **proprietary** (see `../LICENSE`).
- Python dependencies in `requirements.txt` are **third-party software** under their own licenses (see `THIRD_PARTY_LICENSES.md`).
- `wheelhouse/` is intentionally **not** committed to git (it is ignored) so users build/download their own wheels as needed.

## Production Data Notes (For Later)

If/when you train on real production AOI images:

- Treat PCB images, layouts, and BOM-related visuals as **confidential** by default.
- Keep datasets and trained weights in controlled storage (access control, audit, backups).
- Define retention and deletion rules (especially for failed builds/experiments).
- If any images/metadata could include people, screens, or workspace context, check privacy/legal requirements before broader distribution.

## Live GUI Monitor (Optional)

If you want a live view of progress + the latest images during generation/training/eval:

```bash
cd Simple-Sim
./gui/run.sh
```

Notes:
- This uses `tkinter`. On Ubuntu/Debian you may need: `sudo apt-get install python3-tk`

## Single Image Prediction

```bash
cd Simple-Sim
.venv/bin/python scripts/predict.py --model outputs/models/run_0001.pt --image outputs/sim_data/runs/run_0001/images/000000.png
```

## Batch Prediction / Scoring (Like Eval, With History)

```bash
cd Simple-Sim
./predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test --save-preds
```

History table options:
```bash
./predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test \
  --history-scope model --history-metric macro_f1 --history-limit 10

./predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test \
  --history-scope dataset --history-metric accuracy --history-split any --history-limit 20

./predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test \
  --history-scope all --history-metric critical_fn_rate --history-critical-class MISALIGNED --history-limit 10
```

Notes:
- `scripts/batch_predict.py` includes guards to prevent scoring a dataset whose class list does not match the model checkpoint.
- `--model` may also point to a **model bundle directory** (multi-profile). In that case the script resolves the best checkpoint for the dataset profile using `dataset_manifest.json`.

### 2. Generate Dataset

Generate 400 samples (100 per class) with reference configuration:

```bash
.venv/bin/python scripts/generate.py \
  --config configs/run_0001.yaml \
  --out outputs/sim_data/runs/run_0001
```

**Expected output:**
- 400 images (256×256 px) in ~2 minutes
- 70/15/15 train/val/test split
- JSONL metadata and labels
- Stratified splits by (domain, class)

### 3. Validate Dataset

Run quality checks on generated dataset:

```bash
.venv/bin/python tools/validate_dataset.py \
  --data outputs/sim_data/runs/run_0001
```

**Checks performed:**
- Schema validation (JSONL parsing)
- Row count matching
- ID consistency
- Image file existence and readability
- Split overlap detection
- Class distribution
- Determinism verification

### 4. Train Model

Train ResNet18 classifier for 10 epochs:

```bash
.venv/bin/python scripts/train.py \
  --data outputs/sim_data/runs/run_0001 \
  --out outputs/models/run_0001.pt
```

**Expected results:**
- Training completes without errors
- Val accuracy > 70% (target: > 90%)
- Best model saved by macro-F1 score

### 5. Evaluate on Test Set

Run frozen test evaluation:

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
  "schema_version": 1,
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
  "augment": {...}
}
```

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
├── tests/                     # Unit tests
└── outputs/
    ├── sim_data/runs/         # Generated datasets
    └── models/                # Trained models
```

## GUI Profile Support (New)

The GUI is profile-aware:
- Pipeline tab includes a profile selector (and shows the selected dataset/model profiles)
- Model picker is filtered to show only models compatible with the selected profile
- Compatibility indicators and info dialogs help prevent accidentally mixing components

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
- 3D rendering with Blender (MVP: 2D OpenCV only)
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
