# Simple-Sim: Synthetic PCB Defect Detection System

Complete M0 MVP implementation for generating synthetic PCB defect datasets, training classifiers, and evaluating performance.

![Simple-Sim Pipeline Dashboard](images/pipeline-dashboard-simple-sim-v1.0.png)

## Overview

Simple-Sim generates deterministic synthetic datasets of PCB component defects for training and evaluating machine learning models. The system uses 2D OpenCV-based rendering to create realistic ROI images with four defect classes:

- **OK**: Component within tolerance
- **MISSING**: Component not present
- **MISALIGNED**: Excessive shift or rotation
- **TOMBSTONE**: Component tilted ≥75°

## Features

- **Deterministic generation**: Every sample reproducible from (run_seed, domain, index)
- **Contract-based design**: Strict JSONL/YAML schemas with validation
- **Quality gates**: Validation at each pipeline stage
- **Modular structure**: Training/eval code decoupled from generation
- **Comprehensive metrics**: Accuracy, precision, recall, F1, confusion matrix, FN rates

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

### 2. Generate Dataset

Generate 400 samples (100 per class) with reference configuration:

```bash
python scripts/generate.py \
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
python tools/validate_dataset.py \
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
python scripts/train.py \
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
python scripts/eval.py \
  --data outputs/sim_data/runs/run_0001 \
  --model outputs/models/run_0001.pt
```

**Outputs:**
- Console: Formatted metrics table
- File: `outputs/models/report_run_0001.json`
- Success criteria check results

## Configuration

Configuration files (`configs/*.yaml`) define all dataset and training parameters. See `configs/run_0001.yaml` for annotated reference.

### Key Sections

- **run**: Run ID, master seed, schema version
- **roi**: Image dimensions, mm/pixel scaling
- **classes**: Sample counts per class
- **tolerances**: Classification thresholds (shift, rotation, tilt)
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
│   └── run_0001.yaml          # Reference configuration
├── simple_sim/                # Core package
│   ├── schema.py              # Data contracts
│   ├── config.py              # Config validation
│   ├── rng.py                 # Deterministic seeding
│   ├── dataset_store.py       # Atomic dataset writing
│   ├── defects.py             # Defect classification
│   ├── generator_2d.py        # OpenCV rendering
│   ├── splits.py              # Stratified splitting
│   ├── metrics.py             # Evaluation metrics
│   └── data_loader.py         # PyTorch Dataset
├── scripts/
│   ├── generate.py            # Dataset generation
│   ├── train.py               # Model training
│   └── eval.py                # Test evaluation
├── tools/
│   └── validate_dataset.py    # Dataset validation
├── tests/                     # Unit tests
└── outputs/
    ├── sim_data/runs/         # Generated datasets
    └── models/                # Trained models
```

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
- BRIDGE defect class (MVP: 4 classes only)

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

MIT License - See LICENSE file for details.
