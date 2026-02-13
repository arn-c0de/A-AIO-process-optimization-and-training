#!/bin/bash
# Complete pipeline execution script for Simple-Sim M0 MVP

set -e  # Exit on error

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Ensure torch/torchvision caches are writable (pretrained weights, etc.).
export TORCH_HOME="${TORCH_HOME:-${SCRIPT_DIR}/outputs/torch_cache}"
EVENT_LOG="${EVENT_LOG:-${SCRIPT_DIR}/outputs/live/events.jsonl}"
mkdir -p "$(dirname "${EVENT_LOG}")"
: > "${EVENT_LOG}"
export SIMPLE_SIM_EVENT_LOG="${EVENT_LOG}"

echo "============================================================"
echo "Simple-Sim M0 MVP Pipeline"
echo "============================================================"
echo ""

# Configuration (override via environment)
CONFIG="${CONFIG:-configs/run_0001.yaml}"
DATA_DIR="${DATA_DIR:-outputs/sim_data/runs/run_0001}"
MODEL_PATH="${MODEL_PATH:-outputs/models/run_0001.pt}"

# Virtualenv bootstrap (non-interactive).
#
# Network-free installs:
#   export WHEELHOUSE=/path/to/wheels
#   ./run_pipeline.sh
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

PIP_FLAGS=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
    PIP_FLAGS+=(--no-index --find-links "${WHEELHOUSE}")
fi
if [[ -n "${PIP_FLAGS_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    PIP_FLAGS+=(${PIP_FLAGS_EXTRA})
fi

echo "Installing dependencies from requirements.txt"
if ! "${PYTHON}" -m pip install "${PIP_FLAGS[@]}" -r requirements.txt; then
    echo ""
    echo "Dependency install failed."
    echo "If you are offline, set WHEELHOUSE to a directory of pre-downloaded wheels:"
    echo "  WHEELHOUSE=/path/to/wheelhouse ./run_pipeline.sh"
    exit 1
fi

# Step 1: Generate dataset
echo "============================================================"
echo "STEP 1: Generating dataset"
echo "============================================================"
echo "Config: $CONFIG"
echo "Output: $DATA_DIR"
echo ""

"${PYTHON}" scripts/generate.py \
    --config "$CONFIG" \
    --out "$DATA_DIR"

echo ""
echo "Dataset generation complete"
echo ""

# Step 2: Validate dataset
echo "============================================================"
echo "STEP 2: Validating dataset"
echo "============================================================"
echo ""

"${PYTHON}" tools/validate_dataset.py \
    --data "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "Dataset validation failed"
    exit 1
fi

echo ""
echo "Dataset validation passed"
echo ""

# Step 3: Display sample statistics
echo "============================================================"
echo "STEP 3: Dataset statistics"
echo "============================================================"
echo ""

"${PYTHON}" -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from simple_sim.schema import read_jsonl, LabelRow
from simple_sim.splits import read_split

data_dir = Path('$DATA_DIR')
label_rows = read_jsonl(data_dir / 'labels.jsonl', LabelRow)

# Class distribution
class_counts = {}
for row in label_rows:
    class_counts[row.class_name] = class_counts.get(row.class_name, 0) + 1

print('Overall class distribution:')
for class_name in sorted(class_counts.keys()):
    print(f'  {class_name}: {class_counts[class_name]}')

# Split sizes
print()
print('Split sizes:')
for split_name in ['train', 'val', 'test']:
    split_ids = read_split(data_dir / 'splits' / f'{split_name}.txt')
    print(f'  {split_name}: {len(split_ids)} samples')
"

echo ""

# Step 4: Train model
echo "============================================================"
echo "STEP 4: Training model"
echo "============================================================"
echo "Model will be saved to: $MODEL_PATH"
echo ""

"${PYTHON}" scripts/train.py \
    --data "$DATA_DIR" \
    --out "$MODEL_PATH"

echo ""
echo "Training complete"
echo ""

# Step 5: Evaluate model
echo "============================================================"
echo "STEP 5: Evaluating on test set"
echo "============================================================"
echo ""

"${PYTHON}" scripts/eval.py \
    --data "$DATA_DIR" \
    --model "$MODEL_PATH"

echo ""
echo "Evaluation complete"
echo ""

# Step 6: Summary
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  Dataset: $DATA_DIR"
echo "  Model: $MODEL_PATH"
echo "  Report: outputs/models/report_run_0001.json"
echo ""
echo "Next steps:"
echo "  1. Inspect report: cat outputs/models/report_run_0001.json"
echo "  2. View sample images: ls $DATA_DIR/images/"
echo "  3. Check training metrics in console output above"
echo ""
echo "============================================================"
