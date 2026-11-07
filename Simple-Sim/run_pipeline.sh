#!/bin/bash
# Complete pipeline execution script for Simple-Sim M0 MVP

set -e  # Exit on error

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
# shellcheck source=tools/_common.sh
source "${SCRIPT_DIR}/tools/_common.sh"

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
DATASET_MODE="${DATASET_MODE:-new}"

# Virtualenv bootstrap (non-interactive): creates ${VENV_DIR:-.venv} and installs requirements.
# Network-free installs: WHEELHOUSE=/path/to/wheels ./run_pipeline.sh
sim_bootstrap_venv

# Step 1: Generate dataset
echo "============================================================"
echo "STEP 1: Generating dataset"
echo "============================================================"
echo "Config: $CONFIG"
echo "Output: $DATA_DIR"
echo ""

"${PYTHON}" -m simple_sim generate \
    --config "$CONFIG" \
    --out "$DATA_DIR" \
    $([[ "${DATASET_MODE}" == "extend" ]] && echo "--extend")

echo ""
echo "Dataset generation complete"
echo ""

# Step 2: Validate dataset
echo "============================================================"
echo "STEP 2: Validating dataset"
echo "============================================================"
echo ""

if ! "${PYTHON}" -m simple_sim validate --data "$DATA_DIR"; then
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

# DATA_DIR is passed via the environment so paths with quotes don't break the Python source.
DATA_DIR="$DATA_DIR" "${PYTHON}" -c "
import os
from simple_sim.dataset_index import DatasetIndex

index = DatasetIndex(os.environ['DATA_DIR'])
counts = index.class_distribution()
print('Overall class distribution:')
for name in sorted(counts):
    print(f'  {name}: {counts[name]}')
print()
print('Split sizes:')
for name, n in index.split_sizes().items():
    print(f'  {name}: {n} samples')
"

echo ""

# Step 4: Train model
echo "============================================================"
echo "STEP 4: Training model"
echo "============================================================"
echo "Model will be saved to: $MODEL_PATH"
echo ""

"${PYTHON}" -m simple_sim train \
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

"${PYTHON}" -m simple_sim eval \
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
# eval writes the report next to the model (report_<model stem>.json; `eval --json` returns report_path).
REPORT_PATH="$(dirname "$MODEL_PATH")/report_$(basename "${MODEL_PATH%.*}").json"
echo "  Report: $REPORT_PATH"
echo ""
echo "Next steps:"
echo "  1. Inspect report: cat $REPORT_PATH"
echo "  2. View sample images: ls $DATA_DIR/images/"
echo "  3. Check training metrics in console output above"
echo ""
echo "============================================================"
