#!/bin/bash
# Batch predict/evaluate a trained model on a dataset split.
#
# Examples:
#   cd Simple-Sim
#   ./predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test
#
# Write per-sample predictions:
#   ./predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test --save-preds

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=".venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}."
  echo "Run the pipeline once (./run_pipeline.sh) or install deps:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

exec "${PYTHON}" scripts/batch_predict.py "$@"

