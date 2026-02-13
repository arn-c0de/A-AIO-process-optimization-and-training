#!/bin/bash
# Sample a few images per class from a dataset split and check predictions vs GT.
#
# Examples:
#   cd Simple-Sim
#   ./sample_predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test --per-class 5
#
# Faster/smaller batch:
#   ./sample_predict.sh --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --split test --per-class 3 --batch-size 16

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=".venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}."
  echo "Install deps first:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

exec "${PYTHON}" scripts/sample_check.py "$@"

