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

# shellcheck source=tools/_common.sh
source "${SCRIPT_DIR}/tools/_common.sh"
sim_require_python

exec "${PYTHON}" scripts/batch_predict.py "$@"
