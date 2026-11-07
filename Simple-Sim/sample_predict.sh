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

# shellcheck source=tools/_common.sh
source "${SCRIPT_DIR}/tools/_common.sh"
sim_require_python

exec "${PYTHON}" -m simple_sim sample-check "$@"
