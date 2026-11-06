#!/bin/bash
# Start the Simple-Sim Professional Monitor GUI (Multi-tab).
#
# Online:
#   cd Simple-Sim
#   ./gui/run.sh
#
# Offline (use local wheelhouse):
#   cd Simple-Sim
#   WHEELHOUSE=wheelhouse ./gui/run.sh

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/tools/_common.sh"
cd "${SIM_ROOT}"
sim_bootstrap_venv

exec "${PYTHON}" gui/monitor.py
