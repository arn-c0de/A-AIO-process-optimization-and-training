#!/bin/bash
# Root convenience launcher for Simple-Sim.
#
# Default: start the GUI monitor.
#   cd Simple-Sim
#   ./run.sh
#
# Offline:
#   cd Simple-Sim
#   WHEELHOUSE=wheelhouse ./run.sh
#
# Pipeline without GUI:
#   cd Simple-Sim
#   WHEELHOUSE=wheelhouse ./run.sh pipeline

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-gui}"

case "${MODE}" in
  gui)
    exec ./gui/run.sh
    ;;
  pipeline)
    exec ./run_pipeline.sh
    ;;
  *)
    echo "Usage: ./run.sh [gui|pipeline]"
    exit 2
    ;;
esac

