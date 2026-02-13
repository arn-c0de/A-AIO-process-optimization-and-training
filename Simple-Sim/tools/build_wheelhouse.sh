#!/bin/bash
# Download wheels for offline installation.
#
# Usage (on a machine with internet):
#   cd Simple-Sim
#   ./tools/build_wheelhouse.sh wheelhouse
#
# Then copy the wheelhouse directory to the offline machine and run:
#   WHEELHOUSE=/path/to/wheelhouse ./run_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${1:-wheelhouse}"
python3 -m venv .venv-wheelhouse
.venv-wheelhouse/bin/python -m pip download -r requirements.txt -d "${OUT_DIR}"

echo "Wheelhouse written to: ${ROOT_DIR}/${OUT_DIR}"
