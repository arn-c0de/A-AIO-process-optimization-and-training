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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

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

echo "Installing dependencies (if needed)"
"${PYTHON}" -m pip install "${PIP_FLAGS[@]}" -r requirements.txt

exec "${PYTHON}" gui/monitor.py

