#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SIM_ROOT}/.." && pwd)"

PY=""
if [[ -x "${SIM_ROOT}/.venv/bin/python" ]]; then
  PY="${SIM_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "error: no python found (expected ${SIM_ROOT}/.venv/bin/python or ${REPO_ROOT}/.venv/bin/python)" >&2
  exit 1
fi

cd -- "${SIM_ROOT}"
exec "${PY}" tools/update_sample_gallery.py "$@"

