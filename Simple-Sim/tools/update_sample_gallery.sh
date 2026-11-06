#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if ! sim_find_python --fallback; then
  echo "error: no python found (expected ${SIM_ROOT}/.venv/bin/python or ${REPO_ROOT}/.venv/bin/python)" >&2
  exit 1
fi

cd -- "${SIM_ROOT}"
exec "${PYTHON}" tools/update_sample_gallery.py "$@"
