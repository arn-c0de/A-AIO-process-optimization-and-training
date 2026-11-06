#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
sim_find_python --fallback || { echo "error: no python found" >&2; exit 1; }

exec "$PYTHON" "$SIM_ROOT/tools/profile_editor/app.py" --sim-root "$SIM_ROOT" "$@"
