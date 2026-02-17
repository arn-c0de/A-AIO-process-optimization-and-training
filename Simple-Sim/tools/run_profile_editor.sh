#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "$HERE/.." && pwd)"

PY="${SIM_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

exec "$PY" "$SIM_ROOT/tools/profile_editor/app.py" --sim-root "$SIM_ROOT" "$@"
