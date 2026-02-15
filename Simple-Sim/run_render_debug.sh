#!/usr/bin/env bash
set -euo pipefail

# Runs the 3D debug preview renderer with a local venv, installing only the
# missing Python packages needed for the preview script.
#
# Usage examples:
#   ./run_render_debug.sh
#   ./run_render_debug.sh --all --non-interactive --dry-run
#   ./run_render_debug.sh --profiles chip_0603_resistor_3d@1 --non-interactive

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

REQ_FILE="$HERE/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[error] Missing requirements file: $REQ_FILE" >&2
  exit 2
fi

VENV_DIR=""
for cand in "venv" ".venv"; do
  if [[ -x "$HERE/$cand/bin/python" ]]; then
    VENV_DIR="$HERE/$cand"
    break
  fi
done
if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$HERE/.venv"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

req_has() {
  # case-insensitive match of a bare requirement name at start of line
  local name="$1"
  grep -Eiq "^${name}([<=> ].*)?$" "$REQ_FILE" >/dev/null 2>&1
}

missing_mods=()
python - <<'PY' || missing_mods+=("numpy")
import numpy  # noqa: F401
PY
python - <<'PY' || missing_mods+=("yaml")
import yaml  # noqa: F401
PY
python - <<'PY' || missing_mods+=("pil")
from PIL import Image  # noqa: F401
PY

if (( ${#missing_mods[@]} )); then
  echo "[info] Missing Python modules in venv: ${missing_mods[*]}"
  python -m pip install -q --upgrade pip

  # Install minimal deps (fast) but ensure they're listed in requirements.txt.
  to_install=()
  for m in "${missing_mods[@]}"; do
    case "$m" in
      numpy)
        if req_has "numpy"; then to_install+=("numpy"); fi
        ;;
      yaml)
        if req_has "PyYAML"; then to_install+=("PyYAML"); fi
        ;;
      pil)
        if req_has "pillow"; then to_install+=("pillow"); fi
        ;;
    esac
  done

  if (( ${#to_install[@]} )); then
    python -m pip install "${to_install[@]}"
  else
    echo "[warn] Could not map missing modules to requirements.txt entries; falling back to full requirements install." >&2
    python -m pip install -r "$REQ_FILE"
  fi
fi

exec python "$HERE/scripts/render_debug_previews.py" "$@"
