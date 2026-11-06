#!/usr/bin/env bash
set -euo pipefail

# Runs the debug preview renderer (2D OpenCV, 3D Blender, or both) with a local venv,
# installing only the missing Python packages needed for the preview script.
#
# Usage examples:
#   ./run_render_debug.sh
#   ./run_render_debug.sh --backend opencv_2d --all --non-interactive
#   ./run_render_debug.sh --backend blender_3d --profiles chip_0603_resistor_3d@2 --non-interactive
#   ./run_render_debug.sh --backend both --all --non-interactive

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# shellcheck source=tools/_common.sh
source "$HERE/tools/_common.sh"

REQ_FILE="$HERE/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[error] Missing requirements file: $REQ_FILE" >&2
  exit 2
fi

# Prefer an existing venv (VENV_DIR, active VIRTUAL_ENV, .venv, venv); else create .venv.
if ! sim_find_python; then
  python3 -m venv "$HERE/.venv"
  PYTHON="$HERE/.venv/bin/python"
fi
PYTHON_BIN="$PYTHON"
VENV_DIR="$(dirname "$(dirname "$PYTHON_BIN")")"

# shellcheck disable=SC1090
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
fi

req_has() {
  # case-insensitive match of a bare requirement name at start of line
  local name="$1"
  grep -Eiq "^${name}([<=> ].*)?$" "$REQ_FILE" >/dev/null 2>&1
}

missing_mods=()
"$PYTHON_BIN" - <<'PY' || missing_mods+=("numpy")
import numpy  # noqa: F401
PY
"$PYTHON_BIN" - <<'PY' || missing_mods+=("yaml")
import yaml  # noqa: F401
PY
"$PYTHON_BIN" - <<'PY' || missing_mods+=("pil")
from PIL import Image  # noqa: F401
PY
"$PYTHON_BIN" - <<'PY' || missing_mods+=("cv2")
import cv2  # noqa: F401
PY

if (( ${#missing_mods[@]} )); then
  echo "[info] Missing Python modules in venv: ${missing_mods[*]}"
  # Best-effort: must not abort offline runs under `set -e`.
  "$PYTHON_BIN" -m pip install -q --upgrade pip || true

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
      cv2)
        if req_has "opencv-python"; then to_install+=("opencv-python"); fi
        ;;
    esac
  done

  if (( ${#to_install[@]} )); then
    "$PYTHON_BIN" -m pip install "${to_install[@]}"
  else
    echo "[warn] Could not map missing modules to requirements.txt entries; falling back to full requirements install." >&2
    "$PYTHON_BIN" -m pip install -r "$REQ_FILE"
  fi
fi

SETTINGS_FILE="${RENDER_DEBUG_SETTINGS_FILE:-$HERE/outputs/gui/settings.json}"

exec "$PYTHON_BIN" "$HERE/scripts/render_debug_previews.py" --settings-file "$SETTINGS_FILE" "$@"
