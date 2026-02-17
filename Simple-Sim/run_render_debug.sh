#!/usr/bin/env bash
set -euo pipefail

# Runs the debug preview renderer (2D OpenCV, 3D Blender, or both) with a local venv,
# installing only the missing Python packages needed for the preview script.
#
# Usage examples:
#   ./run_render_debug.sh
#   ./run_render_debug.sh --backend opencv_2d --all --non-interactive
#   ./run_render_debug.sh --backend blender_3d --profiles chip_0603_resistor_3d@1 --non-interactive
#   ./run_render_debug.sh --backend both --all --non-interactive

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_HERE="$(cd "$HERE" && pwd -P)"
cd "$HERE"

REQ_FILE="$HERE/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[error] Missing requirements file: $REQ_FILE" >&2
  exit 2
fi

VENV_DIR=""

# Prefer currently active venv, then local candidates from both logical and
# physical script paths (helps when the same repo is mounted via different roots).
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  VENV_DIR="$VIRTUAL_ENV"
else
  for base in "$HERE" "$REAL_HERE"; do
    for cand in ".venv" "venv"; do
      if [[ -x "$base/$cand/bin/python" ]]; then
        VENV_DIR="$base/$cand"
        break 2
      fi
    done
  done
fi

if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$HERE/.venv"
  python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] No python executable found in virtual environment: $VENV_DIR" >&2
  exit 2
fi

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
  "$PYTHON_BIN" -m pip install -q --upgrade pip

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
