#!/bin/bash
# 3D-Board-Renderer launcher (photoreal 3D PCB from DeepVision data, BACKLOG-Sim/board3d-plan.md).
#
#   cd Simple-Sim
#   ./3d-board-viewer.sh                     # viewer / defect editor in the browser (default)
#   ./3d-board-viewer.sh viewer              # same; opens http://127.0.0.1:8765
#   ./3d-board-viewer.sh import              # (re)read the product from DeepVision into outputs/board3d/<product>/<side>/
#   ./3d-board-viewer.sh fit                 # measure the real part poses on the scan (ICs, arrays, ...)
#   ./3d-board-viewer.sh render              # whole board, photoreal, compared with the scan
#   ./3d-board-viewer.sh probe               # 20 x 20 mm test window
#   ./3d-board-viewer.sh dataset [sheets]    # labelled defect crops (default 10 sheets x 8 parts)
#   ./3d-board-viewer.sh all                 # import + fit + render
#
# Settings (environment):
#   PRODUCT=<id> SIDE=top         product / side in the DeepVision store; remembered locally after the first start
#                                 (outputs/board3d/.product), or the only product already imported
#   PHOTOS=<dir>                  folder of phone photos of the product (import; default: none)
#   DV=../../DeepVision-Web       DeepVision checkout (read only)
#   SOURCE=photos|mounted|scan    viewer texture: fused stills of the mounted board (default), SimpleAOI's render of
#                                 the mounted board, or the SMD flatbed scan
#   PORT=8765                     viewer port (127.0.0.1 only)
#   SAMPLES=128 DEVICE=GPU        Cycles samples / device (falls back to the CPU when the GPU is full)
#   NO_BROWSER=1                  do not open the browser
#
# Blender runs on the GPU only outside a sandbox that hides /dev/nvidia*; run this script from a normal terminal.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/tools/_common.sh"
sim_require_python

# Product: $PRODUCT, else the one remembered locally (outputs/board3d/.product, git-ignored like all board data),
# else the only product already imported under outputs/board3d/. No product name lives in the repo.
if [[ -z "${PRODUCT:-}" && -f outputs/board3d/.product ]]; then
  PRODUCT="$(<outputs/board3d/.product)"
fi
if [[ -z "${PRODUCT:-}" ]]; then
  mapfile -t KNOWN < <(find outputs/board3d -mindepth 2 -maxdepth 3 -name board3d.json 2>/dev/null | awk -F/ '{print $3}' | sort -u)
  if [[ ${#KNOWN[@]} -eq 1 ]]; then
    PRODUCT="${KNOWN[0]}"
  else
    echo "Which product? Start once with PRODUCT=<product id in the DeepVision store> ./3d-board-viewer.sh" >&2
    [[ ${#KNOWN[@]} -gt 1 ]] && echo "Already imported: ${KNOWN[*]}" >&2
    exit 2
  fi
fi
mkdir -p outputs/board3d && printf '%s\n' "${PRODUCT}" > outputs/board3d/.product
PHOTOS="${PHOTOS:-}"
SIDE="${SIDE:-top}"
DV="${DV:-${SCRIPT_DIR}/../../DeepVision-Web}"
SOURCE="${SOURCE:-photos}"
PORT="${PORT:-8765}"
SAMPLES="${SAMPLES:-128}"
DEVICE="${DEVICE:-GPU}"
BOARD3D="outputs/board3d/${PRODUCT}/${SIDE}"
MODE="${1:-viewer}"

need_blender() {
  if ! command -v "${SIMPLE_SIM_BLENDER:-blender}" >/dev/null 2>&1; then
    echo "Blender not found: install it or set SIMPLE_SIM_BLENDER=/path/to/blender" >&2
    exit 1
  fi
}

ensure_board3d() {
  if [[ ! -f "${BOARD3D}/board3d.json" ]]; then
    echo "== no board3d yet for ${PRODUCT} ${SIDE}: importing from ${DV}"
    do_import
  fi
}

do_import() {
  "${PYTHON}" tools/board3d_import.py --dv "${DV}" --product "${PRODUCT}" --side "${SIDE}" ${PHOTOS:+--photos "${PHOTOS}"}
}

do_fit() {
  ensure_board3d
  "${PYTHON}" -m simple_sim.board3d.fit --board3d "${BOARD3D}"
}

do_render() {
  ensure_board3d
  need_blender
  "${PYTHON}" tools/board3d_render.py --board3d "${BOARD3D}" --full --name full --samples "${SAMPLES}" --device "${DEVICE}"
  echo "compare image: ${SCRIPT_DIR}/${BOARD3D}/renders/full_compare.png"
}

case "${MODE}" in
  viewer)
    ensure_board3d
    need_blender
    URL="http://127.0.0.1:${PORT}/"
    echo "== Web UI: ${URL}  (answers once the scene is ready; the first start builds mosaic + GLB, a few minutes)"
    if [[ -z "${NO_BROWSER:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
      # open the browser once the server answers (the first start exports the 3D scene, ~1 min).
      # setsid: the browser gets its own session, so Ctrl+C on the server does not reach a browser started here.
      ( for _ in $(seq 1 600); do
          if curl -fs -o /dev/null "${URL}" 2>/dev/null; then
            if command -v setsid >/dev/null 2>&1; then
              setsid -f xdg-open "${URL}" </dev/null >/dev/null 2>&1
            else
              nohup xdg-open "${URL}" </dev/null >/dev/null 2>&1 &
            fi
            break
          fi
          sleep 1
        done ) &
    fi
    exec "${PYTHON}" tools/board3d_viewer.py --board3d "${BOARD3D}" --source "${SOURCE}" --port "${PORT}"
    ;;
  import)
    do_import
    ;;
  fit)
    do_fit
    ;;
  render)
    do_render
    ;;
  probe)
    ensure_board3d
    need_blender
    "${PYTHON}" tools/board3d_render.py --board3d "${BOARD3D}" --window 64 41 20 20 --name probe --samples "${SAMPLES}" --device "${DEVICE}"
    echo "compare image: ${SCRIPT_DIR}/${BOARD3D}/renders/probe_compare.png"
    ;;
  dataset)
    ensure_board3d
    need_blender
    "${PYTHON}" tools/board3d_dataset.py --board3d "${BOARD3D}" --sheets "${2:-10}" --per-sheet 8 --samples 64 --device "${DEVICE}" --name "run$(date +%Y%m%d-%H%M%S)"
    ;;
  all)
    do_import
    do_fit
    do_render
    ;;
  *)
    echo "Usage: ./3d-board-viewer.sh [viewer|import|fit|render|probe|dataset [sheets]|all]"
    exit 2
    ;;
esac
