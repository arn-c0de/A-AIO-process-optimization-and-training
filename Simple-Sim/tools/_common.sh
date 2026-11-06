# Shared helpers for Simple-Sim shell launchers. Source it, do not execute it:
#   source "<Simple-Sim>/tools/_common.sh"
#
# Sets SIM_ROOT (the Simple-Sim directory) and REPO_ROOT (its parent) and defines:
#   sim_find_python [--fallback]  set PYTHON to the project venv python
#                                 ($VENV_DIR, $VIRTUAL_ENV, .venv, venv, ../.venv);
#                                 with --fallback use python3 if none exists. Returns 1 if not found.
#   sim_require_python            sim_find_python or print install hints and exit 1
#   sim_bootstrap_venv            create ${VENV_DIR:-.venv} if missing and pip install
#                                 requirements.txt (honours WHEELHOUSE, PIP_FLAGS_EXTRA)

SIM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${SIM_ROOT}/.." && pwd)"

sim_find_python() {
  local cand
  for cand in \
    ${VENV_DIR:+"${VENV_DIR}"} \
    ${VIRTUAL_ENV:+"${VIRTUAL_ENV}"} \
    "${SIM_ROOT}/.venv" \
    "${SIM_ROOT}/venv" \
    "${REPO_ROOT}/.venv"; do
    # Relative VENV_DIR values are resolved against SIM_ROOT.
    [[ "${cand}" == /* ]] || cand="${SIM_ROOT}/${cand}"
    if [[ -x "${cand}/bin/python" ]]; then
      PYTHON="${cand}/bin/python"
      return 0
    fi
  done
  if [[ "${1:-}" == "--fallback" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    return 0
  fi
  return 1
}

sim_require_python() {
  if ! sim_find_python; then
    echo "Missing virtualenv python (looked for ${SIM_ROOT}/.venv/bin/python)." >&2
    echo "Run the pipeline once (./run_pipeline.sh) or install deps:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
  fi
}

sim_bootstrap_venv() {
  local venv="${VENV_DIR:-.venv}"
  [[ "${venv}" == /* ]] || venv="${SIM_ROOT}/${venv}"
  PYTHON="${venv}/bin/python"
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating venv at ${venv}"
    python3 -m venv "${venv}"
  fi

  local pip_flags=()
  if [[ -n "${WHEELHOUSE:-}" ]]; then
    pip_flags+=(--no-index --find-links "${WHEELHOUSE}")
  fi
  if [[ -n "${PIP_FLAGS_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    pip_flags+=(${PIP_FLAGS_EXTRA})
  fi

  echo "Installing dependencies from requirements.txt"
  # ${arr[@]+...} keeps an empty array safe under `set -u` on bash < 4.4 (macOS ships 3.2).
  if ! "${PYTHON}" -m pip install ${pip_flags[@]+"${pip_flags[@]}"} -r "${SIM_ROOT}/requirements.txt"; then
    echo "" >&2
    echo "Dependency install failed." >&2
    echo "If you are offline, set WHEELHOUSE to a directory of pre-downloaded wheels:" >&2
    echo "  WHEELHOUSE=/path/to/wheelhouse $0" >&2
    exit 1
  fi
}
