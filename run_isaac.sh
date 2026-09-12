#!/usr/bin/env bash
# Launch the Isaac Sim demo, locating Isaac and working around two things that
# routinely break it:
#
#   1. Isaac's own python.sh does `source ${SCRIPT_DIR}/setup_python_env.sh`
#      unquoted, so an install path containing a space fails outright. This
#      project's Isaac lives in "~/Robotics Tools/isaac". We route through a
#      space-free symlink; bash's `pwd` resolves logically, so Isaac sees a
#      clean path.
#   2. Isaac's Python has no idea where this project lives, so PYTHONPATH is
#      set explicitly rather than relying on the working directory.
#
# Usage:  ./run_isaac.sh [--plants 64] [--seconds 10] [--gui]

set -euo pipefail

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LINK_DIR="${HOME}/.cache/plantsim"
mkdir -p "${LINK_DIR}"

find_isaac() {
    if [[ -n "${ISAAC_SIM_PATH:-}" && -x "${ISAAC_SIM_PATH}/python.sh" ]]; then
        echo "${ISAAC_SIM_PATH}"; return 0
    fi
    local candidates=(
        "${HOME}/isaac-sim"
        "${HOME}/Robotics Tools/isaac"
        "${HOME}/.local/share/ov/pkg"/isaac-sim-*
        "/opt/isaac-sim"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "${c}/python.sh" ]]; then echo "${c}"; return 0; fi
    done
    return 1
}

if ! ISAAC_DIR="$(find_isaac)"; then
    cat >&2 <<'MSG'
Isaac Sim was not found.

Searched: $ISAAC_SIM_PATH, ~/isaac-sim, ~/Robotics Tools/isaac,
          ~/.local/share/ov/pkg/isaac-sim-*, /opt/isaac-sim

Install it from https://developer.nvidia.com/isaac/sim, or point this script
at an existing install:

    export ISAAC_SIM_PATH=/path/to/isaac-sim
    ./run_isaac.sh

Everything else in this project runs without Isaac Sim.
MSG
    exit 1
fi

# Route around spaces in the install path via a stable space-free symlink.
if [[ "${ISAAC_DIR}" == *" "* ]]; then
    SAFE_ISAAC="${LINK_DIR}/isaac"
    ln -sfn "${ISAAC_DIR}" "${SAFE_ISAAC}"
    echo "[isaac] install path contains a space; using symlink ${SAFE_ISAAC}"
    ISAAC_DIR="${SAFE_ISAAC}"
fi

VERSION="$(cat "${ISAAC_DIR}/VERSION" 2>/dev/null || echo unknown)"
echo "[isaac] ${ISAAC_DIR}  (version ${VERSION})"

if ! nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'MSG'
[isaac] WARNING: no NVIDIA driver is reachable (nvidia-smi failed).
        Isaac Sim will start, but cannot create a GPU device, so --gui will
        not render. Headless physics and USD export still work: the run
        bakes the motion into a .usd file you can render on a working
        machine later.
MSG
fi

# The project must be importable from Isaac's interpreter.
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${ISAAC_DIR}/python.sh" "${PROJECT_DIR}/src/scripts/05_isaac_sim.py" "$@"
