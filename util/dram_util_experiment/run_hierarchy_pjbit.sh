#!/usr/bin/env bash
# Run memory-hierarchy lower-bound pJ/bit experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in \
        "${VIRTUAL_ENV:-}/bin/python" \
        "${CONDA_PREFIX:-}/bin/python" \
        /home/bang001/miniforge3/envs/ssc21env/bin/python \
        "$(command -v python3 || true)"; do
        if [[ -x "$cand" ]] && "$cand" -c "import cupy, nvtx, pynvml, matplotlib" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
elif ! "$PY" -c "import cupy, nvtx, pynvml, matplotlib" >/dev/null 2>&1; then
    echo "[err] PY does not provide cupy/nvtx/pynvml/matplotlib: $PY" >&2
    echo "      If using sudo with a venv, run:" >&2
    echo "      sudo env PY=/path/to/venv/bin/python ./run_hierarchy_pjbit.sh ..." >&2
    exit 1
fi

if [[ -z "$PY" ]]; then
    echo "[err] cupy/nvtx/pynvml/matplotlib environment required" >&2
    echo "      pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib" >&2
    echo "      sudo can reset venv PATH; use:" >&2
    echo "      sudo env PY=/path/to/venv/bin/python ./run_hierarchy_pjbit.sh ..." >&2
    exit 1
fi

export PY
exec "$PY" hierarchy_pjbit_cupy.py "$@"
