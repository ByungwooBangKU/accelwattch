#!/usr/bin/env bash
# Run read/write DRAM bandwidth + marginal-power pJ/bit measurement.
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
    echo "      sudo env PY=/path/to/venv/bin/python ./run_pjbit_cupy.sh ..." >&2
    exit 1
fi

if [[ -z "$PY" ]]; then
    echo "[err] cupy/nvtx/pynvml/matplotlib 환경 필요" >&2
    echo "      pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib" >&2
    echo "      sudo는 venv 환경을 지울 수 있으므로 필요하면 다음처럼 실행:" >&2
    echo "      sudo env PY=/path/to/venv/bin/python ./run_pjbit_cupy.sh ..." >&2
    exit 1
fi

export PY
exec "$PY" dram_pjbit_cupy.py "$@"
