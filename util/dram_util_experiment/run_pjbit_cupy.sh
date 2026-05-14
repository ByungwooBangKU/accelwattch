#!/usr/bin/env bash
# Run read/write DRAM bandwidth + marginal-power pJ/bit measurement.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in \
        /home/bang001/miniforge3/envs/ssc21env/bin/python \
        "$(command -v python3 || true)"; do
        if [[ -x "$cand" ]] && "$cand" -c "import cupy, nvtx, pynvml, matplotlib" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi

if [[ -z "$PY" ]]; then
    echo "[err] cupy/nvtx/pynvml/matplotlib 환경 필요" >&2
    echo "      pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib" >&2
    exit 1
fi

exec "$PY" dram_pjbit_cupy.py "$@"
