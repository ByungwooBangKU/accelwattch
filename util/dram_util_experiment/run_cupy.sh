#!/usr/bin/env bash
# CuPy 기반 DRAM utilization 실험 (nvcc 불필요).
#
# 사용:
#   ./run_cupy.sh                      # 기본 (25/50/75/100% x 10s, 1 GiB)
#   ./run_cupy.sh --buf-bytes $((8*1024**3))  # A100 등에서 버퍼 확장
#   ./run_cupy.sh --targets 10 30 60 90       # 커스텀 계단
#
# 환경: CuPy + pynvml + nvtx + matplotlib
#   pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib

set -euo pipefail
cd "$(dirname "$0")"

# ssc21env 에 이미 설치되어 있으면 그걸 쓰고, 아니면 현재 python 사용
PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in \
        /home/bang001/miniforge3/envs/ssc21env/bin/python \
        "$(command -v python3 || true)"; do
        if [[ -x "$cand" ]] && "$cand" -c "import cupy, nvtx, pynvml" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "[err] cupy/nvtx/pynvml 이 설치된 python 을 못 찾음. PY=... 환경변수로 지정하거나" >&2
    echo "       pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib" >&2
    exit 1
fi

echo "[info] using: $PY"
exec "$PY" dram_util_cupy.py "$@"
