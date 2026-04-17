#!/usr/bin/env bash
# CuPy 실험을 nsys 로 프로파일링.
#
# nsys 가 설치되어 있으면 이 스크립트가 가장 풍부한 정보 제공:
#   - NVTX range (util_25 / util_50 / util_75 / util_100 / gap) 타임라인
#   - CUDA kernel 런치 트레이스
#   - GPU Metrics (DRAM Read Throughput %) 시계열  (--gpu-metrics-device)
#
# 사용:
#   ./run_nsys_cupy.sh                    # 기본 (25/50/75/100% x 10s)
#   ./run_nsys_cupy.sh --device 0         # 특정 GPU 선택
#   ./run_nsys_cupy.sh --phase-seconds 5  # 실험 시간 단축
#   ./run_nsys_cupy.sh --no-metrics       # GPU metrics 샘플링 끔 (WSL2/권한 제약시)

set -euo pipefail
cd "$(dirname "$0")"

DO_METRICS=1
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --no-metrics) DO_METRICS=0 ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done

command -v nsys >/dev/null || { echo "[err] nsys not found — Nsight Systems 필요"; exit 1; }

# python 자동 탐지 (run_cupy.sh 와 동일 로직)
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
[[ -z "$PY" ]] && { echo "[err] cupy/nvtx/pynvml 환경 필요 — pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib"; exit 1; }

OUT_DIR="reports"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/nsys_cupy_$STAMP"

NSYS_EXTRA=()
if [[ $DO_METRICS -eq 1 ]]; then
    NSYS_EXTRA+=(--gpu-metrics-device=all --gpu-metrics-frequency=10000)
    echo "[info] GPU metrics sampling ON (WSL2/권한 이슈 시 --no-metrics 옵션 사용)"
fi

echo "[info] python: $PY"
echo "[info] nsys output: $REPORT.nsys-rep"

nsys profile \
    --output="$REPORT" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none \
    "${NSYS_EXTRA[@]}" \
    "$PY" dram_util_cupy.py "${PASS_ARGS[@]}"

echo
echo "[nsys stats]"
nsys stats --report cuda_gpu_kern_sum,nvtx_sum "$REPORT.nsys-rep" 2>&1 | tail -80 || true

echo
echo "[ok] GUI:  nsys-ui $REPORT.nsys-rep"
echo "     Timeline 의 NVTX 행 + GPU Metrics 행에서"
echo "     util_25/50/75/100 phase 별 DRAM Read Throughput 확인"
