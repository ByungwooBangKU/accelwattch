#!/usr/bin/env bash
# 사용법:
#   ./run_nsys.sh                 # 빌드 + nsys 프로파일 수집 + 결과 분석
#   ./run_nsys.sh --no-build      # 빌드 생략 (바이너리 재사용)
#   ./run_nsys.sh --no-analyze    # 분석 스킵
#
# 필요: CUDA toolkit (nvcc) + Nsight Systems (nsys) (CUDA 12.x 권장)
# WSL2: Windows측이 아닌 WSL용 nsys 사용 (`sudo apt install nsight-systems-2024.*`)

set -euo pipefail
cd "$(dirname "$0")"

DO_BUILD=1
DO_ANALYZE=1
for a in "$@"; do
    case "$a" in
        --no-build)   DO_BUILD=0 ;;
        --no-analyze) DO_ANALYZE=0 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

OUT_DIR="reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT_DIR/dram_util_$STAMP"
mkdir -p "$OUT_DIR"

command -v nvcc >/dev/null || { echo "nvcc not found — CUDA toolkit 필요"; exit 1; }
command -v nsys >/dev/null || { echo "nsys not found — Nsight Systems 필요"; exit 1; }

if [[ $DO_BUILD -eq 1 ]]; then
    echo "[build] nvcc ..."
    make -B
fi

# GPU metrics 샘플링은 root/관리자 권한이 필요할 수 있음 (ERR_NVGPUCTRPERM)
# 해결: `sudo systemctl edit nvidia-persistenced` 또는
#       `/etc/modprobe.d/nvidia.conf` 에 `options nvidia NVreg_RestrictProfilingToAdminUsers=0`
# WSL2에서는 현재(2025) GPU metrics 샘플링이 제한적일 수 있음 — 안되면 --gpu-metrics-device 제거
#
# 주요 옵션:
#   --gpu-metrics-device=0        GPU0 하드웨어 카운터 샘플링 (DRAM read/write throughput 포함)
#   --gpu-metrics-frequency=10000 10 kHz 샘플링 (기본 10 Hz)
#   --trace=cuda,nvtx             CUDA API + NVTX phase 라벨
#   --sample=none                 CPU 샘플링 끔 (DRAM 실험에는 불필요)

echo "[nsys] profiling -> $REPORT.nsys-rep"
nsys profile \
    --output="$REPORT" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-device=0 \
    --gpu-metrics-frequency=10000 \
    ./dram_util

# 리포트 요약 (kernel summary + nvtx range summary)
echo
echo "[nsys stats] kernel & nvtx summary"
nsys stats --report cuda_gpu_kern_sum,nvtx_sum "$REPORT.nsys-rep" || true

# sqlite 변환 (분석 단계에서 사용)
if [[ $DO_ANALYZE -eq 1 ]]; then
    echo
    echo "[nsys export] sqlite"
    nsys export --type=sqlite --force-overwrite=true \
        --output="$REPORT.sqlite" "$REPORT.nsys-rep"

    echo
    echo "[analyze] per-phase DRAM read utilization"
    python3 ./analyze.py "$REPORT.sqlite"
fi

echo
echo "[ok] Nsight Systems GUI에서 열기:"
echo "  nsys-ui $REPORT.nsys-rep"
echo "  -> Timeline 의 'GPU Metrics' 행에서 DRAM Read Throughput % 확인"
echo "  -> NVTX 행에 util_25 / util_50 / util_75 / util_100 라벨 표시"
