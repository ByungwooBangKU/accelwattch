#!/usr/bin/env bash
# A100 80GB 프리셋:
#   - nvcc arch: sm_80 (A100)
#   - 버퍼 8 GiB (HBM2e 80GB에 여유, L2 40 MiB 대비 ~200x)
#   - HBM2e 피크 ≈ 2039 GB/s
#
# 사용: ./run_nsys_a100.sh [--no-build|--no-analyze]

set -euo pipefail
cd "$(dirname "$0")"

export SM=80
export DRAM_BUF_BYTES=$((8 * 1024 * 1024 * 1024))

echo "[preset] A100 80GB: SM=$SM  DRAM_BUF_BYTES=$DRAM_BUF_BYTES (8 GiB)"
exec ./run_nsys.sh "$@"
