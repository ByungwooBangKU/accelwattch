#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
NVIDIA_SMI="$(command -v nvidia-smi)"
sudo "${NVIDIA_SMI}" -i "${GPU_ID}" -rgc || true
sudo "${NVIDIA_SMI}" -i "${GPU_ID}" -rmc || true
"${NVIDIA_SMI}" -i "${GPU_ID}" --query-gpu=name,clocks.gr,clocks.sm,clocks.mem,pstate,power.limit,temperature.gpu --format=csv
