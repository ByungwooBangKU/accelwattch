#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SM_CLOCK_MHZ="${2:-}"
MEM_CLOCK_MHZ="${3:-}"
NVIDIA_SMI="$(command -v nvidia-smi)"

if [[ -z "${SM_CLOCK_MHZ}" || -z "${MEM_CLOCK_MHZ}" ]]; then
  echo "Usage: $0 GPU_ID SM_CLOCK_MHZ MEM_CLOCK_MHZ" >&2
  echo "Example: sudo $0 0 1410 1593" >&2
  exit 1
fi

sudo "${NVIDIA_SMI}" -i "${GPU_ID}" -pm 1
# Lock graphics/SM clocks. Some datacenter GPUs expose application clocks instead;
# if -lgc fails, set application clocks manually for that platform.
sudo "${NVIDIA_SMI}" -i "${GPU_ID}" -lgc "${SM_CLOCK_MHZ},${SM_CLOCK_MHZ}"
sudo "${NVIDIA_SMI}" -i "${GPU_ID}" -lmc "${MEM_CLOCK_MHZ},${MEM_CLOCK_MHZ}"
"${NVIDIA_SMI}" -i "${GPU_ID}" --query-gpu=name,clocks.gr,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv
