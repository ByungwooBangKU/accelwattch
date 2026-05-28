#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
OUT="${2:-results/env.txt}"
mkdir -p "$(dirname "${OUT}")"
{
  echo "# date"
  date --iso-8601=seconds || date
  echo
  echo "# nvidia-smi -L"
  nvidia-smi -L || true
  echo
  echo "# nvidia-smi -q"
  nvidia-smi -i "${GPU_ID}" -q || true
  echo
  echo "# nvcc --version"
  nvcc --version || true
  echo
  echo "# uname -a"
  uname -a
} | tee "${OUT}"
