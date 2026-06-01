#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
OUT="${2:-results/env.txt}"
BINARY="${3:-build/fp16_energy_bench}"
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
  echo "# nvidia-smi compact GPU query"
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=index,uuid,pci.bus_id,name,driver_version,power.limit,power.draw,pstate,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv || true
  echo
  echo "# nvidia-smi compute apps"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv || true
  echo
  echo "# nvcc --version"
  nvcc --version || true
  echo
  echo "# fp16_energy_bench --help"
  if [[ -x "${BINARY}" ]]; then
    "${BINARY}" --help || true
  else
    echo "binary not found or not executable: ${BINARY}"
  fi
  echo
  echo "# fp16_energy_bench CUDA runtime probe"
  if [[ -x "${BINARY}" ]]; then
    PROBE_JSON="${OUT}.runtime_probe.json"
    "${BINARY}" \
      --device "${GPU_ID}" \
      --kernel baseline_nop \
      --blocks 1 \
      --threads 32 \
      --iters 1 \
      --warmup 0 \
      --repeats 1 \
      --suppress-output-store \
      --json-out "${PROBE_JSON}" || true
    if [[ -f "${PROBE_JSON}" ]]; then
      cat "${PROBE_JSON}"
      echo
    fi
  else
    echo "skipped; binary not found or not executable: ${BINARY}"
  fi
  echo
  echo "# CUDA binary resource usage"
  if command -v cuobjdump >/dev/null 2>&1 && [[ -x "${BINARY}" ]]; then
    cuobjdump --dump-resource-usage "${BINARY}" || true
  else
    echo "cuobjdump not found or binary missing; use CMake ptxas output (-Xptxas=-v) for registers/thread."
  fi
  echo
  echo "# uname -a"
  uname -a
} 2>&1 | tee "${OUT}"
