#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./build/fp16_energy_bench}"
OUTDIR="${2:-results/ncu}"
GPU_ID="${3:-0}"
mkdir -p "${OUTDIR}"

# Keep validation runs shorter than power runs. The section names are more stable
# across Nsight Compute versions than individual metric names.
COMMON=(--device "${GPU_ID}" --blocks 0 --blocks-per-sm 4 --threads 256 --warmup 1 --repeats 1 --unroll 8)

run_ncu() {
  local name="$1"; shift
  ncu \
    --target-processes all \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SourceCounters \
    --print-summary per-kernel \
    --log-file "${OUTDIR}/${name}.ncu.txt" \
    "${BIN}" "${COMMON[@]}" "$@"
}

run_ncu fp16_half2 --kernel fp16_half2 --iters 20000
run_ncu baseline_nop --kernel baseline_nop --iters 20000
run_ncu baseline_regmove --kernel baseline_regmove --iters 20000
run_ncu tensor_baseline_u32 --kernel tensor_baseline_u32 --iters 20000
run_ncu tensor_baseline_f32 --kernel tensor_baseline_f32 --iters 20000
run_ncu tensor_mma_f16acc --kernel tensor_mma_f16acc --iters 20000
run_ncu tensor_mma_f32acc --kernel tensor_mma_f32acc --iters 20000

echo "Nsight Compute reports written to ${OUTDIR}"
echo "Acceptance checks: no local spill, low L1/L2/DRAM traffic in P0 timed kernels, expected FP16/HMMA instruction path."
