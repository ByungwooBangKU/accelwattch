#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./build/fp16_energy_bench}"
OUTDIR="${2:-results/ncu}"
GPU_ID="${3:-0}"
mkdir -p "${OUTDIR}"

# Keep validation runs shorter than power runs. The section names are more stable
# across Nsight Compute versions than individual metric names.
COMMON=(--device "${GPU_ID}" --blocks 0 --blocks-per-sm 4 --threads 256 --warmup 1 --repeats 1 --unroll 8)
NCU_METRICS="${NCU_METRICS:-smsp__inst_executed_pipe_tensor_op_hmma.sum,smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes_read.sum,lts__t_bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum}"

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
    --metrics "${NCU_METRICS}" \
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

python3 "$(dirname "$0")/validate_ncu_reports.py" --input "${OUTDIR}" --outdir "${OUTDIR}"

echo "Nsight Compute reports written to ${OUTDIR}"
echo "Acceptance checks: no local spill, low L1/L2/DRAM traffic in P0 timed kernels, expected FP16/HMMA instruction path."
echo "Validation summary: ${OUTDIR}/ncu_validation_summary.csv"
