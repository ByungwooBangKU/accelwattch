#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./build/fp16_energy_bench}"
OUTDIR="${2:-results/ncu_no_l2_thread_sweep}"
GPU_ID="${3:-0}"
THREADS_CSV="${4:-32,64,128,256,512,1024}"
mkdir -p "${OUTDIR}"

IFS=',' read -r -a THREADS_LIST <<< "${THREADS_CSV}"

DEFAULT_NCU_METRICS="smsp__inst_executed_pipe_tensor_op_hmma.sum,smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes_read.sum,lts__t_bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_METRICS="${NCU_METRICS:-${DEFAULT_NCU_METRICS}}"
NCU_BLOCKS_PER_SM="${NCU_BLOCKS_PER_SM:-8}"
NCU_UNROLL="${NCU_UNROLL:-8}"
NCU_WARMUP="${NCU_WARMUP:-1}"
NCU_REPEATS="${NCU_REPEATS:-1}"
NCU_ITERS="${NCU_ITERS:-20000}"
NCU_SUPPRESS_OUTPUT_STORE="${NCU_SUPPRESS_OUTPUT_STORE:-1}"
NCU_REQUIRE_TENSOR_ACTIVITY="${NCU_REQUIRE_TENSOR_ACTIVITY:-1}"
NCU_MIN_TENSOR_ACTIVITY_PCT="${NCU_MIN_TENSOR_ACTIVITY_PCT:-0.0}"
NCU_SUPPRESS_OUTPUT_STORE_BOOL="false"

COMMON=(
  --device "${GPU_ID}"
  --blocks 0
  --blocks-per-sm "${NCU_BLOCKS_PER_SM}"
  --warmup "${NCU_WARMUP}"
  --repeats "${NCU_REPEATS}"
  --unroll "${NCU_UNROLL}"
)
if [[ "${NCU_SUPPRESS_OUTPUT_STORE}" != "0" ]]; then
  COMMON+=(--suppress-output-store)
  NCU_SUPPRESS_OUTPUT_STORE_BOOL="true"
fi

run_ncu() {
  local name="$1"; shift
  local kernel_regex="$1"; shift
  "${NCU_BIN}" \
    --target-processes all \
    --kernel-name regex:".*${kernel_regex}.*" \
    --section LaunchStats \
    --section Occupancy \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --metrics "${NCU_METRICS}" \
    --print-summary per-kernel \
    --log-file "${OUTDIR}/${name}.ncu.txt" \
    "${BIN}" "${COMMON[@]}" "$@"
}

for threads in "${THREADS_LIST[@]}"; do
  run_ncu "tensor_mma_f16acc_t${threads}" "tensor_mma_f16acc" \
    --kernel tensor_mma_f16acc \
    --threads "${threads}" \
    --iters "${NCU_ITERS}"
  run_ncu "tensor_baseline_u32_t${threads}" "tensor_baseline_u32" \
    --kernel tensor_baseline_u32 \
    --threads "${threads}" \
    --iters "${NCU_ITERS}"
done

VALIDATE_ARGS=(
  --input "${OUTDIR}" \
  --outdir "${OUTDIR}" \
  --benchmark-blocks-per-sm "${NCU_BLOCKS_PER_SM}" \
  --benchmark-unroll "${NCU_UNROLL}" \
  --benchmark-suppress-output-store "${NCU_SUPPRESS_OUTPUT_STORE_BOOL}" \
  --benchmark-warmup "${NCU_WARMUP}" \
  --benchmark-repeats "${NCU_REPEATS}" \
  --benchmark-iters "${NCU_ITERS}" \
  --min-tensor-activity-pct "${NCU_MIN_TENSOR_ACTIVITY_PCT}"
)
if [[ "${NCU_REQUIRE_TENSOR_ACTIVITY}" != "0" ]]; then
  VALIDATE_ARGS+=(--require-tensor-activity)
fi

python3 "$(dirname "$0")/validate_ncu_reports.py" "${VALIDATE_ARGS[@]}"

echo "Nsight Compute no-L2 thread-sweep reports written to ${OUTDIR}"
echo "Acceptance check: timed tensor/baseline kernels should show no material L1/L2/DRAM global memory workload beyond profiler noise."
echo "Validation summary: ${OUTDIR}/ncu_validation_summary.csv"
