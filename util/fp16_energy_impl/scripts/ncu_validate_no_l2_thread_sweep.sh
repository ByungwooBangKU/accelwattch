#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./build/fp16_energy_bench}"
OUTDIR="${2:-results/ncu_no_l2_thread_sweep}"
GPU_ID="${3:-0}"
THREADS_CSV="${4:-32,64,128,256,512,1024}"
mkdir -p "${OUTDIR}"

IFS=',' read -r -a THREADS_LIST <<< "${THREADS_CSV}"

NCU_METRICS="${NCU_METRICS:-smsp__inst_executed_pipe_tensor_op_hmma.sum,smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes_read.sum,lts__t_bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum}"

COMMON=(
  --device "${GPU_ID}"
  --blocks 0
  --blocks-per-sm 8
  --warmup 1
  --repeats 1
  --unroll 8
  --suppress-output-store
)

run_ncu() {
  local name="$1"; shift
  local kernel_regex="$1"; shift
  ncu \
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
    --iters 20000
  run_ncu "tensor_baseline_u32_t${threads}" "tensor_baseline_u32" \
    --kernel tensor_baseline_u32 \
    --threads "${threads}" \
    --iters 20000
done

python3 "$(dirname "$0")/validate_ncu_reports.py" --input "${OUTDIR}" --outdir "${OUTDIR}"

echo "Nsight Compute no-L2 thread-sweep reports written to ${OUTDIR}"
echo "Acceptance check: timed tensor/baseline kernels should show no material L1/L2/DRAM global memory workload beyond profiler noise."
echo "Validation summary: ${OUTDIR}/ncu_validation_summary.csv"
