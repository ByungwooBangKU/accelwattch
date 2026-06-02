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
NCU_BLOCKS_PER_SM_CSV="${NCU_BLOCKS_PER_SM_CSV:-${NCU_BLOCKS_PER_SM}}"
NCU_UNROLL="${NCU_UNROLL:-8}"
NCU_WARMUP="${NCU_WARMUP:-1}"
NCU_REPEATS="${NCU_REPEATS:-1}"
NCU_ITERS="${NCU_ITERS:-20000}"
NCU_SUPPRESS_OUTPUT_STORE="${NCU_SUPPRESS_OUTPUT_STORE:-1}"
NCU_REQUIRE_TENSOR_ACTIVITY="${NCU_REQUIRE_TENSOR_ACTIVITY:-1}"
NCU_MIN_TENSOR_ACTIVITY_PCT="${NCU_MIN_TENSOR_ACTIVITY_PCT:-0.0}"
NCU_TEST_KERNEL="${NCU_TEST_KERNEL:-tensor_mma_f16acc}"
if [[ -z "${NCU_BASELINE_KERNEL:-}" ]]; then
  case "${NCU_TEST_KERNEL}" in
    tensor_mma_f16acc) NCU_BASELINE_KERNEL="tensor_baseline_mov" ;;
    tensor_mma_f32acc) NCU_BASELINE_KERNEL="tensor_baseline_f32" ;;
    fp16_half2) NCU_BASELINE_KERNEL="baseline_regmove" ;;
    *) NCU_BASELINE_KERNEL="tensor_baseline_mov" ;;
  esac
fi
NCU_SUPPRESS_OUTPUT_STORE_BOOL="false"
NCU_FAILURES_CSV="${OUTDIR}/ncu_run_failures.csv"

if [[ "${NCU_SUPPRESS_OUTPUT_STORE}" != "0" ]]; then
  NCU_SUPPRESS_OUTPUT_STORE_BOOL="true"
fi
IFS=',' read -r -a BLOCKS_PER_SM_LIST <<< "${NCU_BLOCKS_PER_SM_CSV}"

csv_field() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "${value}"
}

run_ncu() {
  local name="$1"; shift
  local blocks_per_sm="$1"; shift
  local kernel_regex="$1"; shift
  local log_file="${OUTDIR}/${name}.ncu.txt"
  local common=(
    --device "${GPU_ID}"
    --blocks 0
    --blocks-per-sm "${blocks_per_sm}"
    --warmup "${NCU_WARMUP}"
    --repeats "${NCU_REPEATS}"
    --unroll "${NCU_UNROLL}"
  )
  if [[ "${NCU_SUPPRESS_OUTPUT_STORE}" != "0" ]]; then
    common+=(--suppress-output-store)
  fi
  local cmd=(
    "${NCU_BIN}"
    --target-processes all \
    --kernel-name regex:".*${kernel_regex}.*" \
    --section LaunchStats \
    --section Occupancy \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --metrics "${NCU_METRICS}" \
    --print-summary per-kernel \
    --log-file "${log_file}" \
    "${BIN}" "${common[@]}" "$@"
  )
  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    if [[ ! -f "${NCU_FAILURES_CSV}" ]]; then
      printf 'name,returncode,log_file,command\n' > "${NCU_FAILURES_CSV}"
    fi
    local escaped_command
    escaped_command="$(printf '%q ' "${cmd[@]}")"
    {
      csv_field "${name}"; printf ',%s,' "${rc}"
      csv_field "${log_file}"; printf ','
      csv_field "${escaped_command}"; printf '\n'
    } >> "${NCU_FAILURES_CSV}"
    echo "Nsight Compute failed for ${name} with exit code ${rc}; see ${log_file}" >&2
    echo "Failure record: ${NCU_FAILURES_CSV}" >&2
    return "${rc}"
  fi
}

for threads in "${THREADS_LIST[@]}"; do
  for blocks_per_sm in "${BLOCKS_PER_SM_LIST[@]}"; do
    suffix="t${threads}"
    if [[ "${NCU_BLOCKS_PER_SM_CSV}" != "${NCU_BLOCKS_PER_SM}" || "${blocks_per_sm}" != "${NCU_BLOCKS_PER_SM}" ]]; then
      suffix="${suffix}_b${blocks_per_sm}"
    fi
    run_ncu "${NCU_TEST_KERNEL}_${suffix}" "${blocks_per_sm}" "${NCU_TEST_KERNEL}" \
      --kernel "${NCU_TEST_KERNEL}" \
      --threads "${threads}" \
      --iters "${NCU_ITERS}"
    run_ncu "${NCU_BASELINE_KERNEL}_${suffix}" "${blocks_per_sm}" "${NCU_BASELINE_KERNEL}" \
      --kernel "${NCU_BASELINE_KERNEL}" \
      --threads "${threads}" \
      --iters "${NCU_ITERS}"
  done
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
