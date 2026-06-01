#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
ORIGINAL_INVOCATION="$(printf '%q ' "$0" "${ORIGINAL_ARGS[@]}")"

GPU_ID=0
NVIDIA_SMI_ID=""
CUDA_ARCH=""
OUTDIR=""
REPEAT=10
SAMPLE_MS=100
THREADS_CSV="32,64,96,128,160,192,224,256,288,320,384"
BUILD_DIR="build"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS:-}"
NVCC_BIN="${NVCC_BIN:-nvcc}"
NCU_BIN="${NCU_BIN:-ncu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
SKIP_BUILD=0
SKIP_PREFLIGHT=0
ALLOW_COMPUTE_APPS=0
DIAGNOSTIC_NO_NCU=0
CALIBRATE_MATRIX=1
TARGET_TEST_S=1.0
TARGET_BASELINE_S=1.0
MAX_CALIBRATED_REPEATS=1000

usage() {
  cat <<'USAGE'
Usage: run_strict_fp16_pipeline.sh [options]

Runs the strict FP16 Tensor Core pJ/bit pipeline:
  build -> env capture -> runtime preflight -> matrix repeat calibration
  -> structural-baseline thread sweep -> analyze -> Nsight Compute no-L2 validation
  -> quality gate --require-ncu --require-ncu-tensor-activity

Options:
  --gpu N              CUDA/NVML GPU index [0]
  --nvidia-smi-id ID   Physical nvidia-smi id/UUID for telemetry [auto]
  --cuda-arch ARCH     CMake CUDA architecture, e.g. A100=80, RTX3090=86, H100=90
  --outdir DIR         Result directory [auto timestamp under results/strict_fp16_*]
  --repeat N           Matrix repeat count [10]
  --sample-ms N        nvidia-smi power sample interval [100]
  --threads CSV        Threads/block list for NCU validation [32,64,96,128,160,192,224,256,288,320,384]
  --build-dir DIR      Build directory relative to fp16_energy_impl [build]
  --skip-build         Reuse existing binary
  --skip-preflight     Skip tool/GPU/process checks before build
  --allow-compute-apps Allow active compute processes on the target GPU during preflight
  --no-calibrate-matrix
                        Use configs/fp16_matmul_thread_sweep_fine.json without GPU-specific repeat calibration
  --target-test-s S     Target test CUDA-event duration for calibrated matrix [1.0]
  --target-baseline-s S Target baseline CUDA-event duration for calibrated matrix [1.0]
  --max-calibrated-repeats N
                        Upper bound for calibrated per-role repeats [1000]
  --diagnostic-no-ncu  Do not require NCU in quality gate; for local diagnostic only
  -h, --help           Show this help

Environment overrides:
  CMAKE_BIN, CMAKE_CUDA_FLAGS, NVCC_BIN, NCU_BIN, PYTHON_BIN, NVIDIA_SMI_BIN,
  MPLCONFIGDIR, NCU_METRICS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --nvidia-smi-id) NVIDIA_SMI_ID="$2"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --threads) THREADS_CSV="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --allow-compute-apps) ALLOW_COMPUTE_APPS=1; shift ;;
    --no-calibrate-matrix) CALIBRATE_MATRIX=0; shift ;;
    --target-test-s) TARGET_TEST_S="$2"; shift 2 ;;
    --target-baseline-s) TARGET_BASELINE_S="$2"; shift 2 ;;
    --max-calibrated-repeats) MAX_CALIBRATED_REPEATS="$2"; shift 2 ;;
    --diagnostic-no-ncu) DIAGNOSTIC_NO_NCU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${CUDA_ARCH}" ]]; then
  cat >&2 <<'EOF'
--cuda-arch is required. Use:
  A100  -> 80
  RTX 3090 -> 86
  H100  -> 90
EOF
  exit 2
fi

if [[ -z "${OUTDIR}" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="${ROOT}/results/strict_fp16_gpu${GPU_ID}_sm${CUDA_ARCH}_${stamp}"
elif [[ "${OUTDIR}" != /* ]]; then
  OUTDIR="${ROOT}/${OUTDIR}"
fi

BUILD_PATH="${ROOT}/${BUILD_DIR}"
BINARY="${BUILD_PATH}/fp16_energy_bench"
NCDIR="${OUTDIR}/ncu_no_l2_thread_sweep"
ENV_OUT="${OUTDIR}/env_gpu${GPU_ID}.txt"
BUILD_LOG="${OUTDIR}/build_ptxas.log"
RESOURCE_DIR="${OUTDIR}/resource_audit"
BASE_MATRIX="${ROOT}/configs/fp16_matmul_thread_sweep_fine.json"
MATRIX_PATH="${BASE_MATRIX}"
START_MANIFEST="${OUTDIR}/strict_pipeline_manifest_start.json"
FINAL_MANIFEST="${OUTDIR}/strict_pipeline_manifest.json"
PIPELINE_PREFLIGHT_JSON="${OUTDIR}/strict_pipeline_preflight.json"
PIPELINE_PREFLIGHT_CSV="${OUTDIR}/strict_pipeline_preflight.csv"

if [[ -z "${NVIDIA_SMI_ID}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a CUDA_VISIBLE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    NVIDIA_SMI_ID="${CUDA_VISIBLE_LIST[GPU_ID]:-${GPU_ID}}"
  else
    NVIDIA_SMI_ID="${GPU_ID}"
  fi
fi

mkdir -p "${OUTDIR}"

prepend_env_paths() {
  local var_name="$1"
  local paths_csv="$2"
  local existing="${!var_name:-}"
  local updated=""
  local -a paths=()
  IFS=',' read -r -a paths <<< "${paths_csv}"
  for path in "${paths[@]}"; do
    if [[ -z "${path}" ]]; then
      continue
    fi
    if [[ -z "${updated}" ]]; then
      updated="${path}"
    else
      updated="${updated}:${path}"
    fi
  done
  if [[ -z "${updated}" ]]; then
    return
  fi
  if [[ -n "${existing}" ]]; then
    printf -v "${var_name}" '%s:%s' "${updated}" "${existing}"
  else
    printf -v "${var_name}" '%s' "${updated}"
  fi
  export "${var_name}"
}

apply_cuda_flag_env_overrides() {
  if [[ -z "${CMAKE_CUDA_FLAGS}" ]]; then
    return
  fi

  local -a include_paths=()
  local -a library_paths=()
  local -a cuda_flags=()
  local flag
  local expect_include=0
  local expect_library=0
  read -r -a cuda_flags <<< "${CMAKE_CUDA_FLAGS}"
  for flag in "${cuda_flags[@]}"; do
    if [[ "${expect_include}" -eq 1 ]]; then
      include_paths+=("${flag}")
      expect_include=0
      continue
    fi
    if [[ "${expect_library}" -eq 1 ]]; then
      library_paths+=("${flag}")
      expect_library=0
      continue
    fi

    case "${flag}" in
      -I)
        expect_include=1
        ;;
      -I*)
        include_paths+=("${flag#-I}")
        ;;
      -isystem)
        expect_include=1
        ;;
      -isystem=*)
        include_paths+=("${flag#-isystem=}")
        ;;
      --include-path=*)
        include_paths+=("${flag#--include-path=}")
        ;;
      -L)
        expect_library=1
        ;;
      -L*)
        library_paths+=("${flag#-L}")
        ;;
      --library-path=*)
        library_paths+=("${flag#--library-path=}")
        ;;
    esac
  done

  if [[ "${#include_paths[@]}" -gt 0 ]]; then
    local include_csv
    include_csv="$(IFS=','; echo "${include_paths[*]}")"
    prepend_env_paths CPATH "${include_csv}"
  fi
  if [[ "${#library_paths[@]}" -gt 0 ]]; then
    local library_csv
    library_csv="$(IFS=','; echo "${library_paths[*]}")"
    prepend_env_paths LIBRARY_PATH "${library_csv}"
  fi
}

write_manifest() {
  local status="$1"
  local out="$2"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/write_strict_manifest.py" \
    --out "${out}" \
    --status "${status}" \
    --root "${ROOT}" \
    --outdir "${OUTDIR}" \
    --binary "${BINARY}" \
    --build-dir "${BUILD_DIR}" \
    --build-log "${BUILD_LOG}" \
    --pipeline-preflight-json "${PIPELINE_PREFLIGHT_JSON}" \
    --pipeline-preflight-csv "${PIPELINE_PREFLIGHT_CSV}" \
    --resource-dir "${RESOURCE_DIR}" \
    --ncu-dir "${NCDIR}" \
    --base-matrix "${BASE_MATRIX}" \
    --matrix "${MATRIX_PATH}" \
    --gpu "${GPU_ID}" \
    --nvidia-smi-id "${NVIDIA_SMI_ID}" \
    --cuda-arch "${CUDA_ARCH}" \
    --repeat "${REPEAT}" \
    --sample-ms "${SAMPLE_MS}" \
    --threads "${THREADS_CSV}" \
    --skip-build "${SKIP_BUILD}" \
    --skip-preflight "${SKIP_PREFLIGHT}" \
    --allow-compute-apps "${ALLOW_COMPUTE_APPS}" \
    --diagnostic-no-ncu "${DIAGNOSTIC_NO_NCU}" \
    --calibrate-matrix "${CALIBRATE_MATRIX}" \
    --target-test-s "${TARGET_TEST_S}" \
    --target-baseline-s "${TARGET_BASELINE_S}" \
    --max-calibrated-repeats "${MAX_CALIBRATED_REPEATS}" \
    --invocation "${ORIGINAL_INVOCATION}" \
    --cmake-bin "${CMAKE_BIN}" \
    --nvcc-bin "${NVCC_BIN}" \
    --ncu-bin "${NCU_BIN}" \
    --python-bin "${PYTHON_BIN}" \
    --nvidia-smi-bin "${NVIDIA_SMI_BIN}"
}

PIPELINE_DONE=0
on_pipeline_exit() {
  local rc=$?
  if [[ "${rc}" -ne 0 && "${PIPELINE_DONE}" -eq 0 ]]; then
    echo "Strict FP16 pipeline failed with exit code ${rc}; writing failed manifest: ${FINAL_MANIFEST}" >&2
    set +e
    write_manifest failed "${FINAL_MANIFEST}" >/dev/null 2>&1
    set -e
  fi
  return "${rc}"
}
trap on_pipeline_exit EXIT

write_manifest started "${START_MANIFEST}"

if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
  PREFLIGHT_ARGS=(
    --spec "pipeline:${GPU_ID}:${CUDA_ARCH}:${NVIDIA_SMI_ID}"
    --out-json "${PIPELINE_PREFLIGHT_JSON}"
    --out-csv "${PIPELINE_PREFLIGHT_CSV}"
    --cmake-bin "${CMAKE_BIN}"
    --nvcc-bin "${NVCC_BIN}"
    --ncu-bin "${NCU_BIN}"
    --python-bin "${PYTHON_BIN}"
    --nvidia-smi-bin "${NVIDIA_SMI_BIN}"
  )
  if [[ "${DIAGNOSTIC_NO_NCU}" -eq 0 ]]; then
    PREFLIGHT_ARGS+=(--require-ncu)
  fi
  if [[ "${ALLOW_COMPUTE_APPS}" -eq 1 ]]; then
    PREFLIGHT_ARGS+=(--allow-compute-apps)
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight_strict_architecture_suite.py" "${PREFLIGHT_ARGS[@]}"
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  apply_cuda_flag_env_overrides
  CMAKE_CONFIGURE_ARGS=(
    -S "${ROOT}"
    -B "${BUILD_PATH}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}"
  )
  if [[ -n "${CMAKE_CUDA_FLAGS}" ]]; then
    CMAKE_CONFIGURE_ARGS+=(-DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS}")
  fi
  "${CMAKE_BIN}" "${CMAKE_CONFIGURE_ARGS[@]}"
  "${CMAKE_BIN}" --build "${BUILD_PATH}" --clean-first -j 2 2>&1 | tee "${BUILD_LOG}"
fi

NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN}" \
  "${SCRIPT_DIR}/query_env.sh" "${NVIDIA_SMI_ID}" "${ENV_OUT}" "${BINARY}" "${GPU_ID}"

RUNTIME_PREFLIGHT_JSON="${OUTDIR}/runtime_preflight.json"
ARCH_PREFLIGHT_JSON="${OUTDIR}/architecture_preflight.json"
"${BINARY}" \
  --device "${GPU_ID}" \
  --kernel baseline_nop \
  --blocks 1 \
  --threads 32 \
  --iters 1 \
  --warmup 0 \
  --repeats 1 \
  --suppress-output-store \
  --json-out "${RUNTIME_PREFLIGHT_JSON}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_architecture.py" \
  --input "${RUNTIME_PREFLIGHT_JSON}" \
  --cuda-arch "${CUDA_ARCH}" \
  --strict-chip \
  --require-common-hmma \
  --out "${ARCH_PREFLIGHT_JSON}"

if [[ "${CALIBRATE_MATRIX}" -eq 1 ]]; then
  MATRIX_PATH="${OUTDIR}/calibrated_matrix.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/calibrate_matrix.py" \
    --matrix "${BASE_MATRIX}" \
    --out-matrix "${MATRIX_PATH}" \
    --outdir "${OUTDIR}" \
    --binary "${BINARY}" \
    --gpu "${GPU_ID}" \
    --target-test-s "${TARGET_TEST_S}" \
    --target-baseline-s "${TARGET_BASELINE_S}" \
    --max-repeats "${MAX_CALIBRATED_REPEATS}"
fi

write_manifest started "${START_MANIFEST}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" \
  --binary "${BINARY}" \
  --matrix "${MATRIX_PATH}" \
  --gpu "${GPU_ID}" \
  --nvidia-smi "${NVIDIA_SMI_BIN}" \
  --nvidia-smi-id "${NVIDIA_SMI_ID}" \
  --sample-ms "${SAMPLE_MS}" \
  --repeat "${REPEAT}" \
  --outdir "${OUTDIR}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_strict}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_results.py" --input "${OUTDIR}"

RESOURCE_ARGS=(--result-dir "${OUTDIR}" --outdir "${RESOURCE_DIR}" --cuda-arch "${CUDA_ARCH}")
if [[ -f "${BUILD_LOG}" ]]; then
  RESOURCE_ARGS+=(--ptxas-log "${BUILD_LOG}")
else
  RESOURCE_ARGS+=(--allow-missing)
fi
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_strict}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_kernel_resources.py" "${RESOURCE_ARGS[@]}"

NCU_BIN="${NCU_BIN}" \
  "${SCRIPT_DIR}/ncu_validate_no_l2_thread_sweep.sh" \
  "${BINARY}" \
  "${NCDIR}" \
  "${GPU_ID}" \
  "${THREADS_CSV}"

QUALITY_ARGS=(--input "${OUTDIR}" --ncu-summary "${NCDIR}/ncu_validation_summary.csv")
if [[ "${DIAGNOSTIC_NO_NCU}" -eq 0 ]]; then
  QUALITY_ARGS+=(--require-ncu)
  QUALITY_ARGS+=(--require-ncu-tensor-activity)
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_strict}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/quality_gate.py" "${QUALITY_ARGS[@]}"

write_manifest completed "${FINAL_MANIFEST}"
PIPELINE_DONE=1

cat <<EOF
Strict FP16 pipeline complete:
  results: ${OUTDIR}
  start manifest: ${START_MANIFEST}
  final manifest: ${FINAL_MANIFEST}
  resource audit: ${RESOURCE_DIR}/kernel_resource_summary.csv
  NCU validation: ${NCDIR}/ncu_validation_summary.csv
  quality gate: ${OUTDIR}/quality_gates.csv
  selected target summary: ${OUTDIR}/quality_gate_summary.json
EOF
