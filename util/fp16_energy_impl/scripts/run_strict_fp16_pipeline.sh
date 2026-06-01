#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID=0
NVIDIA_SMI_ID=""
CUDA_ARCH=""
OUTDIR=""
REPEAT=10
SAMPLE_MS=100
THREADS_CSV="32,64,96,128,160,192,224,256,288,320,384"
BUILD_DIR="build"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_BUILD=0
DIAGNOSTIC_NO_NCU=0

usage() {
  cat <<'USAGE'
Usage: run_strict_fp16_pipeline.sh [options]

Runs the strict FP16 Tensor Core pJ/bit pipeline:
  build -> env capture -> structural-baseline thread sweep -> analyze
  -> Nsight Compute no-L2 validation -> quality gate --require-ncu

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
  --diagnostic-no-ncu  Do not require NCU in quality gate; for local diagnostic only
  -h, --help           Show this help

Environment overrides:
  CMAKE_BIN, PYTHON_BIN, MPLCONFIGDIR, NCU_METRICS
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

if [[ -z "${NVIDIA_SMI_ID}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a CUDA_VISIBLE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    NVIDIA_SMI_ID="${CUDA_VISIBLE_LIST[GPU_ID]:-${GPU_ID}}"
  else
    NVIDIA_SMI_ID="${GPU_ID}"
  fi
fi

mkdir -p "${OUTDIR}"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  "${CMAKE_BIN}" -S "${ROOT}" -B "${BUILD_PATH}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
  "${CMAKE_BIN}" --build "${BUILD_PATH}" --clean-first -j 2 2>&1 | tee "${BUILD_LOG}"
fi

"${SCRIPT_DIR}/query_env.sh" "${NVIDIA_SMI_ID}" "${ENV_OUT}" "${BINARY}" "${GPU_ID}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" \
  --binary "${BINARY}" \
  --matrix "${ROOT}/configs/fp16_matmul_thread_sweep_fine.json" \
  --gpu "${GPU_ID}" \
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

"${SCRIPT_DIR}/ncu_validate_no_l2_thread_sweep.sh" \
  "${BINARY}" \
  "${NCDIR}" \
  "${GPU_ID}" \
  "${THREADS_CSV}"

QUALITY_ARGS=(--input "${OUTDIR}" --ncu-summary "${NCDIR}/ncu_validation_summary.csv")
if [[ "${DIAGNOSTIC_NO_NCU}" -eq 0 ]]; then
  QUALITY_ARGS+=(--require-ncu)
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_strict}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/quality_gate.py" "${QUALITY_ARGS[@]}"

cat <<EOF
Strict FP16 pipeline complete:
  results: ${OUTDIR}
  resource audit: ${RESOURCE_DIR}/kernel_resource_summary.csv
  NCU validation: ${NCDIR}/ncu_validation_summary.csv
  quality gate: ${OUTDIR}/quality_gates.csv
  selected target summary: ${OUTDIR}/quality_gate_summary.json
EOF
