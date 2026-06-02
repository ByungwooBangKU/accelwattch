#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID=0
CUDA_ARCH=86
GPU_KIND="rtx3090"
CUDA_VERSION="12.1"
ENV_PREFIX="/tmp/fp16-energy-sm86-cuda121"
OUTDIR=""
THREADS_CSV="32,64,96,128,160,192,224,256,288,320,384"
NCU_BLOCKS_PER_SM_CSV="1,2,4,8"
MODE="full"
USE_SUDO=0
SKIP_TOOLCHAIN_INSTALL=0

usage() {
  cat <<'USAGE'
Usage: run_rtx3090_strict_ncu_after_counter_enable.sh [options]

Run the RTX 3090 strict FP16 NCU flow after NVIDIA performance counters
have been enabled on the host.

Default mode runs the full strict pipeline:
  build -> preflight -> NCU permission probe -> sweep -> NCU no-L2 validation
  -> quality gate -> selected-target work-slope.

Options:
  --gpu N                 CUDA/NVML GPU index [0]
  --cuda-arch ARCH        CUDA architecture [86]
  --env-prefix PATH       Conda env prefix [/tmp/fp16-energy-sm86-cuda121]
  --cuda-version X.Y      Conda CUDA toolkit version [12.1]
  --outdir DIR            Result directory [timestamped strict_fp16_launch_shape_rtx3090_ncu_*]
  --threads CSV           Measurement/NCU validation thread list
  --ncu-blocks-per-sm-csv CSV
                           Blocks/SM list [1,2,4,8]
  --probe-only            Build the benchmark and run only the short NCU permission probe
  --use-sudo              Run the probe/pipeline via sudo -E env ...
                           Use only on native Linux/admin-only profiling setups.
                           On WSL2, enable counters in Windows NVIDIA Control Panel first.
  --skip-toolchain-install
                           Do not create/update the conda toolchain; require env file to exist
  -h, --help              Show this help

Examples:
  # Fast permission check after enabling counters on the Windows host
  ./scripts/run_rtx3090_strict_ncu_after_counter_enable.sh --probe-only

  # Full strict RTX 3090 run
  ./scripts/run_rtx3090_strict_ncu_after_counter_enable.sh

  # Native Linux only, if profiling is restricted to admin users
  ./scripts/run_rtx3090_strict_ncu_after_counter_enable.sh --use-sudo
USAGE
}

log() {
  printf '[rtx3090-strict-ncu] %s\n' "$*" >&2
}

die() {
  printf '[rtx3090-strict-ncu] error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --env-prefix) ENV_PREFIX="$2"; shift 2 ;;
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --threads) THREADS_CSV="$2"; shift 2 ;;
    --ncu-blocks-per-sm-csv) NCU_BLOCKS_PER_SM_CSV="$2"; shift 2 ;;
    --probe-only) MODE="probe"; shift ;;
    --use-sudo) USE_SUDO=1; shift ;;
    --skip-toolchain-install) SKIP_TOOLCHAIN_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [[ "${CUDA_ARCH}" != "86" ]]; then
  log "warning: this helper is named for RTX 3090; using custom CUDA arch ${CUDA_ARCH}"
fi

cd "${ROOT}"

CUDA_TAG="${CUDA_VERSION//./}"
ENV_FILE="${ROOT}/env/toolchain_${GPU_KIND}_sm${CUDA_ARCH}_cuda${CUDA_TAG}.sh"

INSTALL_ARGS=(
  --gpu "${GPU_ID}"
  --gpu-kind "${GPU_KIND}"
  --cuda-arch "${CUDA_ARCH}"
  --cuda-version "${CUDA_VERSION}"
  --env-prefix "${ENV_PREFIX}"
)
if [[ "${SKIP_TOOLCHAIN_INSTALL}" -eq 1 ]]; then
  [[ -f "${ENV_FILE}" ]] || die "missing ${ENV_FILE}; run without --skip-toolchain-install first"
else
  if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    INSTALL_ARGS+=(--no-install)
  fi
  INSTALL_ARGS+=(--no-verify)
  "${SCRIPT_DIR}/install_gpu_toolchain.sh" "${INSTALL_ARGS[@]}"
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

[[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ -x "${CMAKE_BIN}" ]] || die "CMAKE_BIN is not executable: ${CMAKE_BIN}"
[[ -x "${NVCC_BIN}" ]] || die "NVCC_BIN is not executable: ${NVCC_BIN}"
[[ -x "${NCU_BIN}" ]] || die "NCU_BIN is not executable: ${NCU_BIN}"

log "toolchain env: ${ENV_FILE}"
log "ncu: ${NCU_BIN}"
"${NCU_BIN}" --version | sed -n '1,4p'

if [[ -z "${OUTDIR}" ]]; then
  OUTDIR="${ROOT}/results/strict_fp16_launch_shape_rtx3090_ncu_$(date +%Y%m%d_%H%M%S)"
elif [[ "${OUTDIR}" != /* ]]; then
  OUTDIR="${ROOT}/${OUTDIR}"
fi
mkdir -p "${OUTDIR}"

run_maybe_sudo() {
  if [[ "${USE_SUDO}" -eq 1 ]]; then
    sudo -E env \
      PATH="${PATH}" \
      CPATH="${CPATH:-}" \
      LIBRARY_PATH="${LIBRARY_PATH:-}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      CMAKE_BIN="${CMAKE_BIN}" \
      CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS:-}" \
      NVCC_BIN="${NVCC_BIN}" \
      NCU_BIN="${NCU_BIN}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN}" \
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_rtx3090_sm86}" \
      "$@"
  else
    "$@"
  fi
}

if [[ "${MODE}" == "probe" ]]; then
  BUILD_PATH="${ROOT}/build"
  BINARY="${BUILD_PATH}/fp16_energy_bench"
  log "building benchmark for NCU permission probe"
  CMAKE_CONFIGURE_ARGS=(
    -S "${ROOT}" \
    -B "${BUILD_PATH}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}"
  )
  if [[ -n "${CMAKE_CUDA_FLAGS:-}" ]]; then
    CMAKE_CONFIGURE_ARGS+=(-DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS}")
  fi
  "${CMAKE_BIN}" "${CMAKE_CONFIGURE_ARGS[@]}"
  "${CMAKE_BIN}" --build "${BUILD_PATH}" --parallel 2

  PROBE_DIR="${OUTDIR}/ncu_permission_probe_manual"
  log "running short NCU permission probe: ${PROBE_DIR}"
  run_maybe_sudo \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/probe_ncu_permissions.py" \
      --binary "${BINARY}" \
      --outdir "${PROBE_DIR}" \
      --gpu "${GPU_ID}" \
      --ncu-bin "${NCU_BIN}"
  log "probe passed: ${PROBE_DIR}/ncu_permission_probe.json"
  printf 'RESULT_DIR=%s\n' "${OUTDIR}"
  exit 0
fi

log "running full strict RTX 3090 pipeline"
log "result dir: ${OUTDIR}"
run_maybe_sudo \
  "${SCRIPT_DIR}/run_strict_fp16_pipeline.sh" \
    --gpu "${GPU_ID}" \
    --cuda-arch "${CUDA_ARCH}" \
    --matrix configs/fp16_matmul_launch_shape_sweep.json \
    --threads "${THREADS_CSV}" \
    --ncu-blocks-per-sm-csv "${NCU_BLOCKS_PER_SM_CSV}" \
    --run-work-slope \
    --outdir "${OUTDIR}"

printf 'RESULT_DIR=%s\n' "${OUTDIR}"
