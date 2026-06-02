#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID=0
GPU_KIND="auto"
CUDA_ARCH="auto"
CUDA_VERSION="12.1"
ENV_NAME=""
ENV_PREFIX=""
MANAGER="auto"
ENV_FILE=""
RUN_FILE=""
NVIDIA_SMI_BIN=""
INSTALL=1
VERIFY=1
FORCE_RECREATE=0
BUILD_SMOKE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: install_gpu_toolchain.sh [options]

Install a user-space CUDA/Nsight Compute toolchain for the strict FP16
pJ/bit pipeline and write a sourceable toolchain env file.

This script does not install or upgrade the NVIDIA driver. The host driver
must already expose libcuda/NVML through nvidia-smi.

Options:
  --gpu N                 CUDA/NVML GPU index for detection [0]
  --gpu-kind KIND         auto, rtx3090, a100, h100, ga102, ga100, gh100 [auto]
  --cuda-arch ARCH        CUDA arch override, e.g. 86, 80, 90 [auto]
  --cuda-version X.Y      Conda CUDA toolkit version [12.1]
  --env-name NAME         Conda env name [fp16-energy-sm<arch>-cuda<XYY>]
  --env-prefix PATH       Conda env prefix instead of name
  --manager NAME          auto, mamba, conda [auto]
  --nvidia-smi PATH       nvidia-smi path [auto]
  --env-file PATH         Output env file [env/toolchain_<kind>_sm<arch>.sh]
  --run-file PATH         Output example run script [env/run_strict_<kind>_sm<arch>.sh]
  --no-install            Do not create/update the conda env; only write files/check
  --no-verify             Skip tool version/import checks
  --force-recreate        Remove and recreate an existing env
  --build-smoke           Configure/build fp16_energy_bench after install
  --dry-run               Print planned commands without executing them
  -h, --help              Show this help

Examples:
  # RTX 3090 / GA102
  ./scripts/install_gpu_toolchain.sh --gpu-kind rtx3090

  # A100 / GA100
  ./scripts/install_gpu_toolchain.sh --gpu-kind a100

  # H100 / GH100
  ./scripts/install_gpu_toolchain.sh --gpu-kind h100

After installation:
  source env/toolchain_<kind>_sm<arch>.sh
  ./scripts/run_strict_fp16_pipeline.sh --gpu 0 --cuda-arch <arch> ...
USAGE
}

log() {
  printf '[fp16-toolchain] %s\n' "$*" >&2
}

die() {
  printf '[fp16-toolchain] error: %s\n' "$*" >&2
  exit 1
}

quote_cmd() {
  printf '%q ' "$@"
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY RUN: '
    quote_cmd "$@"
    printf '\n'
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --gpu-kind) GPU_KIND="$2"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --env-prefix) ENV_PREFIX="$2"; shift 2 ;;
    --manager) MANAGER="$2"; shift 2 ;;
    --nvidia-smi) NVIDIA_SMI_BIN="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --run-file) RUN_FILE="$2"; shift 2 ;;
    --no-install) INSTALL=0; shift ;;
    --no-verify) VERIFY=0; shift ;;
    --force-recreate) FORCE_RECREATE=1; shift ;;
    --build-smoke) BUILD_SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "${GPU_KIND}" in
  auto|rtx3090|a100|h100|ga102|ga100|gh100) ;;
  *) die "--gpu-kind must be auto, rtx3090, a100, h100, ga102, ga100, or gh100" ;;
esac

case "${MANAGER}" in
  auto|mamba|conda) ;;
  *) die "--manager must be auto, mamba, or conda" ;;
esac

detect_nvidia_smi() {
  if [[ -n "${NVIDIA_SMI_BIN}" ]]; then
    printf '%s\n' "${NVIDIA_SMI_BIN}"
    return
  fi
  if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    printf '%s\n' /usr/lib/wsl/lib/nvidia-smi
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    command -v nvidia-smi
    return
  fi
  printf '%s\n' nvidia-smi
}

detect_gpu_name() {
  local smi="$1"
  local name
  name="$("${smi}" -i "${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed 's/^ *//;s/ *$//' || true)"
  case "${name}" in
    Failed\ to\ initialize*|Failed\ to\ properly*|Unable\ to\ determine*) printf '\n' ;;
    *) printf '%s\n' "${name}" ;;
  esac
}

normalize_gpu_kind() {
  local kind="$1"
  local name="$2"
  if [[ "${kind}" != "auto" ]]; then
    case "${kind}" in
      rtx3090) printf 'rtx3090\n' ;;
      a100) printf 'a100\n' ;;
      h100) printf 'h100\n' ;;
      ga102) printf 'rtx3090\n' ;;
      ga100) printf 'a100\n' ;;
      gh100) printf 'h100\n' ;;
    esac
    return
  fi

  case "${name}" in
    *RTX*3090*|*GeForce*3090*) printf 'rtx3090\n' ;;
    *A100*) printf 'a100\n' ;;
    *H100*) printf 'h100\n' ;;
    *)
      if [[ "${CUDA_ARCH}" == "auto" ]]; then
        die "could not infer GPU kind from nvidia-smi name '${name:-unknown}'; pass --gpu-kind or --cuda-arch"
      fi
      printf 'custom\n'
      ;;
  esac
}

arch_for_kind() {
  local kind="$1"
  case "${kind}" in
    rtx3090) printf '86\n' ;;
    a100) printf '80\n' ;;
    h100) printf '90\n' ;;
    custom) printf '%s\n' "${CUDA_ARCH}" ;;
    *) die "no CUDA arch mapping for GPU kind '${kind}'" ;;
  esac
}

select_manager() {
  if [[ "${MANAGER}" == "mamba" || "${MANAGER}" == "conda" ]]; then
    command -v "${MANAGER}" >/dev/null 2>&1 || die "${MANAGER} not found in PATH"
    command -v "${MANAGER}"
    return
  fi
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  die "neither mamba nor conda was found in PATH"
}

env_selector_args() {
  if [[ -n "${ENV_PREFIX}" ]]; then
    printf '%s\n' "-p" "${ENV_PREFIX}"
  else
    printf '%s\n' "-n" "${ENV_NAME}"
  fi
}

env_exists() {
  if [[ -n "${ENV_PREFIX}" ]]; then
    [[ -d "${ENV_PREFIX}" ]]
    return
  fi
  "${MANAGER_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

resolve_env_prefix() {
  if [[ -n "${ENV_PREFIX}" ]]; then
    cd "${ENV_PREFIX}" && pwd
    return
  fi
  local prefix
  prefix="$("${MANAGER_BIN}" env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF; exit}')"
  if [[ -z "${prefix}" ]]; then
    die "could not resolve conda env prefix for ${ENV_NAME}"
  fi
  printf '%s\n' "${prefix}"
}

make_abs_under_root() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${ROOT}" "${path}"
  fi
}

NVIDIA_SMI_BIN="$(detect_nvidia_smi)"
GPU_NAME="$(detect_gpu_name "${NVIDIA_SMI_BIN}")"
GPU_KIND="$(normalize_gpu_kind "${GPU_KIND}" "${GPU_NAME}")"

if [[ "${CUDA_ARCH}" == "auto" ]]; then
  CUDA_ARCH="$(arch_for_kind "${GPU_KIND}")"
fi

case "${CUDA_ARCH}" in
  80|86|90) ;;
  *) log "using custom CUDA arch ${CUDA_ARCH}" ;;
esac

CUDA_TAG="${CUDA_VERSION//./}"
if [[ -z "${ENV_NAME}" && -z "${ENV_PREFIX}" ]]; then
  ENV_NAME="fp16-energy-sm${CUDA_ARCH}-cuda${CUDA_TAG}"
fi

if [[ -z "${ENV_FILE}" ]]; then
  ENV_FILE="env/toolchain_${GPU_KIND}_sm${CUDA_ARCH}_cuda${CUDA_TAG}.sh"
fi
if [[ -z "${RUN_FILE}" ]]; then
  RUN_FILE="env/run_strict_${GPU_KIND}_sm${CUDA_ARCH}_cuda${CUDA_TAG}.sh"
fi
ENV_FILE="$(make_abs_under_root "${ENV_FILE}")"
RUN_FILE="$(make_abs_under_root "${RUN_FILE}")"

MANAGER_BIN="$(select_manager)"
mapfile -t ENV_SELECTOR < <(env_selector_args)

SPECS=(
  "python=3.10"
  "cmake"
  "ninja"
  "matplotlib"
  "nsight-compute"
  "cuda-version=${CUDA_VERSION}"
  "cuda-nvcc=${CUDA_VERSION}"
  "cuda-cudart-dev=${CUDA_VERSION}"
  "cuda-cudart-static_linux-64=${CUDA_VERSION}"
  "cuda-cccl=${CUDA_VERSION}"
  "cuda-driver-dev=${CUDA_VERSION}"
  "cuda-libraries-dev=${CUDA_VERSION}"
)

log "GPU kind: ${GPU_KIND}"
log "GPU name: ${GPU_NAME:-unknown}"
log "CUDA arch: ${CUDA_ARCH}"
log "Conda CUDA toolkit: ${CUDA_VERSION}"
log "Manager: ${MANAGER_BIN}"
log "nvidia-smi: ${NVIDIA_SMI_BIN}"

if [[ "${INSTALL}" -eq 1 ]]; then
  if env_exists; then
    if [[ "${FORCE_RECREATE}" -eq 1 ]]; then
      log "removing existing env"
      run_cmd "${MANAGER_BIN}" env remove -y "${ENV_SELECTOR[@]}"
      log "creating env"
      run_cmd "${MANAGER_BIN}" create -y "${ENV_SELECTOR[@]}" -c nvidia -c conda-forge "${SPECS[@]}"
    else
      log "updating existing env"
      run_cmd "${MANAGER_BIN}" install -y "${ENV_SELECTOR[@]}" -c nvidia -c conda-forge "${SPECS[@]}"
    fi
  else
    log "creating env"
    run_cmd "${MANAGER_BIN}" create -y "${ENV_SELECTOR[@]}" -c nvidia -c conda-forge "${SPECS[@]}"
  fi
else
  log "skipping conda env create/update"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "dry-run complete; no env file was written"
  exit 0
fi

if ! env_exists; then
  die "conda env does not exist; remove --no-install or pass a valid --env-name/--env-prefix"
fi

ENV_PREFIX="$(resolve_env_prefix)"
CMAKE_BIN="${ENV_PREFIX}/bin/cmake"
NVCC_BIN="${ENV_PREFIX}/bin/nvcc"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
NCU_BIN="${ENV_PREFIX}/bin/ncu"
if [[ ! -x "${NCU_BIN}" ]] || ! "${NCU_BIN}" --version >/dev/null 2>&1; then
  NCU_BIN="$(find "${ENV_PREFIX}" -path '*nsight-compute-*' -type f -name ncu -perm -111 2>/dev/null | sort | head -n 1 || true)"
fi
if [[ ! -x "${NCU_BIN}" ]] || ! "${NCU_BIN}" --version >/dev/null 2>&1; then
  NCU_BIN="$(find "${ENV_PREFIX}/nsight-compute" -type f -name ncu -perm -111 2>/dev/null | sort | head -n 1 || true)"
fi
[[ -x "${CMAKE_BIN}" ]] || die "cmake not found at ${CMAKE_BIN}"
[[ -x "${NVCC_BIN}" ]] || die "nvcc not found at ${NVCC_BIN}"
[[ -x "${PYTHON_BIN}" ]] || die "python not found at ${PYTHON_BIN}"
[[ -x "${NCU_BIN}" ]] || die "ncu not found under ${ENV_PREFIX}"
NCU_BIN_DIR="$(dirname "${NCU_BIN}")"

CUDA_INCLUDE_DIR=""
for candidate in \
  "${ENV_PREFIX}/targets/x86_64-linux/include" \
  "${ENV_PREFIX}/include"; do
  if [[ -f "${candidate}/cuda_runtime.h" ]]; then
    CUDA_INCLUDE_DIR="${candidate}"
    break
  fi
done
[[ -n "${CUDA_INCLUDE_DIR}" ]] || die "CUDA include dir with cuda_runtime.h not found under ${ENV_PREFIX}"

CUDA_LIB_DIR=""
for candidate in \
  "${ENV_PREFIX}/targets/x86_64-linux/lib" \
  "${ENV_PREFIX}/lib"; do
  if [[ -d "${candidate}" ]]; then
    CUDA_LIB_DIR="${candidate}"
    break
  fi
done
[[ -n "${CUDA_LIB_DIR}" ]] || die "CUDA lib dir not found under ${ENV_PREFIX}"

mkdir -p "$(dirname "${ENV_FILE}")" "$(dirname "${RUN_FILE}")"
cat > "${ENV_FILE}" <<EOF
# Generated by scripts/install_gpu_toolchain.sh
# GPU kind: ${GPU_KIND}
# GPU name: ${GPU_NAME:-unknown}
# CUDA arch: ${CUDA_ARCH}
# Conda CUDA toolkit: ${CUDA_VERSION}
export FP16_GPU_KIND="${GPU_KIND}"
export FP16_CUDA_ARCH="${CUDA_ARCH}"
export FP16_CUDA_TOOLKIT_VERSION="${CUDA_VERSION}"
export FP16_CONDA_PREFIX="${ENV_PREFIX}"
export CMAKE_BIN="${CMAKE_BIN}"
export NVCC_BIN="${NVCC_BIN}"
export NCU_BIN="${NCU_BIN}"
export PYTHON_BIN="${PYTHON_BIN}"
export NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN}"
export CMAKE_CUDA_FLAGS="-I${CUDA_INCLUDE_DIR} -L${CUDA_LIB_DIR}"
export CPATH="${CUDA_INCLUDE_DIR}\${CPATH:+:\${CPATH}}"
export LIBRARY_PATH="${CUDA_LIB_DIR}\${LIBRARY_PATH:+:\${LIBRARY_PATH}}"
export PATH="${NCU_BIN_DIR}:${ENV_PREFIX}/bin:\${PATH}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-/tmp/mpl_fp16_${GPU_KIND}_sm${CUDA_ARCH}}"
EOF
chmod +x "${ENV_FILE}"

cat > "${RUN_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
source "${ENV_FILE}"
./scripts/run_strict_fp16_pipeline.sh \\
  --gpu "${GPU_ID}" \\
  --cuda-arch "${CUDA_ARCH}" \\
  --matrix configs/fp16_matmul_launch_shape_sweep.json \\
  --threads 32,64,96,128,160,192,224,256,288,320,384 \\
  --ncu-blocks-per-sm-csv 1,2,4,8 \\
  --run-work-slope \\
  --outdir "results/strict_fp16_launch_shape_${GPU_KIND}_sm${CUDA_ARCH}"
EOF
chmod +x "${RUN_FILE}"

log "wrote env file: ${ENV_FILE}"
log "wrote run file: ${RUN_FILE}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_${GPU_KIND}_sm${CUDA_ARCH}}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${VERIFY}" -eq 1 ]]; then
  log "verifying toolchain"
  run_cmd "${CMAKE_BIN}" --version
  run_cmd "${NVCC_BIN}" --version
  run_cmd "${NCU_BIN}" --version
  run_cmd "${PYTHON_BIN}" - <<'PY'
import matplotlib
print("python/matplotlib OK", matplotlib.__version__)
PY
  if ! run_cmd "${NVIDIA_SMI_BIN}" -L; then
    log "warning: nvidia-smi check failed; fix host driver/NVML access before strict measurement"
  fi
fi

if [[ "${BUILD_SMOKE}" -eq 1 ]]; then
  log "building smoke binary"
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  SMOKE_BUILD="/tmp/fp16_energy_toolchain_smoke_${GPU_KIND}_sm${CUDA_ARCH}"
  run_cmd "${CMAKE_BIN}" -S "${ROOT}" -B "${SMOKE_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}"
  run_cmd "${CMAKE_BIN}" --build "${SMOKE_BUILD}" --parallel 2
  log "smoke binary: ${SMOKE_BUILD}/fp16_energy_bench"
fi

cat <<EOF

Next:
  cd "${ROOT}"
  source "${ENV_FILE}"
  ./scripts/run_strict_fp16_pipeline.sh --gpu ${GPU_ID} --cuda-arch ${CUDA_ARCH} --run-work-slope --outdir results/strict_fp16_${GPU_KIND}_sm${CUDA_ARCH}

Launch-shape sweep:
  "${RUN_FILE}"

NCU profiling permission note:
  If Nsight Compute fails with ERR_NVGPUCTRPERM, enable NVIDIA performance
  counters through the system/cluster policy or run the NCU validation step
  with appropriate administrator-granted profiling permission.
EOF
