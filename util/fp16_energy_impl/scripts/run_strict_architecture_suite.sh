#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"

OUTDIR=""
REPEAT=10
SAMPLE_MS=100
THREADS_CSV="32,64,96,128,160,192,224,256,288,320,384"
BUILD_DIR="build"
CALIBRATE_MATRIX=1
TARGET_TEST_S=1.0
TARGET_BASELINE_S=1.0
MAX_CALIBRATED_REPEATS=1000
DIAGNOSTIC_NO_NCU=0
REQUIRE_ARCHITECTURES="ga100,gh100,ga102"
REQUIRE_COUNTER_TRACE=0
REQUIRE_NCU_TENSOR_ACTIVITY=0
CONTINUE_ON_FAIL=0
SKIP_PREFLIGHT=0
ALLOW_COMPUTE_APPS=0
NO_POSTPROCESS=0
NO_FAIL=0
DRY_RUN=0
SPECS=()

usage() {
  cat <<'USAGE'
Usage: run_strict_architecture_suite.sh --spec LABEL:GPU:CUDA_ARCH[:NVIDIA_SMI_ID] [--spec ...] [options]

Runs strict FP16 pipeline jobs for one or more architecture targets, then runs
strict audit, architecture comparison, visualization, and Markdown report.

Examples:
  ./scripts/run_strict_architecture_suite.sh \
    --spec a100:0:80 \
    --spec rtx3090:1:86 \
    --spec h100:2:90 \
    --outdir results/strict_fp16_suite

  ./scripts/run_strict_architecture_suite.sh \
    --spec h100:0:90:GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    --require-architectures gh100 \
    --outdir results/strict_fp16_h100_suite

Options:
  --spec SPEC          LABEL:CUDA_GPU:CUDA_ARCH[:NVIDIA_SMI_ID]. Repeatable.
                       CUDA_GPU is the CUDA ordinal used by the benchmark.
                       NVIDIA_SMI_ID may be a physical index or GPU UUID.
  --outdir DIR         Suite output root [auto timestamp under results/strict_fp16_suite_*]
  --repeat N           Matrix repeat count passed to each strict run [10]
  --sample-ms N        nvidia-smi power sample interval [100]
  --threads CSV        Threads/block list [32,64,96,128,160,192,224,256,288,320,384]
  --build-dir DIR      Build directory relative to fp16_energy_impl [build]
  --no-calibrate-matrix
                       Use the base matrix without GPU-specific repeat calibration
  --target-test-s S     Target test CUDA-event duration for calibration [1.0]
  --target-baseline-s S Target baseline CUDA-event duration for calibration [1.0]
  --max-calibrated-repeats N
                       Upper bound for calibrated per-role repeats [1000]
  --diagnostic-no-ncu  Pass diagnostic no-NCU mode to each strict run
  --require-architectures CSV
                       Required architecture chips for postprocess [ga100,gh100,ga102]
  --require-counter-trace-agreement
                       Make NVML-counter/power-trace agreement a hard audit gate
  --require-ncu-tensor-activity
                       Make selected NCU tensor activity evidence a hard audit gate
  --continue-on-fail   Continue remaining specs after a strict run fails
  --skip-preflight     Skip suite-level tool/GPU/process checks before long runs
  --allow-compute-apps Allow active compute processes on target GPUs during preflight
  --no-postprocess     Run pipelines only, skip audit/compare/report
  --no-fail            Write artifacts but return success even if a run/postprocess fails
  --dry-run            Print commands without executing them
  -h, --help           Show this help

Environment overrides:
  CMAKE_BIN, PYTHON_BIN, NVIDIA_SMI_BIN, MPLCONFIGDIR, NCU_METRICS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec) SPECS+=("$2"); shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --threads) THREADS_CSV="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --no-calibrate-matrix) CALIBRATE_MATRIX=0; shift ;;
    --target-test-s) TARGET_TEST_S="$2"; shift 2 ;;
    --target-baseline-s) TARGET_BASELINE_S="$2"; shift 2 ;;
    --max-calibrated-repeats) MAX_CALIBRATED_REPEATS="$2"; shift 2 ;;
    --diagnostic-no-ncu) DIAGNOSTIC_NO_NCU=1; shift ;;
    --require-architectures) REQUIRE_ARCHITECTURES="$2"; shift 2 ;;
    --require-counter-trace-agreement) REQUIRE_COUNTER_TRACE=1; shift ;;
    --require-ncu-tensor-activity) REQUIRE_NCU_TENSOR_ACTIVITY=1; shift ;;
    --continue-on-fail) CONTINUE_ON_FAIL=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --allow-compute-apps) ALLOW_COMPUTE_APPS=1; shift ;;
    --no-postprocess) NO_POSTPROCESS=1; shift ;;
    --no-fail) NO_FAIL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${#SPECS[@]}" -eq 0 ]]; then
  echo "At least one --spec is required." >&2
  usage >&2
  exit 2
fi

if [[ -z "${OUTDIR}" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="${ROOT}/results/strict_fp16_suite_${stamp}"
elif [[ "${OUTDIR}" != /* ]]; then
  OUTDIR="${ROOT}/${OUTDIR}"
fi

quote_cmd() {
  local out=()
  local item
  for item in "$@"; do
    out+=("$(printf "%q" "${item}")")
  done
  printf '%s\n' "${out[*]}"
}

safe_label() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

validate_spec_field() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == *","* || "${value}" == *"/"* ]]; then
    echo "Invalid ${name} in --spec: ${value}" >&2
    exit 2
  fi
}

mkdir -p "${OUTDIR}/runs"
PREFLIGHT_JSON="${OUTDIR}/strict_architecture_suite_preflight.json"
PREFLIGHT_CSV="${OUTDIR}/strict_architecture_suite_preflight.csv"
RUN_STATUS_CSV="${OUTDIR}/strict_architecture_suite_runs.csv"
SUITE_SUMMARY_JSON="${OUTDIR}/strict_architecture_suite_summary.json"
printf 'label,gpu,cuda_arch,nvidia_smi_id,result_dir,status,exit_code\n' > "${RUN_STATUS_CSV}"

COMPLETED_DIRS=()
SUITE_FAILED=0
EARLY_RC=""
POSTPROCESS_RC=""

if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
  PREFLIGHT_ARGS=()
  for spec in "${SPECS[@]}"; do
    PREFLIGHT_ARGS+=(--spec "${spec}")
  done
  PREFLIGHT_ARGS+=(
    --out-json "${PREFLIGHT_JSON}"
    --out-csv "${PREFLIGHT_CSV}"
    --cmake-bin "${CMAKE_BIN}"
    --python-bin "${PYTHON_BIN}"
    --nvidia-smi-bin "${NVIDIA_SMI_BIN}"
  )
  if [[ "${DIAGNOSTIC_NO_NCU}" -eq 0 ]]; then
    PREFLIGHT_ARGS+=(--require-ncu)
  fi
  if [[ "${ALLOW_COMPUTE_APPS}" -eq 1 ]]; then
    PREFLIGHT_ARGS+=(--allow-compute-apps)
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    PREFLIGHT_ARGS+=(--dry-run)
  fi
  if [[ "${NO_FAIL}" -eq 1 ]]; then
    PREFLIGHT_ARGS+=(--no-fail)
  fi

  set +e
  "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight_strict_architecture_suite.py" "${PREFLIGHT_ARGS[@]}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "Suite preflight failed. See ${PREFLIGHT_JSON} and ${PREFLIGHT_CSV}." >&2
    exit "${rc}"
  fi
fi

for spec in "${SPECS[@]}"; do
  IFS=':' read -r label gpu cuda_arch nvidia_smi_id extra <<< "${spec}"
  if [[ -n "${extra:-}" ]]; then
    echo "Invalid --spec ${spec}; expected LABEL:GPU:CUDA_ARCH[:NVIDIA_SMI_ID]." >&2
    exit 2
  fi
  validate_spec_field "label" "${label:-}"
  validate_spec_field "gpu" "${gpu:-}"
  validate_spec_field "cuda_arch" "${cuda_arch:-}"

  run_label="$(safe_label "${label}")"
  run_dir="${OUTDIR}/runs/${run_label}"
  cmd=(
    "${SCRIPT_DIR}/run_strict_fp16_pipeline.sh"
    --gpu "${gpu}"
    --cuda-arch "${cuda_arch}"
    --outdir "${run_dir}"
    --repeat "${REPEAT}"
    --sample-ms "${SAMPLE_MS}"
    --threads "${THREADS_CSV}"
    --build-dir "${BUILD_DIR}"
    --target-test-s "${TARGET_TEST_S}"
    --target-baseline-s "${TARGET_BASELINE_S}"
    --max-calibrated-repeats "${MAX_CALIBRATED_REPEATS}"
  )
  if [[ -n "${nvidia_smi_id:-}" ]]; then
    cmd+=(--nvidia-smi-id "${nvidia_smi_id}")
  fi
  if [[ "${CALIBRATE_MATRIX}" -eq 0 ]]; then
    cmd+=(--no-calibrate-matrix)
  fi
  if [[ "${DIAGNOSTIC_NO_NCU}" -eq 1 ]]; then
    cmd+=(--diagnostic-no-ncu)
  fi

  echo "Strict FP16 suite target: ${label} (gpu=${gpu}, cuda_arch=${cuda_arch}, out=${run_dir})"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY RUN: CMAKE_BIN=%q PYTHON_BIN=%q NVIDIA_SMI_BIN=%q %s\n' \
      "${CMAKE_BIN}" "${PYTHON_BIN}" "${NVIDIA_SMI_BIN}" "$(quote_cmd "${cmd[@]}")"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${label}" "${gpu}" "${cuda_arch}" "${nvidia_smi_id:-}" "${run_dir}" "dry_run" "0" >> "${RUN_STATUS_CSV}"
    COMPLETED_DIRS+=("${run_dir}")
    continue
  fi

  set +e
  CMAKE_BIN="${CMAKE_BIN}" PYTHON_BIN="${PYTHON_BIN}" NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN}" "${cmd[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${label}" "${gpu}" "${cuda_arch}" "${nvidia_smi_id:-}" "${run_dir}" "completed" "${rc}" >> "${RUN_STATUS_CSV}"
    COMPLETED_DIRS+=("${run_dir}")
  else
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${label}" "${gpu}" "${cuda_arch}" "${nvidia_smi_id:-}" "${run_dir}" "failed" "${rc}" >> "${RUN_STATUS_CSV}"
    SUITE_FAILED=1
    if [[ "${CONTINUE_ON_FAIL}" -eq 0 ]]; then
      EARLY_RC="${rc}"
      break
    fi
  fi
done

POSTPROCESS_DIR="${OUTDIR}/postprocess"
if [[ "${NO_POSTPROCESS}" -eq 0 && "${#COMPLETED_DIRS[@]}" -gt 0 ]]; then
  post_cmd=(
    "${SCRIPT_DIR}/postprocess_strict_architectures.sh"
    --outdir "${POSTPROCESS_DIR}"
    --require-architectures "${REQUIRE_ARCHITECTURES}"
  )
  if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
    post_cmd+=(--suite-preflight-json "${PREFLIGHT_JSON}" --suite-preflight-csv "${PREFLIGHT_CSV}")
  fi
  if [[ "${REQUIRE_COUNTER_TRACE}" -eq 1 ]]; then
    post_cmd+=(--require-counter-trace-agreement)
  fi
  if [[ "${REQUIRE_NCU_TENSOR_ACTIVITY}" -eq 1 ]]; then
    post_cmd+=(--require-ncu-tensor-activity)
  fi
  if [[ "${NO_FAIL}" -eq 1 ]]; then
    post_cmd+=(--no-fail)
  fi
  post_cmd+=("${COMPLETED_DIRS[@]}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY RUN: $(quote_cmd "${post_cmd[@]}")"
  else
    set +e
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_suite}" "${post_cmd[@]}"
    rc=$?
    set -e
    POSTPROCESS_RC="${rc}"
    if [[ "${rc}" -ne 0 ]]; then
      SUITE_FAILED=1
    fi
  fi
fi

SUMMARY_ARGS=(--out "${SUITE_SUMMARY_JSON}" --outdir "${OUTDIR}" --run-status-csv "${RUN_STATUS_CSV}")
for spec in "${SPECS[@]}"; do
  SUMMARY_ARGS+=(--spec "${spec}")
done
if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
  SUMMARY_ARGS+=(--preflight-json "${PREFLIGHT_JSON}")
fi
if [[ -n "${POSTPROCESS_DIR}" ]]; then
  SUMMARY_ARGS+=(--postprocess-dir "${POSTPROCESS_DIR}")
fi
if [[ -n "${POSTPROCESS_RC}" ]]; then
  SUMMARY_ARGS+=(--postprocess-exit-code "${POSTPROCESS_RC}")
fi
if [[ "${NO_POSTPROCESS}" -eq 1 || "${DRY_RUN}" -eq 1 || "${#COMPLETED_DIRS[@]}" -eq 0 ]]; then
  SUMMARY_ARGS+=(--postprocess-skipped)
fi
if [[ "${SUITE_FAILED}" -ne 0 ]]; then
  SUMMARY_ARGS+=(--suite-failed 1)
else
  SUMMARY_ARGS+=(--suite-failed 0)
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  SUMMARY_ARGS+=(--dry-run)
fi
if [[ "${SKIP_PREFLIGHT}" -eq 1 ]]; then
  SUMMARY_ARGS+=(--skip-preflight)
fi
if [[ "${NO_POSTPROCESS}" -eq 1 ]]; then
  SUMMARY_ARGS+=(--no-postprocess)
fi
SUMMARY_ARGS+=(--require-architectures "${REQUIRE_ARCHITECTURES}")
"${PYTHON_BIN}" "${SCRIPT_DIR}/write_strict_suite_summary.py" "${SUMMARY_ARGS[@]}"

cat <<EOF
Strict FP16 architecture suite complete:
  output root: ${OUTDIR}
  preflight: ${PREFLIGHT_JSON}
  suite summary: ${SUITE_SUMMARY_JSON}
  run status: ${RUN_STATUS_CSV}
  postprocess: ${POSTPROCESS_DIR}
  completed runs: ${#COMPLETED_DIRS[@]}
  failed: ${SUITE_FAILED}
EOF

if [[ "${NO_FAIL}" -eq 0 && "${SUITE_FAILED}" -ne 0 ]]; then
  if [[ -n "${EARLY_RC}" && "${EARLY_RC}" -ne 0 ]]; then
    exit "${EARLY_RC}"
  fi
  if [[ -n "${POSTPROCESS_RC}" && "${POSTPROCESS_RC}" -ne 0 ]]; then
    exit "${POSTPROCESS_RC}"
  fi
  exit 1
fi
