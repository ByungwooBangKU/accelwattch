#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTDIR=""
SUITE_PREFLIGHT_JSON=""
SUITE_PREFLIGHT_CSV=""
REQUIRE_ARCHITECTURES="ga100,gh100,ga102"
REQUIRE_KERNEL="tensor_mma_f16acc"
REQUIRE_BASELINE="tensor_baseline_mov"
NO_FAIL=0
REQUIRE_COUNTER_TRACE=0
REQUIRE_NCU_TENSOR_ACTIVITY=1
INPUTS=()

usage() {
  cat <<'USAGE'
Usage: postprocess_strict_architectures.sh [options] RESULT_DIR...

Runs the multi-GPU strict FP16 postprocess flow:
  strict audit -> architecture comparison figures -> Markdown report

Options:
  --outdir DIR          Output root [results/strict_fp16_postprocess_<timestamp>]
  --suite-preflight-json FILE
                        Suite-level strict_architecture_suite_preflight.json evidence
  --suite-preflight-csv FILE
                        Suite-level strict_architecture_suite_preflight.csv evidence
  --require-architectures CSV
                        Required architecture chips [ga100,gh100,ga102]
  --require-kernel KERNEL
                        Required test kernel [tensor_mma_f16acc]
  --require-baseline KERNEL
                        Required baseline kernel [tensor_baseline_mov]
  --require-counter-trace-agreement
                        Make NVML-counter/power-trace agreement a hard audit gate
  --require-ncu-tensor-activity
                        Keep selected NCU tensor activity evidence as a hard audit gate [default]
  --no-require-ncu-tensor-activity
                        Diagnostic mode: do not hard-fail missing selected NCU tensor activity
  --no-fail             Always return success after writing artifacts
  -h, --help            Show this help

Environment overrides:
  PYTHON_BIN, MPLCONFIGDIR
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2 ;;
    --suite-preflight-json) SUITE_PREFLIGHT_JSON="$2"; shift 2 ;;
    --suite-preflight-csv) SUITE_PREFLIGHT_CSV="$2"; shift 2 ;;
    --require-architectures) REQUIRE_ARCHITECTURES="$2"; shift 2 ;;
    --require-kernel) REQUIRE_KERNEL="$2"; shift 2 ;;
    --require-baseline) REQUIRE_BASELINE="$2"; shift 2 ;;
    --require-counter-trace-agreement) REQUIRE_COUNTER_TRACE=1; shift ;;
    --require-ncu-tensor-activity) REQUIRE_NCU_TENSOR_ACTIVITY=1; shift ;;
    --no-require-ncu-tensor-activity) REQUIRE_NCU_TENSOR_ACTIVITY=0; shift ;;
    --no-fail) NO_FAIL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) INPUTS+=("$1"); shift ;;
  esac
done

while [[ $# -gt 0 ]]; do
  INPUTS+=("$1")
  shift
done

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "At least one RESULT_DIR is required." >&2
  usage >&2
  exit 2
fi

if [[ -z "${OUTDIR}" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="${ROOT}/results/strict_fp16_postprocess_${stamp}"
elif [[ "${OUTDIR}" != /* ]]; then
  OUTDIR="${ROOT}/${OUTDIR}"
fi

ABS_INPUTS=()
for input in "${INPUTS[@]}"; do
  if [[ "${input}" = /* ]]; then
    ABS_INPUTS+=("${input}")
  else
    ABS_INPUTS+=("${ROOT}/${input}")
  fi
done

AUDIT_DIR="${OUTDIR}/strict_fp16_audit"
COMPARE_DIR="${OUTDIR}/architecture_compare_fp16"
ARCH_MODEL_DIR="${OUTDIR}/architecture_models"
REPORT_DIR="${OUTDIR}/strict_fp16_report"

mkdir -p "${OUTDIR}"

AUDIT_ARGS=(
  --input "${ABS_INPUTS[@]}"
  --outdir "${AUDIT_DIR}"
  --require-architectures "${REQUIRE_ARCHITECTURES}"
  --require-kernel "${REQUIRE_KERNEL}"
  --require-baseline "${REQUIRE_BASELINE}"
  --no-fail
)
if [[ "${REQUIRE_COUNTER_TRACE}" -eq 1 ]]; then
  AUDIT_ARGS+=(--require-counter-trace-agreement)
fi
if [[ "${REQUIRE_NCU_TENSOR_ACTIVITY}" -eq 1 ]]; then
  AUDIT_ARGS+=(--require-ncu-tensor-activity)
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_postprocess}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_strict_results.py" "${AUDIT_ARGS[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_postprocess}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_architectures.py" \
  --input "${ABS_INPUTS[@]}" \
  --outdir "${COMPARE_DIR}" \
  --audit-dir "${AUDIT_DIR}" \
  --require-architectures "${REQUIRE_ARCHITECTURES}" \
  --require-kernel "${REQUIRE_KERNEL}" \
  --require-baseline "${REQUIRE_BASELINE}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_postprocess}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/architecture_models.py" \
  --outdir "${ARCH_MODEL_DIR}" \
  --require-architectures "${REQUIRE_ARCHITECTURES}" \
  --fail-on-model-error-pct 1.0 \
  --fail-on-missing-metadata

REPORT_ARGS=(
  --audit-dir "${AUDIT_DIR}" \
  --compare-dir "${COMPARE_DIR}" \
  --architecture-model-dir "${ARCH_MODEL_DIR}" \
  --outdir "${REPORT_DIR}" \
  --require-architectures "${REQUIRE_ARCHITECTURES}"
)
if [[ -n "${SUITE_PREFLIGHT_JSON}" ]]; then
  REPORT_ARGS+=(--suite-preflight-json "${SUITE_PREFLIGHT_JSON}")
fi
if [[ -n "${SUITE_PREFLIGHT_CSV}" ]]; then
  REPORT_ARGS+=(--suite-preflight-csv "${SUITE_PREFLIGHT_CSV}")
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_fp16_postprocess}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/report_strict_results.py" "${REPORT_ARGS[@]}"

read -r AUDIT_PASS REPORT_REQUIREMENTS_PASS OVERALL_PASS < <("${PYTHON_BIN}" - \
  "${AUDIT_DIR}/strict_result_audit.json" \
  "${REPORT_DIR}/fp16_strict_report_requirements.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
requirements_path = Path(sys.argv[2])

audit_pass = False
if audit_path.exists():
    with audit_path.open() as f:
        payload = json.load(f)
    audit_pass = bool(payload.get("overall_pass"))

requirements_pass = False
if requirements_path.exists():
    with requirements_path.open() as f:
        rows = list(csv.DictReader(f))
    requirements_pass = bool(rows) and all(str(row.get("status", "")) == "pass" for row in rows)

overall_pass = audit_pass and requirements_pass
print(
    ("true" if audit_pass else "false"),
    ("true" if requirements_pass else "false"),
    ("true" if overall_pass else "false"),
)
PY
)

cat <<EOF
Strict FP16 postprocess complete:
  output root: ${OUTDIR}
  audit: ${AUDIT_DIR}/strict_result_audit.csv
  compare: ${COMPARE_DIR}/architecture_best_fp16.csv
  architecture models: ${ARCH_MODEL_DIR}/architecture_model_summary.csv
  report: ${REPORT_DIR}/fp16_strict_report.md
  audit_pass: ${AUDIT_PASS}
  report_requirements_pass: ${REPORT_REQUIREMENTS_PASS}
  overall_pass: ${OVERALL_PASS}
EOF

if [[ "${NO_FAIL}" -eq 0 && "${OVERALL_PASS}" != "true" ]]; then
  exit 1
fi
