#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTDIR=""
SUITE_PREFLIGHT_JSON=""
SUITE_PREFLIGHT_CSV=""
REQUIRE_ARCHITECTURES="ga100,gh100,ga102"
NO_FAIL=0
REQUIRE_COUNTER_TRACE=0
REQUIRE_NCU_TENSOR_ACTIVITY=0
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
  --require-counter-trace-agreement
                        Make NVML-counter/power-trace agreement a hard audit gate
  --require-ncu-tensor-activity
                        Make selected NCU tensor activity evidence a hard audit gate
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
    --require-counter-trace-agreement) REQUIRE_COUNTER_TRACE=1; shift ;;
    --require-ncu-tensor-activity) REQUIRE_NCU_TENSOR_ACTIVITY=1; shift ;;
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
REPORT_DIR="${OUTDIR}/strict_fp16_report"

mkdir -p "${OUTDIR}"

AUDIT_ARGS=(
  --input "${ABS_INPUTS[@]}"
  --outdir "${AUDIT_DIR}"
  --require-architectures "${REQUIRE_ARCHITECTURES}"
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
  --outdir "${COMPARE_DIR}"

REPORT_ARGS=(
  --audit-dir "${AUDIT_DIR}" \
  --compare-dir "${COMPARE_DIR}" \
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

OVERALL_PASS="$("${PYTHON_BIN}" - "${AUDIT_DIR}/strict_result_audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit(0)
with path.open() as f:
    payload = json.load(f)
print("true" if payload.get("overall_pass") else "false")
PY
)"

cat <<EOF
Strict FP16 postprocess complete:
  output root: ${OUTDIR}
  audit: ${AUDIT_DIR}/strict_result_audit.csv
  compare: ${COMPARE_DIR}/architecture_best_fp16.csv
  report: ${REPORT_DIR}/fp16_strict_report.md
  overall_pass: ${OVERALL_PASS}
EOF

if [[ "${NO_FAIL}" -eq 0 && "${OVERALL_PASS}" != "true" ]]; then
  exit 1
fi
