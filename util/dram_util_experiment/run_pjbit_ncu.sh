#!/usr/bin/env bash
# Run Nsight Compute validation for DRAM/L2 counters.
#
# This is intentionally separate from the NVML power run. Nsight Compute may
# replay kernels and perturb timing/power, so its output is only for traffic
# validation, not pJ/bit calculation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./run_pjbit_ncu.sh --device 0 --tag a100_ncu --buf-bytes 8589934592

Options:
  --device N             CUDA/NVML GPU index. Default: 0
  --tag TAG              Output tag. Default: ncu_validation
  --modes "..."          read, write, or "read write". Default: read write
  --write-patterns "..." Write patterns. Default: random
  --target N             Active target to profile. Default: 100
  --buf-bytes N          Buffer bytes per mode. Default: dram_pjbit_cupy.py auto
  --phase-seconds N      Short NCU validation phase length. Default: 1
  --idle-seconds N       Idle baseline length inside validation process. Default: 0.2
  --window-ms N          Duty window. Default: 200
  --out-dir DIR          Output directory. Default: reports
  --ncu-bin PATH         Nsight Compute CLI. Default: ncu
  --ncu-set NAME         NCU metric set when --ncu-metrics is empty. Default: full
  --ncu-metrics CSV      Explicit NCU metric list. Overrides --ncu-set.
  --launch-skip N        Kernel launches to skip before profiling. Default: 2
  --launch-count N       Kernel launches to profile. Default: 1
  --                    Extra args passed to run_pjbit_cupy.sh
EOF
}

DEVICE="0"
TAG="ncu_validation"
MODES_STR="read write"
WRITE_PATTERNS_STR="random"
TARGET="100"
BUF_BYTES=""
PHASE_SECONDS="1"
IDLE_SECONDS="0.2"
WINDOW_MS="200"
OUT_DIR="reports"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="full"
NCU_METRICS=""
LAUNCH_SKIP="2"
LAUNCH_COUNT="1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --modes) MODES_STR="$2"; shift 2 ;;
        --write-patterns) WRITE_PATTERNS_STR="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --buf-bytes) BUF_BYTES="$2"; shift 2 ;;
        --phase-seconds) PHASE_SECONDS="$2"; shift 2 ;;
        --idle-seconds) IDLE_SECONDS="$2"; shift 2 ;;
        --window-ms) WINDOW_MS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --ncu-bin) NCU_BIN="$2"; shift 2 ;;
        --ncu-set) NCU_SET="$2"; shift 2 ;;
        --ncu-metrics) NCU_METRICS="$2"; shift 2 ;;
        --launch-skip) LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count) LAUNCH_COUNT="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "[err] unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v "$NCU_BIN" >/dev/null 2>&1; then
    echo "[err] Nsight Compute CLI not found: $NCU_BIN" >&2
    echo "      Install Nsight Compute or pass --ncu-bin /path/to/ncu." >&2
    exit 1
fi

CUPY_RUNNER="$SCRIPT_DIR/run_pjbit_cupy.sh"
if [[ ! -x "$CUPY_RUNNER" ]]; then
    echo "[err] missing executable: $CUPY_RUNNER" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
read -r -a MODES <<< "$MODES_STR"
read -r -a WRITE_PATTERNS <<< "$WRITE_PATTERNS_STR"

common_app_args=(
    --device "$DEVICE"
    --targets "$TARGET"
    --phase-seconds "$PHASE_SECONDS"
    --idle-seconds "$IDLE_SECONDS"
    --window-ms "$WINDOW_MS"
    --poll-hz 10
    --gap-seconds 0.1
    --phase-order target-major
    --cal-passes 1
    --cal-repeats 1
    --out-dir "$OUT_DIR"
)
if [[ -n "$BUF_BYTES" ]]; then
    common_app_args+=(--buf-bytes "$BUF_BYTES")
fi

run_one() {
    local label="$1"
    local kernel="$2"
    shift 2
    local report_base="$OUT_DIR/${TAG}_${label}"

    local ncu_args=(
        "$NCU_BIN"
        --target-processes all
        --kernel-name "regex:${kernel}"
        --launch-skip "$LAUNCH_SKIP"
        --launch-count "$LAUNCH_COUNT"
        --force-overwrite
        --export "$report_base"
    )
    if [[ -n "$NCU_METRICS" ]]; then
        ncu_args+=(--metrics "$NCU_METRICS")
    else
        ncu_args+=(--set "$NCU_SET")
    fi

    echo
    echo "[ncu] $label kernel=$kernel report=${report_base}.ncu-rep"
    "${ncu_args[@]}" "$CUPY_RUNNER" "${common_app_args[@]}" "$@" "${EXTRA_ARGS[@]}"
}

for mode in "${MODES[@]}"; do
    case "$mode" in
        read)
            run_one "read" "stream_read" --modes read --tag "${TAG}_read"
            ;;
        write)
            for pattern in "${WRITE_PATTERNS[@]}"; do
                run_one "write_${pattern}" "stream_write" \
                    --modes write --write-patterns "$pattern" --tag "${TAG}_write_${pattern}"
            done
            ;;
        *)
            echo "[err] unknown mode for --modes: $mode" >&2
            exit 2
            ;;
    esac
done

echo
echo "[done] Nsight Compute reports: $OUT_DIR/${TAG}_*.ncu-rep"
