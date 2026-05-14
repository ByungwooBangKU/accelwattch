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
  --ncu-set NAME         NCU metric set fallback when auto metrics are unavailable. Default: full
  --ncu-metrics CSV      Explicit metric CSV, "auto", or "set" for --ncu-set. Default: auto
  --launch-skip N        Kernel launches to skip before profiling. Default: 2
  --launch-count N       Kernel launches to profile. Default: 1
  --flat-out-dir         Write directly to --out-dir instead of DIR/<gpu>_<YYYYMMDDHHMM>
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
NCU_METRICS="auto"
LAUNCH_SKIP="2"
LAUNCH_COUNT="1"
FLAT_OUT_DIR="0"
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
        --flat-out-dir) FLAT_OUT_DIR="1"; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "[err] unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

sanitize_name() {
    local value="$1"
    local safe
    safe="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    safe="${safe//[^a-z0-9]/_}"
    while [[ "$safe" == _* ]]; do safe="${safe#_}"; done
    while [[ "$safe" == *_ ]]; do safe="${safe%_}"; done
    printf '%s' "${safe:-gpu}"
}

gpu_name_for_output() {
    local name
    name="$(nvidia-smi --id="$DEVICE" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [[ -z "$name" ]]; then
        name="gpu${DEVICE}"
    fi
    printf '%s' "$name"
}

join_by_comma() {
    local IFS=,
    printf '%s' "$*"
}

metric_is_available() {
    local metric="$1"
    local escaped="${metric//./\\.}"
    grep -Eq "(^|[[:space:]])${escaped}([[:space:]]|$)" <<< "$AVAILABLE_NCU_METRICS"
}

select_auto_ncu_metrics() {
    local selected=()
    local candidates=(
        gpu__time_duration.sum
        dram__bytes_read.sum
        dram__bytes_write.sum
        dram__sectors_read.sum
        dram__sectors_write.sum
        dram__throughput.avg.pct_of_peak_sustained_elapsed
        lts__t_sectors_op_read.sum
        lts__t_sectors_op_write.sum
        lts__t_sectors_srcunit_tex_op_read.sum
        lts__t_sectors_srcunit_tex_op_write.sum
        lts__t_sectors_srcunit_l1_op_read.sum
        lts__t_sectors_srcunit_l1_op_write.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
        sm__inst_executed_pipe_lsu.sum
        sm__inst_executed_pipe_fma.sum
        sm__inst_executed_pipe_tensor.sum
        smsp__inst_executed_pipe_lsu.sum
        smsp__inst_executed_pipe_fma.sum
        smsp__inst_executed_pipe_tensor.sum
    )

    for metric in "${candidates[@]}"; do
        if metric_is_available "$metric"; then
            selected+=("$metric")
        fi
    done

    join_by_comma "${selected[@]}"
}

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

print_ncu_permission_help() {
    cat >&2 <<'EOF'
[err] Nsight Compute cannot access NVIDIA GPU performance counters.
      This is ERR_NVGPUCTRPERM, not a DRAM experiment failure.

      The NVML power experiment still works without --ncu-profile.
      To run NCU validation on Linux, use one of:
        1. Run the NCU wrapper with elevated privilege, e.g. sudo ./run_pjbit_ncu.sh ...
        2. Ask the system administrator to enable non-admin performance counters:
           options nvidia NVreg_RestrictProfilingToAdminUsers=0
           in a /etc/modprobe.d/*.conf file, then reboot or reload the NVIDIA module.
        3. In a container, enable counters on the host and launch the container with SYS_ADMIN capability.

      Check the current driver setting:
        grep RmProfilingAdminOnly /proc/driver/nvidia/params
EOF
}

BASE_OUT_DIR="$OUT_DIR"
GPU_OUTPUT_NAME="$(gpu_name_for_output)"
if [[ "$FLAT_OUT_DIR" != "1" ]]; then
    RUN_STAMP="$(date +%Y%m%d%H%M)"
    OUT_DIR="$BASE_OUT_DIR/$(sanitize_name "$GPU_OUTPUT_NAME")_${RUN_STAMP}"
fi
mkdir -p "$OUT_DIR"
echo "[info] output=$OUT_DIR"
read -r -a MODES <<< "$MODES_STR"
read -r -a WRITE_PATTERNS <<< "$WRITE_PATTERNS_STR"

NCU_METRIC_ARGS=()
if [[ "$NCU_METRICS" == "auto" ]]; then
    AVAILABLE_NCU_METRICS="$("$NCU_BIN" --query-metrics 2>/dev/null || true)"
    AUTO_NCU_METRICS="$(select_auto_ncu_metrics)"
    if [[ -n "$AUTO_NCU_METRICS" ]]; then
        NCU_METRIC_ARGS=(--metrics "$AUTO_NCU_METRICS")
        echo "[info] ncu-metrics=auto selected compact metric list"
        echo "[info] ncu-metrics=$AUTO_NCU_METRICS"
    else
        NCU_METRIC_ARGS=(--set "$NCU_SET")
        echo "[warn] ncu-metrics=auto found no known metrics; falling back to --set $NCU_SET" >&2
    fi
elif [[ "$NCU_METRICS" == "set" ]]; then
    NCU_METRIC_ARGS=(--set "$NCU_SET")
elif [[ -n "$NCU_METRICS" ]]; then
    NCU_METRIC_ARGS=(--metrics "$NCU_METRICS")
else
    NCU_METRIC_ARGS=(--set "$NCU_SET")
fi

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
    --flat-output
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
    ncu_args+=("${NCU_METRIC_ARGS[@]}")

    echo
    echo "[ncu] $label kernel=$kernel report=${report_base}.ncu-rep"
    local log_file="${report_base}.ncu.log"

    set +e
    "${ncu_args[@]}" "$CUPY_RUNNER" "${common_app_args[@]}" "$@" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log_file"
    local ncu_status=${PIPESTATUS[0]}
    set -e

    if grep -q "ERR_NVGPUCTRPERM" "$log_file"; then
        print_ncu_permission_help
        echo "[err] NCU log: $log_file" >&2
        exit 13
    fi

    if [[ "$ncu_status" -ne 0 ]]; then
        echo "[err] Nsight Compute failed with exit code $ncu_status" >&2
        echo "[err] NCU log: $log_file" >&2
        exit "$ncu_status"
    fi
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
