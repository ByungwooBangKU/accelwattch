#!/usr/bin/env bash
# Run Nsight Compute validation for hierarchy_pjbit_cupy.py workloads.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./run_hierarchy_ncu.sh --device 0 --tag h100_hierarchy_ncu --dram-buf-bytes 8589934592

Options:
  --device N             CUDA/NVML GPU index. Default: 0
  --tag TAG              Output tag. Default: hierarchy_ncu
  --workloads "..."      Workloads to profile. Default: representative read/write zero
  --write-patterns "..." Patterns passed to hierarchy runner. Default: zero const address toggle random
  --dram-buf-bytes N     DRAM-stream buffer bytes. Default: hierarchy runner auto
  --l2-buf-bytes N       L2-resident buffer bytes. Default: hierarchy runner auto
  --phase-seconds N      Short NCU validation phase length. Default: 1
  --idle-seconds N       Idle baseline inside validation process. Default: 0.2
  --out-dir DIR          Output directory. Default: reports
  --flat-out-dir         Write directly to --out-dir
  --ncu-bin PATH         Nsight Compute CLI. Default: ncu; common install paths are auto-detected
  --ncu-metrics CSV      Explicit metric CSV, "auto", or "set" for --ncu-set. Default: auto
  --ncu-set NAME         Metric set fallback. Default: full
  --launch-skip N        Kernel launches to skip before profiling. Default: 2
  --launch-count N       Kernel launches to profile. Default: 1
  --                    Extra args passed to run_hierarchy_pjbit.sh
EOF
}

DEVICE="0"
TAG="hierarchy_ncu"
WORKLOADS_STR="l2_read dram_read l2_write_zero dram_write_zero"
WRITE_PATTERNS_STR="zero const address toggle random"
DRAM_BUF_BYTES=""
L2_BUF_BYTES=""
PHASE_SECONDS="1"
IDLE_SECONDS="0.2"
OUT_DIR="reports"
FLAT_OUT_DIR="0"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_METRICS="auto"
NCU_SET="full"
LAUNCH_SKIP="2"
LAUNCH_COUNT="1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --workloads) WORKLOADS_STR="$2"; shift 2 ;;
        --write-patterns) WRITE_PATTERNS_STR="$2"; shift 2 ;;
        --dram-buf-bytes) DRAM_BUF_BYTES="$2"; shift 2 ;;
        --l2-buf-bytes) L2_BUF_BYTES="$2"; shift 2 ;;
        --phase-seconds) PHASE_SECONDS="$2"; shift 2 ;;
        --idle-seconds) IDLE_SECONDS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --flat-out-dir) FLAT_OUT_DIR="1"; shift ;;
        --ncu-bin|--ncubin|--ncu_bin) NCU_BIN="$2"; shift 2 ;;
        --ncu-metrics) NCU_METRICS="$2"; shift 2 ;;
        --ncu-set) NCU_SET="$2"; shift 2 ;;
        --launch-skip) LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count) LAUNCH_COUNT="$2"; shift 2 ;;
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

resolve_ncu_bin() {
    local requested="$1"
    local found
    if [[ "$requested" == */* ]]; then
        [[ -x "$requested" ]] && printf '%s' "$requested" && return 0
        return 1
    fi
    found="$(command -v "$requested" 2>/dev/null || true)"
    if [[ -n "$found" && -x "$found" ]]; then
        printf '%s' "$found"
        return 0
    fi
    local old_nullglob
    old_nullglob="$(shopt -p nullglob || true)"
    shopt -s nullglob
    local candidates=(
        /usr/local/cuda/bin/ncu
        /usr/local/cuda-*/bin/ncu
        /usr/local/cuda/nsight-compute-*/ncu
        /usr/local/cuda-*/nsight-compute-*/ncu
        /opt/nvidia/nsight-compute/ncu
        /opt/nvidia/nsight-compute/*/ncu
        /usr/local/NVIDIA-Nsight-Compute*/ncu
    )
    eval "$old_nullglob"
    for found in "${candidates[@]}"; do
        [[ -x "$found" ]] && printf '%s' "$found" && return 0
    done
    return 1
}

workload_kernel_regex() {
    local workload="$1"
    case "$workload" in
        control_*_read) printf '%s' "hierarchy_control_read" ;;
        l2_read|dram_read) printf '%s' "hierarchy_read" ;;
        control_*_write_*) printf '%s' "hierarchy_control_write" ;;
        l2_write_*|dram_write_*) printf '%s' "hierarchy_write" ;;
        *) echo "[err] cannot infer kernel for workload: $workload" >&2; exit 2 ;;
    esac
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
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
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

if ! NCU_BIN_RESOLVED="$(resolve_ncu_bin "$NCU_BIN")"; then
    echo "[err] Nsight Compute CLI not found: $NCU_BIN" >&2
    echo "      Install Nsight Compute or pass --ncu-bin /path/to/ncu." >&2
    exit 1
fi
NCU_BIN="$NCU_BIN_RESOLVED"

BASE_OUT_DIR="$OUT_DIR"
if [[ "$FLAT_OUT_DIR" != "1" ]]; then
    OUT_DIR="$BASE_OUT_DIR/$(sanitize_name "$(gpu_name_for_output)")_$(date +%Y%m%d%H%M)"
fi
mkdir -p "$OUT_DIR"

NCU_METRIC_ARGS=()
if [[ "$NCU_METRICS" == "auto" ]]; then
    AVAILABLE_NCU_METRICS="$("$NCU_BIN" --query-metrics 2>/dev/null || true)"
    AUTO_NCU_METRICS="$(select_auto_ncu_metrics)"
    if [[ -n "$AUTO_NCU_METRICS" ]]; then
        NCU_METRIC_ARGS=(--metrics "$AUTO_NCU_METRICS")
        echo "[info] ncu-metrics=$AUTO_NCU_METRICS"
    else
        NCU_METRIC_ARGS=(--set "$NCU_SET")
        echo "[warn] no known auto NCU metrics found; falling back to --set $NCU_SET" >&2
    fi
elif [[ "$NCU_METRICS" == "set" ]]; then
    NCU_METRIC_ARGS=(--set "$NCU_SET")
else
    NCU_METRIC_ARGS=(--metrics "$NCU_METRICS")
fi

read -r -a WORKLOADS <<< "$WORKLOADS_STR"
read -r -a WRITE_PATTERNS <<< "$WRITE_PATTERNS_STR"

for workload in "${WORKLOADS[@]}"; do
    kernel="$(workload_kernel_regex "$workload")"
    report_base="$OUT_DIR/${TAG}_${workload}"
    app_cmd=(
        "$SCRIPT_DIR/run_hierarchy_pjbit.sh"
        --device "$DEVICE"
        --tag "${TAG}_${workload}"
        --write-patterns "${WRITE_PATTERNS[@]}"
        --only-workload "$workload"
        --phase-seconds "$PHASE_SECONDS"
        --idle-seconds "$IDLE_SECONDS"
        --out-dir "$OUT_DIR"
        --flat-output
        --cal-passes 1
        --cal-repeats 1
    )
    if [[ -n "$DRAM_BUF_BYTES" ]]; then
        app_cmd+=(--dram-buf-bytes "$DRAM_BUF_BYTES")
    fi
    if [[ -n "$L2_BUF_BYTES" ]]; then
        app_cmd+=(--l2-buf-bytes "$L2_BUF_BYTES")
    fi
    app_cmd+=("${EXTRA_ARGS[@]}")

    echo
    echo "[ncu] workload=$workload kernel=$kernel report=${report_base}.ncu-rep"
    set +e
    "$NCU_BIN" \
        --target-processes all \
        --kernel-name "regex:${kernel}" \
        --launch-skip "$LAUNCH_SKIP" \
        --launch-count "$LAUNCH_COUNT" \
        --force-overwrite \
        --export "$report_base" \
        "${NCU_METRIC_ARGS[@]}" \
        "${app_cmd[@]}" 2>&1 | tee "${report_base}.ncu.log"
    status=${PIPESTATUS[0]}
    set -e
    if grep -q "ERR_NVGPUCTRPERM" "${report_base}.ncu.log"; then
        echo "[err] NCU performance counter permission denied." >&2
        echo "      Run sudo -v first or ask admin to set NVreg_RestrictProfilingToAdminUsers=0." >&2
        exit 13
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "[err] NCU failed with exit code $status; log=${report_base}.ncu.log" >&2
        exit "$status"
    fi
done

echo
echo "[done] hierarchy NCU reports: $OUT_DIR/${TAG}_*.ncu-rep"
