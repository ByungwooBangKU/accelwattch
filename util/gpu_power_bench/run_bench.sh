#!/usr/bin/env bash
# Launcher for the GPU power benchmark.
#
# Usage:
#   ./run_bench.sh                             # full sweep on GPU 0 (default)
#   ./run_bench.sh --quick                     # shorter sweep on GPU 0
#   ./run_bench.sh --device 1 --tag h100       # single GPU, tagged output
#
# Multi-GPU (same-node, parallel — for variance analysis):
#   ./run_bench.sh --num-gpus 8                # run on GPUs 0..7 in parallel
#   ./run_bench.sh --devices "0,2,4,6" --tag h100
#   ./run_bench.sh --num-gpus 4 --sequential   # one at a time (cleaner thermal)
#
# All extra args (--window-ms, --llm-shapes, --suite, --cases,
# --include-fused, --attn-shape, --mlp-shape, …) are forwarded to
# gpu_power_bench.py.
#
# Log layout (multi-GPU AND single-GPU) :
#     reports/gpu_power_<tag>_<MMDD_hhmm>/         (= $RUN_DIR)
#         single.log                  (single-GPU run)
#         gpu_power_bench_*.csv       (single-GPU run, all CSVs + PNGs here)
#         *.png
#
#         gpu0/                       (multi-GPU run — one subdir per GPU)
#             gpu0.log
#             gpu_power_bench_*.csv   (CSVs + PNGs colocated with the log)
#             *.png
#         gpu1/
#             gpu1.log
#             ...
#
# Logs AND data are colocated per-GPU so post-mortem inspection only
# needs ONE directory per card. multi_gpu_analysis.py walks $RUN_DIR
# recursively to find all per-GPU CSVs.
# Logs are NEVER deleted on success/failure — you can `tail -f` while
# the run is in progress and inspect them after.
#
# Thermal note: parallel runs share the node's cooling budget and back-plane
# temperatures, so cross-GPU variance measured this way includes cooling
# asymmetry. Use --sequential for a tighter per-GPU thermal profile at the
# cost of N× wall time.

# NOTE: deliberately NOT using `set -u` — an empty forwarded-args array
# would trigger "unbound variable" on bash ≤ 4.3, silently killing every
# launched subshell before the per-GPU log redirect takes effect. That
# reproduced the "only 1 gpu log ever appears" symptom users hit.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

PYTHON="${PYTHON:-python3}"
# Force line-buffered Python stdout. Without this, the `tee` pipe below
# flips Python from line-buffered (terminal) to block-buffered (~8 KB)
# so the first ~30s of "import torch / nvmlInit / preflight / first cell"
# prints land in the buffer and only flush en-bloc when the first cell's
# E_total/P_avg lines push them past the threshold. Looks exactly like
# a hang. Setting PYTHONUNBUFFERED here makes every print() show up
# immediately under tee. Override RUN_BENCH_UNBUFFERED=0 if you have a
# specific reason to want buffered output (e.g. perf-sensitive CSV
# generation that you're piping through grep).
if [[ "${RUN_BENCH_UNBUFFERED:-1}" == "1" ]]; then
    export PYTHONUNBUFFERED=1
fi
# Seconds to wait between parallel launches. Staggering avoids a thundering
# herd on nvmlInit / torch.cuda.set_device that has been observed to drop
# 7/8 processes simultaneously on some driver versions. Override with
# RUN_BENCH_STAGGER=0 to disable.
STAGGER_S="${RUN_BENCH_STAGGER:-3}"

if ! command -v "$PYTHON" >/dev/null; then
    echo "error: $PYTHON not found" >&2
    exit 1
fi

"$PYTHON" -c "import torch, pynvml, nvtx, matplotlib, pandas" 2>/dev/null || {
    echo "missing python deps — install via: $PYTHON -m pip install -r requirements.txt"
    exit 1
}

mkdir -p reports

# ---- argv parsing (we only intercept our new flags; rest forwards) ----
NUM_GPUS=""
DEVICES=""
SEQUENTIAL=0
BASE_TAG=""
SINGLE_DEVICE="0"
SUDO_PSTATE=0
FORWARD=()

AUTO_ANALYZE=1   # default ON for single-GPU; multi-GPU paths disable it below.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)         NUM_GPUS="$2"; shift 2 ;;
        --num-gpus=*)       NUM_GPUS="${1#*=}"; shift ;;
        --devices)          DEVICES="$2"; shift 2 ;;
        --devices=*)        DEVICES="${1#*=}"; shift ;;
        --sequential)       SEQUENTIAL=1; shift ;;
        --no-auto-analyze)  AUTO_ANALYZE=0; shift ;;
        --sudo-pstate)      SUDO_PSTATE=1; FORWARD+=("$1"); shift ;;
        --tag)              BASE_TAG="$2"; FORWARD+=("--tag" "$2"); shift 2 ;;
        --tag=*)            BASE_TAG="${1#*=}"; FORWARD+=("$1"); shift ;;
        --device)           SINGLE_DEVICE="$2"; FORWARD+=("--device" "$2"); shift 2 ;;
        --device=*)         SINGLE_DEVICE="${1#*=}"; FORWARD+=("$1"); shift ;;
        *)                  FORWARD+=("$1"); shift ;;
    esac
done

# Build the device list:
#   1. --devices "0,2,4,6"  → exact list
#   2. --num-gpus N         → 0..N-1
#   3. (neither)            → single default run, forwards raw args
DEVS=()
if [[ -n "$DEVICES" ]]; then
    IFS=',' read -ra DEVS <<< "$DEVICES"
elif [[ -n "$NUM_GPUS" ]]; then
    for ((i=0; i<NUM_GPUS; i++)); do DEVS+=("$i"); done
fi

SUDO_KEEPALIVE_PID=""
cleanup_sudo_keepalive() {
    if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
        kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
        wait "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
    fi
}

start_sudo_keepalive() {
    command -v sudo >/dev/null || {
        echo "error: --sudo-pstate requested but sudo is not available" >&2
        exit 1
    }
    echo "[sudo-pstate] requesting sudo once for selected-GPU clock reset helpers"
    sudo -v || {
        echo "error: sudo authentication failed; rerun without --sudo-pstate or ask the admin" >&2
        exit 1
    }
    (
        while true; do
            sudo -n -v >/dev/null 2>&1 || exit
            sleep 60
        done
    ) &
    SUDO_KEEPALIVE_PID=$!
    trap cleanup_sudo_keepalive EXIT
}

if (( SUDO_PSTATE == 1 )); then
    start_sudo_keepalive
fi

enable_persistence_for_devices() {
    [[ "${RUN_BENCH_PERSISTENCE:-1}" == "1" ]] || return 0
    command -v nvidia-smi >/dev/null || return 0
    command -v sudo >/dev/null || return 0
    local dev
    for dev in "$@"; do
        # Keep this selected-device only. A device-less persistence-mode
        # command on a shared multi-GPU node can affect GPUs this run
        # does not own. Resolve the physical NVML index because
        # CUDA_VISIBLE_DEVICES may remap CUDA --device indices.
        local smi_dev
        smi_dev=$("$PYTHON" - "$dev" 2>/dev/null <<'PY'
import sys
import pynvml
from power_monitor import resolve_nvml_handle

dev = int(sys.argv[1])
pynvml.nvmlInit()
try:
    handle, _ = resolve_nvml_handle(dev)
    print(pynvml.nvmlDeviceGetIndex(handle))
finally:
    pynvml.nvmlShutdown()
PY
)
        if [[ -z "$smi_dev" ]]; then
            if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
                echo "[warn] could not map CUDA --device $dev to a physical nvidia-smi index;" >&2
                echo "       skip persistence setup to avoid touching the wrong GPU" >&2
                continue
            fi
            smi_dev="$dev"
        fi
        sudo -n nvidia-smi -i "$smi_dev" -pm 1 >/dev/null 2>&1 || true
    done
}

PERSIST_DEVS=()
if [[ ${#DEVS[@]} -gt 0 ]]; then
    PERSIST_DEVS=("${DEVS[@]}")
else
    PERSIST_DEVS=("$SINGLE_DEVICE")
fi
enable_persistence_for_devices "${PERSIST_DEVS[@]}"

# Strip any --tag the user passed — we re-add a per-GPU tag below.
STRIPPED=()
skip_next=0
for a in "${FORWARD[@]+"${FORWARD[@]}"}"; do
    if (( skip_next )); then skip_next=0; continue; fi
    case "$a" in
        --tag)    skip_next=1 ;;
        --tag=*)  ;;   # drop
        --device) skip_next=1 ;;
        --device=*) ;;
        *)        STRIPPED+=("$a") ;;
    esac
done

# ---- per-experiment log directory ------------------------------------
# Logs land in reports/gpu_power_<tag>_<MMDD_hhmm>/ so a series of runs
# (different tags / different days) doesn't pile up in one giant
# reports/logs/ folder. The MMDD_hhmm stamp is fixed at script start —
# all per-GPU logs from this run share it. Override RUN_DIR to pick
# your own location (e.g. an NFS mount with more space).
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
LOG_TAG="${BASE_TAG:-default}"
RUN_DIR="${RUN_DIR:-reports/gpu_power_${LOG_TAG}_${RUN_STAMP}}"
mkdir -p "$RUN_DIR"
echo "[run-dir] logs → $RUN_DIR/"

# ${var[@]+expand} is the "only expand if set" trick — produces an empty
# argv on unset/empty arrays without tripping set -u (which we don't use
# here anyway, but this also works under it). Use this everywhere we
# splice STRIPPED into a command line.
run_one() {
    local dev="$1"
    local tag_suffix="gpu${dev}"
    local full_tag
    if [[ -n "$BASE_TAG" ]]; then
        full_tag="${BASE_TAG}_${tag_suffix}"
    else
        full_tag="$tag_suffix"
    fi
    # Per-GPU subdir holds EVERYTHING for this card : log + CSVs + plots.
    # Both gpu_power_bench.py (--out-dir) and analyze.py (defaults to
    # same dir as CSV) write to this directory, so a single `ls` on
    # $subdir gives the operator everything from this card's run.
    local subdir="$RUN_DIR/gpu${dev}"
    mkdir -p "$subdir"
    local log="$subdir/gpu${dev}.log"
    # Create the log file BEFORE the subprocess runs so even a fast-dying
    # process leaves evidence. Record the command being executed and the
    # PID so ps / tail can find it.
    {
        echo "== run_bench.sh launch =="
        echo "dev        : $dev"
        echo "tag        : $full_tag"
        echo "out_dir    : $subdir"
        echo "stripped   : ${STRIPPED[*]+"${STRIPPED[*]}"}"
        echo "cmdline    : $PYTHON gpu_power_bench.py --device $dev --tag $full_tag --out-dir $subdir ${STRIPPED[*]+"${STRIPPED[*]}"}"
        echo "pid        : $$  (subshell)"
        echo "start_time : $(date --iso-8601=seconds 2>/dev/null || date)"
        echo "-- subprocess stdout/stderr below --"
    } > "$log"
    echo "[launch] GPU $dev  tag=$full_tag  out_dir=$subdir  log=$log"
    "$PYTHON" gpu_power_bench.py \
        --device "$dev" \
        --tag "$full_tag" \
        --out-dir "$subdir" \
        ${STRIPPED[@]+"${STRIPPED[@]}"} \
        >> "$log" 2>&1
}

# Fallback: no multi-GPU flag given → single-GPU sweep, then optionally
# chain analyze.py automatically. Use --no-auto-analyze to keep just the
# raw CSV (e.g. when the analyse step will be done elsewhere).
if [[ ${#DEVS[@]} -eq 0 ]]; then
    # Single-GPU sweep — same per-experiment dir convention as the
    # multi-GPU path. We tee through `single.log` so the user still
    # sees live progress AND the log is preserved for later analysis
    # (matches multi-GPU's gpu<N>.log convention). CSVs + PNGs land
    # in $RUN_DIR alongside the log via --out-dir.
    sweep_log="$RUN_DIR/single.log"
    "$PYTHON" gpu_power_bench.py \
        --out-dir "$RUN_DIR" \
        ${FORWARD[@]+"${FORWARD[@]}"} 2>&1 | tee "$sweep_log"
    sweep_rc=${PIPESTATUS[0]}
    if (( sweep_rc != 0 )); then
        echo "[error] gpu_power_bench.py exited with code $sweep_rc" >&2
        echo "[error] full log preserved at $sweep_log" >&2
        exit $sweep_rc
    fi
    if (( AUTO_ANALYZE == 1 )); then
        # gpu_power_bench.py writes a "[OUTPUT_CSV] <path>" line at the end
        # — pluck it out so we don't have to glob. Log is preserved either
        # way for later inspection.
        csv_line=$(grep -E '^\[OUTPUT_CSV\] ' "$sweep_log" | tail -n 1)
        if [[ -z "$csv_line" ]]; then
            echo "[warn] sweep finished but no [OUTPUT_CSV] marker found — skip auto-analyze" >&2
            echo "[info] log: $sweep_log"
            exit 0
        fi
        csv_path="${csv_line#\[OUTPUT_CSV\] }"
        echo
        echo "[auto-analyze] running analyze.py on $csv_path"
        echo "                (skip with --no-auto-analyze; or run later: python3 analyze.py <csv>)"
        echo "[info] sweep log preserved: $sweep_log"
        echo
        "$PYTHON" analyze.py "$csv_path"
        exit $?
    fi
    echo "[info] sweep log: $sweep_log"
    exit 0
fi

echo "[info] multi-GPU run: devices=${DEVS[*]}  mode=$([[ $SEQUENTIAL == 1 ]] && echo sequential || echo parallel)"
if (( SEQUENTIAL == 0 )) && (( STAGGER_S > 0 )) && (( ${#DEVS[@]} > 1 )); then
    echo "[info] staggering parallel launches by ${STAGGER_S}s each (override: RUN_BENCH_STAGGER=0)"
fi

# Remember which PID ran which GPU so we can report per-GPU failures
# and tail the matching log on error.
declare -A PID_DEV=()
declare -A PID_LOG=()

if (( SEQUENTIAL )); then
    sequential_rc=0
    for dev in "${DEVS[@]}"; do
        if ! run_one "$dev"; then
            echo "[fail] GPU $dev sweep exited non-zero" >&2
            sequential_rc=1
        fi
    done
    rc=$sequential_rc
else
    pids=()
    for dev in "${DEVS[@]}"; do
        run_one "$dev" &
        pid=$!
        pids+=("$pid")
        PID_DEV[$pid]="$dev"
        # Stagger so concurrent nvmlInit / torch.cuda.set_device don't
        # collide. Only between launches, not after the last one.
        if (( STAGGER_S > 0 )); then sleep "$STAGGER_S"; fi
    done
    rc=0
    failed_devs=()
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            :
        else
            exit_code=$?
            dev="${PID_DEV[$pid]}"
            echo "[fail] GPU $dev (pid $pid) exited with code $exit_code" >&2
            failed_devs+=("$dev")
            rc=1
        fi
    done
    if (( rc != 0 )); then
        echo "" >&2
        echo "[warn] ${#failed_devs[@]} / ${#DEVS[@]} GPU sweeps failed: ${failed_devs[*]}" >&2
        echo "[warn] tail of each failed log (last 30 lines):" >&2
        for dev in "${failed_devs[@]}"; do
            ftag="gpu${dev}"
            log="$RUN_DIR/${ftag}/${ftag}.log"
            echo "" >&2
            echo "────── $log ──────" >&2
            if [[ -f "$log" ]]; then
                tail -n 30 "$log" >&2
            else
                echo "(log file does not exist — subshell died before redirect took effect;"  >&2
                echo " try RUN_BENCH_STAGGER=10 ./run_bench.sh … to spread out launches,"  >&2
                echo " or --sequential to isolate the failing GPU.)" >&2
            fi
        done
    fi
fi

echo
if (( rc != 0 )); then
    echo "[warn] at least one GPU's sweep failed — see per-log tails above." >&2
    # Build an easy copy-paste retry line for just the failing devices.
    # `failed_devs` is only populated in the parallel branch; guard with
    # a default expansion so it's safe either way.
    fd=("${failed_devs[@]+"${failed_devs[@]}"}")
    if (( ${#fd[@]} > 0 )); then
        retry_csv="${fd[*]}"
        retry_csv="${retry_csv// /,}"
        retry_hint="./run_bench.sh --devices '${retry_csv}'"
        [[ -n "$BASE_TAG" ]] && retry_hint+=" --tag ${BASE_TAG}"
        if (( ${#STRIPPED[@]} > 0 )); then
            retry_hint+=" ${STRIPPED[*]}"
        fi
        echo "       Retry only the failing cards:" >&2
        echo "         $retry_hint" >&2
    else
        echo "       To retry only the failing cards, use --devices '<csv>'." >&2
    fi
else
    echo "[done] per-GPU CSVs written under $RUN_DIR/gpu*/"
    if (( AUTO_ANALYZE == 1 )); then
        # Step 1 : per-GPU analyze.py — generates this card's plots
        # next to its CSV (analyze.py defaults --out-dir to CSV parent).
        echo "[auto-analyze] running analyze.py for each per-GPU CSV"
        echo "                (skip with --no-auto-analyze)"
        for dev in "${DEVS[@]}"; do
            gpulog="$RUN_DIR/gpu${dev}/gpu${dev}.log"
            if [[ ! -f "$gpulog" ]]; then
                echo "[skip-analyze] gpu${dev}: log missing — sweep died early"
                continue
            fi
            csv_line=$(grep -E '^\[OUTPUT_CSV\] ' "$gpulog" | tail -n 1)
            if [[ -z "$csv_line" ]]; then
                echo "[skip-analyze] gpu${dev}: no [OUTPUT_CSV] marker (sweep failed?)"
                continue
            fi
            csv_path="${csv_line#\[OUTPUT_CSV\] }"
            echo "[auto-analyze] gpu${dev}: $csv_path"
            "$PYTHON" analyze.py "$csv_path" || echo "[warn] analyze.py failed for gpu${dev}"
        done
        # Step 2 : cross-GPU variance analysis on the run dir.
        echo
        echo "[auto-analyze] running multi_gpu_analysis.py on $RUN_DIR"
        if [[ -n "$BASE_TAG" ]]; then
            "$PYTHON" multi_gpu_analysis.py "$RUN_DIR" "${#DEVS[@]}" --tag "$BASE_TAG" || rc=$?
        else
            "$PYTHON" multi_gpu_analysis.py "$RUN_DIR" "${#DEVS[@]}" || rc=$?
        fi
    else
        echo "  next: python3 analyze.py $RUN_DIR/gpu<N>/gpu_power_bench_*.csv  # per-GPU plots"
        echo "  next: python3 multi_gpu_analysis.py $RUN_DIR ${#DEVS[@]}${BASE_TAG:+ --tag $BASE_TAG}"
    fi
fi
exit $rc
