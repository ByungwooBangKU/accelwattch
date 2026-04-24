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
# All extra args (--window-ms, --llm-shapes, …) are forwarded to gpu_power_bench.py.
# With multi-GPU, each process gets its own --tag suffix "_gpu<N>" and its own
# log file under reports/logs/ so you can `tail -f reports/logs/gpu0.log`.
#
# Thermal note: parallel runs share the node's cooling budget and back-plane
# temperatures, so cross-GPU variance measured this way includes cooling
# asymmetry. Use --sequential for a tighter per-GPU thermal profile at the
# cost of N× wall time.

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null; then
    echo "error: $PYTHON not found" >&2
    exit 1
fi

"$PYTHON" -c "import torch, pynvml, nvtx, matplotlib, pandas" 2>/dev/null || {
    echo "missing python deps — install via: $PYTHON -m pip install -r requirements.txt"
    exit 1
}

if command -v nvidia-smi >/dev/null; then
    sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
fi

mkdir -p reports reports/logs

# ---- argv parsing (we only intercept our new flags; rest forwards) ----
NUM_GPUS=""
DEVICES=""
SEQUENTIAL=0
BASE_TAG=""
FORWARD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)         NUM_GPUS="$2"; shift 2 ;;
        --num-gpus=*)       NUM_GPUS="${1#*=}"; shift ;;
        --devices)          DEVICES="$2"; shift 2 ;;
        --devices=*)        DEVICES="${1#*=}"; shift ;;
        --sequential)       SEQUENTIAL=1; shift ;;
        --tag)              BASE_TAG="$2"; FORWARD+=("--tag" "$2"); shift 2 ;;
        --tag=*)            BASE_TAG="${1#*=}"; FORWARD+=("$1"); shift ;;
        # --device is a single-GPU flag we leave as-is when no multi-GPU
        # option is given (preserves existing single-GPU behaviour).
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

# Strip any --tag the user passed — we re-add a per-GPU tag below.
STRIPPED=()
skip_next=0
for a in "${FORWARD[@]}"; do
    if (( skip_next )); then skip_next=0; continue; fi
    case "$a" in
        --tag)    skip_next=1 ;;
        --tag=*)  ;;   # drop
        *)        STRIPPED+=("$a") ;;
    esac
done

run_one() {
    local dev="$1"
    local tag_suffix="gpu${dev}"
    local full_tag
    if [[ -n "$BASE_TAG" ]]; then
        full_tag="${BASE_TAG}_${tag_suffix}"
    else
        full_tag="$tag_suffix"
    fi
    local log="reports/logs/${full_tag}.log"
    echo "[launch] GPU $dev  tag=$full_tag  log=$log"
    "$PYTHON" gpu_power_bench.py \
        --device "$dev" \
        --tag "$full_tag" \
        "${STRIPPED[@]}" \
        > "$log" 2>&1
}

# Fallback: no multi-GPU flag given → original single-GPU behaviour.
if [[ ${#DEVS[@]} -eq 0 ]]; then
    # Pass the user's original FORWARD (which still contains their --tag).
    exec "$PYTHON" gpu_power_bench.py "${FORWARD[@]}"
fi

echo "[info] multi-GPU run: devices=${DEVS[*]}  mode=$([[ $SEQUENTIAL == 1 ]] && echo sequential || echo parallel)"

if (( SEQUENTIAL )); then
    for dev in "${DEVS[@]}"; do
        run_one "$dev"
    done
else
    pids=()
    for dev in "${DEVS[@]}"; do
        run_one "$dev" &
        pids+=($!)
    done
    # Wait for all; propagate failure but don't short-circuit the others.
    rc=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then rc=1; fi
    done
    if (( rc != 0 )); then
        echo "[warn] at least one GPU's sweep failed — see reports/logs/" >&2
    fi
fi

echo
echo "[done] per-GPU CSVs written under reports/. Next:"
echo "  python3 multi_gpu_analysis.py reports/ ${#DEVS[@]}${BASE_TAG:+ --tag $BASE_TAG}"
