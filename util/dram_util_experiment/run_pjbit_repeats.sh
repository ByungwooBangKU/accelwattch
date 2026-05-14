#!/usr/bin/env bash
# Run repeated DRAM pJ/bit experiments and generate repeat summary CSV/PNG.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./run_pjbit_repeats.sh --profile rtx3090 --tag rtx3090_integrated_patterns
  ./run_pjbit_repeats.sh --profile a100-8gib --device 0 --tag a100_8gib_patterns
  ./run_pjbit_repeats.sh --profile h100-8gib --device 0 --tag h100_8gib_patterns

Options:
  --profile NAME          auto, rtx3090, a100, a100-8gib, h100, h100-8gib
  --device N             CUDA/NVML GPU index. Default: 0
  --tag TAG              Base tag. Each run appends _rep1/_rep2/...
  --repeats N            Number of repeats. Default: 3
  --targets "..."        Override target list. Quote the list.
  --write-patterns "..." Override write pattern list. Quote the list.
  --buf-bytes N          Override buffer bytes per mode.
  --phase-seconds N      Default: 20
  --idle-seconds N       Default: 15
  --window-ms N          Default: 200
  --poll-hz N            Default: 100
  --gap-seconds N        Default: 1.0
  --phase-order NAME     target-major or workload-major. Default: target-major
  --ncu-profile          Run separate Nsight Compute DRAM/L2 validation after repeats
  --ncu-only             Skip NVML repeats and run only Nsight Compute validation
  --ncu-bin PATH         Nsight Compute CLI. Default: ncu
  --ncu-set NAME         NCU metric set when --ncu-metrics is empty. Default: full
  --ncu-metrics CSV      Explicit NCU metric list. Overrides --ncu-set.
  --ncu-phase-seconds N  NCU validation phase length. Default: 1
  --ncu-buf-bytes N      NCU validation buffer bytes. Default: same as --buf-bytes
  --ncu-launch-skip N    Kernel launches to skip before profiling. Default: 2
  --ncu-launch-count N   Kernel launches to profile. Default: 1
  --out-dir DIR          Default: reports
  --                    Extra args passed to run_pjbit_cupy.sh
EOF
}

PROFILE="auto"
DEVICE="0"
TAG=""
REPEATS="3"
TARGETS_STR=""
WRITE_PATTERNS_STR="zero const address random toggle"
BUF_BYTES=""
PHASE_SECONDS="20"
IDLE_SECONDS="15"
WINDOW_MS="200"
POLL_HZ="100"
GAP_SECONDS="1.0"
PHASE_ORDER="target-major"
NCU_PROFILE="0"
NCU_ONLY="0"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="full"
NCU_METRICS=""
NCU_PHASE_SECONDS="1"
NCU_BUF_BYTES=""
NCU_LAUNCH_SKIP="2"
NCU_LAUNCH_COUNT="1"
OUT_DIR="reports"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --targets) TARGETS_STR="$2"; shift 2 ;;
        --write-patterns) WRITE_PATTERNS_STR="$2"; shift 2 ;;
        --buf-bytes) BUF_BYTES="$2"; shift 2 ;;
        --phase-seconds) PHASE_SECONDS="$2"; shift 2 ;;
        --idle-seconds) IDLE_SECONDS="$2"; shift 2 ;;
        --window-ms) WINDOW_MS="$2"; shift 2 ;;
        --poll-hz) POLL_HZ="$2"; shift 2 ;;
        --gap-seconds) GAP_SECONDS="$2"; shift 2 ;;
        --phase-order) PHASE_ORDER="$2"; shift 2 ;;
        --ncu-profile) NCU_PROFILE="1"; shift ;;
        --ncu-only) NCU_ONLY="1"; NCU_PROFILE="1"; shift ;;
        --ncu-bin) NCU_BIN="$2"; shift 2 ;;
        --ncu-set) NCU_SET="$2"; shift 2 ;;
        --ncu-metrics) NCU_METRICS="$2"; shift 2 ;;
        --ncu-phase-seconds) NCU_PHASE_SECONDS="$2"; shift 2 ;;
        --ncu-buf-bytes) NCU_BUF_BYTES="$2"; shift 2 ;;
        --ncu-launch-skip) NCU_LAUNCH_SKIP="$2"; shift 2 ;;
        --ncu-launch-count) NCU_LAUNCH_COUNT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "[err] unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$PROFILE" == "auto" ]]; then
    GPU_NAME="$(nvidia-smi --id="$DEVICE" --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if [[ "$GPU_NAME" == *"H100"* ]]; then
        PROFILE="h100-8gib"
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        PROFILE="a100-8gib"
    else
        PROFILE="rtx3090"
    fi
    echo "[info] auto profile selected: $PROFILE"
fi

case "$PROFILE" in
    rtx3090)
        DEFAULT_TAG="rtx3090_integrated_patterns"
        DEFAULT_TARGETS="0 25 50 75 100"
        DEFAULT_BUF_BYTES="1073741824"
        ;;
    a100)
        DEFAULT_TAG="a100_patterns"
        DEFAULT_TARGETS="0 50 75 100"
        DEFAULT_BUF_BYTES=""
        ;;
    a100-8gib)
        DEFAULT_TAG="a100_8gib_patterns"
        DEFAULT_TARGETS="0 50 75 100"
        DEFAULT_BUF_BYTES="8589934592"
        ;;
    h100)
        DEFAULT_TAG="h100_patterns"
        DEFAULT_TARGETS="0 50 75 100"
        DEFAULT_BUF_BYTES=""
        ;;
    h100-8gib)
        DEFAULT_TAG="h100_8gib_patterns"
        DEFAULT_TARGETS="0 50 75 100"
        DEFAULT_BUF_BYTES="8589934592"
        ;;
    *)
        echo "[err] unknown profile: $PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

TAG="${TAG:-$DEFAULT_TAG}"
TARGETS_STR="${TARGETS_STR:-$DEFAULT_TARGETS}"
BUF_BYTES="${BUF_BYTES:-$DEFAULT_BUF_BYTES}"

read -r -a TARGETS <<< "$TARGETS_STR"
read -r -a WRITE_PATTERNS <<< "$WRITE_PATTERNS_STR"

PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in \
        /home/bang001/miniforge3/envs/ssc21env/bin/python \
        "$(command -v python3 || true)"; do
        if [[ -x "$cand" ]] && "$cand" -c "import matplotlib" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "[err] Python with matplotlib is required for repeat summary plotting" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

CUPY_RUNNER="$SCRIPT_DIR/run_pjbit_cupy.sh"
if [[ ! -x "$CUPY_RUNNER" ]]; then
    echo "[err] missing executable: $CUPY_RUNNER" >&2
    echo "      Check that util/dram_util_experiment/run_pjbit_cupy.sh exists after git pull." >&2
    exit 1
fi
NCU_RUNNER="$SCRIPT_DIR/run_pjbit_ncu.sh"
if [[ "$NCU_PROFILE" == "1" && ! -x "$NCU_RUNNER" ]]; then
    echo "[err] missing executable: $NCU_RUNNER" >&2
    echo "      Check that util/dram_util_experiment/run_pjbit_ncu.sh exists after git pull." >&2
    exit 1
fi

echo "[info] profile=$PROFILE device=$DEVICE repeats=$REPEATS tag=$TAG"
echo "[info] targets=${TARGETS[*]}"
echo "[info] write-patterns=${WRITE_PATTERNS[*]}"
echo "[info] phase-seconds=$PHASE_SECONDS idle-seconds=$IDLE_SECONDS window-ms=$WINDOW_MS poll-hz=$POLL_HZ gap-seconds=$GAP_SECONDS phase-order=$PHASE_ORDER"
if [[ "$NCU_PROFILE" == "1" ]]; then
    echo "[info] ncu-profile=on ncu-only=$NCU_ONLY ncu-bin=$NCU_BIN ncu-set=$NCU_SET ncu-phase-seconds=$NCU_PHASE_SECONDS"
fi
if [[ -n "$BUF_BYTES" ]]; then
    echo "[info] buf-bytes=$BUF_BYTES"
else
    echo "[info] buf-bytes=auto max(1 GiB, 64 x L2)"
fi

if [[ "$NCU_ONLY" != "1" ]]; then
    for rep in $(seq 1 "$REPEATS"); do
        REP_TAG="${TAG}_rep${rep}"
        cmd=(
            "$CUPY_RUNNER"
            --device "$DEVICE"
            --modes read write
            --write-patterns "${WRITE_PATTERNS[@]}"
            --targets "${TARGETS[@]}"
            --phase-seconds "$PHASE_SECONDS"
            --idle-seconds "$IDLE_SECONDS"
            --poll-hz "$POLL_HZ"
            --gap-seconds "$GAP_SECONDS"
            --phase-order "$PHASE_ORDER"
            --window-ms "$WINDOW_MS"
            --out-dir "$OUT_DIR"
            --tag "$REP_TAG"
        )
        if [[ -n "$BUF_BYTES" ]]; then
            cmd+=(--buf-bytes "$BUF_BYTES")
        fi
        cmd+=("${EXTRA_ARGS[@]}")

        echo
        echo "[run] repeat $rep/$REPEATS tag=$REP_TAG"
        "${cmd[@]}"
    done

    SUMMARY_CSV="$OUT_DIR/${TAG}_repeat_summary.csv"
    SUMMARY_PNG="$OUT_DIR/${TAG}_repeat_summary.png"

    echo
    echo "[summarize] $TAG"
    "$PY" summarize_pjbit_repeats.py \
        "$OUT_DIR/*${TAG}_rep*_analysis.csv" \
        --out "$SUMMARY_CSV" \
        --plot-out "$SUMMARY_PNG"

    echo
    echo "[done] repeat summary: $SUMMARY_CSV"
    echo "[done] repeat summary plot: $SUMMARY_PNG"
    echo "[done] per-run images: $OUT_DIR/*${TAG}_rep*.png"
fi

if [[ "$NCU_PROFILE" == "1" ]]; then
    NCU_EFFECTIVE_BUF_BYTES="${NCU_BUF_BYTES:-$BUF_BYTES}"
    ncu_cmd=(
        "$NCU_RUNNER"
        --device "$DEVICE"
        --tag "${TAG}_ncu"
        --modes "read write"
        --write-patterns "$WRITE_PATTERNS_STR"
        --phase-seconds "$NCU_PHASE_SECONDS"
        --out-dir "$OUT_DIR"
        --ncu-bin "$NCU_BIN"
        --ncu-set "$NCU_SET"
        --launch-skip "$NCU_LAUNCH_SKIP"
        --launch-count "$NCU_LAUNCH_COUNT"
        --window-ms "$WINDOW_MS"
    )
    if [[ -n "$NCU_METRICS" ]]; then
        ncu_cmd+=(--ncu-metrics "$NCU_METRICS")
    fi
    if [[ -n "$NCU_EFFECTIVE_BUF_BYTES" ]]; then
        ncu_cmd+=(--buf-bytes "$NCU_EFFECTIVE_BUF_BYTES")
    fi

    echo
    echo "[ncu] DRAM/L2 counter validation"
    "${ncu_cmd[@]}"
fi
