#!/usr/bin/env bash
# DEPRECATED — kept as a thin alias to `run_bench.sh --suite soc`.
#
# As of the SoC-merge PR, the SoC envelope (static / max / leakage) is a
# regular test case in gpu_power_bench.py. Run it via :
#
#     ./run_bench.sh --suite soc --device 0 --tag h100        # SoC only
#     ./run_bench.sh --cases soc --device 0 --tag h100        # equivalent
#     ./run_bench.sh --suite all --device 0 --tag h100        # full + SoC
#     ./run_bench.sh --num-gpus 8 --suite soc --tag h100      # multi-GPU
#
# Argument forwarding : this script translates the legacy --no-leakage
# and --leakage-* flags it used to accept into the new --soc-* names so
# old shell scripts keep working. Anything else is passed through.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

echo "[deprecated] run_soc_bench.sh — forwarding to ./run_bench.sh --suite soc"
echo "             new path: ./run_bench.sh --suite soc [--device N] [--tag X]"
echo

# Translate the few flag names that drifted between the two scripts.
# Legacy form on the LHS, new form on the RHS.
forwarded=()
skip_next=0
for arg in "$@"; do
    if (( skip_next )); then forwarded+=("$arg"); skip_next=0; continue; fi
    case "$arg" in
        --no-max)            forwarded+=("--soc-max-seconds" "0") ;;
        --no-leakage)        forwarded+=("--soc-leakage-cycles" "0") ;;
        --static-seconds)    forwarded+=("--soc-static-seconds");    skip_next=1 ;;
        --static-seconds=*)  forwarded+=("--soc-static-seconds=${arg#*=}") ;;
        --max-seconds)       forwarded+=("--soc-max-seconds");       skip_next=1 ;;
        --max-seconds=*)     forwarded+=("--soc-max-seconds=${arg#*=}") ;;
        --leakage-cycles)    forwarded+=("--soc-leakage-cycles");    skip_next=1 ;;
        --leakage-cycles=*)  forwarded+=("--soc-leakage-cycles=${arg#*=}") ;;
        --leakage-stress-s)  forwarded+=("--soc-leakage-stress-s");  skip_next=1 ;;
        --leakage-stress-s=*) forwarded+=("--soc-leakage-stress-s=${arg#*=}") ;;
        --leakage-decay-s)   forwarded+=("--soc-leakage-decay-s");   skip_next=1 ;;
        --leakage-decay-s=*) forwarded+=("--soc-leakage-decay-s=${arg#*=}") ;;
        --leak-window-s)     forwarded+=("--soc-leak-window-s");     skip_next=1 ;;
        --leak-window-s=*)   forwarded+=("--soc-leak-window-s=${arg#*=}") ;;
        --matmul-K)          forwarded+=("--soc-matmul-K");          skip_next=1 ;;
        --matmul-K=*)        forwarded+=("--soc-matmul-K=${arg#*=}") ;;
        --dtype)             forwarded+=("--soc-dtype");             skip_next=1 ;;
        --dtype=*)           forwarded+=("--soc-dtype=${arg#*=}") ;;
        --mode)              forwarded+=("--soc-mode");              skip_next=1 ;;
        --mode=*)            forwarded+=("--soc-mode=${arg#*=}") ;;
        *)                   forwarded+=("$arg") ;;
    esac
done

exec ./run_bench.sh --suite soc "${forwarded[@]+"${forwarded[@]}"}"
