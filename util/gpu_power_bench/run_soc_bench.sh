#!/usr/bin/env bash
# Launcher for the SoC power-envelope bench (static / max / leakage).
#
# Usage:
#   ./run_soc_bench.sh                           # default: GPU 0, fp16/tc, K=16384
#   ./run_soc_bench.sh --device 3 --tag blackwell
#   ./run_soc_bench.sh --device 0 --dtype bf16 --matmul-K 12288
#   ./run_soc_bench.sh --no-leakage              # skip the 5-cycle leakage phase
#
# Estimated wall time with defaults: ~10 min (60s static + 60s max +
# 5x(20s+30s) leakage + cooldown gaps). Override --static-seconds /
# --max-seconds / --leakage-cycles to shrink.
#
# All extra args are forwarded to soc_power_bench.py.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null; then
    echo "error: $PYTHON not found" >&2
    exit 1
fi

"$PYTHON" -c "import torch, pynvml, matplotlib" 2>/dev/null || {
    echo "missing python deps — install via: $PYTHON -m pip install -r requirements.txt"
    exit 1
}

if command -v nvidia-smi >/dev/null; then
    sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
fi

mkdir -p reports

exec "$PYTHON" soc_power_bench.py "$@"
