#!/usr/bin/env bash
# Launcher for the GPU power benchmark.
#
# Usage:
#   ./run_bench.sh                        # full sweep on GPU 0
#   ./run_bench.sh --quick                # shorter sweep
#   ./run_bench.sh --device 1 --tag h100  # multi-GPU, tagged output
#
# All extra args are forwarded to gpu_power_bench.py.

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# Pick a python with torch + pynvml installed. Override with $PYTHON=... if needed.
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null; then
    echo "error: $PYTHON not found" >&2
    exit 1
fi

# Preflight is run inside gpu_power_bench.py, but a dry-run check here gives
# a faster error if deps are missing.
"$PYTHON" -c "import torch, pynvml, nvtx, matplotlib, pandas" 2>/dev/null || {
    echo "missing python deps — install via: $PYTHON -m pip install -r requirements.txt"
    exit 1
}

# Recommended kernel-level tweaks. Silently skipped if we lack privileges.
if command -v nvidia-smi >/dev/null; then
    sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
fi

mkdir -p reports
exec "$PYTHON" gpu_power_bench.py "$@"
