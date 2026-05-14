#!/usr/bin/env bash
# Root-level launcher for Nsight Compute DRAM/L2 validation.
set -euo pipefail
cd "$(dirname "$0")/util/dram_util_experiment"
exec ./run_pjbit_ncu.sh "$@"
