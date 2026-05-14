#!/usr/bin/env bash
# Root-level launcher for the DRAM pJ/bit experiment.
set -euo pipefail
cd "$(dirname "$0")/util/dram_util_experiment"
exec ./run_pjbit_cupy.sh "$@"
