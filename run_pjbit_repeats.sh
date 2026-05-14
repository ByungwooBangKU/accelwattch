#!/usr/bin/env bash
# Root-level launcher for repeated DRAM pJ/bit experiments.
set -euo pipefail
cd "$(dirname "$0")/util/dram_util_experiment"
exec ./run_pjbit_repeats.sh "$@"
