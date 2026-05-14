#!/usr/bin/env bash
# Root-level compatibility alias for the typo run_pjbit_cupsh.sh.
set -euo pipefail
cd "$(dirname "$0")/util/dram_util_experiment"
exec ./run_pjbit_cupsh.sh "$@"
