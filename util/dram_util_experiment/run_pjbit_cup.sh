#!/usr/bin/env bash
# Compatibility alias for the common typo run_pjbit_cup.sh.
set -euo pipefail
cd "$(dirname "$0")"
exec ./run_pjbit_cupy.sh "$@"
