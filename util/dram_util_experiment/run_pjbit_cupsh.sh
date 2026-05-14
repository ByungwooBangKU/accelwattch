#!/usr/bin/env bash
# Compatibility alias for the typo run_pjbit_cupsh.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec ./run_pjbit_cupy.sh "$@"
