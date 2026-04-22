#!/usr/bin/env bash
# Install Transformer Engine correctly for the gpu_power_bench fp8_te variant.
#
# The bare `transformer-engine` PyPI package is a META package — installing
# it alone errors at import / runtime:
#
#   RuntimeError: Found empty `transformer-engine` meta package installed.
#   Install `transformer-engine` with framework extensions via
#   'pip3 install --no-build-isolation transformer-engine[pytorch,jax]==VERSION'
#
# This script runs the correct command, after checking the prerequisites
# (nvcc, torch with CUDA) so the build doesn't fail halfway through.
#
# Works for both A100 (TE falls back to FP16 TC — still useful to record)
# and H100 (native FP8 TC).
#
# Usage:
#   ./install_transformer_engine.sh                # install latest
#   TE_VERSION=1.11.0 ./install_transformer_engine.sh   # pin version
#   PYTHON=python3.10 ./install_transformer_engine.sh   # choose interpreter

set -euo pipefail

PYTHON="${PYTHON:-python3}"
TE_VERSION="${TE_VERSION:-}"     # empty → latest
EXTRA="${TE_EXTRA:-pytorch}"     # change to "pytorch,jax" if you need JAX too

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
grn()   { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()   { printf '\033[33m%s\033[0m\n' "$*"; }

echo "== Transformer Engine install helper for gpu_power_bench =="
echo

# ---- 1. python ----
if ! command -v "$PYTHON" >/dev/null; then
    red "error: $PYTHON not found on PATH"
    echo "   → set PYTHON=... e.g. PYTHON=python3.10 $0"
    exit 1
fi
py_ver=$("$PYTHON" -c 'import sys; print(".".join(str(v) for v in sys.version_info[:2]))')
echo "python : $PYTHON  ($py_ver)"

# ---- 2. nvcc ----
if ! command -v nvcc >/dev/null; then
    red "error: nvcc not found on PATH — TE compiles CUDA kernels from source."
    echo "   Install CUDA toolkit, e.g.:"
    echo "     Ubuntu: sudo apt install cuda-toolkit-12-1"
    echo "     conda:  conda install -c nvidia cuda-toolkit=12.1"
    echo "   and make sure \$CUDA_HOME is set and \$PATH includes \$CUDA_HOME/bin."
    exit 1
fi
nvcc_ver=$(nvcc --version | awk '/release/ {print $6}' | tr -d ,)
echo "nvcc   : $(command -v nvcc)  ($nvcc_ver)"

# ---- 3. torch with CUDA ----
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    red "error: torch not installed in this interpreter"
    echo "   → pip install torch  (match CUDA, e.g. --index-url https://download.pytorch.org/whl/cu121)"
    exit 1
fi
torch_ver=$("$PYTHON" -c "import torch; print(torch.__version__)")
torch_cuda=$("$PYTHON" -c "import torch; print(torch.version.cuda)")
torch_has_cuda=$("$PYTHON" -c "import torch; print(int(torch.cuda.is_available()))")
echo "torch  : $torch_ver  (cuda=$torch_cuda, is_available=$torch_has_cuda)"
if [[ "$torch_has_cuda" != "1" ]]; then
    red "error: torch.cuda.is_available()=False — fix this before installing TE"
    exit 1
fi

# ---- 4. GPU capability note (informational, doesn't block install) ----
if command -v nvidia-smi >/dev/null; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d ' ')
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "gpu    : $gpu  (compute_cap=$cc)"
    if [[ "${cc%%.*}" -lt 9 ]]; then
        ylw "note: this GPU is pre-Hopper — TE fp8_autocast falls back to FP16 TC."
        ylw "      Installing TE is still useful: the fp8_te variant runs under the"
        ylw "      same code path, tagged in its notes column as 'TE fallback'."
    fi
fi

# ---- 5. install TE ----
if [[ -n "$TE_VERSION" ]]; then
    PKG="transformer-engine[$EXTRA]==$TE_VERSION"
else
    PKG="transformer-engine[$EXTRA]"
fi
echo
echo "installing : $PKG"
echo "this compiles CUDA kernels — expect 5–10 minutes, several GB RAM used."
echo

# --no-build-isolation is REQUIRED so TE's setup.py sees the installed
# torch + nvcc and links against them.
set -x
"$PYTHON" -m pip install --no-build-isolation "$PKG"
set +x

# ---- 6. verify ----
# Not enough to check `import transformer_engine.pytorch` — the torch backend
# shared library is loaded lazily on the first te.Linear() call.  We actually
# construct a tiny Linear and run it under fp8_autocast so a missing / ABI-
# mismatched .so surfaces here instead of mid-benchmark.
echo
"$PYTHON" - <<'PY'
import sys, traceback
try:
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    print("  te version:", getattr(te, "__version__", "unknown"))
    linear = te.Linear(64, 64, bias=False, params_dtype=torch.float16)
    if torch.cuda.is_available():
        linear = linear.cuda()
        x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        r = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.E4M3,
                                     amax_history_len=4, amax_compute_algo="max")
        with te.fp8_autocast(enabled=True, fp8_recipe=r):
            linear(x)
        print("  fp8_autocast forward pass: OK")
    else:
        print("  (no CUDA device visible — skipped fp8_autocast probe)")
except Exception as e:
    print("  VERIFY FAILED:", type(e).__name__, e)
    traceback.print_exc()
    sys.exit(1)
PY
rc=$?
if [[ $rc -eq 0 ]]; then
    grn "OK — transformer_engine installed and the torch backend .so loads."
    echo "   Run preflight again: $PYTHON preflight.py"
else
    red "install finished but the te.Linear + fp8_autocast runtime probe failed."
    echo "   Common causes:"
    echo "     * torch version mismatch — rebuild TE against THIS torch:"
    echo "         $PYTHON -m pip install --force-reinstall --no-build-isolation '$PKG'"
    echo "     * missing CUDA libs on loader path — set LD_LIBRARY_PATH to CUDA libs"
    echo "     * cudnn-dev / nvcc too old for the TE version"
    exit 1
fi
