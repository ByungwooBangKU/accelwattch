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
# (nvcc version, torch CUDA, toolkit-vs-bundle match) so the build doesn't
# die 5 minutes in with a cryptic message.
#
# Two-phase install:
#   1. Try a prebuilt binary wheel from NVIDIA's PyPI index (no nvcc needed).
#      Succeeds on most common torch versions and avoids the CUDA toolkit
#      quagmire entirely.
#   2. Fall back to a source build (requires CUDA ≥ 12.0 toolkit) if the
#      prebuilt wheel doesn't match this torch / python combination.
#
# Works for both A100 (TE falls back to FP16 TC — still useful to record)
# and H100 (native FP8 TC).
#
# Usage:
#   ./install_transformer_engine.sh                   # auto — prebuilt, then source
#   TE_NO_PREBUILT=1 ./install_transformer_engine.sh  # skip prebuilt attempt
#   TE_VERSION=2.13.0 ./install_transformer_engine.sh # pin version
#   PYTHON=python3.10 ./install_transformer_engine.sh # choose interpreter

set -euo pipefail

PYTHON="${PYTHON:-python3}"
TE_VERSION="${TE_VERSION:-}"        # empty → latest
EXTRA="${TE_EXTRA:-pytorch}"        # change to "pytorch,jax" if you need JAX too
TE_NO_PREBUILT="${TE_NO_PREBUILT:-0}"
# TE requires CUDA toolkit ≥ 12.0 for TE ≥ 1.x (and ≥ 12.1 for TE 2.x).
# We check against 12.0 as the minimum and print a stronger warning if
# the toolkit is below what the latest TE series needs.
TE_MIN_CUDA_MAJOR=12
TE_MIN_CUDA_MINOR=0

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
grn()   { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()   { printf '\033[33m%s\033[0m\n' "$*"; }

cuda_cmp() {   # cuda_cmp major minor >= wantM wantm  → exits 0 if LHS ≥ RHS
    local have_M=$1 have_m=$2 want_M=$3 want_m=$4
    if (( have_M > want_M )); then return 0; fi
    if (( have_M < want_M )); then return 1; fi
    (( have_m >= want_m ))
}

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

# ---- 2. torch with CUDA ----
# Do this BEFORE the nvcc check so we can compare the two CUDA versions
# explicitly and flag the common "torch wheel bundles new CUDA but system
# nvcc is old" trap that makes TE's source build fail cryptically.
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    red "error: torch not installed in this interpreter"
    echo "   → pip install torch  (match CUDA, e.g. --index-url https://download.pytorch.org/whl/cu121)"
    exit 1
fi
torch_ver=$("$PYTHON" -c "import torch; print(torch.__version__)")
torch_cuda=$("$PYTHON" -c "import torch; print(torch.version.cuda)")
torch_has_cuda=$("$PYTHON" -c "import torch; print(int(torch.cuda.is_available()))")
echo "torch  : $torch_ver  (torch.version.cuda=$torch_cuda, is_available=$torch_has_cuda)"
if [[ "$torch_has_cuda" != "1" ]]; then
    red "error: torch.cuda.is_available()=False — fix this before installing TE"
    exit 1
fi
torch_cuda_major=${torch_cuda%%.*}
torch_cuda_minor=${torch_cuda#*.}
torch_cuda_minor=${torch_cuda_minor%%.*}

# ---- 3. nvcc version parsing (only needed for source build) ----
have_nvcc=0
nvcc_major=0
nvcc_minor=0
nvcc_ver="(missing)"
if command -v nvcc >/dev/null; then
    have_nvcc=1
    # nvcc --version prints e.g.  "Cuda compilation tools, release 12.1, V12.1.105"
    nvcc_ver=$(nvcc --version | awk -F 'release ' '/release/ {print $2}' | awk -F ',' '{print $1}')
    nvcc_major=${nvcc_ver%%.*}
    nvcc_minor=${nvcc_ver#*.}
    nvcc_minor=${nvcc_minor%%.*}
    # Dereference the toolkit path so users see where it actually points.
    nvcc_path=$(command -v nvcc)
    cuda_home_effective=$(dirname "$(dirname "$(readlink -f "$nvcc_path")")")
    echo "nvcc   : $nvcc_path  (release $nvcc_ver  → $cuda_home_effective)"
else
    ylw "nvcc   : NOT FOUND on PATH"
    ylw "         → source-build path is unavailable; will only try prebuilt wheels."
fi

# Common pitfall diagnostic: if nvcc's CUDA is way behind torch's bundled
# CUDA, the source build will likely fail with a version-minimum check
# inside TE's setup.py (the exact error the user reported: 'Transformer
# Engine requires CUDA 12.0 or newer' even though torch.version.cuda=12.8).
if [[ $have_nvcc -eq 1 ]]; then
    if ! cuda_cmp "$nvcc_major" "$nvcc_minor" "$TE_MIN_CUDA_MAJOR" "$TE_MIN_CUDA_MINOR"; then
        red ""
        red "         !!!  SOURCE BUILD WILL FAIL  !!!"
        red ""
        red "  Your system toolkit ($nvcc_ver) is below the TE minimum ($TE_MIN_CUDA_MAJOR.$TE_MIN_CUDA_MINOR)."
        if [[ "$torch_cuda_major" -ge $TE_MIN_CUDA_MAJOR ]]; then
            red "  torch.version.cuda reports $torch_cuda because the torch wheel BUNDLES a newer"
            red "  CUDA runtime — but TE compiles from source against your system's nvcc, not"
            red "  torch's bundle. To build TE you need a system CUDA toolkit ≥ $TE_MIN_CUDA_MAJOR.$TE_MIN_CUDA_MINOR."
        fi
        echo
        echo "  Two ways out:"
        echo "   (a) [easy]  Use NVIDIA's prebuilt TE wheel — no nvcc needed:"
        echo "         $PYTHON -m pip install --no-build-isolation \\"
        echo "             --extra-index-url https://pypi.nvidia.com \\"
        echo "             'transformer-engine[$EXTRA]'"
        echo "       This script will try this path automatically below unless"
        echo "       TE_NO_PREBUILT=1 is set."
        echo
        echo "   (b) [involved]  Install a newer CUDA toolkit:"
        echo "         conda install -c nvidia cuda-toolkit=12.1"
        echo "       or system package: sudo apt install cuda-toolkit-12-1"
        echo "       then re-run this script."
        echo
    elif [[ "$nvcc_major" != "$torch_cuda_major" ]]; then
        ylw "  note: system nvcc=$nvcc_ver but torch.version.cuda=$torch_cuda —"
        ylw "        mismatched major versions can still build, but may produce"
        ylw "        binaries that fail to load at runtime. Prefer matching majors."
    fi
fi

# ---- 4. GPU capability note ----
if command -v nvidia-smi >/dev/null; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d ' ')
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "gpu    : $gpu  (compute_cap=$cc)"
    if [[ "${cc%%.*}" -lt 9 ]]; then
        ylw "note: this GPU is pre-Hopper — TE fp8_autocast falls back to FP16 TC."
        ylw "      Installing TE is still useful; the fp8_te variant runs under the"
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
echo

installed_ok=0

# 5a. Try NVIDIA's prebuilt wheel index first.  Wheels there are compiled
#     against specific (torch, cuda) combos — when a match exists, pip
#     picks it up automatically; when none matches, pip falls back to
#     the source tarball which will error out on our CUDA version check.
if [[ "$TE_NO_PREBUILT" != "1" ]]; then
    echo "[attempt 1/2] prebuilt wheel from https://pypi.nvidia.com …"
    if "$PYTHON" -m pip install --no-build-isolation \
            --extra-index-url https://pypi.nvidia.com \
            "$PKG"; then
        if "$PYTHON" -c "import transformer_engine.pytorch" 2>/dev/null; then
            installed_ok=1
            grn "[attempt 1/2] prebuilt wheel installed."
        else
            ylw "[attempt 1/2] prebuilt install succeeded but transformer_engine.pytorch"
            ylw "              still doesn't import — falling through to source build."
        fi
    else
        ylw "[attempt 1/2] prebuilt wheel path failed (no matching wheel, or install error)."
    fi
fi

# 5b. Source build fallback (only reached if prebuilt didn't work).
if [[ $installed_ok -ne 1 ]]; then
    if [[ $have_nvcc -ne 1 ]]; then
        red "cannot source-build without nvcc; see the two options above."
        exit 1
    fi
    if ! cuda_cmp "$nvcc_major" "$nvcc_minor" "$TE_MIN_CUDA_MAJOR" "$TE_MIN_CUDA_MINOR"; then
        red "system CUDA ($nvcc_ver) is below TE minimum ($TE_MIN_CUDA_MAJOR.$TE_MIN_CUDA_MINOR) —"
        red "source build would fail. Fix the toolkit or pass TE_NO_PREBUILT=0 to"
        red "retry the prebuilt wheel with a different (older) pinned TE_VERSION."
        exit 1
    fi
    echo
    echo "[attempt 2/2] source build (this compiles CUDA kernels — 5–10 minutes, several GB RAM)"
    echo
    set -x
    "$PYTHON" -m pip install --no-build-isolation "$PKG"
    set +x
    installed_ok=1
fi

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
    echo "     * torch ABI mismatch — force-reinstall against THIS torch:"
    echo "         $PYTHON -m pip install --force-reinstall --no-build-isolation '$PKG'"
    echo "     * missing CUDA libs on loader path — set LD_LIBRARY_PATH to CUDA libs"
    echo "     * cudnn-dev / nvcc too old for this TE version"
    exit 1
fi
