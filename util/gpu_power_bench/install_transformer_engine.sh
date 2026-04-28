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

# ---- 1.5  venv detection -----------------------------------------------
# A common cause of mystery errors is installing TE into the system
# python while torch/cuda live in a venv (or vice-versa). Refuse to
# proceed unless we can confirm which interpreter is in use, and warn
# loudly when it isn't a venv. The user can override with TE_ALLOW_NO_VENV=1.
in_venv=$("$PYTHON" -c 'import sys; print(int(sys.prefix != sys.base_prefix))')
py_prefix=$("$PYTHON" -c 'import sys; print(sys.prefix)')
py_site=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [[ "$in_venv" == "1" ]]; then
    grn "venv   : ACTIVE  $py_prefix"
    echo "         site-packages: $py_site"
else
    ylw "venv   : NOT a venv  ($py_prefix)"
    if [[ "${TE_ALLOW_NO_VENV:-0}" != "1" ]]; then
        red ""
        red "  Refusing to install into the system Python."
        red "  Activate your venv first :"
        red "      source path/to/venv/bin/activate"
        red "      ./install_transformer_engine.sh"
        red ""
        red "  Or override (NOT recommended) :"
        red "      TE_ALLOW_NO_VENV=1 ./install_transformer_engine.sh"
        exit 1
    fi
    ylw "         TE_ALLOW_NO_VENV=1 set — proceeding into system Python"
fi

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

# ---- 3.5  Detect nvidia-* pip wheels in the venv -----------------------
# torch on cu13 ships sibling pip packages : nvidia-cublas-cu13,
# nvidia-cudnn-cu13, etc. Their `.so` files live under
#   $venv/lib/python3.X/site-packages/nvidia/{cublas,cudnn}/lib/
# When the venv has a NEWER patch of these than the system /usr/local/cuda
# toolkit (a frequent cu13 case — pip ships 13.1.x while the toolkit is
# still 13.0.x), TE built against system headers fails at runtime with
# `undefined symbol: cublasLt*_internal, version libcublasLt.so.13`
# because the older system .so gets loaded first.
#
# The fix is to point BOTH the build and the runtime loader at the venv
# wheels, so they see the same .so. We collect the paths here and apply
# them in the build step + write them into te_env.sh at the end.
PIP_NV_LIB_PATHS=()
PIP_NV_INC_PATHS=()

probe_pip_nv() {
    local pkg="$1"          # e.g. nvidia.cublas
    local pip_name="$2"     # e.g. nvidia-cublas-cu13
    local pkg_dir
    if ! pkg_dir=$("$PYTHON" -c "import $pkg, os; print(os.path.dirname($pkg.__file__))" 2>/dev/null); then
        echo "  $pip_name : NOT installed in this venv"
        return 1
    fi
    local ver
    ver=$("$PYTHON" -m pip show "$pip_name" 2>/dev/null | awk '/^Version:/ {print $2}')
    echo "  $pip_name : $ver  →  $pkg_dir"
    [[ -d "$pkg_dir/lib"     ]] && PIP_NV_LIB_PATHS+=("$pkg_dir/lib")
    [[ -d "$pkg_dir/include" ]] && PIP_NV_INC_PATHS+=("$pkg_dir/include")
    return 0
}

echo
echo "venv-managed CUDA libraries (pip wheels) :"
probe_pip_nv "nvidia.cublas" "nvidia-cublas-cu${torch_cuda_major}" || true
probe_pip_nv "nvidia.cudnn"  "nvidia-cudnn-cu${torch_cuda_major}"  || true
probe_pip_nv "nvidia.cusparse" "nvidia-cusparse-cu${torch_cuda_major}" || true
probe_pip_nv "nvidia.cusolver" "nvidia-cusolver-cu${torch_cuda_major}" || true
# Some torch wheels also bundle nvidia-nccl, but it doesn't intersect TE.

# Compare a key cuBLAS symbol availability between system and venv .so
# so the user sees WHY we're shadowing system libs.
sys_cublas="/usr/local/cuda-${nvcc_major}.${nvcc_minor}/targets/x86_64-linux/lib/libcublasLt.so.${nvcc_major}"
[[ -f "$sys_cublas" ]] || sys_cublas="/usr/local/cuda-${nvcc_major}.${nvcc_minor}/lib64/libcublasLt.so.${nvcc_major}"
venv_cublas=""
if [[ ${#PIP_NV_LIB_PATHS[@]} -gt 0 ]]; then
    for d in "${PIP_NV_LIB_PATHS[@]}"; do
        if [[ -f "$d/libcublasLt.so.${nvcc_major}" ]]; then
            venv_cublas="$d/libcublasLt.so.${nvcc_major}"
            break
        fi
    done
fi
if [[ -f "$sys_cublas" && -n "$venv_cublas" ]]; then
    sym="cublasLtGroupedMatrixLayoutInit"
    sys_has=$(nm -D "$sys_cublas"  2>/dev/null | grep -c "$sym" || true)
    venv_has=$(nm -D "$venv_cublas" 2>/dev/null | grep -c "$sym" || true)
    echo "  cuBLASLt $sym symbol :"
    echo "      system ($sys_cublas) : $sys_has occurrences"
    echo "      venv   ($venv_cublas) : $venv_has occurrences"
    if (( venv_has > 0 && sys_has == 0 )); then
        ylw "  → venv cuBLASLt is NEWER than system. Will prefer venv at"
        ylw "    build & runtime to avoid 'undefined symbol' at import time."
    elif (( sys_has > 0 && venv_has == 0 )); then
        ylw "  → system cuBLASLt is newer than venv. May cause runtime issues."
        ylw "    Consider: $PYTHON -m pip install --upgrade nvidia-cublas-cu${torch_cuda_major}"
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

# ---- 4.5  Wire the venv-managed libs into build & runtime env --------
# Build-time : LIBRARY_PATH (linker) + CPATH (include search) so the
#              source build of TE links against THE SAME .so the runtime
#              loader will find.
# Runtime    : LD_LIBRARY_PATH so dlopen prefers venv .so over system.
#
# We PREPEND, not append — the system /usr/local/cuda paths are normally
# either already on the loader path or picked up by torch's own RPATH,
# and we want venv to win.
if [[ ${#PIP_NV_LIB_PATHS[@]} -gt 0 ]]; then
    venv_lib_csv=$(IFS=:; echo "${PIP_NV_LIB_PATHS[*]}")
    export LD_LIBRARY_PATH="$venv_lib_csv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$venv_lib_csv${LIBRARY_PATH:+:$LIBRARY_PATH}"
    echo
    echo "[env] LD_LIBRARY_PATH prepended with venv pip-wheel libs :"
    for d in "${PIP_NV_LIB_PATHS[@]}"; do echo "        $d"; done
fi
if [[ ${#PIP_NV_INC_PATHS[@]} -gt 0 ]]; then
    venv_inc_csv=$(IFS=:; echo "${PIP_NV_INC_PATHS[*]}")
    export CPATH="$venv_inc_csv${CPATH:+:$CPATH}"
    echo "[env] CPATH prepended with venv pip-wheel headers :"
    for d in "${PIP_NV_INC_PATHS[@]}"; do echo "        $d"; done
fi
# CUDA_HOME defaults to whatever nvcc resolves to. TE's setup.py reads
# this — make it explicit so a stray $CUDA_HOME from a different shell
# doesn't sneak in.
if [[ $have_nvcc -eq 1 && -d "$cuda_home_effective" ]]; then
    export CUDA_HOME="$cuda_home_effective"
    echo "[env] CUDA_HOME=$CUDA_HOME"
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

# ---- 5.5  Write te_env.sh helper for future shells -------------------
# Re-running gpu_power_bench.py from a fresh shell would otherwise miss
# the LD_LIBRARY_PATH / CPATH settings we computed. Drop a sourceable
# helper next to this script so the user runs:
#   source util/gpu_power_bench/te_env.sh
# and gets the same env without re-running the installer.
te_env="$(dirname "$0")/te_env.sh"
{
    echo "#!/usr/bin/env bash"
    echo "# Auto-generated by install_transformer_engine.sh on $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "# Source this to get the same library-path setup the TE install used."
    echo "# It pins venv-managed nvidia-cublas/cudnn/etc above the system toolkit"
    echo "# so dlopen finds the same .so TE was linked against at build time."
    echo "#"
    echo "# Usage:  source $(realpath "$te_env" 2>/dev/null || echo "$te_env")"
    echo
    if [[ ${#PIP_NV_LIB_PATHS[@]} -gt 0 ]]; then
        echo "# venv pip-wheel CUDA libraries (nvidia-cublas-cu*, nvidia-cudnn-cu*, …)"
        for d in "${PIP_NV_LIB_PATHS[@]}"; do
            echo "export LD_LIBRARY_PATH=\"$d\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
        done
    fi
    if [[ ${#PIP_NV_INC_PATHS[@]} -gt 0 ]]; then
        for d in "${PIP_NV_INC_PATHS[@]}"; do
            echo "export CPATH=\"$d\${CPATH:+:\$CPATH}\""
        done
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        echo "export CUDA_HOME=\"$CUDA_HOME\""
    fi
    echo
    echo "echo \"[te_env] LD_LIBRARY_PATH prefixed with venv pip-wheel libs (nvidia-cublas etc.)\""
} > "$te_env"
chmod +x "$te_env" 2>/dev/null || true
grn "[env] wrote $te_env  — source it before running gpu_power_bench.py in a new shell"

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
    if [[ -f "$te_env" ]]; then
        echo "   In a fresh shell, first: source $te_env"
    fi
    exit 0
fi

# ---- Verify failed — try once more with explicit LD_PRELOAD of venv libs ---
# The most common late failure is "undefined symbol: cublasLt*_internal,
# version libcublasLt.so.13" — torch's bundle / system cuBLAS gets
# loaded before we can override. LD_PRELOAD forces the venv .so up
# front, BEFORE python starts importing torch. If this retry succeeds,
# we tell the user exactly what to source going forward.
if [[ ${#PIP_NV_LIB_PATHS[@]} -gt 0 ]]; then
    preload_libs=()
    for d in "${PIP_NV_LIB_PATHS[@]}"; do
        for so in "$d"/libcublasLt.so.* "$d"/libcudnn.so.*; do
            [[ -f "$so" ]] && preload_libs+=("$so")
        done
    done
    if [[ ${#preload_libs[@]} -gt 0 ]]; then
        ylw ""
        ylw "[verify] first probe failed — retrying with LD_PRELOAD of venv libs"
        ylw "         (this nails down the cuBLAS/cuDNN that TE was linked against)"
        preload_csv=$(IFS=:; echo "${preload_libs[*]}")
        LD_PRELOAD="$preload_csv" "$PYTHON" - <<'PY' || rc2=$?
import sys, traceback
try:
    import torch, transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    linear = te.Linear(64, 64, bias=False, params_dtype=torch.float16)
    if torch.cuda.is_available():
        linear = linear.cuda()
        x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        r = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.E4M3,
                                     amax_history_len=4, amax_compute_algo="max")
        with te.fp8_autocast(enabled=True, fp8_recipe=r):
            linear(x)
        print("  fp8_autocast forward pass: OK (with LD_PRELOAD)")
except Exception as e:
    print("  VERIFY (LD_PRELOAD) FAILED:", type(e).__name__, e)
    traceback.print_exc(); sys.exit(1)
PY
        rc2=${rc2:-0}
        if [[ $rc2 -eq 0 ]]; then
            grn ""
            grn "OK — TE works when venv libs are LD_PRELOAD'd."
            grn "Use this in your normal shell to make it permanent :"
            echo "    source $te_env"
            grn "or, equivalently:"
            echo "    export LD_LIBRARY_PATH=\"$(IFS=:; echo "${PIP_NV_LIB_PATHS[*]}"):\${LD_LIBRARY_PATH:-}\""
            exit 0
        fi
    fi
fi

red "install finished but the te.Linear + fp8_autocast runtime probe failed."
echo "   Common causes:"
echo "     * torch ABI mismatch — force-reinstall against THIS torch:"
echo "         $PYTHON -m pip install --force-reinstall --no-build-isolation '$PKG'"
echo "     * missing CUDA libs on loader path — try:"
echo "         source $te_env"
echo "         $PYTHON -c 'import transformer_engine.pytorch'"
echo "     * cudnn-dev / nvcc too old for this TE version"
echo "     * NVIDIA prebuilt wheel mismatch — pin a different version, e.g.:"
echo "         pip uninstall -y transformer-engine transformer-engine-torch"
echo "         TE_VERSION=2.5 ./install_transformer_engine.sh"
exit 1
