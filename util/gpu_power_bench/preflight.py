#!/usr/bin/env python3
"""Pre-flight checks before running the GPU power benchmark.

Verifies:
  * nvidia-smi visible and driver alive
  * pynvml, torch, matplotlib, pandas importable
  * A CUDA device is visible and compute capability is known
  * FP8 dtype availability on the installed torch build
  * NVML power-reading support (required for Joule integration)
  * Persistence mode (recommended for stable clocks / minimal driver latency)

Run as a standalone: `python3 preflight.py` → exits non-zero on fatal gaps.
Or import `check()` which returns a dict with the findings.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class PreflightResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


REQUIRED_PKGS = ("torch", "numpy", "pynvml", "nvtx", "matplotlib", "pandas")


# Distribution names differ from import names for some required packages
# (most notably `pynvml` ships in the `nvidia-ml-py` distribution, and
# older NVTX wheels expose no `__version__` attribute on the module).
# Map import-name → list of plausible distribution names so we can
# fall back to importlib.metadata when the module-level attribute is
# missing.
_DIST_NAME_FALLBACKS = {
    "pynvml": ["nvidia-ml-py", "nvidia-ml-py3", "pynvml"],
    "nvtx":   ["nvtx", "nvtx-plugins"],
}


def _pkg_version(name: str, mod) -> str:
    """Best-effort version string. Tries module __version__ first, then
    importlib.metadata.version() against several distribution-name
    candidates. Returns "?" only when every probe fails."""
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        return "?"
    for dist in _DIST_NAME_FALLBACKS.get(name, [name]):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return "?"


def _check_pkgs(r: PreflightResult) -> None:
    for name in REQUIRED_PKGS:
        try:
            mod = importlib.import_module(name)
            r.info[f"{name}_version"] = _pkg_version(name, mod)
        except ImportError as e:
            r.fail(f"missing python package: {name} ({e}) — pip install -r requirements.txt")


def _check_nvidia_smi(r: PreflightResult) -> None:
    if shutil.which("nvidia-smi") is None:
        r.fail("nvidia-smi not on PATH — install NVIDIA driver")
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,power.management",
             "--format=csv,noheader"],
            text=True, timeout=10)
        r.info["nvidia_smi_query"] = out.strip()
    except subprocess.SubprocessError as e:
        r.fail(f"nvidia-smi failed: {e}")


def _parse_cuda_version(s: str) -> tuple[int, int] | None:
    """Parse '12.1' or '12.8.1' or '11.6' → (major, minor).  Returns None
    if the string doesn't look like a CUDA version."""
    if not s:
        return None
    parts = s.strip().split(".")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return None


def _check_nvcc_toolkit(r: PreflightResult) -> None:
    """Parse `nvcc --version` and compare it to torch.version.cuda.

    Why: Transformer Engine compiles from source against the SYSTEM CUDA
    toolkit (nvcc), not torch's bundled CUDA runtime. When a server has
    a pip-installed torch 2.x+cu128 but a stale system nvcc (e.g. CUDA
    11.x from `/usr/local/cuda`), TE's setup.py refuses with:
        RuntimeError: Transformer Engine requires CUDA 12.0 or newer
    even though torch.version.cuda reads 12.8. Surface this up-front so
    the user knows BEFORE starting the 10-minute compile.
    """
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        r.info["nvcc"] = "NOT FOUND (no nvcc on PATH)"
        return
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, timeout=10)
    except subprocess.SubprocessError as e:
        r.info["nvcc"] = f"error invoking nvcc: {e}"
        return
    # nvcc --version emits e.g. "Cuda compilation tools, release 12.1, V12.1.105"
    nvcc_ver_str = ""
    for line in out.splitlines():
        if "release" in line:
            after = line.split("release", 1)[1]
            nvcc_ver_str = after.split(",", 1)[0].strip()
            break
    r.info["nvcc"] = f"{nvcc_path}  (release {nvcc_ver_str or '?'})"

    nvcc_ver = _parse_cuda_version(nvcc_ver_str)
    try:
        import torch
        torch_cuda = _parse_cuda_version(torch.version.cuda)
    except ImportError:
        torch_cuda = None

    # TE minimum CUDA for the current 2.x line is 12.0.
    if nvcc_ver is not None and nvcc_ver < (12, 0):
        msg = (f"system CUDA toolkit is {nvcc_ver[0]}.{nvcc_ver[1]} — "
               f"Transformer Engine requires ≥ 12.0 to build from source")
        if torch_cuda and torch_cuda >= (12, 0):
            msg += (f" (your torch reports {torch_cuda[0]}.{torch_cuda[1]} "
                    f"because the wheel bundles its own runtime — TE ignores that "
                    f"and uses nvcc)")
        msg += (". Either install a newer toolkit (conda install -c nvidia "
                "cuda-toolkit=12.1) or use the prebuilt wheel path — "
                "./install_transformer_engine.sh now tries pypi.nvidia.com first")
        r.warn(msg)
    elif (nvcc_ver is not None and torch_cuda is not None
            and nvcc_ver[0] != torch_cuda[0]):
        r.warn(f"system nvcc major ({nvcc_ver[0]}.{nvcc_ver[1]}) differs from "
               f"torch.version.cuda ({torch_cuda[0]}.{torch_cuda[1]}); "
               "TE binaries may build but fail to load at runtime.")


def _check_cuda_and_fp8(r: PreflightResult) -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        r.fail("torch.cuda.is_available() = False — no CUDA runtime or no visible GPU")
        return
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cc = torch.cuda.get_device_capability(idx)
    r.info["cuda_device"] = f"[{idx}] {name}"
    r.info["compute_capability"] = f"{cc[0]}.{cc[1]}"
    r.info["torch_cuda"] = torch.version.cuda

    has_fp8_dtype = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")
    r.info["fp8_dtype_in_torch"] = has_fp8_dtype
    if not has_fp8_dtype:
        r.warn("torch build lacks float8_e4m3fn/float8_e5m2 — upgrade to torch>=2.1 for FP8 benchmarks")
    # Native FP8 tensor cores: Hopper (sm_90) + Ada (sm_89 partial). A100 is sm_80 → FP8 emulated.
    r.info["fp8_native"] = cc[0] >= 9
    if cc == (8, 0):
        r.warn("A100 (sm_80) detected: FP8 elementwise EMULATED (no native tensor-core FP8); "
               "matmul_fp8_te variant will fall back to FP16 TC (auto-tagged in notes)")
    elif cc[0] < 8:
        r.warn(f"compute capability {cc[0]}.{cc[1]} is older than A100 — FP16 tensor cores may be slow or absent")

    # Tensor Core support matrix — informational, useful in the report.
    # Use tuple comparison (cc >= (M, m)) for the version checks: the old
    # `cc[0] >= 7 and cc[1] >= 2` form falsely returned False on Ampere
    # (cc=8.0) and Hopper (cc=9.0) for int8_tc because minor=0 fails the
    # minor-version test, even though those generations clearly inherit
    # Turing's INT8 tensor cores.
    tc = {
        "fp16_tc":  cc >= (7, 0),    # Volta+
        "bf16_tc":  cc >= (8, 0),    # Ampere+
        "tf32_tc":  cc >= (8, 0),    # Ampere+
        "int8_tc":  cc >= (7, 5),    # Turing+
        "fp8_tc":   cc >= (9, 0),    # Hopper+
    }
    r.info["tensor_core_support"] = ", ".join(f"{k}={v}" for k, v in tc.items())

    # Transformer Engine — required for the fp8_te matmul variant.
    # Three failure modes that each produce a different symptom:
    #   (a) `transformer_engine` not installed at all            → ImportError
    #   (b) meta-package stub only (no [pytorch] extra) —        → ImportError /
    #       `import transformer_engine.pytorch` fails              ModuleNotFoundError
    #   (c) pytorch extension .so missing or ABI-mismatched —    → OSError /
    #       import passes but `te.Linear(...)` errors at runtime   RuntimeError
    # The CI-friendly way to distinguish (b) and (c) is to actually instantiate
    # a tiny te.Linear and run a forward pass under fp8_autocast; if that
    # succeeds, the install is genuinely usable.
    te_status = "NOT installed"
    te_ok = False
    try:
        import transformer_engine  # noqa: F401
        te_status = getattr(transformer_engine, "__version__", "installed")
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe as te_recipe
            # Runtime probe: forces the torch-backend .so to load.
            linear = te.Linear(64, 64, bias=False, params_dtype=torch.float16)
            if torch.cuda.is_available():
                linear = linear.cuda()
                x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
                recipe = te_recipe.DelayedScaling(
                    fp8_format=te_recipe.Format.E4M3,
                    amax_history_len=4, amax_compute_algo="max")
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    linear(x)
            te_ok = True
        except (OSError, RuntimeError) as e:
            te_status = f"torch-backend BROKEN ({type(e).__name__}: {str(e)[:120]})"
        except ImportError as e:
            te_status = f"META-PACKAGE ONLY (no [pytorch] extra): {e}"
    except ImportError:
        pass
    r.info["transformer_engine"] = te_status
    if not te_ok:
        hint = ("./install_transformer_engine.sh   "
                "# or: pip install --no-build-isolation 'transformer-engine[pytorch]'")
        if cc[0] >= 9:
            r.warn(f"H100 detected but TE missing/broken — fp8_te variant will be skipped. Fix: {hint}")
        else:
            r.warn(f"TE missing/broken — matmul_fp8_te will be skipped (would fall back to FP16 TC anyway). "
                   f"To run the fallback for cross-GPU comparison: {hint}")


def _check_pynvml(r: PreflightResult) -> None:
    try:
        import pynvml
    except ImportError:
        return
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        r.fail(f"nvmlInit failed: {e}")
        return
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        # Power: we integrate mW samples → Joules. Required.
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
            r.info["idle_power_w"] = f"{power_mw/1000:.1f}"
        except pynvml.NVMLError as e:
            r.fail(f"nvmlDeviceGetPowerUsage not supported on this GPU: {e}")
        # Temperature: required for cool-down.
        try:
            t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            r.info["gpu_temp_c"] = t
        except pynvml.NVMLError as e:
            r.warn(f"nvmlDeviceGetTemperature failed ({e}) — cool-down disabled")
        # Persistence mode (recommended, not required).
        try:
            pm = pynvml.nvmlDeviceGetPersistenceMode(h)
            r.info["persistence_mode"] = bool(pm)
            if not pm:
                r.warn("persistence mode OFF — run `sudo nvidia-smi -pm 1` for steadier clocks")
        except pynvml.NVMLError:
            pass
        # ECC (informational).
        try:
            ecc_cur, ecc_pend = pynvml.nvmlDeviceGetEccMode(h)
            r.info["ecc"] = f"current={bool(ecc_cur)} pending={bool(ecc_pend)}"
        except pynvml.NVMLError:
            pass
    finally:
        pynvml.nvmlShutdown()


def check() -> PreflightResult:
    r = PreflightResult()
    _check_pkgs(r)
    _check_nvidia_smi(r)
    _check_nvcc_toolkit(r)
    _check_cuda_and_fp8(r)
    _check_pynvml(r)
    return r


def print_report(r: PreflightResult) -> None:
    print("=" * 72)
    print("GPU power benchmark — pre-flight check")
    print("=" * 72)
    for k, v in r.info.items():
        print(f"  {k:28s} {v}")
    if r.warnings:
        print("\n[warnings]")
        for w in r.warnings:
            print(f"  - {w}")
    if r.errors:
        print("\n[errors]")
        for e in r.errors:
            print(f"  ! {e}")
    print()
    print("PASS" if r.ok else "FAIL")


if __name__ == "__main__":
    res = check()
    print_report(res)
    sys.exit(0 if res.ok else 1)
