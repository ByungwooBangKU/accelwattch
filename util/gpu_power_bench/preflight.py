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


def _check_pkgs(r: PreflightResult) -> None:
    for name in REQUIRED_PKGS:
        try:
            mod = importlib.import_module(name)
            r.info[f"{name}_version"] = getattr(mod, "__version__", "?")
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
    tc = {
        "fp16_tc":  cc[0] >= 7,                 # Volta+
        "bf16_tc":  cc[0] >= 8,                 # Ampere+
        "tf32_tc":  cc[0] >= 8,                 # Ampere+
        "int8_tc":  cc[0] >= 7 and cc[1] >= 2,  # Turing+
        "fp8_tc":   cc[0] >= 9,                 # Hopper+
    }
    r.info["tensor_core_support"] = ", ".join(f"{k}={v}" for k, v in tc.items())

    # Transformer Engine — required for the fp8_te matmul variant.
    # The bare `transformer-engine` PyPI package is a META package that
    # errors at runtime; the real install needs `[pytorch]` extra AND
    # `--no-build-isolation`.  Detect both "not installed" and "meta
    # package stub" cases and surface the exact install command.
    te_status = "NOT installed"
    try:
        import transformer_engine  # noqa: F401
        try:
            import transformer_engine.pytorch  # noqa: F401
            te_status = getattr(transformer_engine, "__version__", "installed")
        except Exception as e:
            te_status = f"META-PACKAGE ONLY (broken): {e.__class__.__name__}"
    except ImportError:
        pass
    r.info["transformer_engine"] = te_status
    if te_status != "installed" and not te_status.replace(".", "").isdigit() \
            and not te_status[0:1].isdigit():
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
