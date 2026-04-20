#!/usr/bin/env python3
"""The ten GPU power benchmarks: {FP8, FP16} × {MUL, ADD, Softmax, GeLU, LayerNorm}.

Each benchmark is a small factory that returns a callable `f()` plus metadata
describing the workload. The driver calls `f()` K times in a tight loop
between NVML power samples; total_ops = K × ops_per_call, and Joules/op is
(E_total − P_static × t_total) / total_ops.

FP8 notes:
  * `torch.float8_e4m3fn` lands in torch 2.1+. A100 (sm_80) has no native FP8
    tensor-core path, so elementwise ops are computed by promoting through
    fp16/fp32; H100 (sm_90) runs FP8 on HW. Either way, the *energy* to
    complete the workload is what we actually want to measure — the benchmark
    just reports what the silicon does.
  * For Softmax / GeLU / LayerNorm, reductions and non-linearities run in
    fp32 math even when the input is fp16/fp8 (standard numerical practice).
    We quote FLOP counts only as first-order estimates; `joule_per_element`
    is the precision-agnostic primary metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch


# FLOPs-per-element estimates for each op, used only for reporting
# joule-per-FLOP as a secondary metric. The primary metric is joule-per-element.
FLOPS_PER_ELEMENT = {
    "mul":       1,   # a * b
    "add":       1,   # a + b
    "softmax":   5,   # max, sub, exp, sum, div
    "gelu":      8,   # tanh-approx: x, x^3, mul, add, tanh, add, mul, mul
    "layernorm": 8,   # mean, var, sub, div, mul, add (amortized across dim)
}


@dataclass
class BenchSpec:
    name: str              # "fp16_mul", "fp8_softmax", ...
    op: str                # "mul" | "add" | "softmax" | "gelu" | "layernorm"
    dtype_label: str       # "fp16" | "fp8"
    shape: tuple[int, ...]
    n_elements: int        # total element count
    flops_per_call: int    # estimated FLOPs in one call to f()
    run: Callable[[], None]
    notes: str = ""


# ---------- dtype resolution ------------------------------------------------

def _resolve_dtype(label: str) -> tuple[torch.dtype, Optional[torch.dtype]]:
    """Return (storage_dtype, compute_dtype or None).

    For fp8, storage is float8_e4m3fn but compute is bf16/fp16 (elementwise
    kernels don't operate natively on fp8 — they cast, compute, cast back).
    """
    if label == "fp16":
        return torch.float16, None
    if label == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch build has no float8_e4m3fn (need torch>=2.1)")
        return torch.float8_e4m3fn, torch.float16
    raise ValueError(f"unknown dtype label {label!r}")


def _alloc_like(shape, dtype, device, compute_dtype=None):
    """Create a random tensor of the requested dtype.

    torch.randn does not support float8 directly — we draw in fp32 and cast.
    """
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.randn(*shape, dtype=dtype, device=device)
    # fp8 path: draw in fp32, scale into fp8's small range, cast.
    x = torch.randn(*shape, dtype=torch.float32, device=device) * 0.25
    return x.to(dtype)


# ---------- op kernels (one-call closures) ----------------------------------

def _make_mul(shape, dtype_label, device) -> Callable[[], None]:
    dt, ct = _resolve_dtype(dtype_label)
    a = _alloc_like(shape, dt, device, ct)
    b = _alloc_like(shape, dt, device, ct)
    if ct is None:
        def f():
            torch.mul(a, b)
        return f
    # FP8: cast → compute → cast back. Mirrors what real FP8 inference kernels do.
    def f():
        a16 = a.to(ct)
        b16 = b.to(ct)
        out = torch.mul(a16, b16)
        out.to(dt)
    return f


def _make_add(shape, dtype_label, device) -> Callable[[], None]:
    dt, ct = _resolve_dtype(dtype_label)
    a = _alloc_like(shape, dt, device, ct)
    b = _alloc_like(shape, dt, device, ct)
    if ct is None:
        def f():
            torch.add(a, b)
        return f
    def f():
        a16 = a.to(ct)
        b16 = b.to(ct)
        out = torch.add(a16, b16)
        out.to(dt)
    return f


def _make_softmax(shape, dtype_label, device) -> Callable[[], None]:
    dt, ct = _resolve_dtype(dtype_label)
    x = _alloc_like(shape, dt, device, ct)
    if ct is None:
        def f():
            torch.nn.functional.softmax(x, dim=-1)
        return f
    def f():
        x16 = x.to(ct)
        out = torch.nn.functional.softmax(x16, dim=-1)
        out.to(dt)
    return f


def _make_gelu(shape, dtype_label, device) -> Callable[[], None]:
    dt, ct = _resolve_dtype(dtype_label)
    x = _alloc_like(shape, dt, device, ct)
    if ct is None:
        def f():
            torch.nn.functional.gelu(x, approximate="tanh")
        return f
    def f():
        x16 = x.to(ct)
        out = torch.nn.functional.gelu(x16, approximate="tanh")
        out.to(dt)
    return f


def _make_layernorm(shape, dtype_label, device) -> Callable[[], None]:
    dt, ct = _resolve_dtype(dtype_label)
    x = _alloc_like(shape, dt, device, ct)
    norm_shape = (shape[-1],)
    # weight/bias must match the compute dtype so PyTorch doesn't up-promote.
    w_dtype = ct if ct is not None else dt
    weight = torch.ones(norm_shape, dtype=w_dtype, device=device)
    bias = torch.zeros(norm_shape, dtype=w_dtype, device=device)
    if ct is None:
        def f():
            torch.nn.functional.layer_norm(x, norm_shape, weight, bias, eps=1e-5)
        return f
    def f():
        x16 = x.to(ct)
        out = torch.nn.functional.layer_norm(x16, norm_shape, weight, bias, eps=1e-5)
        out.to(dt)
    return f


_BUILDERS = {
    "mul":       _make_mul,
    "add":       _make_add,
    "softmax":   _make_softmax,
    "gelu":      _make_gelu,
    "layernorm": _make_layernorm,
}


# ---------- shape helpers ---------------------------------------------------

def _shape_for(op: str, n_elements: int) -> tuple[int, ...]:
    """Pick a 1-D or 2-D shape with n_elements total.

    Elementwise ops (mul/add/gelu) → 1-D. Reductions (softmax/layernorm)
    → 2-D with a fixed "feature" dim so the reduction cost per row stays
    realistic (matches a transformer's hidden size).
    """
    if op in ("mul", "add", "gelu"):
        return (n_elements,)
    # softmax, layernorm: reduction along last dim; fix D = 1024 (transformer-ish)
    D = 1024
    M = max(1, n_elements // D)
    # snap n_elements to M*D so downstream math is exact
    return (M, D)


def build(op: str, dtype_label: str, n_elements: int,
          device: str | torch.device = "cuda") -> BenchSpec:
    if op not in _BUILDERS:
        raise ValueError(f"unknown op {op!r} (choices: {list(_BUILDERS)})")
    if dtype_label not in ("fp16", "fp8"):
        raise ValueError(f"unknown dtype {dtype_label!r}")
    shape = _shape_for(op, n_elements)
    actual_n = math.prod(shape)
    fn = _BUILDERS[op](shape, dtype_label, device)
    flops = actual_n * FLOPS_PER_ELEMENT[op]
    name = f"{dtype_label}_{op}"
    notes = ""
    if dtype_label == "fp8" and torch.cuda.get_device_capability()[0] < 9:
        notes = "fp8 emulated (no native FP8 tensor cores on this GPU)"
    return BenchSpec(
        name=name, op=op, dtype_label=dtype_label,
        shape=shape, n_elements=actual_n,
        flops_per_call=flops, run=fn, notes=notes,
    )


# ---------- the canonical list of 10 benchmarks -----------------------------

OPS = ("mul", "add", "softmax", "gelu", "layernorm")
DTYPES = ("fp16", "fp8")


def all_specs(n_elements: int, device="cuda") -> list[BenchSpec]:
    return [build(op, dt, n_elements, device=device) for dt in DTYPES for op in OPS]
