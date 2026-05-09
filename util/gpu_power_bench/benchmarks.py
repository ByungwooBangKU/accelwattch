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
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


# FLOPs-per-element estimates for each op, used only for reporting
# joule-per-FLOP as a secondary metric. The primary metric is joule-per-element.
FLOP_PER_ELEMENT = {
    "mul":       1,   # a * b
    "add":       1,   # a + b
    "softmax":   5,   # max, sub, exp, sum, div
    "gelu":      8,   # tanh-approx: x, x^3, mul, add, tanh, add, mul, mul
    "layernorm": 8,   # mean, var, sub, div, mul, add (amortized across dim)
    # STREAM-style probes are intentionally compute-light so all of the
    # measured energy attributes to memory traffic.
    "stream_copy":  0,   # no compute, pure data movement
    "stream_scale": 1,   # one mul per element
    "stream_triad": 2,   # one mul + one add per element
    # stream_read / stream_write — single-direction memory probes for
    # the R/W decomposition. Treated as compute-light (FLOP=0) for the
    # same reason as stream_copy: we want pJ/bit to capture memory cost
    # only, not a tiny incidental sum-reduction. Without these entries,
    # build() raises KeyError when --dram-bw-test enables them.
    "stream_read":  0,
    "stream_write": 0,
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
    # HW path the main FLOPs actually execute on. One of:
    #   "CUDA core"                    — SIMT lanes / regular CUDA cores
    #   "Tensor Core"                  — native TC mma
    #   "Tensor Core (FP16 fallback)"  — fp8_te on pre-Hopper GPUs
    compute_unit: str = "CUDA core"
    # True when the HW path does NOT match what a naive reader would assume
    # from the benchmark name. Examples:
    #   * fp8 elementwise on any GPU — PyTorch has no native FP8 elementwise
    #     kernel so we cast fp8→fp16, compute in fp16, cast back. The reported
    #     energy includes cast-kernel overhead, NOT "real" FP8 compute cost.
    #   * matmul_fp8_te on A100 — Transformer Engine falls back to FP16
    #     Tensor Core; the measurement is an FP16-TC number, not FP8.
    emulated: bool = False
    # Cache/locality regime — classified from working-set vs L2 size. Lets the
    # analyser separate L2-bound from DRAM-bound points instead of forcing a
    # single regression line across a regime change.
    #   "l2_hit_100"   — working set ≤ L2/4, comfortably L2-resident
    #   "l2_hit_75"    — L2/4 < working set ≤ L2/2
    #   "l2_hit_50"    — L2/2 < working set ≤ 2·L2
    #   "l2_hit_25"    — 2·L2 < working set ≤ 4·L2
    #   "l2_hit_0"     — working set > 4·L2, DRAM streaming
    #   "unknown"      — L2 size unavailable (legacy rows / non-CUDA)
    cache_regime: str = "unknown"
    # Fine-grained classification of WHAT the measured energy actually
    # represents. `emulated` is a True/False flag ; this string carries
    # more nuance for downstream plot/CSV. Values :
    #   "native_or_standard"           — the obvious case (fp16 mul, bf16 matmul, etc.)
    #   "emulated_cast_compute_cast"   — fp8 elementwise : FP8 storage,
    #                                    FP16 compute via cast-compute-cast.
    #                                    NOT native FP8 op energy.
    #   "native_or_te_fp8_tensorcore"  — H100 native FP8 path via TE
    #                                    (matmul_fp8_te / attention_flash fp8).
    #   "te_fp16_fallback"             — TE on pre-Hopper : silently runs
    #                                    FP16 TC. NOT a real FP8 measurement.
    # Used by analyze.py to decide hatch patterns, "fp8 native" headlines,
    # and exclude emulated rows from FP8 energy claims.
    path_semantics: str = "native_or_standard"
    notes: str = ""
    # Extra per-case metadata persisted by gpu_power_bench.py.
    # L2 probes use this for logical traffic bits, policy, and kernel geometry.
    extra: dict = field(default_factory=dict)


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


# ---------- STREAM-style DRAM-bandwidth probes -------------------------------
# Pure-streaming kernels modeled after McCalpin's classic STREAM benchmark.
# Used for deriving pJ/bit of DRAM traffic at large working sets:
#   copy   : y = x                 (1 read  + 1 write = 2N bytes/call)
#   scale  : y = α · x             (1 read  + 1 write = 2N bytes/call)
#   triad  : y = α · x + z         (2 reads + 1 write = 3N bytes/call)
# These have minimal compute (one fused MAD at most) so the dynamic energy
# we measure is dominated by HBM traffic — the measurement boundary is
# board-level so it includes the L2 → HBM PHY → DRAM cell path. Compare
# the resulting pJ/bit at the l2_hit_0 cache regime to literature values
# (see README §3.5).
def _make_copy(shape, dtype_label, device):
    dt, ct = _resolve_dtype(dtype_label)
    a = _alloc_like(shape, dt, device, ct)
    out = _alloc_like(shape, dt, device, ct)   # pre-allocated to avoid alloc traffic
    def f():
        out.copy_(a)
    return f


def _make_scale(shape, dtype_label, device):
    dt, ct = _resolve_dtype(dtype_label)
    a = _alloc_like(shape, dt, device, ct)
    if ct is None:
        alpha = torch.tensor(1.5, dtype=dt, device=device)
        def f():
            torch.mul(a, alpha)
        return f
    # fp8 path — promote, scale, demote (still BW-bound at large N)
    alpha = torch.tensor(1.5, dtype=ct, device=device)
    def f():
        a16 = a.to(ct)
        out = torch.mul(a16, alpha)
        out.to(dt)
    return f


def _make_triad(shape, dtype_label, device):
    dt, ct = _resolve_dtype(dtype_label)
    a = _alloc_like(shape, dt, device, ct)
    z = _alloc_like(shape, dt, device, ct)
    if ct is None:
        alpha = torch.tensor(1.5, dtype=dt, device=device)
        def f():
            torch.add(z, a, alpha=1.5)   # y = z + α·a  (PyTorch fused)
        return f
    alpha = torch.tensor(1.5, dtype=ct, device=device)
    def f():
        a16 = a.to(ct); z16 = z.to(ct)
        out = torch.add(z16, a16, alpha=1.5)
        out.to(dt)
    return f


def _make_read_only(shape, dtype_label, device):
    """Pure-read probe: y = sum(x).
    Reads N elements from HBM; writes a single scalar — output traffic is
    negligible (~0 of N bytes). Used to isolate the **DRAM read** energy
    when sampled at the l2_hit_0 cache regime."""
    dt, _ = _resolve_dtype(dtype_label)
    a = _alloc_like(shape, dt, device)
    if dtype_label == "fp8":
        # FP8 has no native sum kernel — promote in-register to fp16 for
        # the reduction. The cost we measure is still dominated by the
        # fp8 LOAD from HBM (1 byte/elem), which is what we want.
        def f():
            torch.sum(a.to(torch.float16))
        return f
    def f():
        torch.sum(a)
    return f


def _make_write_only(shape, dtype_label, device):
    """Pure-write probe: y.fill_(constant).
    Writes N elements to HBM; reads ~0 of x (the fill value is broadcast
    per warp). Used to isolate the **DRAM write** energy at l2_hit_0."""
    dt, _ = _resolve_dtype(dtype_label)
    out = _alloc_like(shape, dt, device)
    def f():
        out.fill_(1.5)
    return f


_STREAM_BUILDERS = {
    "stream_copy":  _make_copy,
    "stream_scale": _make_scale,
    "stream_triad": _make_triad,
    # Single-direction probes — let analyze split the average pJ/bit into
    # separate read and write components.
    "stream_read":  _make_read_only,
    "stream_write": _make_write_only,
}

# Bytes touched per call for each op (read+write counts × N × bytes_per_elem).
# Used by analyze.py to compute pJ/bit. Keys cover both the regular elementwise
# ops and the STREAM probes.
_RW_PER_CALL = {
    "mul": 3, "add": 3,                       # 2 reads + 1 write
    "gelu": 2, "softmax": 2, "layernorm": 2,  # 1 read  + 1 write
    "stream_copy": 2, "stream_scale": 2,
    "stream_triad": 3,
    # Pure single-direction kernels — bytes_traffic = 1 × N × bpe in each
    # case, but the SEMANTICS differ: stream_read's pJ/bit is purely DRAM
    # read cost, stream_write's purely DRAM write cost.
    "stream_read":  1,
    "stream_write": 1,
}


def bytes_per_call(op: str, n_elements: int, dtype_label: str) -> int:
    """Bytes that an op-kernel touches in a single call. For DRAM-bound
    workloads (l2_hit_0) this equals DRAM bytes/call; for L2-resident
    workloads it equals L2 traffic (with most of it staying on-chip)."""
    rw = _RW_PER_CALL.get(op)
    if rw is None:
        return 0
    return rw * n_elements * _dtype_bytes(dtype_label)


# ---------- shape helpers ---------------------------------------------------

def _shape_for(op: str, n_elements: int) -> tuple[int, ...]:
    """Pick a 1-D or 2-D shape with n_elements total.

    Elementwise ops (mul/add/gelu) → 1-D. Reductions (softmax/layernorm)
    → 2-D with a fixed "feature" dim so the reduction cost per row stays
    realistic (matches a transformer's hidden size).
    """
    if op in ("mul", "add", "gelu",
              "stream_copy", "stream_scale", "stream_triad",
              "stream_read", "stream_write"):
        return (n_elements,)
    # softmax, layernorm: reduction along last dim; fix D = 1024 (transformer-ish)
    D = 1024
    M = max(1, n_elements // D)
    # snap n_elements to M*D so downstream math is exact
    return (M, D)


# ---------- L2 size + cache-regime classifier ------------------------------

def _dtype_bytes(label: str) -> int:
    """Storage bytes per element for a dtype label. Note: fp8 storage is
    1 byte, but our cast-compute-cast benchmark materialises full fp16
    intermediates too — the working-set for cache classification tracks the
    DOMINANT resident tensor, which is the fp16 intermediate (2 bytes/elem)
    when the path is emulated, and the storage dtype otherwise."""
    if label == "fp16":
        return 2
    if label == "fp8":
        # cast-compute-cast → fp16 intermediates dominate traffic
        return 2
    if label == "bf16":
        return 2
    if label == "fp32" or label == "tf32":
        return 4
    return 2   # conservative fallback


def get_l2_bytes(device: int = 0) -> int:
    """Return L2 cache size in bytes for the given CUDA device (0 if unknown)."""
    try:
        props = torch.cuda.get_device_properties(device)
        # Attribute name varies across PyTorch versions.
        for attr in ("L2_cache_size", "l2_cache_size"):
            v = getattr(props, attr, None)
            if v:
                return int(v)
    except Exception:
        pass
    return 0


def classify_cache_regime(working_set_bytes: int, l2_bytes: int) -> str:
    """Return a 5-bucket cache-locality regime for a given working set.

    The buckets are log-symmetric around the L2 size: L2/4, L2/2, 2·L2, 4·L2.
    That gives a steady progression from "easily fits" to "grossly exceeds",
    which makes the five per-regime k_op coefficients read as a trend rather
    than the 3-bucket step function the earlier version produced.

      ws ≤ L2/4              → l2_hit_100 : comfortably L2-resident, ~100% hit
      L2/4 < ws ≤ L2/2       → l2_hit_75  : just inside L2, still mostly hit
      L2/2 < ws ≤ 2·L2       → l2_hit_50  : right at / around L2, thrashing
      2·L2 < ws ≤ 4·L2       → l2_hit_25  : clearly larger than L2, spillover
      ws > 4·L2              → l2_hit_0   : DRAM streaming, ~0% L2 hit

    Labels embed the approximate L2 hit rate so plot legends and CSV
    columns read directly as "experiments at ~75% cache hit".
    """
    if l2_bytes <= 0:
        return "unknown"
    if working_set_bytes <= l2_bytes / 4:
        return "l2_hit_100"
    if working_set_bytes <= l2_bytes / 2:
        return "l2_hit_75"
    if working_set_bytes <= 2 * l2_bytes:
        return "l2_hit_50"
    if working_set_bytes <= 4 * l2_bytes:
        return "l2_hit_25"
    return "l2_hit_0"


def _elementwise_working_set(op: str, n_elements: int, bytes_per_elem: int) -> int:
    """Bytes touched per kernel for a given op at a given size.

    - mul/add : 2 reads + 1 write = 3·N·bytes_per_elem
    - stream_read / stream_write : one tensor = 1·N·bytes_per_elem
    - stream_copy / stream_scale : 1 read + 1 write = 2·N·bytes_per_elem
    - stream_triad : 2 reads + 1 write = 3·N·bytes_per_elem
    - gelu    : 1 read + 1 write = 2·N·bytes_per_elem
    - softmax / layernorm : 1 read + 1 write = 2·N·bytes_per_elem (+ weight/bias
      are O(D) so negligible relative to N·D)
    This is what the L2 actually has to hold (transient) for the kernel to
    complete — the figure used to classify the regime.
    """
    if op in ("stream_read", "stream_write"):
        rw = 1
    elif op in ("mul", "add", "stream_triad"):
        rw = 3
    else:
        rw = 2
    return rw * n_elements * bytes_per_elem


def cache_sweep_points(op: str, dtype_label: str, l2_bytes: int) -> list[int]:
    """Return 5 N values that each land in the centre of one cache regime.

    Targets are the geometric centre of each bucket on a log scale, so each
    point is unambiguously inside its regime after rounding:

      bucket        ws target           approx L2 hit
      l2_hit_100    L2/8                ~100 %
      l2_hit_75     L2·√(1/4 · 1/2) ≈ L2·0.35 (≈ L2/3)   ~75 %
      l2_hit_50     L2                  ~50 %
      l2_hit_25     L2·√(2 · 4) ≈ 2.83·L2 (≈ 3·L2)       ~25 %
      l2_hit_0      8·L2                ~0 %
    """
    if l2_bytes <= 0:
        # Fall back to 5 spread-out defaults when L2 size is unknown.
        return [1 << 19, 1 << 21, 1 << 23, 1 << 25, 1 << 27]
    b = _dtype_bytes(dtype_label)
    rw = 3 if op in ("mul", "add") else 2
    def n_for(ws): return max(1 << 14, int(ws) // (rw * b))
    targets = [l2_bytes / 8,
               l2_bytes / 3,
               l2_bytes,
               3 * l2_bytes,
               8 * l2_bytes]
    return [n_for(t) for t in targets]


def build(op: str, dtype_label: str, n_elements: int,
          device: str | torch.device = "cuda") -> BenchSpec:
    builders = {**_BUILDERS, **_STREAM_BUILDERS}
    if op not in builders:
        raise ValueError(f"unknown op {op!r} (choices: {list(builders)})")
    if dtype_label not in ("fp16", "fp8"):
        raise ValueError(f"unknown dtype {dtype_label!r}")
    shape = _shape_for(op, n_elements)
    actual_n = math.prod(shape)
    fn = builders[op](shape, dtype_label, device)
    # `.get(..., 0)` defends against the historic foot-gun of registering
    # a new op in _BUILDERS / _STREAM_BUILDERS but forgetting the matching
    # FLOP_PER_ELEMENT entry — the cell will still run, just with
    # FLOPs=0 (treated as compute-light, J/FLOP comes out NaN per
    # gpu_power_bench.py's zero-divisor guard). Without this, you'd see
    # "build failed: '<op>'" with KeyError, exactly the symptom the
    # stream_read / stream_write addition triggered.
    flops = actual_n * FLOP_PER_ELEMENT.get(op, 0)
    name = f"{dtype_label}_{op}"
    # Elementwise ops never hit Tensor Cores — TC is matmul-only silicon.
    compute_unit = "CUDA core"
    # FP8 elementwise is ALWAYS cast-compute-cast (regardless of GPU) because
    # PyTorch has no native FP8 elementwise kernel. So the measurement on
    # *any* GPU reflects cast-kernel overhead, not true FP8 compute cost.
    emulated = (dtype_label == "fp8")
    notes = ""
    if dtype_label == "fp8":
        notes = ("fp8 emulated via FP16 cast-compute-cast "
                 "(no native FP8 elementwise kernel in PyTorch)")
    # Cache regime — requires knowing L2 size on the ACTUAL target device.
    dev_idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
    l2 = get_l2_bytes(dev_idx)
    ws = _elementwise_working_set(op, actual_n, _dtype_bytes(dtype_label))
    regime = classify_cache_regime(ws, l2)
    # Path semantics : fp8 elementwise is ALWAYS cast-compute-cast in
    # PyTorch (no native FP8 elementwise kernel). Other dtypes run their
    # natural path.
    path_semantics = ("emulated_cast_compute_cast" if dtype_label == "fp8"
                      else "native_or_standard")
    return BenchSpec(
        name=name, op=op, dtype_label=dtype_label,
        shape=shape, n_elements=actual_n,
        flops_per_call=flops, run=fn,
        compute_unit=compute_unit, emulated=emulated,
        cache_regime=regime, path_semantics=path_semantics, notes=notes,
    )


# ---------- the canonical list of 10 benchmarks -----------------------------

OPS = ("mul", "add", "softmax", "gelu", "layernorm")
DTYPES = ("fp16", "fp8")


def all_specs(n_elements: int, device="cuda") -> list[BenchSpec]:
    return [build(op, dt, n_elements, device=device) for dt in DTYPES for op in OPS]


# ============================================================================
# Matmul variants — "Tensor Core vs CUDA Core" + "native FP8 via Transformer
# Engine" comparison axis. These add up to 5 variants that sweep over the
# matrix side length K (M = N = K for a square-square GEMM).
# ============================================================================
#
# What each variant runs:
#   matmul_fp32_simt : fp32 inputs, TF32 explicitly DISABLED → CUDA cores (SIMT)
#                      MAD.  This is the Tensor-Core-off baseline.  Same FLOPs
#                      as TC paths, vastly more joules — that delta is the
#                      "Tensor Core energy advantage".
#   matmul_tf32_tc   : fp32 inputs, TF32 enabled → TF32 Tensor Core path.
#                      Ampere+ (sm_80+) only.  10-bit mantissa, 8-bit exponent.
#   matmul_fp16_tc   : fp16 inputs → FP16 Tensor Core (wmma).  Both A100 & H100.
#   matmul_bf16_tc   : bf16 inputs → BF16 Tensor Core.  Same peak as FP16 but
#                      different numerics (more dynamic range).  Both A100/H100.
#   matmul_fp8_te    : FP8 (E4M3) via Transformer Engine fp8_autocast on
#                      te.Linear.  On H100 this hits native FP8 Tensor Cores;
#                      TE on pre-Hopper falls back to FP16 TC — benchmark is
#                      auto-skipped or tagged as "fallback" accordingly.
#
# Why these are the right variants:
#   * fp32_simt is the only way to force "no Tensor Core" for a GEMM; PyTorch
#     dispatches fp16/bf16 matmul to Tensor Core unconditionally on Ampere+.
#   * tf32 vs fp16 TC on the same GPU measures the energy cost of
#     mantissa width while keeping the same HW unit.
#   * fp8 TE is the unique H100 capability we want to price vs A100 FP16 TC.
#
# Load axis: matrix side length K (M = N = K).  FLOPs per call = 2·K³.
# Memory:    3 × K² elements (A, B, out).  K=12288 fp32 ≈ 1.7 GB — fits on
# both 80 GB A100 (HBM2E) and 80 GB H100 with large margin.
# ============================================================================

# (dtype_label, compute_mode) — compute_mode ∈ {"simt", "tc", "te"}
MATMUL_VARIANTS: tuple[tuple[str, str], ...] = (
    ("fp32", "simt"),
    ("tf32", "tc"),
    ("fp16", "tc"),
    ("bf16", "tc"),
    ("fp8",  "te"),
)


_TORCH_DTYPES = {
    "fp32": torch.float32,
    "tf32": torch.float32,   # TF32 uses fp32 storage, TC kernel differs by flag
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_matmul_fp32_simt(M, N, K, device):
    a = torch.randn(M, K, dtype=torch.float32, device=device)
    b = torch.randn(K, N, dtype=torch.float32, device=device)
    def f():
        # Flag flip is a cheap python bool assignment; re-asserting every call
        # is defensive against another benchmark's builder having flipped it.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.matmul(a, b)
    return f


def _make_matmul_tf32_tc(M, N, K, device):
    a = torch.randn(M, K, dtype=torch.float32, device=device)
    b = torch.randn(K, N, dtype=torch.float32, device=device)
    def f():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.matmul(a, b)
    return f


def _make_matmul_halfprec_tc(M, N, K, dtype, device):
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    def f():
        torch.matmul(a, b)
    return f


def _make_matmul_fp8_te(M, N, K, device):
    """FP8 GEMM via Transformer Engine. Native Tensor Core path on H100 only.

    Two distinct failure modes we need to surface with actionable messages:

    1. Module not importable   → user never installed TE (or installed bare
       `transformer-engine` meta-package without the `[pytorch]` extra).
       Raises ImportError.
    2. Module imports, but the compiled torch backend shared-object isn't
       loadable at runtime — typically because the installed TE wheel was
       built against a different torch version than the one we're running,
       or CUDA libs aren't on the loader path. This surfaces as OSError
       ("cannot open shared object file") or RuntimeError ("could not find
       shared object file for transformer engine torch lib"). The import
       line alone doesn't trigger it — `te.Linear(...)` / `fp8_autocast(...)`
       does — so we build and warm the module here to fail early with a
       clear message instead of a mid-sweep traceback.
    """
    _install_hint = (
        "Run ./install_transformer_engine.sh (it checks prerequisites and "
        "uses --no-build-isolation + the [pytorch] extra, which is the "
        "combination that avoids the meta-package and shared-object traps)."
    )
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe
    except ImportError as e:
        raise RuntimeError(
            f"transformer_engine not importable ({e}). {_install_hint}"
        ) from e
    except (OSError, RuntimeError) as e:
        # Import line can itself load the torch backend .so lazily on some
        # TE versions; catch that here too.
        raise RuntimeError(
            f"transformer_engine imported but its torch backend shared "
            f"library failed to load ({type(e).__name__}: {e}). This usually "
            f"means the installed TE wheel was built against a different "
            f"torch version, or the meta-package stub is installed. "
            f"{_install_hint}"
        ) from e

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    # te.Linear expects (in_features, out_features); we map K → N.
    # bias=False keeps the kernel pure GEMM for apples-to-apples FLOP counting.
    try:
        linear = te.Linear(K, N, bias=False, params_dtype=torch.float16).to(device)
        fp8_recipe = te_recipe.DelayedScaling(
            fp8_format=te_recipe.Format.E4M3,
            amax_history_len=16,
            amax_compute_algo="max",
        )
        # Warmup — 5 iterations + sync. Two reasons:
        #   (a) forces lazy load of libtransformer_engine_torch.so so a
        #       missing / ABI-mismatched build fails during build() not
        #       inside the power-sampled loop;
        #   (b) catches Blackwell + small-M edge cases where TE's amax
        #       buffer maintenance corrupts CUDA state on the second or
        #       third call (single-warmup wasn't enough — see README
        #       §8.3.3). Once we sync 5 calls cleanly, the cell is safe.
        for _ in range(5):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                linear(x)
        torch.cuda.synchronize()
    except (OSError, RuntimeError) as e:
        msg = str(e)
        if ("shared object" in msg.lower()
                or "transformer_engine_torch" in msg
                or "libtransformer_engine" in msg):
            raise RuntimeError(
                f"transformer_engine loaded but the torch backend shared "
                f"library is missing or unloadable ({type(e).__name__}: "
                f"{e}). {_install_hint}"
            ) from e
        raise

    def f():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            linear(x)
    return f


def build_matmul(K_size: int, dtype_label: str, mode: str,
                 device: str | torch.device = "cuda") -> BenchSpec:
    """Build one matmul BenchSpec (M = N = K = K_size)."""
    M = N = K = K_size
    cc = torch.cuda.get_device_capability()
    notes = ""
    compute_unit = "Tensor Core"   # default for all matmul variants except SIMT
    emulated = False

    if (dtype_label, mode) == ("fp32", "simt"):
        fn = _make_matmul_fp32_simt(M, N, K, device)
        compute_unit = "CUDA core"
    elif (dtype_label, mode) == ("tf32", "tc"):
        if cc[0] < 8:
            raise RuntimeError(f"TF32 requires Ampere (sm_80) or newer (this GPU is sm_{cc[0]}{cc[1]})")
        fn = _make_matmul_tf32_tc(M, N, K, device)
    elif (dtype_label, mode) == ("fp16", "tc"):
        fn = _make_matmul_halfprec_tc(M, N, K, torch.float16, device)
    elif (dtype_label, mode) == ("bf16", "tc"):
        if cc[0] < 8:
            raise RuntimeError(f"BF16 requires Ampere (sm_80) or newer")
        fn = _make_matmul_halfprec_tc(M, N, K, torch.bfloat16, device)
    elif (dtype_label, mode) == ("fp8", "te"):
        fn = _make_matmul_fp8_te(M, N, K, device)
        if cc[0] < 9:
            # Pre-Hopper (A100 etc.) has no FP8 Tensor Core — TE falls back to
            # the FP16 TC path. The measurement is therefore an FP16-TC number
            # masquerading as FP8, and should be flagged.
            compute_unit = "Tensor Core (FP16 fallback)"
            emulated = True
            notes = ("fp8_te on pre-Hopper: Transformer Engine falls back to "
                     "FP16 Tensor Core — NOT a native FP8 measurement")
    else:
        raise ValueError(f"unknown matmul variant ({dtype_label!r}, {mode!r})")

    flops = 2 * M * N * K
    n_out = M * N  # output element count (for sanity; primary metric is J/FLOP)
    name = f"matmul_{dtype_label}_{mode}"
    # Cache regime for matmul: working set = A (M·K) + B (K·N) + C (M·N). Note
    # matmul has intrinsic reuse (each element of A/B read K times) so even
    # when the full working set exceeds L2, tile-level reuse recovers a big
    # fraction of hits. The regime label still distinguishes "small GEMMs
    # where everything fits" from "big GEMMs where tiles thrash".
    ws_bytes = (M * K + K * N + M * N) * _dtype_bytes(dtype_label)
    dev_idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
    regime = classify_cache_regime(ws_bytes, get_l2_bytes(dev_idx))
    # Path semantics for matmul :
    #   fp8_te native (Hopper, cc≥9)   → "native_or_te_fp8_tensorcore"
    #   fp8_te fallback (pre-Hopper)   → "te_fp16_fallback"
    #   other matmul variants          → "native_or_standard"
    if (dtype_label, mode) == ("fp8", "te"):
        path_semantics = ("te_fp16_fallback" if emulated
                          else "native_or_te_fp8_tensorcore")
    else:
        path_semantics = "native_or_standard"
    return BenchSpec(
        name=name, op="matmul", dtype_label=dtype_label,
        shape=(M, N, K), n_elements=n_out,
        flops_per_call=flops, run=fn,
        compute_unit=compute_unit, emulated=emulated,
        cache_regime=regime, path_semantics=path_semantics, notes=notes,
    )


def matmul_all_specs(K_size: int, device="cuda") -> list[BenchSpec]:
    """Build every matmul variant at one K.  Variants that error (e.g. fp8_te
    on a system without transformer_engine) are silently skipped; the caller
    can note which were built."""
    out = []
    for dtype_label, mode in MATMUL_VARIANTS:
        try:
            out.append(build_matmul(K_size, dtype_label, mode, device))
        except Exception:
            continue
    return out


# ============================================================================
# LLM-shape matmul sweep — same builders as build_matmul() but with real
# layer shapes from a representative large model (gpt-oss-120B), and M
# (= token count T) swept explicitly rather than coupled to K. The square
# sweep above is kept because square R² is the cleanest signal for linearity;
# this one is for "what does one inference step of layer X actually cost?".
# ============================================================================
#
# Hidden dim d = 2880. K and N hardcoded per role below. Every preset is
# a plain linear layer (Y = X @ W) with bias=False so the measurement is
# pure GEMM — no epilogue bias-add to muddle the FLOP count.
#
# Override shapes with LLM_SHAPES["my_layer"] = (K, N) from python.
# ============================================================================

LLM_SHAPES: dict[str, tuple[int, int]] = {
    # name      : (K  = reduction dim,  N = output dim)
    "qkv":      (2880, 5120),    # merged QKV projection
    "q_only":   (2880, 4096),    # Q head projection
    "kv":       (2880,  512),    # K/V head projection (GQA — K ≫ N, skinny)
    "attn_o":   (4096, 2880),    # attention output projection
    "router":   (2880,  128),    # MoE gate (very skinny output)
    "mlp1":     (2880, 5760),    # MoE expert up-projection
    "mlp2":     (2880, 2880),    # MoE expert down-projection
    "lm_head":  (2880, 201088),  # unembedding to vocab (extremely fat)
}

# Default token-count (= M dim) sweep spanning decode → long-context prefill.
DEFAULT_LLM_TS: tuple[int, ...] = (1, 256, 2048, 8192, 32768)


def _make_llm_matmul(M: int, K: int, N: int, dtype: torch.dtype,
                     device) -> Callable[[], None]:
    """Non-TE linear-layer GEMM: Y = X @ W.  For fp16/bf16/tf32/fp32."""
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(K, N, dtype=dtype, device=device)
    def f():
        torch.matmul(x, w)
    return f


def _make_llm_matmul_fp32_simt(M: int, K: int, N: int, device) -> Callable[[], None]:
    x = torch.randn(M, K, dtype=torch.float32, device=device)
    w = torch.randn(K, N, dtype=torch.float32, device=device)
    def f():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.matmul(x, w)
    return f


def _make_llm_matmul_tf32_tc(M: int, K: int, N: int, device) -> Callable[[], None]:
    x = torch.randn(M, K, dtype=torch.float32, device=device)
    w = torch.randn(K, N, dtype=torch.float32, device=device)
    def f():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.matmul(x, w)
    return f


def _make_llm_matmul_fp8_te(M: int, K: int, N: int, device) -> Callable[[], None]:
    """FP8 via Transformer Engine te.Linear — same wrapper as build_matmul's
    fp8 path but M/K/N are arbitrary (not forced square)."""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe
    except (ImportError, OSError, RuntimeError) as e:
        raise RuntimeError(
            f"transformer_engine not usable ({type(e).__name__}: {e}); "
            f"run ./install_transformer_engine.sh"
        ) from e
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    linear = te.Linear(K, N, bias=False, params_dtype=torch.float16).to(device)
    fp8_recipe = te_recipe.DelayedScaling(
        fp8_format=te_recipe.Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="max",
    )
    # Warmup 5×: same Blackwell + small-M (T=1 decode) safeguard as the
    # square-matmul builder. If TE's amax-buffer dance is going to trip
    # the CUDA context, it shows up here, where build() can re-raise as
    # a clean RuntimeError that's caught at plan-build time and skips
    # the cell instead of poisoning the whole sweep.
    try:
        for _ in range(5):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                linear(x)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "illegal memory access" in str(e) or "CUDA error" in str(e):
            raise RuntimeError(
                f"transformer_engine fp8_autocast failed during warmup "
                f"on shape (M={M}, K={K}, N={N}) — likely Blackwell + "
                f"small-M (decode-size) issue. CUDA context is now "
                f"unrecoverable in this process. README §8.3.3 has the "
                f"workarounds. Original: {e}"
            ) from e
        raise
    def f():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            linear(x)
    return f


def build_llm_matmul(preset: str, T: int, dtype_label: str, mode: str,
                     device: str | torch.device = "cuda",
                     shapes: dict[str, tuple[int, int]] | None = None) -> BenchSpec:
    """Build one LLM-shape matmul BenchSpec.

    Args:
      preset: key into LLM_SHAPES (or `shapes` if given) — picks (K, N).
      T: token count = M dim.  Sweep this axis to see J/FLOP as a function
         of batch/sequence length.
      dtype_label / mode: same as build_matmul (fp32/simt, tf32/tc,
         fp16/tc, bf16/tc, fp8/te).
      shapes: optional override dict — useful when the caller wants to
         sweep a model different from gpt-oss-120B.
    """
    table = shapes if shapes is not None else LLM_SHAPES
    if preset not in table:
        raise ValueError(f"unknown llm preset {preset!r}; "
                         f"choices: {sorted(table)}")
    K, N = table[preset]
    M = T
    cc = torch.cuda.get_device_capability()
    compute_unit = "Tensor Core"
    emulated = False
    notes = ""

    if (dtype_label, mode) == ("fp32", "simt"):
        fn = _make_llm_matmul_fp32_simt(M, K, N, device)
        compute_unit = "CUDA core"
    elif (dtype_label, mode) == ("tf32", "tc"):
        if cc[0] < 8:
            raise RuntimeError(f"TF32 requires Ampere (sm_80) or newer (this GPU is sm_{cc[0]}{cc[1]})")
        fn = _make_llm_matmul_tf32_tc(M, K, N, device)
    elif (dtype_label, mode) == ("fp16", "tc"):
        fn = _make_llm_matmul(M, K, N, torch.float16, device)
    elif (dtype_label, mode) == ("bf16", "tc"):
        if cc[0] < 8:
            raise RuntimeError("BF16 requires Ampere (sm_80) or newer")
        fn = _make_llm_matmul(M, K, N, torch.bfloat16, device)
    elif (dtype_label, mode) == ("fp8", "te"):
        fn = _make_llm_matmul_fp8_te(M, K, N, device)
        if cc[0] < 9:
            compute_unit = "Tensor Core (FP16 fallback)"
            emulated = True
            notes = "fp8_te on pre-Hopper falls back to FP16 Tensor Core"
    else:
        raise ValueError(f"unknown matmul variant ({dtype_label!r}, {mode!r})")

    flops = 2 * M * N * K
    n_out = M * N
    name = f"llm_{preset}_{dtype_label}_{mode}"
    ws_bytes = (M * K + K * N + M * N) * _dtype_bytes(dtype_label)
    dev_idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
    regime = classify_cache_regime(ws_bytes, get_l2_bytes(dev_idx))
    notes = (notes + f" | llm_preset={preset} shape=({M}x{K})@({K}x{N})").strip(" |")
    if (dtype_label, mode) == ("fp8", "te"):
        path_semantics = ("te_fp16_fallback" if emulated
                          else "native_or_te_fp8_tensorcore")
    else:
        path_semantics = "native_or_standard"
    return BenchSpec(
        name=name, op="matmul_llm", dtype_label=dtype_label,
        shape=(M, K, N), n_elements=n_out,
        flops_per_call=flops, run=fn,
        compute_unit=compute_unit, emulated=emulated,
        cache_regime=regime, path_semantics=path_semantics, notes=notes,
    )


def llm_matmul_footprint_bytes(preset: str, T: int, dtype_label: str,
                               shapes: dict[str, tuple[int, int]] | None = None) -> int:
    """Worst-case HBM footprint (A + B + C) for pre-flight memory budgeting.

    This is exposed as a module-level function so gpu_power_bench.py can
    drop (preset, T) combinations that would exceed the per-cell HBM
    budget BEFORE it tries to allocate them.
    """
    table = shapes if shapes is not None else LLM_SHAPES
    K, N = table[preset]
    b = _dtype_bytes(dtype_label)
    # Three tensors (X, W, Y); fp8_te additionally keeps an fp16 weight
    # shadow for the amax history, so inflate by 1.5× in the fp8 case.
    base = (T * K + K * N + T * N) * b
    if dtype_label == "fp8":
        base = int(base * 1.5) + K * N * 2
    return base


# ============================================================================
# Fused vs Standalone (G11 / P1.4) — measure softmax / gelu / layernorm both
# as standalone PyTorch ops (already in `_BUILDERS` above) AND as part of a
# fused kernel (FlashAttention / linear+gelu / ln+linear). Standalone J/elem
# is dominated by HBM round-trip ; the fused variant amortises HBM across
# the surrounding matmul. Comparison via subtraction (analyze.py decomposition).
#
# Default shapes are taken from openai/gpt-oss-120b/config.json :
#   * attention : B=1, H_q=64, H_kv=8, N_q=N_kv=2048, D_head=64
#                 (full-attention layer ; sliding-window layer N_kv=128 not
#                  measured in phase 1)
#   * MLP       : M=2048, D_in=D_out=2880  (per-expert MoE intermediate ;
#                 1 of top-4 active experts)
# CLI overrides : --attn-shape, --mlp-shape (gpu_power_bench.py).
# Caveat — GPT-OSS uses SiLU/SwiGLU + RMSNorm, not GeLU/LayerNorm. Phase 1
# keeps the user-named ops for structure comparison ; SiLU/RMSNorm phase
# is REVIEW.md G12 / P2.4. See README §3.7.
# ============================================================================

# One-shot guard so the FP8 DPA verification hint prints once per process,
# not on every rebaseline / per-cell rebuild.
_FP8_DPA_VERIFY_HINTED: dict = {}


FUSED_VARIANTS = (
    "attention_flash",        # SDPA flash kernel (fp16/bf16) ; fp8 routes to TE
                              # via _make_attention_flash_fp8 (legacy convenience)
    "attention_flash_te",     # ALWAYS Transformer Engine DotProductAttention.
                              # Used to remove the SDPA-vs-TE backend confounder
                              # in the cross-dtype attention compare. fp8 here
                              # is identical to `attention_flash` fp8 — plan-
                              # builder skips it to avoid duplicate measurement.
    "attention_qkv_matmul",   # Q@K + (Q@K)@V only (softmax replaced by identity) — subtraction baseline
    "linear_gelu",            # gelu(linear(x)+b) — torch.compile epilogue fusion
    "linear_baseline_gelu",   # linear(x)+b only — same shape, no activation — subtraction baseline
    "ln_linear",              # linear(layer_norm(x))+b — torch.compile pre-norm fusion
    "linear_baseline_ln",     # linear(x)+b only — matches ln_linear shape — subtraction baseline
)


def _sdpa_kernel_ctx(enable_flash=True, enable_cudnn=True,
                     enable_math=False, enable_mem_efficient=False):
    """Cross-version SDPA backend selector. PyTorch's API moved from
    `torch.backends.cuda.sdp_kernel` → `torch.nn.attention.sdpa_kernel`.
    Try the new path first, fall back to the legacy one, then to a
    no-op. The benchmark warns if math/mem-efficient runs unintentionally
    (verifiable post-hoc with `fusion_check.py`)."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backends = []
        if enable_flash: backends.append(SDPBackend.FLASH_ATTENTION)
        if enable_cudnn:
            try: backends.append(SDPBackend.CUDNN_ATTENTION)
            except AttributeError: pass
        if enable_math: backends.append(SDPBackend.MATH)
        if enable_mem_efficient: backends.append(SDPBackend.EFFICIENT_ATTENTION)
        return sdpa_kernel(backends)
    except Exception:
        pass
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash, enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


def _try_native_gqa_supported() -> bool:
    """Probe whether SDPA's `enable_gqa` kwarg is available (PyTorch >= 2.5)."""
    try:
        q = torch.zeros(1, 1, 1, 8, dtype=torch.float16, device="cuda")
        torch.nn.functional.scaled_dot_product_attention(q, q, q, enable_gqa=True)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def _make_attention_flash(B: int, H_q: int, H_kv: int, N_q: int, N_kv: int,
                          D_head: int, dtype: torch.dtype, device,
                          causal: bool = False) -> Callable[[], None]:
    """Full fused attention via SDPA + FlashAttention backend.

    GQA path : K/V have H_kv heads, Q has H_q. PyTorch >= 2.5 supports this
    natively via `enable_gqa=True` ; older versions need manual
    `repeat_interleave`. The fusion_check.py PoC verifies which path is
    available before the sweep.
    """
    q = torch.randn(B, H_q,  N_q, D_head, dtype=dtype, device=device)
    k = torch.randn(B, H_kv, N_kv, D_head, dtype=dtype, device=device)
    v = torch.randn(B, H_kv, N_kv, D_head, dtype=dtype, device=device)
    enable_gqa = (H_q != H_kv) and _try_native_gqa_supported()
    if H_q != H_kv and not enable_gqa:
        groups = H_q // H_kv
        k = k.repeat_interleave(groups, dim=1).contiguous()
        v = v.repeat_interleave(groups, dim=1).contiguous()
    sdpa_kwargs = {"is_causal": causal}
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True

    def f():
        with _sdpa_kernel_ctx(enable_flash=True, enable_cudnn=True,
                              enable_math=False, enable_mem_efficient=False):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    return f


def _make_attention_flash_fp8(B: int, H_q: int, H_kv: int,
                               N_q: int, N_kv: int, D_head: int,
                               device, causal: bool = False
                               ) -> tuple[Callable[[], None], bool, bool]:
    """FP8 fused attention via Transformer Engine + `fp8_autocast`.

    Returns (run_fn, emulated_flag, fp8_dpa_recipe_accepted) :
      * emulated_flag : True when GPU is pre-Hopper (no native FP8 attn)
        and TE silently falls back to a half-precision code path.
      * fp8_dpa_recipe_accepted : True when TE's DelayedScaling accepted
        `fp8_dpa=True` (i.e. TE >= 1.x — the FP8 DPA cuDNN sub-backend 2
        is REQUESTED). False means TE rejected the kwarg → recipe fell
        back to defaults and FP8 DPA backend is NOT guaranteed (operator
        should re-run with NVTE_DEBUG=1 to verify backend selection).

    Implementation : `te.DotProductAttention` accepts (Q, K, V) tensors in
    `bshd` (Batch-Seq-Head-Dim) layout and dispatches to cuDNN / FlashAttention
    internally ; under `te.fp8_autocast(enabled=True, recipe=DelayedScaling
    (E4M3))` the whole attention chain (Q@Kᵀ → softmax → P@V) runs in FP8 on
    Hopper. GQA is handled via `num_gqa_groups=H_kv` (TE param name).

    No subtraction baseline yet — TE does not expose a public batched-fp8-gemm
    API that we can use to construct an "attention_qkv_matmul_fp8" baseline
    cleanly. So fp8 is INFORMATIONAL (full attention energy only) for now ;
    decomposition (matmul + softmax-residual) is fp16/bf16 only. See README
    §3.7 / TestCases A.5 / REVIEW.md G12.
    """
    _install_hint = ("Run ./install_transformer_engine.sh — the same script "
                     "used by matmul_fp8_te.")
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe
    except (ImportError, OSError, RuntimeError) as e:
        raise RuntimeError(
            f"transformer_engine not usable for fp8 attention "
            f"({type(e).__name__}: {e}). {_install_hint}") from e

    # `te.DotProductAttention` lives at the top of the te.pytorch namespace
    # in TE >= 1.0. Older versions exposed it under te.attention only.
    DPA = getattr(te, "DotProductAttention", None)
    if DPA is None:
        try:
            from transformer_engine.pytorch.attention import DotProductAttention as DPA
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"te.DotProductAttention unavailable (TE version too old) "
                f"({type(e).__name__}: {e}). Upgrade TE >= 1.0 to use "
                f"fp8 fused attention.") from e

    cc = torch.cuda.get_device_capability()
    is_hopper = cc[0] >= 9
    emulated = not is_hopper   # TE on pre-Hopper falls back to FP16 attention

    # bshd layout : (B, S, H, D). TE prefers this for attention.
    q = torch.randn(B, N_q,  H_q,  D_head, dtype=torch.float16, device=device)
    k = torch.randn(B, N_kv, H_kv, D_head, dtype=torch.float16, device=device)
    v = torch.randn(B, N_kv, H_kv, D_head, dtype=torch.float16, device=device)

    try:
        dpa = DPA(
            num_attention_heads=H_q,
            kv_channels=D_head,
            num_gqa_groups=H_kv,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal" if causal else "no_mask",
        ).to(device)
    except TypeError:
        # Some TE versions don't accept all of these kwargs (older API).
        # Try the minimal signature.
        try:
            dpa = DPA(
                num_attention_heads=H_q,
                kv_channels=D_head,
                attention_dropout=0.0,
            ).to(device)
        except Exception as e:
            raise RuntimeError(
                f"te.DotProductAttention construction failed "
                f"({type(e).__name__}: {e}); your TE version may have a "
                f"different API. Upgrade TE or skip fp8 attention.") from e

    # FP8 attention recipe — `fp8_dpa=True` is REQUIRED to select cuDNN
    # FusedAttention sub-backend 2 (the FP8 DPA path). Without it the
    # default DelayedScaling does NOT route attention's two GEMMs through
    # FP8 — the dtype label would be misleading. fp8_dpa landed in TE
    # 1.x ; older versions silently ignore the kwarg, so we try-except.
    #
    # Verification (operator should run once on H100) :
    #   NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=1 \
    #     python3 gpu_power_bench.py --fused-dtypes fp8 ...
    # Look for "FusedAttention sub-backend 2" / "FP8 DPA" in the log.
    # `NVTE_FUSED_ATTN_BACKEND=2` can force the FP8 backend if multiple
    # are available.
    _recipe_kwargs = dict(
        fp8_format=te_recipe.Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="max",
    )
    try:
        fp8_recipe = te_recipe.DelayedScaling(fp8_dpa=True, **_recipe_kwargs)
        fp8_dpa_recipe_accepted = True
    except TypeError as e:
        # TE < 1.x : fp8_dpa kwarg not yet present. Fall back gracefully
        # but warn loudly that FP8 DPA backend may NOT be selected.
        print(f"[fused-attn-fp8] WARN te_recipe.DelayedScaling rejected "
              f"`fp8_dpa=True` ({e}). FP8 DPA cuDNN sub-backend 2 may NOT "
              f"be active — measurement could be FP16/BF16 attention "
              f"masquerading as fp8. Upgrade Transformer Engine >= 1.x "
              f"to enable explicit FP8 DPA. Recipe falling back to "
              f"DelayedScaling defaults.")
        fp8_recipe = te_recipe.DelayedScaling(**_recipe_kwargs)
        fp8_dpa_recipe_accepted = False

    # Warmup 5× — same Blackwell + small-N safeguard as matmul_fp8_te.
    # If TE's amax-buffer dance is going to trip the CUDA context, surface
    # it here as a clean RuntimeError so build() can skip the cell.
    try:
        for _ in range(5):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                dpa(q, k, v)
        torch.cuda.synchronize()
    except RuntimeError as e:
        msg = str(e)
        if "illegal memory access" in msg or "CUDA error" in msg:
            raise RuntimeError(
                f"te.DotProductAttention fp8 warmup failed on shape "
                f"(B={B}, H_q={H_q}, H_kv={H_kv}, N_q={N_q}, N_kv={N_kv}, "
                f"D={D_head}). CUDA context unrecoverable in this process. "
                f"Original: {e}") from e
        raise

    # One-time verification hint — operator should run once with NVTE_DEBUG
    # to confirm FP8 DPA sub-backend 2 was actually selected. We can't
    # introspect that ourselves from Python (TE doesn't expose it).
    if not _FP8_DPA_VERIFY_HINTED.get("hinted"):
        print(f"[fused-attn-fp8] To VERIFY FP8 DPA backend was selected "
              f"(cuDNN sub-backend 2), re-run once with :\n"
              f"   NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=1 \\\n"
              f"     ./run_bench.sh --suite quick --fused-dtypes fp8 ...\n"
              f" and grep the log for 'sub-backend 2' / 'FP8 DPA'. If "
              f"sub-backend 2 is NOT chosen, force it with "
              f"NVTE_FUSED_ATTN_BACKEND=2.")
        _FP8_DPA_VERIFY_HINTED["hinted"] = True

    def f():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            dpa(q, k, v)
    return f, emulated, fp8_dpa_recipe_accepted


def _make_attention_flash_te_halfprec(B: int, H_q: int, H_kv: int,
                                       N_q: int, N_kv: int, D_head: int,
                                       dtype: torch.dtype, device,
                                       causal: bool = False
                                       ) -> Callable[[], None]:
    """fp16 / bf16 attention via Transformer Engine `DotProductAttention`
    WITHOUT `fp8_autocast`. Same TE backend as the fp8 path so the
    `attention_flash_te` family across (fp16, bf16, fp8) constitutes a
    backend-controlled pure-dtype comparison.

    Existing `attention_flash` for fp16/bf16 uses PyTorch SDPA's flash
    backend (fast but different code path), and existing `attention_flash`
    for fp8 already uses TE. So adding this halfprec TE variant gives
    operators :
      * SDPA (fp16) vs TE (fp16) → backend effect (this plot)
      * TE (fp16) vs TE (fp8)    → pure dtype effect (no backend mix)
    """
    _install_hint = ("Run ./install_transformer_engine.sh — same script "
                     "used by matmul_fp8_te.")
    try:
        import transformer_engine.pytorch as te
    except (ImportError, OSError, RuntimeError) as e:
        raise RuntimeError(
            f"transformer_engine not usable for attention_flash_te "
            f"({type(e).__name__}: {e}). {_install_hint}") from e

    DPA = getattr(te, "DotProductAttention", None)
    if DPA is None:
        try:
            from transformer_engine.pytorch.attention import DotProductAttention as DPA
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"te.DotProductAttention unavailable (TE version too old). "
                f"Upgrade TE >= 1.0. ({type(e).__name__}: {e})") from e

    # bshd layout (TE preferred).
    q = torch.randn(B, N_q,  H_q,  D_head, dtype=dtype, device=device)
    k = torch.randn(B, N_kv, H_kv, D_head, dtype=dtype, device=device)
    v = torch.randn(B, N_kv, H_kv, D_head, dtype=dtype, device=device)

    try:
        dpa = DPA(
            num_attention_heads=H_q,
            kv_channels=D_head,
            num_gqa_groups=H_kv,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal" if causal else "no_mask",
        ).to(device)
    except TypeError:
        dpa = DPA(
            num_attention_heads=H_q,
            kv_channels=D_head,
            attention_dropout=0.0,
        ).to(device)

    # Warmup 5× — no fp8_autocast, just plain half-precision through TE.
    try:
        for _ in range(5):
            dpa(q, k, v)
        torch.cuda.synchronize()
    except RuntimeError as e:
        msg = str(e)
        if "illegal memory access" in msg or "CUDA error" in msg:
            raise RuntimeError(
                f"te.DotProductAttention halfprec warmup failed on shape "
                f"(B={B}, H_q={H_q}, H_kv={H_kv}, N_q={N_q}, N_kv={N_kv}, "
                f"D={D_head}). Original: {e}") from e
        raise

    def f():
        dpa(q, k, v)
    return f


def _make_attention_qkv_matmul(B: int, H_q: int, H_kv: int, N_q: int, N_kv: int,
                               D_head: int, dtype: torch.dtype, device,
                               causal: bool = False) -> Callable[[], None]:
    """Subtraction baseline for `attention_flash` :  Q@Kᵀ then result@V,
    NO softmax / scaling / masking. Same input shape, same FLOPs (4·B·H_q·
    N_q·N_kv·D excluding the softmax FLOPs which are tiny in standalone).

    Energy attribution :
        J(attention_flash) − J(attention_qkv_matmul) ≈
            J of (online streaming softmax + max/sum/exp/scale/rescale +
                  whatever masking the flash kernel applies)
    """
    q = torch.randn(B, H_q,  N_q, D_head, dtype=dtype, device=device)
    k = torch.randn(B, H_kv, N_kv, D_head, dtype=dtype, device=device)
    v = torch.randn(B, H_kv, N_kv, D_head, dtype=dtype, device=device)
    if H_q != H_kv:
        groups = H_q // H_kv
        k = k.repeat_interleave(groups, dim=1).contiguous()
        v = v.repeat_interleave(groups, dim=1).contiguous()
    # We use `torch.matmul` chains : Q @ K.T → (Q@K.T) @ V. NO softmax.
    # The shape of Q@K.T is (B, H_q, N_q, N_kv).
    def f():
        s = torch.matmul(q, k.transpose(-1, -2))
        torch.matmul(s, v)
    return f


# ---------- helper : torch.compile fusion with TE fallback ------------------

_COMPILE_FALLBACK_WARNED: dict[str, bool] = {}


def _compile_or_eager(fn_eager, label: str, force: str = "auto"):
    """Wrap fn_eager with torch.compile ; fall back to eager + warning if
    the compile / first call raises. `force` ∈ {"auto", "compile", "eager"}.

    Returns (callable, backend_label). backend_label ∈ {"compile", "eager"}.
    """
    if force == "eager":
        return fn_eager, "eager"
    if force in ("auto", "compile"):
        try:
            compiled = torch.compile(fn_eager, mode="reduce-overhead", dynamic=False)
            return compiled, "compile"
        except Exception as e:
            if not _COMPILE_FALLBACK_WARNED.get(label):
                _COMPILE_FALLBACK_WARNED[label] = True
                print(f"[fused] torch.compile failed for {label} "
                      f"({type(e).__name__}: {e}) — falling back to eager. "
                      f"Energy will include kernel-launch overhead per op "
                      f"(NOT representative of fused).")
            return fn_eager, "eager"
    return fn_eager, "eager"


def _make_linear_gelu(M: int, D_in: int, D_out: int, dtype: torch.dtype,
                      device, force_backend: str = "auto") -> tuple[Callable[[], None], str]:
    """`gelu(linear(x) + b)` fused via torch.compile. Returns (run_fn, backend).
    backend ∈ {"compile", "eager"} — caller can flag emulated=1 when "eager"."""
    x = torch.randn(M, D_in, dtype=dtype, device=device)
    W = torch.randn(D_in, D_out, dtype=dtype, device=device) * (1.0 / D_in ** 0.5)
    b = torch.zeros(D_out, dtype=dtype, device=device)

    def fn_eager(x=x):
        torch.nn.functional.gelu(x @ W + b, approximate="tanh")

    fn_compiled, backend = _compile_or_eager(fn_eager, "linear_gelu", force_backend)
    # Warm the compile so the first sampled call doesn't include compile time.
    try:
        for _ in range(3):
            fn_compiled()
        torch.cuda.synchronize()
    except Exception as e:
        # Fallback if compiled fn fails on first call.
        print(f"[fused] linear_gelu compiled call failed ({e}); using eager.")
        fn_compiled, backend = fn_eager, "eager"
        for _ in range(3): fn_compiled()
        torch.cuda.synchronize()
    return fn_compiled, backend


def _make_linear_baseline_gelu(M: int, D_in: int, D_out: int, dtype: torch.dtype,
                               device, force_backend: str = "auto") -> tuple[Callable[[], None], str]:
    """Subtraction baseline for `linear_gelu` : pure `linear(x)+b`, no GeLU.
    Compiled with the same backend so kernel-launch overhead cancels in subtraction."""
    x = torch.randn(M, D_in, dtype=dtype, device=device)
    W = torch.randn(D_in, D_out, dtype=dtype, device=device) * (1.0 / D_in ** 0.5)
    b = torch.zeros(D_out, dtype=dtype, device=device)

    def fn_eager(x=x):
        _ = x @ W + b

    fn_compiled, backend = _compile_or_eager(fn_eager, "linear_baseline_gelu", force_backend)
    try:
        for _ in range(3):
            fn_compiled()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[fused] linear_baseline_gelu compiled call failed ({e}); using eager.")
        fn_compiled, backend = fn_eager, "eager"
        for _ in range(3): fn_compiled()
        torch.cuda.synchronize()
    return fn_compiled, backend


def _make_ln_linear(M: int, D_in: int, D_out: int, dtype: torch.dtype,
                    device, force_backend: str = "auto") -> tuple[Callable[[], None], str]:
    """`linear(layer_norm(x)) + b` fused via torch.compile. Pre-norm pattern."""
    x = torch.randn(M, D_in, dtype=dtype, device=device)
    g = torch.ones(D_in,  dtype=dtype, device=device)
    bln = torch.zeros(D_in, dtype=dtype, device=device)
    W = torch.randn(D_in, D_out, dtype=dtype, device=device) * (1.0 / D_in ** 0.5)
    blin = torch.zeros(D_out, dtype=dtype, device=device)

    def fn_eager(x=x):
        h = torch.nn.functional.layer_norm(x, (D_in,), g, bln, eps=1e-5)
        _ = h @ W + blin

    fn_compiled, backend = _compile_or_eager(fn_eager, "ln_linear", force_backend)
    try:
        for _ in range(3):
            fn_compiled()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[fused] ln_linear compiled call failed ({e}); using eager.")
        fn_compiled, backend = fn_eager, "eager"
        for _ in range(3): fn_compiled()
        torch.cuda.synchronize()
    return fn_compiled, backend


def _make_linear_baseline_ln(M: int, D_in: int, D_out: int, dtype: torch.dtype,
                             device, force_backend: str = "auto") -> tuple[Callable[[], None], str]:
    """Subtraction baseline for `ln_linear` : pure `linear(x)+b`, no LN.
    Same shape as ln_linear; compiled with the same backend."""
    return _make_linear_baseline_gelu(M, D_in, D_out, dtype, device, force_backend)


# ---------- public entry point ----------------------------------------------

def build_fused(variant: str, dtype_label: str,
                attn_shape: tuple[int, int, int, int, int, int] = (1, 64, 8, 2048, 2048, 64),
                mlp_shape: tuple[int, int, int] = (2048, 2880, 2880),
                causal: bool = False,
                fusion_backend: str = "auto",
                device: str | torch.device = "cuda") -> BenchSpec:
    """Build one fused-vs-standalone variant (G11 / P1.4).

    Args:
      variant: one of FUSED_VARIANTS.
      dtype_label: "fp16" | "bf16". fp8 fused is REVIEW.md G12 / P2.4.
      attn_shape: (B, H_q, H_kv, N_q, N_kv, D_head) for attention variants.
      mlp_shape:  (M, D_in, D_out) for MLP / LN variants.
      causal: if True, attention uses causal mask (halves softmax cost).
      fusion_backend: "auto" | "compile" | "eager".

    Returns BenchSpec ; `emulated=1` indicates fusion did NOT take effect
    (eager fallback) — analyze.py honours this when computing residuals.
    """
    if variant not in FUSED_VARIANTS:
        raise ValueError(f"unknown fused variant {variant!r}; "
                         f"choices: {FUSED_VARIANTS}")
    if dtype_label not in ("fp16", "bf16", "fp8"):
        raise ValueError(f"fused variants only support fp16/bf16/fp8 (got {dtype_label!r}); "
                         f"other dtypes are G12/P2.4 follow-up")
    if dtype_label == "fp8" and variant != "attention_flash":
        raise ValueError(
            f"fp8 fused only supports `attention_flash` for now (got "
            f"{variant!r}). fp8 needs a baseline (e.g. attention_qkv_matmul "
            f"in fp8) for full decomposition — that requires a public "
            f"batched-fp8-gemm API which TE doesn't expose. fp8 MLP fused "
            f"variants (linear_gelu / ln_linear) are also pending. "
            f"See REVIEW.md G12 / P2.4.")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp8":  torch.float16}[dtype_label]   # fp8 stores as fp16 ; TE autocasts internally
    notes = ""
    backend_used = "tc"  # default for fused — output reaches Tensor Cores
    emulated = False

    if variant in ("attention_flash", "attention_flash_te", "attention_qkv_matmul"):
        B, H_q, H_kv, N_q, N_kv, D_head = attn_shape
        if H_q % H_kv != 0:
            raise ValueError(f"attention: H_q ({H_q}) must be divisible by H_kv ({H_kv})")
        if dtype_label == "fp8":
            # FP8 path : TE DotProductAttention + fp8_autocast.
            # Same builder is used for both `attention_flash` and
            # `attention_flash_te` — fp8 always goes through TE so the
            # two variants are identical at fp8. plan-builder skips
            # the duplicate.
            fn, emulated, fp8_dpa_accepted = _make_attention_flash_fp8(
                B, H_q, H_kv, N_q, N_kv, D_head, device, causal=causal)
            notes_extra = (f" ; fp8_dpa_requested={'true' if fp8_dpa_accepted else 'false'}"
                           f" (verify cuDNN sub-backend 2 selection with "
                           f"NVTE_DEBUG=1 NVTE_FUSED_ATTN_BACKEND=2)")
            notes = ("fp8 FlashAttention via te.DotProductAttention + "
                     "fp8_autocast(E4M3)")
            if emulated:
                notes += " (pre-Hopper FP16 fallback)"
            notes += notes_extra
        elif variant == "attention_flash_te":
            # NEW : fp16/bf16 routed through TE DotProductAttention (no
            # fp8_autocast) — backend-controlled pure-dtype comparison
            # against the fp8 row of the same variant.
            fn = _make_attention_flash_te_halfprec(
                B, H_q, H_kv, N_q, N_kv, D_head, dtype, device, causal=causal)
            notes = ("FlashAttention via Transformer Engine DPA "
                     "(backend-controlled vs SDPA `attention_flash`)")
        else:
            builder = (_make_attention_flash if variant == "attention_flash"
                       else _make_attention_qkv_matmul)
            fn = builder(B, H_q, H_kv, N_q, N_kv, D_head, dtype, device, causal=causal)
            if variant == "attention_flash":
                notes = "FlashAttention via SDPA (forced flash/cudnn backend)"
            else:
                notes = "subtraction baseline : Q@Kᵀ + (Q@Kᵀ)@V, no softmax"
        # FLOPs : Q@Kᵀ = 2·B·H_q·N_q·N_kv·D , (Q@Kᵀ)@V = 2·B·H_q·N_q·N_kv·D
        # → 4·B·H_q·N_q·N_kv·D total (softmax FLOPs negligible here).
        flops = 4 * B * H_q * N_q * N_kv * D_head
        n_out = B * H_q * N_q * D_head
        shape = attn_shape
        compute_unit = ("Tensor Core (FP8)" if dtype_label == "fp8" and not emulated
                        else "Tensor Core (FP16 fallback)" if emulated
                        else "Tensor Core")
        if causal: notes += " ; causal=1"

    elif variant in ("linear_gelu", "linear_baseline_gelu",
                     "ln_linear", "linear_baseline_ln"):
        M, D_in, D_out = mlp_shape
        builder_map = {
            "linear_gelu":          _make_linear_gelu,
            "linear_baseline_gelu": _make_linear_baseline_gelu,
            "ln_linear":            _make_ln_linear,
            "linear_baseline_ln":   _make_linear_baseline_ln,
        }
        fn, backend_used = builder_map[variant](M, D_in, D_out, dtype, device,
                                                fusion_backend)
        # FLOPs : linear = 2·M·D_in·D_out + bias add (M·D_out).
        # Activation / norm FLOPs are tiny (~8·M·D_out vs 2·M·D_in·D_out
        # which dominates when D_in is large) — bundled into total flops
        # as "compute" but not separately costed.
        flops = 2 * M * D_in * D_out
        if variant == "linear_gelu":
            flops += 8 * M * D_out         # tanh-approx GeLU
        elif variant == "ln_linear":
            flops += 8 * M * D_in          # LN over D_in
        n_out = M * D_out
        shape = mlp_shape
        compute_unit = "Tensor Core"
        notes = f"backend={backend_used}"
        emulated = (backend_used == "eager")
        if emulated:
            notes += " (FUSION FAILED — eager fallback, residual will inflate)"

    else:
        raise AssertionError(f"unhandled fused variant {variant!r}")

    name = variant   # CSV `variant` column = exactly the variant key
    # cache_regime — fused inputs typically span multiple L2-sized
    # tensors (Q+K+V or x+W+y). Compute working set ; classify.
    if "attention" in variant:
        B, H_q, H_kv, N_q, N_kv, D_head = attn_shape
        ws = (B * (H_q + 2 * H_kv) * N_kv * D_head + B * H_q * N_q * D_head) * 2  # bf16/fp16
    else:
        M, D_in, D_out = mlp_shape
        ws = (M * D_in + D_in * D_out + M * D_out) * 2
    dev_idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
    regime = classify_cache_regime(ws, get_l2_bytes(dev_idx))

    # Path semantics for fused : fp8 attention via TE = native (or
    # fallback on pre-Hopper). Other dtypes / non-attention variants
    # = standard. Linear+gelu / ln+linear that fall back to eager are
    # NOT "emulated cast" — the activation/norm just isn't fused. We
    # leave those as native_or_standard ; the `emulated` flag already
    # marks them for plot hatching.
    if "attention" in variant and dtype_label == "fp8":
        path_semantics = ("te_fp16_fallback" if emulated
                          else "native_or_te_fp8_tensorcore")
    else:
        path_semantics = "native_or_standard"
    return BenchSpec(
        name=name, op=variant, dtype_label=dtype_label,
        shape=tuple(shape), n_elements=int(n_out),
        flops_per_call=int(flops), run=fn,
        compute_unit=compute_unit, emulated=emulated,
        cache_regime=regime, path_semantics=path_semantics, notes=notes,
    )


# ---------- L2/SRAM resident traffic probes ----------------------------------
# These probes deliberately avoid PyTorch high-level ops.  A small CUDA
# extension launches one kernel whose INNER loop repeats the same L2-sized
# window many times, so NVML sees a multi-Joule signal instead of a single
# sub-mJ L2 pass.  The reported coefficient is an L2-hit *traffic path* energy
# (L2 array + slice/fabric + datapath), not isolated SRAM bit-cell energy.

_L2_OPS = ("reg_spin", "l2_read_hit", "l2_write_hit", "l2_copy_hit", "l2_sliding_delta")
_L2_EXT = None


def _cuda_target_include_paths() -> list[str]:
    """Return extra CUDA include dirs needed by conda-packaged toolkits.

    Recent conda CUDA packages can place CCCL headers such as `nv/target`
    under targets/x86_64-linux/include instead of the top-level include dir.
    torch.utils.cpp_extension adds CUDA_HOME/include, but not always this
    target-specific path.
    """
    import os
    import sys
    from pathlib import Path

    roots = []
    for key in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    roots.append(sys.prefix)

    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        target_include = Path(root) / "targets" / "x86_64-linux" / "include"
        candidates = [
            target_include,
            target_include / "cccl",
        ]
        for path in candidates:
            has_cuda_cccl = (
                (path / "nv" / "target").exists()
                or (path / "thrust" / "complex.h").exists()
                or (path / "cub" / "cub.cuh").exists()
            )
            if not has_cuda_cccl:
                continue
            s = str(path)
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _cuda_target_library_paths() -> list[str]:
    """Return CUDA library dirs needed by conda-packaged toolkits.

    Some conda CUDA installs expose the usable runtime library under
    targets/x86_64-linux/lib while the top-level lib/libcudart.so symlink can
    be absent or stale.  torch's extension linker still adds the top-level lib
    dir, so prepend the target-specific lib dir through LIBRARY_PATH before
    building the L2 extension.
    """
    import os
    import sys
    from pathlib import Path

    roots = []
    for key in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    roots.append(sys.prefix)

    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        candidates = [
            Path(root) / "targets" / "x86_64-linux" / "lib",
            Path(root) / "lib64",
            Path(root) / "lib",
        ]
        for path in candidates:
            if not (
                (path / "libcudart.so").exists()
                or (path / "libcudart.so.12").exists()
                or (path / "libcudart_static.a").exists()
            ):
                continue
            s = str(path)
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _prepend_env_path(name: str, paths: list[str]) -> None:
    """Prepend unique paths to a colon-separated environment variable."""
    import os

    existing = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    merged: list[str] = []
    seen: set[str] = set()
    for path in [*paths, *existing]:
        if path and path not in seen:
            merged.append(path)
            seen.add(path)
    if merged:
        os.environ[name] = os.pathsep.join(merged)


def _l2_extension():
    """Compile/load the CUDA extension used by build_l2_probe()."""
    global _L2_EXT
    if _L2_EXT is not None:
        return _L2_EXT
    from torch.utils.cpp_extension import load_inline

    _prepend_env_path("LIBRARY_PATH", _cuda_target_library_paths())

    cpp_src = r'''
#include <torch/extension.h>

void l2_reg_spin_launcher(torch::Tensor out, long n_words, int repeat_inner, int block_size);
void l2_read_hit_launcher(torch::Tensor x, torch::Tensor out, long n_words, int repeat_inner, int block_size);
void l2_write_hit_launcher(torch::Tensor y, long n_words, int repeat_inner, int block_size);
void l2_copy_hit_launcher(torch::Tensor x, torch::Tensor y, long n_words, int repeat_inner, int block_size);
void l2_sliding_delta_launcher(torch::Tensor x, torch::Tensor out, long window_words,
                               long cold_words, long delta_words, int repeat_inner, int block_size);
void l2_reset_persisting_launcher();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reg_spin", &l2_reg_spin_launcher, "register/control-loop baseline");
  m.def("read_hit", &l2_read_hit_launcher, "L2 resident read-hit probe");
  m.def("write_hit", &l2_write_hit_launcher, "L2 resident store-hit probe");
  m.def("copy_hit", &l2_copy_hit_launcher, "L2 resident copy-hit probe");
  m.def("sliding_delta", &l2_sliding_delta_launcher, "sliding-window L2/HBM delta probe");
  m.def("reset_persisting_l2", &l2_reset_persisting_launcher, "cudaCtxResetPersistingL2Cache wrapper");
}
'''

    cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_U32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be torch.int32")
#define CUDA_CHECK(call) do { cudaError_t err = (call); TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err)); } while (0)

static inline int grid_for(long n_words, int block_size) {
  long blocks = (n_words + block_size - 1) / block_size;
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  return (int)blocks;
}

__global__ void reg_spin_kernel(uint32_t* out, long n_words, int repeat_inner) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  uint32_t acc = (uint32_t)(tid + 0x9e3779b9u);
  for (int r = 0; r < repeat_inner; ++r) {
    for (long i = tid; i < n_words; i += stride) {
      // Similar loop/control shape to the memory probes, but no data traffic.
      acc ^= (uint32_t)i + 0x7f4a7c15u + (acc << 6) + (acc >> 2);
    }
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

__global__ void read_hit_kernel(const uint32_t* __restrict__ x, uint32_t* out,
                                long n_words, int repeat_inner) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  // One in-kernel warm pass helps populate L2.  Measurement metadata excludes it;
  // the regression intercept absorbs this one-time fill cost.
  for (long i = tid; i < n_words; i += stride) acc += __ldcg(x + i);
  for (int r = 0; r < repeat_inner; ++r) {
    for (long i = tid; i < n_words; i += stride) acc += __ldcg(x + i);
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

__global__ void write_hit_kernel(uint32_t* y, long n_words, int repeat_inner) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  uint32_t v = (uint32_t)(tid ^ 0xa5a5a5a5u);
  // Allocate/warm target lines before the measured repeated stores.
  for (long i = tid; i < n_words; i += stride) y[i] = v;
  for (int r = 0; r < repeat_inner; ++r) {
    v += (uint32_t)r + 1u;
    for (long i = tid; i < n_words; i += stride) y[i] = v;
  }
}

__global__ void copy_hit_kernel(const uint32_t* __restrict__ x, uint32_t* y,
                                long n_words, int repeat_inner) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (long i = tid; i < n_words; i += stride) {
    uint32_t v = __ldcg(x + i);
    y[i] = v;
    acc += v;
  }
  for (int r = 0; r < repeat_inner; ++r) {
    for (long i = tid; i < n_words; i += stride) {
      uint32_t v = __ldcg(x + i);
      y[i] = v + (uint32_t)r;
      acc += v;
    }
  }
  if (threadIdx.x == 0) y[blockIdx.x % n_words] = acc;
}

__global__ void sliding_delta_kernel(const uint32_t* __restrict__ x, uint32_t* out,
                                     long window_words, long cold_words,
                                     long delta_words, int repeat_inner) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  long max_start = (cold_words - window_words) > 1L ? (cold_words - window_words) : 1L;
  for (int r = 0; r < repeat_inner; ++r) {
    long start = (delta_words <= 0) ? 0 : ((long)r * delta_words) % max_start;
    for (long i = tid; i < window_words; i += stride) {
      acc += __ldcg(x + start + i);
    }
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

void l2_reg_spin_launcher(torch::Tensor out, long n_words, int repeat_inner, int block_size) {
  CHECK_CUDA(out); CHECK_U32(out);
  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = grid_for(n_words, block_size);
  reg_spin_kernel<<<grid, block_size, 0, stream>>>(reinterpret_cast<uint32_t*>(out.data_ptr<int>()), n_words, repeat_inner);
  CUDA_CHECK(cudaGetLastError());
}

void l2_read_hit_launcher(torch::Tensor x, torch::Tensor out, long n_words, int repeat_inner, int block_size) {
  CHECK_CUDA(x); CHECK_CUDA(out); CHECK_U32(x); CHECK_U32(out);
  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = grid_for(n_words, block_size);
  read_hit_kernel<<<grid, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(x.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(out.data_ptr<int>()), n_words, repeat_inner);
  CUDA_CHECK(cudaGetLastError());
}

void l2_write_hit_launcher(torch::Tensor y, long n_words, int repeat_inner, int block_size) {
  CHECK_CUDA(y); CHECK_U32(y);
  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = grid_for(n_words, block_size);
  write_hit_kernel<<<grid, block_size, 0, stream>>>(reinterpret_cast<uint32_t*>(y.data_ptr<int>()), n_words, repeat_inner);
  CUDA_CHECK(cudaGetLastError());
}

void l2_copy_hit_launcher(torch::Tensor x, torch::Tensor y, long n_words, int repeat_inner, int block_size) {
  CHECK_CUDA(x); CHECK_CUDA(y); CHECK_U32(x); CHECK_U32(y);
  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = grid_for(n_words, block_size);
  copy_hit_kernel<<<grid, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(x.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(y.data_ptr<int>()), n_words, repeat_inner);
  CUDA_CHECK(cudaGetLastError());
}

void l2_sliding_delta_launcher(torch::Tensor x, torch::Tensor out, long window_words,
                               long cold_words, long delta_words, int repeat_inner, int block_size) {
  CHECK_CUDA(x); CHECK_CUDA(out); CHECK_U32(x); CHECK_U32(out);
  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = grid_for(window_words, block_size);
  sliding_delta_kernel<<<grid, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(x.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(out.data_ptr<int>()), window_words, cold_words, delta_words, repeat_inner);
  CUDA_CHECK(cudaGetLastError());
}

void l2_reset_persisting_launcher() {
  CUDA_CHECK(cudaCtxResetPersistingL2Cache());
}
'''
    _L2_EXT = load_inline(
        name="gpu_power_l2_probe_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=None,
        extra_include_paths=_cuda_target_include_paths(),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _L2_EXT


def build_l2_probe(
    op: str,
    dtype_label: str,
    working_set_bytes: int,
    repeat_inner: int,
    delta_bytes: int = 0,
    cold_pool_bytes: int = 0,
    device: str | torch.device = "cuda",
    use_persisting_l2: bool = False,
    block_size: int = 256,
) -> BenchSpec:
    """Build one custom-kernel L2/SRAM traffic probe.

    The logical traffic counters in `extra` are per *single spec.run() call*;
    gpu_power_bench.py multiplies them by the measured outer iteration count
    before writing CSV.  The unit under test is an L2-hit traffic path, not an
    isolated SRAM cell.
    """
    if op not in _L2_OPS:
        raise ValueError(f"unknown L2 op {op!r} (choices: {_L2_OPS})")
    if dtype_label not in ("uint32", "fp32"):
        raise ValueError("L2 probes currently use uint32/fp32-sized 4B words only")
    if working_set_bytes <= 0:
        raise ValueError("working_set_bytes must be positive")
    if repeat_inner <= 0:
        raise ValueError("repeat_inner must be positive")

    dev = torch.device(device)
    word_bytes = 4
    n_words = max(1, int(working_set_bytes) // word_bytes)
    working_set_bytes = n_words * word_bytes
    delta_bytes = max(0, int(delta_bytes))
    delta_bytes = min(delta_bytes, working_set_bytes)
    delta_words = delta_bytes // word_bytes
    cold_pool_bytes = int(cold_pool_bytes or working_set_bytes)
    if op == "l2_sliding_delta":
        cold_pool_bytes = max(cold_pool_bytes, working_set_bytes + max(delta_bytes, 1))
    cold_words = max(n_words, cold_pool_bytes // word_bytes)
    cold_pool_bytes = cold_words * word_bytes

    ext = _l2_extension()
    out_words = min(65535, max(1, (n_words + block_size - 1) // block_size))
    out = torch.empty(out_words, dtype=torch.int32, device=dev)

    # int32 is intentional: traffic path is 4B words; dtype_label is metadata.
    x = y = None
    if op in ("l2_read_hit", "l2_copy_hit", "l2_sliding_delta"):
        alloc_words = cold_words if op == "l2_sliding_delta" else n_words
        x = torch.arange(alloc_words, dtype=torch.int32, device=dev)
    if op in ("l2_write_hit", "l2_copy_hit"):
        y = torch.empty(n_words, dtype=torch.int32, device=dev)

    # Reset persisting lines from previous probes.  Actual access-policy-window
    # set-aside is deliberately not claimed here; cache policy is controlled by
    # __ldcg and validated later with Nsight Compute counters.
    try:
        ext.reset_persisting_l2()
    except Exception:
        pass

    def f():
        if op == "reg_spin":
            ext.reg_spin(out, n_words, int(repeat_inner), int(block_size))
        elif op == "l2_read_hit":
            ext.read_hit(x, out, n_words, int(repeat_inner), int(block_size))
        elif op == "l2_write_hit":
            ext.write_hit(y, n_words, int(repeat_inner), int(block_size))
        elif op == "l2_copy_hit":
            ext.copy_hit(x, y, n_words, int(repeat_inner), int(block_size))
        elif op == "l2_sliding_delta":
            ext.sliding_delta(x, out, n_words, cold_words, delta_words,
                              int(repeat_inner), int(block_size))

    # Build/warm once so JIT/extension and initial allocation are outside the
    # measured section. gpu_power_bench.py will still do its standard warmups.
    f()
    torch.cuda.synchronize(dev)

    read_bits = write_bits = hbm_bits = 0
    if op == "l2_read_hit":
        read_bits = working_set_bytes * repeat_inner * 8
    elif op == "l2_write_hit":
        write_bits = working_set_bytes * repeat_inner * 8
    elif op == "l2_copy_hit":
        read_bits = working_set_bytes * repeat_inner * 8
        write_bits = working_set_bytes * repeat_inner * 8
    elif op == "l2_sliding_delta":
        hit_bytes = max(0, working_set_bytes - delta_bytes)
        read_bits = hit_bytes * repeat_inner * 8
        hbm_bits = delta_bytes * repeat_inner * 8
    # reg_spin intentionally stays zero.

    l2 = get_l2_bytes(dev.index or 0)
    policy = "ld.global.cg" + ("+persisting_requested" if use_persisting_l2 else "+no_persisting")
    notes = ("L2-hit traffic path energy probe; not isolated SRAM bit-cell energy. "
             "Uses custom CUDA kernel with in-kernel repeat_inner loop and __ldcg reads; "
             "validate hit rate with separate Nsight Compute counter run.")
    if use_persisting_l2:
        notes += " Persisting L2 was requested in metadata; this prototype does not rely on it."

    return BenchSpec(
        name=f"{op}_{working_set_bytes >> 20}mib_R{repeat_inner}"
             + (f"_D{delta_bytes >> 10}k" if op == "l2_sliding_delta" else ""),
        op=op, dtype_label=dtype_label,
        shape=(n_words,), n_elements=n_words,
        flops_per_call=0, run=f,
        compute_unit="L2/cache path", emulated=False,
        cache_regime=classify_cache_regime(working_set_bytes, l2),
        path_semantics="l2_hit_path_probe",
        notes=notes,
        extra={
            "working_set_bytes": working_set_bytes,
            "repeat_inner": int(repeat_inner),
            "delta_bytes": int(delta_bytes),
            "estimated_l2_read_bits_per_call": int(read_bits),
            "estimated_l2_write_bits_per_call": int(write_bits),
            "estimated_l2_total_bits_per_call": int(read_bits + write_bits),
            "estimated_hbm_refill_bits_per_call": int(hbm_bits),
            "l2_policy": policy,
            "block_size": int(block_size),
            "grid_size": int(out_words),
            "kernel_version": "l2_probe_v1_cuda_ext_cg",
            "cold_pool_bytes": int(cold_pool_bytes if op == "l2_sliding_delta" else 0),
        },
    )
