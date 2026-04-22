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
    #   "l2_resident"  — working set ≤ L2/2, ≈ 100% L2 hit after iter 1
    #   "l2_partial"   — working set within [L2/2, 2·L2], thrashing / ~50%
    #   "dram_stream"  — working set ≥ 2·L2, ≈ 0% L2 hit
    #   "unknown"      — L2 size unavailable (legacy rows / non-CUDA)
    cache_regime: str = "unknown"
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
    """Return a coarse cache-locality regime for a given working set.

    Boundaries are set so each regime has a clear interpretation:
      * working set ≤ L2/2              → fits with margin → ~100% L2 hit after
                                          the first iteration (data stays resident
                                          across subsequent kernel launches).
      * L2/2 < working set ≤ 2·L2       → thrashing / partial hit (~50% as a
                                          ballpark; actual depends on access
                                          pattern and L2 replacement policy).
      * working set > 2·L2              → every kernel streams from DRAM;
                                          effective L2 hit rate near 0%.
    These thresholds are deliberately conservative — the inner boundaries
    aren't sharp, but they place each sweep point unambiguously in one bucket.
    """
    if l2_bytes <= 0:
        return "unknown"
    if working_set_bytes <= l2_bytes / 2:
        return "l2_resident"
    if working_set_bytes <= 2 * l2_bytes:
        return "l2_partial"
    return "dram_stream"


def _elementwise_working_set(op: str, n_elements: int, bytes_per_elem: int) -> int:
    """Bytes touched per kernel for a given op at a given size.

    - mul/add : 2 reads + 1 write = 3·N·bytes_per_elem
    - gelu    : 1 read + 1 write = 2·N·bytes_per_elem
    - softmax / layernorm : 1 read + 1 write = 2·N·bytes_per_elem (+ weight/bias
      are O(D) so negligible relative to N·D)
    This is what the L2 actually has to hold (transient) for the kernel to
    complete — the figure used to classify the regime.
    """
    if op in ("mul", "add"):
        rw = 3
    else:
        rw = 2
    return rw * n_elements * bytes_per_elem


def cache_sweep_points(op: str, dtype_label: str, l2_bytes: int) -> list[int]:
    """Return 3 N values that land cleanly in the 3 regimes for this op/dtype.

    Target working-set ratios vs L2:
      l2_resident : L2/8    (solid 100% hit)
      l2_partial  : L2      (right at threshold → ~50%)
      dram_stream : 8·L2    (clear DRAM streaming)
    """
    if l2_bytes <= 0:
        # Fall back to 3 spread-out defaults — same as before this feature.
        return [1 << 19, 1 << 23, 1 << 27]
    b = _dtype_bytes(dtype_label)
    rw = 3 if op in ("mul", "add") else 2
    def n_for(ws): return max(1 << 14, int(ws) // (rw * b))
    return [n_for(l2_bytes / 8), n_for(l2_bytes), n_for(8 * l2_bytes)]


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
    return BenchSpec(
        name=name, op=op, dtype_label=dtype_label,
        shape=shape, n_elements=actual_n,
        flops_per_call=flops, run=fn,
        compute_unit=compute_unit, emulated=emulated,
        cache_regime=regime, notes=notes,
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
# Memory:    3 × K² elements (A, B, out).  K=8192 fp32 ≈ 768 MiB — fits on
# both A100 40GB and H100.
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
        # Warmup call — also forces the torch backend .so load path so
        # a broken install fails during build() rather than inside the
        # power-sampled loop.
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            linear(x)
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
    return BenchSpec(
        name=name, op="matmul", dtype_label=dtype_label,
        shape=(M, N, K), n_elements=n_out,
        flops_per_call=flops, run=fn,
        compute_unit=compute_unit, emulated=emulated,
        cache_regime=regime, notes=notes,
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
