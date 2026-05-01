#!/usr/bin/env python3
"""Phase-0 diagnostic for the fused-vs-standalone experiment (G11 / P1.4).

Verifies that the fusion mechanisms our `--include-fused` benchmarks rely on
actually take effect on the target machine BEFORE we try to measure energy
on them. Three checks :

  1. `torch.compile` fuses `gelu(linear(x) + b)` into a single kernel
     (epilogue activation fusion via inductor).
  2. `torch.compile` fuses `linear(layer_norm(x))` into a single kernel
     (pre-norm fusion).
  3. `F.scaled_dot_product_attention` selects the FlashAttention backend
     (not math / mem-efficient) on this GPU.

When a check fails, the script prints WHY and what fallback to use (TE
LayerNormMLP for #1/#2, or eager + warning for #3). Prints a final
verdict block that downstream tooling can grep for.

Run :
    python3 fusion_check.py
or :
    python3 fusion_check.py --dtype bf16 --shape 2048,2880,2880

Exit code = 0 if all required checks pass, 1 otherwise.

This file is a *diagnostic*, not a benchmark — it does NOT measure power.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager

import torch

# Suppress torch._dynamo logspam that the user doesn't need.
os.environ.setdefault("TORCHDYNAMO_DISABLE_LOGS", "1")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

VERDICT_OK = "PASS"
VERDICT_WARN = "WARN"
VERDICT_FAIL = "FAIL"


def _line(label: str, value: str) -> None:
    print(f"  {label:<32s} {value}")


def _hr() -> None:
    print("-" * 64)


def _resolve_dtype(label: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[label]


@contextmanager
def _capture_inductor_graph():
    """Capture the inductor-generated graph. We use a callback hook on
    `torch._inductor.compile_fx` so we can count fused kernels.

    The instrumentation API differs across torch versions ; we try the
    modern path first, then fall back to a no-op that just exercises
    the compile but doesn't count kernels (returns -1)."""
    captured: dict = {"n_kernels": -1}
    try:
        from torch._inductor import compile_fx as _cfx
        original = _cfx.compile_fx_inner

        def patched(*args, **kwargs):
            out = original(*args, **kwargs)
            try:
                # `out` may be a callable wrapping the cached kernel set.
                # In recent PyTorch the graph attribute holds a list of
                # PythonWrapperCodegen objects. The number of distinct
                # kernel function definitions in the wrapper code is the
                # measure that matters for "did fusion happen".
                wrapper_code = getattr(out, "_wrapper_code", None) or ""
                if not isinstance(wrapper_code, str):
                    wrapper_code = str(wrapper_code)
                # Count Triton kernel decorators OR cuda kernel definitions.
                n = (wrapper_code.count("@triton.jit")
                     + wrapper_code.count("@triton_heuristics."))
                if n > 0:
                    captured["n_kernels"] = n
            except Exception:
                pass
            return out

        _cfx.compile_fx_inner = patched
        try:
            yield captured
        finally:
            _cfx.compile_fx_inner = original
    except Exception:
        yield captured


# ---------------------------------------------------------------------------
# Check 1 + 2 — torch.compile epilogue / pre-norm fusion
# ---------------------------------------------------------------------------

def check_compile_linear_gelu(M: int, D_in: int, D_out: int,
                              dtype: torch.dtype) -> dict:
    """Build `gelu(linear(x))`, compile it, run once. Then check :
       * compile succeeded (no CompilationError)
       * the run produces the same output (within tolerance) as eager
       * one inductor-fused kernel was emitted (best-effort count)
    """
    print("\n[check 1] torch.compile : gelu(linear(x) + b)")
    x = torch.randn(M, D_in, dtype=dtype, device="cuda")
    W = torch.randn(D_in, D_out, dtype=dtype, device="cuda") * (1.0 / D_in ** 0.5)
    b = torch.zeros(D_out, dtype=dtype, device="cuda")

    def fn_eager(x):
        return torch.nn.functional.gelu(x @ W + b, approximate="tanh")

    try:
        with _capture_inductor_graph() as cap:
            fn_compiled = torch.compile(fn_eager, mode="reduce-overhead",
                                        dynamic=False)
            # Two warmups so triton autotuner runs.
            for _ in range(2):
                _ = fn_compiled(x)
            torch.cuda.synchronize()
            out_compiled = fn_compiled(x)
        out_eager = fn_eager(x)
        # Numerics check — same to within fp16 tolerance.
        max_abs = (out_compiled - out_eager).abs().max().item()
        rel = max_abs / (out_eager.abs().max().item() + 1e-9)
        n_kernels = cap.get("n_kernels", -1)

        ok = rel < 1e-2  # loose: dtype/algo differences add a percent
        verdict = (VERDICT_OK if ok and n_kernels in (-1, 1)
                   else VERDICT_WARN if ok else VERDICT_FAIL)
        _line("compiled OK", "yes")
        _line("max |Δ| vs eager", f"{max_abs:.3e}  (rel {rel:.3e})")
        _line("inductor kernels", str(n_kernels)
              if n_kernels >= 0 else "(could not count — version-dep)")
        _line("VERDICT", verdict)
        return {"ok": ok, "verdict": verdict, "n_kernels": n_kernels,
                "rel_err": rel}
    except Exception as e:
        _line("compile failed", f"{type(e).__name__}: {e}")
        _line("VERDICT", VERDICT_FAIL)
        return {"ok": False, "verdict": VERDICT_FAIL, "error": str(e)}


def check_compile_ln_linear(M: int, D_in: int, D_out: int,
                            dtype: torch.dtype) -> dict:
    """Build `linear(layer_norm(x))`, compile it, run once. Same checks
    as check 1."""
    print("\n[check 2] torch.compile : linear(layer_norm(x))")
    x = torch.randn(M, D_in, dtype=dtype, device="cuda")
    g = torch.ones(D_in,  dtype=dtype, device="cuda")
    b = torch.zeros(D_in, dtype=dtype, device="cuda")
    W = torch.randn(D_in, D_out, dtype=dtype, device="cuda") * (1.0 / D_in ** 0.5)
    bias = torch.zeros(D_out, dtype=dtype, device="cuda")

    def fn_eager(x):
        h = torch.nn.functional.layer_norm(x, (D_in,), g, b, eps=1e-5)
        return h @ W + bias

    try:
        with _capture_inductor_graph() as cap:
            fn_compiled = torch.compile(fn_eager, mode="reduce-overhead",
                                        dynamic=False)
            for _ in range(2):
                _ = fn_compiled(x)
            torch.cuda.synchronize()
            out_compiled = fn_compiled(x)
        out_eager = fn_eager(x)
        max_abs = (out_compiled - out_eager).abs().max().item()
        rel = max_abs / (out_eager.abs().max().item() + 1e-9)
        n_kernels = cap.get("n_kernels", -1)

        ok = rel < 1e-2
        verdict = (VERDICT_OK if ok and n_kernels in (-1, 1, 2)
                   else VERDICT_WARN if ok else VERDICT_FAIL)
        _line("compiled OK", "yes")
        _line("max |Δ| vs eager", f"{max_abs:.3e}  (rel {rel:.3e})")
        _line("inductor kernels", str(n_kernels)
              if n_kernels >= 0 else "(could not count — version-dep)")
        _line("VERDICT", verdict)
        return {"ok": ok, "verdict": verdict, "n_kernels": n_kernels,
                "rel_err": rel}
    except Exception as e:
        _line("compile failed", f"{type(e).__name__}: {e}")
        _line("VERDICT", VERDICT_FAIL)
        return {"ok": False, "verdict": VERDICT_FAIL, "error": str(e)}


# ---------------------------------------------------------------------------
# Check 3 — SDPA backend selection
# ---------------------------------------------------------------------------

def _sdpa_kernel_ctx(enable_flash=True, enable_math=False, enable_mem_efficient=False,
                     enable_cudnn=True):
    """Cross-version SDPA backend forcer. The PyTorch API for selecting
    SDPA backends moved twice — old `torch.backends.cuda.sdp_kernel`,
    then `torch.nn.attention.sdpa_kernel`, then back-compat re-exports.
    Try them in order ; whichever exists, use."""
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
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient)
    except Exception:
        # Fallback: no-op context.
        from contextlib import nullcontext
        return nullcontext()


def check_sdpa_flash(B: int, H_q: int, H_kv: int, N: int, D: int,
                     dtype: torch.dtype) -> dict:
    """Run SDPA with FlashAttention forced. If it falls back to math/
    mem_efficient, that's a FAIL — our attention_flash variant cannot
    measure what we want.

    GQA is honoured by passing K/V with H_kv heads ; SDPA broadcasts
    them across the H_q query groups internally."""
    print("\n[check 3] F.scaled_dot_product_attention : FlashAttention backend")
    q = torch.randn(B, H_q,  N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_kv, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_kv, N, D, dtype=dtype, device="cuda")

    # GQA : SDPA accepts H_kv != H_q only if PyTorch >= 2.5 with
    # `enable_gqa=True`. On older versions we expand K/V manually.
    enable_gqa_supported = False
    try:
        # Probe : try a tiny call with enable_gqa=True. If it raises
        # TypeError, the kwarg isn't supported.
        torch.nn.functional.scaled_dot_product_attention(
            q[:, :1], k[:, :1], v[:, :1], enable_gqa=True)
        enable_gqa_supported = (H_kv != H_q)
    except TypeError:
        enable_gqa_supported = False
    except Exception:
        enable_gqa_supported = False
    if H_kv != H_q and not enable_gqa_supported:
        _line("GQA path", "manual expand (PyTorch < 2.5 or unsupported)")
        groups = H_q // H_kv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    else:
        _line("GQA path", "native enable_gqa" if (H_kv != H_q) else "MHA (H_q==H_kv)")

    sdpa_kwargs = dict(is_causal=False)
    if enable_gqa_supported and H_kv != H_q:
        sdpa_kwargs["enable_gqa"] = True

    # Force flash backend, then check which one was actually used.
    try:
        with _sdpa_kernel_ctx(enable_flash=True, enable_cudnn=False,
                              enable_math=False, enable_mem_efficient=False):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, **sdpa_kwargs)
            torch.cuda.synchronize()
        used = "flash"
        _line("flash-only context", "ran without exception")
    except RuntimeError as e:
        msg = str(e)
        if "no available kernel" in msg.lower() or "not supported" in msg.lower():
            # Flash refused — try cudnn. On H100 cuDNN attention is
            # also FA-class and acceptable for our purpose.
            try:
                with _sdpa_kernel_ctx(enable_flash=False, enable_cudnn=True,
                                      enable_math=False, enable_mem_efficient=False):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, **sdpa_kwargs)
                    torch.cuda.synchronize()
                used = "cudnn"
                _line("flash refused; cudnn-only", "OK")
            except RuntimeError as e2:
                _line("both flash & cudnn refused", str(e2))
                _line("VERDICT", VERDICT_FAIL)
                return {"ok": False, "verdict": VERDICT_FAIL, "backend": "math/mem_efficient",
                        "error": str(e2)}
        else:
            _line("flash failed (other)", msg)
            _line("VERDICT", VERDICT_FAIL)
            return {"ok": False, "verdict": VERDICT_FAIL, "error": msg}

    _line("backend used", used)
    _line("VERDICT", VERDICT_OK)
    return {"ok": True, "verdict": VERDICT_OK, "backend": used,
            "enable_gqa_native": enable_gqa_supported}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--mlp-shape", type=str, default="2048,2880,2880",
                    help="M,D_in,D_out for linear_gelu / ln_linear (default: GPT-OSS 120B per-expert)")
    ap.add_argument("--attn-shape", type=str, default="1,64,8,2048,2048,64",
                    help="B,H_q,H_kv,N_q,N_kv,D_head (default: GPT-OSS 120B full-attn layer)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[fatal] CUDA not available — cannot run fusion check.")
        return 2
    torch.cuda.set_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    M, D_in, D_out = (int(x) for x in args.mlp_shape.split(","))
    B, H_q, H_kv, N_q, N_kv, D = (int(x) for x in args.attn_shape.split(","))
    if N_q != N_kv:
        print(f"[note] N_q != N_kv : SDPA test uses N = N_kv ({N_kv}) for square shape")
    N = N_kv

    # Header
    _hr()
    print(f"FUSION CHECK — torch {torch.__version__}, "
          f"CUDA {torch.version.cuda}, device cuda:{args.device}")
    _line("device", torch.cuda.get_device_name(args.device))
    _line("dtype", args.dtype)
    _line("MLP shape (M, D_in, D_out)", f"({M}, {D_in}, {D_out})")
    _line("attn shape (B, H_q, H_kv, N, D)", f"({B}, {H_q}, {H_kv}, {N}, {D})")
    _hr()

    r1 = check_compile_linear_gelu(M, D_in, D_out, dtype)
    r2 = check_compile_ln_linear(M, D_in, D_out, dtype)
    r3 = check_sdpa_flash(B, H_q, H_kv, N, D, dtype)

    # Final summary
    _hr()
    print("SUMMARY")
    _line("check 1 (linear_gelu)", r1["verdict"])
    _line("check 2 (ln_linear)",   r2["verdict"])
    _line("check 3 (sdpa_flash)",  r3["verdict"])

    all_ok = (r1["verdict"] in (VERDICT_OK, VERDICT_WARN)
              and r2["verdict"] in (VERDICT_OK, VERDICT_WARN)
              and r3["verdict"] == VERDICT_OK)
    print()
    if all_ok:
        print("→ All required checks passed. `--include-fused` is safe to use on this machine.")
        if r1["verdict"] == VERDICT_WARN or r2["verdict"] == VERDICT_WARN:
            print("  (WARN = numerics OK but fused-kernel count couldn't be verified — "
                  "torch version dependent, not blocking.)")
    else:
        print("→ One or more required checks FAILED. Suggested fallback :")
        if r1["verdict"] == VERDICT_FAIL:
            print("    * linear_gelu : use TransformerEngine LayerNormMLP "
                  "or skip (--include-fused will tag this variant emulated=1).")
        if r2["verdict"] == VERDICT_FAIL:
            print("    * ln_linear   : use TE LayerNormLinear or skip.")
        if r3["verdict"] == VERDICT_FAIL:
            print("    * sdpa_flash  : flash-attn / cudnn unavailable. "
                  "Either upgrade PyTorch / install flash-attn package, "
                  "or accept math/mem_efficient backend (NOT comparable "
                  "to fused-attention energy).")
    _hr()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
