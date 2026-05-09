#!/usr/bin/env python3
"""GPU power benchmark driver — Joule per operation, static/dynamic split.

Runs 10 benchmarks (FP8/FP16 × MUL/ADD/Softmax/GeLU/LayerNorm) across a
sweep of load sizes and emits a CSV where each row is one (benchmark, load)
measurement. Power is integrated from NVML samples → Joules. The idle /
static power measured at startup is subtracted from the average workload
power to isolate *dynamic* energy.

Output columns (reports/gpu_power_bench_<gpu>_<stamp>.csv):
  gpu, compute_cap, op, dtype, n_elements, shape,
  iters, wall_s, flops_per_call, total_flops, total_elements,
  static_power_w, avg_power_w, dyn_power_w,
  total_energy_j, static_energy_j, dyn_energy_j,
  j_per_element_total, j_per_element_dyn,
  j_per_flop_total, j_per_flop_dyn,
  avg_temp_c, peak_temp_c, sm_clk_mhz, mem_clk_mhz, notes

Usage:
  python3 gpu_power_bench.py                 # defaults (full sweep)
  python3 gpu_power_bench.py --quick         # small sweep (~2 min)
  python3 gpu_power_bench.py --ops mul add   # subset
  python3 gpu_power_bench.py --dtypes fp16
  python3 gpu_power_bench.py --loads 1048576 16777216
  python3 gpu_power_bench.py --no-cooldown   # skip thermal wait
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

import pynvml
import torch

import benchmarks as bm
import gpu_profiles as gp
from power_monitor import (PowerSampler, measure_static_power,
                            wait_for_cooldown, wait_for_pstate_idle,
                            force_p8_for_measurement,
                            resolve_nvml_handle)
import preflight
# SoC envelope phase + plot helpers — same code path the standalone
# soc_power_bench.py uses, just composed here so we can run sweep + SoC
# in one process. Imported lazily-friendly (top-level is fine; module
# loads matplotlib only inside its plot fns).
import soc_power_bench as sb


# ---- defaults --------------------------------------------------------------

# Elementwise load sweep. Hit every cache regime with ≥ 2 points so the
# per-regime regression in analyze.py isn't a single-point fallback, and
# push the upper end high enough to touch realistic LLM activation sizes
# (a 32 × 2048 × 4096 BF16 activation = 256M elements).
#
# Working-set rule of thumb for mul/add (3·N·bytes_per_elem) on fp16:
#     N                  ws         regime on A100 (L2=40MB)    H100 (L2=50MB)
#     128K (1<<17)       0.75 MB    l2_resident                 l2_resident
#     1M   (1<<20)       6 MB       l2_resident                 l2_resident
#     2M   (1<<21)      12 MB       l2_resident                 l2_resident
#     8M   (1<<23)      48 MB       l2_partial (just past L2)   l2_partial
#     16M  (1<<24)      96 MB       l2_partial                  l2_partial
#     32M  (1<<25)     192 MB       dram_stream                 dram_stream
#     64M  (1<<26)     384 MB       dram_stream                 dram_stream
#     128M (1<<27)     768 MB       dram_stream                 dram_stream
#     256M (1<<28)     1.5 GB       dram_stream                 dram_stream
#     512M (1<<29)     3.0 GB       dram_stream                 dram_stream
#     1G   (1<<30)     6.0 GB       l2_hit_0 (~8% of 80GB A100)   l2_hit_0
#
# Memory safety: the largest point (1G mul fp16 ≈ 6 GB) fits on 80 GB
# A100 HBM2E with huge margin (~8%). fp8 emulation adds fp16
# intermediates (~3× the footprint), so for smaller-memory GPUs the
# memory-aware cap in _filter_loads() below drops cells that would
# exceed 25 % of HBM. That keeps the default sweep identical across
# A100 80GB / H100 80GB unless a smaller GPU is targeted.
DEFAULT_LOADS = [
    1 << 17,   # 128K  — launch-overhead regime (every op is l2_resident here)
    1 << 20,   # 1M
    1 << 21,   # 2M
    1 << 23,   # 8M    — upper l2_resident / lower l2_partial
    1 << 24,   # 16M   — solid l2_partial for mul/add (was a blind spot)
    1 << 25,   # 32M   — entering dram_stream
    1 << 26,   # 64M
    1 << 27,   # 128M
    1 << 28,   # 256M  — one realistic-size transformer activation
    1 << 29,   # 512M  — deep dram_stream, catches BW-saturation plateau
    1 << 30,   # 1G    — ~8% of 80 GB A100 HBM2E; headroom for fp8 emulation
]
QUICK_LOADS = [1 << 20, 1 << 22, 1 << 24]

# Matrix side length K (M = N = K). FLOPs per call = 2·K³, memory = 3·K²
# elements.
#
# Default set is "GPT-OSS-120B aware" — the hidden / attn / MLP-intermediate
# K values used by the reference 120B-class model are explicitly included so
# that square sweep results can be cross-checked against the asymmetric
# --llm-shapes results (which use the same K's). See README §3.2.
#
#     1024, 2048      — TC sweet spot; small enough to avoid BW saturation
#     2880            — GPT-OSS hidden dim (qkv / q_only / kv / mlp / lm_head)
#     4096            — GPT-OSS attn_o input (head_dim × heads)
#     5760            — GPT-OSS MLP intermediate (mlp1 out / mlp2 in)
#     8192, 12288     — large GEMM regime, BW saturation visible on fp32
#
# K=12288 fp32 needs 3·K²·4B ≈ 1.7 GB — fits on 80 GB A100/H100 with margin.
# K=512..1536 dropped vs old default — H100 fp8_te falls below NVML noise
# floor at K<2880 (README §8.3.4) and the tiny launch-overhead cell at
# K=512 was never load-bearing for k_op fits.
#
# Override per-run with `--matmul-sizes K1 K2 ...` to e.g. extend to 16384
# for fp8-only fp8_te runs or to drop large K on small-VRAM cards.
DEFAULT_MATMUL_SIZES = [1024, 2048, 2880, 4096, 5760, 8192, 12288]
QUICK_MATMUL_SIZES = [1024, 2880, 4096]


# Fraction of the device's total HBM that one elementwise cell may consume.
# At 0.25, an 80 GB A100 (HBM2E) caps at 20 GB per-cell; 80 GB H100
# caps at 20 GB; 40 GB cards (older A100 / V100) cap at 10 GB — still
# enough for 1B fp16 mul (6 GB) but will drop the 1G fp8 cell (9 GB).
_MEM_SAFETY_FRACTION = 0.25


# ---- test-suite presets ----------------------------------------------------
# Each suite is a dict of argparse-style attribute overrides applied to the
# parser's defaults BEFORE the user's argv is parsed. So the user can still
# override any individual flag explicitly. Suites compose the canonical
# benchmark categories (elementwise, matmul, cache, dram, llm) into named
# bundles so a user doesn't have to memorise 8 flags to run a focused test.
#
# Usage:
#   ./run_bench.sh                           # full validation suite
#   ./run_bench.sh --suite smoke             # 5-min sanity check
#   ./run_bench.sh --suite cache --tag h100   # pure cache regime sweep
#   ./run_bench.sh --suite full --tag h100    # everything + drift correction
#
# A user may override any field, e.g.:
#   ./run_bench.sh --suite full --no-matmul   # skip matmul portion of full
# SUITES are bundles of test-case selections + tuning. Each sets `cases`
# (the canonical list of test categories to run) plus any per-suite
# parameter overrides. Legacy fields (no_matmul, llm_shapes, etc.) are
# omitted here — the new path uses `cases` everywhere — but the legacy
# CLI flags themselves still work for backward compatibility.
SUITES: dict[str, dict] = {
    "smoke": {
        # Minimum-effort sanity check — quick mode + elementwise only.
        # ~5 minutes. Good for verifying the pipeline before spending
        # 30 min on the real sweep.
        "_doc":   "5-min sanity check (elementwise only, quick)",
        "cases":  ("elementwise",),
        "quick":  True,
    },
    "powermodel": {
        # Original baseline benchmark — elementwise + matmul, no extras.
        # Produces the canonical k_op coefficient table.
        "_doc":  "elementwise + matmul, no extra probes (legacy baseline)",
        "cases": ("elementwise", "matmul"),
    },
    "cache": {
        # Focused 5-point cache-regime sweep (--cache-sweep). Each
        # (op, dtype) gets one cell per regime bucket. Matmul kept on —
        # its per-K size sweep spans regimes naturally.
        "_doc":         "focused 5-point per-regime cache sweep + matmul",
        "cases":        ("elementwise", "matmul"),
        "cache_sweep":  True,
    },
    "dram": {
        # STREAM-style probes only. Purest signal for pJ/bit derivation.
        "_doc":  "STREAM-style DRAM bandwidth probes only (pJ/bit)",
        "cases": ("dram",),
    },
    "llm": {
        # Real LLM layer shapes (gpt-oss-120B presets) at 5 token counts.
        "_doc":  "gpt-oss-120B layer shapes only (qkv / mlp / lm_head etc.)",
        "cases": ("llm-matmul",),
    },
    "l2": {
        # L2/SRAM resident traffic path probe. Produces logical-traffic
        # pJ/bit estimates; Nsight Compute counters are a separate validation run.
        "_doc":             "L2/SRAM resident traffic probe (pJ/bit path estimate)",
        "cases":            ("l2",),
        "rebaseline_every": 10,
        "window_ms":        8000.0,
    },
    "soc": {
        # SoC envelope only — static / max / leakage. ~5 min.
        # Replaces the old soc_power_bench.py / run_soc_bench.sh combo.
        "_doc":  "SoC envelope only (static / max / leakage, ~5 min)",
        "cases": ("soc",),
    },
    "full": {
        # All built-in experiment cases, plus fused variants and drift
        # correction.  Users may pass --no-fused for dependency debugging.
        "_doc":             "all cases including L2 + SoC + fused + periodic re-baseline",
        "cases":            ("elementwise", "matmul", "llm-matmul", "dram", "l2", "soc"),
        "include_fused":    True,
        "rebaseline_every": 20,
        "window_ms":        6000.0,
    },
    "all": {
        # Alias of full kept for users who expect "all" to mean literally
        # every built-in case.
        "_doc":             "alias of full: all cases including L2 + SoC + fused",
        "cases":            ("elementwise", "matmul", "llm-matmul", "dram", "l2", "soc"),
        "include_fused":    True,
        "rebaseline_every": 20,
        "window_ms":        6000.0,
    },
    "fp8-mece": {
        # FP8-focused MECE characterisation. Long --window-ms because
        # FP8 dynamic energy is small and sensitive to NVML noise floor.
        # Includes fused so attention_flash fp8 is captured. Recommended
        # to wrap with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=1
        # NVTE_FUSED_ATTN_BACKEND=2` to verify FP8 DPA cuDNN sub-backend 2.
        "_doc":             "FP8-focused MECE (long window + rebaseline + fused, ~50 min)",
        "cases":            ("elementwise", "matmul", "dram"),
        "rebaseline_every": 20,
        "window_ms":        6000.0,
        "include_fused":    True,
        "fused_dtypes":     ("fp16", "bf16", "fp8"),
    },
}


def _argv_has_flag(argv: list[str], *names: str) -> bool:
    return any(a == name or a.startswith(name + "=")
               for a in argv for name in names)


def _implicit_suite_from_argv(argv: list[str]) -> str | None:
    """Default to the full validation suite unless argv selects a scope.

    Device/tag/output flags are not experiment-scope choices, so
    `run_bench.sh --device 0` should still run the full component sweep.
    Legacy scope flags keep their old behavior, while bare `--quick`
    maps to the smoke suite so the documented quick path stays short.
    """
    if _argv_has_flag(argv, "--suite", "--cases"):
        return None
    quick = _argv_has_flag(argv, "--quick")
    legacy_scope = _argv_has_flag(
        argv,
        "--no-elementwise",
        "--no-matmul",
        "--llm-shapes",
        "--dram-bw-test",
        "--cache-sweep",
    )
    if quick and not legacy_scope:
        return "smoke"
    if not quick and not legacy_scope:
        return "full"
    return None


def _apply_suite_to_parser(parser, suite_name: str) -> None:
    """Push a suite's overrides into the parser's default values so the
    user's explicit argv still wins."""
    if suite_name not in SUITES:
        raise SystemExit(
            f"unknown --suite {suite_name!r}. choices: {sorted(SUITES)}")
    overrides = {k: v for k, v in SUITES[suite_name].items()
                 if not k.startswith("_")}
    if overrides:
        parser.set_defaults(**overrides)


def _cell_memory_bytes(op: str, dtype_label: str, n_elements: int) -> int:
    """Conservative upper bound on DRAM footprint of a single elementwise cell.

    mul/add                       : 3 tensors (a, b, out)
    gelu                          : 2 tensors (in, out)
    softmax/layernorm             : 2 tensors (in, out) — reduction in fp32 done in-register
    stream_copy / stream_scale    : 2 tensors (in, out)
    stream_triad                  : 3 tensors (a, z, out)
    stream_read                   : 1 tensor  (in only — output is a scalar)
    stream_write                  : 1 tensor  (out only)
    fp8 path                      : each tensor may also materialise a full fp16
                                    intermediate (cast-compute-cast), so ×1.5.
    """
    if op in ("mul", "add", "stream_triad"):
        tensors = 3
    elif op in ("stream_read", "stream_write"):
        tensors = 1
    else:
        tensors = 2
    bytes_per_elem = {"fp16": 2, "fp8": 1}.get(dtype_label, 2)
    base = tensors * n_elements * bytes_per_elem
    # fp8 emulation path: add the fp16 shadow tensors.
    if dtype_label == "fp8":
        base += tensors * n_elements * 2
    return base


def _filter_loads(loads: list[int], ops: list[str], dtypes: list[str],
                  hbm_bytes: int) -> list[int]:
    """Drop load values that would exceed the per-cell HBM budget for any
    of the planned (op, dtype) combinations. Returns the loads that are
    safe for every combo; drops are logged so the user knows why the big
    cells disappeared.
    """
    if hbm_bytes <= 0:
        return loads
    budget = int(hbm_bytes * _MEM_SAFETY_FRACTION)
    kept: list[int] = []
    dropped: list[tuple[int, str, int]] = []
    for N in loads:
        worst = max(_cell_memory_bytes(op, dt, N) for op in ops for dt in dtypes)
        if worst <= budget:
            kept.append(N)
        else:
            worst_case = max(((op, dt, _cell_memory_bytes(op, dt, N))
                              for op in ops for dt in dtypes),
                             key=lambda t: t[2])
            dropped.append((N, f"{worst_case[0]}/{worst_case[1]}", worst_case[2]))
    if dropped:
        print(f"[memcheck] HBM={hbm_bytes/(1<<30):.1f} GB, "
              f"budget={budget/(1<<30):.1f} GB per cell ({int(_MEM_SAFETY_FRACTION*100)}%)")
        for N, worst, mb in dropped:
            print(f"[memcheck]   dropped N={N:,} "
                  f"(worst case {worst} ≈ {mb/(1<<30):.2f} GB)")
    return kept




def _parse_l2_repeats(values: list[str], window_bytes: int,
                       target_energy_j: float, k_guess_pj_bit: float) -> list[int]:
    """Return repeat_inner values for one L2 window.

    `auto` sizes the centre point from target_energy ≈ k_guess·W_bits·R,
    then returns a 4-point R sweep so analyze.py can fit fixed-W slopes and
    absorb one-time fill/launch overhead in the intercept.
    """
    if values == ["auto"] or "auto" in values:
        k_j_per_bit = max(k_guess_pj_bit, 1e-6) * 1e-12
        bits = max(1, int(window_bytes) * 8)
        base = int(math.ceil(max(target_energy_j, 0.1) / (k_j_per_bit * bits)))
        base = max(64, min(4_000_000, base))
        reps = sorted({max(16, base // 4), max(32, base // 2), base,
                       min(8_000_000, base * 2)})
        return reps
    reps: list[int] = []
    for v in values:
        try:
            r = int(v)
        except ValueError as e:
            raise SystemExit(f"bad --l2-repeat-inner value {v!r}: expected int or auto") from e
        if r <= 0:
            raise SystemExit("--l2-repeat-inner values must be positive")
        reps.append(r)
    return sorted(set(reps))




def _variant_key(plan: dict) -> tuple:
    """Key for suppressing repeated failures.

    Historical behavior ignores load_value so a poisoned CUDA context does not
    re-trigger for every K/N. L2 probes can legitimately fail only for one large
    working-set/cold-pool allocation, so include load/delta in that case.
    """
    base = (plan.get("op"), plan.get("dtype"), plan.get("mode"), plan.get("llm_preset", ""))
    if plan.get("category") == "l2":
        return base + (plan.get("load_value"),)
    return base


def _slugify(name: str) -> str:
    s = re.sub(r"(?i)nvidia|geforce|pcie|sxm\d?|\bhbm\d*\b|\bon\b", "", name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "gpu"


def _nvml_value(fn, default=""):
    try:
        return fn()
    except Exception:
        return default


def _gpu_spec_snapshot(profile_key: str, profile: dict, observed_profile: str,
                       profile_status: str, profile_reason: str,
                       gpu_name: str, cc: tuple[int, int], device: int,
                       handle, hbm_bytes: int, l2_reported_bytes: int,
                       l2_effective_bytes: int, l2_source: str) -> dict:
    power_limit_w = _nvml_value(
        lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, "")
    power_min_w = _nvml_value(
        lambda: pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[0] / 1000.0, "")
    power_max_w = _nvml_value(
        lambda: pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0, "")
    sm_max_mhz = _nvml_value(
        lambda: pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM), "")
    mem_max_mhz = _nvml_value(
        lambda: pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM), "")
    mig_current = ""
    mig_pending = ""
    mig_mode = _nvml_value(lambda: pynvml.nvmlDeviceGetMigMode(handle), None)
    if isinstance(mig_mode, tuple) and len(mig_mode) >= 2:
        mig_current, mig_pending = mig_mode[0], mig_mode[1]

    observed_cc = f"{cc[0]}.{cc[1]}"
    return {
        "gpu_profile": profile_key,
        "gpu_profile_label": profile.get("label", ""),
        "gpu_profile_status": profile_status,
        "gpu_profile_reason": profile_reason,
        "observed_profile": observed_profile,
        "gpu": gpu_name,
        "device": device,
        "compute_cap": observed_cc,
        "expected_compute_cap": profile.get("expected_cc", ""),
        "arch_expected": profile.get("arch", ""),
        "memory_type_expected": profile.get("memory_type", ""),
        "memory_capacity_expected_gb": profile.get("memory_capacity_gb", ""),
        "memory_total_gb": f"{hbm_bytes/(1<<30):.3f}" if hbm_bytes else "",
        "peak_bw_expected_gbps": profile.get("peak_bw_gbps", ""),
        "l2_expected_mb": profile.get("l2_mb", ""),
        "l2_reported_mb": f"{l2_reported_bytes/(1<<20):.3f}" if l2_reported_bytes else "",
        "l2_effective_mb": f"{l2_effective_bytes/(1<<20):.3f}" if l2_effective_bytes else "",
        "l2_source": l2_source,
        "power_envelope_expected_w": profile.get("power_envelope_w", ""),
        "power_limit_w": f"{power_limit_w:.3f}" if isinstance(power_limit_w, (int, float)) else "",
        "power_limit_min_w": f"{power_min_w:.3f}" if isinstance(power_min_w, (int, float)) else "",
        "power_limit_max_w": f"{power_max_w:.3f}" if isinstance(power_max_w, (int, float)) else "",
        "sm_clock_max_mhz": sm_max_mhz,
        "mem_clock_max_mhz": mem_max_mhz,
        "mig_mode_current": mig_current,
        "mig_mode_pending": mig_pending,
        "native_fp8_headline_allowed": int(gp.native_fp8_headline_allowed(profile_key, observed_cc)),
        "native_bf16_headline_allowed": int(bool(profile.get("native_bf16", False))),
        "profile_role": profile.get("role", ""),
    }


def _print_fused_failure_hint(err_msg: str) -> None:
    msg = err_msg.lower()
    print("  [fused] fused cell skipped. For full H100 component validation, "
          "fused support should be installed and working.")
    print("  [fused] check/install:")
    print("  [fused]   cd util/gpu_power_bench")
    print("  [fused]   ./install_transformer_engine.sh")
    print("  [fused]   python3 preflight.py --device 0")
    if any(x in msg for x in ("transformer_engine", "fp8", "dotproductattention", "nvte")):
        print("  [fused] Transformer Engine is required for fp8 fused attention "
              "and TE attention paths.")
    if any(x in msg for x in ("flash", "scaled_dot_product", "sdpa", "cudnn")):
        print("  [fused] PyTorch SDPA/FlashAttention backend was not usable. "
              "Check torch/CUDA/cuDNN compatibility.")
    print("  [fused] To bypass fused only for debugging, rerun with --no-fused.")


def _apply_effective_l2_cache_regime(spec: bm.BenchSpec, l2_bytes: int,
                                     l2_source: str) -> None:
    """Fill cache_regime from the effective L2 size when builders could not.

    Some torch builds do not expose device L2 size for RTX 3090, while the
    selected GPU profile still has the value needed for workload-size
    classification. Builders run before they know that profile fallback, so
    the driver patches unknown regimes here.
    """
    if spec.cache_regime != "unknown" or l2_bytes <= 0:
        return
    try:
        if spec.path_semantics == "l2_hit_path_probe" or spec.compute_unit == "L2/cache path":
            ws = int(spec.extra.get("working_set_bytes", 0) or 0)
            if ws <= 0:
                return
        elif spec.op in bm.FLOP_PER_ELEMENT:
            ws = bm._elementwise_working_set(  # noqa: SLF001 - local benchmark metadata helper.
                spec.op, spec.n_elements, bm._dtype_bytes(spec.dtype_label))
        elif spec.op in ("matmul", "matmul_llm") and len(spec.shape) == 3:
            a, b, c = spec.shape
            ws = (a * b + b * c + a * c) * bm._dtype_bytes(spec.dtype_label)
        elif spec.op in bm.FUSED_VARIANTS:
            if "attention" in spec.op and len(spec.shape) == 6:
                B, H_q, H_kv, N_q, N_kv, D_head = spec.shape
                ws = (B * (H_q + 2 * H_kv) * N_kv * D_head
                      + B * H_q * N_q * D_head) * 2
            elif len(spec.shape) == 3:
                M, D_in, D_out = spec.shape
                ws = (M * D_in + D_in * D_out + M * D_out) * 2
            else:
                return
        else:
            return
    except Exception:
        return
    spec.cache_regime = bm.classify_cache_regime(int(ws), l2_bytes)
    spec.extra["cache_regime_source"] = l2_source


def pick_iters(spec: bm.BenchSpec, target_ms: float, ms_per_call: float) -> int:
    """Choose iter count so one measurement window is roughly target_ms long.

    A longer window → lower NVML quantization noise on the integrated energy.
    We floor at 16 iters so launch overhead doesn't dominate at tiny sizes,
    and cap at 1e6 so 0.01 ms kernels don't explode.
    """
    if ms_per_call <= 0:
        return 100
    iters = int(round(target_ms / ms_per_call))
    return max(16, min(1_000_000, iters))


def time_one_call(spec: bm.BenchSpec, warmup: int = 5, reps: int = 8) -> float:
    """Return milliseconds per call (best-of-N)."""
    for _ in range(warmup):
        spec.run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(reps):
        start.record()
        spec.run()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def run_measurement(spec: bm.BenchSpec, sampler: PowerSampler,
                    window_ms: float) -> dict:
    """Measure one (op, dtype, load) cell."""
    ms_per_call = time_one_call(spec)
    iters = pick_iters(spec, target_ms=window_ms, ms_per_call=ms_per_call)

    # A second warm-up so the clocks are already boosted before we start the
    # NVML stopwatch — the first sample window otherwise catches the clock
    # ramp and inflates dyn power estimates.
    for _ in range(10):
        spec.run()
    torch.cuda.synchronize()

    sampler.set_phase(f"{spec.name}_N{spec.n_elements}")
    t_start = time.perf_counter() - sampler.t0
    for _ in range(iters):
        spec.run()
    torch.cuda.synchronize()
    t_end = time.perf_counter() - sampler.t0
    sampler.set_phase("gap")

    wall_s = t_end - t_start
    energy_total_j = sampler.energy_joules(t_start, t_end)
    avg_power = sampler.avg_power(t_start, t_end)
    avg_temp = sampler.avg_temp(t_start, t_end)
    peak_temp = sampler.peak_temp(t_start, t_end)
    # Snapshot clocks at the end of the run (representative of steady state).
    last = next((s for s in reversed(sampler.samples)
                 if s.phase.startswith(spec.name)), None)
    sm_clk = last.sm_mhz if last else -1
    mem_clk = last.mem_mhz if last else -1

    return {
        "iters": iters,
        "ms_per_call": ms_per_call,
        "wall_s": wall_s,
        "total_energy_j": energy_total_j,
        "avg_power_w": avg_power,
        "avg_temp_c": avg_temp,
        "peak_temp_c": peak_temp,
        "sm_clk_mhz": sm_clk,
        "mem_clk_mhz": mem_clk,
    }


def main() -> int:
    argv = sys.argv[1:]
    user_set_dtypes = _argv_has_flag(argv, "--dtypes")
    user_set_l2_windows = _argv_has_flag(argv, "--l2-window-mb")
    user_set_l2_delta = _argv_has_flag(argv, "--l2-delta-kb")
    user_set_cases = _argv_has_flag(argv, "--cases")
    user_set_fused = _argv_has_flag(argv, "--include-fused", "--no-fused")

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=("Test-suite presets (--suite NAME):\n"
                + "\n".join(f"  {k:11s} {v.get('_doc','')}"
                            for k, v in SUITES.items())))
    ap.add_argument("--suite", choices=list(SUITES), default=None,
                    help="apply a predefined bundle of flags. See epilog "
                         "above. Individual flags after --suite still "
                         "override the suite's defaults. If no suite/cases "
                         "or legacy scope flags are given, defaults to full.")
    # ---- Test cases — orthogonal axis to suites ----------------------------
    # `--cases` selects which experiment categories to run, decoupled from
    # the suite presets. When set, it's the authoritative source — any
    # legacy --no-elementwise / --no-matmul / --llm-shapes / --dram-bw-test
    # flags are IGNORED so the user gets exactly the cases they asked for.
    # When NOT set, the existing legacy flag behaviour is preserved
    # (backward compat for old scripts).
    #
    # Cases:
    #   elementwise  — A.1 elementwise sweep (mul/add/softmax/gelu/layernorm)
    #   matmul       — A.3 square matmul (5 dtype/mode variants × K-sweep)
    #   llm-matmul   — A.4 LLM-shape matmul (8 preset × T-sweep)
    #   dram         — A.2 STREAM-style probes (read/write/copy/scale/triad)
    #   l2           — A.6 L2/SRAM resident traffic path probe
    #   soc          — B   SoC envelope (static / max / leakage)
    ALL_CASES = ("elementwise", "matmul", "llm-matmul", "dram", "l2", "soc")
    ap.add_argument("--cases", nargs="+", choices=ALL_CASES, default=None,
                    metavar="CASE",
                    help="explicit list of test cases to run. Choices: "
                         + " / ".join(ALL_CASES) + ". When set, overrides "
                         "suite cases and legacy --no-* flags. If used with "
                         "--suite full/all, fused is disabled unless "
                         "--include-fused is also explicit. When unset, "
                         "derived from legacy flags (--no-elementwise / "
                         "--no-matmul / --llm-shapes / --dram-bw-test) for "
                         "back-compat.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gpu-profile", choices=gp.profile_choices(),
                    default=gp.DEFAULT_GPU_PROFILE,
                    help="experiment profile for GPU-specific defaults and headline gating. "
                         "Default is h100_sxm because H100 SXM is the primary target.")
    ap.add_argument("--allow-non-headline", action="store_true",
                    help="do not filter profile-unsupported dtype/variant choices. "
                         "Rows still record headline_status=NOT_HEADLINE/PROXY.")
    ap.add_argument("--ops", nargs="+",
                    default=list(bm.OPS),
                    choices=list(bm.OPS))
    ap.add_argument("--dtypes", nargs="+",
                    default=None,
                    choices=list(bm.DTYPES))
    ap.add_argument("--loads", type=int, nargs="+", default=None,
                    help="tensor element counts; default: 256K..256M sweep")
    ap.add_argument("--quick", action="store_true",
                    help="short sweep for smoke-testing")
    ap.add_argument("--window-ms", type=float, default=3000.0,
                    help="target measurement window per cell (ms). Longer = lower NVML noise; "
                         "for the energy-decomposition workflow (MECE plot in analyze.py), "
                         "6000+ is recommended because the small-cell numbers it subtracts "
                         "between are most exposed to the ±5..10 W NVML noise floor. "
                         "3000 ms gives ≈60 power samples per cell at 20 Hz NVML update rate.")
    ap.add_argument("--static-seconds", type=float, default=12.0,
                    help="idle time to measure static/baseline power")
    ap.add_argument("--cooldown-c", type=int, default=50,
                    help="°C threshold to reach between experiments (set -1 to disable)")
    ap.add_argument("--cooldown-min-s", type=float, default=5.0,
                    help="minimum idle time before starting a cell, even if already below "
                         "--cooldown-c. Ensures HBM / VRM residual heat has time to dissipate.")
    ap.add_argument("--cooldown-timeout", type=float, default=180.0)
    ap.add_argument("--no-cooldown", action="store_true",
                    help="skip thermal cool-down between cells")
    ap.add_argument("--pstate-idle-wait", type=float, default=30.0,
                    help="before each static / re-baseline measurement, "
                         "block until the GPU's SM clock drops below "
                         "PSTATE_IDLE_CLOCK_THRESHOLD_MHZ (default 500) for "
                         "3 consecutive samples — proves P8 idle was actually "
                         "reached. 0 = disable. wait_for_cooldown only checks "
                         "TEMPERATURE; P-state hysteresis can keep clocks in "
                         "P0 even after temp drops, inflating P_static by "
                         "30..50 W and triggering the '0/N samples reached "
                         "P8' warning. Recommended: 30s on H100, longer if "
                         "boost-clock lock survives. Use --sudo-pstate when "
                         "the selected GPU allows sudo clock resets.")
    ap.add_argument("--sudo-pstate", action="store_true",
                    help="allow selected-GPU P-state reset helpers to run "
                         "`sudo -n nvidia-smi -i <selected_gpu> -rgc` during "
                         "baseline/rebaseline. This does not run the benchmark "
                         "as root. run_bench.sh asks for sudo once and keeps "
                         "the timestamp alive; if calling this Python file "
                         "directly, run `sudo -v` first.")
    ap.add_argument("--rebaseline-every", type=int, default=0,
                    help="re-measure idle / static power every N cells "
                         "(0 = once at start, no drift correction). 20 is a "
                         "good default for a 130-cell sweep — adds about "
                         "1–2 minutes of wall time but tracks the 1–3 W of "
                         "P_static drift that builds up over a 30-minute run.")
    ap.add_argument("--rebaseline-seconds", type=float, default=4.0,
                    help="duration of each re-baseline measurement (s). "
                         "Shorter than --static-seconds because we already "
                         "have a thermal context and only need a quick "
                         "re-anchor. Default 4 s.")
    ap.add_argument("--out-dir", type=str, default="reports")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output filenames (separate runs / configs)")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--poll-hz", type=int, default=100)
    ap.add_argument("--power-source", choices=["legacy", "instant", "average"],
                    default="legacy",
                    help="NVML power source to integrate. 'legacy' "
                         "(default) = nvmlDeviceGetPowerUsage, matches "
                         "nvidia-smi exactly. 'instant' = "
                         "NVML_FI_DEV_POWER_INSTANT (~1ms cadence) — "
                         "captures transients but on idle GPUs reads "
                         "+5..40 W higher than nvidia-smi due to DMA / "
                         "telemetry / heartbeat spikes the averaged "
                         "source smooths away. 'average' = "
                         "NVML_FI_DEV_POWER_AVERAGE running average.")
    # --- matmul (Tensor Core vs CUDA-core + TE FP8) ---
    ap.add_argument("--no-matmul", action="store_true",
                    help="skip the matmul (Tensor Core / SIMT) sweep")
    ap.add_argument("--no-elementwise", action="store_true",
                    help="skip the elementwise sweep. Useful for re-running "
                         "just the matmul tail of a previous run — e.g. after "
                         "fixing a transformer_engine install and wanting the "
                         "8 fp8_te cells without redoing the 90 elementwise "
                         "cells. Combine with --matmul-variants fp8:te to "
                         "target matmul_fp8_te specifically.")
    ap.add_argument("--cache-sweep", action="store_true",
                    help="override --loads with exactly 3 elementwise points "
                         "per (op, dtype) targeting the three cache regimes: "
                         "L2-resident (~100%% L2 hit), L2-partial (~50%%), "
                         "DRAM-stream (~0%% L2 hit). Sizes are derived from "
                         "the detected L2 capacity of --device. Use this to "
                         "isolate cache-locality effects on energy.")
    ap.add_argument("--matmul-sizes", type=int, nargs="+", default=None,
                    help="square matrix side lengths (M=N=K); default: 512..8192")
    ap.add_argument("--matmul-variants", nargs="+", default=None,
                    help='matmul variants "dtype:mode" (default: all 5). '
                         'choices: fp32:simt tf32:tc fp16:tc bf16:tc fp8:te')
    # --- LLM-shape matmul sweep (real inference layer shapes) ---
    ap.add_argument("--llm-shapes", action="store_true",
                    help="enable the LLM-shape matmul sweep (real inference "
                         "shapes for a representative 120B-class model: "
                         "QKV / KV / MLP1 / MLP2 / router / LM-head, with "
                         "the token-count M dim swept separately from the "
                         "fixed K / N dims). Disabled by default because it "
                         "adds ~40 cells; the square sweep stays enabled.")
    ap.add_argument("--llm-presets", nargs="+", default=None,
                    help="subset of LLM_SHAPES keys to benchmark. Default: all 8.")
    ap.add_argument("--llm-ts", type=int, nargs="+", default=None,
                    help="token-count (M dim) sweep for --llm-shapes; "
                         "default: 1 256 2048 8192 32768")
    ap.add_argument("--llm-dtypes", nargs="+", default=None,
                    help='LLM-shape dtype:mode list (default: bf16:tc). '
                         'choices: fp32:simt tf32:tc fp16:tc bf16:tc fp8:te')
    # --- DRAM bandwidth probe (STREAM-style copy/scale/triad) ---
    ap.add_argument("--dram-bw-test", action="store_true",
                    help="add STREAM-style probes (stream_copy / stream_scale "
                         "/ stream_triad) at large working sets so analyze.py "
                         "can derive pJ/bit of DRAM traffic. Each probe is "
                         "compute-light, so the dynamic energy is dominated "
                         "by HBM ↔ DRAM movement. See README §3.5 for the "
                         "literature comparison points (HBM2 ≈ 7 pJ/bit, "
                         "HBM3 ≈ 4 pJ/bit) and what our board-level "
                         "measurement boundary actually includes.")
    ap.add_argument("--dram-bw-loads", type=int, nargs="+", default=None,
                    help="working-set-target N values for --dram-bw-test "
                         "(default: 4 sizes deep into the l2_hit_0 regime)")
    # --- L2/SRAM resident traffic path probe --------------------------------
    ap.add_argument("--l2-window-mb", type=int, nargs="+",
                    default=None,
                    help="L2 probe working-set windows in MiB. Keep below full "
                         "L2 to reduce replacement/HBM contamination. Default "
                         "comes from --gpu-profile (H100: 16/24/32/40 MiB).")
    ap.add_argument("--l2-repeat-inner", nargs="+", default=["auto"],
                    help="repeat_inner values for the in-kernel loop, or auto. "
                         "auto derives a small R sweep around --l2-target-energy-j "
                         "using --l2-k-guess-pj-bit.")
    ap.add_argument("--l2-target-energy-j", type=float, default=10.0,
                    help="target logical incremental L2 energy used to size "
                         "auto repeat_inner values (J).")
    ap.add_argument("--l2-k-guess-pj-bit", type=float, default=1.0,
                    help="initial L2-hit path pJ/bit guess for auto R sizing.")
    ap.add_argument("--l2-delta-kb", type=int, nargs="+",
                    default=None,
                    help="sliding-window delta sizes in KiB. Delta=0 is the "
                         "L2-resident reference; nonzero deltas are validation, "
                         "not the primary estimator. Default comes from "
                         "--gpu-profile.")
    ap.add_argument("--l2-cold-pool-gb", type=float, default=4.0,
                    help="cold pool size for l2_sliding_delta (GiB).")
    ap.add_argument("--l2-dtypes", nargs="+", default=["uint32"],
                    choices=["uint32", "fp32"],
                    help="storage word label for L2 probes. Both are 4B words; "
                         "uint32 is the default to avoid FP arithmetic effects.")
    l2_persist = ap.add_mutually_exclusive_group()
    l2_persist.add_argument("--l2-use-persisting", dest="l2_use_persisting",
                            action="store_true",
                            help="record persisting-L2 intent in metadata. The "
                                 "prototype uses __ldcg and counter validation; "
                                 "MIG may disable persisting set-aside.")
    l2_persist.add_argument("--l2-no-persisting", dest="l2_use_persisting",
                            action="store_false",
                            help="do not request persisting-L2 metadata")
    ap.set_defaults(l2_use_persisting=False)
    # ---- Fused vs Standalone (G11 / P1.4) --------------------------------
    # 6 new variants : attention_flash + attention_qkv_matmul (subtraction
    # baseline), linear_gelu + linear_baseline_gelu, ln_linear +
    # linear_baseline_ln. Each variant runs as a SINGLE cell at fixed
    # shape (no load sweep — the load axis is the shape itself, fixed by
    # GPT-OSS 120B defaults). analyze.py pairs full ↔ baseline by op
    # group and computes residual + bootstrap CI. See README §3.7.
    ap.add_argument("--include-fused", action="store_true",
                    help="add the fused-vs-standalone variants. Enabled by "
                         "--suite full/all. See README §3.7 / TestCases A.5 "
                         "/ REVIEW.md G11.")
    ap.add_argument("--no-fused", dest="include_fused", action="store_false",
                    help="disable fused variants even if the selected suite "
                         "enables them. Use only for dependency debugging.")
    ap.set_defaults(include_fused=False)
    ap.add_argument("--attn-shape", type=str, default="1,64,8,2048,2048,64",
                    help="attention_flash / attention_qkv_matmul shape — "
                         "B,H_q,H_kv,N_q,N_kv,D_head. Default = GPT-OSS 120B "
                         "full-attention layer.")
    ap.add_argument("--mlp-shape", type=str, default="2048,2880,2880",
                    help="linear_gelu / ln_linear shape — M,D_in,D_out. "
                         "Default = GPT-OSS 120B per-expert MoE intermediate.")
    ap.add_argument("--fused-causal", action="store_true",
                    help="attention_flash : use causal mask. Halves "
                         "softmax cost (lower bound). Default off "
                         "= upper bound estimate.")
    ap.add_argument("--fused-fusion-backend", choices=["auto", "compile", "eager"],
                    default="auto",
                    help="how to enforce fusion for linear_gelu / ln_linear. "
                         "'auto' tries torch.compile and falls back to eager "
                         "with a warning. 'eager' disables fusion entirely "
                         "(diagnostic only — residuals will inflate).")
    ap.add_argument("--fused-dtypes", nargs="+", default=None,
                    help="dtypes for fused variants. Default = profile fused "
                         "dtypes, or the intersection of --dtypes with "
                         "{fp16, bf16, fp8} when --dtypes is explicit. fp8 "
                         "only applies to attention_flash (via Transformer "
                         "Engine + fp8_autocast) — all other fused "
                         "variants (qkv_matmul baseline, linear_gelu, "
                         "ln_linear) remain fp16/bf16. fp8 MLP fused is "
                         "REVIEW.md G12 / P2.4 follow-up.")
    # ---- SoC envelope (the case formerly known as soc_power_bench.py) ----
    # Runs static/max/leakage phases as a sidecar. Same defaults as the
    # standalone soc_power_bench.py (~5 min wall). All flags accept the
    # exact value names from the standalone script's CLI; only namespace
    # (`--soc-*`) is added so they don't collide with sweep flags.
    ap.add_argument("--soc-static-seconds", type=float, default=20.0,
                    help="SoC envelope: idle-baseline phase duration")
    ap.add_argument("--soc-max-seconds", type=float, default=30.0,
                    help="SoC envelope: max-power GEMM phase duration")
    ap.add_argument("--soc-leakage-cycles", type=int, default=5,
                    help="SoC envelope: number of stress/decay cycles")
    ap.add_argument("--soc-leakage-stress-s", type=float, default=10.0,
                    help="SoC envelope: GEMM stress duration per leakage cycle")
    ap.add_argument("--soc-leakage-decay-s", type=float, default=15.0,
                    help="SoC envelope: post-stress idle decay per cycle")
    ap.add_argument("--soc-leak-window-s", type=float, default=1.0,
                    help="SoC envelope: window after stress-stop for hot-leakage avg")
    ap.add_argument("--soc-matmul-K", type=int, default=16384,
                    help="SoC envelope: square GEMM K for max/stress phases")
    ap.add_argument("--soc-dtype", type=str, default="fp16",
                    choices=["fp32", "tf32", "fp16", "bf16", "fp8"],
                    help="SoC envelope: GEMM dtype")
    ap.add_argument("--soc-mode", type=str, default="tc",
                    choices=["simt", "tc", "te"],
                    help="SoC envelope: compute path (simt/tc/te)")
    args = ap.parse_args()

    selected_suite = args.suite or _implicit_suite_from_argv(argv)

    # If a suite was named or inferred, re-parse so its overrides become
    # defaults but explicit user flags still take precedence. Also print the
    # resolved config so the user can see what the suite expanded into.
    if selected_suite:
        if args.suite is None:
            ap.set_defaults(suite=selected_suite)
        _apply_suite_to_parser(ap, selected_suite)
        args = ap.parse_args()
        sd = SUITES[selected_suite]
        flags = ", ".join(f"{k}={v}" for k, v in sd.items() if not k.startswith("_"))
        if selected_suite == args.suite and _argv_has_flag(argv, "--suite"):
            origin = "selected"
        else:
            origin = "default"
        print(f"[suite] '{selected_suite}' ({origin}) → "
              f"{flags or '(no overrides — legacy default)'}")
        if user_set_cases and not user_set_fused and args.include_fused:
            args.include_fused = False
            print("[suite] explicit --cases overrides the suite fused default; "
                  "add --include-fused to keep fused variants.")

    profile_key = args.gpu_profile if args.gpu_profile != "auto" else gp.DEFAULT_GPU_PROFILE
    profile = gp.GPU_PROFILES[profile_key]
    if args.dtypes is None:
        args.dtypes = gp.profile_default_dtypes(profile_key)
    if args.l2_window_mb is None:
        args.l2_window_mb = gp.profile_l2_windows_mb(profile_key)
    if args.l2_delta_kb is None:
        args.l2_delta_kb = gp.profile_l2_delta_kb(profile_key)

    # ---- Resolve which test cases to actually run ---------------------
    # Order of precedence:
    #   1. --cases X Y Z          (explicit; overrides legacy flags)
    #   2. SUITES[suite]["cases"] (suite-level default; set by re-parse above)
    #   3. legacy --no-* / --llm-shapes / --dram-bw-test flags. A bare
    #      command is handled earlier by the implicit full suite.
    if args.cases is not None:
        cases = set(args.cases)
        # User asked for explicit cases — silently ignore legacy --no-* /
        # --llm-shapes / --dram-bw-test flags (they're noise here).
    else:
        cases = set()
        if not args.no_elementwise:
            cases.add("elementwise")
        if not args.no_matmul:
            cases.add("matmul")
        if args.llm_shapes:
            cases.add("llm-matmul")
        if args.dram_bw_test:
            cases.add("dram")
        # SoC envelope is opt-in via --cases or via the soc / all suites.
        # Suites set args.cases (form 1), so this `else` branch only fires
        # when neither --cases nor a suite that includes soc was given.

    if not cases:
        print("[error] no test cases selected — pass --cases ... or pick a suite")
        return 1
    print(f"[cases] {' '.join(sorted(cases))}")

    if args.quick:
        loads = QUICK_LOADS
        matmul_sizes = QUICK_MATMUL_SIZES
    else:
        loads = args.loads or DEFAULT_LOADS
        matmul_sizes = args.matmul_sizes or DEFAULT_MATMUL_SIZES

    # Parse matmul variants list.
    if args.matmul_variants is None:
        matmul_variants = list(bm.MATMUL_VARIANTS)
    else:
        matmul_variants = []
        for v in args.matmul_variants:
            parts = v.split(":")
            if len(parts) != 2:
                print(f"bad --matmul-variants entry {v!r} (expected dtype:mode)")
                return 1
            matmul_variants.append(tuple(parts))

    # ---- preflight ---------------------------------------------------------
    if not args.skip_preflight:
        pf = preflight.check(device=args.device)
        preflight.print_report(pf)
        if not pf.ok:
            print("preflight failed — fix the above or re-run with --skip-preflight")
            return 1

    # ---- NVML + device -----------------------------------------------------
    # Two ways the user can pin the run to a specific GPU:
    #   (a) `--device N`                          — the canonical path
    #   (b) `CUDA_VISIBLE_DEVICES=N` env var      — common shell habit
    #
    # CUDA_VISIBLE_DEVICES is a CUDA-side filter: torch only sees the listed
    # GPUs and re-numbers them starting at 0. NVML, however, ignores this
    # env var and ALWAYS addresses GPUs by their physical index. So the
    # naive `nvmlDeviceGetHandleByIndex(args.device)` reads the WRONG card
    # whenever CUDA_VISIBLE_DEVICES restricts visibility — silently. The
    # fix below pins NVML to torch's actual GPU via PCI bus ID, which is
    # the same identifier on both sides and is unaffected by remapping.
    pynvml.nvmlInit()
    torch.cuda.set_device(args.device)
    gpu_name = torch.cuda.get_device_name(args.device)
    cc = torch.cuda.get_device_capability(args.device)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    handle, nvml_resolution = resolve_nvml_handle(args.device)
    observed_cc = f"{cc[0]}.{cc[1]}"
    resolved_key, resolved_profile, observed_profile, observed_reason = gp.resolve_gpu_profile(
        args.gpu_profile, gpu_name, observed_cc)
    if args.gpu_profile == "auto" and resolved_key != profile_key:
        profile_key = resolved_key
        profile = resolved_profile
        if not user_set_dtypes:
            args.dtypes = gp.profile_default_dtypes(profile_key)
        if not user_set_l2_windows:
            args.l2_window_mb = gp.profile_l2_windows_mb(profile_key)
        if not user_set_l2_delta:
            args.l2_delta_kb = gp.profile_l2_delta_kb(profile_key)
    profile_status, profile_status_reason = gp.profile_cc_status(profile_key, observed_cc)
    profile_reason = "; ".join(x for x in (observed_reason, profile_status_reason) if x)
    if profile_reason:
        print(f"[profile] WARN {profile_reason}")

    kept_dtypes, dropped_dtypes = gp.filter_with_profile(
        list(args.dtypes), gp.profile_default_dtypes(profile_key),
        label="--dtypes", allow_non_headline=args.allow_non_headline)
    if dropped_dtypes:
        print(f"[profile] dropped non-headline --dtypes for {profile_key}: {dropped_dtypes} "
              f"(use --allow-non-headline to keep them)")
    args.dtypes = kept_dtypes

    kept_variants, dropped_variants = gp.filter_with_profile(
        matmul_variants, gp.profile_matmul_variants(profile_key),
        label="--matmul-variants", allow_non_headline=args.allow_non_headline)
    if dropped_variants:
        dropped_fmt = [f"{d}:{m}" for d, m in dropped_variants]
        print(f"[profile] dropped non-headline --matmul-variants for {profile_key}: "
              f"{dropped_fmt} (use --allow-non-headline to keep them)")
    matmul_variants = kept_variants
    soc_variant = (args.soc_dtype, args.soc_mode)
    if "soc" in cases and not args.allow_non_headline:
        allowed_soc = gp.profile_matmul_variants(profile_key)
        if soc_variant not in allowed_soc:
            fallback = ("fp16", "tc") if ("fp16", "tc") in allowed_soc else allowed_soc[0]
            print(f"[profile] changed non-headline SoC GEMM {args.soc_dtype}:{args.soc_mode} "
                  f"to {fallback[0]}:{fallback[1]} for {profile_key} "
                  f"(use --allow-non-headline to keep it)")
            args.soc_dtype, args.soc_mode = fallback
    if "by index" in nvml_resolution and cvd:
        print(f"[warn] CUDA_VISIBLE_DEVICES={cvd!r} is set but we could not "
              f"resolve the GPU by PCI bus id / UUID — NVML reads may target "
              f"the WRONG physical GPU. Prefer `--device {args.device}` "
              f"without CUDA_VISIBLE_DEVICES, or upgrade torch.")
    gpu_slug = _slugify(gpu_name)
    print(f"\n[info] GPU={gpu_name}  cc={cc[0]}.{cc[1]}  slug={gpu_slug}")
    print(f"[profile] requested={args.gpu_profile}  active={profile_key} "
          f"({gp.profile_label(profile_key)})  status={profile_status}")
    print(f"[info] NVML handle resolved {nvml_resolution}"
          + (f"  (CUDA_VISIBLE_DEVICES={cvd})" if cvd else ""))
    print(f"[info] ops={args.ops}  dtypes={args.dtypes}  loads={loads}")

    # ---- static / baseline power ------------------------------------------
    # Drop any stale allocations, sync, then sit idle. This is the P_static we
    # will subtract to isolate dynamic (workload) power.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if not args.no_cooldown and args.cooldown_c > 0:
        wait_for_cooldown(handle, target_c=args.cooldown_c,
                          timeout_s=args.cooldown_timeout,
                          min_s=args.cooldown_min_s)
    # Wait for actual P8 idle (boost-clock hysteresis can keep SMs at P0
    # for tens of seconds even after temperature drops).
    if args.pstate_idle_wait > 0:
        wait_for_pstate_idle(handle, timeout_s=args.pstate_idle_wait)
    # Aggressively force P8 (NVML clock-lock or `nvidia-smi -rgc`) so the
    # static measurement actually captures cold-idle, not P0-locked boost
    # idle. The restore() callback unlocks clocks AFTER the measurement,
    # so the workload phase keeps full DVFS.
    if args.pstate_idle_wait > 0:
        p8_ctx = force_p8_for_measurement(
            handle, use_sudo=args.sudo_pstate)
    else:
        p8_ctx = {"success": False, "method": "skipped", "restore": None}
    print(f"[baseline] measuring static power for {args.static_seconds:.1f}s …")
    baseline = measure_static_power(handle, seconds=args.static_seconds,
                                    hz=args.poll_hz,
                                    power_source=args.power_source)
    if p8_ctx.get("restore"):
        p8_ctx["restore"]()
    p_static = baseline["power_w_mean"]
    print(f"[baseline] static power = {p_static:.1f} ± {baseline['power_w_std']:.2f} W  "
          f"(min {baseline['power_w_min']:.1f} W, max {baseline['power_w_max']:.1f} W, "
          f"temp {baseline['temp_c_mean']:.1f}°C, n={baseline['n']})")
    if baseline.get("pstate_filter_note"):
        print(f"[baseline] {baseline['pstate_filter_note']}")
    # Sanity check: if stdev > 5% of mean, the "idle" wasn't really idle
    # (background kernel, clock ramp, another process) — warn so the user
    # knows the P_static they're subtracting is noisy.
    if p_static > 0 and baseline["power_w_std"] / p_static > 0.05:
        print(f"[baseline] WARN idle power stdev is {100*baseline['power_w_std']/p_static:.1f}% "
              f"of mean — another process may be using the GPU, or clocks are ramping. "
              f"Check the baseline plot before trusting dyn-power numbers.")

    # ---- sampler (runs for the whole sweep, phase is toggled per cell) ----
    sampler = PowerSampler(handle, hz=args.poll_hz,
                           power_source=args.power_source)
    print(f"[info] power source: {sampler.power_source}")
    sampler.start()

    # ---- Build a unified list of "plans" — one per (op × dtype × load) ----
    # Each plan has a `build()` callable that allocates tensors just-in-time
    # so we never hold more than one cell's tensors at once.
    plans: list[dict] = []
    l2_reported_bytes = bm.get_l2_bytes(args.device)
    l2_bytes = l2_reported_bytes
    l2_source = "torch"
    if l2_bytes <= 0:
        try:
            profile_l2_mb = float(profile.get("l2_mb", 0) or 0)
        except (TypeError, ValueError):
            profile_l2_mb = 0.0
        if profile_l2_mb > 0:
            l2_bytes = int(profile_l2_mb * (1 << 20))
            l2_source = "gpu_profile"
    if l2_bytes > 0:
        if l2_reported_bytes > 0:
            print(f"[info] L2 cache size: {l2_bytes/(1<<20):.1f} MB "
                  f"(used to classify cache_regime for every row)")
        else:
            print(f"[info] L2 cache size not reported by torch — using "
                  f"{profile_key} profile value {l2_bytes/(1<<20):.1f} MB "
                  f"to classify cache_regime")
    else:
        print("[info] L2 cache size not reported by torch — cache_regime=unknown")
    # Memory budget: drop elementwise loads that would OOM on this HBM.
    # Done up-front (before the sampler starts any work) so the user sees
    # the filter decisions logged at the top of the run.
    try:
        hbm_bytes = int(torch.cuda.get_device_properties(args.device).total_memory)
    except Exception:
        hbm_bytes = 0
    if hbm_bytes > 0:
        print(f"[info] HBM total: {hbm_bytes/(1<<30):.1f} GB "
              f"(per-cell budget: {int(_MEM_SAFETY_FRACTION*100)}% = "
              f"{hbm_bytes*_MEM_SAFETY_FRACTION/(1<<30):.1f} GB)")
    spec_snapshot = _gpu_spec_snapshot(
        profile_key, profile, observed_profile, profile_status, profile_reason,
        gpu_name, cc, args.device, handle, hbm_bytes, l2_reported_bytes,
        l2_bytes, l2_source if l2_bytes > 0 else "")
    print(f"[profile] expected memory={profile.get('memory_type')} "
          f"{profile.get('memory_capacity_gb')}GB, expected L2={profile.get('l2_mb')}MB, "
          f"power envelope={profile.get('power_envelope_w')}W")
    safe_loads = _filter_loads(loads, list(args.ops), list(args.dtypes), hbm_bytes)
    if "elementwise" in cases:
        for dtype in args.dtypes:
            for op in args.ops:
                # In cache-sweep mode, the global `--loads` is ignored and each
                # (op, dtype) gets exactly 3 points sized for L2-resident /
                # L2-partial / DRAM-stream regimes. Otherwise use the global
                # load list (which itself may span all three regimes naturally).
                per_op_loads = (bm.cache_sweep_points(op, dtype, l2_bytes)
                                if args.cache_sweep else safe_loads)
                for N in per_op_loads:
                    plans.append({
                        "category": "elementwise",
                        "mode": "elementwise",
                        "op": op, "dtype": dtype,
                        "load_name": "n_elements", "load_value": N,
                        "build": (lambda op=op, dtype=dtype, N=N:
                                  bm.build(op, dtype, N, device="cuda")),
                    })
    if "matmul" in cases:
        for dtype_label, mode in matmul_variants:
            for K in matmul_sizes:
                plans.append({
                    "category": "matmul",
                    "mode": mode,
                    "op": "matmul", "dtype": dtype_label,
                    "load_name": "K_size", "load_value": K,
                    "build": (lambda K=K, d=dtype_label, m=mode:
                              bm.build_matmul(K, d, m, device="cuda")),
                })
    # DRAM bandwidth probes — opt-in. Pure-streaming kernels at deep
    # l2_hit_0 PLUS small l2_hit_100 baseline points. Both regimes are
    # required so analyze.compute_dram_marginal() can subtract the L2-
    # resident slope from the DRAM-streaming slope and report the
    # incremental L2→HBM cost proxy. Without the small-N baseline,
    # STREAM kernels have no marginal row in `_dram_marginal.csv`.
    if "dram" in cases:
        if args.dram_bw_loads is not None:
            stream_loads = args.dram_bw_loads
        elif l2_bytes > 0:
            # Targets : 2 deep l2_hit_100 baselines (ws = L2/16, L2/8) +
            # 4 l2_hit_0 streaming points (ws = 8/16/32/64 × L2).
            # Use rw=2 / 2-byte dtype for sizing (the probe whose ws/N is
            # smallest, so the others are even more deeply DRAM-bound).
            base = max(1 << 14, int(l2_bytes / (2 * 2)))
            stream_loads = [
                base // 16,                           # l2_hit_100 baseline
                base // 8,                            # l2_hit_100 baseline
                8 * base, 16 * base, 32 * base, 64 * base,  # l2_hit_0
            ]
        else:
            stream_loads = [1 << 22, 1 << 23,
                            1 << 27, 1 << 28, 1 << 29, 1 << 30]
        # Apply the same memory budget filter used for the elementwise sweep.
        stream_loads = _filter_loads(stream_loads,
                                     ["stream_copy", "stream_scale", "stream_triad",
                                      "stream_read", "stream_write"],
                                     list(args.dtypes), hbm_bytes)
        print(f"[dram-bw] {len(stream_loads)} working-set sizes: {stream_loads}")
        for dtype in args.dtypes:
            # 5 STREAM-style probes:
            #   stream_read   pure read  (sum reduction)         → DRAM read pJ/bit
            #   stream_write  pure write (fill_)                  → DRAM write pJ/bit
            #   stream_copy   1R + 1W    (out.copy_ from x)       → cross-check (R+W)/2
            #   stream_scale  1R + 1W    (y = α·x)
            #   stream_triad  2R + 1W    (y = α·x + z)
            for op in ("stream_read",  "stream_write",
                       "stream_copy",  "stream_scale", "stream_triad"):
                for N in stream_loads:
                    plans.append({
                        "category": "elementwise",
                        "mode": "elementwise",
                        "op": op, "dtype": dtype,
                        "load_name": "n_elements", "load_value": N,
                        "build": (lambda op=op, dtype=dtype, N=N:
                                  bm.build(op, dtype, N, device="cuda")),
                    })
    # L2/SRAM resident traffic path probes — custom CUDA extension with an
    # in-kernel repeat loop.  These rows estimate an L2-hit traffic *path*
    # coefficient, not isolated SRAM cell energy.  Sliding-delta rows are
    # validation only; primary estimates come from reg_spin-subtracted
    # l2_read_hit / l2_write_hit fixed-W R sweeps.
    if "l2" in cases:
        l2_windows = [int(mb) * (1 << 20) for mb in args.l2_window_mb]
        l2_cold_pool = int(float(args.l2_cold_pool_gb) * (1 << 30))
        l2_budget = int(hbm_bytes * _MEM_SAFETY_FRACTION) if hbm_bytes > 0 else 0
        print("[l2] L2-hit path probe: windows MiB="
              f"{args.l2_window_mb}, deltas KiB={args.l2_delta_kb}, "
              f"dtypes={args.l2_dtypes}, persisting={args.l2_use_persisting}")
        for dtype in args.l2_dtypes:
            for W in l2_windows:
                if l2_budget and W * 2 > l2_budget:
                    print(f"[l2][memcheck] drop W={W/(1<<20):.1f} MiB "
                          f"(read+write buffers may exceed 25% HBM budget)")
                    continue
                repeats = _parse_l2_repeats(args.l2_repeat_inner, W,
                                            args.l2_target_energy_j,
                                            args.l2_k_guess_pj_bit)
                for R in repeats:
                    for op in ("reg_spin", "l2_read_hit", "l2_write_hit", "l2_copy_hit"):
                        plans.append({
                            "category": "l2",
                            "mode": "l2_probe",
                            "op": op, "dtype": dtype,
                            "load_name": "working_set_bytes", "load_value": W,
                            "build": (lambda op=op, dtype=dtype, W=W, R=R:
                                      bm.build_l2_probe(
                                          op=op, dtype_label=dtype,
                                          working_set_bytes=W,
                                          repeat_inner=R,
                                          device="cuda",
                                          use_persisting_l2=args.l2_use_persisting)),
                        })
                # Sliding validation: use the largest auto/explicit repeat so
                # E(Δ)-E(0) rises above NVML noise.  Delta=0 reference included.
                slide_R = max(repeats)
                for dkb in args.l2_delta_kb:
                    D = int(dkb) * 1024
                    if D > W:
                        print(f"[l2] skip sliding delta {dkb} KiB for "
                              f"W={W/(1<<20):.1f} MiB (delta > W)")
                        continue
                    alloc = max(l2_cold_pool, W + max(D, 4096))
                    if l2_budget and alloc > l2_budget:
                        print(f"[l2][memcheck] drop sliding W={W/(1<<20):.1f} MiB "
                              f"D={dkb} KiB (cold pool {alloc/(1<<30):.2f} GiB "
                              f"> budget {l2_budget/(1<<30):.2f} GiB)")
                        continue
                    plans.append({
                        "category": "l2",
                        "mode": "l2_probe",
                        "op": "l2_sliding_delta", "dtype": dtype,
                        "load_name": "working_set_bytes", "load_value": W,
                        "build": (lambda dtype=dtype, W=W, R=slide_R, D=D, alloc=alloc:
                                  bm.build_l2_probe(
                                      op="l2_sliding_delta", dtype_label=dtype,
                                      working_set_bytes=W,
                                      repeat_inner=R, delta_bytes=D,
                                      cold_pool_bytes=alloc,
                                      device="cuda",
                                      use_persisting_l2=args.l2_use_persisting)),
                    })
    # LLM-shape sweep — opt-in via --llm-shapes. We pre-filter (preset, T)
    # combos by HBM budget using llm_matmul_footprint_bytes(), so the fat
    # lm_head @ T=32768 cell doesn't blow memory on smaller GPUs.
    if "llm-matmul" in cases:
        presets = args.llm_presets or list(bm.LLM_SHAPES.keys())
        unknown = [p for p in presets if p not in bm.LLM_SHAPES]
        if unknown:
            print(f"[error] unknown llm preset(s): {unknown}. "
                  f"choices: {sorted(bm.LLM_SHAPES)}")
            return 1
        llm_ts = args.llm_ts or list(bm.DEFAULT_LLM_TS)
        if args.llm_dtypes is None:
            llm_dtypes = [("bf16", "tc")]
        else:
            llm_dtypes = []
            for v in args.llm_dtypes:
                parts = v.split(":")
                if len(parts) != 2:
                    print(f"bad --llm-dtypes entry {v!r} (expected dtype:mode)")
                    return 1
                llm_dtypes.append(tuple(parts))
        llm_dtypes, dropped_llm_dtypes = gp.filter_with_profile(
            llm_dtypes, gp.profile_llm_dtypes(profile_key),
            label="--llm-dtypes", allow_non_headline=args.allow_non_headline)
        if dropped_llm_dtypes:
            dropped_fmt = [f"{d}:{m}" for d, m in dropped_llm_dtypes]
            print(f"[profile] dropped non-headline --llm-dtypes for {profile_key}: "
                  f"{dropped_fmt} (use --allow-non-headline to keep them)")
        budget = int(hbm_bytes * _MEM_SAFETY_FRACTION) if hbm_bytes > 0 else 0
        dropped_llm: list[tuple[str, int, str, int]] = []
        for dtype_label, mode in llm_dtypes:
            for preset in presets:
                for T in llm_ts:
                    fp = bm.llm_matmul_footprint_bytes(preset, T, dtype_label)
                    if budget > 0 and fp > budget:
                        dropped_llm.append((preset, T, dtype_label, fp))
                        continue
                    plans.append({
                        "category": "matmul_llm",
                        "mode": mode,
                        "op": "matmul", "dtype": dtype_label,
                        "load_name": "T",
                        "load_value": T,
                        "llm_preset": preset,
                        "build": (lambda p=preset, t=T, d=dtype_label, m=mode:
                                  bm.build_llm_matmul(p, t, d, m, device="cuda")),
                    })
        if dropped_llm:
            print(f"[memcheck] dropped {len(dropped_llm)} LLM-shape cells "
                  f"that would exceed the {int(_MEM_SAFETY_FRACTION*100)}% "
                  f"HBM budget:")
            for preset, T, dt, fp in dropped_llm:
                print(f"[memcheck]   {preset:>8s} @ T={T:<6d} {dt:>4s}  "
                      f"≈ {fp/(1<<30):.2f} GB")

    # Fused vs Standalone (G11 / P1.4). full/all suites enable this by
    # default; non-full suites can still opt in with --include-fused. Each
    # variant runs as a single cell at fixed shape (no load sweep). Shape
    # comes from CLI (--attn-shape / --mlp-shape, GPT-OSS 120B defaults).
    if args.include_fused:
        try:
            attn_shape = tuple(int(x) for x in args.attn_shape.split(","))
            assert len(attn_shape) == 6, "expected 6 numbers"
        except (ValueError, AssertionError) as e:
            print(f"[error] bad --attn-shape {args.attn_shape!r}: {e}. "
                  f"Expected B,H_q,H_kv,N_q,N_kv,D_head")
            return 1
        try:
            mlp_shape = tuple(int(x) for x in args.mlp_shape.split(","))
            assert len(mlp_shape) == 3, "expected 3 numbers"
        except (ValueError, AssertionError) as e:
            print(f"[error] bad --mlp-shape {args.mlp_shape!r}: {e}. "
                  f"Expected M,D_in,D_out")
            return 1
        # dtype filter : intersection with {fp16, bf16, fp8}.
        # NOTE — fp8 fused only supports `attention_flash` (TE-based).
        # All other fused variants (qkv_matmul baseline, MLP variants)
        # remain fp16/bf16-only ; G12/P2.4 follow-up will add fp8 paths.
        if args.fused_dtypes is not None:
            fused_dtypes = list(args.fused_dtypes)
        elif not user_set_dtypes:
            fused_dtypes = gp.profile_fused_dtypes(profile_key)
        else:
            fused_dtypes = [d for d in args.dtypes if d in ("fp16", "bf16", "fp8")]
            if not fused_dtypes:
                fused_dtypes = ["bf16"]
                print(f"[fused] --dtypes had no fp16/bf16/fp8; defaulting fused to bf16")
        unsupported = [d for d in fused_dtypes if d not in ("fp16", "bf16", "fp8")]
        if unsupported:
            print(f"[error] fused only supports fp16/bf16/fp8; got {unsupported}.")
            return 1
        fused_dtypes, dropped_fused_dtypes = gp.filter_with_profile(
            fused_dtypes, gp.profile_fused_dtypes(profile_key),
            label="--fused-dtypes", allow_non_headline=args.allow_non_headline)
        if dropped_fused_dtypes:
            print(f"[profile] dropped non-headline --fused-dtypes for {profile_key}: "
                  f"{dropped_fused_dtypes} (use --allow-non-headline to keep them)")
        n_fused_cells = 0
        for dtype in fused_dtypes:
            for variant in bm.FUSED_VARIANTS:
                # fp8 only supports attention_flash for now (other fused
                # variants don't have an fp8 path yet — see G12/P2.4b).
                # `attention_flash_te` for fp8 is identical to
                # `attention_flash` for fp8 (both go through TE), so skip
                # the duplicate to avoid double-measurement.
                if dtype == "fp8" and variant not in ("attention_flash",):
                    continue
                # The "load value" here is the fixed shape — encoded as a
                # human-readable summary so each cell is uniquely keyed.
                if variant in ("attention_flash", "attention_flash_te",
                               "attention_qkv_matmul"):
                    load_value = (f"B{attn_shape[0]}_Hq{attn_shape[1]}_"
                                  f"Hkv{attn_shape[2]}_Nq{attn_shape[3]}_"
                                  f"Nkv{attn_shape[4]}_D{attn_shape[5]}")
                else:
                    load_value = f"M{mlp_shape[0]}_Din{mlp_shape[1]}_Dout{mlp_shape[2]}"
                plans.append({
                    "category": "fused",
                    "mode": "fused",
                    "op": variant, "dtype": dtype,
                    "load_name": "shape",
                    "load_value": load_value,
                    "build": (lambda v=variant, d=dtype,
                                     a=attn_shape, m=mlp_shape,
                                     c=args.fused_causal,
                                     fb=args.fused_fusion_backend:
                              bm.build_fused(v, d, attn_shape=a, mlp_shape=m,
                                             causal=c, fusion_backend=fb,
                                             device="cuda")),
                })
                n_fused_cells += 1
        print(f"[fused] {n_fused_cells} fused cells added "
              f"(dtypes={fused_dtypes}, attn={attn_shape}, mlp={mlp_shape}, "
              f"causal={args.fused_causal}, backend={args.fused_fusion_backend} ; "
              f"fp8 → attention_flash only)")

    total_cells = len(plans)
    print(f"[info] scheduling {total_cells} cells "
          f"(elementwise={sum(1 for p in plans if p['category']=='elementwise')}, "
          f"matmul={sum(1 for p in plans if p['category']=='matmul')}, "
          f"matmul_llm={sum(1 for p in plans if p['category']=='matmul_llm')}, "
          f"l2={sum(1 for p in plans if p['category']=='l2')}, "
          f"fused={sum(1 for p in plans if p['category']=='fused')})")

    rows: list[dict] = []
    # Re-baseline drift bookkeeping. p_static_ts records the wall clock
    # at which the active idle-power estimate was taken, so each row can
    # report `baseline_age_s` — how stale the P_static was when used.
    p_static_ts = time.perf_counter()
    rebaseline_history: list[dict] = [{
        "after_cell": 0,
        "p_static_w": p_static,
        "p_static_w_std": baseline.get("power_w_std", float("nan")),
        "duration_s": baseline.get("duration_s", args.static_seconds),
        "wall_ts": p_static_ts,
        "kind": "initial",
    }]
    # Counters for the post-sweep clip-bias report.
    n_clip_power = 0
    n_clip_energy = 0
    # Variants (op, dtype, mode, llm_preset) that have already crashed once.
    # Any later cell sharing the same key gets skipped without retry — the
    # CUDA fault repeats deterministically and we'd just waste time.
    broken_variants: set[tuple[str, str, str, str]] = set()
    fatal_error: str | None = None
    fused_hint_printed = False
    try:
        for i, plan in enumerate(plans, 1):
            # Skip cells from variants we've already proven broken in this run.
            variant_key = _variant_key(plan)
            if variant_key in broken_variants:
                print(f"\n[{i}/{total_cells}] {plan['op']}_{plan['dtype']}_{plan['mode']}"
                      + (f"·{plan['llm_preset']}" if plan.get("llm_preset") else "")
                      + f"  {plan['load_name']}={plan['load_value']}  — SKIPPED "
                      f"(this variant crashed earlier)")
                continue
            # Periodic re-baseline. Done BEFORE the per-cell cooldown so
            # the fresh idle measurement also serves as a thermal
            # stabilisation window. Skipped on cell #1 (we already have a
            # fresh baseline from program start) and when the user opts
            # out by passing --rebaseline-every 0.
            if (args.rebaseline_every > 0 and i > 1
                    and (i - 1) % args.rebaseline_every == 0):
                # The active sampler is logging this idle period — tag it
                # so analyse can find/exclude these intervals.
                sampler.set_phase(f"rebaseline_after_{i-1}")
                # Drain any pending GPU work before measuring idle.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Same P8-wait as the initial baseline so re-baselines
                # don't get hit by the P0 hysteresis that always follows
                # an active cell.
                if args.pstate_idle_wait > 0:
                    wait_for_pstate_idle(handle,
                                         timeout_s=args.pstate_idle_wait,
                                         verbose=False)
                # Force P8 around each rebaseline too — quiet (verbose=False)
                # so a 130-cell sweep doesn't print 6 force-p8 noise blocks.
                rb_p8 = (force_p8_for_measurement(handle, verbose=False,
                                                  use_sudo=args.sudo_pstate)
                          if args.pstate_idle_wait > 0
                          else {"success": False, "restore": None})
                rb = measure_static_power(handle,
                                          seconds=args.rebaseline_seconds,
                                          hz=args.poll_hz,
                                          power_source=args.power_source)
                if rb_p8.get("restore"):
                    rb_p8["restore"]()
                new_p = rb["power_w_mean"]
                drift = new_p - p_static
                print(f"\n[rebaseline @ cell {i-1}/{total_cells}] "
                      f"P_static {p_static:.2f} W → {new_p:.2f} W "
                      f"(Δ {drift:+.2f} W, σ {rb.get('power_w_std', 0):.2f} W, "
                      f"{rb.get('temp_c_mean', -1):.1f}°C)")
                p_static = new_p
                p_static_ts = time.perf_counter()
                rebaseline_history.append({
                    "after_cell": i - 1,
                    "p_static_w": p_static,
                    "p_static_w_std": rb.get("power_w_std", float("nan")),
                    "duration_s": rb.get("duration_s", args.rebaseline_seconds),
                    "wall_ts": p_static_ts,
                    "kind": "periodic",
                })
                sampler.set_phase("gap")

            label = f"{plan['op']}_{plan['dtype']}_{plan['mode']}"
            print(f"\n[{i}/{total_cells}] {label}  {plan['load_name']}={plan['load_value']}")

            try:
                spec = plan["build"]()
            except Exception as e:
                err_msg = str(e)
                # Same fatal-marker / broken_variants logic as the
                # run_measurement except below. A build() failure on Blackwell
                # fp8_te typically means the prior cell's measurement loop
                # poisoned the CUDA context, and the *first* CUDA call this
                # cell makes (the warmup inside the build helper) is what
                # surfaces it. Without this, we'd "skip" each remaining cell
                # of the broken variant with a fresh build attempt — burning
                # time on a deterministic, unrecoverable fault.
                fatal_markers = (
                    "illegal memory access",
                    "CUDA error",
                    "an asynchronous CUDA error",
                    "device-side assert",
                    "CUDA call failed",
                )
                fatal = any(m in err_msg for m in fatal_markers)
                variant_key = _variant_key(plan)
                broken_variants.add(variant_key)
                if fatal:
                    print(f"  !! FATAL CUDA error during build at {label} "
                          f"{plan['load_name']}={plan['load_value']}: {err_msg}")
                    if plan.get("category") == "fused" and not fused_hint_printed:
                        _print_fused_failure_hint(err_msg)
                        fused_hint_printed = True
                    print(f"  !! CUDA context is now unrecoverable — "
                          f"flushing {len(rows)} completed cells to CSV "
                          f"and exiting early. Re-run without the offending "
                          f"variant (e.g. --matmul-variants without fp8:te, "
                          f"--no-fused, or skip --llm-shapes).")
                    fatal_error = err_msg
                    break
                print(f"  ! build failed: {err_msg}  — skipped")
                if plan.get("category") == "fused" and not fused_hint_printed:
                    _print_fused_failure_hint(err_msg)
                    fused_hint_printed = True
                continue
            _apply_effective_l2_cache_regime(spec, l2_bytes, l2_source)
            tag = spec.compute_unit + ("  [EMULATED]" if spec.emulated else "")
            print(f"  compute_unit: {tag}")
            print(f"  cache_regime: {spec.cache_regime}")
            if spec.notes:
                print(f"  note: {spec.notes}")

            cooldown_info = {"final_temp_c": -1, "elapsed_s": 0.0, "reached": False}
            if not args.no_cooldown and args.cooldown_c > 0:
                cooldown_info = wait_for_cooldown(
                    handle, target_c=args.cooldown_c,
                    timeout_s=args.cooldown_timeout,
                    min_s=args.cooldown_min_s,
                    verbose=False)
                print(f"  cooldown: {cooldown_info['elapsed_s']:.1f}s → "
                      f"{cooldown_info['final_temp_c']}°C "
                      f"{'(reached)' if cooldown_info['reached'] else '(TIMEOUT)'}")

            try:
                meas = run_measurement(spec, sampler, window_ms=args.window_ms)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  ! OOM at load={plan['load_value']}: {e}")
                del spec
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            except Exception as e:
                err_msg = str(e)
                # Some errors poison the CUDA context for the entire process
                # — every subsequent CUDA call (including empty_cache and the
                # next cell's tensor allocation) will re-raise the SAME error.
                # In that case continuing is pointless: we'd just produce
                # garbage. Detect those, dump the rows we already have, and
                # exit cleanly so the user keeps everything completed so far.
                fatal_markers = (
                    "illegal memory access",
                    "CUDA error",
                    "an asynchronous CUDA error",
                    "device-side assert",
                    "CUDA call failed",
                )
                fatal = any(m in err_msg for m in fatal_markers)
                # Track that THIS variant is broken so subsequent cells of the
                # same (op, dtype, mode, llm_preset) get skipped without trying
                # — the CUDA context is the same, the fault repeats.
                variant_key = _variant_key(plan)
                broken_variants.add(variant_key)
                if fatal:
                    print(f"  !! FATAL CUDA error at {label} "
                          f"{plan['load_name']}={plan['load_value']}: {err_msg}")
                    if plan.get("category") == "fused" and not fused_hint_printed:
                        _print_fused_failure_hint(err_msg)
                        fused_hint_printed = True
                    print(f"  !! CUDA context is now unrecoverable — "
                          f"flushing {len(rows)} completed cells to CSV "
                          f"and exiting early. Re-run without the offending "
                          f"variant (e.g. --no-matmul / drop --llm-shapes / "
                          f"--matmul-variants without fp8:te / --no-fused).")
                    fatal_error = err_msg
                    break  # exit the for-loop; the finally: + post-loop save
                           # still write whatever we accumulated.
                print(f"  ! run failed: {err_msg}")
                if plan.get("category") == "fused" and not fused_hint_printed:
                    _print_fused_failure_hint(err_msg)
                    fused_hint_printed = True
                del spec
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            # Energy decomposition: we subtract the baseline / static power
            # from the measured average power to get the dynamic
            # (workload-attributable) component. The clip-to-zero is the
            # standard convention because negative dyn is non-physical, but
            # at very low load NVML noise + P_static drift CAN push the raw
            # difference slightly below zero — clipping then biases the
            # low-load mean upward, which flattens the log-log regression
            # slope. We therefore record BOTH the raw (signed) value and
            # the clipped one so analyse can quantify the bias and warn.
            dyn_power_raw = meas["avg_power_w"] - p_static
            dyn_power = max(0.0, dyn_power_raw)
            static_energy = p_static * meas["wall_s"]
            dyn_energy_raw = meas["total_energy_j"] - static_energy
            dyn_energy = max(0.0, dyn_energy_raw)
            if dyn_power_raw < 0: n_clip_power += 1
            if dyn_energy_raw < 0: n_clip_energy += 1
            baseline_age_s = time.perf_counter() - p_static_ts

            total_elements = spec.n_elements * meas["iters"]
            total_flops = spec.flops_per_call * meas["iters"]
            headline_eligible, headline_status, headline_reason = gp.headline_status(
                profile_key,
                category=plan["category"],
                op=plan["op"],
                dtype=plan["dtype"],
                mode=plan["mode"],
                compute_unit=spec.compute_unit,
                emulated=bool(spec.emulated),
                observed_cc=observed_cc,
            )

            row = {
                "gpu": gpu_name,
                "compute_cap": f"{cc[0]}.{cc[1]}",
                "gpu_profile": profile_key,
                "gpu_profile_status": profile_status,
                "gpu_profile_reason": profile_reason,
                "observed_profile": observed_profile,
                "memory_type_expected": spec_snapshot.get("memory_type_expected", ""),
                "memory_capacity_expected_gb": spec_snapshot.get("memory_capacity_expected_gb", ""),
                "memory_total_gb": spec_snapshot.get("memory_total_gb", ""),
                "peak_bw_expected_gbps": spec_snapshot.get("peak_bw_expected_gbps", ""),
                "l2_expected_mb": spec_snapshot.get("l2_expected_mb", ""),
                "l2_reported_mb": spec_snapshot.get("l2_reported_mb", ""),
                "l2_effective_mb": spec_snapshot.get("l2_effective_mb", ""),
                "l2_source": spec_snapshot.get("l2_source", ""),
                "power_envelope_expected_w": spec_snapshot.get("power_envelope_expected_w", ""),
                "power_limit_w": spec_snapshot.get("power_limit_w", ""),
                "native_fp8_headline_allowed": spec_snapshot.get("native_fp8_headline_allowed", ""),
                "headline_eligible": headline_eligible,
                "headline_status": headline_status,
                "headline_reason": headline_reason,
                "category": plan["category"],
                "op": plan["op"],
                "dtype": plan["dtype"],
                "mode": plan["mode"],
                "variant": spec.name,
                # Empty except for matmul_llm — identifies which LLM layer
                # shape (qkv / mlp1 / lm_head / …) this row came from. Lets
                # analyze group the sweep by layer role.
                "llm_preset": plan.get("llm_preset", ""),
                # HW path the op actually runs on — see benchmarks.BenchSpec.
                # "CUDA core" | "Tensor Core" | "Tensor Core (FP16 fallback)"
                "compute_unit": spec.compute_unit,
                # 1 if this is NOT the HW path a naive reader would assume
                # (fp8 elementwise anywhere, fp8_te on pre-Hopper, etc.)
                "emulated": int(bool(spec.emulated)),
                # Fine-grained classification — see BenchSpec.path_semantics docstring.
                #   native_or_standard / emulated_cast_compute_cast /
                #   native_or_te_fp8_tensorcore / te_fp16_fallback
                "path_semantics": spec.path_semantics,
                # Cache-locality regime classified from working-set vs L2.
                # "l2_resident" / "l2_partial" / "dram_stream" / "unknown"
                "cache_regime": spec.cache_regime,
                "n_elements": spec.n_elements,
                "shape": "x".join(str(s) for s in spec.shape),
                "load_name": plan["load_name"],
                "load_value": plan["load_value"],
                "iters": meas["iters"],
                "ms_per_call": f"{meas['ms_per_call']:.4f}",
                "wall_s": f"{meas['wall_s']:.4f}",
                "flops_per_call": spec.flops_per_call,
                "total_flops": total_flops,
                "total_elements": total_elements,
                "static_power_w": f"{p_static:.3f}",
                # baseline_age_s — how stale (s) the active P_static was
                # when this cell ran. 0 if --rebaseline-every is on and we
                # just refreshed; up to ~30 min if not.
                "baseline_age_s": f"{baseline_age_s:.1f}",
                "avg_power_w":    f"{meas['avg_power_w']:.3f}",
                "dyn_power_w":    f"{dyn_power:.3f}",
                # Pre-clip dynamic — sometimes negative when noise + drift
                # nudge a low-load measurement below P_static. Lets analyse
                # see exactly how much clipping inflated the low end.
                "dyn_power_w_raw":  f"{dyn_power_raw:.3f}",
                "total_energy_j":  f"{meas['total_energy_j']:.4f}",
                "static_energy_j": f"{static_energy:.4f}",
                "dyn_energy_j":    f"{dyn_energy:.4f}",
                "dyn_energy_j_raw":f"{dyn_energy_raw:.4f}",
                "j_per_element_total": (f"{meas['total_energy_j']/total_elements:.3e}"
                                        if total_elements else ""),
                "j_per_element_dyn":   (f"{dyn_energy/total_elements:.3e}"
                                        if total_elements else ""),
                # FLOP-normalised energy is undefined for the STREAM probes
                # (stream_copy has flops_per_call=0 by design — pure data
                # movement). Write empty so pandas reads it as NaN and
                # analyse can drop those rows from J/FLOP regressions
                # without ZeroDivisionError mid-sweep.
                "j_per_flop_total":    (f"{meas['total_energy_j']/total_flops:.3e}"
                                        if total_flops else ""),
                "j_per_flop_dyn":      (f"{dyn_energy/total_flops:.3e}"
                                        if total_flops else ""),
                "start_temp_c":      cooldown_info["final_temp_c"],
                "avg_temp_c":        f"{meas['avg_temp_c']:.1f}",
                "peak_temp_c":       meas["peak_temp_c"],
                "temp_rise_c":       (meas["peak_temp_c"] - cooldown_info["final_temp_c"]
                                      if cooldown_info["final_temp_c"] >= 0 else -1),
                "cooldown_elapsed_s": f"{cooldown_info['elapsed_s']:.2f}",
                "cooldown_reached":   int(bool(cooldown_info["reached"])),
                "sm_clk_mhz":   meas["sm_clk_mhz"],
                "mem_clk_mhz":  meas["mem_clk_mhz"],
                # L2/SRAM resident traffic probe metadata. Blank for non-L2 rows.
                "working_set_bytes": "",
                "repeat_inner": "",
                "delta_bytes": "",
                "estimated_l2_read_bits": "",
                "estimated_l2_write_bits": "",
                "estimated_l2_total_bits": "",
                "estimated_hbm_refill_bits": "",
                "l2_policy": "",
                "block_size": "",
                "grid_size": "",
                "kernel_version": "",
                "cold_pool_bytes": "",
                "notes":        spec.notes,
            }
            extra = getattr(spec, "extra", {}) or {}
            if extra:
                outer_iters = int(meas["iters"])
                row.update({
                    "working_set_bytes": extra.get("working_set_bytes", ""),
                    "repeat_inner": extra.get("repeat_inner", ""),
                    "delta_bytes": extra.get("delta_bytes", ""),
                    "estimated_l2_read_bits": int(extra.get("estimated_l2_read_bits_per_call", 0)) * outer_iters,
                    "estimated_l2_write_bits": int(extra.get("estimated_l2_write_bits_per_call", 0)) * outer_iters,
                    "estimated_l2_total_bits": int(extra.get("estimated_l2_total_bits_per_call", 0)) * outer_iters,
                    "estimated_hbm_refill_bits": int(extra.get("estimated_hbm_refill_bits_per_call", 0)) * outer_iters,
                    "l2_policy": extra.get("l2_policy", ""),
                    "block_size": extra.get("block_size", ""),
                    "grid_size": extra.get("grid_size", ""),
                    "kernel_version": extra.get("kernel_version", ""),
                    "cold_pool_bytes": extra.get("cold_pool_bytes", ""),
                })
            rows.append(row)
            print(f"  E_total={meas['total_energy_j']:.3f} J, "
                  f"E_dyn={dyn_energy:.3f} J, "
                  f"P_avg={meas['avg_power_w']:.1f} W "
                  f"(dyn {dyn_power:.1f} W), "
                  f"T={cooldown_info['final_temp_c']}→{meas['peak_temp_c']}°C "
                  f"(Δ{row['temp_rise_c']}, avg {meas['avg_temp_c']:.1f}°C), "
                  f"iters={meas['iters']}, {meas['wall_s']:.2f}s")
            print(f"  J/elem (dyn)={row['j_per_element_dyn']}  "
                  f"J/FLOP (dyn)={row['j_per_flop_dyn']}")

            # Free before the next cell to keep peak memory bounded.
            del spec
            # Force any pending async CUDA errors to surface NOW — TE's
            # amax-buffer fault on Blackwell often stays queued until the
            # next CUDA call, which is the *next cell's* build(). Synching
            # here means we attribute the fault to the cell that caused
            # it, mark that variant broken, and exit cleanly.
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                err_msg = str(e)
                variant_key = _variant_key(plan)
                broken_variants.add(variant_key)
                fatal_markers = (
                    "illegal memory access",
                    "CUDA error",
                    "an asynchronous CUDA error",
                    "device-side assert",
                    "CUDA call failed",
                )
                if any(m in err_msg for m in fatal_markers):
                    print(f"  !! deferred CUDA error surfaced after {label} "
                          f"{plan['load_name']}={plan['load_value']}: {err_msg}")
                    print(f"  !! CUDA context is poisoned — flushing "
                          f"{len(rows)} completed cells and exiting.")
                    fatal_error = err_msg
                    break
                # Non-fatal: just skip this variant going forward.
                print(f"  ! post-cell cleanup failed: {err_msg}  "
                      f"— marking {variant_key[0]}_{variant_key[1]}_{variant_key[2]} broken")
    finally:
        try:
            sampler.stop()
        except Exception:
            pass
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # ---- write CSV ---------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}.csv"
    csv_saved = False
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        csv_saved = True
        if fatal_error:
            print(f"\n[save] {csv_path}  ({len(rows)} rows — partial; "
                  f"sweep aborted by '{fatal_error}')")
            print(f"[recover] all completed cells are saved. To finish "
                  f"the sweep, re-run dropping the variant that crashed "
                  f"(e.g. --matmul-variants without fp8:te, or skip "
                  f"--llm-shapes).")
        else:
            print(f"\n[save] {csv_path}  ({len(rows)} rows)")
    else:
        # Empty rows is normal for a soc-only run (cases={"soc"}). In any
        # other case it's a sign the sweep produced nothing, which is a
        # genuine error.
        if cases == {"soc"}:
            print("\n[info] no per-cell rows — soc-only run; skipping main CSV")
        else:
            print("\n[warn] no rows collected — nothing saved")
            return 2

    spec_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}_gpu_spec_snapshot.csv"
    with open(spec_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in spec_snapshot.items():
            w.writerow([k, v])
    print(f"[save] {spec_path}")

    # ---- clip-bias report ------------------------------------------------
    # Tells the user how often the dyn = max(0, raw) clip fired. Frequent
    # clipping at low load is a sign that P_static was too high (drift),
    # not necessarily that the kernel is genuinely above idle.
    n_total = len(rows)
    if n_total > 0:
        pct_p = 100.0 * n_clip_power  / n_total
        pct_e = 100.0 * n_clip_energy / n_total
        if n_clip_power > 0 or n_clip_energy > 0:
            print(f"\n[clip] dyn_power_w  clipped to 0 on {n_clip_power}/{n_total} cells ({pct_p:.1f}%)")
            print(f"[clip] dyn_energy_j clipped to 0 on {n_clip_energy}/{n_total} cells ({pct_e:.1f}%)")
            print(f"[clip] inspect dyn_power_w_raw / dyn_energy_j_raw columns to see "
                  f"the unclipped residual; large clip rate (≥ 20%) at small N means "
                  f"P_static drifted above the true idle — consider --rebaseline-every 20.")
        else:
            print(f"\n[clip] no cells required clipping — dyn_raw stayed positive on all "
                  f"{n_total} cells.")

    # ---- re-baseline history sidecar -------------------------------------
    # One row per baseline measurement (initial + each periodic refresh).
    # analyse can plot P_static(time) to verify whether drift is monotone
    # (rack warming up) vs random (background noise).
    rebaseline_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}_rebaseline.csv"
    with open(rebaseline_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["after_cell", "kind", "p_static_w", "p_static_w_std",
                    "duration_s", "wall_ts"])
        for r in rebaseline_history:
            w.writerow([r["after_cell"], r["kind"],
                        f"{r['p_static_w']:.3f}",
                        f"{r['p_static_w_std']:.3f}",
                        f"{r['duration_s']:.2f}",
                        f"{r['wall_ts']:.2f}"])
    if len(rebaseline_history) > 1:
        # Quick drift summary
        ps_vals = [r["p_static_w"] for r in rebaseline_history]
        drift_total = ps_vals[-1] - ps_vals[0]
        drift_max = max(ps_vals) - min(ps_vals)
        print(f"[rebaseline] {len(rebaseline_history)} P_static measurements over the sweep; "
              f"net drift {drift_total:+.2f} W, range {drift_max:.2f} W")
    print(f"[save] {rebaseline_path}")

    # Dump the idle / static-power baseline as a sidecar so analyze.py can
    # draw a P_static(t) plot (flat line = idle was clean; sawtooth = bad).
    # We also emit a 1-row stats CSV so downstream tools don't have to
    # re-compute the mean/stdev.
    baseline_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}_baseline.csv"
    with open(baseline_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "power_w", "temp_c"])
        for t_s, p_w, t_c in baseline.get("samples", []):
            w.writerow([f"{t_s:.4f}", f"{p_w:.3f}", t_c])
    stats_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}_baseline_stats.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "duration_s", "hz", "n",
                    "p_static_w_mean", "p_static_w_std",
                    "p_static_w_min", "p_static_w_max",
                    "temp_c_mean", "temp_c_min", "temp_c_max"])
        w.writerow([gpu_name, baseline.get("duration_s", args.static_seconds),
                    baseline.get("hz", args.poll_hz), baseline["n"],
                    f"{baseline['power_w_mean']:.3f}",
                    f"{baseline['power_w_std']:.3f}",
                    f"{baseline['power_w_min']:.3f}",
                    f"{baseline['power_w_max']:.3f}",
                    f"{baseline['temp_c_mean']:.2f}",
                    baseline.get("temp_c_min", -1),
                    baseline.get("temp_c_max", -1)])
    print(f"[save] {baseline_path}  ({len(baseline.get('samples', []))} idle samples)")
    print(f"[save] {stats_path}   (P_static = {p_static:.2f} W)")

    # Also dump raw power samples for the whole run so analyze.py can draw a
    # global timeline (power, temp, clocks) with phase labels.
    raw_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}_samples.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "power_w", "temp_c", "sm_mhz", "mem_mhz",
                    "gpu_util", "mem_util", "phase"])
        for s in sampler.samples:
            w.writerow([f"{s.t:.4f}", f"{s.power_w:.3f}", s.temp_c,
                        s.sm_mhz, s.mem_mhz, s.gpu_util, s.mem_util, s.phase])
    print(f"[save] {raw_path}  ({len(sampler.samples)} NVML samples)")

    # ---- SoC envelope (run as a sidecar phase if requested) -----------------
    # When 'soc' is in cases, run the static / max / leakage phases that
    # used to live in soc_power_bench.py. Outputs are filename-tied to the
    # main sweep CSV (same gpu / timestamp / tag) so analyze and
    # multi_gpu_analysis can find them via the existing pattern.
    if "soc" in cases:
        # Re-init NVML — it was shut down in the sweep's finally:. Resolve
        # the same physical GPU via PCI bus id (CUDA_VISIBLE_DEVICES safe).
        pynvml.nvmlInit()
        try:
            soc_handle, _ = resolve_nvml_handle(args.device)

            print(f"\n========== SoC envelope ==========")
            print(f"[soc] static={args.soc_static_seconds}s  "
                  f"max={args.soc_max_seconds}s  "
                  f"leakage={args.soc_leakage_cycles}x("
                  f"{args.soc_leakage_stress_s}s+{args.soc_leakage_decay_s}s)")

            # GEMM build is DEFERRED until after the static phase. The 5x
            # warmup inside _make_matmul_*() pins clocks to P0 and NVIDIA
            # driver hysteresis keeps them there for ~30..60s after — if
            # we built first, static would read the boost-clock idle
            # power (110-120 W on H100), not the true cold idle (~70 W).
            # See README §3.4 / §9.6 for the full discussion.
            soc_spec = None

            soc_sampler = PowerSampler(soc_handle, hz=args.poll_hz,
                                       power_source=args.power_source)
            print(f"[soc] power source: {soc_sampler.power_source}")
            soc_sampler.start()
            soc_sampler.set_phase("startup")

            soc_summary = {"gpu_name": gpu_name, "gpu_slug": gpu_slug,
                           "cc_major": cc[0], "cc_minor": cc[1],
                           "matmul_K": args.soc_matmul_K,
                           "dtype": args.soc_dtype, "mode": args.soc_mode,
                           "power_source": soc_sampler.power_source,
                           "sample_hz": args.poll_hz}
            soc_cycles_meta: list[dict] = []
            soc_static_t = soc_max_t = None
            soc_fatal: str | None = None

            try:
                # ---- Static phase first (cold-idle, no GEMM warmup yet) ---
                if args.cooldown_c > 0:
                    soc_sampler.set_phase("cooldown_pre_static")
                    wait_for_cooldown(soc_handle, target_c=args.cooldown_c,
                                      timeout_s=args.cooldown_timeout, verbose=False)
                print(f"\n[soc/static] {args.soc_static_seconds}s idle …")
                soc_static_t = sb.phase_static(soc_sampler, args.soc_static_seconds)

                # ---- NOW build the GEMM ----------------------------------
                try:
                    soc_spec = bm.build_matmul(args.soc_matmul_K, args.soc_dtype,
                                               args.soc_mode, device="cuda")
                    print(f"[soc] GEMM: K={args.soc_matmul_K}  "
                          f"{args.soc_dtype}/{args.soc_mode}  "
                          f"compute_unit={soc_spec.compute_unit}")
                except Exception as e:
                    print(f"[soc] failed to build GEMM "
                          f"({args.soc_dtype}/{args.soc_mode} K={args.soc_matmul_K}): {e}")
                    soc_fatal = f"build: {e}"
                    soc_spec = None

                # ---- Max phase --------------------------------------------
                if soc_spec is not None and not soc_fatal:
                    if args.cooldown_c > 0:
                        soc_sampler.set_phase("cooldown_pre_max")
                        wait_for_cooldown(soc_handle, target_c=args.cooldown_c,
                                          timeout_s=args.cooldown_timeout, verbose=False)
                    print(f"[soc/max] {args.soc_max_seconds}s GEMM …")
                    try:
                        soc_max_t = sb.phase_max(soc_sampler, soc_spec, args.soc_max_seconds)
                    except RuntimeError as e:
                        print(f"[soc/max] failed: {e}")
                        soc_fatal = f"max: {e}"

                # ---- Leakage phase ----------------------------------------
                if soc_spec is not None and not soc_fatal:
                    if args.cooldown_c > 0:
                        soc_sampler.set_phase("cooldown_pre_leakage")
                        wait_for_cooldown(soc_handle, target_c=args.cooldown_c,
                                          timeout_s=args.cooldown_timeout, verbose=False)
                    print(f"[soc/leakage] {args.soc_leakage_cycles} cycles …")
                    try:
                        soc_cycles_meta = sb.phase_leakage(
                            soc_sampler, soc_spec,
                            args.soc_leakage_cycles,
                            args.soc_leakage_stress_s,
                            args.soc_leakage_decay_s)
                    except RuntimeError as e:
                        print(f"[soc/leakage] failed mid-cycle: {e}")
                        soc_fatal = f"leakage: {e}"
            finally:
                soc_sampler.stop()

            if soc_fatal:
                soc_summary["fatal_error"] = soc_fatal

            # ---- SoC stats from the in-memory timeseries -------------
            soc_samples = soc_sampler.samples

            def _slice_stats(t0: float, t1: float) -> dict:
                ps = [s.power_w for s in soc_samples
                      if t0 <= s.t <= t1 and s.power_w >= 0]
                ts = [s.temp_c for s in soc_samples
                      if t0 <= s.t <= t1 and s.temp_c >= 0]
                if not ps:
                    return {"power_mean": -1, "power_peak": -1,
                            "temp_mean": -1, "temp_peak": -1, "n": 0}
                return {
                    "power_mean": sum(ps) / len(ps),
                    "power_peak": max(ps),
                    "temp_mean":  (sum(ts) / len(ts)) if ts else -1,
                    "temp_peak":  max(ts) if ts else -1,
                    "n": len(ps),
                }

            if soc_static_t:
                st = _slice_stats(*soc_static_t)
                soc_summary["static_seconds"]      = args.soc_static_seconds
                soc_summary["static_power_w_mean"] = round(st["power_mean"], 3)
                soc_summary["static_power_w_peak"] = round(st["power_peak"], 3)
                soc_summary["static_temp_c_mean"]  = round(st["temp_mean"], 1)
                soc_summary["static_temp_c_peak"]  = st["temp_peak"]
            if soc_max_t:
                mx = _slice_stats(*soc_max_t)
                soc_summary["max_seconds"]         = args.soc_max_seconds
                soc_summary["max_power_w_mean"]    = round(mx["power_mean"], 3)
                soc_summary["max_power_w_peak"]    = round(mx["power_peak"], 3)
                soc_summary["max_temp_c_mean"]     = round(mx["temp_mean"], 1)
                soc_summary["max_temp_c_peak"]     = mx["temp_peak"]

            leak_powers, leak_temps = [], []
            for c in soc_cycles_meta:
                d0 = c["decay_t0"]
                st = _slice_stats(d0, d0 + args.soc_leak_window_s)
                c["hot_power_w_mean"] = round(st["power_mean"], 3)
                c["hot_temp_c_mean"]  = round(st["temp_mean"], 1)
                c["hot_temp_c_peak"]  = st["temp_peak"]
                st_stress = _slice_stats(c["stress_t0"], c["stress_t1"])
                c["stress_temp_c_peak"] = st_stress["temp_peak"]
                c["stress_power_w_mean"] = round(st_stress["power_mean"], 3)
                if st["power_mean"] > 0:
                    leak_powers.append(st["power_mean"])
                    leak_temps.append(st["temp_mean"])
            if leak_powers:
                soc_summary["leakage_cycles"]       = len(leak_powers)
                soc_summary["leakage_window_s"]     = args.soc_leak_window_s
                soc_summary["leakage_power_w_mean"] = round(sum(leak_powers)/len(leak_powers), 3)
                soc_summary["leakage_power_w_min"]  = round(min(leak_powers), 3)
                soc_summary["leakage_power_w_max"]  = round(max(leak_powers), 3)
                soc_summary["leakage_temp_c_mean"]  = round(sum(leak_temps)/len(leak_temps), 1)
                if "static_power_w_mean" in soc_summary:
                    soc_summary["leakage_minus_static_w"] = round(
                        soc_summary["leakage_power_w_mean"] -
                        soc_summary["static_power_w_mean"], 3)

            # ---- write SoC sidecars -----------------------------------
            stem = csv_path.stem
            soc_summary_path = out_dir / f"{stem}_soc_summary.csv"
            with open(soc_summary_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in soc_summary.items():
                    w.writerow([k, v])
                w.writerow([])
                if soc_cycles_meta:
                    w.writerow(["leakage_cycle", "stress_temp_c_peak",
                                "stress_power_w_mean", "hot_temp_c_peak",
                                "hot_power_w_mean", "hot_minus_static_w"])
                    stat_w = soc_summary.get("static_power_w_mean", 0) or 0
                    for c in soc_cycles_meta:
                        w.writerow([c["cycle"] + 1,
                                    c.get("stress_temp_c_peak", -1),
                                    c.get("stress_power_w_mean", -1),
                                    c.get("hot_temp_c_peak", -1),
                                    c.get("hot_power_w_mean", -1),
                                    round(c.get("hot_power_w_mean", 0) - stat_w, 3)])
            print(f"[save] {soc_summary_path}")

            soc_ts_path = out_dir / f"{stem}_soc_timeseries.csv"
            with open(soc_ts_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "power_w", "temp_c", "sm_mhz",
                            "mem_mhz", "gpu_util", "phase"])
                for s in soc_samples:
                    w.writerow([f"{s.t:.4f}", f"{s.power_w:.4f}",
                                s.temp_c, s.sm_mhz, s.mem_mhz,
                                s.gpu_util, s.phase])
            print(f"[save] {soc_ts_path}")

            phase_png            = out_dir / f"{stem}_soc_phases.png"
            leakage_png          = out_dir / f"{stem}_soc_leakage.png"
            leakage_enlarged_png = out_dir / f"{stem}_soc_leakage_enlarged.png"
            summary_png          = out_dir / f"{stem}_soc_summary.png"
            sb.plot_phase_timeline(soc_samples, soc_summary, phase_png)
            leakage_t_png = out_dir / f"{stem}_soc_leakage_temperature.png"
            if soc_cycles_meta:
                sb.plot_leakage_decay(soc_samples, soc_cycles_meta,
                                      soc_summary, leakage_png)
                sb.plot_leakage_decay_zoomed(soc_samples, soc_cycles_meta,
                                             soc_summary, leakage_enlarged_png,
                                             x_max=3.0, y_min=50.0, y_max=150.0)
                # Leakage(T) curve fit (PR B / P2.2 / G7) — Arrhenius +
                # linear sanity-check on the (T, P) decay-window pairs.
                leak_t_fit = sb.fit_leakage_temperature(soc_samples, soc_cycles_meta)
                if leak_t_fit["n_points"] >= 5:
                    for k, v in leak_t_fit.items():
                        if isinstance(v, tuple):
                            soc_summary[f"leakage_t_{k}_min"] = v[0]
                            soc_summary[f"leakage_t_{k}_max"] = v[1]
                        else:
                            soc_summary[f"leakage_t_{k}"] = v
                    sb.plot_leakage_temperature(
                        soc_samples, soc_cycles_meta, leak_t_fit,
                        leakage_t_png, gpu_name)
            sb.plot_summary_bars(soc_summary, summary_png)
            print(f"[save] {phase_png}")
            if soc_cycles_meta:
                print(f"[save] {leakage_png}")
                print(f"[save] {leakage_enlarged_png}")
            print(f"[save] {summary_png}")
        finally:
            try: pynvml.nvmlShutdown()
            except Exception: pass

    if csv_saved:
        print("\nnext: python3 analyze.py {}  # generate linearity plots".format(csv_path))
        # Machine-readable last line — run_bench.sh greps this to chain
        # `analyze.py <csv>` automatically. Keep the prefix exact.
        print(f"[OUTPUT_CSV] {csv_path}")
    else:
        print("\n[info] no benchmark CSV emitted; analysis plots are only available for per-cell sweeps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
