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
import re
import sys
import time
from pathlib import Path

import pynvml
import torch

import benchmarks as bm
from power_monitor import PowerSampler, measure_static_power, wait_for_cooldown
import preflight


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
# elements.  Dense around the TC sweet spot (1024–4096) + one tiny (512,
# launch overhead) + three big (6144, 8192, 12288) to expose BW saturation
# on FP32 and push matmul into dram_stream cache regime. K=12288 fp32 ≈
# 1.7 GB — fits on both A100/H100 with margin.
DEFAULT_MATMUL_SIZES = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288]
QUICK_MATMUL_SIZES = [1024, 2048, 4096]


# Fraction of the device's total HBM that one elementwise cell may consume.
# At 0.25, an 80 GB A100 (HBM2E) caps at 20 GB per-cell; 80 GB H100
# caps at 20 GB; 40 GB cards (older A100 / V100) cap at 10 GB — still
# enough for 1B fp16 mul (6 GB) but will drop the 1G fp8 cell (9 GB).
_MEM_SAFETY_FRACTION = 0.25


def _cell_memory_bytes(op: str, dtype_label: str, n_elements: int) -> int:
    """Conservative upper bound on DRAM footprint of a single elementwise cell.

    mul/add           : 3 tensors (a, b, out)
    gelu              : 2 tensors (in, out)
    softmax/layernorm : 2 tensors (in, out) — reduction in fp32 done in-register
    fp8 path          : each of those tensors may also materialise a full fp16
                        intermediate (cast-compute-cast), so multiply by 3.
    """
    if op in ("mul", "add"):
        tensors = 3
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


def _slugify(name: str) -> str:
    s = re.sub(r"(?i)nvidia|geforce|pcie|sxm\d?|\bhbm\d*\b|\bon\b", "", name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "gpu"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ops", nargs="+",
                    default=list(bm.OPS),
                    choices=list(bm.OPS))
    ap.add_argument("--dtypes", nargs="+",
                    default=list(bm.DTYPES),
                    choices=list(bm.DTYPES))
    ap.add_argument("--loads", type=int, nargs="+", default=None,
                    help="tensor element counts; default: 256K..256M sweep")
    ap.add_argument("--quick", action="store_true",
                    help="short sweep for smoke-testing")
    ap.add_argument("--window-ms", type=float, default=3000.0,
                    help="target measurement window per cell (ms). Longer = lower NVML noise; "
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
    args = ap.parse_args()

    if args.no_elementwise and args.no_matmul:
        print("[error] --no-elementwise and --no-matmul together select zero cells")
        return 1

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
        pf = preflight.check()
        preflight.print_report(pf)
        if not pf.ok:
            print("preflight failed — fix the above or re-run with --skip-preflight")
            return 1

    # ---- NVML + device -----------------------------------------------------
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    torch.cuda.set_device(args.device)
    gpu_name = torch.cuda.get_device_name(args.device)
    cc = torch.cuda.get_device_capability(args.device)
    gpu_slug = _slugify(gpu_name)
    print(f"\n[info] GPU={gpu_name}  cc={cc[0]}.{cc[1]}  slug={gpu_slug}")
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
    print(f"[baseline] measuring static power for {args.static_seconds:.1f}s …")
    baseline = measure_static_power(handle, seconds=args.static_seconds,
                                    hz=args.poll_hz)
    p_static = baseline["power_w_mean"]
    print(f"[baseline] static power = {p_static:.1f} ± {baseline['power_w_std']:.2f} W  "
          f"(min {baseline['power_w_min']:.1f} W, max {baseline['power_w_max']:.1f} W, "
          f"temp {baseline['temp_c_mean']:.1f}°C, n={baseline['n']})")
    # Sanity check: if stdev > 5% of mean, the "idle" wasn't really idle
    # (background kernel, clock ramp, another process) — warn so the user
    # knows the P_static they're subtracting is noisy.
    if p_static > 0 and baseline["power_w_std"] / p_static > 0.05:
        print(f"[baseline] WARN idle power stdev is {100*baseline['power_w_std']/p_static:.1f}% "
              f"of mean — another process may be using the GPU, or clocks are ramping. "
              f"Check the baseline plot before trusting dyn-power numbers.")

    # ---- sampler (runs for the whole sweep, phase is toggled per cell) ----
    sampler = PowerSampler(handle, hz=args.poll_hz)
    sampler.start()

    # ---- Build a unified list of "plans" — one per (op × dtype × load) ----
    # Each plan has a `build()` callable that allocates tensors just-in-time
    # so we never hold more than one cell's tensors at once.
    plans: list[dict] = []
    l2_bytes = bm.get_l2_bytes(args.device)
    if l2_bytes > 0:
        print(f"[info] L2 cache size: {l2_bytes/(1<<20):.1f} MB "
              f"(used to classify cache_regime for every row)")
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
    safe_loads = _filter_loads(loads, list(args.ops), list(args.dtypes), hbm_bytes)
    if not args.no_elementwise:
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
    if not args.no_matmul:
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
    # DRAM bandwidth probes — opt-in. Pure-streaming kernels at large N
    # so the dyn energy is dominated by HBM traffic. analyze.py converts
    # these into pJ/bit. We pick load sizes deep in the l2_hit_0 regime
    # (working set ≥ 8·L2) so cache reuse is essentially zero.
    if args.dram_bw_test:
        if args.dram_bw_loads is not None:
            stream_loads = args.dram_bw_loads
        elif l2_bytes > 0:
            # Targets ws ∈ {8, 16, 32, 64} × L2 — solidly l2_hit_0.
            # Use rw=2 / 2-byte dtype for sizing (the probe whose ws/N is
            # smallest, so the others are even more deeply DRAM-bound).
            base = max(1 << 14, int(l2_bytes / (2 * 2)))
            stream_loads = [8 * base, 16 * base, 32 * base, 64 * base]
        else:
            stream_loads = [1 << 27, 1 << 28, 1 << 29, 1 << 30]
        # Apply the same memory budget filter used for the elementwise sweep.
        stream_loads = _filter_loads(stream_loads,
                                     ["stream_copy", "stream_scale", "stream_triad"],
                                     list(args.dtypes), hbm_bytes)
        print(f"[dram-bw] {len(stream_loads)} working-set sizes: {stream_loads}")
        for dtype in args.dtypes:
            for op in ("stream_copy", "stream_scale", "stream_triad"):
                for N in stream_loads:
                    plans.append({
                        "category": "elementwise",
                        "mode": "elementwise",
                        "op": op, "dtype": dtype,
                        "load_name": "n_elements", "load_value": N,
                        "build": (lambda op=op, dtype=dtype, N=N:
                                  bm.build(op, dtype, N, device="cuda")),
                    })
    # LLM-shape sweep — opt-in via --llm-shapes. We pre-filter (preset, T)
    # combos by HBM budget using llm_matmul_footprint_bytes(), so the fat
    # lm_head @ T=32768 cell doesn't blow memory on smaller GPUs.
    if args.llm_shapes:
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
    total_cells = len(plans)
    print(f"[info] scheduling {total_cells} cells "
          f"(elementwise={sum(1 for p in plans if p['category']=='elementwise')}, "
          f"matmul={sum(1 for p in plans if p['category']=='matmul')}, "
          f"matmul_llm={sum(1 for p in plans if p['category']=='matmul_llm')})")

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
    try:
        for i, plan in enumerate(plans, 1):
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
                rb = measure_static_power(handle,
                                          seconds=args.rebaseline_seconds,
                                          hz=args.poll_hz)
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
                print(f"  ! build failed: {e}  — skipped")
                continue
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
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  ! run failed: {e}")
                del spec
                torch.cuda.empty_cache()
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

            row = {
                "gpu": gpu_name,
                "compute_cap": f"{cc[0]}.{cc[1]}",
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
                "j_per_element_total": f"{meas['total_energy_j']/total_elements:.3e}",
                "j_per_element_dyn":   f"{dyn_energy/total_elements:.3e}",
                "j_per_flop_total":    f"{meas['total_energy_j']/total_flops:.3e}",
                "j_per_flop_dyn":      f"{dyn_energy/total_flops:.3e}",
                "start_temp_c":      cooldown_info["final_temp_c"],
                "avg_temp_c":        f"{meas['avg_temp_c']:.1f}",
                "peak_temp_c":       meas["peak_temp_c"],
                "temp_rise_c":       (meas["peak_temp_c"] - cooldown_info["final_temp_c"]
                                      if cooldown_info["final_temp_c"] >= 0 else -1),
                "cooldown_elapsed_s": f"{cooldown_info['elapsed_s']:.2f}",
                "cooldown_reached":   int(bool(cooldown_info["reached"])),
                "sm_clk_mhz":   meas["sm_clk_mhz"],
                "mem_clk_mhz":  meas["mem_clk_mhz"],
                "notes":        spec.notes,
            }
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
            torch.cuda.empty_cache()
    finally:
        sampler.stop()
        pynvml.nvmlShutdown()

    # ---- write CSV ---------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"gpu_power_bench_{gpu_slug}_{stamp}{suffix}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[save] {csv_path}  ({len(rows)} rows)")
    else:
        print("\n[warn] no rows collected — nothing saved")
        return 2

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

    print("\nnext: python3 analyze.py {}  # generate linearity plots".format(csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
