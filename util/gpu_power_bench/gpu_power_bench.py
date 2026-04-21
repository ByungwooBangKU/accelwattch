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

DEFAULT_LOADS = [
    1 << 18,   # 256K
    1 << 20,   # 1M
    1 << 22,   # 4M
    1 << 24,   # 16M
    1 << 26,   # 64M
    1 << 28,   # 256M
]
QUICK_LOADS = [1 << 20, 1 << 22, 1 << 24]

# Matrix side length K (M = N = K). FLOPs per call = 2·K³, memory = 3·K² elements.
DEFAULT_MATMUL_SIZES = [512, 1024, 2048, 4096, 8192]
QUICK_MATMUL_SIZES = [1024, 2048, 4096]


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
    ap.add_argument("--window-ms", type=float, default=1500.0,
                    help="target measurement window per cell (ms). Longer = lower NVML noise.")
    ap.add_argument("--static-seconds", type=float, default=8.0,
                    help="idle time to measure static/baseline power")
    ap.add_argument("--cooldown-c", type=int, default=50,
                    help="°C threshold to reach between experiments (set -1 to disable)")
    ap.add_argument("--cooldown-timeout", type=float, default=120.0)
    ap.add_argument("--no-cooldown", action="store_true",
                    help="skip thermal cool-down between cells")
    ap.add_argument("--out-dir", type=str, default="reports")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output filenames (separate runs / configs)")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--poll-hz", type=int, default=100)
    # --- matmul (Tensor Core vs CUDA-core + TE FP8) ---
    ap.add_argument("--no-matmul", action="store_true",
                    help="skip the matmul (Tensor Core / SIMT) sweep")
    ap.add_argument("--matmul-sizes", type=int, nargs="+", default=None,
                    help="square matrix side lengths (M=N=K); default: 512..8192")
    ap.add_argument("--matmul-variants", nargs="+", default=None,
                    help='matmul variants "dtype:mode" (default: all 5). '
                         'choices: fp32:simt tf32:tc fp16:tc bf16:tc fp8:te')
    args = ap.parse_args()

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
                          timeout_s=args.cooldown_timeout)
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
    for dtype in args.dtypes:
        for op in args.ops:
            for N in loads:
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
    total_cells = len(plans)
    print(f"[info] scheduling {total_cells} cells "
          f"(elementwise={sum(1 for p in plans if p['category']=='elementwise')}, "
          f"matmul={sum(1 for p in plans if p['category']=='matmul')})")

    rows: list[dict] = []
    try:
        for i, plan in enumerate(plans, 1):
            label = f"{plan['op']}_{plan['dtype']}_{plan['mode']}"
            print(f"\n[{i}/{total_cells}] {label}  {plan['load_name']}={plan['load_value']}")

            try:
                spec = plan["build"]()
            except Exception as e:
                print(f"  ! build failed: {e}  — skipped")
                continue
            if spec.notes:
                print(f"  note: {spec.notes}")

            if not args.no_cooldown and args.cooldown_c > 0:
                wait_for_cooldown(handle, target_c=args.cooldown_c,
                                  timeout_s=args.cooldown_timeout,
                                  verbose=False)

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
            # (measured idle at program start) from the measured average
            # power to get the dynamic (workload-attributable) component.
            dyn_power = max(0.0, meas["avg_power_w"] - p_static)
            static_energy = p_static * meas["wall_s"]
            dyn_energy = max(0.0, meas["total_energy_j"] - static_energy)

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
                "avg_power_w":    f"{meas['avg_power_w']:.3f}",
                "dyn_power_w":    f"{dyn_power:.3f}",
                "total_energy_j":  f"{meas['total_energy_j']:.4f}",
                "static_energy_j": f"{static_energy:.4f}",
                "dyn_energy_j":    f"{dyn_energy:.4f}",
                "j_per_element_total": f"{meas['total_energy_j']/total_elements:.3e}",
                "j_per_element_dyn":   f"{dyn_energy/total_elements:.3e}",
                "j_per_flop_total":    f"{meas['total_energy_j']/total_flops:.3e}",
                "j_per_flop_dyn":      f"{dyn_energy/total_flops:.3e}",
                "avg_temp_c":   f"{meas['avg_temp_c']:.1f}",
                "peak_temp_c":  meas["peak_temp_c"],
                "sm_clk_mhz":   meas["sm_clk_mhz"],
                "mem_clk_mhz":  meas["mem_clk_mhz"],
                "notes":        spec.notes,
            }
            rows.append(row)
            print(f"  E_total={meas['total_energy_j']:.3f} J, "
                  f"E_dyn={dyn_energy:.3f} J, "
                  f"P_avg={meas['avg_power_w']:.1f} W "
                  f"(dyn {dyn_power:.1f} W), "
                  f"T={meas['avg_temp_c']:.1f}°C (peak {meas['peak_temp_c']}), "
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
