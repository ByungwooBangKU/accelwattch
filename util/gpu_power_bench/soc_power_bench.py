#!/usr/bin/env python3
"""SoC power-envelope bench: static / max / leakage.

Three phases:

1. **static**     : `--static-seconds` of pure idle (clocks gate, no kernel
                    running). Reports mean/min/max idle power. Should match
                    the static_power_w baseline that gpu_power_bench.py
                    subtracts in its k_op decomposition — useful as a
                    cross-check.

2. **max**        : `--max-seconds` of a large GEMM driven hard enough to
                    saturate the SMs and approach TGP. Reports peak and
                    mean power, peak and mean temperature. The accompanying
                    plot shows the P(t) ramp + T(t) thermal-soak curve so
                    you can see how long it takes to saturate.

3. **leakage**    : `--leakage-cycles` cycles of {`--leakage-stress-s` of
                    GEMM stress, `--leakage-decay-s` of idle decay}. After
                    each stress phase the silicon is hot, so its leakage
                    current is much higher than at cold-static. We
                    measure mean power in the first second of decay (kernel
                    has stopped, GPU is still hot) — that's `P_hot_idle`.
                    `P_hot_idle - P_static` is the temperature-dependent
                    leakage component. Report the per-cycle value AND the
                    5-cycle mean.

Outputs (under --out-dir, default ./reports):

  soc_power_<gpu>_<stamp>[_<tag>]_summary.csv      one-row stats
  soc_power_<gpu>_<stamp>[_<tag>]_timeseries.csv   raw 100Hz samples
  soc_power_<gpu>_<stamp>[_<tag>]_phases.png       full-run P(t) + T(t)
  soc_power_<gpu>_<stamp>[_<tag>]_leakage.png      5 overlaid decay curves
  soc_power_<gpu>_<stamp>[_<tag>]_summary.png      static/max/leakage bars
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynvml
import torch

import benchmarks as bm
from power_monitor import PowerSampler, wait_for_cooldown


# ---------------------------------------------------------------------------
# Helpers (PCI bus id resolution + slugify mirror gpu_power_bench.py so
# multi-GPU users get the same per-card semantics).
# ---------------------------------------------------------------------------
def _slugify(name: str) -> str:
    s = re.sub(r"(?i)nvidia|geforce|pcie|sxm\d?|\bhbm\d*\b|\bon\b", "", name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "gpu"


def _resolve_nvml_handle(device: int):
    """Pin NVML to the same physical card torch sees as `device`.

    Without this, `nvmlDeviceGetHandleByIndex(device)` reads the wrong
    card whenever CUDA_VISIBLE_DEVICES restricts visibility (NVML ignores
    that env var). PCI bus id is the same on both sides so it's the
    canonical bridge.
    """
    try:
        pci_id = torch.cuda.get_device_properties(device).pci_bus_id
        return pynvml.nvmlDeviceGetHandleByPciBusId(pci_id.encode())
    except Exception:
        return pynvml.nvmlDeviceGetHandleByIndex(device)


# ---------------------------------------------------------------------------
# Phase runners. Each returns (t_start, t_end) in sampler-relative seconds
# so the caller can slice the timeseries for stats.
# ---------------------------------------------------------------------------
def phase_static(sampler: PowerSampler, seconds: float) -> tuple[float, float]:
    """Pure idle. We've already drained CUDA at the call site."""
    sampler.set_phase("static")
    t0 = time.perf_counter() - sampler.t0
    time.sleep(seconds)
    t1 = time.perf_counter() - sampler.t0
    sampler.set_phase("gap")
    return t0, t1


def _run_gemm_for(spec, seconds: float, batch: int = 32) -> None:
    """Loop the GEMM call() until `seconds` has elapsed.

    Calls are batched in groups of `batch` so Python loop overhead doesn't
    starve the GPU between launches — at large K each call already takes
    enough ms that GPU stays saturated, but smaller GEMMs would otherwise
    leave gaps that drop us out of the max-power envelope.
    """
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(batch):
            spec.run()
    torch.cuda.synchronize()


def phase_max(sampler: PowerSampler, spec, seconds: float) -> tuple[float, float]:
    """Drive a large GEMM continuously to push toward TGP."""
    # Warm-up: clocks ramp + cuBLAS / TE picks an algorithm. ~1s should be
    # plenty for the boost-clock to settle.
    sampler.set_phase("max_warmup")
    _run_gemm_for(spec, 1.0)

    sampler.set_phase("max")
    t0 = time.perf_counter() - sampler.t0
    _run_gemm_for(spec, seconds)
    t1 = time.perf_counter() - sampler.t0
    sampler.set_phase("gap")
    return t0, t1


def phase_leakage(sampler: PowerSampler, spec, n_cycles: int,
                  stress_s: float, decay_s: float) -> list[dict]:
    """Repeated stress→stop→decay cycles to expose temperature-dependent leakage.

    Returns a list of per-cycle slice metadata: (stress_t0, stress_t1,
    decay_t0, decay_t1) so the caller can compute hot-leakage power
    by averaging the first 1s of each decay window.
    """
    out = []
    for i in range(n_cycles):
        # ---- stress ----
        sampler.set_phase(f"leak_stress_{i}")
        s0 = time.perf_counter() - sampler.t0
        _run_gemm_for(spec, stress_s)
        s1 = time.perf_counter() - sampler.t0

        # ---- decay (kernel stopped, silicon still hot) ----
        sampler.set_phase(f"leak_decay_{i}")
        d0 = time.perf_counter() - sampler.t0
        time.sleep(decay_s)
        d1 = time.perf_counter() - sampler.t0
        sampler.set_phase("gap")

        out.append({
            "cycle":     i,
            "stress_t0": s0,
            "stress_t1": s1,
            "decay_t0":  d0,
            "decay_t1":  d1,
        })
        print(f"  cycle {i+1}/{n_cycles}  stress={s1-s0:.1f}s  decay={d1-d0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_phase_timeline(samples, summary, out_path: Path) -> None:
    """Whole-run P(t) and T(t) on one figure with phase shading."""
    if not samples:
        return
    ts  = [s.t for s in samples]
    ps  = [s.power_w for s in samples if s.power_w >= 0]
    pts = [s.t for s in samples if s.power_w >= 0]
    Ts  = [s.temp_c for s in samples if s.temp_c >= 0]
    Tts = [s.t for s in samples if s.temp_c >= 0]

    fig, (axP, axT) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    axP.plot(pts, ps, lw=0.8, color="C0")
    axP.set_ylabel("Power (W)")
    axP.set_title(f"SoC power envelope — {summary['gpu_name']}")
    axP.grid(alpha=0.3)
    # Annotate static / max / mean-leakage as horizontal guides.
    axP.axhline(summary["static_power_w_mean"], color="C2", lw=1, ls="--",
                label=f"static {summary['static_power_w_mean']:.1f} W")
    axP.axhline(summary["max_power_w_mean"], color="C3", lw=1, ls="--",
                label=f"max-mean {summary['max_power_w_mean']:.1f} W")
    if summary.get("leakage_power_w_mean", -1) > 0:
        axP.axhline(summary["leakage_power_w_mean"], color="C1", lw=1, ls="--",
                    label=f"hot-leak {summary['leakage_power_w_mean']:.1f} W")
    axP.legend(loc="upper right", fontsize=8)

    axT.plot(Tts, Ts, lw=0.8, color="C3")
    axT.set_ylabel("Temperature (°C)")
    axT.set_xlabel("t (s)")
    axT.grid(alpha=0.3)

    # Phase shading from the per-sample phase tag — collapse runs of same
    # tag to (t_start, t_end) intervals and shade alternating tones.
    intervals = []
    if samples:
        cur_phase = samples[0].phase
        cur_start = samples[0].t
        for s in samples[1:]:
            if s.phase != cur_phase:
                intervals.append((cur_phase, cur_start, s.t))
                cur_phase, cur_start = s.phase, s.t
        intervals.append((cur_phase, cur_start, samples[-1].t))

    color_for = {"static": (0.85, 1.00, 0.85, 0.5),
                 "max":    (1.00, 0.85, 0.85, 0.5)}
    for phase, t0, t1 in intervals:
        c = color_for.get(phase)
        if c is None and phase.startswith("leak_stress"):
            c = (1.00, 0.92, 0.80, 0.4)
        elif c is None and phase.startswith("leak_decay"):
            c = (0.85, 0.90, 1.00, 0.4)
        if c is not None:
            for ax in (axP, axT):
                ax.axvspan(t0, t1, color=c, lw=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_leakage_decay(samples, cycles, summary, out_path: Path) -> None:
    """Overlay the 5 decay curves with t=0 at the moment stress stopped."""
    if not cycles:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    static_w = summary["static_power_w_mean"]
    for c in cycles:
        # Slice the decay window and re-zero its time axis.
        sl = [(s.t - c["decay_t0"], s.power_w, s.temp_c)
              for s in samples
              if c["decay_t0"] <= s.t <= c["decay_t1"] and s.power_w >= 0]
        if not sl:
            continue
        ts = [r[0] for r in sl]
        ps = [r[1] for r in sl]
        ax.plot(ts, ps, lw=1.0, label=f"cycle {c['cycle']+1}", alpha=0.85)
    ax.axhline(static_w, color="k", lw=1, ls=":", label=f"static {static_w:.1f} W")
    ax.set_xlabel("t since stress stop (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Leakage decay (kernel idle, hot silicon) — {summary['gpu_name']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_summary_bars(summary, out_path: Path) -> None:
    """Bar chart: static / max-mean / max-peak / hot-leak (with delta vs static)."""
    labels, values = [], []
    labels.append("static")
    values.append(summary["static_power_w_mean"])
    labels.append("max\n(mean)")
    values.append(summary["max_power_w_mean"])
    labels.append("max\n(peak)")
    values.append(summary["max_power_w_peak"])
    if summary.get("leakage_power_w_mean", -1) > 0:
        labels.append("hot-leak\n(mean of 5)")
        values.append(summary["leakage_power_w_mean"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values,
                  color=["C2", "C3", "C3", "C1"][:len(labels)])
    static_w = summary["static_power_w_mean"]
    for b, v, lab in zip(bars, values, labels):
        delta = v - static_w
        delta_txt = f"\nΔ={delta:+.1f} W" if "static" not in lab else ""
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.1f} W{delta_txt}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Power (W)")
    ax.set_ylim(0, max(values) * 1.20)
    ax.set_title(f"SoC power-envelope summary — {summary['gpu_name']}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="reports")
    ap.add_argument("--sample-hz", type=int, default=100)

    ap.add_argument("--static-seconds", type=float, default=60.0,
                    help="duration of the idle baseline phase")
    ap.add_argument("--max-seconds", type=float, default=60.0,
                    help="duration of the max-power GEMM phase")
    ap.add_argument("--leakage-cycles", type=int, default=5,
                    help="number of stress/decay cycles for the leakage phase")
    ap.add_argument("--leakage-stress-s", type=float, default=20.0,
                    help="GEMM stress duration per leakage cycle")
    ap.add_argument("--leakage-decay-s", type=float, default=30.0,
                    help="post-stress idle (cooldown) per leakage cycle")
    ap.add_argument("--leak-window-s", type=float, default=1.0,
                    help="how long after stress stop to average for hot-leakage "
                         "power (kernel has stopped, silicon hot)")

    ap.add_argument("--matmul-K", type=int, default=16384,
                    help="square GEMM size M=N=K. Larger → more SMs busy → "
                         "closer to TGP. 16384 is roughly the sweet spot for "
                         "Hopper/Blackwell — too small underutilizes, too "
                         "large risks OOM at fp32.")
    ap.add_argument("--dtype", type=str, default="fp16",
                    choices=["fp32", "tf32", "fp16", "bf16", "fp8"])
    ap.add_argument("--mode", type=str, default="tc", choices=["simt", "tc", "te"],
                    help="compute path: simt=CUDA core, tc=Tensor Core, "
                         "te=Transformer Engine fp8 (use with --dtype fp8)")

    ap.add_argument("--cooldown-c", type=int, default=45,
                    help="target temp before each phase. Set 0 to disable.")
    ap.add_argument("--cooldown-timeout", type=float, default=180.0)
    ap.add_argument("--no-leakage", action="store_true")
    ap.add_argument("--no-max", action="store_true")
    args = ap.parse_args()

    pynvml.nvmlInit()
    torch.cuda.set_device(args.device)
    handle = _resolve_nvml_handle(args.device)
    gpu_name = torch.cuda.get_device_name(args.device)
    cc = torch.cuda.get_device_capability(args.device)
    gpu_slug = _slugify(gpu_name)
    print(f"[info] GPU={gpu_name}  cc={cc[0]}.{cc[1]}  slug={gpu_slug}")
    print(f"[info] static={args.static_seconds}s  max={args.max_seconds}s  "
          f"leakage={args.leakage_cycles}x({args.leakage_stress_s}s+"
          f"{args.leakage_decay_s}s)")

    # Build the GEMM spec ONCE — reused for max and leakage stress phases.
    spec = None
    if not args.no_max or not args.no_leakage:
        try:
            spec = bm.build_matmul(args.matmul_K, args.dtype, args.mode,
                                   device="cuda")
            print(f"[info] GEMM: K={args.matmul_K}  {args.dtype}/{args.mode}  "
                  f"compute_unit={spec.compute_unit}")
            if spec.emulated:
                print(f"[warn] GEMM is EMULATED on this GPU: {spec.notes}")
        except Exception as e:
            print(f"[error] failed to build GEMM ({args.dtype}/{args.mode} "
                  f"K={args.matmul_K}): {e}")
            print(f"[error] try a smaller K or a different dtype/mode")
            return 2

    # Single sampler for the whole run so the timeseries is contiguous and
    # we can slice phases out of it for stats / plotting.
    sampler = PowerSampler(handle, hz=args.sample_hz)
    sampler.start()
    sampler.set_phase("startup")

    summary = {"gpu_name": gpu_name, "gpu_slug": gpu_slug,
               "cc_major": cc[0], "cc_minor": cc[1],
               "matmul_K": args.matmul_K,
               "dtype": args.dtype, "mode": args.mode}
    cycles_meta: list[dict] = []
    static_t = max_t = None

    try:
        # --- 1) STATIC ------------------------------------------------------
        if args.cooldown_c > 0:
            sampler.set_phase("cooldown_pre_static")
            wait_for_cooldown(handle, target_c=args.cooldown_c,
                              timeout_s=args.cooldown_timeout, verbose=False)
        print(f"\n[phase] static idle for {args.static_seconds}s")
        static_t = phase_static(sampler, args.static_seconds)

        # --- 2) MAX ---------------------------------------------------------
        if not args.no_max:
            if args.cooldown_c > 0:
                sampler.set_phase("cooldown_pre_max")
                wait_for_cooldown(handle, target_c=args.cooldown_c,
                                  timeout_s=args.cooldown_timeout, verbose=False)
            print(f"\n[phase] max-power GEMM for {args.max_seconds}s "
                  f"(K={args.matmul_K} {args.dtype}/{args.mode})")
            max_t = phase_max(sampler, spec, args.max_seconds)

        # --- 3) LEAKAGE -----------------------------------------------------
        if not args.no_leakage:
            if args.cooldown_c > 0:
                sampler.set_phase("cooldown_pre_leakage")
                wait_for_cooldown(handle, target_c=args.cooldown_c,
                                  timeout_s=args.cooldown_timeout, verbose=False)
            print(f"\n[phase] leakage: {args.leakage_cycles} cycles of "
                  f"{args.leakage_stress_s}s stress + {args.leakage_decay_s}s decay")
            cycles_meta = phase_leakage(sampler, spec, args.leakage_cycles,
                                        args.leakage_stress_s,
                                        args.leakage_decay_s)
    finally:
        sampler.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # ---- compute summary stats from the in-memory timeseries --------------
    samples = sampler.samples

    def _slice_stats(t0: float, t1: float) -> dict:
        ps = [s.power_w for s in samples
              if t0 <= s.t <= t1 and s.power_w >= 0]
        ts = [s.temp_c for s in samples
              if t0 <= s.t <= t1 and s.temp_c >= 0]
        if not ps:
            return {"power_mean": -1, "power_peak": -1, "power_min": -1,
                    "temp_mean": -1, "temp_peak": -1, "n": 0}
        return {
            "power_mean": sum(ps) / len(ps),
            "power_peak": max(ps),
            "power_min":  min(ps),
            "temp_mean":  (sum(ts) / len(ts)) if ts else -1,
            "temp_peak":  max(ts) if ts else -1,
            "n":          len(ps),
        }

    if static_t:
        st = _slice_stats(*static_t)
        summary["static_seconds"]      = args.static_seconds
        summary["static_power_w_mean"] = round(st["power_mean"], 3)
        summary["static_power_w_peak"] = round(st["power_peak"], 3)
        summary["static_temp_c_mean"]  = round(st["temp_mean"], 1)
        summary["static_temp_c_peak"]  = st["temp_peak"]

    if max_t:
        mx = _slice_stats(*max_t)
        summary["max_seconds"]         = args.max_seconds
        summary["max_power_w_mean"]    = round(mx["power_mean"], 3)
        summary["max_power_w_peak"]    = round(mx["power_peak"], 3)
        summary["max_temp_c_mean"]     = round(mx["temp_mean"], 1)
        summary["max_temp_c_peak"]     = mx["temp_peak"]

    # Per-cycle hot-leakage = first --leak-window-s of each decay window.
    leakage_window = args.leak_window_s
    leak_powers, leak_temps = [], []
    for c in cycles_meta:
        d0 = c["decay_t0"]
        st = _slice_stats(d0, d0 + leakage_window)
        c["hot_power_w_mean"] = round(st["power_mean"], 3)
        c["hot_temp_c_mean"]  = round(st["temp_mean"], 1)
        c["hot_temp_c_peak"]  = st["temp_peak"]
        # Pre-decay (i.e. end of stress) temp = peak temp during stress.
        st_stress = _slice_stats(c["stress_t0"], c["stress_t1"])
        c["stress_temp_c_peak"] = st_stress["temp_peak"]
        c["stress_power_w_mean"] = round(st_stress["power_mean"], 3)
        if st["power_mean"] > 0:
            leak_powers.append(st["power_mean"])
            leak_temps.append(st["temp_mean"])
    if leak_powers:
        summary["leakage_cycles"]         = len(leak_powers)
        summary["leakage_window_s"]       = leakage_window
        summary["leakage_power_w_mean"]   = round(sum(leak_powers) / len(leak_powers), 3)
        summary["leakage_power_w_min"]    = round(min(leak_powers), 3)
        summary["leakage_power_w_max"]    = round(max(leak_powers), 3)
        summary["leakage_temp_c_mean"]    = round(sum(leak_temps) / len(leak_temps), 1)
        if "static_power_w_mean" in summary:
            summary["leakage_minus_static_w"] = round(
                summary["leakage_power_w_mean"] - summary["static_power_w_mean"], 3)

    # ---- write outputs -----------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    base = out_dir / f"soc_power_{gpu_slug}_{stamp}{suffix}"

    # Summary CSV — one row, plus per-cycle rows for leakage detail.
    summary_path = base.parent / f"{base.name}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
        w.writerow([])
        if cycles_meta:
            w.writerow(["leakage_cycle", "stress_temp_c_peak",
                        "stress_power_w_mean", "hot_temp_c_peak",
                        "hot_power_w_mean", "hot_minus_static_w"])
            stat_w = summary.get("static_power_w_mean", 0)
            for c in cycles_meta:
                w.writerow([c["cycle"] + 1,
                            c.get("stress_temp_c_peak", -1),
                            c.get("stress_power_w_mean", -1),
                            c.get("hot_temp_c_peak", -1),
                            c.get("hot_power_w_mean", -1),
                            round(c.get("hot_power_w_mean", 0) - stat_w, 3)])

    # Time-series CSV — every sample with phase tag (rounded to keep size sane).
    ts_path = base.parent / f"{base.name}_timeseries.csv"
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "power_w", "temp_c", "sm_mhz", "mem_mhz",
                    "gpu_util", "phase"])
        for s in samples:
            w.writerow([f"{s.t:.4f}", f"{s.power_w:.4f}",
                        s.temp_c, s.sm_mhz, s.mem_mhz, s.gpu_util, s.phase])

    # Plots
    phase_png   = base.parent / f"{base.name}_phases.png"
    leakage_png = base.parent / f"{base.name}_leakage.png"
    summary_png = base.parent / f"{base.name}_summary.png"
    plot_phase_timeline(samples, summary, phase_png)
    if cycles_meta:
        plot_leakage_decay(samples, cycles_meta, summary, leakage_png)
    plot_summary_bars(summary, summary_png)

    # ---- console report ---------------------------------------------------
    print("\n========== SoC power envelope ==========")
    print(f"  GPU            : {gpu_name}")
    if "static_power_w_mean" in summary:
        print(f"  static (idle)  : {summary['static_power_w_mean']:7.2f} W   "
              f"@ {summary['static_temp_c_mean']:.1f}°C "
              f"(peak {summary['static_temp_c_peak']}°C)")
    if "max_power_w_mean" in summary:
        print(f"  max (mean)     : {summary['max_power_w_mean']:7.2f} W   "
              f"@ avg {summary['max_temp_c_mean']:.1f}°C "
              f"(peak {summary['max_temp_c_peak']}°C)")
        print(f"  max (peak)     : {summary['max_power_w_peak']:7.2f} W")
    if "leakage_power_w_mean" in summary:
        print(f"  hot leakage    : {summary['leakage_power_w_mean']:7.2f} W   "
              f"@ {summary['leakage_temp_c_mean']:.1f}°C  "
              f"(mean of {summary['leakage_cycles']} cycles, "
              f"first {leakage_window}s after stress stop)")
        print(f"  Δ vs static    : {summary.get('leakage_minus_static_w', 0):+7.2f} W   "
              f"(temperature-dependent leakage component)")
    print()
    print(f"[save] {summary_path}")
    print(f"[save] {ts_path}")
    print(f"[save] {phase_png}")
    if cycles_meta:
        print(f"[save] {leakage_png}")
    print(f"[save] {summary_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
