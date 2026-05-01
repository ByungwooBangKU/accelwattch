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

  soc_power_<gpu>_<stamp>[_<tag>]_summary.csv           one-row stats
  soc_power_<gpu>_<stamp>[_<tag>]_timeseries.csv        raw 100Hz samples
  soc_power_<gpu>_<stamp>[_<tag>]_phases.png            full-run P(t) + T(t)
  soc_power_<gpu>_<stamp>[_<tag>]_leakage.png           5 overlaid decay curves
  soc_power_<gpu>_<stamp>[_<tag>]_leakage_enlarged.png  zoom: first 3s, 0..150W
  soc_power_<gpu>_<stamp>[_<tag>]_summary.png           static/max/leakage bars
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

    A fatal CUDA fault here (Blackwell amax race for fp8_te, OOM, etc.)
    poisons the whole process — same hazard as gpu_power_bench.py covers
    in README §8.3.3. We re-raise as RuntimeError so the caller can save
    whatever telemetry has already been collected before bailing.
    """
    end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < end:
            for _ in range(batch):
                spec.run()
        torch.cuda.synchronize()
    except Exception as e:
        msg = str(e)
        fatal_markers = ("illegal memory access", "CUDA error",
                         "device-side assert", "CUDA call failed",
                         "out of memory")
        if any(m in msg for m in fatal_markers):
            raise RuntimeError(
                f"CUDA fault during stress GEMM after {time.perf_counter()-end+seconds:.1f}s "
                f"of {seconds:.0f}s: {msg}. Process CUDA context is likely "
                f"poisoned; partial telemetry will still be saved. See "
                f"README §8.3.3 for Blackwell+fp8_te workarounds."
            ) from e
        raise


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
def _add_axis_detail(ax, x_major: float, x_minor: float,
                     y_major: float = None, y_minor: float = None) -> None:
    """Tighten axis ticks + grid for at-a-glance reading.

    Major ticks every `x_major` seconds with labels; minor ticks every
    `x_minor` for sub-second resolution. Grid drawn on both. Y is set
    similarly when y_major/y_minor are given (we leave it auto-scaled
    when None — the data range varies too much across phases to pick
    a sane fixed step).
    """
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(which="major", alpha=0.4)
    ax.grid(which="minor", alpha=0.15, linestyle=":")
    ax.tick_params(which="both", labelsize=9)


def plot_phase_timeline(samples, summary, out_path: Path) -> None:
    """Whole-run P(t) and T(t) on one figure with phase shading.

    Big figure (16x10 in) so x/y ticks remain readable when the run
    spans 200+ seconds. Minor x ticks every 5s, major every 30s; minor
    power-y ticks every 10W, major every 50W.
    """
    if not samples:
        return
    ts  = [s.t for s in samples]
    ps  = [s.power_w for s in samples if s.power_w >= 0]
    pts = [s.t for s in samples if s.power_w >= 0]
    Ts  = [s.temp_c for s in samples if s.temp_c >= 0]
    Tts = [s.t for s in samples if s.temp_c >= 0]

    fig, (axP, axT) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    axP.plot(pts, ps, lw=1.0, color="C0")
    axP.set_ylabel("Power (W)", fontsize=11)
    src = summary.get("power_source", "")
    title = f"SoC power envelope — {summary['gpu_name']}"
    if src:
        title += f"   ({src})"
    axP.set_title(title, fontsize=12)
    # Horizontal guides — only drawn for phases that actually completed.
    # An aborted run may leave any of these missing from `summary`.
    s_w = summary.get("static_power_w_mean")
    if s_w is not None and s_w > 0:
        axP.axhline(s_w, color="C2", lw=1.2, ls="--",
                    label=f"static {s_w:.1f} W")
    m_w = summary.get("max_power_w_mean")
    if m_w is not None and m_w > 0:
        axP.axhline(m_w, color="C3", lw=1.2, ls="--",
                    label=f"max-mean {m_w:.1f} W")
    l_w = summary.get("leakage_power_w_mean")
    if l_w is not None and l_w > 0:
        axP.axhline(l_w, color="C1", lw=1.2, ls="--",
                    label=f"hot-leak {l_w:.1f} W")
    axP.legend(loc="upper right", fontsize=10)

    axT.plot(Tts, Ts, lw=1.0, color="C3")
    axT.set_ylabel("Temperature (°C)", fontsize=11)
    axT.set_xlabel("t (s)", fontsize=11)

    # Tick density tuned to the new ~200s default total wall time.
    # 30s major / 5s minor on x lets you spot a 1-2s feature without
    # squinting; 50W major / 10W minor on power matches the typical
    # per-bin resolution we care about.
    duration = (samples[-1].t - samples[0].t) if samples else 1.0
    x_major = max(10.0, round(duration / 12 / 5) * 5) or 10.0
    x_minor = max(1.0, x_major / 6)
    _add_axis_detail(axP, x_major=x_major, x_minor=x_minor,
                     y_major=50.0, y_minor=10.0)
    _add_axis_detail(axT, x_major=x_major, x_minor=x_minor,
                     y_major=10.0, y_minor=2.0)

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
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _draw_leakage_decay(ax, samples, cycles, summary,
                        x_lim: tuple[float, float] | None,
                        y_lim: tuple[float, float] | None,
                        x_major: float, x_minor: float,
                        y_major: float, y_minor: float) -> None:
    """Shared body for the full and zoomed leakage-decay plots."""
    static_w = summary.get("static_power_w_mean", 0.0) or 0.0
    leak_window = summary.get("leakage_window_s", 1.0)
    for c in cycles:
        # Slice the decay window and re-zero its time axis. When zoomed,
        # we still pull the full slice — matplotlib's xlim handles the
        # crop, and we want the legend / static reference to look the
        # same on both plots.
        sl = [(s.t - c["decay_t0"], s.power_w, s.temp_c)
              for s in samples
              if c["decay_t0"] <= s.t <= c["decay_t1"] and s.power_w >= 0]
        if not sl:
            continue
        ts = [r[0] for r in sl]
        ps = [r[1] for r in sl]
        ax.plot(ts, ps, lw=1.3, label=f"cycle {c['cycle']+1}", alpha=0.85)
    ax.axhline(static_w, color="k", lw=1.2, ls=":",
               label=f"static {static_w:.1f} W")
    ax.axvspan(0, leak_window, color=(0.95, 0.95, 0.30, 0.20),
               label=f"hot-window ({leak_window}s)")
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("t since stress stop (s)", fontsize=11)
    ax.set_ylabel("Power (W)", fontsize=11)
    _add_axis_detail(ax, x_major=x_major, x_minor=x_minor,
                     y_major=y_major, y_minor=y_minor)
    ax.legend(loc="upper right", fontsize=10)


def plot_leakage_decay(samples, cycles, summary, out_path: Path) -> None:
    """Overlay the decay curves with t=0 at the moment stress stopped.

    Full window (the entire `--leakage-decay-s` per cycle). Minor x ticks
    every 1s, y ticks every 5W, with the hot-window shaded so the reader
    can see which samples feed `leakage_power_w_mean`.
    """
    if not cycles:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    src = summary.get("power_source", "")
    title = f"Leakage decay (kernel idle, hot silicon) — {summary['gpu_name']}"
    if src:
        title += f"   ({src})"
    ax.set_title(title, fontsize=12)
    # Decay windows are ~15s in the new defaults — minor every 1s, major
    # every 5s gives readable detail without crowding.
    _draw_leakage_decay(ax, samples, cycles, summary,
                        x_lim=None, y_lim=None,
                        x_major=5.0, x_minor=1.0,
                        y_major=20.0, y_minor=5.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_leakage_decay_zoomed(samples, cycles, summary, out_path: Path,
                              x_max: float = 3.0,
                              y_min: float = 50.0,
                              y_max: float = 150.0) -> None:
    """Zoomed view of the first 3s and (50..150 W) — the region where the
    hot-leakage signal lives.

    Same data as `plot_leakage_decay` but with tighter axis limits and
    finer ticks (0.25s minor / 1s major on x, 5W minor / 25W major on y)
    so the cycle-to-cycle hot-window spread and the rapid drop in the
    first second are easy to read off the plot.

    Includes a twin y-axis for temperature : per-cycle die temp during
    the same decay window, plotted as dashed lines so the reader can
    correlate "how hot was the chip when this leakage power was
    measured" without flipping between plots. Higher temp → higher
    leakage current is the canonical Δ leakage interpretation, so
    temp on the same panel makes that link visible.
    """
    if not cycles:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    src = summary.get("power_source", "")
    title = (f"Leakage decay — first {x_max:.0f}s zoom "
             f"({y_min:.0f}–{y_max:.0f} W) — {summary['gpu_name']}")
    if src:
        title += f"   ({src})"
    ax.set_title(title, fontsize=12)
    _draw_leakage_decay(ax, samples, cycles, summary,
                        x_lim=(0.0, x_max), y_lim=(y_min, y_max),
                        x_major=1.0, x_minor=0.25,
                        y_major=25.0, y_minor=5.0)

    # ---- Temperature overlay on a twin y-axis ---------------------------
    # Per-cycle temp during the same decay window. Same color cycle as
    # the power lines so cycle 1 power and cycle 1 temp share a hue —
    # only the linestyle (solid power, dashed temp) differs. Single
    # legend entry per "Temp (°C)" series to avoid bloating the
    # already-busy legend.
    ax_t = ax.twinx()
    cmap = plt.get_cmap("tab10")
    temp_min, temp_max = float("inf"), float("-inf")
    for i, c in enumerate(cycles):
        sl = [(s.t - c["decay_t0"], s.temp_c) for s in samples
              if c["decay_t0"] <= s.t <= c["decay_t1"] and s.temp_c >= 0]
        if not sl:
            continue
        ts = [r[0] for r in sl]
        Ts = [r[1] for r in sl]
        # Match cycle color (matplotlib default cycle Cn → tab10[n])
        color = cmap(i % 10)
        ax_t.plot(ts, Ts, lw=1.0, ls="--", alpha=0.7, color=color,
                  label=f"cycle {c['cycle']+1} temp" if i == 0 else None)
        temp_min = min(temp_min, min(Ts))
        temp_max = max(temp_max, max(Ts))
    if temp_min < temp_max:
        # Round to 5°C grid + 2°C headroom each side
        ax_t.set_ylim(int(temp_min) - 2, int(temp_max) + 2)
    ax_t.set_ylabel("Temperature (°C, dashed)", fontsize=11, color="#555")
    ax_t.tick_params(axis="y", labelsize=9, colors="#555")
    # Single annotation in the upper-right showing the hot-end span :
    if temp_min < temp_max:
        ax_t.text(0.99, 0.98,
                  f"temp range across cycles : {int(temp_min)}…{int(temp_max)} °C",
                  transform=ax_t.transAxes, ha="right", va="top",
                  fontsize=9, color="#555",
                  bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.85, pad=2))

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary_bars(summary, out_path: Path) -> None:
    """Bar chart: static / max-mean / max-peak / hot-leak (with delta vs static).

    Bigger figure (10x6) and y-axis tick density tuned so the W values
    on each bar are easily read.
    """
    labels, values = [], []
    s_w = summary.get("static_power_w_mean")
    if s_w is not None and s_w > 0:
        labels.append("static")
        values.append(s_w)
    m_w = summary.get("max_power_w_mean")
    if m_w is not None and m_w > 0:
        labels.append("max\n(mean)")
        values.append(m_w)
    m_p = summary.get("max_power_w_peak")
    if m_p is not None and m_p > 0:
        labels.append("max\n(peak)")
        values.append(m_p)
    l_w = summary.get("leakage_power_w_mean")
    if l_w is not None and l_w > 0:
        n = summary.get("leakage_cycles", "?")
        labels.append(f"hot-leak\n(mean of {n})")
        values.append(l_w)

    if not values:
        return  # nothing to plot

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values,
                  color=["C2", "C3", "C3", "C1"][:len(labels)])
    static_w = summary.get("static_power_w_mean", 0.0) or 0.0
    for b, v, lab in zip(bars, values, labels):
        delta = v - static_w
        delta_txt = f"\nΔ={delta:+.1f} W" if "static" not in lab else ""
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.1f} W{delta_txt}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.22)
    src = summary.get("power_source", "")
    title = f"SoC power-envelope summary — {summary['gpu_name']}"
    if src:
        title += f"   ({src})"
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3, which="major")
    ax.grid(axis="y", alpha=0.15, which="minor", linestyle=":")
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Leakage(T) curve fit  (P2.2 / G7)  — extract a power(temperature) model
# from the decay-window (T, P) samples we already collect for the
# leakage plot. Lets the user feed an Arrhenius-like leakage term into
# AccelWattch instead of a single hot-vs-cold delta.
# ---------------------------------------------------------------------------
def fit_leakage_temperature(samples, cycles) -> dict:
    """Fit an exponential power(temperature) model to the decay
    samples — (T, P) pairs collected during each leakage cycle's decay
    window (from t=0 stress-stop to t = decay_s).

    Model :
        P(T) = a + b · exp(c · T)
        where T is die temperature in °C, P in W.

    The exponential captures the Arrhenius-like temperature dependence
    of silicon leakage current (≈ doubles every 10 °C). On idle silicon
    cooling from hot, this is exactly what we measure.

    Falls back to linear fit `P(T) = a + b·T` if exponential fit fails
    (e.g. noisy data, narrow temp range). Both fits are reported in the
    return dict so the user can pick.

    Returns dict with :
        n_points              : number of (T, P) samples used
        temp_range_c          : (min, max) of the fitted T range
        power_range_w         : (min, max) of the fitted P range
        exp_a, exp_b, exp_c   : exponential fit parameters (NaN on fail)
        exp_r2                : R² of the exponential fit
        lin_a, lin_b          : linear fit P = a + b·T
        lin_r2                : R² of the linear fit
        recommended           : "exponential" or "linear" — better R²
    """
    # Aggregate (T, P) pairs from every decay window of every cycle.
    pairs = []
    for c in cycles:
        d0 = c.get("decay_t0")
        d1 = c.get("decay_t1")
        if d0 is None or d1 is None:
            continue
        for s in samples:
            if d0 <= s.t <= d1 and s.power_w >= 0 and s.temp_c >= 0:
                pairs.append((float(s.temp_c), float(s.power_w)))
    out = {
        "n_points": len(pairs),
        "temp_range_c": (float("nan"), float("nan")),
        "power_range_w": (float("nan"), float("nan")),
        "exp_a": float("nan"), "exp_b": float("nan"),
        "exp_c": float("nan"), "exp_r2": float("nan"),
        "lin_a": float("nan"), "lin_b": float("nan"),
        "lin_r2": float("nan"),
        "recommended": "none",
    }
    if len(pairs) < 5:
        return out
    import numpy as _np
    T = _np.array([p[0] for p in pairs], dtype=float)
    P = _np.array([p[1] for p in pairs], dtype=float)
    out["temp_range_c"]  = (float(T.min()), float(T.max()))
    out["power_range_w"] = (float(P.min()), float(P.max()))

    # ---- Linear fit (always works as a sanity check) ----
    try:
        lin_b, lin_a = _np.polyfit(T, P, 1)   # P = lin_a + lin_b · T
        P_lin = lin_a + lin_b * T
        ss_res = _np.sum((P - P_lin) ** 2)
        ss_tot = _np.sum((P - P.mean()) ** 2)
        lin_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out["lin_a"], out["lin_b"], out["lin_r2"] = float(lin_a), float(lin_b), float(lin_r2)
    except Exception:
        pass

    # ---- Exponential fit P = a + b · exp(c · T) ----
    # Use scipy.optimize.curve_fit; if scipy unavailable, fall back to
    # log-linear approximation : log(P − P_min_safe) ≈ log(b) + c·T,
    # with P_min_safe being a small offset to keep log defined.
    exp_ok = False
    try:
        from scipy.optimize import curve_fit
        def _expmodel(T, a, b, c):
            return a + b * _np.exp(c * T)
        # initial guesses : a ≈ P_min, b small positive, c ≈ 0.05
        try:
            (a, b, c), _ = curve_fit(_expmodel, T, P,
                                     p0=[float(P.min()), 1.0, 0.05],
                                     maxfev=8000)
            P_exp = _expmodel(T, a, b, c)
            ss_res = _np.sum((P - P_exp) ** 2)
            ss_tot = _np.sum((P - P.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            out["exp_a"], out["exp_b"], out["exp_c"] = float(a), float(b), float(c)
            out["exp_r2"] = float(r2)
            exp_ok = _np.isfinite(r2)
        except Exception:
            pass
    except ImportError:
        # Log-linear fallback : assume offset = floor(P) − ε.
        try:
            P_offset = float(P.min()) - 0.1
            log_dy = _np.log(_np.maximum(P - P_offset, 1e-6))
            c_est, log_b = _np.polyfit(T, log_dy, 1)
            a = P_offset
            b = float(_np.exp(log_b))
            c = float(c_est)
            P_exp = a + b * _np.exp(c * T)
            ss_res = _np.sum((P - P_exp) ** 2)
            ss_tot = _np.sum((P - P.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            out["exp_a"], out["exp_b"], out["exp_c"] = a, b, c
            out["exp_r2"] = float(r2)
            exp_ok = _np.isfinite(r2)
        except Exception:
            pass

    # Pick the better fit
    if exp_ok and out["exp_r2"] >= out.get("lin_r2", -1):
        out["recommended"] = "exponential"
    elif _np.isfinite(out.get("lin_r2", float("nan"))):
        out["recommended"] = "linear"
    return out


def plot_leakage_temperature(samples, cycles, fit, out_path, gpu: str) -> bool:
    """Scatter (T, P) from leakage decay windows + exponential / linear
    fit overlays. Companion to fit_leakage_temperature().
    """
    if not cycles:
        return False
    pairs = []
    cycle_idx = []
    for i, c in enumerate(cycles):
        d0 = c.get("decay_t0"); d1 = c.get("decay_t1")
        if d0 is None or d1 is None:
            continue
        for s in samples:
            if d0 <= s.t <= d1 and s.power_w >= 0 and s.temp_c >= 0:
                pairs.append((s.temp_c, s.power_w))
                cycle_idx.append(i)
    if len(pairs) < 5:
        return False
    import numpy as _np
    T = _np.array([p[0] for p in pairs], dtype=float)
    P = _np.array([p[1] for p in pairs], dtype=float)
    cycle_idx = _np.array(cycle_idx)

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    for i in sorted(set(cycle_idx)):
        m = cycle_idx == i
        ax.scatter(T[m], P[m], s=24, color=cmap(i % 10), alpha=0.7,
                   label=f"cycle {i+1}", edgecolors="white", linewidths=0.5)

    # Overlay fits
    T_fit = _np.linspace(T.min(), T.max(), 100)
    if _np.isfinite(fit.get("exp_r2", float("nan"))):
        a, b, c = fit["exp_a"], fit["exp_b"], fit["exp_c"]
        P_exp = a + b * _np.exp(c * T_fit)
        ax.plot(T_fit, P_exp, "-", color="#d62728", lw=2.5,
                label=(f"exp fit: P = {a:.1f} + {b:.3f}·exp({c:.4f}·T)"
                       f"   R²={fit['exp_r2']:.3f}"))
    if _np.isfinite(fit.get("lin_r2", float("nan"))):
        a, b = fit["lin_a"], fit["lin_b"]
        P_lin = a + b * T_fit
        ax.plot(T_fit, P_lin, "--", color="#7f7f7f", lw=1.5, alpha=0.7,
                label=(f"linear fit: P = {a:.1f} + {b:.3f}·T   "
                       f"R²={fit['lin_r2']:.3f}"))

    ax.set_xlabel("Die temperature (°C)", fontsize=11)
    ax.set_ylabel("Power (W)", fontsize=11)
    src = " — recommended: " + fit.get("recommended", "?")
    ax.set_title(
        f"Leakage power vs die temperature — {gpu}\n"
        "P(T) extracted from leakage-cycle decay windows. Exponential "
        "form is the canonical Arrhenius-like leakage model; "
        "linear form is the sanity-check baseline." + src,
        fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


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
    ap.add_argument("--power-source", choices=["legacy", "instant", "average"],
                    default="legacy",
                    help="NVML power source. 'legacy' (default) matches "
                         "nvidia-smi. 'instant' captures sub-ms transients "
                         "(useful for the leakage decay) but reads higher "
                         "than nvidia-smi during static phase. 'average' "
                         "is a smoother running mean.")

    # Defaults shrunk vs. the original (60/60/5×(20+30) ≈ 10 min):
    # - static 20s gives ~400 samples at 20Hz NVML — plenty for a tight
    #   idle baseline (std typically <0.5W).
    # - max 30s lets clocks ramp, cuBLAS pick an algorithm, and the
    #   thermal-soak curve reach its steady-state plateau (~5-10s ramp,
    #   then ~20s of saturated reading).
    # - leakage stress 10s heats the silicon enough that the hot-leak
    #   delta is well above NVML noise. decay 15s captures the visible
    #   exponential ramp back toward static; longer just adds wall time.
    # Total compute ~3 min, total wall (with cooldowns) ~5 min.
    ap.add_argument("--static-seconds", type=float, default=20.0,
                    help="duration of the idle baseline phase")
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="duration of the max-power GEMM phase")
    ap.add_argument("--leakage-cycles", type=int, default=5,
                    help="number of stress/decay cycles for the leakage phase")
    ap.add_argument("--leakage-stress-s", type=float, default=10.0,
                    help="GEMM stress duration per leakage cycle")
    ap.add_argument("--leakage-decay-s", type=float, default=15.0,
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

    # GEMM spec is built AFTER the static phase to avoid the 5x warmup
    # inside _make_matmul_*() pinning the GPU into P0 (boost clocks)
    # before we measure idle. NVIDIA's P-state hysteresis keeps the
    # chip in P0 for tens of seconds after recent activity, which
    # would inflate static_power_w by 30..50 W on H100. Static
    # measurement runs first on a truly cold idle, then we build.
    spec = None

    # Single sampler for the whole run so the timeseries is contiguous and
    # we can slice phases out of it for stats / plotting.
    sampler = PowerSampler(handle, hz=args.sample_hz,
                           power_source=args.power_source)
    print(f"[info] power source: {sampler.power_source}")
    sampler.start()
    sampler.set_phase("startup")

    summary = {"gpu_name": gpu_name, "gpu_slug": gpu_slug,
               "cc_major": cc[0], "cc_minor": cc[1],
               "matmul_K": args.matmul_K,
               "dtype": args.dtype, "mode": args.mode,
               "power_source": sampler.power_source,
               "sample_hz": args.sample_hz}
    cycles_meta: list[dict] = []
    static_t = max_t = None
    fatal_error: str | None = None

    try:
        # --- 1) STATIC ------------------------------------------------------
        # IMPORTANT: build_matmul() has NOT been called yet, so no GEMM
        # warmup has run. Combined with the wait_for_cooldown below, this
        # gives us a genuine cold-idle measurement.
        if args.cooldown_c > 0:
            sampler.set_phase("cooldown_pre_static")
            wait_for_cooldown(handle, target_c=args.cooldown_c,
                              timeout_s=args.cooldown_timeout, verbose=False)
        print(f"\n[phase] static idle for {args.static_seconds}s")
        static_t = phase_static(sampler, args.static_seconds)

        # --- BUILD the GEMM (deferred until after static phase) ----------
        # Now we can warm up the matmul without contaminating idle. Also
        # if build itself fails (Blackwell amax race etc.), at least the
        # static reading is preserved.
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
                # Don't return — static phase already produced data; mark
                # max/leakage as unavailable and let the post-loop save
                # write the partial results.
                fatal_error = f"build: {e}"
                spec = None

        # --- 2) MAX ---------------------------------------------------------
        # Each compute phase is in its own try/except so that a CUDA fault
        # in max doesn't kill the leakage data, AND so partial telemetry
        # always reaches CSV/plots. _run_gemm_for() already classifies
        # fatal CUDA markers and raises RuntimeError on them.
        if not args.no_max and spec is not None:
            if args.cooldown_c > 0:
                sampler.set_phase("cooldown_pre_max")
                wait_for_cooldown(handle, target_c=args.cooldown_c,
                                  timeout_s=args.cooldown_timeout, verbose=False)
            print(f"\n[phase] max-power GEMM for {args.max_seconds}s "
                  f"(K={args.matmul_K} {args.dtype}/{args.mode})")
            try:
                max_t = phase_max(sampler, spec, args.max_seconds)
            except RuntimeError as e:
                print(f"[error] max phase failed: {e}")
                fatal_error = f"max: {e}"

        # --- 3) LEAKAGE -----------------------------------------------------
        if not args.no_leakage and not fatal_error and spec is not None:
            if args.cooldown_c > 0:
                sampler.set_phase("cooldown_pre_leakage")
                wait_for_cooldown(handle, target_c=args.cooldown_c,
                                  timeout_s=args.cooldown_timeout, verbose=False)
            print(f"\n[phase] leakage: {args.leakage_cycles} cycles of "
                  f"{args.leakage_stress_s}s stress + {args.leakage_decay_s}s decay")
            try:
                cycles_meta = phase_leakage(sampler, spec, args.leakage_cycles,
                                            args.leakage_stress_s,
                                            args.leakage_decay_s)
            except RuntimeError as e:
                print(f"[error] leakage phase failed mid-cycle: {e}")
                print(f"        partial cycles_meta will be reported")
                fatal_error = f"leakage: {e}"
    finally:
        sampler.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if fatal_error:
        summary["fatal_error"] = fatal_error

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
    phase_png            = base.parent / f"{base.name}_phases.png"
    leakage_png          = base.parent / f"{base.name}_leakage.png"
    leakage_enlarged_png = base.parent / f"{base.name}_leakage_enlarged.png"
    summary_png          = base.parent / f"{base.name}_summary.png"
    plot_phase_timeline(samples, summary, phase_png)
    if cycles_meta:
        plot_leakage_decay(samples, cycles_meta, summary, leakage_png)
        # Zoomed companion: first 3s, 0..150W. Catches the cycle-to-cycle
        # hot-window spread and the rapid first-second drop that the full
        # ~15s view compresses into a few pixels.
        plot_leakage_decay_zoomed(samples, cycles_meta, summary,
                                  leakage_enlarged_png,
                                  x_max=3.0, y_min=50.0, y_max=150.0)
        # Leakage(T) curve fit (P2.2 / G7) — Arrhenius-like exponential
        # plus linear sanity-check, on (T, P) pairs from decay windows.
        leak_t_fit = fit_leakage_temperature(samples, cycles_meta)
        if leak_t_fit["n_points"] >= 5:
            # Bake fit parameters into summary CSV for AccelWattch
            # consumption.
            for k, v in leak_t_fit.items():
                if isinstance(v, tuple):
                    summary[f"leakage_t_{k}_min"] = v[0]
                    summary[f"leakage_t_{k}_max"] = v[1]
                else:
                    summary[f"leakage_t_{k}"] = v
            leakage_t_png = base.parent / f"{base.name}_leakage_temperature.png"
            plot_leakage_temperature(samples, cycles_meta, leak_t_fit,
                                     leakage_t_png, gpu_name)
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
        print(f"[save] {leakage_enlarged_png}")
    print(f"[save] {summary_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
