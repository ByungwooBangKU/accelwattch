#!/usr/bin/env python3
"""Post-process the benchmark CSV into linearity plots and a summary.

For each (op, dtype) pair we plot:
  * total dynamic energy (J) vs load (N_elements)   → should be linear
  * total wall time (s) vs load                     → should be linear
  * J/element (dyn) vs load                         → should be ~flat

The script also computes an R² for the linearity of total_dyn_energy vs total_elements
so you can sanity-check each benchmark's scaling in one column of the summary.

Usage:
  python3 analyze.py reports/gpu_power_bench_a100_*.csv
  python3 analyze.py reports/gpu_power_bench_h100_*.csv --samples reports/..._samples.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination for y ≈ a·x + b (a,b fit by least-squares)."""
    if len(x) < 2:
        return float("nan")
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(op,dtype) summary: slope, intercept, R² of energy vs total_elements."""
    out = []
    for (op, dt), g in df.groupby(["op", "dtype"]):
        g = g.sort_values("total_elements")
        x = g["total_elements"].to_numpy(dtype=float)
        y_dyn = g["dyn_energy_j"].to_numpy(dtype=float)
        y_tot = g["total_energy_j"].to_numpy(dtype=float)
        # Joule per element (dynamic) from the slope of dyn_energy vs total_elements.
        slope_dyn, _ = (np.polyfit(x, y_dyn, 1) if len(x) >= 2 else (float("nan"), 0))
        slope_tot, _ = (np.polyfit(x, y_tot, 1) if len(x) >= 2 else (float("nan"), 0))
        out.append({
            "op": op, "dtype": dt, "n_points": len(g),
            "slope_J_per_elem_dyn":   slope_dyn,
            "slope_J_per_elem_total": slope_tot,
            "R2_dyn_vs_N":            _r2(x, y_dyn),
            "R2_total_vs_N":          _r2(x, y_tot),
            "mean_dyn_power_w":       g["dyn_power_w"].mean(),
            "mean_avg_power_w":       g["avg_power_w"].mean(),
            "mean_temp_c":            g["avg_temp_c"].mean(),
            "peak_temp_c":            g["peak_temp_c"].max(),
        })
    return pd.DataFrame(out)


def plot_linearity(df: pd.DataFrame, out_png: Path, gpu: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = sorted(df["op"].unique())
    dtypes = sorted(df["dtype"].unique(), reverse=True)  # fp16 first

    # 3 rows × (len ops) cols: energy, time, J/elem
    fig, axes = plt.subplots(3, len(ops), figsize=(4 * len(ops), 10),
                             squeeze=False)
    colors = {"fp16": "#1f77b4", "fp8": "#d62728"}
    markers = {"fp16": "o", "fp8": "s"}

    for ci, op in enumerate(ops):
        ax_e = axes[0][ci]
        ax_t = axes[1][ci]
        ax_j = axes[2][ci]
        ax_e.set_title(f"{op} — E_dyn vs load")
        ax_t.set_title(f"{op} — wall time vs load")
        ax_j.set_title(f"{op} — J/elem (dyn)")
        for dt in dtypes:
            g = df[(df.op == op) & (df.dtype == dt)].sort_values("total_elements")
            if g.empty:
                continue
            x  = g["total_elements"].to_numpy(dtype=float)
            ye = g["dyn_energy_j"].to_numpy(dtype=float)
            yt = g["wall_s"].to_numpy(dtype=float)
            yj = g["j_per_element_dyn"].to_numpy(dtype=float)
            r2 = _r2(x, ye)
            ax_e.plot(x, ye, marker=markers[dt], color=colors[dt],
                      label=f"{dt}  R²={r2:.3f}")
            ax_t.plot(x, yt, marker=markers[dt], color=colors[dt], label=dt)
            ax_j.plot(x, yj, marker=markers[dt], color=colors[dt], label=dt)
        for ax in (ax_e, ax_t, ax_j):
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            ax.set_xlabel("total elements (iters × N)")
            ax.legend(fontsize=8)
        ax_e.set_ylabel("dyn energy (J)")
        ax_t.set_ylabel("wall time (s)")
        ax_j.set_ylabel("J / element (dyn)")
        # Log Y for energy/time emphasizes linearity (a line on a log-log plot).
        ax_e.set_yscale("log"); ax_t.set_yscale("log")

    fig.suptitle(f"GPU power benchmark — {gpu}", y=1.00)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


def plot_timeline(samples_csv: Path, out_png: Path, gpu: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(samples_csv)
    fig, (ax_p, ax_t) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax_p.plot(s["t_s"], s["power_w"], lw=0.6, color="#1f77b4")
    ax_p.set_ylabel("power (W)"); ax_p.grid(True, alpha=0.3)
    ax_t.plot(s["t_s"], s["temp_c"], lw=0.6, color="#d62728")
    ax_t.set_xlabel("time (s)"); ax_t.set_ylabel("temp (°C)")
    ax_t.grid(True, alpha=0.3)

    # Shade each phase block along the bottom of the power plot.
    phase = s["phase"].fillna("")
    changes = np.where(phase.values[1:] != phase.values[:-1])[0] + 1
    edges = np.r_[0, changes, len(phase)]
    for a, b in zip(edges[:-1], edges[1:]):
        label = phase.iloc[a]
        if not label or label in ("", "gap", "idle"):
            continue
        t0 = s["t_s"].iloc[a]; t1 = s["t_s"].iloc[b - 1]
        ax_p.axvspan(t0, t1, alpha=0.08, color="orange")

    fig.suptitle(f"Power / temperature timeline — {gpu}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="benchmark CSV (per-cell rows)")
    ap.add_argument("--samples", type=Path, default=None,
                    help="raw NVML samples CSV (for timeline plot)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write plots (default: same dir as csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("empty CSV"); return 1
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(exist_ok=True, parents=True)
    gpu = df["gpu"].iloc[0]
    stem = args.csv.stem

    summary = summarize(df)
    summary_path = out_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.to_string(index=False))

    plot_linearity(df, out_dir / f"{stem}_linearity.png", gpu)

    # Auto-discover samples file if not given.
    if args.samples is None:
        cand = args.csv.with_name(stem + "_samples.csv")
        if cand.exists():
            args.samples = cand
    if args.samples and args.samples.exists():
        plot_timeline(args.samples, out_dir / f"{stem}_timeline.png", gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
