#!/usr/bin/env python3
"""Cross-GPU comparison — e.g. A100 vs H100.

Takes two or more benchmark CSVs (output of gpu_power_bench.py), computes
per-(variant, GPU) regression slopes, and renders:

  * `gpu_compare_<stamp>_summary.csv`
    One row per (variant, GPU) with slope_dyn (= J/element or J/FLOP),
    R², mean dynamic power, mean temperature. Also adds a "ratio vs
    baseline GPU" column so it's easy to read "H100 uses 0.3× the energy
    of A100 on fp16_mul", etc.

  * `gpu_compare_<stamp>_bar.png`
    Grouped bar chart of J/op for every variant, one color per GPU.
    Missing variants (e.g. fp8_te on A100 without TE) are simply skipped.

  * `gpu_compare_<stamp>_heatmap.png`
    (variant × GPU) heatmap of the energy ratio relative to the first-
    listed GPU. Green < 1 → destination GPU is cheaper; red > 1 → costlier.

  * `gpu_compare_<stamp>_static.png`
    Static (idle) power per GPU — this is the additive term in the
    energy model that doesn't depend on workload.

Typical use
-----------
    python3 compare_gpus.py \
        reports/gpu_power_bench_a100_80gb_20260420_120000.csv \
        reports/gpu_power_bench_h100_sxm_20260420_140000.csv \
        --baseline a100_80gb

The GPU name is derived from the "gpu" column of each CSV; `--baseline`
chooses which GPU the ratio columns compare against (defaults to the
first CSV on the command line).

What you should see for a healthy run
-------------------------------------
  * fp32_simt has by far the largest J/FLOP on both GPUs (no Tensor Cores).
  * tf32_tc / fp16_tc / bf16_tc roughly an order of magnitude lower.
  * On H100, matmul_fp8_te lowers it another 2–3× vs fp16_tc.
  * On A100, fp8_te shows a note "TE falls back to FP16 TC" and its J/FLOP
    will land close to fp16_tc, not below it — this is the diagnostic that
    FP8 gains require Hopper silicon.
  * Elementwise (mul/add/softmax/gelu/layernorm) energy is memory-bound, so
    H100 advantage there tracks HBM BW ratio (~1.5–2× cheaper per element).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import linear_fit, summarize


def load_and_summarize(csv: Path) -> tuple[str, pd.DataFrame, float]:
    df = pd.read_csv(csv)
    if df.empty:
        return "", pd.DataFrame(), float("nan")
    gpu = df["gpu"].iloc[0]
    p_static = df["static_power_w"].astype(float).iloc[0]
    s = summarize(df)
    s["gpu"] = gpu
    s["static_power_w"] = p_static
    return gpu, s, p_static


def pivot_on_variant(all_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return (variant × gpu) matrix of the chosen metric."""
    return all_summary.pivot(index="variant", columns="gpu", values=metric)


def add_ratio_column(df: pd.DataFrame, baseline_gpu: str) -> pd.DataFrame:
    """Attach `ratio_vs_baseline` = slope / slope_on_baseline (per variant)."""
    base = df[df["gpu"] == baseline_gpu][["variant", "slope_dyn"]].rename(
        columns={"slope_dyn": "_base_slope"})
    merged = df.merge(base, on="variant", how="left")
    merged["ratio_vs_baseline"] = merged["slope_dyn"] / merged["_base_slope"]
    merged = merged.drop(columns=["_base_slope"])
    merged["baseline_gpu"] = baseline_gpu
    return merged


def plot_bar(all_summary: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Split into elementwise and matmul panels since the metric units differ.
    ew = all_summary[all_summary["category"] == "elementwise"]
    mm = all_summary[all_summary["category"] == "matmul"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [3, 2]})
    gpus = sorted(all_summary["gpu"].unique())
    palette = {g: c for g, c in zip(gpus, ("#1f77b4", "#d62728", "#2ca02c",
                                           "#ff7f0e", "#9467bd"))}

    def _panel(ax, sub, metric_unit):
        if sub.empty:
            ax.set_visible(False)
            return
        variants = sorted(sub["variant"].unique())
        xpos = np.arange(len(variants))
        w = 0.8 / max(1, len(gpus))
        for i, g in enumerate(gpus):
            vals, r2s = [], []
            for v in variants:
                r = sub[(sub["variant"] == v) & (sub["gpu"] == g)]
                vals.append(r["slope_dyn"].iloc[0] if not r.empty else float("nan"))
                r2s.append(r["R2_dyn"].iloc[0] if not r.empty else float("nan"))
            bars = ax.bar(xpos + (i - (len(gpus) - 1) / 2) * w, vals, w,
                          label=g, color=palette[g], alpha=0.9)
            for rect, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            rect.get_height(), f"{v:.1e}",
                            ha="center", va="bottom", fontsize=6, rotation=0)
        ax.set_xticks(xpos)
        ax.set_xticklabels(variants, rotation=30, ha="right")
        ax.set_ylabel(f"slope_dyn ({metric_unit})")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    _panel(axes[0], ew, "J / element")
    axes[0].set_title("Elementwise — dynamic J per element")
    _panel(axes[1], mm, "J / FLOP")
    axes[1].set_title("Matmul — dynamic J per FLOP")
    fig.suptitle("Cross-GPU comparison — per-op energy coefficient")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


def plot_heatmap(all_summary: pd.DataFrame, baseline_gpu: str,
                 out_png: Path) -> None:
    """(variant × GPU) ratio heatmap, log-colored around 1.0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    piv = pivot_on_variant(all_summary, "slope_dyn")
    if baseline_gpu not in piv.columns:
        print(f"[warn] baseline GPU {baseline_gpu!r} not present; using {piv.columns[0]}")
        baseline_gpu = piv.columns[0]
    ratio = piv.div(piv[baseline_gpu], axis=0)

    # Order rows: elementwise first, then matmul, alphabetically within.
    is_matmul = ratio.index.to_series().str.startswith("matmul_")
    ratio = ratio.loc[list(ratio.index[~is_matmul]) + list(ratio.index[is_matmul])]

    fig, ax = plt.subplots(figsize=(max(6, 1 + 1.5 * ratio.shape[1]),
                                    0.4 * len(ratio) + 2))
    im = ax.imshow(ratio.values, aspect="auto", cmap="RdYlGn_r",
                   norm=LogNorm(vmin=0.1, vmax=10.0))
    ax.set_xticks(range(ratio.shape[1]))
    ax.set_xticklabels(ratio.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(ratio)))
    ax.set_yticklabels(ratio.index)
    # Annotate each cell with the ratio.
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            v = ratio.values[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.2f}×"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_title(f"J/op ratio vs baseline ({baseline_gpu})\n"
                 f"< 1 (green) = dest GPU is cheaper; > 1 (red) = more expensive")
    fig.colorbar(im, ax=ax, label="slope_dyn ratio (log scale)")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


def plot_static_power(statics: dict[str, float], out_png: Path) -> None:
    """Idle / static power bar — this is the P_static term in the energy model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gpus = list(statics.keys())
    vals = [statics[g] for g in gpus]
    fig, ax = plt.subplots(figsize=(1.5 * len(gpus) + 3, 4))
    bars = ax.bar(range(len(gpus)), vals, color="#1f77b4", alpha=0.9)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{v:.1f} W", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(gpus))); ax.set_xticklabels(gpus, rotation=20, ha="right")
    ax.set_ylabel("static (idle) power (W)")
    ax.set_title("P_static — the additive term in E(workload) = P_static·T + Σ k_op · N_op")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", type=Path, nargs="+",
                    help="benchmark CSVs from different GPUs to compare")
    ap.add_argument("--baseline", type=str, default=None,
                    help="GPU name (from the CSV's `gpu` column) to use as "
                         "denominator in the ratio column; default = first CSV's GPU")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    if len(args.csvs) < 2:
        print("need at least 2 CSVs to compare")
        return 1

    frames = []
    statics: dict[str, float] = {}
    for csv in args.csvs:
        gpu, s, p_static = load_and_summarize(csv)
        if s.empty:
            print(f"[warn] {csv}: empty summary, skipped"); continue
        frames.append(s)
        statics[gpu] = p_static
        print(f"[load] {csv} → {gpu}  (P_static={p_static:.1f} W, "
              f"{len(s)} variants)")
    if not frames:
        print("no usable CSVs")
        return 2
    all_summary = pd.concat(frames, ignore_index=True)

    baseline_gpu = args.baseline or all_summary["gpu"].iloc[0]
    all_summary = add_ratio_column(all_summary, baseline_gpu)

    args.out_dir.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""

    out_summary = args.out_dir / f"gpu_compare_{stamp}{suffix}_summary.csv"
    all_summary.to_csv(out_summary, index=False)
    print(f"[save] {out_summary}")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.3e}"):
        cols = ["gpu", "variant", "slope_dyn", "ratio_vs_baseline",
                "R2_dyn", "mean_dyn_power_w", "mean_temp_c"]
        print()
        print(all_summary[cols].to_string(index=False))

    plot_bar(all_summary,
             args.out_dir / f"gpu_compare_{stamp}{suffix}_bar.png")
    plot_heatmap(all_summary, baseline_gpu,
                 args.out_dir / f"gpu_compare_{stamp}{suffix}_heatmap.png")
    plot_static_power(statics,
                      args.out_dir / f"gpu_compare_{stamp}{suffix}_static.png")

    print(f"\nbaseline for ratios: {baseline_gpu}")
    print("ratio < 1 ⇒ dest GPU is cheaper per op than baseline; > 1 ⇒ more expensive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
