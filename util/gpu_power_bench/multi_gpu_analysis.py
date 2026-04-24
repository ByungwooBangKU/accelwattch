#!/usr/bin/env python3
"""Cross-GPU variance analysis for same-model GPUs on one node.

Given a reports/ directory containing per-GPU sweeps written by
`./run_bench.sh --num-gpus N`, this tool aggregates them into:

  * per-variant statistics (mean, std, CV%, min, max) across GPUs
  * outlier-GPU detection (|deviation| from mean ≥ threshold·σ)
  * per-GPU idle/static-power ranking  — reveals silicon / cooling lottery
  * per-GPU thermal ranking             — reveals airflow / TIM outliers
  * plots: k_op bars with ±σ, variance heatmap, per-GPU power / temp bars

Typical workflow:
    ./run_bench.sh --num-gpus 8 --tag h100      # launches 8 parallel sweeps
    python3 multi_gpu_analysis.py reports/ 8 --tag h100

The tool discovers CSVs by globbing
`gpu_power_bench_*_<tag>_gpu<N>*.csv` under the reports directory and
deduplicating to one file per GPU (latest timestamp wins).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import summarize


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

_GPU_TAG_RE = re.compile(r"_gpu(\d+)(?:[._-]|$)")


def discover_csvs(reports_dir: Path, num_gpus: int,
                  tag: str | None = None) -> dict[int, Path]:
    """Find one per-GPU CSV per index in 0..num_gpus-1.

    Matching rule — be generous about name shape, but skip sidecars:
      * must contain `_gpu<N>` somewhere in the filename
      * must NOT end in `_baseline.csv`, `_baseline_stats.csv`, `_samples.csv`,
        `_summary.csv`, `_summary_by_regime.csv` (these are not per-cell files)
      * if `--tag` is given, must contain that tag as well
      * when more than one CSV matches an index, the newest mtime wins
    """
    sidecar = ("_baseline.csv", "_baseline_stats.csv",
               "_samples.csv", "_summary.csv",
               "_summary_by_regime.csv")
    candidates: dict[int, list[Path]] = {}
    for p in reports_dir.rglob("gpu_power_bench_*.csv"):
        name = p.name
        if any(name.endswith(s) for s in sidecar):
            continue
        if tag and tag not in name:
            continue
        m = _GPU_TAG_RE.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= num_gpus:
            continue
        candidates.setdefault(idx, []).append(p)
    result: dict[int, Path] = {}
    for idx, files in candidates.items():
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        result[idx] = files[0]
    return result


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_summaries(csvs: dict[int, Path]) -> pd.DataFrame:
    """Load each per-GPU CSV, summarise it, stack. Adds a `gpu_index` column."""
    frames = []
    for idx in sorted(csvs):
        df = pd.read_csv(csvs[idx])
        if df.empty:
            print(f"[warn] gpu{idx}: CSV {csvs[idx].name} is empty — skipping")
            continue
        s = summarize(df)
        s["gpu_index"] = idx
        # Carry the node-side gpu name forward so the report can label
        # which physical GPU each index maps to.
        s["gpu_name"] = df["gpu"].iloc[0] if "gpu" in df.columns else f"gpu{idx}"
        # Idle/static power is one value per run — pull it from the raw df.
        if "static_power_w" in df.columns:
            ps = pd.to_numeric(df["static_power_w"], errors="coerce").dropna()
            s["static_power_w_run"] = float(ps.iloc[0]) if len(ps) else float("nan")
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def variance_table(agg: pd.DataFrame,
                   value_col: str = "slope_dyn") -> pd.DataFrame:
    """One row per variant — mean/std/CV/min/max/outlier_gpus across GPUs.

    CV (coefficient of variation) = std / mean, percent. > 5% is worth
    checking; > 10% is almost certainly a cooling / silicon outlier.
    """
    if agg.empty:
        return pd.DataFrame()
    g = agg.groupby("variant", sort=False)
    rows = []
    for variant, sub in g:
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        if vals.empty:
            continue
        mean = float(vals.mean())
        std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        cv   = (std / mean * 100.0) if mean else float("nan")
        # Outlier GPUs = those ≥ 2σ from mean on this variant.
        if std > 0:
            zmask = (vals - mean).abs() >= 2.0 * std
            outliers = sub.loc[zmask.index[zmask], "gpu_index"].astype(int).tolist()
        else:
            outliers = []
        rows.append({
            "variant":   variant,
            "category":  sub["category"].iloc[0],
            "compute_unit": sub["compute_unit"].iloc[0],
            "n_gpus":    int(len(vals)),
            "mean":      mean,
            "std":       std,
            "cv_percent": cv,
            "min":       float(vals.min()),
            "max":       float(vals.max()),
            "range_rel": (float(vals.max()) - float(vals.min())) / mean * 100.0 if mean else float("nan"),
            "outlier_gpus_2sigma": ",".join(str(x) for x in outliers),
        })
    return pd.DataFrame(rows).sort_values(["category", "variant"]).reset_index(drop=True)


def per_gpu_scalar_table(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per GPU — idle power, mean dyn power, mean/peak temp averaged
    across the whole sweep. This is the 'per-card health card'."""
    if agg.empty:
        return pd.DataFrame()
    rows = []
    for idx, sub in agg.groupby("gpu_index"):
        rows.append({
            "gpu_index": int(idx),
            "gpu_name":  str(sub["gpu_name"].iloc[0]),
            "static_power_w":   float(sub["static_power_w_run"].iloc[0])
                                if "static_power_w_run" in sub else float("nan"),
            "mean_dyn_power_w": float(pd.to_numeric(sub["mean_dyn_power_w"],
                                                    errors="coerce").mean()),
            "mean_temp_c":      float(pd.to_numeric(sub["mean_temp_c"],
                                                    errors="coerce").mean()),
            "peak_temp_c":      float(pd.to_numeric(sub["peak_temp_c"],
                                                    errors="coerce").max()),
            "n_variants":       int(len(sub)),
        })
    return pd.DataFrame(rows).sort_values("gpu_index").reset_index(drop=True)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_coefficient_variance(var_df: pd.DataFrame, out_png: Path,
                              gpu_label: str) -> None:
    """Bar chart of mean k_op per variant, with ±σ error bars. Labels each
    bar with the CV% so outlier-prone variants stand out visually."""
    if var_df.empty:
        return
    plt = _get_mpl()
    ew = var_df[var_df["category"] == "elementwise"]
    mm = var_df[var_df["category"].isin(("matmul", "matmul_llm"))]
    # Two panels for unit-consistent comparisons.
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [3, 2]})

    def _panel(ax, sub, label, unit):
        if sub.empty:
            ax.set_visible(False)
            return
        xs = np.arange(len(sub))
        bars = ax.bar(xs, sub["mean"].values,
                      yerr=sub["std"].values, capsize=4,
                      color="#1f77b4", alpha=0.85, edgecolor="white",
                      error_kw=dict(ecolor="#d62728", lw=1.2))
        # Two-line label: value in pJ + CV%
        for rect, mean, cv in zip(bars, sub["mean"].values, sub["cv_percent"].values):
            txt = f"{mean*1e12:.2f} {unit}\nCV={cv:.1f}%"
            if cv >= 10:
                txt += "  ⚠"
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    txt, ha="center", va="bottom", fontsize=7, linespacing=1.1)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["variant"].values, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"k_op = slope_dyn  ({label})")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"{label} — mean ± σ across GPUs, CV% annotated")
        lo, hi = ax.get_ylim()
        if lo > 0 and np.isfinite(hi):
            ax.set_ylim(lo, hi * 4)

    _panel(axes[0], ew, "J / element (dyn)",  "pJ/elem")
    _panel(axes[1], mm, "J / FLOP (dyn)",    "pJ/FLOP")
    fig.suptitle(f"Cross-GPU variance — {gpu_label}  "
                 f"(⚠ = CV ≥ 10% → likely outlier GPU on this variant)",
                 y=1.00)
    fig.tight_layout(); fig.savefig(out_png, dpi=160, pad_inches=0.3)
    print(f"[save] {out_png}")


def plot_per_gpu_heatmap(agg: pd.DataFrame, var_df: pd.DataFrame,
                         out_png: Path, gpu_label: str) -> None:
    """(variant × gpu_index) heatmap of relative deviation from the cross-
    GPU mean, in percent. Red cells = that GPU is costlier than the pack on
    that variant; green = cheaper. Makes outlier patterns obvious — e.g.
    a single GPU hot on every variant = cooling issue; one variant hot on
    every GPU = variant-specific kernel noise."""
    if agg.empty or var_df.empty:
        return
    plt = _get_mpl()
    from matplotlib.colors import TwoSlopeNorm
    # Build matrix: rows=variant, cols=gpu_index, cells = (val - mean)/mean × 100.
    piv = agg.pivot_table(index="variant", columns="gpu_index",
                          values="slope_dyn", aggfunc="first")
    # Sort rows by category so elementwise / matmul cluster.
    order = var_df.sort_values(["category", "variant"])["variant"].tolist()
    piv = piv.reindex([v for v in order if v in piv.index])
    means = piv.mean(axis=1)
    rel = (piv.sub(means, axis=0).div(means, axis=0) * 100).astype(float)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * piv.shape[1] + 4),
                                    0.35 * len(piv) + 2.5))
    # Center colormap at 0 (no deviation) — saturate beyond ±10%.
    vmax = max(10.0, float(np.nanmax(np.abs(rel.values))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(rel.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(rel.shape[1]))
    ax.set_xticklabels([f"gpu{i}" for i in rel.columns])
    ax.set_yticks(range(len(rel)))
    ax.set_yticklabels(rel.index, fontsize=8)
    # Annotate each cell with % deviation.
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:+.1f}%"
            color = "white" if abs(v) > vmax * 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title(f"k_op deviation from cross-GPU mean (%) — {gpu_label}\n"
                 f"red = this GPU is costlier on this variant, blue = cheaper")
    fig.colorbar(im, ax=ax, label="relative deviation (%)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160, pad_inches=0.3)
    print(f"[save] {out_png}")


def plot_per_gpu_scalars(pg_df: pd.DataFrame, out_png: Path,
                         gpu_label: str) -> None:
    """Three bars per GPU: static power, mean dyn power, peak temp. Ranking
    the cards here reveals cooling or silicon-binning outliers at a glance."""
    if pg_df.empty:
        return
    plt = _get_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    xs = np.arange(len(pg_df))

    ax = axes[0]
    bars = ax.bar(xs, pg_df["static_power_w"].values,
                  color="#6a994e", alpha=0.85, edgecolor="white")
    mean = pg_df["static_power_w"].mean()
    ax.axhline(mean, color="#d62728", lw=1, ls="--",
               label=f"mean = {mean:.1f} W")
    for rect, v in zip(bars, pg_df["static_power_w"].values):
        if not np.isnan(v):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"gpu{i}" for i in pg_df["gpu_index"]],
                                          rotation=0)
    ax.set_ylabel("static (idle) power (W)")
    ax.set_title("Idle power per GPU\n(high = bad silicon / stuck clock)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(xs, pg_df["mean_dyn_power_w"].values,
                  color="#1f77b4", alpha=0.85, edgecolor="white")
    for rect, v in zip(bars, pg_df["mean_dyn_power_w"].values):
        if not np.isnan(v):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"gpu{i}" for i in pg_df["gpu_index"]])
    ax.set_ylabel("mean dyn power (W)")
    ax.set_title("Mean dynamic power per GPU (workload avg)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    w = 0.4
    ax.bar(xs - w/2, pg_df["mean_temp_c"].values, w,
           color="#ffb703", alpha=0.85, label="mean", edgecolor="white")
    ax.bar(xs + w/2, pg_df["peak_temp_c"].values, w,
           color="#d62728", alpha=0.85, label="peak", edgecolor="white")
    for i, (mn, pk) in enumerate(zip(pg_df["mean_temp_c"], pg_df["peak_temp_c"])):
        if not np.isnan(pk):
            ax.text(i + w/2, pk, f"{pk:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([f"gpu{i}" for i in pg_df["gpu_index"]])
    ax.set_ylabel("temperature (°C)")
    ax.set_title("Mean / peak temperature per GPU\n(high peak = airflow / TIM)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"Per-GPU health card — {gpu_label}", y=1.00)
    fig.tight_layout(); fig.savefig(out_png, dpi=160, pad_inches=0.3)
    print(f"[save] {out_png}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-GPU variance analysis for same-model GPUs on one node.")
    ap.add_argument("reports_dir", type=Path,
                    help="directory containing per-GPU benchmark CSVs "
                         "(written by run_bench.sh --num-gpus)")
    ap.add_argument("num_gpus", type=int,
                    help="how many GPUs to expect (will look for _gpu0.._gpu{N-1})")
    ap.add_argument("--tag", type=str, default=None,
                    help="only consider CSVs whose filename contains this tag")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the multi-GPU outputs (default: "
                         "<reports-dir>/multi_gpu_<tag>/)")
    args = ap.parse_args()

    if not args.reports_dir.is_dir():
        print(f"error: {args.reports_dir} is not a directory")
        return 1
    if args.num_gpus <= 0:
        print("error: num_gpus must be > 0"); return 1

    csvs = discover_csvs(args.reports_dir, args.num_gpus, args.tag)
    missing = [i for i in range(args.num_gpus) if i not in csvs]
    print(f"[discover] matched {len(csvs)}/{args.num_gpus} GPU CSVs "
          f"(tag={args.tag!r})")
    for idx in sorted(csvs):
        print(f"  gpu{idx}: {csvs[idx]}")
    if missing:
        print(f"[warn] no CSV found for GPU indices: {missing}")
    if not csvs:
        print("error: nothing to analyse — check --reports-dir / --tag")
        return 2

    # --- aggregate ---
    agg = aggregate_summaries(csvs)
    if agg.empty:
        print("error: all CSVs empty after summarise"); return 2
    var_df = variance_table(agg, value_col="slope_dyn")
    pg_df = per_gpu_scalar_table(agg)

    # --- output dir + file stem ---
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "mgpu"
    out_dir = args.out_dir or (args.reports_dir / f"multi_gpu_{tag}")
    out_dir.mkdir(exist_ok=True, parents=True)
    stem = f"multi_gpu_{tag}_{stamp}"
    label = f"{pg_df['gpu_name'].iloc[0]} × {len(csvs)}" if not pg_df.empty else tag

    # --- CSVs ---
    agg.to_csv(out_dir / f"{stem}_per_gpu_summary.csv", index=False)
    print(f"[save] {out_dir / f'{stem}_per_gpu_summary.csv'}")
    var_df.to_csv(out_dir / f"{stem}_variance.csv", index=False)
    print(f"[save] {out_dir / f'{stem}_variance.csv'}")
    pg_df.to_csv(out_dir / f"{stem}_per_gpu_scalars.csv", index=False)
    print(f"[save] {out_dir / f'{stem}_per_gpu_scalars.csv'}")

    # --- console summary ---
    print("\n== cross-GPU variance on k_op (slope_dyn) ==")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.3e}"):
        cols = ["variant", "category", "n_gpus", "mean", "std",
                "cv_percent", "min", "max", "outlier_gpus_2sigma"]
        print(var_df[cols].to_string(index=False))
    high_cv = var_df[var_df["cv_percent"] >= 10.0]
    if not high_cv.empty:
        print(f"\n⚠  {len(high_cv)} variant(s) with CV ≥ 10% — likely a per-GPU "
              f"outlier on those variants:")
        for _, r in high_cv.iterrows():
            print(f"    {r['variant']}: CV={r['cv_percent']:.1f}%, "
                  f"outlier GPUs (|z|≥2) = [{r['outlier_gpus_2sigma']}]")

    print("\n== per-GPU health card ==")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.2f}"):
        print(pg_df.to_string(index=False))

    # --- plots ---
    plot_coefficient_variance(var_df,
        out_dir / f"{stem}_01_coefficient_variance.png", label)
    plot_per_gpu_heatmap(agg, var_df,
        out_dir / f"{stem}_02_deviation_heatmap.png", label)
    plot_per_gpu_scalars(pg_df,
        out_dir / f"{stem}_03_per_gpu_health.png", label)

    print(f"\n[done] outputs under {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
