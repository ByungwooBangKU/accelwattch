#!/usr/bin/env python3
"""Analyse a benchmark CSV into plots + a power-model summary.

Reads one or more per-cell CSVs written by gpu_power_bench.py and emits:

  1. `<stem>_summary.csv`
     Per (category, op, dtype, mode) line with the linear-fit slope
     (= joule per element or joule per FLOP) and its R². These slopes are
     the primary *power-modeling coefficients* — each benchmark gives one
     number that can go directly into a per-op cost table.

  2. `<stem>_linearity_elementwise.png`
     For every elementwise benchmark (mul/add/softmax/gelu/layernorm × fp16/fp8):
       row 1 : E_dyn (dynamic Joules) vs total_elements      (log-log → straight line ≡ linear)
       row 2 : wall time (s) vs total_elements               (sanity: throughput scaling)
       row 3 : J / element (dynamic)                         (flat = linearity holds)

  3. `<stem>_linearity_matmul.png` (only if the CSV has matmul rows)
     For every matmul variant (fp32_simt / tf32_tc / fp16_tc / bf16_tc / fp8_te):
       row 1 : E_dyn vs total_FLOPs                          (log-log; slope = J/FLOP)
       row 2 : wall time vs total_FLOPs                      (1/wall ∝ throughput)
       row 3 : J / FLOP (dynamic)                            (flat = within linearity)

  4. `<stem>_joule_per_op_bar.png`
     Side-by-side bar chart of the regression slopes (J/elem for
     elementwise, J/FLOP for matmul). This is the "one number per
     benchmark" view — good for reporting and for cross-checking against
     the summary CSV.

  5. `<stem>_static_power.png`  (only if a *_baseline.csv is found)
     Three panels showing the static (idle) power baseline:
       * P_static(t) during the idle window with mean and ±σ band —
         flat = clean idle; drift = clock ramp / stray process.
       * Per-cell stacked bar of static_energy_j vs dyn_energy_j — shows
         how much of each measurement was "just the GPU being on".
       * Static-energy share (%) per cell — high share means the
         workload is too short relative to static overhead; lengthen
         --window-ms if you see >50% static on the biggest loads.

  6. `<stem>_temperature.png`
     Thermal context for every cell — three panels:
       * per-cell (start / avg / peak) temperature bars. Start = the
         cool-down floor the run began from; peak = the hottest sample
         during the measurement window. A large gap between start and
         peak (= temp_rise_c) means the workload heats the die fast.
       * per-cell cool-down elapsed time (s). Flat → no thermal drift
         across the sweep; rising → later cells are starting hotter.
       * J/op (slope_dyn) vs avg_temp_c scatter. If a variant's points
         line up with a positive slope, that op is getting more
         expensive as the GPU warms — flag for temp-aware modelling.

  7. `<stem>_timeline.png`  (only if the companion *_samples.csv exists)
     Global power / temperature / SM-clock trace with each cell shaded —
     useful to eyeball thermal stability and clock throttling.

Why linearity matters for power modeling
----------------------------------------
A simple first-order GPU energy model is:

    E(workload) = P_static · T  +  Σ_i  k_op_i · N_op_i

Where `k_op_i` is the "Joules per op of kind i" coefficient.  If the
E_dyn-vs-N plot is a straight line (R² ≥ 0.99), that assumption holds
for this op on this GPU, and you can use the regression slope as the k_op
coefficient directly.  The bar chart / summary CSV give exactly these
slopes.  Non-linearity (R² lower, or J/elem drifting) signals either
launch-overhead (load too small) or memory-BW saturation (load too
large) — you need to restrict the fit to the linear regime in that case.

Usage
-----
    python3 analyze.py reports/gpu_power_bench_a100_*.csv
    python3 analyze.py reports/gpu_power_bench_h100_*.csv --samples reports/...samples.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# utility: R² for the linear fit y ≈ a·x + b, and the slope itself.
# ---------------------------------------------------------------------------

def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R²).  Slope is the primary power-model coeff."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


# ---------------------------------------------------------------------------
# summary: one regression per (category, op, dtype, mode).
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Power-modeling summary.

    For elementwise rows the regression is E_dyn vs total_elements → slope is
    *J per element*.  For matmul rows we regress against total_FLOPs → slope
    is *J per FLOP* (which is the right axis since FLOPs scale as K³ while
    element count scales as K²).

    Columns `compute_unit` and `emulated` are propagated so downstream tables
    and plots can distinguish "CUDA core" vs "Tensor Core" paths, and flag
    emulated cases (fp8 elementwise, fp8_te on pre-Hopper).
    """
    out = []
    group_keys = ["category", "op", "dtype", "mode"]
    # Some older CSVs may lack "category"/"mode" columns — fall back gracefully.
    for col in group_keys:
        if col not in df.columns:
            df[col] = "elementwise" if col in ("category", "mode") else df.get(col, "")
    # Back-compat: older CSVs won't have compute_unit / emulated columns.
    if "compute_unit" not in df.columns:
        df["compute_unit"] = df["category"].map(
            lambda c: "Tensor Core" if c == "matmul" else "CUDA core")
    if "emulated" not in df.columns:
        df["emulated"] = 0
    for (cat, op, dt, mode), g in df.groupby(group_keys):
        g = g.sort_values("total_elements")
        # Axis choice: FLOPs for matmul (K³ scaling), elements otherwise.
        if cat == "matmul":
            x = g["total_flops"].to_numpy(dtype=float)
            unit = "J/FLOP"
        else:
            x = g["total_elements"].to_numpy(dtype=float)
            unit = "J/element"
        y_dyn = g["dyn_energy_j"].to_numpy(dtype=float)
        y_tot = g["total_energy_j"].to_numpy(dtype=float)
        slope_dyn, _, r2_dyn = linear_fit(x, y_dyn)
        slope_tot, _, r2_tot = linear_fit(x, y_tot)
        # compute_unit / emulated are constant within a group — take the first.
        compute_unit = str(g["compute_unit"].iloc[0])
        emulated = int(bool(g["emulated"].iloc[0]))
        out.append({
            "category": cat, "op": op, "dtype": dt, "mode": mode,
            "variant": f"{op}_{dt}_{mode}" if cat == "matmul" else f"{dt}_{op}",
            "compute_unit": compute_unit,
            "emulated":     emulated,
            "n_points": len(g),
            "fit_axis": unit,
            "slope_dyn":   slope_dyn,   # ← the power-modeling coefficient
            "slope_total": slope_tot,
            "R2_dyn":      r2_dyn,
            "R2_total":    r2_tot,
            "mean_dyn_power_w":  g["dyn_power_w"].mean(),
            "mean_avg_power_w":  g["avg_power_w"].mean(),
            "mean_temp_c":       g["avg_temp_c"].mean(),
            "peak_temp_c":       g["peak_temp_c"].max(),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_linearity_elementwise(df: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """3 × 5 grid : one column per op, rows = E_dyn, wall, J/elem.

    X-axis is total_elements (iters × N). A log-log E_dyn vs N plot should be
    a straight line of slope ≈ 1 if E scales linearly; slope < 1 means
    launch-overhead-dominated, slope > 1 means BW-saturated or TC-bound."""
    ew = df[df["category"] == "elementwise"]
    if ew.empty:
        return
    plt = _get_mpl()
    ops = [o for o in ("mul", "add", "softmax", "gelu", "layernorm")
           if o in ew["op"].unique()]
    dtypes = sorted(ew["dtype"].unique(), reverse=True)  # fp16 first
    fig, axes = plt.subplots(3, len(ops), figsize=(4 * len(ops), 10),
                             squeeze=False)
    colors = {"fp16": "#1f77b4", "fp8": "#d62728"}
    markers = {"fp16": "o", "fp8": "s"}

    for ci, op in enumerate(ops):
        ax_e, ax_t, ax_j = axes[0][ci], axes[1][ci], axes[2][ci]
        ax_e.set_title(f"{op} — E_dyn vs N")
        ax_t.set_title(f"{op} — wall time vs N")
        ax_j.set_title(f"{op} — J/elem (dyn)")
        for dt in dtypes:
            g = ew[(ew.op == op) & (ew.dtype == dt)].sort_values("total_elements")
            if g.empty:
                continue
            x = g["total_elements"].to_numpy(float)
            ye = g["dyn_energy_j"].to_numpy(float)
            yt = g["wall_s"].to_numpy(float)
            yj = g["j_per_element_dyn"].to_numpy(float)
            _, _, r2 = linear_fit(x, ye)
            c = colors.get(dt, "gray"); m = markers.get(dt, "x")
            ax_e.plot(x, ye, marker=m, color=c, label=f"{dt}  R²={r2:.3f}")
            ax_t.plot(x, yt, marker=m, color=c, label=dt)
            ax_j.plot(x, yj, marker=m, color=c, label=dt)
        for ax in (ax_e, ax_t, ax_j):
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            ax.set_xlabel("total elements (iters × N)")
            ax.legend(fontsize=8)
        ax_e.set_ylabel("dyn energy (J)")
        ax_t.set_ylabel("wall time (s)")
        ax_j.set_ylabel("J / element (dyn)")
        ax_e.set_yscale("log"); ax_t.set_yscale("log")

    fig.suptitle(f"Elementwise benchmarks — {gpu}", y=1.00)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


def plot_linearity_matmul(df: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """Matmul: x = total_FLOPs (not elements), slope = J/FLOP."""
    mm = df[df["category"] == "matmul"]
    if mm.empty:
        return
    plt = _get_mpl()
    variants = sorted(mm["variant"].unique())
    if not variants:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), squeeze=False)
    ax_e, ax_t, ax_j = axes[0][0], axes[1][0], axes[2][0]
    ax_e.set_title("matmul — E_dyn vs FLOPs (slope = J/FLOP)")
    ax_t.set_title("matmul — wall time vs FLOPs")
    ax_j.set_title("matmul — J/FLOP (dyn)")
    palette = {"matmul_fp32_simt": "#555555",
               "matmul_tf32_tc":   "#ff7f0e",
               "matmul_fp16_tc":   "#1f77b4",
               "matmul_bf16_tc":   "#2ca02c",
               "matmul_fp8_te":    "#d62728"}
    for v in variants:
        g = mm[mm["variant"] == v].sort_values("total_flops")
        if g.empty:
            continue
        x = g["total_flops"].to_numpy(float)
        ye = g["dyn_energy_j"].to_numpy(float)
        yt = g["wall_s"].to_numpy(float)
        yj = g["j_per_flop_dyn"].to_numpy(float)
        _, _, r2 = linear_fit(x, ye)
        c = palette.get(v, None)
        # Decorate the legend with the actual compute unit (CUDA vs TC) and
        # an "*EMU" marker when the measurement isn't the native HW path
        # (fp8_te on A100 falls back to FP16 TC).
        cu = str(g.get("compute_unit", pd.Series(["Tensor Core"])).iloc[0])
        emu = int(g.get("emulated", pd.Series([0])).iloc[0])
        tag = "TC" if cu.startswith("Tensor") else "CUDA"
        if cu == "Tensor Core (FP16 fallback)":
            tag = "TC·FP16-fallback"
        star = " *EMU" if emu else ""
        ax_e.plot(x, ye, marker="o", color=c,
                  label=f"{v} [{tag}]{star}  R²={r2:.3f}")
        ax_t.plot(x, yt, marker="o", color=c, label=f"{v} [{tag}]{star}")
        ax_j.plot(x, yj, marker="o", color=c, label=f"{v} [{tag}]{star}")
    for ax in (ax_e, ax_t, ax_j):
        ax.set_xscale("log"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("total FLOPs (iters × 2MNK)")
        ax.legend(fontsize=9)
    ax_e.set_ylabel("dyn energy (J)")
    ax_t.set_ylabel("wall time (s)")
    ax_j.set_ylabel("J / FLOP (dyn)")
    ax_e.set_yscale("log"); ax_t.set_yscale("log"); ax_j.set_yscale("log")
    fig.suptitle(f"Matmul (Tensor Core vs CUDA core vs TE FP8) — {gpu}", y=1.00)
    # If any variant was measured on the wrong HW path (e.g. fp8_te on A100
    # silently using FP16 TC), spell that out below the plot so the reader
    # doesn't mistake the FP8 bar for a real FP8 number.
    if "emulated" in mm.columns and int(mm["emulated"].max()) == 1:
        fig.text(0.5, -0.01,
                 "*EMU = emulated / fallback path — NOT a native measurement "
                 "of the named dtype (e.g. fp8_te on pre-Hopper uses FP16 TC)",
                 ha="center", fontsize=8, color="#d62728")
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[save] {out_png}")


def plot_joule_per_op_bar(summary: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """One bar per benchmark with the slope (J/element or J/FLOP).

    Two panels side by side:
      * elementwise — bars grouped by op, color by dtype
      * matmul      — bars by variant
    This is the picture that goes in a report: "here's how much it costs
    per op on this GPU".
    """
    plt = _get_mpl()
    ew = summary[summary["category"] == "elementwise"]
    mm = summary[summary["category"] == "matmul"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 2]})

    # ---- elementwise panel: grouped bar (op on x, dtype as hue) ----
    ax = axes[0]
    if not ew.empty:
        ops = sorted(ew["op"].unique())
        dtypes = sorted(ew["dtype"].unique(), reverse=True)
        xpos = np.arange(len(ops))
        w = 0.8 / max(1, len(dtypes))
        colors = {"fp16": "#1f77b4", "fp8": "#d62728"}
        has_emulated = False
        for i, dt in enumerate(dtypes):
            vals, r2s = [], []
            for op in ops:
                row = ew[(ew.op == op) & (ew.dtype == dt)]
                if row.empty:
                    vals.append(float("nan")); r2s.append(float("nan"))
                else:
                    vals.append(row["slope_dyn"].iloc[0])
                    r2s.append(row["R2_dyn"].iloc[0])
                    if "emulated" in row.columns and int(row["emulated"].iloc[0]):
                        has_emulated = True
            # fp8 bars are drawn with a diagonal hatch to visually flag them
            # as the cast-compute-cast emulation path.
            emu_dt = (dt == "fp8")
            label = f"{dt} [CUDA]" + (" *EMU" if emu_dt else "")
            bars = ax.bar(xpos + (i - (len(dtypes) - 1) / 2) * w, vals, w,
                          label=label, color=colors.get(dt, None), alpha=0.9,
                          hatch="//" if emu_dt else None, edgecolor="white")
            for rect, r2 in zip(bars, r2s):
                if not np.isnan(r2):
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            rect.get_height(), f"R²={r2:.2f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xpos); ax.set_xticklabels(ops)
        ax.set_ylabel("J / element (dynamic)  — regression slope")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        subtitle = "Elementwise — per-op energy coefficient  (all on CUDA cores)"
        if has_emulated:
            subtitle += "\nfp8 bars = cast-compute-cast via FP16 (no native FP8 elementwise in PyTorch)"
        ax.set_title(subtitle, fontsize=10)
    else:
        ax.set_visible(False)

    # ---- matmul panel: one bar per variant ----
    ax = axes[1]
    if not mm.empty:
        order = ["matmul_fp32_simt", "matmul_tf32_tc", "matmul_fp16_tc",
                 "matmul_bf16_tc", "matmul_fp8_te"]
        mm2 = mm.set_index("variant").reindex([v for v in order if v in mm["variant"].values])
        palette = {"matmul_fp32_simt": "#555555",
                   "matmul_tf32_tc":   "#ff7f0e",
                   "matmul_fp16_tc":   "#1f77b4",
                   "matmul_bf16_tc":   "#2ca02c",
                   "matmul_fp8_te":    "#d62728"}
        colors = [palette.get(v, "gray") for v in mm2.index]
        # Hatch fp8_te bar when it was measured on the fallback FP16 TC path
        # (pre-Hopper). Readers should NOT interpret that height as FP8 cost.
        def _emu(v):
            if "emulated" in mm2.columns:
                val = mm2.loc[v, "emulated"]
                return bool(int(val)) if pd.notna(val) else False
            return False
        hatches = ["//" if _emu(v) else None for v in mm2.index]
        bars = ax.bar(range(len(mm2)), mm2["slope_dyn"].values,
                      color=colors, alpha=0.9, edgecolor="white")
        for rect, h in zip(bars, hatches):
            if h:
                rect.set_hatch(h)
        for rect, r2 in zip(bars, mm2["R2_dyn"].values):
            if not np.isnan(r2):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(), f"R²={r2:.2f}",
                        ha="center", va="bottom", fontsize=7)
        # Build tick labels that carry the actual compute unit ([CUDA]/[TC])
        # and a star when the variant was emulated (fp8_te fallback).
        def _tick(v):
            cu = str(mm2.loc[v, "compute_unit"]) if "compute_unit" in mm2.columns else ""
            if cu.startswith("Tensor Core (FP16 fallback)"):
                tag = "TC·FP16-fallback"
            elif cu.startswith("Tensor"):
                tag = "TC"
            elif cu.startswith("CUDA"):
                tag = "CUDA"
            else:
                tag = "?"
            star = " *" if _emu(v) else ""
            return f"{v}\n[{tag}]{star}"
        ax.set_xticks(range(len(mm2)))
        ax.set_xticklabels([_tick(v) for v in mm2.index],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("J / FLOP (dynamic)  — regression slope")
        ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
        title = "Matmul — per-variant energy coefficient"
        if any(hatches):
            title += "\n* hatched bar = emulated (not native for this dtype)"
        ax.set_title(title, fontsize=10)
    else:
        ax.set_visible(False)

    fig.suptitle(f"Power-model coefficients — {gpu}")
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[save] {out_png}")


def plot_static_power(df: pd.DataFrame, baseline_csv: Path | None,
                      out_png: Path, gpu: str) -> None:
    """Static-power diagnostics — the P_static term in E = P_static·T + Σ k_op·N.

    Panel A : idle power trace (if the baseline sidecar exists). A clean
              idle should be a flat horizontal line within ~1 W of the mean.
              Drift or spikes → another process on the GPU or clocks still
              ramping down; the reported P_static will be pessimistic.
    Panel B : stacked bar per cell, static_energy_j on bottom (grey) +
              dyn_energy_j on top (blue for elementwise, orange for matmul).
              Shows how much of each measurement is "just keeping the
              GPU on" vs. "actually running the op".
    Panel C : static-energy share (%) per cell. If this is >50% on any
              cell, increase --window-ms until the dyn part dominates.
    """
    plt = _get_mpl()
    have_trace = baseline_csv is not None and baseline_csv.exists()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2, 1.4], hspace=0.45)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharex=ax_b)

    # ---------------- A: idle-window power trace ----------------
    p_static_mean = float(df["static_power_w"].astype(float).iloc[0])
    if have_trace:
        bdf = pd.read_csv(baseline_csv)
        if not bdf.empty and bdf["power_w"].notna().any():
            t = bdf["t_s"].to_numpy(float)
            p = bdf["power_w"].to_numpy(float)
            mean = float(np.mean(p))
            std = float(np.std(p))
            ax_a.plot(t, p, lw=0.8, color="#1f77b4", label=f"idle power (n={len(p)})")
            ax_a.axhline(mean, color="#d62728", lw=1.2,
                         label=f"mean = {mean:.2f} W")
            ax_a.fill_between(t, mean - std, mean + std, color="#d62728",
                              alpha=0.15, label=f"±σ ({std:.2f} W)")
            if "temp_c" in bdf.columns and bdf["temp_c"].notna().any():
                temp_mean = float(bdf["temp_c"].mean())
                ax_a.set_title(f"P_static(t) during idle window  —  "
                               f"{mean:.2f} ± {std:.2f} W  @ {temp_mean:.1f}°C")
            else:
                ax_a.set_title(f"P_static(t) during idle window  —  "
                               f"{mean:.2f} ± {std:.2f} W")
            ax_a.set_xlabel("time since sampler start (s)")
            ax_a.set_ylabel("power (W)")
            ax_a.grid(True, alpha=0.3)
            ax_a.legend(loc="upper right", fontsize=8)
        else:
            ax_a.text(0.5, 0.5, "baseline CSV empty", ha="center", va="center",
                      transform=ax_a.transAxes)
            ax_a.set_axis_off()
    else:
        ax_a.text(0.5, 0.5,
                  f"no *_baseline.csv found — showing CSV-recorded P_static = "
                  f"{p_static_mean:.2f} W only",
                  ha="center", va="center", transform=ax_a.transAxes)
        ax_a.axhline(p_static_mean, color="#d62728", lw=1.2)
        ax_a.set_ylabel("power (W)")
        ax_a.set_title("P_static (from CSV column — no raw trace)")
        ax_a.grid(True, alpha=0.3)

    # ---------------- B+C: per-cell static vs dynamic energy ----------------
    # Order: elementwise first (by op, dtype), then matmul (by variant, K).
    def _cell_key(r):
        cat = r.get("category", "elementwise")
        if cat == "matmul":
            return (1, r.get("variant", ""), int(r["load_value"]))
        return (0, r["op"], r["dtype"], int(r["load_value"]))

    rows = df.to_dict("records")
    rows.sort(key=_cell_key)
    labels = []
    stat_e = []
    dyn_e = []
    share = []
    colors_dyn = []
    for r in rows:
        se = float(r["static_energy_j"])
        de = float(r["dyn_energy_j"])
        total = se + de
        labels.append(_short_label(r))
        stat_e.append(se)
        dyn_e.append(de)
        share.append(100.0 * se / total if total > 0 else 0.0)
        colors_dyn.append("#ff7f0e" if r.get("category") == "matmul" else "#1f77b4")

    x = np.arange(len(labels))
    ax_b.bar(x, stat_e, color="#b0b0b0", label="static energy (P_static · T)")
    ax_b.bar(x, dyn_e, bottom=stat_e, color=colors_dyn,
             label="dynamic energy (workload)")
    ax_b.set_ylabel("energy (J)")
    ax_b.set_title("Per-cell energy breakdown — grey=static (idle overhead), "
                   "blue=elementwise dyn, orange=matmul dyn")
    ax_b.grid(True, axis="y", alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=9)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([""] * len(labels))

    bars_c = ax_c.bar(x, share, color=colors_dyn, alpha=0.7)
    ax_c.axhline(50.0, color="#d62728", lw=1, ls="--",
                 label="50 % — consider --window-ms ↑")
    for rect, s in zip(bars_c, share):
        ax_c.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                  f"{s:.0f}%", ha="center", va="bottom", fontsize=6)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=80, ha="right", fontsize=7)
    ax_c.set_ylabel("static share (%)")
    ax_c.set_ylim(0, max(100.0, max(share) * 1.1 if share else 100.0))
    ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Static (idle) power diagnostics — {gpu}", y=0.995)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[save] {out_png}")


def plot_temperature(df: pd.DataFrame, summary: pd.DataFrame, out_png: Path,
                     gpu: str) -> None:
    """Thermal diagnostics per cell — complements the static-power plot.

    - Top panel : start / avg / peak temp per cell (grouped bar).
    - Mid panel : cool-down time spent before each cell (s).
    - Bottom panel : J/op vs mean temp scatter per variant — shows whether
      any op's energy cost drifts with temperature.
    """
    plt = _get_mpl()
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.2, 2.2], hspace=0.55)
    ax_t = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1], sharex=ax_t)
    ax_s = fig.add_subplot(gs[2])

    # ---- A+B: per-cell temperatures and cool-down time ----
    def _cell_key(r):
        cat = r.get("category", "elementwise")
        if cat == "matmul":
            return (1, r.get("variant", ""), int(r["load_value"]))
        return (0, r["op"], r["dtype"], int(r["load_value"]))

    rows = df.to_dict("records")
    rows.sort(key=_cell_key)
    labels = [_short_label(r) for r in rows]
    x = np.arange(len(labels))
    start = np.array([float(r.get("start_temp_c", -1)) for r in rows])
    avg = np.array([float(r["avg_temp_c"]) for r in rows])
    peak = np.array([float(r["peak_temp_c"]) for r in rows])
    cd = np.array([float(r.get("cooldown_elapsed_s", 0)) for r in rows])

    w = 0.27
    ax_t.bar(x - w, np.where(start >= 0, start, 0), w, color="#8ecae6",
             label="start (post-cooldown)")
    ax_t.bar(x,      avg,  w, color="#ffb703", label="avg during cell")
    ax_t.bar(x + w,  peak, w, color="#d62728", label="peak")
    ax_t.set_ylabel("temperature (°C)")
    ax_t.set_title("Per-cell thermal state — start (blue), avg (orange), peak (red)")
    ax_t.grid(True, axis="y", alpha=0.3)
    ax_t.legend(loc="upper left", fontsize=9, ncol=3)
    ax_t.tick_params(labelbottom=False)
    # Annotate Δ (temp_rise) above each peak bar so the reader sees how much
    # heat this op added relative to its cool-down floor.
    for xi, s, p in zip(x, start, peak):
        if s >= 0 and p >= 0 and p > s:
            ax_t.text(xi + w, p, f"Δ{int(p - s)}", ha="center", va="bottom",
                      fontsize=6, color="#d62728")

    bars_c = ax_c.bar(x, cd, color="#6a994e", alpha=0.85)
    ax_c.set_ylabel("cooldown (s)")
    ax_c.set_title("Cool-down time before each cell (flat = thermal state uniform across sweep)")
    ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=80, ha="right", fontsize=7)
    for rect, v in zip(bars_c, cd):
        if v > 0:
            ax_c.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                      f"{v:.0f}", ha="center", va="bottom", fontsize=6)

    # ---- C: does J/op drift with temperature? ----
    # For every cell, plot (avg_temp_c, j_per_* dyn) colored by variant.
    variant_col = "variant" if "variant" in df.columns else "op"
    groups = df.groupby(variant_col)
    palette = plt.get_cmap("tab20")
    for i, (name, g) in enumerate(groups):
        xs = g["avg_temp_c"].astype(float).to_numpy()
        cat = g["category"].iloc[0] if "category" in g else "elementwise"
        if cat == "matmul":
            ys = g["j_per_flop_dyn"].astype(float).to_numpy()
        else:
            ys = g["j_per_element_dyn"].astype(float).to_numpy()
        ax_s.scatter(xs, ys, s=28, color=palette(i % 20), alpha=0.75,
                     label=str(name), edgecolors="none")
    ax_s.set_xlabel("avg GPU temperature during cell (°C)")
    ax_s.set_ylabel("J / element (eltwise) or J / FLOP (matmul) — dynamic")
    ax_s.set_yscale("log")
    ax_s.grid(True, alpha=0.3)
    ax_s.set_title("Per-cell energy vs temperature — horizontal cloud = no thermal sensitivity")
    ax_s.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
                ncol=1, frameon=False)

    fig.suptitle(f"Thermal diagnostics — {gpu}", y=0.995)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[save] {out_png}")


def _short_label(r: dict) -> str:
    """Compact cell label used as x-tick in the static-power bar chart."""
    if r.get("category") == "matmul":
        return f"{r.get('variant','matmul')}·K{int(r['load_value'])}"
    return f"{r['dtype']}·{r['op']}·N{int(r['load_value'])}"


def plot_timeline(samples_csv: Path, out_png: Path, gpu: str) -> None:
    plt = _get_mpl()
    s = pd.read_csv(samples_csv)
    fig, (ax_p, ax_t, ax_c) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax_p.plot(s["t_s"], s["power_w"], lw=0.6, color="#1f77b4")
    ax_p.set_ylabel("power (W)"); ax_p.grid(True, alpha=0.3)
    ax_t.plot(s["t_s"], s["temp_c"], lw=0.6, color="#d62728")
    ax_t.set_ylabel("temp (°C)"); ax_t.grid(True, alpha=0.3)
    ax_c.plot(s["t_s"], s["sm_mhz"], lw=0.6, color="#2ca02c", label="SM")
    ax_c.plot(s["t_s"], s["mem_mhz"], lw=0.6, color="#ff7f0e", label="MEM")
    ax_c.set_xlabel("time (s)"); ax_c.set_ylabel("clock (MHz)")
    ax_c.grid(True, alpha=0.3); ax_c.legend()

    # Shade each non-idle phase block.
    phase = s["phase"].fillna("")
    changes = np.where(phase.values[1:] != phase.values[:-1])[0] + 1
    edges = np.r_[0, changes, len(phase)]
    for a, b in zip(edges[:-1], edges[1:]):
        lbl = phase.iloc[a]
        if not lbl or lbl in ("", "gap", "idle"):
            continue
        t0 = s["t_s"].iloc[a]; t1 = s["t_s"].iloc[b - 1]
        for ax in (ax_p, ax_t, ax_c):
            ax.axvspan(t0, t1, alpha=0.06, color="orange")

    fig.suptitle(f"Power / temp / clock timeline — {gpu}")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[save] {out_png}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="benchmark CSV (per-cell rows)")
    ap.add_argument("--samples", type=Path, default=None,
                    help="raw NVML samples CSV (for timeline plot)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="idle / static-power baseline CSV "
                         "(auto-discovered as <stem>_baseline.csv)")
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

    # --- summary / power-model coefficient extraction ---
    summary = summarize(df)
    summary_path = out_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.3e}"):
        cols = ["category", "variant", "compute_unit", "emulated",
                "n_points", "fit_axis",
                "slope_dyn", "R2_dyn", "mean_dyn_power_w",
                "mean_temp_c", "peak_temp_c"]
        print(summary[cols].to_string(index=False))

    # --- plots ---
    plot_linearity_elementwise(df, out_dir / f"{stem}_linearity_elementwise.png", gpu)
    plot_linearity_matmul(df, out_dir / f"{stem}_linearity_matmul.png", gpu)
    plot_joule_per_op_bar(summary, out_dir / f"{stem}_joule_per_op_bar.png", gpu)

    # Static-power diagnostics (auto-discover the baseline sidecar).
    if args.baseline is None:
        cand = args.csv.with_name(stem + "_baseline.csv")
        if cand.exists():
            args.baseline = cand
    plot_static_power(df, args.baseline,
                      out_dir / f"{stem}_static_power.png", gpu)
    plot_temperature(df, summary,
                     out_dir / f"{stem}_temperature.png", gpu)

    # Timeline (auto-discover samples file if not given).
    if args.samples is None:
        cand = args.csv.with_name(stem + "_samples.csv")
        if cand.exists():
            args.samples = cand
    if args.samples and args.samples.exists():
        plot_timeline(args.samples, out_dir / f"{stem}_timeline.png", gpu)

    print("\nHow to read the summary CSV:")
    print("  slope_dyn  — Joules per element (elementwise) / per FLOP (matmul).")
    print("               This is the power-modeling coefficient for this op+GPU.")
    print("  R2_dyn     — linearity of E_dyn ~ load.  ≥0.99 = model assumption holds.")
    print("  Lower R² → restrict your fit to loads in the linear regime, then re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
