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
    # form 1 — explicit CSV path
    python3 analyze.py reports/gpu_power_bench_h100_20260421_*.csv

    # form 2 — point at a reports/ directory and pick by tag
    python3 analyze.py --reports-dir reports/ --tag h100
    python3 analyze.py --reports-dir reports/ --tag a100

    # add a global timeline if you have the sampler sidecar:
    python3 analyze.py reports/foo.csv --samples reports/foo_samples.csv
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
    # Include llm_preset in the group key so each (layer-role × dtype) in the
    # matmul_llm sweep gets its own regression. For non-LLM rows llm_preset
    # is empty string (or missing) and doesn't change the grouping.
    group_keys = ["category", "op", "dtype", "mode", "llm_preset"]
    # Some older CSVs may lack "category"/"mode"/"llm_preset" columns — fall
    # back gracefully.  IMPORTANT: pandas read_csv turns blank cells into
    # NaN by default, and DataFrame.groupby drops rows whose group keys are
    # NaN — which would silently empty out the summary when the CSV has an
    # llm_preset column that's empty on every non-LLM row. We coerce to
    # empty string before the groupby so every row is retained.
    df = df.copy()
    for col in group_keys:
        if col not in df.columns:
            df[col] = "elementwise" if col in ("category", "mode") else ""
        else:
            # Existing column: fill any NaN / None with "" so groupby keeps
            # those rows. `fillna("")` is cheap even when no NaNs exist.
            df[col] = df[col].fillna("").astype(str)
    # Back-compat: older CSVs won't have compute_unit / emulated columns.
    if "compute_unit" not in df.columns:
        df["compute_unit"] = df["category"].map(
            lambda c: "Tensor Core" if c in ("matmul", "matmul_llm") else "CUDA core")
    if "emulated" not in df.columns:
        df["emulated"] = 0
    for (cat, op, dt, mode, preset), g in df.groupby(group_keys):
        g = g.sort_values("total_elements")
        # Axis choice: FLOPs for matmul (K³ scaling), elements otherwise.
        if cat in ("matmul", "matmul_llm"):
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
        # Variant name — encodes preset when this is an LLM-shape row so
        # analyze / compare plots can tell qkv / mlp1 / lm_head apart.
        if cat == "matmul_llm":
            variant = f"llm_{preset}_{dt}_{mode}"
        elif cat == "matmul":
            variant = f"{op}_{dt}_{mode}"
        else:
            variant = f"{dt}_{op}"
        out.append({
            "category": cat, "op": op, "dtype": dt, "mode": mode,
            "llm_preset": preset,
            "variant": variant,
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


def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Same power-model regression as summarize(), but grouped additionally
    by cache_regime.

    Rationale: within a sweep, J/element for elementwise ops jumps by ~1
    order of magnitude across the L2→DRAM boundary. Regressing through
    all cells together produces a slope that's an uninformative mix of
    the two regimes. Splitting the regression per regime yields the
    regime-specific k_op coefficient the power model actually wants
    ("joule per element while cache-resident" vs "while DRAM-streaming").

    Returns one row per (category, op, dtype, mode, cache_regime). Rows
    where the regime has <2 points report slope=NaN / R²=NaN because a
    linear fit needs at least two samples; the median J/element is still
    emitted so a coarser reading is possible.
    """
    if "cache_regime" not in df.columns:
        return pd.DataFrame()
    out = []
    group_keys = ["category", "op", "dtype", "mode", "llm_preset", "cache_regime"]
    # Same NaN-in-group-keys gotcha as summarize() — read_csv produces NaN
    # for blank cells and groupby silently drops those rows. cache_regime
    # NaNs become "unknown" (which is then filtered below); everything else
    # becomes the empty string so non-LLM rows still participate.
    df = df.copy()
    for col in group_keys:
        default = "unknown" if col == "cache_regime" else ""
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)
    if "compute_unit" not in df.columns:
        df["compute_unit"] = df["category"].map(
            lambda c: "Tensor Core" if c in ("matmul", "matmul_llm") else "CUDA core")
    if "emulated" not in df.columns:
        df["emulated"] = 0
    # Back-compat: older CSVs used the 3-bucket vocabulary — fold those
    # onto the 5-bucket equivalents so the summary rows stay consistent
    # regardless of which version produced the data.
    df = df.copy()
    df["cache_regime"] = df["cache_regime"].replace({
        "l2_resident": "l2_hit_100",
        "l2_partial":  "l2_hit_50",
        "dram_stream": "l2_hit_0",
    })
    for (cat, op, dt, mode, preset, regime), g in df.groupby(group_keys):
        if regime == "unknown":
            continue
        g = g.sort_values("total_elements")
        if cat in ("matmul", "matmul_llm"):
            x = g["total_flops"].to_numpy(dtype=float)
            y_per = g["j_per_flop_dyn"].astype(float)
            unit = "J/FLOP"
        else:
            x = g["total_elements"].to_numpy(dtype=float)
            y_per = g["j_per_element_dyn"].astype(float)
            unit = "J/element"
        y_dyn = g["dyn_energy_j"].to_numpy(dtype=float)
        if len(g) >= 2:
            slope_dyn, _, r2_dyn = linear_fit(x, y_dyn)
        else:
            # Single-point regime — use the per-point coefficient directly.
            # This is the one case where we degrade the model to median.
            slope_dyn, r2_dyn = float(y_per.iloc[0]), float("nan")
        compute_unit = str(g["compute_unit"].iloc[0])
        emulated = int(bool(g["emulated"].iloc[0]))
        if cat == "matmul_llm":
            variant = f"llm_{preset}_{dt}_{mode}"
        elif cat == "matmul":
            variant = f"{op}_{dt}_{mode}"
        else:
            variant = f"{dt}_{op}"
        out.append({
            "category": cat, "op": op, "dtype": dt, "mode": mode,
            "llm_preset": preset,
            "cache_regime": regime,
            "variant": variant,
            "compute_unit": compute_unit,
            "emulated":     emulated,
            "n_points":     len(g),
            "fit_axis":     unit,
            "slope_dyn":      slope_dyn,          # regime-specific k_op
            "R2_dyn":         r2_dyn,
            "median_j_per_unit": float(y_per.median()),
            "mean_dyn_power_w":  g["dyn_power_w"].astype(float).mean(),
            "mean_temp_c":       g["avg_temp_c"].astype(float).mean(),
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
    fig, axes = plt.subplots(3, len(ops), figsize=(4.5 * len(ops), 12),
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
    fig.tight_layout(); fig.savefig(out_png, dpi=160)
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
    fig, axes = plt.subplots(3, 1, figsize=(14, 17), squeeze=False)
    ax_e, ax_t, ax_j = axes[0][0], axes[1][0], axes[2][0]
    ax_e.set_title("matmul — E_dyn vs FLOPs (slope = J/FLOP)")
    ax_t.set_title("matmul — wall time vs FLOPs")
    ax_j.set_title("matmul — J/FLOP (dyn)  [annotated with the swept K]")
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
        ks = g["load_value"].to_numpy(int) if "load_value" in g.columns else None
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
        # Annotate each point in the sweep so the reader can see which K the
        # point corresponds to (top panel) and the actual J/FLOP at that K
        # (bottom panel). Matmul slopes depend heavily on problem size, so
        # these numbers are what a power-model consumer actually needs.
        if ks is not None:
            for xi, yi, k in zip(x, ye, ks):
                ax_e.annotate(f"K={k}", (xi, yi),
                              textcoords="offset points", xytext=(5, 5),
                              fontsize=7, color=c, alpha=0.85)
            for xi, yi, k in zip(x, yj, ks):
                ax_j.annotate(f"K={k}\n{yi:.2e}", (xi, yi),
                              textcoords="offset points", xytext=(0, 6),
                              ha="center", fontsize=6.5, color=c, alpha=0.9)
    for ax in (ax_e, ax_t, ax_j):
        ax.set_xscale("log"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("total FLOPs (iters × 2MNK)")
        ax.legend(fontsize=9)
    ax_e.set_ylabel("dyn energy (J)")
    ax_t.set_ylabel("wall time (s)")
    ax_j.set_ylabel("J / FLOP (dyn)")
    ax_e.set_yscale("log"); ax_t.set_yscale("log"); ax_j.set_yscale("log")
    # Give each panel extra headroom on its log y-axis so the per-point
    # "K=..." / "J/FLOP" annotations don't overlap the top of the frame.
    for ax in (ax_e, ax_t, ax_j):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 6)
    fig.suptitle(f"Matmul (Tensor Core vs CUDA core vs TE FP8) — {gpu}", y=1.00)
    # If any variant was measured on the wrong HW path (e.g. fp8_te on A100
    # silently using FP16 TC), spell that out below the plot so the reader
    # doesn't mistake the FP8 bar for a real FP8 number.
    if "emulated" in mm.columns and int(mm["emulated"].max()) == 1:
        fig.text(0.5, -0.01,
                 "*EMU = emulated / fallback path — NOT a native measurement "
                 "of the named dtype (e.g. fp8_te on pre-Hopper uses FP16 TC)",
                 ha="center", fontsize=8, color="#d62728")
    fig.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[save] {out_png}")


def plot_joule_per_op_bar(summary: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """One bar per benchmark with the slope (J/element or J/FLOP).

    Two panels side by side:
      * elementwise — bars grouped by op, color by dtype
      * matmul      — bars by variant
    Each bar is annotated with both the numeric coefficient (pJ/element for
    elementwise, pJ/FLOP for matmul — small-number-friendly units) and the
    regression R², so the plot alone answers "how much does this op cost
    on this GPU" without the reader having to open the CSV.
    """
    plt = _get_mpl()
    ew = summary[summary["category"] == "elementwise"]
    mm = summary[summary["category"] == "matmul"]
    # Figure height bumped vs before: each bar now carries two lines of text
    # ("0.31 pJ / R²=0.99"), which needs extra headroom on a log-scale y-axis.
    fig, axes = plt.subplots(1, 2, figsize=(17, 7),
                             gridspec_kw={"width_ratios": [3, 2]})

    def _annot(rect, value_j, r2, scale_label):
        """Write '0.31 pJ\\nR²=0.99' on top of a bar, skipping NaNs."""
        if value_j is None or np.isnan(value_j):
            return
        # Pick human-friendly precision: large values (>=100 pJ) get one
        # decimal, smaller ones get two, very small (<0.1 pJ) use scientific.
        v_p = value_j * 1e12   # J → pJ
        if abs(v_p) >= 100:
            vtxt = f"{v_p:.0f} {scale_label}"
        elif abs(v_p) >= 1:
            vtxt = f"{v_p:.2f} {scale_label}"
        elif abs(v_p) >= 0.01:
            vtxt = f"{v_p:.3f} {scale_label}"
        else:
            vtxt = f"{v_p:.2e} {scale_label}"
        if np.isnan(r2):
            label = vtxt
        else:
            label = f"{vtxt}\nR²={r2:.3f}"
        ax = rect.axes
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                label, ha="center", va="bottom", fontsize=7,
                linespacing=1.1)

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
            for rect, v, r2 in zip(bars, vals, r2s):
                _annot(rect, v, r2, "pJ/elem")
        ax.set_xticks(xpos); ax.set_xticklabels(ops)
        ax.set_ylabel("J / element (dynamic)  — regression slope")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        subtitle = ("Elementwise — per-op energy coefficient (all on CUDA cores). "
                    "Labels: slope (pJ/elem) + R²")
        if has_emulated:
            subtitle += "\nfp8 bars = cast-compute-cast via FP16 (no native FP8 elementwise in PyTorch)"
        ax.set_title(subtitle, fontsize=10)
        # Bumping the upper ylim by ~1 decade on log scale so the two-line
        # labels don't clip the title.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 3)
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
        for rect, v, r2 in zip(bars, mm2["slope_dyn"].values,
                                mm2["R2_dyn"].values):
            _annot(rect, float(v) if pd.notna(v) else float("nan"),
                   float(r2) if pd.notna(r2) else float("nan"),
                   "pJ/FLOP")
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
        title = ("Matmul — per-variant energy coefficient. "
                 "Labels: slope (pJ/FLOP) + R²")
        if any(hatches):
            title += "\n* hatched bar = emulated (not native for this dtype)"
        ax.set_title(title, fontsize=10)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 3)
    else:
        ax.set_visible(False)

    fig.suptitle(f"Power-model coefficients — {gpu}")
    fig.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches="tight")
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
    # Width scales with the number of cells so the x-tick labels don't stack
    # on top of each other when the sweep is long. Minimum 16" keeps small
    # runs (~10 cells) readable; larger sweeps grow up to 32".
    n_cells = len(df)
    fig_w = max(16, min(32, 0.28 * n_cells + 10))
    fig = plt.figure(figsize=(fig_w, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2.2, 1.6], hspace=0.55)
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
        if cat == "matmul_llm":
            return (2, r.get("llm_preset", ""), r.get("dtype", ""),
                    int(r["load_value"]))
        return (0, r["op"], r["dtype"], int(r["load_value"]))

    rows = df.to_dict("records")
    rows.sort(key=_cell_key)
    labels = []
    stat_e = []
    dyn_e = []
    share = []
    colors_dyn = []
    groups = []           # one entry per cell: the "which experiment" string
    for r in rows:
        se = float(r["static_energy_j"])
        de = float(r["dyn_energy_j"])
        total = se + de
        labels.append(_short_label(r))
        stat_e.append(se)
        dyn_e.append(de)
        share.append(100.0 * se / total if total > 0 else 0.0)
        colors_dyn.append("#ff7f0e" if r.get("category") == "matmul" else "#1f77b4")
        # Group key = the experiment identity minus the swept load value.
        # Every cell in the same sweep shares this key, so we can draw one
        # bracket over the cells that belong to the same benchmark.
        if r.get("category") == "matmul":
            groups.append(str(r.get("variant", "matmul")))
        else:
            groups.append(f"{r.get('dtype', '')}·{r.get('op', '')}")

    x = np.arange(len(labels))
    ax_b.bar(x, stat_e, color="#b0b0b0", label="static energy (P_static · T)")
    ax_b.bar(x, dyn_e, bottom=stat_e, color=colors_dyn,
             label="dynamic energy (workload)")
    ax_b.set_ylabel("energy (J)")
    ax_b.set_title("Per-cell energy breakdown — grey = static (idle overhead), "
                   "blue = elementwise dyn, orange = matmul dyn")
    ax_b.grid(True, axis="y", alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=10)
    ax_b.tick_params(labelbottom=False)

    # --- group brackets: one span per benchmark sweep -----------------------
    # Compute contiguous runs of identical `groups[i]` and draw:
    #   (a) a vertical separator between groups on panels B and C,
    #   (b) a horizontal bracket + label above panel B naming the experiment.
    # This is what the user asked for: "이 구간때 어떤 실험을 했는지 보여줘".
    if groups:
        y_top = (np.array(stat_e) + np.array(dyn_e)).max()
        bracket_y = y_top * 1.08
        text_y = y_top * 1.14
        ax_b.set_ylim(0, y_top * 1.22)
        runs = []
        lo = 0
        for i in range(1, len(groups) + 1):
            if i == len(groups) or groups[i] != groups[lo]:
                runs.append((lo, i - 1, groups[lo]))
                lo = i
        for start, end, name in runs:
            ax_b.plot([start - 0.4, end + 0.4], [bracket_y, bracket_y],
                      color="#333333", lw=1.0, clip_on=False)
            ax_b.plot([start - 0.4, start - 0.4],
                      [bracket_y, bracket_y - y_top * 0.015],
                      color="#333333", lw=1.0, clip_on=False)
            ax_b.plot([end + 0.4, end + 0.4],
                      [bracket_y, bracket_y - y_top * 0.015],
                      color="#333333", lw=1.0, clip_on=False)
            ax_b.text((start + end) / 2.0, text_y, name,
                      ha="center", va="bottom", fontsize=9,
                      fontweight="bold", color="#222222")
        # Dotted separators between sweeps extend across both panels.
        for (_, end, _), (nxt_start, _, _) in zip(runs[:-1], runs[1:]):
            sep = (end + nxt_start) / 2.0
            for ax in (ax_b, ax_c):
                ax.axvline(sep, color="#888888", lw=0.6, ls=":", alpha=0.8)

    bars_c = ax_c.bar(x, share, color=colors_dyn, alpha=0.7)
    ax_c.axhline(50.0, color="#d62728", lw=1, ls="--",
                 label="50 % — consider --window-ms ↑")
    for rect, s in zip(bars_c, share):
        ax_c.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                  f"{s:.0f}%", ha="center", va="bottom", fontsize=7)
    ax_c.set_xticks(x)
    # 45° tick labels read left-to-right much more easily than the old 80°,
    # which at 7pt visually collapsed to vertical streaks for long sweeps.
    ax_c.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_c.set_ylabel("static share (%)")
    ax_c.set_ylim(0, max(100.0, max(share) * 1.1 if share else 100.0))
    ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Static (idle) power diagnostics — {gpu}", y=0.995)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
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
    n_cells = len(df)
    fig_w = max(17, min(32, 0.28 * n_cells + 11))
    fig = plt.figure(figsize=(fig_w, 17))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.2, 2.2], hspace=0.55)
    ax_t = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1], sharex=ax_t)
    ax_s = fig.add_subplot(gs[2])

    # ---- A+B: per-cell temperatures and cool-down time ----
    def _cell_key(r):
        cat = r.get("category", "elementwise")
        if cat == "matmul":
            return (1, r.get("variant", ""), int(r["load_value"]))
        if cat == "matmul_llm":
            return (2, r.get("llm_preset", ""), r.get("dtype", ""),
                    int(r["load_value"]))
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
    # Headroom so the Δ labels don't sit on the title bar.
    tmax = float(np.nanmax(peak)) if len(peak) else 100.0
    ax_t.set_ylim(0, tmax * 1.15 if tmax > 0 else 100)

    bars_c = ax_c.bar(x, cd, color="#6a994e", alpha=0.85)
    ax_c.set_ylabel("cooldown (s)")
    ax_c.set_title("Cool-down time before each cell (flat = thermal state uniform across sweep)")
    ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for rect, v in zip(bars_c, cd):
        if v > 0:
            ax_c.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                      f"{v:.0f}", ha="center", va="bottom", fontsize=7)

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
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[save] {out_png}")


def _short_label(r: dict) -> str:
    """Compact cell label used as x-tick in the static-power bar chart."""
    if r.get("category") == "matmul":
        return f"{r.get('variant','matmul')}·K{int(r['load_value'])}"
    if r.get("category") == "matmul_llm":
        return f"llm·{r.get('llm_preset','?')}·{r.get('dtype','?')}·T{int(r['load_value'])}"
    return f"{r['dtype']}·{r['op']}·N{int(r['load_value'])}"


def plot_cache_regime(df: pd.DataFrame, by_regime: pd.DataFrame,
                      out_png: Path, gpu: str) -> None:
    """Energy-per-operation grouped by cache locality regime, for both
    elementwise and matmul categories.

    Layout is a 2 × 3 grid:
      Row 1 — elementwise (ops: mul / add / softmax / gelu / layernorm)
      Row 2 — matmul      (5 variants: fp32_simt / tf32_tc / fp16_tc / bf16_tc / fp8_te)
    Columns (shared between rows):
      A : per-cell strip of the per-op unit (J/elem or J/FLOP) vs regime.
      B : regime-specific slope_dyn (k_op) from summarize_by_regime, with
          numeric value annotated on each bar.
      C : mean dynamic power (W) per regime — DRAM streaming raises
          steady-state power, not just energy per op.

    Matmul note: matmul has intrinsic reuse (O(K) per element), so even
    DRAM-sized GEMMs often stay closer to compute-bound than elementwise
    DRAM-streaming. The per-regime bars for matmul therefore typically
    show a smaller gap than elementwise — that's physics, not a bug.
    """
    if "cache_regime" not in df.columns:
        return
    plt = _get_mpl()
    # 5-bucket cache-locality vocabulary (new). Legacy 3-bucket CSVs are
    # still readable below — `_keep_row` filters strictly against whatever
    # is present in the data, and the legacy column values
    # (l2_resident / l2_partial / dram_stream) are mapped onto the new
    # labels for plotting so old and new data can analyse the same way.
    regime_order = ["l2_hit_100", "l2_hit_75", "l2_hit_50",
                    "l2_hit_25", "l2_hit_0"]
    regime_hit_rate = {"l2_hit_100": "~100%", "l2_hit_75": "~75%",
                       "l2_hit_50": "~50%",  "l2_hit_25": "~25%",
                       "l2_hit_0":  "~0%"}
    _legacy_to_new = {"l2_resident": "l2_hit_100",
                      "l2_partial":  "l2_hit_50",
                      "dram_stream": "l2_hit_0"}
    regime_x = {r: i for i, r in enumerate(regime_order)}

    # Map legacy labels onto the new 5-bucket vocabulary up-front so the
    # rest of the function can stay single-vocabulary. No-op on new data.
    df = df.copy()
    df["cache_regime"] = df["cache_regime"].replace(_legacy_to_new)

    ew = df[df["category"] == "elementwise"].copy()
    ew = ew[ew["cache_regime"].isin(regime_order)]
    mm = df[df["category"] == "matmul"].copy()
    mm = mm[mm["cache_regime"].isin(regime_order)]
    if ew.empty and mm.empty:
        return
    if not ew.empty:
        ew["cache_regime"] = pd.Categorical(ew["cache_regime"],
                                            categories=regime_order, ordered=True)
    if not mm.empty:
        mm["cache_regime"] = pd.Categorical(mm["cache_regime"],
                                            categories=regime_order, ordered=True)

    n_rows = (1 if not ew.empty else 0) + (1 if not mm.empty else 0)
    fig, axes = plt.subplots(n_rows, 3, figsize=(22, 7 * n_rows),
                             gridspec_kw={"width_ratios": [3, 3, 2]},
                             squeeze=False)

    def _draw_row(row_ax, cat_df, cat_sum, cat_name, keys_key, key_palette,
                  unit_col, unit_label, annot_scale, annot_suffix, title_prefix):
        """Populate one (cat) row across the 3 shared-semantics columns."""
        ax_a, ax_b, ax_c = row_ax
        keys = [k for k in key_palette.keys() if k in cat_df[keys_key].unique()]
        # Fallback: if the palette doesn't cover all keys (user added ops),
        # append remaining keys in deterministic order.
        extras = sorted(k for k in cat_df[keys_key].unique() if k not in keys)
        keys = keys + extras
        extra_cmap = plt.get_cmap("tab20")
        colors = dict(key_palette)
        for i, k in enumerate(extras):
            colors[k] = extra_cmap(i % 20)

        # --- Panel A: per-cell strip ---------------------------------------
        # Strip plot on a log scale — filter non-positive values. When a
        # column has zero legitimate points after the filter (e.g. every
        # matmul cell reported j_per_flop_dyn=0 because of a bad fit), the
        # panel stays empty but we don't crash and don't let matplotlib
        # auto-scale to absurd bounds.
        any_a_pts = False
        for key in keys:
            g = cat_df[cat_df[keys_key] == key]
            if g.empty:
                continue
            ys_raw = pd.to_numeric(g[unit_col], errors="coerce")
            mask = ys_raw.notna() & (ys_raw > 0)
            if not mask.any():
                continue
            xs = g["cache_regime"].map(regime_x).astype(float) \
                + (0.05 * (hash(str(key)) % 7 - 3))
            ax_a.scatter(xs[mask.values], ys_raw[mask].values,
                         s=42, color=colors[key], marker="o",
                         alpha=0.8, edgecolors="white", label=str(key))
            any_a_pts = True
        ax_a.set_xticks(list(regime_x.values()))
        ax_a.set_xticklabels([f"{r}\n({regime_hit_rate[r]} L2 hit)"
                              for r in regime_order])
        ax_a.set_ylabel(f"{unit_label} (per cell, dynamic)")
        if any_a_pts:
            ax_a.set_yscale("log")
        ax_a.grid(True, axis="y", alpha=0.3)
        ax_a.set_title(f"{title_prefix}A. Raw spread — every cell")
        ax_a.legend(fontsize=8, ncol=1, loc="upper left",
                    bbox_to_anchor=(1.0, 1.0))

        # --- Panel B: k_op bars with numeric annotation --------------------
        if cat_sum is None or cat_sum.empty:
            ax_b.set_visible(False)
        else:
            width = 0.8 / max(1, len(keys))
            xpos = np.arange(len(regime_order))
            positive_vals: list[float] = []   # used for safe log-scale ylim
            for i, key in enumerate(keys):
                vals, r2s = [], []
                for r in regime_order:
                    row = cat_sum[(cat_sum[keys_key] == key)
                                  & (cat_sum["cache_regime"] == r)]
                    if row.empty:
                        vals.append(np.nan); r2s.append(np.nan)
                    else:
                        sl = float(row["slope_dyn"].iloc[0])
                        if not np.isfinite(sl) or sl <= 0:
                            # Fall back to the median of per-cell J/unit when
                            # the regression slope is non-physical (NaN, ≤ 0
                            # happens for single-point regimes with intercept
                            # noise). Median is always a real positive J/unit
                            # value if ANY cell had one.
                            med = float(row["median_j_per_unit"].iloc[0])
                            sl = med if np.isfinite(med) and med > 0 else np.nan
                        vals.append(sl)
                        r2s.append(float(row["R2_dyn"].iloc[0]))
                bars = ax_b.bar(xpos + (i - (len(keys)-1)/2) * width, vals, width,
                                label=str(key), color=colors[key], alpha=0.9,
                                edgecolor="white")
                for rect, v, r2 in zip(bars, vals, r2s):
                    if not np.isfinite(v) or v <= 0:
                        continue
                    positive_vals.append(v)
                    v_p = v * annot_scale
                    if abs(v_p) >= 1:
                        vtxt = f"{v_p:.2f}"
                    elif abs(v_p) >= 0.01:
                        vtxt = f"{v_p:.3f}"
                    else:
                        vtxt = f"{v_p:.2e}"
                    txt = f"{vtxt} {annot_suffix}"
                    if np.isfinite(r2):
                        txt += f"\nR²={r2:.2f}"
                    ax_b.text(rect.get_x() + rect.get_width()/2,
                              rect.get_height(), txt,
                              ha="center", va="bottom", fontsize=7,
                              linespacing=1.1)
            ax_b.set_xticks(xpos)
            ax_b.set_xticklabels([f"{r}\n({regime_hit_rate[r]})" for r in regime_order])
            ax_b.set_ylabel(f"k_op = slope_dyn  ({unit_label})")
            # Only switch to log scale when we actually have ≥ 1 positive bar
            # — otherwise matplotlib can't pick finite log bounds and the
            # figure expands unboundedly when bbox_inches="tight" is applied.
            if positive_vals:
                ax_b.set_yscale("log")
                lo = min(positive_vals)
                hi = max(positive_vals)
                # Anchor the limits explicitly — don't ask matplotlib to
                # auto-scale. 0.3× below lowest bar, 5× above highest. This
                # guarantees the figure stays finite even when some bars are
                # NaN / 0 and others are tiny (mixes of both produced the
                # ~1.3 Gpx image earlier).
                ax_b.set_ylim(lo * 0.3, hi * 5)
            ax_b.grid(True, axis="y", alpha=0.3)
            ax_b.set_title(f"{title_prefix}B. k_op per regime "
                           f"(annotated in {annot_suffix})")
            ax_b.legend(fontsize=8, ncol=min(len(keys), 3), loc="upper left")

        # --- Panel C: mean dyn_power_w per regime --------------------------
        mean_p = {}
        for r in regime_order:
            g = cat_df[cat_df["cache_regime"] == r]
            vals = pd.to_numeric(g.get("dyn_power_w", pd.Series(dtype=float)),
                                 errors="coerce").dropna()
            mean_p[r] = float(vals.mean()) if len(vals) else float("nan")
        x_regime = list(range(len(regime_order)))
        y_power  = [mean_p[r] for r in regime_order]
        bar_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
        bars_c = ax_c.bar(x_regime, y_power, color=bar_colors, alpha=0.85)
        for rect, v in zip(bars_c, y_power):
            if not np.isnan(v):
                ax_c.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                          f"{v:.0f} W", ha="center", va="bottom", fontsize=10)
        ax_c.set_xticks(x_regime)
        ax_c.set_xticklabels([f"{r}\n({regime_hit_rate[r]})" for r in regime_order])
        ax_c.set_ylabel("mean dyn power  (W)")
        ax_c.set_title(f"{title_prefix}C. Steady-state dyn power per regime")
        ax_c.grid(True, axis="y", alpha=0.3)
        if y_power and not all(np.isnan(v) for v in y_power):
            ymin, ymax = ax_c.get_ylim()
            ax_c.set_ylim(ymin, ymax * 1.2)

    ew_palette = {"mul": "#1f77b4", "add": "#2ca02c", "softmax": "#d62728",
                  "gelu": "#9467bd", "layernorm": "#ff7f0e"}
    mm_palette = {"matmul_fp32_simt": "#555555", "matmul_tf32_tc": "#ff7f0e",
                  "matmul_fp16_tc":   "#1f77b4", "matmul_bf16_tc": "#2ca02c",
                  "matmul_fp8_te":    "#d62728"}

    ew_sum = (by_regime[by_regime["category"] == "elementwise"]
              if by_regime is not None and not by_regime.empty else pd.DataFrame())
    mm_sum = (by_regime[by_regime["category"] == "matmul"]
              if by_regime is not None and not by_regime.empty else pd.DataFrame())

    r = 0
    if not ew.empty:
        _draw_row(axes[r], ew, ew_sum, "elementwise",
                  keys_key="op", key_palette=ew_palette,
                  unit_col="j_per_element_dyn", unit_label="J / element",
                  annot_scale=1e12, annot_suffix="pJ/elem",
                  title_prefix="[Elementwise]  ")
        r += 1
    if not mm.empty:
        _draw_row(axes[r], mm, mm_sum, "matmul",
                  keys_key="variant", key_palette=mm_palette,
                  unit_col="j_per_flop_dyn", unit_label="J / FLOP",
                  annot_scale=1e12, annot_suffix="pJ/FLOP",
                  title_prefix="[Matmul]  ")

    fig.suptitle(f"Cache-regime breakdown — {gpu}  "
                 "(B. columns show the k_op value the energy model uses)",
                 y=1.00)
    fig.tight_layout()
    # Belt-and-braces: even with the log-scale / positive-value guards above,
    # clamp the final figure size so a single pathological artist can never
    # balloon the PNG into a gigapixel decompression-bomb. 24 inches tall ≈
    # 3840 px at 160 dpi, which is still generous for two stacked rows.
    w_in, h_in = fig.get_size_inches()
    if w_in > 40 or h_in > 24:
        fig.set_size_inches(min(w_in, 40), min(h_in, 24))
    # Fixed pad_inches instead of bbox_inches="tight" — the "tight" mode
    # expands the crop to fit any overflowing artist, which is what let the
    # figure grow to ~2500 in when a log-scale limit went infinite.
    fig.savefig(out_png, dpi=160, pad_inches=0.3)
    print(f"[save] {out_png}")


def plot_llm_matmul(df: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """LLM-shape matmul energy per preset as a function of token count T (= M).

    Two panels:
      A : J/FLOP (dyn) vs T, one line per preset. Horizontal-ish = BW
          cost dominates over compute cost (skinny GEMM territory);
          upward slope = compute cost dominates.
      B : dynamic energy per single forward pass of that layer at each
          T, on log-log axes. The slope of each line is the per-flop
          coefficient; the y-intercept is the fixed launch overhead.
    """
    llm = df[df["category"] == "matmul_llm"].copy()
    if llm.empty:
        return
    plt = _get_mpl()
    presets = sorted(llm["llm_preset"].unique())
    # Fixed palette so layers in the same role always get the same colour
    # across runs / GPUs.
    palette = {
        "qkv":     "#1f77b4",   "q_only": "#17becf",
        "kv":      "#aec7e8",   "attn_o": "#2ca02c",
        "router":  "#9467bd",   "mlp1":   "#ff7f0e",
        "mlp2":    "#d62728",   "lm_head": "#555555",
    }
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))
    for preset in presets:
        g = llm[llm["llm_preset"] == preset].sort_values("load_value")
        if g.empty:
            continue
        T = g["load_value"].to_numpy(float)
        jpf = pd.to_numeric(g["j_per_flop_dyn"], errors="coerce").to_numpy(float)
        dyn = pd.to_numeric(g["dyn_energy_j"], errors="coerce").to_numpy(float)
        iters = pd.to_numeric(g.get("iters", pd.Series(dtype=float)),
                              errors="coerce").to_numpy(float)
        # Energy PER forward pass of the layer (not per-iter batch) = dyn / iters.
        with np.errstate(divide="ignore", invalid="ignore"):
            e_per_forward = np.where(iters > 0, dyn / iters, dyn)
        colour = palette.get(preset, None)
        # Only show points with positive ys for log axes.
        mJ = jpf > 0
        mE = e_per_forward > 0
        if mJ.any():
            ax_a.plot(T[mJ], jpf[mJ], marker="o", color=colour, label=preset)
            for t, y in zip(T[mJ], jpf[mJ]):
                ax_a.annotate(f"T={int(t)}", (t, y),
                              textcoords="offset points", xytext=(5, 5),
                              fontsize=7, color=colour, alpha=0.85)
        if mE.any():
            ax_b.plot(T[mE], e_per_forward[mE], marker="o", color=colour,
                      label=preset)
            for t, y in zip(T[mE], e_per_forward[mE]):
                ax_b.annotate(f"T={int(t)}\n{y*1e3:.2f} mJ", (t, y),
                              textcoords="offset points", xytext=(5, 5),
                              fontsize=6.5, color=colour, alpha=0.85)
    for ax in (ax_a, ax_b):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("token count T  (= M dim)")
        ax.legend(fontsize=8, ncol=2, loc="best")
    ax_a.set_ylabel("J / FLOP  (dynamic)")
    ax_a.set_title("A. Per-FLOP energy vs token count — flat = BW-bound, "
                   "rising = compute-bound")
    ax_b.set_ylabel("J per forward pass of this layer")
    ax_b.set_title("B. Per-call energy vs T  (annotated in mJ)")
    # Pick headroom manually to leave space for the K/N legend entries.
    for ax in (ax_a, ax_b):
        lo, hi = ax.get_ylim()
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            ax.set_ylim(lo, hi * 4)

    # Title records which preset shapes were in play so the PNG is
    # self-documenting (important for multi-model comparisons).
    shape_lines = []
    for preset in presets:
        g = llm[llm["llm_preset"] == preset]
        if g.empty:
            continue
        shape = str(g["shape"].iloc[0])
        shape_lines.append(f"{preset}:{shape}")
    fig.suptitle(f"LLM-shape matmul — {gpu}   "
                 f"({', '.join(shape_lines)})", y=1.00, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=160, pad_inches=0.3)
    print(f"[save] {out_png}")


def plot_timeline(samples_csv: Path, out_png: Path, gpu: str) -> None:
    plt = _get_mpl()
    s = pd.read_csv(samples_csv)
    fig, (ax_p, ax_t, ax_c) = plt.subplots(3, 1, sharex=True, figsize=(16, 10))
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
    fig.tight_layout(); fig.savefig(out_png, dpi=160)
    print(f"[save] {out_png}")


# ---------------------------------------------------------------------------

def _resolve_csv(args) -> Path | None:
    """Find the benchmark CSV from whichever flags the user provided.

    Two input forms are supported:
      (1) positional path:  analyze.py reports/foo.csv
      (2) dir + tag:        analyze.py --reports-dir reports/ --tag h100
    Form (2) globs `gpu_power_bench_*{tag}*.csv` under the directory and
    picks the most recent match (falling back to plain `gpu_power_bench_*.csv`
    when no tag is given).
    """
    if args.csv is not None:
        return args.csv
    if args.reports_dir is None:
        return None
    d = args.reports_dir
    if not d.is_dir():
        print(f"error: --reports-dir {d} does not exist or is not a directory")
        return None
    # Prefer tag-suffixed files (those gpu_power_bench.py writes when --tag
    # is used); otherwise match any per-cell CSV. We deliberately exclude
    # sidecar CSVs (_baseline / _samples / _summary) so the user doesn't
    # accidentally analyze the wrong file.
    patterns = []
    if args.tag:
        patterns += [f"gpu_power_bench_*_{args.tag}.csv",
                     f"gpu_power_bench_*{args.tag}*.csv"]
    else:
        patterns += ["gpu_power_bench_*.csv"]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pat in patterns:
        for p in d.glob(pat):
            if p in seen:
                continue
            name = p.name
            if any(name.endswith(s) for s in ("_baseline.csv", "_baseline_stats.csv",
                                              "_samples.csv", "_summary.csv")):
                continue
            seen.add(p)
            candidates.append(p)
    if not candidates:
        print(f"error: no matching CSV in {d} "
              f"(tag={args.tag!r}, pattern='gpu_power_bench_*.csv')")
        return None
    # Most recently modified file wins.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        print(f"[info] {len(candidates)} CSVs matched — using the most recent:")
        for p in candidates[:5]:
            print(f"         {p.name}")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyse a gpu_power_bench CSV into plots + a summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Positional CSV (form 1) is optional so --reports-dir (form 2) works too.
    ap.add_argument("csv", type=Path, nargs="?", default=None,
                    help="benchmark CSV (per-cell rows); omit to use --reports-dir")
    ap.add_argument("--reports-dir", type=Path, default=None,
                    help="directory to search for gpu_power_bench_*.csv "
                         "(used when the positional CSV is omitted)")
    ap.add_argument("--tag", type=str, default=None,
                    help="only consider CSVs whose filename contains this tag "
                         "(matches gpu_power_bench_*_<tag>.csv)")
    ap.add_argument("--samples", type=Path, default=None,
                    help="raw NVML samples CSV (for timeline plot)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="idle / static-power baseline CSV "
                         "(auto-discovered as <stem>_baseline.csv)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write plots (default: same dir as csv)")
    ap.add_argument("--include-emulated", action="store_true",
                    help="also show emulated ELEMENTWISE cells (fp8 "
                         "cast-compute-cast). Emulated matmul (fp8_te fallback "
                         "on pre-Hopper) is ALWAYS shown regardless of this "
                         "flag, because its number — which should coincide "
                         "with matmul_fp16_tc — is a useful sanity check.")
    args = ap.parse_args()

    csv_path = _resolve_csv(args)
    if csv_path is None:
        ap.print_usage()
        print("error: give a CSV path positionally or use --reports-dir [--tag]")
        return 2
    args.csv = csv_path
    print(f"[input] {csv_path}")

    df = pd.read_csv(args.csv)
    if df.empty:
        print("empty CSV"); return 1
    # Output directory resolution:
    #   1. --out-dir explicitly given  → use it
    #   2. --reports-dir + --tag given → <reports-dir>/<tag>/  (so A100 and
    #                                     H100 plots don't clobber each other)
    #   3. otherwise                   → same directory as the CSV
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.reports_dir is not None and args.tag:
        out_dir = args.reports_dir / args.tag
    else:
        out_dir = args.csv.parent
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"[output] {out_dir}/")
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

    # --- per-regime summary (regime-specific k_op) --------------------------
    # Slope is regressed independently within each cache_regime, so the
    # J/element coefficient doesn't mix L2-bound and DRAM-bound points.
    # This is what the power model should consume when the caller knows
    # the target workload's working-set size.
    by_regime = summarize_by_regime(df)
    if not by_regime.empty:
        by_regime_path = out_dir / f"{stem}_summary_by_regime.csv"
        by_regime.to_csv(by_regime_path, index=False)
        print(f"[save] {by_regime_path}")
        print("\nk_op per cache regime (J/element for elementwise, "
              "J/FLOP for matmul):")
        with pd.option_context("display.width", 200,
                               "display.max_columns", 20,
                               "display.float_format", lambda v: f"{v:.3e}"):
            cols_r = ["variant", "compute_unit", "cache_regime",
                      "n_points", "slope_dyn", "R2_dyn",
                      "median_j_per_unit", "mean_dyn_power_w"]
            # Sort so mul/add/… stay grouped and regimes cycle in order.
            regime_rank = {"l2_hit_100": 0, "l2_hit_75": 1, "l2_hit_50": 2,
                           "l2_hit_25": 3, "l2_hit_0":  4,
                           # legacy fallback
                           "l2_resident": 0, "l2_partial": 2, "dram_stream": 4}
            show = by_regime.assign(
                _rk=by_regime["cache_regime"].map(regime_rank).fillna(99)
            ).sort_values(["variant", "_rk"]).drop(columns="_rk")
            print(show[cols_r].to_string(index=False))

    # --- filter emulated ELEMENTWISE rows out of the plots -----------------
    # We hide only emulated elementwise cells by default, because those are
    # pure cast-compute-cast noise (PyTorch has no native FP8 elementwise
    # kernel — see benchmarks.py). Emulated matmul rows (matmul_fp8_te on
    # A100, where TE falls back to FP16 TC) stay visible: they're informative
    # — the line should land on top of matmul_fp16_tc, which is the whole
    # point of measuring the fallback. The plots already tag those bars as
    # "[TC·FP16-fallback] *EMU" so readers can tell them apart.
    plot_df = df
    plot_summary = summary
    if not args.include_emulated:
        def _keep_row(row) -> bool:
            emu = int(row.get("emulated", 0) or 0)
            if not emu:
                return True
            # Keep emulated matmul rows (A100 fp8_te fallback is useful info).
            return row.get("category", "") == "matmul"
        if "emulated" in df.columns:
            mask = df.apply(_keep_row, axis=1)
            plot_df = df[mask].copy()
        if "emulated" in summary.columns:
            mask = summary.apply(_keep_row, axis=1)
            plot_summary = summary[mask].copy()
        hidden_cells = len(df) - len(plot_df)
        hidden_variants = len(summary) - len(plot_summary)
        if hidden_cells or hidden_variants:
            print(f"[filter] hiding {hidden_variants} emulated elementwise "
                  f"variants ({hidden_cells} rows) from plots — emulated "
                  f"matmul stays visible. Pass --include-emulated to show "
                  f"elementwise fp8 too. Full data: {summary_path.name}.")

    # --- plots ---
    # File-naming convention: every plot file is prefixed with a 2-digit
    # group number so `ls` puts them in the order a reader should consume.
    #   01_powermodel_*   → linearity + coefficient bar
    #   02_cache_*        → cache-regime breakdown
    #   03_baseline_*     → static power diagnostics
    #   04_thermal_*      → thermal diagnostics
    #   05_trace_*        → raw NVML timeline
    plot_linearity_elementwise(plot_df,
        out_dir / f"{stem}_01_powermodel_linearity_elementwise.png", gpu)
    plot_linearity_matmul(plot_df,
        out_dir / f"{stem}_01_powermodel_linearity_matmul.png", gpu)
    plot_joule_per_op_bar(plot_summary,
        out_dir / f"{stem}_01_powermodel_coefficient_bar.png", gpu)
    # Filter the by-regime summary in the same way (elementwise fp8 hidden
    # by default) so annotations on the plot stay consistent with the
    # other plots. Matmul rows are kept either way.
    plot_by_regime = by_regime
    if not args.include_emulated and not by_regime.empty:
        plot_by_regime = by_regime[
            (by_regime["emulated"].astype(int) == 0)
            | (by_regime["category"] == "matmul")
        ].copy()
    plot_cache_regime(plot_df, plot_by_regime,
                      out_dir / f"{stem}_02_cache_regime.png", gpu)
    # LLM-shape matmul plot (only generates if matmul_llm rows present).
    plot_llm_matmul(plot_df,
                    out_dir / f"{stem}_01_powermodel_llm_matmul.png", gpu)

    # Static-power diagnostics (auto-discover the baseline sidecar).
    if args.baseline is None:
        cand = args.csv.with_name(stem + "_baseline.csv")
        if cand.exists():
            args.baseline = cand
    plot_static_power(plot_df, args.baseline,
                      out_dir / f"{stem}_03_baseline_static_power.png", gpu)
    plot_temperature(plot_df, plot_summary,
                     out_dir / f"{stem}_04_thermal_diagnostics.png", gpu)

    # Timeline (auto-discover samples file if not given).
    if args.samples is None:
        cand = args.csv.with_name(stem + "_samples.csv")
        if cand.exists():
            args.samples = cand
    if args.samples and args.samples.exists():
        plot_timeline(args.samples, out_dir / f"{stem}_05_trace_timeline.png", gpu)

    print("\nHow to read the summary CSV:")
    print("  slope_dyn  — Joules per element (elementwise) / per FLOP (matmul).")
    print("               This is the power-modeling coefficient for this op+GPU.")
    print("  R2_dyn     — linearity of E_dyn ~ load.  ≥0.99 = model assumption holds.")
    print("  Lower R² → restrict your fit to loads in the linear regime, then re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
