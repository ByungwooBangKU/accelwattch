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


# ===========================================================================
# Section 1 — Module-level constants
# ===========================================================================
# Everything that several functions need to share lives here so adding a
# new op / regime / dtype / colour only touches one place.
# ---------------------------------------------------------------------------

# 5-bucket cache-locality vocabulary. classify_cache_regime() in
# benchmarks.py emits these labels; analyse uses them to slice and order
# results. Older CSVs (pre-PR #17) used a 3-bucket vocabulary which we
# fold onto these via LEGACY_REGIME_MAP.
REGIME_ORDER = ("l2_hit_100", "l2_hit_75", "l2_hit_50",
                "l2_hit_25",  "l2_hit_0")
REGIME_HIT_PCT = {"l2_hit_100": "~100%", "l2_hit_75": "~75%",
                  "l2_hit_50": "~50%",  "l2_hit_25": "~25%",
                  "l2_hit_0":  "~0%"}
LEGACY_REGIME_MAP = {"l2_resident": "l2_hit_100",
                     "l2_partial":  "l2_hit_50",
                     "dram_stream": "l2_hit_0"}

# Op/variant colour palettes. We keep elementwise + matmul + LLM in
# separate dicts so unrelated benchmarks can't accidentally share a
# colour. Any keys NOT in the palette get tab20 colours assigned in
# whatever order they appear in the data.
PALETTE_ELEMENTWISE_OPS = {
    "mul":       "#1f77b4",
    "add":       "#2ca02c",
    "softmax":   "#d62728",
    "gelu":      "#9467bd",
    "layernorm": "#ff7f0e",
}
PALETTE_MATMUL_VARIANTS = {
    "matmul_fp32_simt": "#555555",
    "matmul_tf32_tc":   "#ff7f0e",
    "matmul_fp16_tc":   "#1f77b4",
    "matmul_bf16_tc":   "#2ca02c",
    "matmul_fp8_te":    "#d62728",
}
PALETTE_LLM_PRESETS = {
    "qkv":     "#1f77b4",   "q_only": "#17becf",
    "kv":      "#aec7e8",   "attn_o": "#2ca02c",
    "router":  "#9467bd",   "mlp1":   "#ff7f0e",
    "mlp2":    "#d62728",   "lm_head": "#555555",
}

# Bytes per element for the dtypes we can encounter in a CSV. Mirrors
# benchmarks._dtype_bytes() but kept here because analyze.py mustn't
# import torch (we want analyze to run on a laptop).
DTYPE_BYTES = {"fp16": 2, "fp8": 2, "bf16": 2, "fp32": 4, "tf32": 4}

# How many tensor reads + writes happen per kernel call, used by
# add_traffic_metrics() to derive bytes_traffic from N + iters.
RW_PER_CALL = {
    "mul": 3, "add": 3,
    "gelu": 2, "softmax": 2, "layernorm": 2,
    "stream_copy": 2, "stream_scale": 2, "stream_triad": 3,
    "stream_read":  1,    # bytes_traffic = N·bpe (read only)
    "stream_write": 1,    # bytes_traffic = N·bpe (write only)
}

# Single-direction stream probes — used by compute_dram_rw_split() to
# isolate read vs write energy. The "ratio" is the (reads, writes) per call.
STREAM_RW_RATIO: dict[str, tuple[int, int]] = {
    "stream_read":  (1, 0),   # pure read
    "stream_write": (0, 1),   # pure write
    "stream_copy":  (1, 1),   # 50/50 — cross-check
    "stream_scale": (1, 1),   # 50/50 — cross-check
    "stream_triad": (2, 1),   # 67/33 — cross-check
}

# Literature reference points for DRAM pJ/bit, drawn as horizontal guide
# lines on the dram-energy plot. Numbers are "full stack" (DRAM cells +
# PHY + controller); see README §3.5 for sources.
DRAM_REFERENCES_PJBIT = {
    "HBM2 (V100)":     7.0,
    "HBM2E (A100)":    5.0,
    "HBM3 (H100)":     3.9,
    "DDR4":            7.0,
    "Horowitz '14 (DRAM core)": 2.5,
}


# ===========================================================================
# Section 2 — Regression helpers
# ===========================================================================
# `linear_fit` is plain OLS, kept for back-compat. `linear_fit_wls` is
# the one we recommend for headline k_op (each point's *relative* error
# weighted equally — matches the log-log linearity plot view).
# `bootstrap_slope_ci` returns a 95% percentile CI on top of either fit.
# ---------------------------------------------------------------------------

def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R²).  Slope is the primary power-model coeff.

    OLS — minimises Σ(y − a·x − b)². Equal weight on every point, which
    over-weights large-N samples whose absolute variance is also larger
    (energy variance ≈ proportional to mean energy). For the heteroscedasticity-
    aware version that's appropriate when reporting `k_op` headlines, see
    `linear_fit_wls()` below.
    """
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def linear_fit_wls(x: np.ndarray, y: np.ndarray,
                   weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Weighted least squares.  Returns (slope, intercept, R²).

    Default weights = 1 / max(y, ε)² — this matches the empirical
    observation that for our power measurements σ(y) is approximately
    proportional to y itself (constant *relative* error). Each point
    therefore contributes equally in log-space, which is exactly what
    the analyst eyes when looking at the log-log linearity plots.

    With explicit weights, this is the same fit a `numpy.polyfit(x, y, 1, w)`
    would produce — but we also return the weighted R², which is what
    the headline `R²` in the summary CSV should reflect.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    if weights is None:
        eps = max(1e-12, float(np.nanmax(np.abs(y))) * 1e-6)
        weights = 1.0 / np.maximum(np.abs(y), eps) ** 2
    w = np.asarray(weights, dtype=float)
    # numpy.polyfit interprets `w` as 1/sigma weights (sqrt(weights) under the
    # hood), so to apply a true 1/var weighting we pass sqrt(weights).
    a, b = np.polyfit(x, y, 1, w=np.sqrt(w))
    y_pred = a * x + b
    sw = np.sum(w)
    if sw <= 0:
        return float(a), float(b), float("nan")
    y_mean_w = np.sum(w * y) / sw
    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - y_mean_w) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def bootstrap_slope_ci(x: np.ndarray, y: np.ndarray,
                       n_resample: int = 1000,
                       confidence: float = 0.95,
                       weighted: bool = True,
                       rng_seed: int = 0) -> tuple[float, float, float]:
    """Percentile-bootstrap confidence interval for the regression slope.

    Resamples (x, y) PAIRS with replacement, refits the slope each time,
    and returns (slope_med, ci_lo, ci_hi). With <2 unique x-values returns
    NaNs. With weighted=True uses linear_fit_wls (matches the headline
    fit method); set False for OLS.

    For our N=9-11 sweep points, percentile bootstrap with 1000 resamples
    is sensible — the resampling distribution is well-resolved at the 2.5%
    and 97.5% quantiles, and the cost is negligible (well under 100 ms
    per variant).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or len(np.unique(x)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    fit = linear_fit_wls if weighted else linear_fit
    slopes = np.empty(n_resample, dtype=float)
    for i in range(n_resample):
        idx = rng.integers(0, n, size=n)
        # Avoid degenerate resamples that picked all-same x.
        if len(np.unique(x[idx])) < 2:
            slopes[i] = np.nan
            continue
        s, _, _ = fit(x[idx], y[idx])
        slopes[i] = s
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(slopes, alpha))
    hi = float(np.quantile(slopes, 1.0 - alpha))
    med = float(np.median(slopes))
    return med, lo, hi


# ===========================================================================
# Section 3 — DataFrame normalisation + traffic-metric derivations
# ===========================================================================
# `add_traffic_metrics(df)` derives bytes_traffic / pj_per_bit_traffic /
# achieved_bw_gbps for the elementwise rows. `_normalize_for_summary(df)`
# fills the column defaults that summarize() / summarize_by_regime() /
# the plot functions all need (cache_regime back-compat, llm_preset
# fillna so groupby doesn't drop rows, etc.). Both helpers always
# return a *copy* so callers don't have to remember.
# ---------------------------------------------------------------------------

def add_traffic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Augment a per-cell df with three derived columns:

      bytes_traffic       — working-set bytes touched per call × iters.
                            For l2_hit_0 cells this is the DRAM traffic;
                            for l2_hit_100 cells it overstates DRAM and
                            instead reflects L2 traffic. The interpretation
                            depends on the cache_regime, so analyze always
                            slices by regime before reporting.
      pj_per_bit_traffic  — dyn_energy_J × 1e12 / (bytes_traffic × 8).
                            At l2_hit_0 this maps to "pJ to move one bit
                            through HBM + PHY + L2 path"; literature
                            HBM2 ≈ 7, HBM2E ≈ 5, HBM3 ≈ 3.9 (full stack).
      achieved_bw_gbps    — bytes_traffic / wall_s / 1e9. Useful for
                            sanity-checking against the GPU's HBM peak;
                            >50% of peak ⇒ truly BW-bound.

    Matmul rows are skipped (their working set has K-fold reuse so the
    naive bytes-per-call ≠ DRAM bytes).
    """
    df = df.copy()
    if "iters" not in df.columns or "n_elements" not in df.columns:
        return df

    def _bytes_per_call(r) -> float:
        if r.get("category") not in (None, "elementwise"):
            return float("nan")
        rw = RW_PER_CALL.get(r.get("op", ""))
        if rw is None:
            return float("nan")
        bpe = DTYPE_BYTES.get(r.get("dtype"), 2)
        try:
            n = int(r.get("n_elements", 0))
        except (TypeError, ValueError):
            return float("nan")
        return float(rw * n * bpe)

    bpc = df.apply(_bytes_per_call, axis=1)
    iters = pd.to_numeric(df.get("iters"), errors="coerce")
    df["bytes_traffic"] = bpc * iters
    dyn = pd.to_numeric(df.get("dyn_energy_j"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pj_per_bit_traffic"] = (dyn * 1e12) / (df["bytes_traffic"] * 8.0)
        wall = pd.to_numeric(df.get("wall_s"), errors="coerce")
        df["achieved_bw_gbps"] = df["bytes_traffic"] / wall / 1e9
    return df


def _normalize_for_summary(df: pd.DataFrame, *,
                           include_cache_regime: bool = False) -> pd.DataFrame:
    """Return a *copy* of df with the columns / values summarize() needs.

    Solves three back-compat / silent-data-loss gotchas in one place so
    summarize() and summarize_by_regime() don't each repeat the boilerplate:

      1. Older CSVs may lack `category` / `mode` / `llm_preset` columns —
         we add safe defaults.
      2. `pandas.read_csv` turns blank cells into NaN, and groupby silently
         DROPS rows whose group keys are NaN (this once silently emptied
         the entire summary on real H100 data — see PR #21). We fillna +
         astype(str) every group-key column.
      3. Pre-PR-#17 CSVs used the 3-bucket cache vocabulary; fold those
         onto the new 5-bucket labels so analysis is unified.

    Pass include_cache_regime=True for summarize_by_regime() / plot
    callers; default False for the basic summarize() that doesn't need
    it.
    """
    df = df.copy()
    group_keys = ["category", "op", "dtype", "mode", "llm_preset"]
    if include_cache_regime:
        group_keys = group_keys + ["cache_regime"]
    for col in group_keys:
        default = ("elementwise" if col in ("category", "mode")
                   else "unknown" if col == "cache_regime" else "")
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)
    if "compute_unit" not in df.columns:
        df["compute_unit"] = df["category"].map(
            lambda c: "Tensor Core" if c in ("matmul", "matmul_llm") else "CUDA core")
    if "emulated" not in df.columns:
        df["emulated"] = 0
    if include_cache_regime:
        df["cache_regime"] = df["cache_regime"].replace(LEGACY_REGIME_MAP)
    return df


def _variant_name(cat: str, op: str, dt: str, mode: str, preset: str) -> str:
    """Stable variant string used as the row identifier in summary CSVs."""
    if cat == "matmul_llm":
        return f"llm_{preset}_{dt}_{mode}"
    if cat == "matmul":
        return f"{op}_{dt}_{mode}"
    return f"{dt}_{op}"


def _fit_one_group(g: pd.DataFrame, x_col: str,
                   *, want_ci: bool = True) -> dict:
    """Run all regressions on one (groupby) cell. Returns a dict that can
    be merged straight into a summary row.

    Computes:
      - OLS slope + R²        (legacy headline; back-compat)
      - WLS slope + R²        (recommended k_op; weights = 1/y²)
      - Bootstrap 95% CI on the WLS slope (1000 resamples)
      - Total-energy OLS slope + R² (no clip, includes static)
      - Unclipped WLS slope + clip_bias_pct from dyn_energy_j_raw
        (PR A; absent on pre-PR-A CSVs → reported as NaN)

    Single-point groups can't fit a slope; we fall back to median y for
    a coarse k_op and emit NaN R² / CI.
    """
    x = g[x_col].to_numpy(dtype=float)
    y_dyn = g["dyn_energy_j"].to_numpy(dtype=float)
    y_tot = g["total_energy_j"].to_numpy(dtype=float)
    n = len(g)
    # Drop clipped-to-zero rows from the DYN regression. The WLS weight
    # is 1/y² (`linear_fit_wls`), so a single y=0 row gets weight ≈ 1/eps²
    # which dominates the fit and drags slope_dyn to ~0. This is the
    # exact symptom on H100 matmul_fp8_te K=1024..2048 — those rows are
    # under the NVML noise floor (README §8.3.4) and got clipped to 0,
    # making the bar plot read 0 J/FLOP for the entire variant. Total-
    # energy regression isn't affected (y_tot never goes to 0; static
    # power × wall_s is always > 0) so we keep its full-row fit.
    pos_mask = (y_dyn > 0) & np.isfinite(y_dyn) & np.isfinite(x)
    n_dropped_clipped = int((~pos_mask).sum())
    x_dyn = x[pos_mask]
    y_dyn_pos = y_dyn[pos_mask]
    n_dyn = len(x_dyn)
    if n_dyn >= 2:
        slope_dyn,    _, r2_dyn     = linear_fit(x_dyn, y_dyn_pos)
        slope_dyn_wls, _, r2_dyn_wls = linear_fit_wls(x_dyn, y_dyn_pos)
        if want_ci:
            _, ci_lo, ci_hi = bootstrap_slope_ci(
                x_dyn, y_dyn_pos, n_resample=1000, confidence=0.95, weighted=True)
        else:
            ci_lo = ci_hi = float("nan")
    elif n_dyn == 1:
        # Single non-clipped point — coarse k_op from that point alone.
        per_point = y_dyn_pos[0] / x_dyn[0] if x_dyn[0] != 0 else float("nan")
        slope_dyn = slope_dyn_wls = per_point
        r2_dyn = r2_dyn_wls = float("nan")
        ci_lo = ci_hi = float("nan")
    else:
        # ALL points clipped → no meaningful dyn k_op. Emit NaN so the
        # bar / heatmap plots draw nothing for this variant rather than
        # showing a misleading 0.
        slope_dyn = slope_dyn_wls = float("nan")
        r2_dyn = r2_dyn_wls = float("nan")
        ci_lo = ci_hi = float("nan")
    # Total-energy fit — uses full row set (no clipping concern there).
    if n >= 2:
        slope_total,  _, r2_total = linear_fit(x, y_tot)
    else:
        slope_total = float("nan")
        r2_total    = float("nan")
    # Clip-bias check (PR A added dyn_energy_j_raw — pre-clip residual).
    slope_dyn_unclipped = float("nan")
    clip_bias_pct = float("nan")
    if "dyn_energy_j_raw" in g.columns:
        y_raw = pd.to_numeric(g["dyn_energy_j_raw"], errors="coerce").to_numpy(float)
        if np.all(np.isfinite(y_raw)) and n >= 2:
            s_raw, _, _ = linear_fit_wls(x, y_raw)
            slope_dyn_unclipped = s_raw
            if np.isfinite(slope_dyn_wls) and slope_dyn_wls != 0:
                clip_bias_pct = (slope_dyn_wls - s_raw) / s_raw * 100.0
    return dict(
        n_points=n,
        # n_points_dyn_fit = how many rows actually fed the dyn regression
        # after clipped-to-zero exclusion. n_dropped_clipped = the rest.
        # When n_points_dyn_fit < n_points, the headline k_op is fitted
        # from a subset of the cells the user thinks they ran — surfaced
        # so a downstream consumer can flag that.
        n_points_dyn_fit=n_dyn,
        n_dropped_clipped=n_dropped_clipped,
        slope_dyn=slope_dyn,
        slope_dyn_wls=slope_dyn_wls,
        slope_dyn_ci_lo=ci_lo,
        slope_dyn_ci_hi=ci_hi,
        slope_dyn_unclipped=slope_dyn_unclipped,
        clip_bias_pct=clip_bias_pct,
        slope_total=slope_total,
        R2_dyn=r2_dyn,
        R2_dyn_wls=r2_dyn_wls,
        R2_total=r2_total,
    )


# ===========================================================================
# Section 4 — Summary builders
# ===========================================================================
# `summarize()` produces one row per (category, op, dtype, mode, llm_preset).
# `summarize_by_regime()` adds cache_regime to the group key so each
# locality bucket gets its own k_op (the regime-specific k_op is what
# the power model should consume when the caller knows the workload's
# working-set size).
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Power-modeling summary — one regression per
    (category, op, dtype, mode, llm_preset).

    For elementwise rows the regression is E_dyn vs total_elements → slope is
    *J per element*.  For matmul rows we regress against total_FLOPs → slope
    is *J per FLOP* (right axis since FLOPs scale as K³ while element count
    scales as K²). Columns `compute_unit` and `emulated` are propagated so
    downstream tables and plots can distinguish "CUDA core" vs "Tensor Core"
    paths and flag emulated cases (fp8 elementwise, fp8_te on pre-Hopper).
    """
    df = _normalize_for_summary(df, include_cache_regime=False)
    group_keys = ["category", "op", "dtype", "mode", "llm_preset"]
    out: list[dict] = []
    for (cat, op, dt, mode, preset), g in df.groupby(group_keys):
        g = g.sort_values("total_elements")
        if cat in ("matmul", "matmul_llm"):
            x_col, unit = "total_flops",    "J/FLOP"
        else:
            x_col, unit = "total_elements", "J/element"
        fit = _fit_one_group(g, x_col, want_ci=True)
        out.append({
            "category": cat, "op": op, "dtype": dt, "mode": mode,
            "llm_preset": preset,
            "variant": _variant_name(cat, op, dt, mode, preset),
            "compute_unit": str(g["compute_unit"].iloc[0]),
            "emulated":     int(bool(g["emulated"].iloc[0])),
            "fit_axis":     unit,
            **fit,
            "mean_dyn_power_w":  g["dyn_power_w"].mean(),
            "mean_avg_power_w":  g["avg_power_w"].mean(),
            "mean_temp_c":       g["avg_temp_c"].mean(),
            "peak_temp_c":       g["peak_temp_c"].max(),
        })
    return pd.DataFrame(out)


def summarize_matmul_per_K(df: pd.DataFrame) -> pd.DataFrame:
    """Per-K k_op rows for every matmul variant — one row per (variant, K).

    Existing `summarize()` produces a single slope across the K range,
    but matmul J/FLOP varies with K (Hopper FP8 sweet spot K ≥ 8192,
    small K shows launch overhead, large K shows sustained Tensor Core
    throughput). This sidecar exposes that K-by-K detail in a
    machine-readable form so downstream tools and notebooks can plot
    or fit the efficiency curve themselves.

    Closes G3 from REVIEW.md §7. Companion plot:
    `_01_powermodel_kop_per_K.png`.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    mm = df[df["category"] == "matmul"].copy()
    if mm.empty:
        return pd.DataFrame()
    keep_cols = [
        "category", "op", "dtype", "mode", "compute_unit", "emulated",
        "load_value",                        # = K for matmul
        "n_elements",                        # = K^2
        "total_flops",                       # = 2*K^3 * iters
        "iters",
        "wall_s",
        "dyn_energy_j", "dyn_energy_j_raw",
        "j_per_flop_dyn",
        "dyn_power_w",
        "cache_regime",
        "peak_temp_c", "avg_temp_c",
    ]
    keep_cols = [c for c in keep_cols if c in mm.columns]
    out = mm[keep_cols].copy()
    out = out.rename(columns={"load_value": "K_size"})
    # Build the variant column (matmul_fp32_simt, etc.) for grouping.
    out["variant"] = ("matmul_" + out["dtype"].astype(str) + "_"
                      + out["mode"].astype(str))
    # Sort for readable output.
    out = out.sort_values(["variant", "K_size"]).reset_index(drop=True)
    return out


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
    df = _normalize_for_summary(df, include_cache_regime=True)
    group_keys = ["category", "op", "dtype", "mode", "llm_preset", "cache_regime"]
    out = []
    for (cat, op, dt, mode, preset, regime), g in df.groupby(group_keys):
        if regime == "unknown":
            continue
        g = g.sort_values("total_elements")
        if cat in ("matmul", "matmul_llm"):
            x_col, unit, y_per_col = "total_flops",    "J/FLOP",    "j_per_flop_dyn"
        else:
            x_col, unit, y_per_col = "total_elements", "J/element", "j_per_element_dyn"
        fit = _fit_one_group(g, x_col, want_ci=True)
        median_j = float(pd.to_numeric(g[y_per_col], errors="coerce").median())
        out.append({
            "category": cat, "op": op, "dtype": dt, "mode": mode,
            "llm_preset": preset,
            "cache_regime": regime,
            "variant": _variant_name(cat, op, dt, mode, preset),
            "compute_unit": str(g["compute_unit"].iloc[0]),
            "emulated":     int(bool(g["emulated"].iloc[0])),
            "fit_axis":     unit,
            "n_points":          fit["n_points"],
            "slope_dyn":         fit["slope_dyn"],
            "slope_dyn_wls":     fit["slope_dyn_wls"],
            "slope_dyn_ci_lo":   fit["slope_dyn_ci_lo"],
            "slope_dyn_ci_hi":   fit["slope_dyn_ci_hi"],
            "R2_dyn":            fit["R2_dyn"],
            "median_j_per_unit": median_j,
            "mean_dyn_power_w":  pd.to_numeric(g["dyn_power_w"], errors="coerce").mean(),
            "mean_temp_c":       pd.to_numeric(g["avg_temp_c"], errors="coerce").mean(),
        })
    return pd.DataFrame(out)


# ===========================================================================
# Section 5 — Plot helpers
# ===========================================================================
# `_get_mpl()` lazy-imports matplotlib (analyse can run on a CSV viewer
# host without it). `_save_fig()` is the canonical save path — fixed
# pad_inches + a 30×18-in figsize cap. `_annot_bar_pj()` writes the
# "0.31 pJ/elem  R²=0.99" two-line annotation on top of any bar.
# ---------------------------------------------------------------------------

def _setup_cjk_fonts(matplotlib_module) -> None:
    """Prepend Korean / CJK-capable fonts to matplotlib's sans-serif fallback
    list so titles / labels with 한글 render instead of emitting "Glyph
    XXXXX missing from font(s) DejaVu Sans" warnings + tofu boxes.

    Tries (in order, first hit wins):
        Noto Sans CJK KR / SC / TC   (most Linux distros via fonts-noto-cjk)
        NanumGothic / NanumBarunGothic (common Korean dev installs)
        Malgun Gothic                  (Windows default Korean)
        AppleGothic                    (macOS default Korean)
        DejaVu Sans                    (final fallback)
    """
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Noto Sans CJK KR", "Noto Sans CJK SC", "Noto Sans CJK TC",
                 "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
                 "AppleGothic", "Source Han Sans KR", "DejaVu Sans"]
    chosen = [f for f in preferred if f in available]
    if not chosen:
        chosen = ["DejaVu Sans"]   # nothing CJK-capable; keep default
    matplotlib_module.rcParams["font.sans-serif"] = (
        chosen + matplotlib_module.rcParams.get("font.sans-serif", []))
    matplotlib_module.rcParams["axes.unicode_minus"] = False  # Korean fonts


def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    _setup_cjk_fonts(matplotlib)
    import matplotlib.pyplot as plt
    return plt


# ===========================================================================
# Section 6 — Plot functions
# ===========================================================================
# Each visible plot is one or more PNGs under `<stem>_<group>_<panel>.png`.
# Groups (lexicographic order = reading order):
#   01_powermodel_*  linearity + coefficient + LLM
#   02_cache_regime_*  per-regime strip / k_op / dyn-power
#   02_dram_energy_*   pJ/bit + sustained BW
#   03_baseline_*      P_static diagnostics
#   04_thermal_*       per-cell temp + cool-down + J vs T
#   05_trace_*         raw NVML timeline (samples sidecar)
# ---------------------------------------------------------------------------

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
    # Layout : 1 row × 3 cols, each panel ~9×9 → roughly square so the
    # log-log slope is read at a 1:1 aspect (slope-of-1 line really looks
    # 45°). Per-panel ~9×9 with title/label margin gives fig (30, 10).
    # User feedback: the old 3-row stacked layout (figsize=(14, 17),
    # i.e. 14×5.7 per panel) made the panels visibly wider than tall,
    # distorting log-log slope perception.
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), squeeze=False)
    ax_e, ax_t, ax_j = axes[0][0], axes[0][1], axes[0][2]
    # Force aspect=1 in DATA units after axis limits are set (call below).
    for ax in (ax_e, ax_t, ax_j):
        ax.set_box_aspect(1)
    ax_e.set_title("matmul — E_dyn vs FLOPs (slope = J/FLOP)")
    ax_t.set_title("matmul — wall time vs FLOPs")
    ax_j.set_title("matmul — J/FLOP (dyn)  [annotated with the swept K]")
    palette = PALETTE_MATMUL_VARIANTS
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


def plot_kop_per_K(per_K_df: pd.DataFrame, out_png: Path, gpu: str) -> bool:
    """Per-K J/FLOP curve for every matmul variant — exposes the Tensor
    Core efficiency ramp that single-slope summaries hide.

    On Hopper, fp8_te has a clear sweet spot at K ≥ 8192 where Tensor
    Cores reach ~67% of theoretical peak. Smaller K is dominated by
    launch overhead + cuBLAS algorithm selection picking smaller tiles.
    Single slope across the K range averages the two regimes and hides
    where the variant actually performs best.

    Closes G3 (P1.2) from REVIEW.md §7. Companion CSV:
    `_summary_matmul_per_K.csv`.

    x : K (log scale)
    y : pJ/FLOP (log scale — order-of-magnitude differences across
        variants need log)
    one line per variant
    """
    if per_K_df is None or per_K_df.empty:
        return False
    plt = _get_mpl()
    df = per_K_df.copy()
    df["pj_per_flop"] = pd.to_numeric(df["j_per_flop_dyn"],
                                      errors="coerce") * 1e12
    df["K_size"] = pd.to_numeric(df["K_size"], errors="coerce")
    df = df[(df["pj_per_flop"] > 0) & df["K_size"].notna()]
    if df.empty:
        return False

    fig, ax = plt.subplots(figsize=(13, 7.5))
    palette = PALETTE_MATMUL_VARIANTS
    variants = sorted(df["variant"].unique(),
                      key=lambda v: ["matmul_fp32_simt", "matmul_tf32_tc",
                                     "matmul_fp16_tc", "matmul_bf16_tc",
                                     "matmul_fp8_te"].index(v)
                                    if v in ["matmul_fp32_simt", "matmul_tf32_tc",
                                             "matmul_fp16_tc", "matmul_bf16_tc",
                                             "matmul_fp8_te"] else 99)
    for v in variants:
        g = df[df["variant"] == v].sort_values("K_size")
        if g.empty:
            continue
        emu = bool(int(g["emulated"].iloc[0])) if "emulated" in g else False
        c = palette.get(v, "gray")
        line_label = v + (" *EMU" if emu else "")
        # marker only — linestyle is set explicitly below to avoid the
        # "redundantly defined" matplotlib warning that "-o" + linestyle
        # kwarg triggered in the previous code path.
        ax.plot(g["K_size"], g["pj_per_flop"], color=c, label=line_label,
                marker="o", markersize=7,
                linewidth=2, alpha=0.85,
                linestyle="--" if emu else "-")
        # Annotate the K with the lowest pJ/FLOP — the variant's "sweet spot"
        if len(g) >= 3:
            row_min = g.loc[g["pj_per_flop"].idxmin()]
            ax.annotate(f"K={int(row_min['K_size'])}\n{row_min['pj_per_flop']:.2f} pJ/FLOP",
                        xy=(row_min["K_size"], row_min["pj_per_flop"]),
                        xytext=(0, -35), textcoords="offset points",
                        ha="center", fontsize=8, color=c,
                        arrowprops=dict(arrowstyle="-", color=c, alpha=0.6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K   (square matmul, M = N = K)", fontsize=11)
    ax.set_ylabel("pJ / FLOP (dynamic)", fontsize=11)
    ax.set_title(
        f"matmul k_op (J/FLOP) — per-K curve — {gpu}\n"
        "Tensor Core efficiency varies with K — small K shows launch / "
        "tile-selection overhead, large K shows sustained throughput. "
        "Single-slope summary averages the two regimes.",
        fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def _annot_bar_pj(rect, value_j, r2, scale_label):
    """Write '0.31 pJ/elem\\nR²=0.99' on top of a bar, skipping NaNs."""
    if value_j is None or np.isnan(value_j):
        return
    v_p = value_j * 1e12   # J → pJ
    if abs(v_p) >= 100:
        vtxt = f"{v_p:.0f} {scale_label}"
    elif abs(v_p) >= 1:
        vtxt = f"{v_p:.2f} {scale_label}"
    elif abs(v_p) >= 0.01:
        vtxt = f"{v_p:.3f} {scale_label}"
    else:
        vtxt = f"{v_p:.2e} {scale_label}"
    label = vtxt if np.isnan(r2) else f"{vtxt}\nR²={r2:.3f}"
    ax = rect.axes
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
            label, ha="center", va="bottom", fontsize=9, linespacing=1.1)


def _save_fig(fig, out_png: Path, dpi: int = 160) -> None:
    """Save with a fixed pad and a hard size cap — defends against the
    gigapixel-PNG bug whenever log axes end up with pathological bounds.

    `tight_layout()` is wrapped + silenced because some plots (MECE
    decompositions, anything with a `fig.text()` caveat box ABOVE or
    BELOW the axes) intentionally place text outside the axes; tight
    layout can't reconcile that and prints "Tight layout not applied"
    warnings even though the saved figure is fine. We rely on
    `bbox_inches="tight"` to actually clip whitespace.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message=r"Tight layout not applied")
        try:
            fig.tight_layout()
        except Exception:
            pass
    w_in, h_in = fig.get_size_inches()
    if w_in > 30 or h_in > 18:
        fig.set_size_inches(min(w_in, 30), min(h_in, 18))
    fig.savefig(out_png, dpi=dpi, pad_inches=0.3, bbox_inches="tight")
    print(f"[save] {out_png}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _coef_bar_elementwise(ew, out_png: Path, gpu: str) -> bool:
    """Standalone elementwise k_op bar chart — full-width, single panel.
    Returns True iff a file was written."""
    if ew.empty:
        return False
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(14, 7))
    ops = sorted(ew["op"].unique())
    dtypes = sorted(ew["dtype"].unique(), reverse=True)
    xpos = np.arange(len(ops))
    w = 0.8 / max(1, len(dtypes))
    colors = {"fp16": "#1f77b4", "fp8": "#d62728"}
    has_emulated = False
    slope_col = "slope_dyn_wls" if "slope_dyn_wls" in ew.columns else "slope_dyn"
    r2_col    = "R2_dyn_wls"    if "R2_dyn_wls"    in ew.columns else "R2_dyn"
    all_positive: list[float] = []
    for i, dt in enumerate(dtypes):
        vals, r2s, errs_lo, errs_hi = [], [], [], []
        for op in ops:
            row = ew[(ew.op == op) & (ew.dtype == dt)]
            if row.empty:
                vals.append(float("nan")); r2s.append(float("nan"))
                errs_lo.append(0.0); errs_hi.append(0.0)
            else:
                v = float(row[slope_col].iloc[0])
                vals.append(v); r2s.append(float(row[r2_col].iloc[0]))
                if ("slope_dyn_ci_lo" in row.columns
                        and pd.notna(row["slope_dyn_ci_lo"].iloc[0])):
                    ci_lo = float(row["slope_dyn_ci_lo"].iloc[0])
                    ci_hi = float(row["slope_dyn_ci_hi"].iloc[0])
                    errs_lo.append(max(0.0, v - ci_lo))
                    errs_hi.append(max(0.0, ci_hi - v))
                else:
                    errs_lo.append(0.0); errs_hi.append(0.0)
                if "emulated" in row.columns and int(row["emulated"].iloc[0]):
                    has_emulated = True
                if np.isfinite(v) and v > 0:
                    all_positive.append(v)
        emu_dt = (dt == "fp8")
        label = f"{dt} [CUDA]" + (" *EMU" if emu_dt else "")
        yerr = [errs_lo, errs_hi] if any(errs_lo) or any(errs_hi) else None
        bars = ax.bar(xpos + (i - (len(dtypes) - 1) / 2) * w, vals, w,
                      yerr=yerr, capsize=3,
                      label=label, color=colors.get(dt, None), alpha=0.9,
                      hatch="//" if emu_dt else None, edgecolor="white",
                      error_kw=dict(ecolor="#444444", lw=0.9))
        for rect, v, r2 in zip(bars, vals, r2s):
            _annot_bar_pj(rect, v, r2, "pJ/elem")
    ax.set_xticks(xpos); ax.set_xticklabels(ops, fontsize=11)
    ax.set_ylabel("J / element (dynamic)  — regression slope")
    if all_positive:
        ax.set_yscale("log")
        # Headroom 1.6× on log scale ≈ 20% of a decade — enough room for
        # the per-bar "X pJ/elem  R²=…" annotation without leaving most
        # of the chart blank above the tallest bar (was 4×, complaint
        # was that y-axis went to 262922 pJ when max bar was ~65k).
        ax.set_ylim(min(all_positive) * 0.4, max(all_positive) * 1.6)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    # Detect whether the source is the l2_hit_0 regime (preferred path)
    # or the cross-regime summary (fallback when by_regime not present).
    regime_label = ""
    if "cache_regime" in ew.columns:
        rs = set(str(x) for x in ew["cache_regime"].dropna().unique())
        if rs == {"l2_hit_0"}:
            regime_label = " @ l2_hit_0 (DRAM streaming — production memory-bound region)"
        elif len(rs) == 1:
            regime_label = f" @ {next(iter(rs))}"
    title = ("Elementwise — per-op energy coefficient (all on CUDA cores)"
             + regime_label
             + ". Labels: slope (pJ/elem) + R²")
    if has_emulated:
        title += "\nfp8 bars = cast-compute-cast via FP16 (no native FP8 elementwise in PyTorch)"
    if regime_label:
        title += ("\nFitted within a single cache regime to avoid R² "
                  "collapse from L2→DRAM nonlinearity.")
    ax.set_title(title, fontsize=11)
    fig.suptitle(f"Power-model coefficient — elementwise — {gpu}")
    _save_fig(fig, out_png)
    return True


def _coef_bar_matmul(mm, out_png: Path, gpu: str) -> bool:
    """Standalone matmul k_op bar chart — full-width, single panel."""
    if mm.empty:
        return False
    plt = _get_mpl()
    order = ["matmul_fp32_simt", "matmul_tf32_tc", "matmul_fp16_tc",
             "matmul_bf16_tc", "matmul_fp8_te"]
    mm2 = mm.set_index("variant").reindex([v for v in order if v in mm["variant"].values])
    if mm2.empty:
        return False
    colors_b = [PALETTE_MATMUL_VARIANTS.get(v, "gray") for v in mm2.index]
    def _emu(v):
        if "emulated" in mm2.columns:
            val = mm2.loc[v, "emulated"]
            return bool(int(val)) if pd.notna(val) else False
        return False
    hatches = ["//" if _emu(v) else None for v in mm2.index]
    slope_col = "slope_dyn_wls" if "slope_dyn_wls" in mm2.columns else "slope_dyn"
    r2_col    = "R2_dyn_wls"    if "R2_dyn_wls"    in mm2.columns else "R2_dyn"
    slope_vals = mm2[slope_col].values
    if "slope_dyn_ci_lo" in mm2.columns and "slope_dyn_ci_hi" in mm2.columns:
        errs_lo, errs_hi = [], []
        for v, lo, hi in zip(slope_vals,
                              mm2["slope_dyn_ci_lo"].values,
                              mm2["slope_dyn_ci_hi"].values):
            if pd.notna(v) and pd.notna(lo) and pd.notna(hi):
                errs_lo.append(max(0.0, float(v) - float(lo)))
                errs_hi.append(max(0.0, float(hi) - float(v)))
            else:
                errs_lo.append(0.0); errs_hi.append(0.0)
        yerr = [errs_lo, errs_hi]
    else:
        yerr = None
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(mm2)), slope_vals, yerr=yerr, capsize=3,
                  color=colors_b, alpha=0.9, edgecolor="white",
                  error_kw=dict(ecolor="#444444", lw=0.9))
    for rect, h in zip(bars, hatches):
        if h:
            rect.set_hatch(h)
    for rect, v, r2 in zip(bars, slope_vals, mm2[r2_col].values):
        _annot_bar_pj(rect, float(v) if pd.notna(v) else float("nan"),
                      float(r2) if pd.notna(r2) else float("nan"),
                      "pJ/FLOP")
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
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("J / FLOP (dynamic)  — regression slope")
    positive = [float(v) for v in slope_vals if pd.notna(v) and float(v) > 0]
    if positive:
        ax.set_yscale("log")
        ax.set_ylim(min(positive) * 0.4, max(positive) * 1.6)
    ax.grid(True, axis="y", alpha=0.3)
    title = ("Matmul — per-variant energy coefficient. "
             "Labels: slope (pJ/FLOP) + R²")
    if any(hatches):
        title += "\n* hatched bar = emulated (not native for this dtype)"
    ax.set_title(title, fontsize=11)
    fig.suptitle(f"Power-model coefficient — matmul — {gpu}")
    _save_fig(fig, out_png)
    return True


def _coef_bar_fp8(summary: pd.DataFrame, out_png: Path, gpu: str) -> bool:
    """fp8-only k_op bar chart : matmul_fp8_te + softmax_fp8 + gelu_fp8 +
    layernorm_fp8 on one figure with two panels (matmul J/FLOP on left,
    elementwise J/element on right) so the unit difference is honest.

    Always includes emulated rows — fp8 elementwise is by definition
    emulated (cast-compute-cast), and on pre-Hopper fp8_te falls back
    to FP16 TC. Both are flagged with a hatched bar pattern. The
    point of THIS plot is to surface fp8 numbers that are otherwise
    hidden by the default emulated-row filter on the main coef-bar.
    """
    if summary.empty:
        return False
    fp8 = summary[summary["dtype"] == "fp8"].copy()
    if fp8.empty:
        return False
    fp8_mm = fp8[fp8["category"] == "matmul"]
    fp8_ew_ops = ("softmax", "gelu", "layernorm")
    fp8_ew = fp8[(fp8["category"] == "elementwise")
                 & (fp8["op"].isin(fp8_ew_ops))]
    if fp8_mm.empty and fp8_ew.empty:
        return False

    plt = _get_mpl()
    fig, (ax_mm, ax_ew) = plt.subplots(1, 2, figsize=(20, 8),
                                       gridspec_kw={"width_ratios": [1, 2]})

    # ---- matmul_fp8_te panel (J/FLOP) ----
    slope_col = "slope_dyn_wls" if "slope_dyn_wls" in fp8_mm.columns else "slope_dyn"
    r2_col    = "R2_dyn_wls"    if "R2_dyn_wls"    in fp8_mm.columns else "R2_dyn"
    if not fp8_mm.empty:
        # Always plotted as a single bar (the only fp8 matmul variant).
        for _, row in fp8_mm.iterrows():
            v = float(row[slope_col]) if pd.notna(row[slope_col]) else float("nan")
            r2 = float(row[r2_col]) if pd.notna(row.get(r2_col, np.nan)) else float("nan")
            emu = bool(int(row.get("emulated", 0)))
            label = "matmul_fp8_te" + (" *EMU" if emu else "")
            bar = ax_mm.bar([0], [v],
                            color="#9467bd", alpha=0.9, edgecolor="white",
                            hatch="//" if emu else None)
            for rect in bar:
                _annot_bar_pj(rect, v, r2, "pJ/FLOP")
            ax_mm.set_xticks([0])
            ax_mm.set_xticklabels([label], fontsize=11, rotation=15, ha="right")
        ax_mm.set_ylabel("J / FLOP (dynamic)", fontsize=11)
        ax_mm.set_title("matmul_fp8_te — k_op (pJ/FLOP)", fontsize=11)
        ax_mm.grid(True, axis="y", alpha=0.3)
        if pd.notna(fp8_mm[slope_col].iloc[0]) and float(fp8_mm[slope_col].iloc[0]) > 0:
            ax_mm.set_ylim(0, float(fp8_mm[slope_col].iloc[0]) * 1.4)
    else:
        ax_mm.set_visible(False)

    # ---- fp8 elementwise panel (J/element) ----
    if not fp8_ew.empty:
        op_order = [op for op in fp8_ew_ops if op in fp8_ew["op"].values]
        xs = np.arange(len(op_order))
        vals, r2s, hatches = [], [], []
        for op in op_order:
            row = fp8_ew[fp8_ew["op"] == op]
            if row.empty:
                vals.append(float("nan")); r2s.append(float("nan"))
                hatches.append(None); continue
            v = float(row[slope_col].iloc[0]) if pd.notna(row[slope_col].iloc[0]) else float("nan")
            r2 = float(row[r2_col].iloc[0]) if pd.notna(row.get(r2_col, pd.Series([np.nan])).iloc[0]) else float("nan")
            emu = bool(int(row.get("emulated", pd.Series([0])).iloc[0]))
            vals.append(v); r2s.append(r2)
            hatches.append("//" if emu else None)
        bars = ax_ew.bar(xs, vals, color="#d62728", alpha=0.9, edgecolor="white")
        for rect, h in zip(bars, hatches):
            if h: rect.set_hatch(h)
        for rect, v, r2 in zip(bars, vals, r2s):
            _annot_bar_pj(rect, v, r2, "pJ/elem")
        ax_ew.set_xticks(xs)
        ax_ew.set_xticklabels([f"{op}_fp8" for op in op_order], fontsize=11)
        ax_ew.set_ylabel("J / element (dynamic)", fontsize=11)
        ax_ew.set_title("fp8 elementwise — k_op (pJ/elem)", fontsize=11)
        ax_ew.grid(True, axis="y", alpha=0.3)
        finite_pos = [v for v in vals if pd.notna(v) and v > 0]
        if finite_pos:
            ax_ew.set_ylim(0, max(finite_pos) * 1.3)
    else:
        ax_ew.set_visible(False)

    fig.suptitle(
        f"FP8 power-model coefficients — {gpu}\n"
        "matmul_fp8_te (left, pJ/FLOP) and fp8 elementwise (right, pJ/elem). "
        "Hatched bar = emulated path (cast-compute-cast or A100 FP16 fallback).",
        fontsize=11)
    _save_fig(fig, out_png)
    return True


def _coef_bar_fp8_per_regime(by_regime: pd.DataFrame, out_png: Path, gpu: str) -> bool:
    """fp8 k_op per cache regime — same 4 ops as `_coef_bar_fp8`, but
    grouped along the cache_regime x-axis so the locality dependence is
    visible.

    Two panels :
      - left : matmul_fp8_te k_op vs regime  (J/FLOP — usually flat,
        ~1.3-1.5× from l2_resident to dram_stream)
      - right : fp8 softmax/gelu/layernorm k_op vs regime  (J/element
        — should drop with hit rate; 5..10× span typical for memory-
        light ops)
    """
    if by_regime is None or by_regime.empty:
        return False
    fp8 = by_regime[by_regime["dtype"] == "fp8"].copy()
    if fp8.empty:
        return False
    fp8["cache_regime"] = fp8["cache_regime"].replace(LEGACY_REGIME_MAP)
    fp8 = fp8[fp8["cache_regime"].isin(REGIME_ORDER)]
    if fp8.empty:
        return False

    fp8_mm = fp8[fp8["category"] == "matmul"]
    fp8_ew_ops = ("softmax", "gelu", "layernorm")
    fp8_ew = fp8[(fp8["category"] == "elementwise")
                 & (fp8["op"].isin(fp8_ew_ops))]
    if fp8_mm.empty and fp8_ew.empty:
        return False

    plt = _get_mpl()
    fig, (ax_mm, ax_ew) = plt.subplots(1, 2, figsize=(20, 8),
                                       gridspec_kw={"width_ratios": [1.2, 2]})

    regime_x = {r: i for i, r in enumerate(REGIME_ORDER)}
    xpos = np.arange(len(REGIME_ORDER))
    slope_col = "slope_dyn"
    r2_col    = "R2_dyn"

    def _bars_for(ax, sub_df, color, op_order, scale_label, scale, is_matmul):
        if sub_df.empty:
            ax.set_visible(False); return
        # If multiple ops, use grouped bars; else single bar per regime
        keys = op_order if op_order else ["matmul_fp8_te"]
        width = 0.8 / max(1, len(keys))
        cmap = plt.get_cmap("tab10")
        positive_vals: list[float] = []
        for i, key in enumerate(keys):
            vals, r2s = [], []
            for r in REGIME_ORDER:
                if is_matmul:
                    row = sub_df[sub_df["cache_regime"] == r]
                else:
                    row = sub_df[(sub_df["op"] == key)
                                 & (sub_df["cache_regime"] == r)]
                if row.empty:
                    vals.append(np.nan); r2s.append(np.nan)
                else:
                    v = float(row[slope_col].iloc[0])
                    if not np.isfinite(v) or v <= 0:
                        med_col = "median_j_per_unit"
                        if med_col in row.columns:
                            med = float(row[med_col].iloc[0])
                            v = med if np.isfinite(med) and med > 0 else np.nan
                    vals.append(v)
                    r2s.append(float(row[r2_col].iloc[0]) if r2_col in row.columns else np.nan)
            bar_col = color if is_matmul else cmap(i % 10)
            bars = ax.bar(xpos + (i - (len(keys)-1)/2) * width, vals, width,
                          label=str(key) + ("_fp8" if not is_matmul else ""),
                          color=bar_col, alpha=0.9, edgecolor="white",
                          hatch="//")
            for rect, v, r2 in zip(bars, vals, r2s):
                if not np.isfinite(v) or v <= 0:
                    continue
                positive_vals.append(v)
                v_p = v * scale
                if abs(v_p) >= 1:
                    vtxt = f"{v_p:.2f}"
                elif abs(v_p) >= 0.01:
                    vtxt = f"{v_p:.3f}"
                else:
                    vtxt = f"{v_p:.2e}"
                txt = f"{vtxt} {scale_label}"
                if np.isfinite(r2):
                    txt += f"\nR²={r2:.2f}"
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                        txt, ha="center", va="bottom", fontsize=8,
                        linespacing=1.1)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{r}\n({REGIME_HIT_PCT[r]})"
                            for r in REGIME_ORDER], fontsize=11)
        if positive_vals:
            ax.set_ylim(0, max(positive_vals) * 1.30)
        ax.grid(True, axis="y", alpha=0.3)
        if not is_matmul:
            ax.legend(fontsize=9)

    if not fp8_mm.empty:
        _bars_for(ax_mm, fp8_mm, "#9467bd", [], "pJ/FLOP", 1e12, True)
        ax_mm.set_ylabel("J / FLOP (dynamic)", fontsize=11)
        ax_mm.set_title("matmul_fp8_te — k_op per cache regime", fontsize=11)
    else:
        ax_mm.set_visible(False)
    if not fp8_ew.empty:
        op_order = [op for op in fp8_ew_ops if op in fp8_ew["op"].values]
        _bars_for(ax_ew, fp8_ew, None, op_order, "pJ/elem", 1e12, False)
        ax_ew.set_ylabel("J / element (dynamic)", fontsize=11)
        ax_ew.set_title("fp8 elementwise — k_op per cache regime", fontsize=11)
    else:
        ax_ew.set_visible(False)

    fig.suptitle(
        f"FP8 k_op per cache regime — {gpu}\n"
        "Hatched bars : emulated path (cast-compute-cast on elementwise; "
        "FP16 fallback on pre-Hopper matmul). "
        "Lower hit-rate regimes pay extra DRAM bytes — slope shows the "
        "memory-cost component of fp8 power.",
        fontsize=11)
    _save_fig(fig, out_png)
    return True


def plot_joule_per_op_bar(summary: pd.DataFrame, out_dir: Path, stem: str,
                          gpu: str,
                          by_regime: pd.DataFrame | None = None) -> None:
    """Save the elementwise and matmul k_op bar charts as TWO SEPARATE
    full-width PNGs — one per panel — so the x-axis labels never get
    cramped. Filenames:
        <stem>_01_powermodel_coef_bar_elementwise.png
        <stem>_01_powermodel_coef_bar_matmul.png
        <stem>_01_powermodel_coef_bar_fp8.png         (NEW)

    For elementwise, prefers `by_regime` filtered to `l2_hit_0` if
    supplied — the cross-regime (single) `summary` slope crosses the
    L2→DRAM boundary where J/element jumps ~10×, so the linear fit
    R² collapses (~0.087 for mul/add reported by user). Within a
    single regime the linear assumption holds, R² → ~1.0.
    Headline elementwise k_op now reflects the DRAM-streaming
    (l2_hit_0) regime — the production-relevant memory-bound region.
    Matmul is unaffected (compute-bound, tile reuse keeps the
    cross-K linear assumption).
    """
    # Elementwise — prefer per-regime fit at l2_hit_0 to avoid the
    # cross-regime non-linearity (R² collapse on mul/add). Fall back to
    # the cross-regime summary if by_regime not provided or empty.
    ew_source = summary[summary["category"] == "elementwise"]
    if by_regime is not None and not by_regime.empty:
        br = by_regime.copy()
        if "cache_regime" in br.columns:
            br["cache_regime"] = br["cache_regime"].replace(LEGACY_REGIME_MAP)
            ew_l2_hit_0 = br[(br["category"] == "elementwise")
                             & (br["cache_regime"] == "l2_hit_0")]
            if not ew_l2_hit_0.empty:
                ew_source = ew_l2_hit_0
    mm = summary[summary["category"] == "matmul"]
    _coef_bar_elementwise(ew_source,
        out_dir / f"{stem}_01_powermodel_coef_bar_elementwise.png", gpu)
    _coef_bar_matmul(mm,
        out_dir / f"{stem}_01_powermodel_coef_bar_matmul.png", gpu)
    # fp8 dedicated panel uses the FULL summary (not the emulated-filtered
    # plot_summary) so fp8 elementwise bars are always present, regardless
    # of whether --include-emulated was passed.
    _coef_bar_fp8(summary,
        out_dir / f"{stem}_01_powermodel_coef_bar_fp8.png", gpu)


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
    # Order: elementwise first (by op, dtype), then matmul (by variant, K),
    # then matmul_llm (by preset, dtype, T), then fused (by variant, dtype).
    def _safe_int(v, default=0):
        # load_value is an int for elementwise/matmul/matmul_llm but a
        # shape-encoded string for `fused` (e.g. "B1_Hq64_..."). Anything
        # non-int returns `default` so the sort key stays type-stable.
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _cell_key(r):
        cat = r.get("category", "elementwise")
        if cat == "matmul":
            return (1, r.get("variant", ""), _safe_int(r["load_value"]))
        if cat == "matmul_llm":
            return (2, r.get("llm_preset", ""), r.get("dtype", ""),
                    _safe_int(r["load_value"]))
        if cat == "fused":
            # Fused : 1 cell per (variant, dtype) at a fixed shape, so the
            # exact load_value text is fine as a stable secondary key.
            return (3, r.get("variant", ""), r.get("dtype", ""),
                    str(r.get("load_value", "")))
        return (0, r["op"], r["dtype"], _safe_int(r["load_value"]))

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


def plot_pstatic_drift_vs_temp(rebaseline_csv: Path, df: pd.DataFrame,
                               out_png: Path, gpu: str) -> bool:
    """P_static drift over the sweep, plotted against avg_temp_c at the
    same wall_ts. Distinguishes thermal-driven drift (correlated with
    rising temperature) from random NVML noise (uncorrelated).

    Closes G8 (P2.3) from REVIEW.md §7. Useful when `--rebaseline-every N`
    is on and the user wants to know whether the reported drift is
    "rack warming up" (slope > 0, R² high) or "background process /
    NVML jitter" (no correlation).

    Two side-by-side panels :
      A : P_static(t) trace over the sweep duration — same as the
          existing _baseline_static_power panel B but at full width
          for clarity.
      B : Scatter (avg_temp_c, p_static_w) per rebaseline event +
          linear fit + Pearson r. Shows thermal correlation.

    Inputs:
      rebaseline_csv : path to <stem>_rebaseline.csv (one row per
                        baseline event — initial + each periodic
                        re-baseline). Columns: after_cell, kind,
                        p_static_w, p_static_w_std, duration_s, wall_ts.
      df              : main per-cell CSV; used only to source
                        avg_temp_c per (wall_ts) by nearest-cell match.
    """
    if not rebaseline_csv.exists():
        return False
    try:
        rb = pd.read_csv(rebaseline_csv)
    except Exception:
        return False
    if rb.empty or "p_static_w" not in rb.columns:
        return False

    rb["p_static_w"]   = pd.to_numeric(rb["p_static_w"],   errors="coerce")
    rb["wall_ts"]      = pd.to_numeric(rb.get("wall_ts",   pd.Series([])), errors="coerce")
    rb = rb[rb["p_static_w"].notna()]
    if rb.empty:
        return False

    # Match each rebaseline event to the nearest cell's avg_temp_c.
    # The sweep CSV doesn't have wall_ts directly — we approximate by
    # cell ordering. If df has 'avg_temp_c', take rolling mean of cells
    # near each rebaseline event.
    avg_temps = []
    if "avg_temp_c" in df.columns and not df.empty:
        cell_temps = pd.to_numeric(df["avg_temp_c"], errors="coerce")
        cell_temps = cell_temps.dropna().tolist()
    else:
        cell_temps = []
    if cell_temps and "after_cell" in rb.columns:
        for after_cell in rb["after_cell"].astype(int).tolist():
            if after_cell <= 0:
                avg_temps.append(float("nan"))
            else:
                idx = min(after_cell - 1, len(cell_temps) - 1)
                # Rolling-window context: 3 cells centred at after_cell
                lo = max(0, idx - 1)
                hi = min(len(cell_temps), idx + 2)
                window = cell_temps[lo:hi]
                avg_temps.append(sum(window) / len(window) if window else float("nan"))
    elif "avg_temp_c" in df.columns:
        # Fallback : use mean of all cells
        mean_t = float(pd.to_numeric(df["avg_temp_c"], errors="coerce").mean())
        avg_temps = [mean_t] * len(rb)
    else:
        avg_temps = [float("nan")] * len(rb)
    rb = rb.assign(avg_temp_c=avg_temps)

    plt = _get_mpl()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6),
                                     gridspec_kw={"width_ratios": [1.4, 1]})

    # ---- Panel A : P_static(t) over the sweep ----
    if rb["wall_ts"].notna().any():
        x_a = rb["wall_ts"].to_numpy(dtype=float)
        x_a = x_a - x_a.min()  # zero at sweep start
        ax_a.set_xlabel("wall time since sweep start (s)", fontsize=11)
    else:
        x_a = np.arange(len(rb))
        ax_a.set_xlabel("rebaseline index", fontsize=11)
    y_a = rb["p_static_w"].to_numpy(dtype=float)
    ax_a.plot(x_a, y_a, "-o", color="#1f77b4", lw=1.6, markersize=6, alpha=0.85)
    ax_a.set_ylabel("P_static (W)", fontsize=11)
    ax_a.set_title("P_static drift over the sweep", fontsize=11)
    if len(y_a) > 1:
        net = y_a[-1] - y_a[0]
        rng = y_a.max() - y_a.min()
        ax_a.text(0.02, 0.97,
                  f"net drift: {net:+.2f} W\nrange: {rng:.2f} W\nn = {len(y_a)}",
                  transform=ax_a.transAxes, va="top", fontsize=10,
                  bbox=dict(facecolor="white", alpha=0.85, pad=4))
    ax_a.grid(True, alpha=0.3)

    # ---- Panel B : P_static vs temperature scatter + fit ----
    have_temp = rb["avg_temp_c"].notna().any()
    if have_temp and len(rb) >= 3:
        T = rb["avg_temp_c"].to_numpy(dtype=float)
        P = rb["p_static_w"].to_numpy(dtype=float)
        m = np.isfinite(T) & np.isfinite(P)
        T, P = T[m], P[m]
        if len(T) >= 3 and (T.max() - T.min()) > 0.1:
            # Linear fit
            b, a = np.polyfit(T, P, 1)
            P_fit = a + b * T
            ss_res = np.sum((P - P_fit) ** 2)
            ss_tot = np.sum((P - P.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            # Pearson r (signed)
            if T.std() > 0 and P.std() > 0:
                r = float(np.corrcoef(T, P)[0, 1])
            else:
                r = float("nan")

            ax_b.scatter(T, P, s=44, color="#d62728", alpha=0.8,
                         edgecolors="white", linewidths=0.5)
            T_line = np.linspace(T.min(), T.max(), 50)
            ax_b.plot(T_line, a + b * T_line, "-", color="#444", lw=1.8,
                      label=(f"P = {a:.2f} + {b:.3f}·T\n"
                             f"R² = {r2:.3f}, Pearson r = {r:+.3f}"))
            ax_b.legend(loc="best", fontsize=9)
            # Verdict box
            if abs(r) > 0.7:
                verdict = "→ thermal-driven drift (correlated with T)"
                color = "#d62728"
            elif abs(r) < 0.3:
                verdict = "→ uncorrelated with T (random noise / background)"
                color = "#2ca02c"
            else:
                verdict = "→ partial correlation (mixed thermal + noise)"
                color = "#ff7f0e"
            ax_b.text(0.5, -0.18, verdict, transform=ax_b.transAxes,
                      ha="center", fontsize=10, color=color, fontweight="bold")
        else:
            ax_b.text(0.5, 0.5,
                      "insufficient temperature variation\n"
                      "(need ≥ 3 rebaseline events with T spread > 0.1 °C)",
                      ha="center", va="center", transform=ax_b.transAxes,
                      fontsize=10, color="#888")
    else:
        ax_b.text(0.5, 0.5,
                  "no avg_temp_c data available\n(need --rebaseline-every >0)",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=10, color="#888")
    ax_b.set_xlabel("avg cell temperature near rebaseline (°C)", fontsize=11)
    ax_b.set_ylabel("P_static (W)", fontsize=11)
    ax_b.set_title("P_static vs temperature — drift origin", fontsize=11)
    ax_b.grid(True, alpha=0.3)

    fig.suptitle(f"P_static drift diagnostics — {gpu}", y=0.99, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[save] {out_png}")
    return True


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
    def _safe_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _cell_key(r):
        cat = r.get("category", "elementwise")
        if cat == "matmul":
            return (1, r.get("variant", ""), _safe_int(r["load_value"]))
        if cat == "matmul_llm":
            return (2, r.get("llm_preset", ""), r.get("dtype", ""),
                    _safe_int(r["load_value"]))
        if cat == "fused":
            return (3, r.get("variant", ""), r.get("dtype", ""),
                    str(r.get("load_value", "")))
        return (0, r["op"], r["dtype"], _safe_int(r["load_value"]))

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
    def _i(v):
        # load_value is int for elementwise/matmul/matmul_llm but a
        # shape-encoded string for `fused`. Render whatever it is as text.
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return str(v)
    if r.get("category") == "matmul":
        return f"{r.get('variant','matmul')}·K{_i(r['load_value'])}"
    if r.get("category") == "matmul_llm":
        return f"llm·{r.get('llm_preset','?')}·{r.get('dtype','?')}·T{_i(r['load_value'])}"
    if r.get("category") == "fused":
        # Compact : "fused·attention_flash·bf16" — load_value (long
        # shape string) is omitted to keep the x-tick narrow.
        return f"fused·{r.get('variant','?')}·{r.get('dtype','?')}"
    return f"{r['dtype']}·{r['op']}·N{_i(r['load_value'])}"


def plot_cache_regime(df: pd.DataFrame, by_regime: pd.DataFrame,
                      out_dir: Path, stem: str, gpu: str,
                      by_regime_unfiltered: pd.DataFrame | None = None) -> None:
    """Energy-per-operation grouped by cache locality regime — one PNG
    per panel for clean x-axes.

    Six output files (when both categories have data):
        <stem>_02_cache_regime_elementwise_strip.png      Panel A elementwise
        <stem>_02_cache_regime_elementwise_kop.png        Panel B elementwise
        <stem>_02_cache_regime_elementwise_dynpower.png   Panel C elementwise
        <stem>_02_cache_regime_matmul_strip.png           Panel A matmul
        <stem>_02_cache_regime_matmul_kop.png             Panel B matmul
        <stem>_02_cache_regime_matmul_dynpower.png        Panel C matmul

    Panel A : per-cell strip of the per-op unit (J/elem or J/FLOP) vs regime.
    Panel B : regime-specific slope_dyn (k_op) from summarize_by_regime,
              annotated with the numeric value on each bar.
    Panel C : mean dynamic power (W) per regime — DRAM streaming raises
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
    # still readable below — the legacy column values
    # (l2_resident / l2_partial / dram_stream) are mapped onto the new
    # labels for plotting so old and new data can analyse the same way.
    regime_order = list(REGIME_ORDER)
    regime_hit_rate = REGIME_HIT_PCT
    regime_x = {r: i for i, r in enumerate(regime_order)}

    # Map legacy labels onto the new 5-bucket vocabulary up-front so the
    # rest of the function can stay single-vocabulary. No-op on new data.
    df = df.copy()
    df["cache_regime"] = df["cache_regime"].replace(LEGACY_REGIME_MAP)

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

    def _resolve_keys(cat_df, keys_key, key_palette):
        keys = [k for k in key_palette.keys() if k in cat_df[keys_key].unique()]
        extras = sorted(k for k in cat_df[keys_key].unique() if k not in keys)
        keys = keys + extras
        extra_cmap = plt.get_cmap("tab20")
        colors = dict(key_palette)
        for i, k in enumerate(extras):
            colors[k] = extra_cmap(i % 20)
        return keys, colors

    def _panel_strip(cat_df, keys_col, keys, colors, unit_col, unit_label,
                     title, out_png):
        fig, ax = plt.subplots(figsize=(14, 7))
        any_pts = False
        for key in keys:
            g = cat_df[cat_df[keys_col] == key]
            if g.empty:
                continue
            ys_raw = pd.to_numeric(g[unit_col], errors="coerce")
            mask = ys_raw.notna() & (ys_raw > 0)
            if not mask.any():
                continue
            xs = g["cache_regime"].map(regime_x).astype(float) \
                + (0.05 * (hash(str(key)) % 7 - 3))
            ax.scatter(xs[mask.values], ys_raw[mask].values,
                       s=58, color=colors[key], marker="o", alpha=0.85,
                       edgecolors="white", linewidths=0.6,
                       label=str(key))
            any_pts = True
        ax.set_xticks(list(regime_x.values()))
        ax.set_xticklabels([f"{r}\n({regime_hit_rate[r]} L2 hit)" for r in regime_order],
                           fontsize=11)
        ax.set_ylabel(f"{unit_label} (per cell, dynamic)")
        if any_pts:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(title)
        if any_pts:
            ax.legend(fontsize=9, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        _save_fig(fig, out_png)

    def _panel_kop(cat_sum, keys, colors, keys_key, unit_label,
                   annot_scale, annot_suffix, title, out_png):
        if cat_sum is None or cat_sum.empty:
            return
        fig, ax = plt.subplots(figsize=(14, 7))
        width = 0.8 / max(1, len(keys))
        xpos = np.arange(len(regime_order))
        positive_vals: list[float] = []
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
                        med = float(row["median_j_per_unit"].iloc[0])
                        sl = med if np.isfinite(med) and med > 0 else np.nan
                    vals.append(sl)
                    r2s.append(float(row["R2_dyn"].iloc[0]))
            bars = ax.bar(xpos + (i - (len(keys)-1)/2) * width, vals, width,
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
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                        txt, ha="center", va="bottom", fontsize=8, linespacing=1.1)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{r}\n({regime_hit_rate[r]})" for r in regime_order],
                           fontsize=11)
        ax.set_ylabel(f"k_op = slope_dyn  ({unit_label})")
        if positive_vals:
            ax.set_yscale("log")
            ax.set_ylim(min(positive_vals) * 0.3, max(positive_vals) * 5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=9, ncol=min(len(keys), 5), loc="upper left")
        _save_fig(fig, out_png)

    def _panel_dynpower(cat_df, title, out_png):
        fig, ax = plt.subplots(figsize=(11, 7))
        mean_p = {}
        for r in regime_order:
            g = cat_df[cat_df["cache_regime"] == r]
            vals = pd.to_numeric(g.get("dyn_power_w", pd.Series(dtype=float)),
                                 errors="coerce").dropna()
            mean_p[r] = float(vals.mean()) if len(vals) else float("nan")
        x_regime = list(range(len(regime_order)))
        y_power  = [mean_p[r] for r in regime_order]
        bar_colors = ["#2ca02c", "#7fc97f", "#ff7f0e", "#fb9a99", "#d62728"]
        bars = ax.bar(x_regime, y_power, color=bar_colors, alpha=0.9, edgecolor="white")
        for rect, v in zip(bars, y_power):
            if not np.isnan(v):
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                        f"{v:.0f} W", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x_regime)
        ax.set_xticklabels([f"{r}\n({regime_hit_rate[r]})" for r in regime_order],
                           fontsize=11)
        ax.set_ylabel("mean dyn power  (W)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if y_power and not all(np.isnan(v) for v in y_power):
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.2)
        _save_fig(fig, out_png)

    ew_palette = PALETTE_ELEMENTWISE_OPS
    mm_palette = PALETTE_MATMUL_VARIANTS

    ew_sum = (by_regime[by_regime["category"] == "elementwise"]
              if by_regime is not None and not by_regime.empty else pd.DataFrame())
    mm_sum = (by_regime[by_regime["category"] == "matmul"]
              if by_regime is not None and not by_regime.empty else pd.DataFrame())

    if not ew.empty:
        keys_key = "op"
        keys, colors = _resolve_keys(ew, keys_key, ew_palette)
        _panel_strip(ew, keys_key, keys, colors, "j_per_element_dyn", "J / element",
                     f"Elementwise raw spread per cell — {gpu}",
                     out_dir / f"{stem}_02_cache_regime_elementwise_strip.png")
        _panel_kop(ew_sum, keys, colors, keys_key, "J / element",
                   1e12, "pJ/elem",
                   f"Elementwise k_op per cache regime — {gpu}  (annotated in pJ/elem)",
                   out_dir / f"{stem}_02_cache_regime_elementwise_kop.png")
        _panel_dynpower(ew,
                        f"Elementwise — steady-state dyn power per cache regime — {gpu}",
                        out_dir / f"{stem}_02_cache_regime_elementwise_dynpower.png")
    if not mm.empty:
        keys_key = "variant"
        keys, colors = _resolve_keys(mm, keys_key, mm_palette)
        _panel_strip(mm, keys_key, keys, colors, "j_per_flop_dyn", "J / FLOP",
                     f"Matmul raw spread per cell — {gpu}",
                     out_dir / f"{stem}_02_cache_regime_matmul_strip.png")
        _panel_kop(mm_sum, keys, colors, keys_key, "J / FLOP",
                   1e12, "pJ/FLOP",
                   f"Matmul k_op per cache regime — {gpu}  (annotated in pJ/FLOP)",
                   out_dir / f"{stem}_02_cache_regime_matmul_kop.png")
        _panel_dynpower(mm,
                        f"Matmul — steady-state dyn power per cache regime — {gpu}",
                        out_dir / f"{stem}_02_cache_regime_matmul_dynpower.png")

    # ---- fp8 dedicated panel ------------------------------------------
    # Even when --include-emulated is OFF (default), surface the fp8
    # elementwise + fp8_te matmul k_op-per-regime since the user asked
    # for an fp8-specific view in PR #55. Uses the UNFILTERED by_regime
    # df (when caller supplied it), so emulated rows are always present
    # here regardless of the --include-emulated flag.
    fp8_src = by_regime_unfiltered if by_regime_unfiltered is not None else by_regime
    if fp8_src is not None and not fp8_src.empty:
        _coef_bar_fp8_per_regime(
            fp8_src,
            out_dir / f"{stem}_02_cache_regime_fp8.png", gpu)


def plot_energy_decomposition(by_regime: pd.DataFrame, out_png: Path,
                              gpu: str) -> bool:
    """MECE energy breakdown for elementwise ops at l2_hit_0.

    For each (op, dtype) pair where we have BOTH `l2_hit_100` and
    `l2_hit_0` measurements, decompose the dyn-energy slope at l2_hit_0
    into three components that are mutually exclusive and collectively
    exhaustive (MECE) :

        Total          = J(op, dtype, l2_hit_0)
        ────────────────────────────────────────
        A) "L2-resident workload"
           = J(op, fp16, l2_hit_100)
           Contains : SM compute + L1 / SMEM transit + register-file
           activity + L2 traffic + kernel launch overhead. NOT further
           decomposable from NVML measurements alone.

        B) "FP8 cast overhead"  (only when dtype = fp8)
           = J(op, fp8, l2_hit_100) − J(op, fp16, l2_hit_100)
           Contains : the extra cost the fp8 emulation path adds on top
           of fp16 — separate cast kernel launches, fp16 intermediate
           tensor materialisation, cast-compute itself. Always 0 for
           fp16 rows by construction.

        C) "DRAM round-trip"
           = J(op, dtype, l2_hit_0) − J(op, dtype, l2_hit_100)
           Contains : the marginal cost of streaming the working set
           through HBM. PR #30's marginal-DRAM technique.

        Identity:  A + B + C  ≡  J(op, dtype, l2_hit_0)
                   exact algebraic — no double-counting, no missing
                   piece, hence MECE.

    What we INTENTIONALLY don't try to decompose :
      * "compute vs L2 vs launch" inside component A — there is no
        compute-only measurement in the suite (PyTorch does not allow
        register-resident microbenchmarking), so any further split would
        be an estimate, not an exact subtraction. Component A stays
        bundled. The plot's text annotation flags this caveat.

    Plot layout :
      * One stacked bar per (op, dtype) at l2_hit_0
      * Three colour-coded layers : resident (green) → cast (orange) →
        DRAM (red), bottom to top
      * Bar height equals total pJ/elem; each layer's percentage of
        total is annotated inline
      * Title spells out the MECE identity so the reader can verify
        on the plot itself
    """
    if by_regime is None or by_regime.empty:
        return False
    ew = by_regime[by_regime["category"] == "elementwise"].copy()
    if ew.empty:
        return False
    ew["cache_regime"] = ew["cache_regime"].replace(LEGACY_REGIME_MAP)

    slope_col = "slope_dyn_wls" if "slope_dyn_wls" in ew.columns else "slope_dyn"

    def _slope(op: str, dtype: str, regime: str):
        sub = ew[(ew["op"] == op) & (ew["dtype"] == dtype)
                 & (ew["cache_regime"] == regime)]
        if sub.empty:
            return None
        v = sub[slope_col].iloc[0]
        if pd.isna(v) or v <= 0:
            return None
        return float(v)

    # Build decomposition rows for every (op, dtype) pair that has the
    # measurements we need. Skip silently when a piece is missing — the
    # plot just omits that bar rather than confabulating.
    #
    # Excluded ops : `add` is intentionally dropped. Per user feedback —
    # add_fp8 / add_fp16 are too trivial (1 FLOP/elem, identical pattern
    # to mul) to be worth the bar real estate in this MECE view. Keeping
    # mul as the lone simple-op reference is enough.
    EXCLUDED_OPS = {"add"}
    bars = []
    for op in sorted(ew["op"].unique()):
        if op in EXCLUDED_OPS:
            continue
        for dtype in ("fp16", "fp8"):
            j_self_l2  = _slope(op, dtype, "l2_hit_100")
            j_self_dr  = _slope(op, dtype, "l2_hit_0")
            if j_self_l2 is None or j_self_dr is None:
                continue
            j_fp16_l2 = _slope(op, "fp16", "l2_hit_100") if dtype == "fp8" else None
            # Component A — "resident workload" (compute + L2 + launch)
            # For fp16 : just j_self_l2.
            # For fp8 : the fp16 baseline of the same op (so cast overhead
            # is attributed cleanly to component B).
            if dtype == "fp8":
                if j_fp16_l2 is None:
                    # Without an fp16 baseline we can't separate cast
                    # cleanly — skip rather than misattribute.
                    continue
                A = j_fp16_l2
                B = j_self_l2 - j_fp16_l2     # cast overhead
            else:
                A = j_self_l2
                B = 0.0
            C = j_self_dr - j_self_l2          # DRAM round-trip
            total = A + B + C                  # = j_self_dr  (identity)
            bars.append({
                "label": f"{op}_{dtype}",
                "op":    op,
                "dtype": dtype,
                "A":     A,
                "B":     B,
                "C":     C,
                "total": total,
                "j_self_dr": j_self_dr,         # for sanity check
            })
    if not bars:
        return False

    plt = _get_mpl()
    # 2-row layout : main chart on top, caveat box on its own row below.
    # Reserves a dedicated chunk of the figure for the caveat so it's
    # never clipped by tight_layout / bbox_inches gymnastics. Also
    # taller than before (was 7.5 in.) so segments at the bottom of the
    # stack remain visible even when one component is 99 % of total.
    fig, (ax, ax_caveat) = plt.subplots(
        2, 1,
        figsize=(max(13, 1.6 * len(bars) + 5), 11),
        gridspec_kw={"height_ratios": [9, 1.2], "hspace": 0.05})
    ax_caveat.set_axis_off()
    xs = np.arange(len(bars))

    A_vals = [b["A"] * 1e12 for b in bars]
    B_vals = [b["B"] * 1e12 for b in bars]
    C_vals = [b["C"] * 1e12 for b in bars]

    ax.bar(xs, A_vals, color="#2ca02c", edgecolor="white",
           label="A) L2-resident workload\n(compute + L2 + launch)")
    ax.bar(xs, B_vals, bottom=A_vals, color="#ff7f0e", edgecolor="white",
           label="B) FP8 cast overhead\n(cast-compute-cast)")
    A_plus_B = [a + b for a, b in zip(A_vals, B_vals)]
    ax.bar(xs, C_vals, bottom=A_plus_B, color="#d62728", edgecolor="white",
           label="C) DRAM round-trip\n(marginal HBM cost)")

    def _fmt_pj(v):
        """Format pJ value with reasonable precision."""
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if abs(v) >= 0.01:
            return f"{v:.3f}"
        return f"{v:.2e}"

    # Per-bar annotations : total at top + per-component value+pct.
    # Strategy : if a segment is too small (< 6% of total) for an inline
    # label, render the label OUTSIDE the bar with a leader line so the
    # value is still visible. Otherwise render inline (white text on
    # the coloured segment).
    INLINE_PCT_THRESHOLD = 6.0
    LEADER_X_OFFSET = 0.42  # how far right of the bar to place outside labels
    for i, b in enumerate(bars):
        total_pj = b["total"] * 1e12
        a_pct = 100.0 * b["A"] / b["total"] if b["total"] > 0 else 0
        b_pct = 100.0 * b["B"] / b["total"] if b["total"] > 0 else 0
        c_pct = 100.0 * b["C"] / b["total"] if b["total"] > 0 else 0

        # Total above the stack
        ax.text(xs[i], total_pj * 1.02,
                f"Σ = {_fmt_pj(total_pj)} pJ/elem",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        segments = [
            ("A", A_vals[i], a_pct, A_vals[i] / 2,
             0,  # bottom of A
             "white"),
            ("B", B_vals[i], b_pct, A_vals[i] + B_vals[i] / 2,
             A_vals[i],  # bottom of B
             "white"),
            ("C", C_vals[i], c_pct, A_vals[i] + B_vals[i] + C_vals[i] / 2,
             A_vals[i] + B_vals[i],  # bottom of C
             "white"),
        ]
        # Use a separate y for outside labels so they don't pile up.
        outside_y_cursor = total_pj * 0.05  # start near bottom of bar
        for name, value_pj, pct, mid_y, _bot, _color in segments:
            if value_pj <= 0:
                continue
            label = f"{name}: {_fmt_pj(value_pj)} pJ\n({pct:.1f}%)"
            if pct >= INLINE_PCT_THRESHOLD:
                # Inline — fits comfortably in the segment
                ax.text(xs[i], mid_y, label,
                        ha="center", va="center", fontsize=8.5,
                        color=_color, fontweight="bold", linespacing=1.1)
            else:
                # Outside — leader line from segment to right-side text
                ax.annotate(
                    label,
                    xy=(xs[i] + 0.30, mid_y),                       # bar edge
                    xytext=(xs[i] + LEADER_X_OFFSET, outside_y_cursor),
                    fontsize=8, ha="left", va="center",
                    color="#222",
                    arrowprops=dict(arrowstyle="-", color="#888",
                                    lw=0.8, alpha=0.7,
                                    connectionstyle="arc3,rad=0.0"),
                    bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                              alpha=0.92, pad=2))
                outside_y_cursor += total_pj * 0.06   # next outside label below

    ax.set_xticks(xs)
    ax.set_xticklabels([b["label"] for b in bars], rotation=20, ha="right",
                       fontsize=10)
    ax.set_ylabel("pJ / element  (dynamic, at l2_hit_0)", fontsize=11)
    # Headroom for the Σ label above each bar.
    if any(v > 0 for v in [b["total"] for b in bars]):
        ax.set_ylim(0, max(b["total"] for b in bars) * 1.15 * 1e12)
    ax.set_title(
        f"MECE energy decomposition — elementwise @ l2_hit_0 — {gpu}\n"
        "A + B + C  ≡  J(op, dtype, l2_hit_0)   (algebraic identity → no overlap, no missing piece)\n"
        "A = J(op, fp16, l2_hit_100) ;  B = J(op, fp8, l2_hit_100) − J(op, fp16, l2_hit_100) ;  "
        "C = J(op, dtype, l2_hit_0) − J(op, dtype, l2_hit_100)",
        fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=1, bbox_to_anchor=(1.01, 1.0))

    # Caveat row — guaranteed visible because it lives in its own subplot.
    ax_caveat.text(
        0.5, 0.5,
        "Note : component A (resident workload) bundles compute + L2 transit + "
        "kernel-launch overhead because no NVML measurement in this suite isolates "
        "pure compute. Further breakdown of A would require an estimate "
        "(e.g. FLOP × J_per_FLOP_reference), which is NOT MECE — "
        "it intentionally remains a single bucket.",
        ha="center", va="center", fontsize=9, color="#333",
        wrap=True,
        bbox=dict(facecolor="#f0f0f0", edgecolor="#bbbbbb", pad=6))

    _save_fig(fig, out_png)
    return True


def plot_energy_decomposition_matmul(by_regime: pd.DataFrame, out_png: Path,
                                     gpu: str) -> bool:
    """MECE energy breakdown for MATMUL variants at l2_hit_0.

    Unlike the elementwise version (PR #58) which has 3 components
    (resident / fp8-cast-overhead / DRAM), matmul has only 2 :

        Total      = J(matmul, variant, l2_hit_0)
        ──────────────────────────────────────────────
        A) "L2-resident workload"
           = J(matmul, variant, l2_hit_100)
           bundles SM/TC compute + L2 traffic + register file +
           kernel launch. NOT further decomposable.

        C) "DRAM round-trip"
           = J(matmul, variant, l2_hit_0) − J(matmul, variant, l2_hit_100)
           marginal HBM cost.

        Identity:  A + C ≡ J(matmul, variant, l2_hit_0)   ← MECE

    Why no "B" cast-overhead term :
      * On H100, matmul_fp8_te is NATIVE (no emulation) → fp8 vs fp16
        delta is a *hardware advantage*, not an overhead. Calling it
        "cast" would be misleading.
      * On A100, fp8_te falls back to FP16 TC → fp8 vs fp16 delta ≈ 0.
        Again no cast overhead.
      * Each variant gets its own bar (5 bars for 5 variants), each
        decomposed into its own A + C. Comparison across variants is
        the point — see Tensor Core gap (fp32_simt vs fp16_tc) and
        FP8 advantage (fp16_tc vs fp8_te) by reading off the bar
        heights.

    CRITICAL CAVEAT — matmul cache regime is approximate :
      `classify_cache_regime()` uses *logical* working set
      (3·K²·bpe). cuBLAS / TE matmul kernels do tile reuse, so the
      ACTUAL DRAM traffic is much less than logical (each input
      element is reused O(K) times within an SM tile cache). That
      means C is a noisy upper bound on real DRAM cost. The plot's
      caveat box flags this.

    Closes G4 (P2.1) from REVIEW.md §7.
    """
    if by_regime is None or by_regime.empty:
        return False
    mm = by_regime[by_regime["category"] == "matmul"].copy()
    if mm.empty:
        return False
    mm["cache_regime"] = mm["cache_regime"].replace(LEGACY_REGIME_MAP)

    slope_col = "slope_dyn_wls" if "slope_dyn_wls" in mm.columns else "slope_dyn"

    def _slope(variant: str, regime: str):
        sub = mm[(mm["variant"] == variant) & (mm["cache_regime"] == regime)]
        if sub.empty:
            return None
        v = sub[slope_col].iloc[0]
        if pd.isna(v) or v <= 0:
            return None
        return float(v)

    # 5 variants in canonical order. fp8_te gets a 3-component breakdown
    # (A_fp16_baseline + B_fp8_overhead + C_DRAM); the other 4 stay
    # 2-component (A + C). The B term captures the *emulation /
    # cast / scaling* delta vs the FP16 TC baseline:
    #   * On A100  : matmul_fp8_te falls back to FP16 TC (`emulated=1`),
    #                so B = J(fp8_te) − J(fp16_tc) is the FP16-fallback
    #                overhead. Typically small but positive.
    #   * On H100+ : matmul_fp8_te is NATIVE; B can be near 0 or even
    #                NEGATIVE (fp8 path is more efficient than fp16).
    #                When B ≤ 0 we render the fp8_te bar as 2-component
    #                (its own A + C, no B layer) and add a "FP8 native
    #                advantage" annotation showing the saved energy.
    variant_order = ["matmul_fp32_simt", "matmul_tf32_tc",
                     "matmul_fp16_tc", "matmul_bf16_tc",
                     "matmul_fp8_te"]
    bars = []
    fp16_baseline_l2 = _slope("matmul_fp16_tc", "l2_hit_100")  # may be None
    for variant in variant_order:
        if variant not in mm["variant"].values:
            continue
        A_self = _slope(variant, "l2_hit_100")
        T = _slope(variant, "l2_hit_0")
        if A_self is None or T is None:
            continue
        emu_row = mm[mm["variant"] == variant]
        emu = bool(int(emu_row["emulated"].iloc[0])) if "emulated" in emu_row.columns else False
        # 3-component path only for fp8_te WHEN the fp16 baseline exists
        # AND emulation overhead is positive. Otherwise fall back to 2-
        # component to keep the stacked bar honest.
        is_fp8 = (variant == "matmul_fp8_te")
        if is_fp8 and fp16_baseline_l2 is not None:
            A = fp16_baseline_l2
            B = A_self - fp16_baseline_l2  # may be ≤ 0 on Hopper native
        else:
            A = A_self
            B = 0.0
        if B < 0:
            # Hopper native fp8 — render as 2-component, save B for annotation.
            B_advantage = -B  # positive number; how much fp8 saved vs fp16
            A = A_self  # use fp8's own L2-resident
            B = 0.0
        else:
            B_advantage = 0.0
        C = T - A - B
        bars.append({
            "variant": variant, "A": A, "B": B, "C": C, "total": T,
            "emulated": emu, "is_fp8": is_fp8, "B_advantage": B_advantage,
        })
    if not bars:
        return False

    plt = _get_mpl()
    # 2-row layout : main chart + dedicated caveat row. hspace is wide
    # enough that x-tick labels don't kiss the caveat box ; height_ratios
    # 9:1.6 keeps the caveat readable but compact. Total figure height
    # 9 in (was 11) — bbox_inches="tight" in _save_fig crops empty space
    # below the caveat so the saved PNG isn't dominated by whitespace.
    fig, (ax, ax_caveat) = plt.subplots(
        2, 1,
        figsize=(max(12, 1.8 * len(bars) + 5), 9.0),
        gridspec_kw={"height_ratios": [9, 1.6], "hspace": 0.32})
    ax_caveat.set_axis_off()
    xs = np.arange(len(bars))

    # Convert J/FLOP to pJ/FLOP for display
    A_vals = [b["A"] * 1e12 for b in bars]
    B_vals = [b["B"] * 1e12 for b in bars]
    C_vals = [b["C"] * 1e12 for b in bars]
    T_vals = [b["total"] * 1e12 for b in bars]

    # palette
    palette = PALETTE_MATMUL_VARIANTS
    base_colors = [palette.get(b["variant"], "gray") for b in bars]

    # Stack order : A (solid) → B (orange = cast/emu overhead) → C (hatched DRAM)
    legend_a_used = False
    legend_b_used = False
    legend_c_used = False
    for i, b in enumerate(bars):
        # A
        ax.bar(xs[i], A_vals[i], color=base_colors[i], edgecolor="white",
               label=("A) L2-resident workload (compute + L2 + launch)\n"
                      "    fp8: uses fp16 baseline; non-fp8: uses own l2_hit_100")
                     if not legend_a_used else None,
               alpha=0.95)
        legend_a_used = True
        # B — cast / emulation overhead, only positive layer (Hopper native
        # fp8's negative case rendered as 2-component instead).
        if B_vals[i] > 0:
            ax.bar(xs[i], B_vals[i], bottom=A_vals[i],
                   color="#ff7f0e", edgecolor="white",
                   label=("B) FP8 cast / emulation overhead\n"
                          "    = J(fp8_te) − J(fp16_tc) at l2_hit_100\n"
                          "    (positive = emulation / cast cost)")
                         if not legend_b_used else None,
                   alpha=0.85)
            legend_b_used = True
        # C — DRAM
        ax.bar(xs[i], C_vals[i], bottom=A_vals[i] + B_vals[i],
               color=base_colors[i], edgecolor="white", hatch="///",
               label=("C) DRAM round-trip (marginal HBM cost)"
                      if not legend_c_used else None),
               alpha=0.55)
        legend_c_used = True

    def _fmt_pj(v):
        """Format pJ value with reasonable precision for matmul (typ 0.01-100)."""
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if abs(v) >= 0.01:
            return f"{v:.3f}"
        return f"{v:.2e}"

    # Annotations : total above + per-component value+pct.
    # Inline label when component pct is large enough; otherwise a
    # leader line points to a label outside the bar (right side).
    # On log y-scale we use offset_points for the outside labels so
    # vertical spacing is uniform regardless of magnitude.
    INLINE_PCT_THRESHOLD = 8.0
    LEADER_X_OFFSET = 0.42
    for i, b in enumerate(bars):
        total = b["total"] if b["total"] > 0 else 1
        a_pct = 100.0 * b["A"] / total
        b_pct = 100.0 * b["B"] / total
        c_pct = 100.0 * b["C"] / total
        # Total above the stack
        ax.text(xs[i], T_vals[i] * 1.04,
                f"Σ = {_fmt_pj(T_vals[i])} pJ/FLOP",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        # Geometric midpoints work better on a log axis than arithmetic.
        def _gmid(top, bot):
            if top <= 0 or bot <= 0:
                return (top + bot) / 2.0
            import math
            return math.exp((math.log(top) + math.log(bot)) / 2.0)

        a_top = A_vals[i]
        b_top = A_vals[i] + B_vals[i]
        c_top = A_vals[i] + B_vals[i] + C_vals[i]
        segments = [
            ("A", A_vals[i], a_pct,
             _gmid(a_top, max(a_top * 1e-6, 1e-9)) if a_top > 0 else 0,
             "white"),
            ("B", B_vals[i], b_pct, _gmid(b_top, a_top) if B_vals[i] > 0 else 0,
             "white"),
            ("C", C_vals[i], c_pct, _gmid(c_top, b_top) if C_vals[i] > 0 else 0,
             "black"),
        ]
        outside_offset_points = -10.0  # cumulative pt offset for leader labels
        for name, value_pj, pct, mid_y, txt_color in segments:
            if value_pj <= 0:
                continue
            label = f"{name}: {_fmt_pj(value_pj)} pJ\n({pct:.1f}%)"
            if pct >= INLINE_PCT_THRESHOLD and mid_y > 0:
                ax.text(xs[i], mid_y, label,
                        ha="center", va="center", fontsize=8.5,
                        color=txt_color, fontweight="bold", linespacing=1.1)
            else:
                # Outside leader. Use display coords for the text by
                # offsetting from the bar-edge anchor in points.
                ax.annotate(
                    label,
                    xy=(xs[i] + 0.30, mid_y if mid_y > 0 else value_pj),
                    xytext=(40, outside_offset_points),
                    textcoords="offset points",
                    fontsize=8, ha="left", va="center", color="#222",
                    arrowprops=dict(arrowstyle="-", color="#888",
                                    lw=0.8, alpha=0.7),
                    bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                              alpha=0.92, pad=2))
                outside_offset_points -= 30.0  # stack subsequent labels
        # FP8 native advantage annotation : when B was negative, surface
        # the saved energy as a positive number so reader sees "fp8 is
        # cheaper by X pJ/FLOP than fp16 baseline".
        if b["is_fp8"] and b["B_advantage"] > 0:
            adv_pj = b["B_advantage"] * 1e12
            ax.annotate(
                f"FP8 native advantage:\n−{_fmt_pj(adv_pj)} pJ/FLOP\nvs fp16_tc baseline",
                xy=(xs[i], T_vals[i]),
                xytext=(0, 40), textcoords="offset points",
                ha="center", fontsize=8, color="#2ca02c", fontweight="bold",
                bbox=dict(facecolor="#e6f4ea", edgecolor="#2ca02c", pad=3))

    labels = [b["variant"] + (" *EMU" if b["emulated"] else "") for b in bars]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("pJ / FLOP   (dynamic, at l2_hit_0)", fontsize=11)
    ax.set_yscale("log")
    # Explicit ylim — the "Σ = X pJ/FLOP" labels above each bar otherwise
    # push matplotlib's auto-ylim by a full extra decade, leaving most of
    # the chart blank above the tallest bar. 1.5× cap on log scale gives
    # the labels enough room without wasting half the canvas.
    if T_vals:
        positive_T = [v for v in T_vals if v > 0]
        positive_min = [v for v in (A_vals + C_vals) if v > 0]
        if positive_T and positive_min:
            ax.set_ylim(min(positive_min) * 0.5, max(positive_T) * 1.5)
    ax.set_title(
        f"MECE energy decomposition — matmul @ l2_hit_0 — {gpu}\n"
        "fp8_te uses 3 components (A: fp16 baseline, B: cast/emulation "
        "overhead, C: DRAM); the other 4 variants use 2 components (A + C). "
        "B = J(fp8_te) − J(fp16_tc) at l2_hit_100. When B ≤ 0 (Hopper "
        "native), fp8_te falls back to 2 components plus an "
        "'FP8 native advantage' annotation showing the saved energy.\n"
        "Identity (per variant) :  A + B + C ≡ J(variant, l2_hit_0)   ← MECE",
        fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, ncol=1, bbox_to_anchor=(1.01, 1.0))

    # Caveat row — guaranteed visible because it lives in its own subplot.
    ax_caveat.text(
        0.5, 0.5,
        "CRITICAL CAVEAT : matmul's `cache_regime` is based on LOGICAL "
        "working set (3·K²·bpe). cuBLAS/TE matmul kernels reuse each "
        "input element O(K) times via SM tile cache, so actual DRAM "
        "traffic ≪ logical. Component C is therefore a NOISY UPPER BOUND "
        "on real DRAM cost — interpret as 'extra cost when working set "
        "exceeds L2', not literal HBM bytes. README §3.5.3.",
        ha="center", va="center", fontsize=9, color="#333", wrap=True,
        bbox=dict(facecolor="#fff2cc", edgecolor="#d6a800", pad=6))

    _save_fig(fig, out_png)
    return True


# ============================================================================
# Fused vs Standalone (G11 / P1.4) — decomposition + plots
# ============================================================================
#
# Pair each "full" fused variant with its "baseline" (no softmax / no
# activation / no LN) and compute the residual energy that's attributable
# to the op INSIDE the fused kernel. Compare against the existing
# `softmax` / `gelu` / `layernorm` standalone measurements to expose the
# difference (often ≫ 1× because standalone includes HBM round-trip).
#
# Pairing :
#   ("softmax",   "attention_flash"     , "attention_qkv_matmul")
#   ("gelu",      "linear_gelu"         , "linear_baseline_gelu")
#   ("layernorm", "ln_linear"           , "linear_baseline_ln")
#
# Statistical significance heuristic — NVML measurement noise has typical
# CV ~3..5% per cell ; subtraction of two independent noisy measurements
# inflates the variance by √2. So if the residual is < ~10% of the full
# measurement, it's within noise and we label "not statistically
# distinguishable from zero" (analogous to the M(esidual) < 2σ rule).
# ============================================================================

_FUSED_PAIRS = (
    ("softmax",   "attention_flash",     "attention_qkv_matmul"),
    ("gelu",      "linear_gelu",         "linear_baseline_gelu"),
    ("layernorm", "ln_linear",           "linear_baseline_ln"),
)
_FUSED_NOISE_FLOOR_PCT = 5.0 * np.sqrt(2.0)   # ~7.1%


def summarize_fused_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Pair fused full/baseline rows by op group and compute residual energy.

    Returns one row per (op_group, dtype) with :
      * j_per_call_full           — fused full kernel energy (e.g. attention_flash)
      * j_per_call_baseline       — same shape, op stripped (e.g. attention_qkv_matmul)
      * j_per_call_residual       — full − baseline ; "fused op contribution"
      * residual_pct_of_full      — residual / full × 100
      * stat_significant          — 1 if |residual_pct| ≥ 2 × NVML noise floor (~14%)
      * j_per_element_residual    — residual / n_elements_full
      * j_standalone_per_element  — standalone op J/elem at l2_hit_0 (for ratio)
      * ratio_residual_to_standalone — residual / standalone (per-elem) ;
                                       1 = no difference, ≪ 1 = fused is much cheaper
      * fusion_emulated           — 1 if torch.compile fell back to eager (residual unreliable)

    Empty DataFrame when no fused rows exist (sweep didn't use --include-fused).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    fused = df[df["category"] == "fused"].copy()
    if fused.empty:
        return pd.DataFrame()
    # Per-call dynamic energy = dyn_energy_j / iters. Use this rather than
    # j_per_element_dyn because n_elements meaning differs across variants.
    fused["dyn_energy_j_n"] = pd.to_numeric(fused["dyn_energy_j"], errors="coerce")
    fused["iters_n"]        = pd.to_numeric(fused["iters"], errors="coerce")
    fused["j_per_call_dyn"] = fused["dyn_energy_j_n"] / fused["iters_n"]
    fused["n_elem_n"]       = pd.to_numeric(fused["n_elements"], errors="coerce")

    # Standalone J/elem at l2_hit_0 — for ratio comparison. Use the
    # dyn_energy_j / total_elements averaged over the largest cells
    # (l2_hit_0 regime) per (op, dtype). Falls back to NaN when missing.
    standalone = df[df["category"] == "elementwise"].copy()
    standalone["jpe"] = pd.to_numeric(standalone["j_per_element_dyn"], errors="coerce")
    if "cache_regime" in standalone.columns:
        standalone["cache_regime"] = standalone["cache_regime"].replace(LEGACY_REGIME_MAP)
        # l2_hit_0 = pure DRAM-streaming = closest to "standalone op cost
        # in real workload" because LLM tensors typically don't fit L2.
        sa_l2_hit_0 = standalone[standalone["cache_regime"] == "l2_hit_0"]
        sa_summary = (sa_l2_hit_0.groupby(["op", "dtype"])["jpe"].median()
                      if not sa_l2_hit_0.empty else
                      standalone.groupby(["op", "dtype"])["jpe"].median())
    else:
        sa_summary = standalone.groupby(["op", "dtype"])["jpe"].median()

    rows = []
    for op_group, full_var, base_var in _FUSED_PAIRS:
        for dtype in sorted(fused["dtype"].unique()):
            sf = fused[(fused["op"] == full_var) & (fused["dtype"] == dtype)]
            sb = fused[(fused["op"] == base_var) & (fused["dtype"] == dtype)]
            if sf.empty or sb.empty:
                continue
            J_full = float(sf["j_per_call_dyn"].iloc[0])
            J_base = float(sb["j_per_call_dyn"].iloc[0])
            J_res = J_full - J_base
            n_elem = int(sf["n_elem_n"].iloc[0]) if not pd.isna(sf["n_elem_n"].iloc[0]) else 0
            residual_pct = 100.0 * J_res / J_full if J_full > 0 else float("nan")
            stat_sig = (not pd.isna(residual_pct)
                        and abs(residual_pct) >= 2.0 * _FUSED_NOISE_FLOOR_PCT)
            jpe_residual = J_res / n_elem if n_elem > 0 else float("nan")
            jpe_standalone = float(sa_summary.get((op_group, dtype), float("nan")))
            ratio = (jpe_residual / jpe_standalone
                     if jpe_standalone > 0 and not pd.isna(jpe_residual)
                     else float("nan"))
            fusion_emu = int(sf["emulated"].astype(int).iloc[0]) if "emulated" in sf.columns else 0
            rows.append({
                "op_group": op_group,
                "dtype": dtype,
                "full_variant": full_var,
                "baseline_variant": base_var,
                "j_per_call_full":     J_full,
                "j_per_call_baseline": J_base,
                "j_per_call_residual": J_res,
                "residual_pct_of_full":     residual_pct,
                "residual_pct_noise_floor": _FUSED_NOISE_FLOOR_PCT,
                "stat_significant":         int(bool(stat_sig)),
                "n_elements_full":          n_elem,
                "j_per_element_residual":   jpe_residual,
                "j_per_element_standalone": jpe_standalone,
                "ratio_residual_to_standalone": ratio,
                "fusion_emulated": fusion_emu,
                "shape_full": str(sf["shape"].iloc[0]) if "shape" in sf.columns else "",
            })
    return pd.DataFrame(rows)


def plot_fused_vs_standalone_bar(decomp_df: pd.DataFrame, out_png: Path,
                                 gpu: str) -> bool:
    """Grouped bar — 3 op groups × 2 metrics (standalone J/elem vs
    fused-residual J/elem). Annotates ratio fused/standalone and flags
    "near noise floor" residuals. Returns True if rendered.
    """
    if decomp_df is None or decomp_df.empty:
        return False
    plt = _get_mpl()
    # One panel per dtype so fp16/bf16 don't get squashed.
    dtypes = sorted(decomp_df["dtype"].unique())
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 6.5),
                             squeeze=False)
    op_order = [p[0] for p in _FUSED_PAIRS]
    for ax_idx, dtype in enumerate(dtypes):
        ax = axes[0, ax_idx]
        sub = decomp_df[decomp_df["dtype"] == dtype].set_index("op_group")
        sub = sub.reindex([o for o in op_order if o in sub.index])
        if sub.empty:
            ax.set_visible(False); continue
        xs = np.arange(len(sub))
        w = 0.35
        std_vals = (sub["j_per_element_standalone"] * 1e12).values
        res_vals = (sub["j_per_element_residual"]   * 1e12).values
        ax.bar(xs - w/2, std_vals, w, color="#1f77b4", edgecolor="white",
               label="standalone (PyTorch op, l2_hit_0)")
        # Highlight residuals that aren't statistically distinguishable from 0.
        bar_colors = ["#a4d4a4" if s else "#d4a4a4" for s in sub["stat_significant"].values]
        ax.bar(xs + w/2, res_vals, w, color=bar_colors, edgecolor="black",
               hatch=["" if s else "//" for s in sub["stat_significant"].values],
               label="fused-residual (full − baseline)")
        # Annotate ratio above the residual bar
        for i, (_, row) in enumerate(sub.iterrows()):
            if not pd.isna(row["ratio_residual_to_standalone"]):
                ratio = row["ratio_residual_to_standalone"]
                tag = ""
                if not row["stat_significant"]:
                    tag = "\n(within noise)"
                if row["fusion_emulated"]:
                    tag += "\n⚠ fusion failed"
                ax.text(xs[i] + w/2, res_vals[i],
                        f"ratio = {ratio:.2f}×{tag}",
                        ha="center", va="bottom", fontsize=8, linespacing=1.1)
            ax.text(xs[i] - w/2, std_vals[i],
                    f"{std_vals[i]:.1f} pJ", ha="center", va="bottom", fontsize=8)
            ax.text(xs[i] + w/2, res_vals[i] if res_vals[i] > 0 else 0,
                    f"{res_vals[i]:+.2f} pJ", ha="center", va="top", fontsize=7,
                    color="#444")
        ax.set_xticks(xs); ax.set_xticklabels(sub.index, fontsize=11)
        ax.set_ylabel("pJ / element  (dynamic)")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.set_title(f"{dtype} — fused vs standalone")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        f"Fused-vs-Standalone — softmax / gelu / layernorm — {gpu}\n"
        f"blue = standalone PyTorch op at l2_hit_0  ;  green = fused-residual "
        f"(stat. distinguishable from 0)  ;  red+hatch = within NVML noise floor "
        f"({2*_FUSED_NOISE_FLOOR_PCT:.1f}%).  ratio = fused-residual / standalone "
        f"(≪ 1 = HBM-bound standalone overstates fused)",
        y=1.02, fontsize=10)
    _save_fig(fig, out_png)
    return True


def plot_attention_decomposition(decomp_df: pd.DataFrame, df: pd.DataFrame,
                                 out_png: Path, gpu: str) -> bool:
    """Stacked bar — for each dtype, attention_flash energy split into
    matmul (QKᵀ + (QKᵀ)V) + softmax-residual.

    `decomp_df` row corresponds to `op_group="softmax"` :
        J_qkv_matmul = j_per_call_baseline   (the matmul-only baseline)
        J_softmax    = j_per_call_residual   (what flash adds beyond matmul)
        J_full       = j_per_call_full

    Returns True if rendered.
    """
    if decomp_df is None or decomp_df.empty:
        return False
    sm = decomp_df[decomp_df["op_group"] == "softmax"]
    if sm.empty:
        return False
    plt = _get_mpl()
    # Wider figure to accommodate the legend OUTSIDE the axes (right side).
    fig, (ax, ax_caveat) = plt.subplots(
        2, 1, figsize=(max(11, 2.5 * len(sm) + 6), 8.5),
        gridspec_kw={"height_ratios": [9, 1.2], "hspace": 0.22})
    ax_caveat.set_axis_off()

    xs = np.arange(len(sm))
    # mJ / call so the numbers are readable (typical attention call ≈ 1..50 mJ)
    qkv_vals  = (sm["j_per_call_baseline"].values) * 1e3
    sm_vals   = (sm["j_per_call_residual"].values) * 1e3
    full_vals = (sm["j_per_call_full"].values)     * 1e3

    # Decomposition rendering depends on the SIGN of the residual :
    #
    #   residual ≥ 0  → flash total = matmul-baseline + softmax-residual
    #                   (decomposable stacked bar — the textbook case)
    #
    #   residual < 0  → flash total < matmul-baseline (i.e. the FUSED kernel
    #                   is more efficient than the 2-call matmul baseline,
    #                   even before adding softmax). NOT decomposable as a
    #                   stack — would be misleading. Instead we render the
    #                   ACTUAL flash bar and a horizontal reference line at
    #                   matmul-baseline height with a "saves X mJ" tag.
    #
    # Either way the visible bar height = ACTUAL fused flash energy, so
    # cross-bar comparison stays honest.
    for i in range(len(sm)):
        q, s, t = qkv_vals[i], sm_vals[i], full_vals[i]
        if s >= 0:
            # textbook stacked decomposition
            ax.bar(xs[i], q, color="#1f77b4", edgecolor="white", alpha=0.85,
                   label=("matmul-baseline (Q@Kᵀ + (Q@Kᵀ)V)"
                          if i == 0 else None))
            ax.bar(xs[i], s, bottom=q, color="#ff7f0e", edgecolor="white",
                   alpha=0.85,
                   label=("softmax-residual = J(flash) − J(matmul-baseline)"
                          if i == 0 else None))
            # total label
            ax.text(xs[i], t * 1.02, f"flash = {t:.2f} mJ",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            # in-bar
            if q / max(t, 1e-9) > 0.08:
                ax.text(xs[i], q / 2, f"matmul\n{q:.2f} mJ\n({100*q/t:.0f}%)",
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold", linespacing=1.1)
            if s / max(t, 1e-9) > 0.05:
                ax.text(xs[i], q + s / 2, f"softmax\n+{s:.2f} mJ\n({100*s/t:.0f}%)",
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold", linespacing=1.1)
        else:
            # Flash beats separate matmul-pair → not decomposable as a stack.
            # Show ACTUAL flash bar in green, plus dashed reference line at
            # matmul-baseline height with a "savings" annotation.
            ax.bar(xs[i], t, color="#2ca02c", edgecolor="white", alpha=0.85,
                   label=("flash total (more efficient than the matmul-pair)"
                          if (i == 0 or sm_vals[:i].min() >= 0) else None))
            # reference line at matmul-baseline
            ax.hlines(q, xs[i] - 0.4, xs[i] + 0.4,
                      colors="#1f77b4", linestyles="--", linewidth=1.5,
                      label=("matmul-baseline reference (sum of separate matmul-pair)"
                             if (i == 0 or sm_vals[:i].min() >= 0) else None))
            # in-bar : flash total
            ax.text(xs[i], t / 2, f"flash\n{t:.2f} mJ",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold", linespacing=1.1)
            # baseline label above the dashed line
            ax.text(xs[i], q * 1.02, f"matmul-baseline = {q:.2f} mJ",
                    ha="center", va="bottom", fontsize=8, color="#1f77b4",
                    fontweight="bold")
            # "savings" annotation between bar top and reference line
            saved_mJ = q - t
            saved_pct = 100.0 * saved_mJ / q if q > 0 else 0
            ax.annotate(
                f"flash saves\n{saved_mJ:.2f} mJ\n({saved_pct:.1f}% of baseline)",
                xy=(xs[i] + 0.1, (t + q) / 2),
                xytext=(15, 0), textcoords="offset points",
                fontsize=8, ha="left", va="center", color="#2ca02c",
                fontweight="bold", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color="#2ca02c",
                                lw=0.8, alpha=0.8),
                bbox=dict(facecolor="#e6f4ea", edgecolor="#2ca02c", pad=3))

    labels = [f"{r['dtype']}\n{r.get('shape_full', '')}"
              for _, r in sm.iterrows()]
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Energy per attention call  (mJ)")
    ax.grid(True, axis="y", alpha=0.3)
    # Headroom for the "flash = X mJ" label above each bar.
    visible_max = float(max(np.max(full_vals), np.max(qkv_vals)))
    ax.set_ylim(0, visible_max * 1.18)
    # Legend OUTSIDE the axes (right side) so it never covers data.
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, frameon=True)
    ax.set_title(
        f"Attention decomposition (G11 / P1.4) — {gpu}\n"
        f"residual ≥ 0  →  stacked bar : matmul-baseline + softmax-residual.   "
        f"residual < 0  →  flash beats baseline ; bar height = actual flash, "
        f"dashed line = matmul-baseline reference.",
        fontsize=10)

    ax_caveat.text(
        0.5, 0.5,
        "CAVEAT : `attention_qkv_matmul` baseline computes Q@Kᵀ then "
        "(Q@Kᵀ)@V using `torch.matmul` — NOT a fused matmul-pair. The "
        "kernel-launch + intermediate-write overhead of the 2-call "
        "baseline is therefore PRESENT in the baseline but ABSENT in the "
        "fused flash kernel. When residual is NEGATIVE this overhead "
        "exceeds the streaming-softmax cost — the 'savings' you see are "
        "really 'matmul-pair launch overhead − fused softmax cost', not a "
        "lower bound on softmax energy. README §3.7.6 / TestCases A.5.",
        ha="center", va="center", fontsize=9, color="#333", wrap=True,
        bbox=dict(facecolor="#fff2cc", edgecolor="#d6a800", pad=6))
    _save_fig(fig, out_png)
    return True


def plot_llm_matmul(df: pd.DataFrame, out_dir: Path, stem: str, gpu: str) -> None:
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
    palette = PALETTE_LLM_PRESETS
    # Pre-compute per-preset arrays once, then re-use across both panels.
    preset_data: dict[str, dict] = {}
    for preset in presets:
        g = llm[llm["llm_preset"] == preset].sort_values("load_value")
        if g.empty:
            continue
        T = g["load_value"].to_numpy(float)
        jpf = pd.to_numeric(g["j_per_flop_dyn"], errors="coerce").to_numpy(float)
        dyn = pd.to_numeric(g["dyn_energy_j"], errors="coerce").to_numpy(float)
        iters = pd.to_numeric(g.get("iters", pd.Series(dtype=float)),
                              errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_per_forward = np.where(iters > 0, dyn / iters, dyn)
        preset_data[preset] = dict(T=T, jpf=jpf, e_per_forward=e_per_forward,
                                   shape=str(g["shape"].iloc[0]))

    shape_lines = [f"{p}:{d['shape']}" for p, d in preset_data.items()]
    suptitle_shapes = ", ".join(shape_lines)

    # --- Panel A : J/FLOP vs T ---
    fig_a, ax_a = plt.subplots(figsize=(15, 7))
    for preset, d in preset_data.items():
        colour = palette.get(preset, None)
        mJ = d["jpf"] > 0
        if mJ.any():
            ax_a.plot(d["T"][mJ], d["jpf"][mJ], marker="o", color=colour, label=preset)
            for t, y in zip(d["T"][mJ], d["jpf"][mJ]):
                ax_a.annotate(f"T={int(t)}", (t, y),
                              textcoords="offset points", xytext=(5, 5),
                              fontsize=8, color=colour, alpha=0.9)
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlabel("token count T  (= M dim)")
    ax_a.set_ylabel("J / FLOP  (dynamic)")
    ax_a.legend(fontsize=9, ncol=2, loc="best")
    ax_a.set_title(f"LLM-shape matmul — Per-FLOP energy vs T — {gpu}\n"
                   f"flat = BW-bound, rising = compute-bound. Shapes: {suptitle_shapes}",
                   fontsize=10)
    lo, hi = ax_a.get_ylim()
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
        ax_a.set_ylim(lo, hi * 4)
    _save_fig(fig_a, out_dir / f"{stem}_01_powermodel_llm_jperflop.png")

    # --- Panel B : per-call energy (mJ) vs T ---
    fig_b, ax_b = plt.subplots(figsize=(15, 7))
    for preset, d in preset_data.items():
        colour = palette.get(preset, None)
        mE = d["e_per_forward"] > 0
        if mE.any():
            ax_b.plot(d["T"][mE], d["e_per_forward"][mE], marker="o",
                      color=colour, label=preset)
            for t, y in zip(d["T"][mE], d["e_per_forward"][mE]):
                ax_b.annotate(f"T={int(t)}\n{y*1e3:.2f} mJ", (t, y),
                              textcoords="offset points", xytext=(5, 5),
                              fontsize=7, color=colour, alpha=0.9)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlabel("token count T  (= M dim)")
    ax_b.set_ylabel("J per forward pass of this layer")
    ax_b.legend(fontsize=9, ncol=2, loc="best")
    ax_b.set_title(f"LLM-shape matmul — per-call energy vs T — {gpu}\n"
                   f"(annotated in mJ). Shapes: {suptitle_shapes}",
                   fontsize=10)
    lo, hi = ax_b.get_ylim()
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
        ax_b.set_ylim(lo, hi * 4)
    _save_fig(fig_b, out_dir / f"{stem}_01_powermodel_llm_per_call.png")


def plot_dram_energy(df: pd.DataFrame, out_dir: Path, stem: str, gpu: str,
                     hbm_peak_gbps: float | None = None) -> None:
    """pJ/bit + achieved BW per cell, sliced by cache_regime.

    Two panels:
      A : pJ/bit per cell, x = cache_regime (l2_hit_100 → l2_hit_0).
          Each marker is one cell, colored by op. l2_hit_0 cluster is
          the "DRAM cost" — the interesting number. l2_hit_100 cluster
          is the "L2 cost" — it should be ~5–10× lower.
          Horizontal dashed lines mark literature HBM reference points.
      B : achieved sustained BW (GB/s) at l2_hit_0, by op + dtype.
          Reveals whether each kernel is truly BW-bound (close to HBM
          peak) or compute-bound (well below peak). HBM peak (if known)
          drawn as a dashed line.
    """
    if "pj_per_bit_traffic" not in df.columns:
        return
    ew = df[(df.get("category") == "elementwise")].copy()
    if ew.empty or ew["pj_per_bit_traffic"].notna().sum() == 0:
        return
    plt = _get_mpl()
    regime_order = list(REGIME_ORDER)
    ew["cache_regime"] = ew["cache_regime"].replace(LEGACY_REGIME_MAP)
    ew = ew[ew["cache_regime"].isin(regime_order)]
    if ew.empty:
        return
    ew["pj_per_bit_traffic"] = pd.to_numeric(ew["pj_per_bit_traffic"], errors="coerce")
    ew["achieved_bw_gbps"]   = pd.to_numeric(ew["achieved_bw_gbps"], errors="coerce")
    ew = ew[ew["pj_per_bit_traffic"] > 0]
    if ew.empty:
        return

    palette_cmap = plt.get_cmap("tab10")
    ops = sorted(ew["op"].unique())
    op_color = {op: palette_cmap(i % 10) for i, op in enumerate(ops)}
    regime_x = {r: i for i, r in enumerate(regime_order)}

    # --- Panel A : pJ/bit strip, regime on x ---
    fig_a, ax_a = plt.subplots(figsize=(15, 7))
    for op in ops:
        for dt, marker in (("fp16", "o"), ("fp8", "s"),
                           ("bf16", "^"), ("fp32", "D"), ("tf32", "v")):
            g = ew[(ew["op"] == op) & (ew["dtype"] == dt)]
            if g.empty:
                continue
            xs = g["cache_regime"].map(regime_x).astype(float) \
                + (0.04 * (hash(op + dt) % 7 - 3))
            ys = g["pj_per_bit_traffic"]
            ax_a.scatter(xs, ys, s=44, color=op_color[op], marker=marker,
                         alpha=0.85, edgecolors="white", label=f"{op} {dt}")
    # Reference lines (only meaningful in the DRAM-streaming regime — but
    # we draw them across the whole panel for context). Two lines from
    # DRAM_REFERENCES_PJBIT can share a y-value (HBM2 and DDR4 are both
    # 7.0 pJ/bit), so we stagger labels horizontally in 3 columns and
    # paint a white background bbox so the dashed line doesn't bleed
    # through the glyphs.
    ref_colors = ["#888888", "#666666", "#444444", "#aa5555", "#55aaaa"]
    n_regimes = len(regime_order)
    label_xs = [n_regimes - 1 + 0.05,    # just right of last regime tick
                -0.55,                   # well left of first regime tick
                (n_regimes - 1) / 2.0]   # middle, between regime ticks
    label_has = ["left", "right", "center"]
    label_bbox = dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85)
    for i, ((label, val), c) in enumerate(
            zip(DRAM_REFERENCES_PJBIT.items(), ref_colors)):
        ax_a.axhline(val, color=c, ls="--", lw=1, alpha=0.7)
        col = i % len(label_xs)
        ax_a.text(label_xs[col], val, f" {label} ≈ {val} pJ/bit ",
                  fontsize=8, color=c,
                  va="center", ha=label_has[col],
                  bbox=label_bbox)
    ax_a.set_xticks(list(regime_x.values()))
    ax_a.set_xticklabels([f"{r}\n({REGIME_HIT_PCT[r]} L2 hit)" for r in regime_order],
                         fontsize=11)
    # Linear scale (was log). At log scale the literature reference lines
    # crowd into a narrow band near the bottom of the panel and the
    # measured points spread thinly, which made the comparison hard to
    # read. Linear gives the eye a fair side-by-side; outlier high-FLOP
    # ops (gelu / softmax / layernorm) just push the upper bound a bit.
    ax_a.set_yscale("linear")
    # Pad x range so the left/right label columns aren't clipped.
    ax_a.set_xlim(-0.85, n_regimes - 1 + 1.05)
    ax_a.set_ylabel("pJ / bit  (working-set traffic)")
    ax_a.set_title(f"DRAM energy — per-cell pJ/bit by cache regime — {gpu}\n"
                   "l2_hit_0 ≈ DRAM cost; dashed = literature reference (full stack)\n"
                   "marginal pJ/bit (l2_hit_0 − l2_hit_100) is the cleaner DRAM-only "
                   "estimate — see dram_energy_marginal.png and README §3.5")
    ax_a.grid(True, axis="y", alpha=0.3)
    ax_a.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save_fig(fig_a, out_dir / f"{stem}_02_dram_energy_pjbit.png")

    # --- Panel B : achieved BW at l2_hit_0 ---
    drm = ew[ew["cache_regime"] == "l2_hit_0"]
    if drm.empty:
        return
    agg = drm.groupby(["op", "dtype"]).agg(
        bw_med=("achieved_bw_gbps", "median"),
        n=("achieved_bw_gbps", "size"),
    ).reset_index()
    agg = agg.sort_values(["op", "dtype"]).reset_index(drop=True)
    fig_b, ax_b = plt.subplots(figsize=(12, 7))
    xs = np.arange(len(agg))
    bars = ax_b.bar(xs, agg["bw_med"], color=[op_color[o] for o in agg["op"]],
                    alpha=0.9, edgecolor="white")
    for rect, v in zip(bars, agg["bw_med"]):
        if not np.isnan(v):
            ax_b.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                      f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"{r['op']}\n{r['dtype']}" for _, r in agg.iterrows()],
                         rotation=0, fontsize=10)
    ax_b.set_ylabel("achieved sustained BW  (GB/s)")
    ax_b.set_title(f"DRAM-streaming sustained BW per kernel — {gpu}  "
                   "(median across N at l2_hit_0)")
    ax_b.grid(True, axis="y", alpha=0.3)
    if hbm_peak_gbps:
        ax_b.axhline(hbm_peak_gbps, color="#d62728", ls="--", lw=1.2,
                     label=f"HBM peak ≈ {hbm_peak_gbps:.0f} GB/s")
        ax_b.legend(fontsize=9)
    _save_fig(fig_b, out_dir / f"{stem}_02_dram_energy_bw.png")


# ---------------------------------------------------------------------------
# DRAM read/write split + marginal cost (added in PR #30)
# ---------------------------------------------------------------------------

def compute_dram_rw_split(df: pd.DataFrame) -> pd.DataFrame:
    """For each dtype at l2_hit_0, derive **separate** read and write
    pJ/bit from the stream_read / stream_write probes, plus the mixed
    measurements (stream_copy / stream_scale / stream_triad) as cross-
    check.

    Returns one row per (dtype, op) carrying:
      pj_per_bit_med    — median over the swept N values
      n_cells           — how many cells contributed
      r_per_call        — number of reads per kernel call
      w_per_call        — number of writes per kernel call
      role              — "READ", "WRITE", "MIXED" — for plot grouping

    Also computes the "implied" pJ/bit for each MIXED kernel as
    `(r·R + w·W)/(r+w)` and returns it as `pj_per_bit_implied` so the
    user can eyeball the read/write decomposition's self-consistency.
    """
    if "pj_per_bit_traffic" not in df.columns:
        return pd.DataFrame()
    drm = df[(df.get("category") == "elementwise") &
             (df["op"].isin(STREAM_RW_RATIO))].copy()
    if drm.empty:
        return pd.DataFrame()
    drm["cache_regime"] = drm["cache_regime"].replace(LEGACY_REGIME_MAP)
    drm = drm[drm["cache_regime"] == "l2_hit_0"]
    drm["pj_per_bit_traffic"] = pd.to_numeric(drm["pj_per_bit_traffic"],
                                              errors="coerce")
    drm = drm[drm["pj_per_bit_traffic"] > 0]
    if drm.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    # Separate cache for read / write per dtype so we can compute the
    # MIXED-kernel implied number from the same dtype's measurements.
    pure: dict[tuple[str, str], float] = {}   # (dtype, "read"|"write") → pJ/bit
    for (dt, op), g in drm.groupby(["dtype", "op"]):
        med = float(g["pj_per_bit_traffic"].median())
        r, w = STREAM_RW_RATIO[op]
        if (r, w) == (1, 0):
            role = "READ";  pure[(dt, "read")]  = med
        elif (r, w) == (0, 1):
            role = "WRITE"; pure[(dt, "write")] = med
        else:
            role = "MIXED"
        rows.append(dict(dtype=dt, op=op, role=role,
                         r_per_call=r, w_per_call=w,
                         pj_per_bit_med=med, n_cells=len(g)))
    out = pd.DataFrame(rows)
    # Add implied pJ/bit for MIXED rows.
    def _implied(row):
        if row["role"] != "MIXED":
            return float("nan")
        R = pure.get((row["dtype"], "read"))
        W = pure.get((row["dtype"], "write"))
        if R is None or W is None:
            return float("nan")
        r, w = row["r_per_call"], row["w_per_call"]
        return (r * R + w * W) / (r + w)
    out["pj_per_bit_implied"] = out.apply(_implied, axis=1)
    out["implied_error_pct"] = (out["pj_per_bit_med"] - out["pj_per_bit_implied"]) \
                                / out["pj_per_bit_implied"] * 100.0
    return out.sort_values(["dtype", "role", "op"]).reset_index(drop=True)


def compute_dram_marginal(df: pd.DataFrame) -> pd.DataFrame:
    """Marginal DRAM cost — the extra pJ/bit a kernel pays when its
    working set spills out of L2 into HBM. For each (op, dtype) that has
    cells in BOTH the l2_hit_100 and l2_hit_0 regimes we report:

        J_per_byte(l2_hit_100)        # baseline: SM compute + L2 traffic
        J_per_byte(l2_hit_0)          # SM compute + L2 transit + HBM stream
        marginal_pJ_per_bit = (J_dram - J_l2) × 1e12 / 8

    The marginal number is closer to the *literature* "DRAM-only"
    energy because the on-chip components (SM compute, L2 lookup, NoC
    routing) cancel between the two regimes. README §3.5.

    Cells where dyn_energy_j was clipped to 0 (P_static drift) are
    excluded — they'd silently set the L2 baseline to 0 and inflate
    the marginal.
    """
    if ("pj_per_bit_traffic" not in df.columns
            or "bytes_traffic" not in df.columns):
        return pd.DataFrame()
    ew = df[df.get("category") == "elementwise"].copy()
    if ew.empty:
        return pd.DataFrame()
    ew["cache_regime"] = ew["cache_regime"].replace(LEGACY_REGIME_MAP)
    ew["dyn_energy_j"] = pd.to_numeric(ew["dyn_energy_j"], errors="coerce")
    ew["bytes_traffic"] = pd.to_numeric(ew["bytes_traffic"], errors="coerce")
    ew = ew[(ew["dyn_energy_j"] > 0) & (ew["bytes_traffic"] > 0)]
    if ew.empty:
        return pd.DataFrame()
    ew["j_per_byte"] = ew["dyn_energy_j"] / ew["bytes_traffic"]

    rows: list[dict] = []
    for (op, dt), g in ew.groupby(["op", "dtype"]):
        l2 = g[g["cache_regime"] == "l2_hit_100"]["j_per_byte"]
        dr = g[g["cache_regime"] == "l2_hit_0"]["j_per_byte"]
        if l2.empty or dr.empty:
            continue
        l2_med = float(l2.median()); dr_med = float(dr.median())
        delta = dr_med - l2_med
        rows.append(dict(
            op=op, dtype=dt,
            l2_J_per_byte=l2_med,
            dram_J_per_byte=dr_med,
            marginal_pJ_per_bit=delta * 1e12 / 8.0,
            direct_dram_pJ_per_bit=dr_med * 1e12 / 8.0,
            n_l2_cells=len(l2),
            n_dram_cells=len(dr),
        ))
    return pd.DataFrame(rows).sort_values(["op", "dtype"]).reset_index(drop=True)


def plot_dram_rw_split(rw_df: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """Bar chart: read vs write vs mixed pJ/bit per dtype (l2_hit_0).

    Reference dashed lines: the literature HBM3 numbers (read ~3.0,
    write ~4.5 pJ/bit). Mixed bars show their measured value next to
    the "implied" value derived from the read/write decomposition —
    the % error tells the user how well the linear-model assumption
    (mixed = (r·R + w·W)/(r+w)) holds on this card.
    """
    if rw_df.empty:
        return
    plt = _get_mpl()
    dtypes = sorted(rw_df["dtype"].unique())
    op_order = ["stream_read", "stream_write",
                "stream_copy", "stream_scale", "stream_triad"]
    role_color = {"READ": "#1f77b4", "WRITE": "#d62728", "MIXED": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(15, 7))
    bar_w = 0.8 / max(1, len(dtypes))
    xs = np.arange(len(op_order))
    for i, dt in enumerate(dtypes):
        sub = rw_df[rw_df["dtype"] == dt].set_index("op")
        vals = [float(sub.loc[op, "pj_per_bit_med"]) if op in sub.index else float("nan")
                for op in op_order]
        roles = [str(sub.loc[op, "role"]) if op in sub.index else "" for op in op_order]
        offset = (i - (len(dtypes) - 1) / 2) * bar_w
        bars = ax.bar(xs + offset, vals, bar_w,
                      color=[role_color.get(r, "#999") for r in roles],
                      edgecolor="white", alpha=0.9, label=dt,
                      hatch=["", "", "//", "//", "//"])
        for rect, v, op in zip(bars, vals, op_order):
            if not np.isfinite(v):
                continue
            row = sub.loc[op] if op in sub.index else None
            txt = f"{v:.2f}"
            if row is not None and pd.notna(row.get("pj_per_bit_implied")):
                imp = row["pj_per_bit_implied"]; err = row["implied_error_pct"]
                txt += f"\n(impl {imp:.2f}, Δ{err:+.0f}%)"
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    txt, ha="center", va="bottom", fontsize=8, linespacing=1.05)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{op}\n({STREAM_RW_RATIO[op][0]}R+{STREAM_RW_RATIO[op][1]}W)"
                        for op in op_order], fontsize=10)
    ax.set_ylabel("pJ / bit  (l2_hit_0, board-level)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    # Reference lines.
    ax.axhline(3.0, color="#1f77b4", ls="--", lw=1, alpha=0.6)
    ax.text(len(op_order) - 0.5, 3.0, " HBM3 read ≈ 3.0", fontsize=8,
            color="#1f77b4", va="center")
    ax.axhline(4.5, color="#d62728", ls="--", lw=1, alpha=0.6)
    ax.text(len(op_order) - 0.5, 4.5, " HBM3 write ≈ 4.5", fontsize=8,
            color="#d62728", va="center")
    ax.axhline(3.9, color="#444444", ls=":", lw=1, alpha=0.6)
    ax.text(len(op_order) - 0.5, 3.9, " HBM3 avg ≈ 3.9", fontsize=8,
            color="#444444", va="center")
    ax.set_title(f"DRAM read vs write energy — {gpu}\n"
                 "Blue=read-only / Red=write-only / Grey=mixed "
                 "(implied = (r·R+w·W)/(r+w))", fontsize=11)
    ax.legend(title="dtype", fontsize=9, loc="upper left")
    _save_fig(fig, out_png)


def plot_dram_marginal(marg_df: pd.DataFrame, out_png: Path, gpu: str) -> None:
    """Bar chart contrasting two DRAM-cost interpretations:

      direct  = pJ/bit at l2_hit_0 (= board-level full-stack)
      marginal = (J_dram − J_l2) per byte → cancels SM-compute and L2
                  routing; closer to "DRAM cells + PHY + controller"
                  literature definition

    Marginal is usually 1–2 pJ/bit *lower* than direct because the SM
    compute and L2 transit baseline are subtracted. If marginal is
    *negative* on some op, P_static is wrong (the l2_hit_100 measurement
    is artificially high, pushing direct-l2 above direct-dram).
    """
    if marg_df.empty:
        return
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(15, 7))
    keys = [f"{r['op']}·{r['dtype']}" for _, r in marg_df.iterrows()]
    xs = np.arange(len(keys))
    w = 0.4
    bars_d = ax.bar(xs - w/2, marg_df["direct_dram_pJ_per_bit"], w,
                    color="#999999", alpha=0.85, edgecolor="white",
                    label="direct (full stack at l2_hit_0)")
    bars_m = ax.bar(xs + w/2, marg_df["marginal_pJ_per_bit"], w,
                    color="#1f77b4", alpha=0.9, edgecolor="white",
                    label="marginal (DRAM-only ≈ direct − l2_hit_100)")
    for rect, v in zip(bars_d, marg_df["direct_dram_pJ_per_bit"]):
        if pd.notna(v):
            ax.text(rect.get_x()+rect.get_width()/2, rect.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for rect, v in zip(bars_m, marg_df["marginal_pJ_per_bit"]):
        if pd.notna(v):
            colour = "#d62728" if v < 0 else "black"
            ax.text(rect.get_x()+rect.get_width()/2, rect.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                    color=colour, fontweight="bold" if v < 0 else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("pJ / bit")
    # Reference lines for the marginal interpretation. The full literature
    # set spans HBM2 (~7) → HBM2E (~5) → HBM3 (~3.9) → DRAM core (~2.5).
    # Show the band that brackets typical Hopper/Ampere measurements so a
    # reader can immediately see whether their marginal lands in the
    # expected window for THEIR HBM generation.
    ax.axhline(5.0, color="#aa5555", ls="--", lw=1, alpha=0.7)
    ax.text(len(keys) - 0.5, 5.0, " HBM2E (A100) ≈ 5.0", fontsize=8,
            color="#aa5555", va="center")
    ax.axhline(3.9, color="#444444", ls="--", lw=1, alpha=0.7)
    ax.text(len(keys) - 0.5, 3.9, " HBM3 (H100) ≈ 3.9", fontsize=8,
            color="#444444", va="center")
    ax.axhline(2.5, color="#888888", ls=":", lw=1, alpha=0.7)
    ax.text(len(keys) - 0.5, 2.5, " Horowitz '14 DRAM core ≈ 2.5", fontsize=8,
            color="#888888", va="center")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"DRAM energy: direct vs marginal — {gpu}\n"
                 "marginal cancels SM/L2 baseline → closer to literature DRAM-stack value\n"
                 "(red, bold values: marginal < 0 — likely P_static drift problem)",
                 fontsize=11)
    ax.legend(loc="upper left")
    _save_fig(fig, out_png)


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


# ===========================================================================
# Section 7 — CLI / main
# ===========================================================================
# `_resolve_csv()` accepts either a positional CSV path or a
# `--reports-dir` + `--tag` pair (auto-discovers the latest matching
# `gpu_power_bench_*_<tag>.csv`). `main()` orchestrates: load → augment
# (traffic metrics) → summarise (per-cell + per-regime) → plot all.
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
    # sidecar CSVs so the user doesn't accidentally analyze the wrong file.
    # Two families to filter:
    #   gpu_power_bench.py writes : _baseline / _baseline_stats / _samples
    #                               / _summary / _rebaseline
    #   analyze.py writes back   : _summary / _summary_by_regime
    #                               / _dram_rw_split / _dram_marginal
    # Without this, the most-recent-mtime tiebreak eventually picks one of
    # analyze.py's own outputs (next re-run after a successful run drops
    # in newer-mtime sidecars) and crashes with KeyError.
    SIDECAR_SUFFIXES = (
        "_baseline.csv", "_baseline_stats.csv",
        "_samples.csv", "_summary.csv", "_rebaseline.csv",
        "_summary_by_regime.csv",
        "_dram_rw_split.csv", "_dram_marginal.csv",
    )
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
            if any(name.endswith(s) for s in SIDECAR_SUFFIXES):
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
    # Augment with derived DRAM-energy metrics (cheap; later filters reuse
    # the same df). For non-elementwise rows the new columns stay NaN.
    df = add_traffic_metrics(df)
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
    # Defensive: gpu column is written by current gpu_power_bench.py but
    # missing in (a) very old CSVs, (b) sidecars that slipped past the
    # _resolve_csv filter. Fall back to a slug parsed from the filename
    # so we still produce plots instead of crashing.
    if "gpu" in df.columns and not df["gpu"].empty:
        gpu = df["gpu"].iloc[0]
    else:
        # Filename pattern: gpu_power_bench_<slug>_<ts>[_<tag>].csv
        stem_parts = args.csv.stem.split("_")
        gpu = "_".join(stem_parts[3:5]) if len(stem_parts) >= 5 else args.csv.stem
        print(f"[warn] CSV has no 'gpu' column — derived from filename: {gpu!r}. "
              f"Re-run gpu_power_bench.py if you want the proper GPU name "
              f"(this CSV is likely from an older sweep or a sidecar).")
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
                "slope_dyn", "slope_dyn_wls",
                "slope_dyn_ci_lo", "slope_dyn_ci_hi",
                "R2_dyn_wls", "clip_bias_pct",
                "mean_dyn_power_w", "mean_temp_c"]
        # Older summaries may not have the new columns; show what's there.
        cols = [c for c in cols if c in summary.columns]
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

    # Per-K matmul k_op sidecar (P1.2 / G3) — exposes Tensor Core
    # efficiency curve that single-slope summaries hide.
    matmul_per_K = summarize_matmul_per_K(df)
    if not matmul_per_K.empty:
        mm_per_K_path = out_dir / f"{stem}_summary_matmul_per_K.csv"
        matmul_per_K.to_csv(mm_per_K_path, index=False)
        print(f"[save] {mm_per_K_path}")
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
    # Coefficient bar — split into two full-width PNGs (elementwise / matmul)
    # so neither x-axis is cramped against the other panel.
    # Pass by_regime so the elementwise headline bar can use the
    # l2_hit_0 regime slope (R² collapse fix — see plot_joule_per_op_bar
    # docstring). Matmul still uses the cross-K summary (compute-bound,
    # tile reuse keeps within-K linearity).
    plot_joule_per_op_bar(plot_summary, out_dir, stem, gpu,
                          by_regime=by_regime)
    # Filter the by-regime summary in the same way (elementwise fp8 hidden
    # by default) so annotations on the plot stay consistent with the
    # other plots. Matmul rows are kept either way.
    plot_by_regime = by_regime
    if not args.include_emulated and not by_regime.empty:
        plot_by_regime = by_regime[
            (by_regime["emulated"].astype(int) == 0)
            | (by_regime["category"] == "matmul")
        ].copy()
    # Cache-regime — 6 separate per-panel PNGs (per-cell strip / k_op bar /
    # mean dyn power × elementwise / matmul) so x-axis stays readable.
    # Pass the UNFILTERED by_regime so the fp8-dedicated panel (added in
    # PR #55) always shows emulated rows, regardless of --include-emulated.
    plot_cache_regime(plot_df, plot_by_regime, out_dir, stem, gpu,
                      by_regime_unfiltered=by_regime)
    # MECE energy decomposition — break dyn_energy at l2_hit_0 into
    # resident workload / fp8 cast overhead / DRAM round-trip. Uses the
    # UNFILTERED by_regime so fp8 rows participate.
    plot_energy_decomposition(
        by_regime,
        out_dir / f"{stem}_03_energy_decomposition_mece.png", gpu)
    # MECE for matmul (P2.1 / G4) — 2 components (no fp8 cast term;
    # matmul fp8_te is native on H100, FP16-fallback on A100).
    plot_energy_decomposition_matmul(
        by_regime,
        out_dir / f"{stem}_03_energy_decomposition_matmul_mece.png", gpu)
    # Per-K J/FLOP curve for every matmul variant (P1.2 / G3) — exposes
    # Tensor Core efficiency ramp that single-slope summary hides.
    if not matmul_per_K.empty:
        plot_kop_per_K(
            matmul_per_K,
            out_dir / f"{stem}_01_powermodel_kop_per_K.png", gpu)
    # Fused-vs-Standalone decomposition (G11 / P1.4) — fires only when
    # the sweep used --include-fused. Pairs full ↔ baseline, computes
    # residual energy attributable to softmax / gelu / layernorm INSIDE
    # the fused kernel, compares to standalone J/elem at l2_hit_0.
    fused_decomp = summarize_fused_decomposition(df)
    if not fused_decomp.empty:
        decomp_path = out_dir / f"{stem}_fused_decomposition.csv"
        fused_decomp.to_csv(decomp_path, index=False)
        print(f"[save] {decomp_path}")
        plot_fused_vs_standalone_bar(fused_decomp,
            out_dir / f"{stem}_03_fused_vs_standalone_bar.png", gpu)
        plot_attention_decomposition(fused_decomp, df,
            out_dir / f"{stem}_03_attention_decomposition.png", gpu)
        # Tidy console summary so user sees ratios without opening CSV.
        print("\n== Fused-vs-Standalone decomposition (G11 / P1.4) ==")
        with pd.option_context("display.width", 160,
                               "display.float_format", lambda v: f"{v:.3e}"):
            cols = ["op_group", "dtype",
                    "j_per_element_residual", "j_per_element_standalone",
                    "ratio_residual_to_standalone",
                    "residual_pct_of_full", "stat_significant",
                    "fusion_emulated"]
            print(fused_decomp[cols].to_string(index=False))
    # DRAM-bandwidth energy — 2 separate PNGs (pJ/bit strip + sustained BW).
    plot_dram_energy(plot_df, out_dir, stem, gpu)
    # DRAM read/write split (only fires when stream_read / stream_write
    # probes are present — i.e. user ran --dram-bw-test).
    rw_df = compute_dram_rw_split(plot_df)
    if not rw_df.empty:
        rw_path = out_dir / f"{stem}_dram_rw_split.csv"
        rw_df.to_csv(rw_path, index=False)
        print(f"[save] {rw_path}")
        plot_dram_rw_split(rw_df,
            out_dir / f"{stem}_02_dram_energy_rw_split.png", gpu)
        # Also print a tidy summary so the user sees R / W numbers without
        # opening the CSV.
        print("\n== DRAM read vs write energy (l2_hit_0, pJ/bit) ==")
        with pd.option_context("display.width", 160,
                               "display.float_format", lambda v: f"{v:.3f}"):
            cols = ["dtype", "op", "role", "r_per_call", "w_per_call",
                    "n_cells", "pj_per_bit_med", "pj_per_bit_implied",
                    "implied_error_pct"]
            print(rw_df[cols].to_string(index=False))
    # DRAM marginal cost (l2_hit_0 minus l2_hit_100) — works on any sweep
    # that has cells in both regimes (no extra probe needed).
    marg_df = compute_dram_marginal(plot_df)
    if not marg_df.empty:
        marg_path = out_dir / f"{stem}_dram_marginal.csv"
        marg_df.to_csv(marg_path, index=False)
        print(f"[save] {marg_path}")
        plot_dram_marginal(marg_df,
            out_dir / f"{stem}_02_dram_energy_marginal.png", gpu)
        print("\n== DRAM marginal cost (direct vs marginal pJ/bit) ==")
        with pd.option_context("display.width", 160,
                               "display.float_format", lambda v: f"{v:.3f}"):
            cols = ["op", "dtype", "n_l2_cells", "n_dram_cells",
                    "direct_dram_pJ_per_bit", "marginal_pJ_per_bit"]
            print(marg_df[cols].to_string(index=False))
        neg = marg_df[marg_df["marginal_pJ_per_bit"] < 0]
        if not neg.empty:
            print(f"\n[warn] {len(neg)} kernel(s) have NEGATIVE marginal — "
                  f"P_static is probably wrong (l2_hit_100 measurement "
                  f"sits above l2_hit_0 in raw J). Re-run with "
                  f"--rebaseline-every 20.")
    # LLM-shape matmul — 2 separate PNGs (J/FLOP-vs-T + per-call energy).
    plot_llm_matmul(plot_df, out_dir, stem, gpu)

    # --- console summary: pJ/bit at l2_hit_0 (DRAM streaming) -------------
    if "pj_per_bit_traffic" in df.columns:
        dram = df[(df.get("category") == "elementwise")
                  & (df["cache_regime"].replace({
                      "l2_resident":"l2_hit_100","l2_partial":"l2_hit_50",
                      "dram_stream":"l2_hit_0"}) == "l2_hit_0")]
        if not dram.empty and dram["pj_per_bit_traffic"].notna().any():
            print("\n== DRAM-streaming energy (l2_hit_0 cells) ==")
            print("    Per-kernel pJ/bit (board-level — includes HBM cells, "
                  "PHY, controller, on-chip routing):")
            with pd.option_context("display.width", 160,
                                   "display.float_format", lambda v: f"{v:.2f}"):
                tbl = (dram.groupby(["op", "dtype"])
                          .agg(n=("pj_per_bit_traffic", "size"),
                               pj_per_bit_med=("pj_per_bit_traffic", "median"),
                               bw_gbps_med=("achieved_bw_gbps", "median"))
                          .reset_index())
                print(tbl.to_string(index=False))
            print("    Reference (full stack): HBM2 ≈ 7, HBM2E ≈ 5, HBM3 ≈ 3.9 "
                  "pJ/bit. Our boundary is wider (also captures L2 → HBM "
                  "transit + idle controller overhead) so values 1.5–3× "
                  "above the reference are normal.")

    # Static-power diagnostics (auto-discover the baseline sidecar).
    if args.baseline is None:
        cand = args.csv.with_name(stem + "_baseline.csv")
        if cand.exists():
            args.baseline = cand
    plot_static_power(plot_df, args.baseline,
                      out_dir / f"{stem}_03_baseline_static_power.png", gpu)
    # P_static drift vs temperature scatter (P2.3 / G8) — auto-detect
    # the rebaseline sidecar; no-op if --rebaseline-every wasn't used.
    rebaseline_cand = args.csv.with_name(stem + "_rebaseline.csv")
    if rebaseline_cand.exists():
        plot_pstatic_drift_vs_temp(
            rebaseline_cand, plot_df,
            out_dir / f"{stem}_03_baseline_pstatic_vs_temp.png", gpu)
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
