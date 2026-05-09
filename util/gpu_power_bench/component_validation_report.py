#!/usr/bin/env python3
"""Build multi-GPU component-energy validation reports.

This is intentionally a post-processing tool. It does not run benchmarks and
does not change the measurement path. It consumes gpu_power_bench CSVs and
analysis sidecars, then emits the cross-GPU tables/plots needed to check that
RTX 3090, A100 SXM, H100 SXM, and H100 PCIe results are interpreted with the
right memory, L2, power-envelope, and FP8 assumptions.

Tables are written to --out-dir. PNGs are written to --image-dir with
category-numbered filenames so reports stay stable across runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import analyze
import gpu_profiles as gp


SIDECAR_SUFFIXES = (
    "_baseline.csv",
    "_baseline_stats.csv",
    "_samples.csv",
    "_summary.csv",
    "_summary_by_regime.csv",
    "_summary_matmul_per_K.csv",
    "_dram_rw_split.csv",
    "_dram_marginal.csv",
    "_rebaseline.csv",
    "_plot_skip_reasons.csv",
    "_soc_summary.csv",
    "_soc_timeseries.csv",
    "_gpu_spec_snapshot.csv",
    "_fused_decomposition.csv",
    "_02_l2_summary.csv",
    "_02_l2_per_window.csv",
    "_02_l2_fit_points.csv",
    "_02_l2_validation_summary.csv",
    "_02_l2_skip_reasons.csv",
    "_02_l2_refill_summary.csv",
    "_02_l2_refill_fit_points.csv",
    "_02_l2_refill_validation_summary.csv",
    "_02_l2_refill_skip_reasons.csv",
)


GPU_PROFILES = gp.GPU_PROFILES


COVERAGE_COMPONENTS = (
    "static_idle",
    "thermal_leakage",
    "fp32_simt_compute",
    "tf32_tensor_core",
    "fp16_tensor_core",
    "bf16_tensor_core",
    "native_fp8_tensor_core",
    "memory_read_write",
    "l2_hit_path",
    "hbm_to_l2_refill_path",
    "soc_hbm_phy_to_l2_proxy",
    "standalone_nonlinear",
    "fused_nonlinear",
)


def is_main_benchmark_csv(path: Path) -> bool:
    if not path.name.startswith("gpu_power_bench_") or path.suffix != ".csv":
        return False
    return not any(path.name.endswith(s) for s in SIDECAR_SUFFIXES)


def find_benchmark_csvs(reports_dir: Path, tags: list[str] | None) -> list[Path]:
    paths = [p for p in reports_dir.rglob("gpu_power_bench_*.csv") if is_main_benchmark_csv(p)]
    if tags:
        lowered = [t.lower() for t in tags]
        paths = [p for p in paths if any(t in p.name.lower() for t in lowered)]
    return sorted(paths)


def infer_profile(*texts: object) -> str:
    return gp.infer_gpu_profile(*texts)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_key_value_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with path.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    out[str(row[0])] = str(row[1])
    except Exception:
        return {}
    return out


def companion(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix)


def normalize_cc(value: object) -> str:
    return gp.normalize_cc(value)


def status_for_cc(profile: str, observed: str) -> tuple[str, str]:
    return gp.profile_cc_status(profile, observed)


def load_or_compute_summary(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    existing = read_csv_safe(companion(csv_path, "_summary.csv"))
    if not existing.empty:
        return existing
    if df.empty:
        return pd.DataFrame()
    try:
        return analyze.summarize(df)
    except Exception:
        return pd.DataFrame()


def load_or_compute_dram_rw(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    existing = read_csv_safe(companion(csv_path, "_dram_rw_split.csv"))
    if not existing.empty:
        return existing
    if df.empty:
        return pd.DataFrame()
    try:
        return analyze.compute_dram_rw_split(df)
    except Exception:
        return pd.DataFrame()


def load_or_compute_l2(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    existing = read_csv_safe(companion(csv_path, "_02_l2_summary.csv"))
    if not existing.empty:
        return existing
    if df.empty:
        return pd.DataFrame()
    try:
        l2_summary, _, _, _, _ = analyze.compute_l2_energy(df)
        return l2_summary
    except Exception:
        return pd.DataFrame()


def load_or_compute_l2_refill(df: pd.DataFrame, csv_path: Path, l2_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    existing = read_csv_safe(companion(csv_path, "_02_l2_refill_summary.csv"))
    if not existing.empty:
        return existing
    if df.empty:
        return pd.DataFrame()
    try:
        refill_summary, _, _, _ = analyze.compute_l2_refill_energy(df, l2_summary=l2_summary)
        return refill_summary
    except Exception:
        return pd.DataFrame()


def build_gpu_spec_matrix(csvs: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in csvs:
        df = read_csv_safe(csv_path)
        sidecar = read_key_value_csv(companion(csv_path, "_gpu_spec_snapshot.csv"))
        gpu_name = str(df["gpu"].iloc[0]) if "gpu" in df and not df.empty else ""
        cc = normalize_cc(df["compute_cap"].iloc[0] if "compute_cap" in df and not df.empty else "")
        profile = str(sidecar.get("gpu_profile") or (
            df["gpu_profile"].iloc[0] if "gpu_profile" in df and not df.empty else "") or
            infer_profile(csv_path, gpu_name, cc))
        spec = GPU_PROFILES.get(profile, {})
        status = str(sidecar.get("gpu_profile_status") or status_for_cc(profile, cc)[0])
        reason = str(sidecar.get("gpu_profile_reason") or status_for_cc(profile, cc)[1])
        rows.append({
            "source_csv": str(csv_path),
            "gpu_profile": profile,
            "gpu_reported": gpu_name,
            "compute_cap_reported": cc,
            "expected_compute_cap": spec.get("expected_cc", ""),
            "arch": spec.get("arch", ""),
            "memory_type": spec.get("memory_type", ""),
            "memory_capacity_gb": spec.get("memory_capacity_gb", np.nan),
            "memory_total_gb": sidecar.get("memory_total_gb", ""),
            "peak_bw_gbps": spec.get("peak_bw_gbps", np.nan),
            "l2_mb": spec.get("l2_mb", np.nan),
            "l2_reported_mb": sidecar.get("l2_reported_mb", ""),
            "l2_effective_mb": sidecar.get("l2_effective_mb", ""),
            "l2_source": sidecar.get("l2_source", ""),
            "power_envelope_w": spec.get("power_envelope_w", np.nan),
            "power_limit_w": sidecar.get("power_limit_w", ""),
            "mig_mode_current": sidecar.get("mig_mode_current", ""),
            "native_fp8_headline_allowed": bool(spec.get("native_fp8", False)),
            "role": spec.get("role", ""),
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def collect_runtime_specs() -> pd.DataFrame:
    rows = []
    try:
        import torch

        for dev in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(dev)
            profile = infer_profile(props.name, f"{props.major}.{props.minor}")
            spec = GPU_PROFILES.get(profile, {})
            status, reason = status_for_cc(profile, f"{props.major}.{props.minor}")
            rows.append({
                "source": "torch",
                "device": dev,
                "gpu_profile": profile,
                "gpu_reported": props.name,
                "compute_cap_reported": f"{props.major}.{props.minor}",
                "memory_total_gb": props.total_memory / (1 << 30),
                "l2_cache_mb": getattr(props, "l2_cache_size", 0) / (1 << 20),
                "expected_memory_type": spec.get("memory_type", ""),
                "expected_l2_mb": spec.get("l2_mb", np.nan),
                "status": status,
                "reason": reason,
            })
    except Exception as e:
        rows.append({"source": "torch", "status": "WARN", "reason": str(e)})

    try:
        q = "name,memory.total,power.limit,clocks.max.sm,clocks.max.memory,mig.mode.current"
        cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            for idx, line in enumerate(proc.stdout.splitlines()):
                vals = [v.strip() for v in line.split(",")]
                while len(vals) < 6:
                    vals.append("")
                profile = infer_profile(vals[0])
                rows.append({
                    "source": "nvidia-smi",
                    "device": idx,
                    "gpu_profile": profile,
                    "gpu_reported": vals[0],
                    "memory_total_mib": vals[1],
                    "power_limit_w": vals[2],
                    "clocks_max_sm_mhz": vals[3],
                    "clocks_max_memory_mhz": vals[4],
                    "mig_mode_current": vals[5],
                    "status": "INFO",
                    "reason": "",
                })
        else:
            rows.append({"source": "nvidia-smi", "status": "WARN", "reason": proc.stderr.strip()})
    except Exception as e:
        rows.append({"source": "nvidia-smi", "status": "WARN", "reason": str(e)})
    return pd.DataFrame(rows)


def _status_from_r2(r2: object, *, min_pass: float = 0.98, min_low: float = 0.90) -> str:
    try:
        v = float(r2)
    except Exception:
        return "LOW_CONF"
    if not math.isfinite(v):
        return "LOW_CONF"
    if v >= min_pass:
        return "PASS"
    if v >= min_low:
        return "LOW_CONF"
    return "FAIL"


def _coef_row(profile: str, csv_path: Path, coefficient: str, unit: str,
              value: object, source: str, status: str, caveat: str = "",
              ci_lo: object = np.nan, ci_hi: object = np.nan, r2: object = np.nan,
              gpu: str = "", headline_eligible: object = "",
              headline_status: str = "", headline_reason: str = "") -> dict:
    return {
        "gpu_profile": profile,
        "gpu": gpu,
        "coefficient": coefficient,
        "unit": unit,
        "value": value,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "r2": r2,
        "status": status,
        "caveat": caveat,
        "headline_eligible": headline_eligible,
        "headline_status": headline_status,
        "headline_reason": headline_reason,
        "source": source,
        "source_csv": str(csv_path),
    }


def build_coefficient_table(csvs: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in csvs:
        df = read_csv_safe(csv_path)
        if not df.empty:
            df = analyze.add_traffic_metrics(df)
        gpu = str(df["gpu"].iloc[0]) if "gpu" in df and not df.empty else ""
        sidecar = read_key_value_csv(companion(csv_path, "_gpu_spec_snapshot.csv"))
        profile = str(sidecar.get("gpu_profile") or (
            df["gpu_profile"].iloc[0] if "gpu_profile" in df and not df.empty else "") or
            infer_profile(csv_path, gpu))
        observed_cc = df["compute_cap"].iloc[0] if "compute_cap" in df and not df.empty else sidecar.get("compute_cap", "")
        summary = load_or_compute_summary(df, csv_path)
        if not summary.empty:
            for _, r in summary.iterrows():
                cat = str(r.get("category", ""))
                op = str(r.get("op", ""))
                dt = str(r.get("dtype", ""))
                mode = str(r.get("mode", ""))
                variant = str(r.get("variant", ""))
                if cat in ("matmul", "matmul_llm"):
                    coeff = f"k_{variant}"
                elif cat == "elementwise":
                    coeff = f"k_standalone_{dt}_{op}"
                elif cat == "fused":
                    coeff = f"k_fused_full_{dt}_{op}"
                else:
                    coeff = f"k_{variant or cat or op}"
                status = _status_from_r2(r.get("R2_dyn_wls", np.nan))
                caveat = ""
                emulated = int(r.get("emulated", 0) or 0)
                h_eligible, h_status, h_reason = gp.headline_status(
                    profile, category=cat, op=op, dtype=dt, mode=mode,
                    emulated=bool(emulated), observed_cc=observed_cc)
                if h_status == "NOT_HEADLINE":
                    status = "NOT_HEADLINE"
                elif h_status == "PROXY" and status == "PASS":
                    status = "LOW_CONF"
                if h_reason:
                    caveat = h_reason
                if emulated:
                    caveat = (caveat + "; " if caveat else "") + "emulated path"
                rows.append(_coef_row(
                    profile, csv_path, coeff, str(r.get("fit_axis", "")),
                    r.get("slope_dyn_wls", r.get("slope_dyn", np.nan)),
                    "summary", status, caveat,
                    r.get("slope_dyn_ci_lo", np.nan),
                    r.get("slope_dyn_ci_hi", np.nan),
                    r.get("R2_dyn_wls", np.nan), gpu,
                    h_eligible, h_status, h_reason))

        rw = load_or_compute_dram_rw(df, csv_path)
        if not rw.empty:
            for _, r in rw.iterrows():
                role = str(r.get("role", r.get("op", ""))).replace(" ", "_")
                dt = str(r.get("dtype", ""))
                status = "PASS" if pd.notna(r.get("pj_per_bit_med", np.nan)) else "FAIL"
                h_eligible, h_status, h_reason = gp.headline_status(
                    profile, category="elementwise", op=f"stream_{role.lower()}",
                    dtype=dt, mode="elementwise", observed_cc=observed_cc)
                rows.append(_coef_row(
                    profile, csv_path, f"k_mem_{role}_{dt}", "pJ/bit",
                    r.get("pj_per_bit_med", np.nan), "dram_rw_split",
                    status, f"{GPU_PROFILES.get(profile, {}).get('memory_type', 'unknown')} board-level memory path",
                    gpu=gpu, headline_eligible=h_eligible,
                    headline_status=h_status, headline_reason=h_reason))

        l2 = load_or_compute_l2(df, csv_path)
        if not l2.empty:
            r = l2.iloc[0]
            for col, coeff in (
                ("k_l2_read_pj_per_bit", "k_l2_read_hit_path"),
                ("k_l2_write_pj_per_bit", "k_l2_write_hit_path"),
                ("k_l2_copy_effective_pj_per_bit", "k_l2_copy_effective_path"),
            ):
                h_eligible, h_status, h_reason = gp.headline_status(
                    profile, category="l2", op=coeff, dtype="uint32",
                    mode="l2_probe", observed_cc=observed_cc)
                rows.append(_coef_row(
                    profile, csv_path, coeff, "pJ/bit", r.get(col, np.nan),
                    "l2_summary", str(r.get("status", "LOW_CONF")),
                    "L2-hit traffic path, not isolated SRAM bit-cell energy",
                    r.get(col.replace("pj_per_bit", "ci_lo"), np.nan),
                    r.get(col.replace("pj_per_bit", "ci_hi"), np.nan),
                    gpu=gpu, headline_eligible=h_eligible,
                    headline_status=h_status, headline_reason=h_reason))

        refill = load_or_compute_l2_refill(df, csv_path, l2)
        if not refill.empty:
            r = refill.iloc[0]
            if profile.startswith("h100"):
                h_eligible, h_status, h_reason = (
                    1, "HEADLINE", "h100_cp_async_refill_path_proxy_not_pure_component")
            else:
                h_eligible, h_status, h_reason = (
                    1, "PROXY", "prefetch_global_l2_fallback_not_h100_cp_async_headline")
            rows.append(_coef_row(
                profile, csv_path, "k_hbm_to_l2_refill_path", "pJ/bit",
                r.get("k_hbm_to_l2_refill_path_pj_bit", np.nan),
                "l2_refill_summary", str(r.get("status", "LOW_CONF")),
                "cold-hot HBM -> L2 refill path proxy; not isolated HBM cell or PHY energy",
                r.get("k_hbm_to_l2_refill_ci_lo", np.nan),
                r.get("k_hbm_to_l2_refill_ci_hi", np.nan),
                r.get("r2", np.nan), gpu,
                h_eligible, h_status, h_reason))

            residual = pd.to_numeric(
                pd.Series([r.get("k_soc_hbm_phy_to_l2_fill_proxy_pj_bit", np.nan)]),
                errors="coerce").iloc[0]
            if pd.notna(residual):
                residual_status = str(r.get("status", "LOW_CONF"))
                if residual_status == "PASS":
                    residual_status = "LOW_CONF"
                rows.append(_coef_row(
                    profile, csv_path, "k_soc_hbm_phy_to_l2_fill_proxy", "pJ/bit",
                    residual, "l2_refill_summary", residual_status,
                    "assumption-dependent residual: refill path minus assumed HBM interface and L2 fill proxy",
                    gpu=gpu, headline_eligible=h_eligible,
                    headline_status="PROXY", headline_reason="derived_from_boundary_assumption"))

        soc = read_key_value_csv(companion(csv_path, "_soc_summary.csv"))
        if soc:
            for key, coeff in (
                ("static_power_w_mean", "P_static"),
                ("leakage_minus_static_w", "P_leak_hot_delta"),
                ("max_power_w_mean", "P_max_soc"),
            ):
                if key in soc:
                    rows.append(_coef_row(profile, csv_path, coeff, "W", soc[key], "soc_summary", "PASS", gpu=gpu))

    return pd.DataFrame(rows)


def build_coverage_matrix(csvs: Iterable[Path], coeffs: pd.DataFrame) -> pd.DataFrame:
    profiles = ["rtx3090", "a100_sxm", "h100_sxm", "h100_pcie"]
    rows = []

    def classify(matches: pd.DataFrame) -> tuple[str, str]:
        if matches.empty:
            return "missing", ""
        statuses = set(str(s).upper() for s in matches.get("status", []))
        headlines = set(str(s).upper() for s in matches.get("headline_status", []))
        if "PASS" in statuses and "NOT_HEADLINE" not in headlines:
            return "pass", f"{len(matches)} coefficient(s)"
        if "FAIL" in statuses and not (statuses - {"FAIL"}):
            return "fail", f"{len(matches)} failing coefficient(s)"
        if "NOT_HEADLINE" in statuses or "NOT_HEADLINE" in headlines:
            return "not_headline", "measured only as fallback/proxy"
        if "LOW_CONF" in statuses or "PROXY" in headlines:
            return "low_conf", f"{len(matches)} low-confidence/proxy coefficient(s)"
        return "low_conf", f"{len(matches)} coefficient(s), status={sorted(statuses)}"

    for profile in profiles:
        c = coeffs[coeffs["gpu_profile"] == profile] if not coeffs.empty else pd.DataFrame()
        for comp in COVERAGE_COMPONENTS:
            status = "missing"
            reason = ""
            if comp == "native_fp8_tensor_core" and not GPU_PROFILES[profile]["native_fp8"]:
                status = "not_applicable"
                reason = "no native FP8 Tensor Core headline for this GPU"
            elif comp == "static_idle":
                status, reason = classify(c[c["coefficient"] == "P_static"] if not c.empty else pd.DataFrame())
            elif comp == "thermal_leakage":
                status, reason = classify(c[c["coefficient"] == "P_leak_hot_delta"] if not c.empty else pd.DataFrame())
            elif comp == "fp32_simt_compute":
                status, reason = classify(c[c["coefficient"].str.contains("fp32_simt", na=False)] if not c.empty else pd.DataFrame())
            elif comp == "tf32_tensor_core":
                status, reason = classify(c[c["coefficient"].str.contains("tf32_tc", na=False)] if not c.empty else pd.DataFrame())
            elif comp == "fp16_tensor_core":
                status, reason = classify(c[c["coefficient"].str.contains("fp16_tc", na=False)] if not c.empty else pd.DataFrame())
            elif comp == "bf16_tensor_core":
                status, reason = classify(c[c["coefficient"].str.contains("bf16_tc", na=False)] if not c.empty else pd.DataFrame())
            elif comp == "native_fp8_tensor_core":
                status, reason = classify(c[c["coefficient"].str.contains("fp8_te", na=False)] if not c.empty else pd.DataFrame())
            elif comp == "memory_read_write":
                status, reason = classify(c[c["coefficient"].str.startswith("k_mem_")] if not c.empty else pd.DataFrame())
            elif comp == "l2_hit_path":
                status, reason = classify(c[c["coefficient"].str.startswith("k_l2_")] if not c.empty else pd.DataFrame())
            elif comp == "hbm_to_l2_refill_path":
                status, reason = classify(c[c["coefficient"] == "k_hbm_to_l2_refill_path"] if not c.empty else pd.DataFrame())
            elif comp == "soc_hbm_phy_to_l2_proxy":
                status, reason = classify(c[c["coefficient"] == "k_soc_hbm_phy_to_l2_fill_proxy"] if not c.empty else pd.DataFrame())
            elif comp == "standalone_nonlinear":
                status, reason = classify(c[c["coefficient"].str.contains("softmax|gelu|layernorm", case=False, na=False)] if not c.empty else pd.DataFrame())
            elif comp == "fused_nonlinear":
                status, reason = classify(c[c["coefficient"].str.startswith("k_fused_")] if not c.empty else pd.DataFrame())
            rows.append({
                "gpu_profile": profile,
                "gpu": GPU_PROFILES[profile]["label"],
                "component": comp,
                "status": status,
                "reason": reason,
            })
    return pd.DataFrame(rows)


def build_model_vs_measured(csvs: Iterable[Path], coeffs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if coeffs.empty:
        return pd.DataFrame()
    for csv_path in csvs:
        df = read_csv_safe(csv_path)
        if df.empty:
            continue
        sidecar = read_key_value_csv(companion(csv_path, "_gpu_spec_snapshot.csv"))
        profile = str(sidecar.get("gpu_profile") or (
            df["gpu_profile"].iloc[0] if "gpu_profile" in df and not df.empty else "") or
            infer_profile(csv_path, df["gpu"].iloc[0] if "gpu" in df else ""))
        c = coeffs[(coeffs["gpu_profile"] == profile) & (coeffs["source"] == "summary")].copy()
        c["value_num"] = pd.to_numeric(c["value"], errors="coerce")
        for _, r in df.iterrows():
            cat = str(r.get("category", ""))
            op = str(r.get("op", ""))
            dt = str(r.get("dtype", ""))
            mode = str(r.get("mode", ""))
            preset = str(r.get("llm_preset", "") or "")
            if cat == "matmul":
                variant = f"matmul_{dt}_{mode}"
                x_col = "total_flops"
            elif cat == "matmul_llm":
                variant = f"llm_{preset}_{dt}_{mode}"
                x_col = "total_flops"
            elif cat == "elementwise":
                variant = f"k_standalone_{dt}_{op}"
                x_col = "total_elements"
            elif cat == "fused":
                variant = f"k_fused_full_{dt}_{op}"
                x_col = "total_elements"
            else:
                continue
            match = c[c["coefficient"].isin([f"k_{variant}", variant])]
            match = match[match["value_num"].notna()]
            if match.empty:
                continue
            k = float(match.iloc[0]["value_num"])
            x = pd.to_numeric(pd.Series([r.get(x_col, np.nan)]), errors="coerce").iloc[0]
            measured = pd.to_numeric(pd.Series([r.get("dyn_energy_j", np.nan)]), errors="coerce").iloc[0]
            if not (np.isfinite(k) and np.isfinite(x) and np.isfinite(measured) and measured != 0):
                continue
            pred = k * x
            rows.append({
                "gpu_profile": profile,
                "gpu": r.get("gpu", ""),
                "category": cat,
                "op": op,
                "dtype": dt,
                "mode": mode,
                "variant": variant,
                "measured_dyn_energy_j": measured,
                "model_dyn_energy_j": pred,
                "delta_j": pred - measured,
                "delta_pct": 100.0 * (pred - measured) / measured,
                "source_csv": str(csv_path),
            })
    return pd.DataFrame(rows)


def _num(value: object, default: float = np.nan) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _coef_value(coeffs: pd.DataFrame, profile: str, coefficient: str,
                source: str | None = None) -> float:
    if coeffs.empty:
        return np.nan
    c = coeffs[(coeffs["gpu_profile"] == profile) & (coeffs["coefficient"] == coefficient)].copy()
    if source is not None and "source" in c:
        c = c[c["source"] == source]
    if c.empty:
        return np.nan
    if "status" in c:
        c = c[~c["status"].astype(str).str.upper().isin(["FAIL", "NOT_HEADLINE"])]
    if c.empty:
        return np.nan
    vals = pd.to_numeric(c["value"], errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def _row_profile(csv_path: Path, df: pd.DataFrame) -> str:
    sidecar = read_key_value_csv(companion(csv_path, "_gpu_spec_snapshot.csv"))
    return str(sidecar.get("gpu_profile") or (
        df["gpu_profile"].iloc[0] if "gpu_profile" in df and not df.empty else "") or
        infer_profile(csv_path, df["gpu"].iloc[0] if "gpu" in df and not df.empty else ""))


def _row_memory_bits(row: pd.Series) -> tuple[float, float]:
    op = str(row.get("op", ""))
    ratio = analyze.STREAM_RW_RATIO.get(op)
    bits = _num(row.get("bytes_traffic"), np.nan) * 8.0
    if not math.isfinite(bits) or bits <= 0:
        return np.nan, np.nan
    if ratio is None:
        rw = analyze.RW_PER_CALL.get(op)
        if not rw:
            return np.nan, np.nan
        # Elementwise/nonlinear traffic has no exact read/write split in the
        # CSV, so use the logical tensor mix convention used by the benchmark.
        if op in ("mul", "add"):
            ratio = (2, 1)
        elif op in ("gelu", "softmax", "layernorm", "stream_copy", "stream_scale"):
            ratio = (1, 1)
        else:
            ratio = (rw, 0)
    reads, writes = ratio
    denom = max(reads + writes, 1)
    return bits * reads / denom, bits * writes / denom


def build_component_reconstruction(csvs: Iterable[Path], coeffs: pd.DataFrame) -> pd.DataFrame:
    """Partial component reconstruction of each measured cell.

    This is deliberately conservative.  It classifies what can be assigned
    from measured coefficients and leaves the rest as unmodeled residual
    instead of pretending that NVML can separate L1/register/front-end power.
    """
    rows = []
    if coeffs.empty:
        return pd.DataFrame()
    for csv_path in csvs:
        df = read_csv_safe(csv_path)
        if df.empty:
            continue
        df = analyze.add_traffic_metrics(df)
        profile = _row_profile(csv_path, df)
        static_w = _coef_value(coeffs, profile, "P_static", "soc_summary")
        if not math.isfinite(static_w):
            static_w = np.nan
        k_l2_read = _coef_value(coeffs, profile, "k_l2_read_hit_path", "l2_summary")
        k_l2_write = _coef_value(coeffs, profile, "k_l2_write_hit_path", "l2_summary")

        for _, r in df.iterrows():
            cat = str(r.get("category", ""))
            op = str(r.get("op", ""))
            dt = str(r.get("dtype", ""))
            mode = str(r.get("mode", ""))
            preset = str(r.get("llm_preset", "") or "")
            measured_total = _num(r.get("total_energy_j"))
            measured_dyn = _num(r.get("dyn_energy_j"))
            wall_s = _num(r.get("wall_s"))
            if not math.isfinite(measured_dyn):
                continue

            compute_j = 0.0
            memory_j = 0.0
            l2_j = 0.0
            nonlinear_j = 0.0
            assigned = []

            if cat == "matmul":
                coeff = f"k_matmul_{dt}_{mode}"
                k = _coef_value(coeffs, profile, coeff, "summary")
                x = _num(r.get("total_flops"))
                if math.isfinite(k) and math.isfinite(x):
                    compute_j = k * x
                    assigned.append("compute")
            elif cat == "matmul_llm":
                coeff = f"k_llm_{preset}_{dt}_{mode}"
                k = _coef_value(coeffs, profile, coeff, "summary")
                x = _num(r.get("total_flops"))
                if math.isfinite(k) and math.isfinite(x):
                    compute_j = k * x
                    assigned.append("compute_llm_shape")
            elif cat == "fused":
                coeff = f"k_fused_full_{dt}_{op}"
                k = _coef_value(coeffs, profile, coeff, "summary")
                x = _num(r.get("total_elements"))
                if math.isfinite(k) and math.isfinite(x):
                    nonlinear_j = k * x
                    assigned.append("fused_full")
            elif cat == "l2":
                read_bits = _num(r.get("estimated_l2_read_bits"), 0.0)
                write_bits = _num(r.get("estimated_l2_write_bits"), 0.0)
                if math.isfinite(k_l2_read) and read_bits > 0:
                    l2_j += k_l2_read * 1e-12 * read_bits
                    assigned.append("l2_read")
                if math.isfinite(k_l2_write) and write_bits > 0:
                    l2_j += k_l2_write * 1e-12 * write_bits
                    assigned.append("l2_write")
            elif cat in ("elementwise", "stream"):
                read_bits, write_bits = _row_memory_bits(r)
                k_read_row = _coef_value(coeffs, profile, f"k_mem_READ_{dt}", "dram_rw_split")
                k_write_row = _coef_value(coeffs, profile, f"k_mem_WRITE_{dt}", "dram_rw_split")
                if not math.isfinite(k_read_row):
                    k_read_row = _coef_value(coeffs, profile, "k_mem_READ_fp16", "dram_rw_split")
                if not math.isfinite(k_write_row):
                    k_write_row = _coef_value(coeffs, profile, "k_mem_WRITE_fp16", "dram_rw_split")
                if math.isfinite(k_read_row) and math.isfinite(read_bits) and read_bits > 0:
                    memory_j += k_read_row * 1e-12 * read_bits
                    assigned.append("mem_read")
                if math.isfinite(k_write_row) and math.isfinite(write_bits) and write_bits > 0:
                    memory_j += k_write_row * 1e-12 * write_bits
                    assigned.append("mem_write")
                if op in ("softmax", "gelu", "layernorm"):
                    coeff = f"k_standalone_{dt}_{op}"
                    k = _coef_value(coeffs, profile, coeff, "summary")
                    x = _num(r.get("total_elements"))
                    if math.isfinite(k) and math.isfinite(x):
                        # Standalone nonlinear slopes include memory traffic.
                        # Store the excess over memory as nonlinear residual.
                        nonlinear_j = max(0.0, k * x - memory_j)
                        assigned.append("nonlinear_residual")

            model_dyn = compute_j + memory_j + l2_j + nonlinear_j
            residual = measured_dyn - model_dyn
            residual_pct = 100.0 * residual / measured_dyn if measured_dyn else np.nan
            matched = math.isfinite(residual_pct) and abs(residual_pct) <= 10
            if not assigned:
                scope = "unclassified"
            elif matched:
                scope = "matched_within_10pct"
            else:
                scope = "partial_component_sum"
            static_j = static_w * wall_s if math.isfinite(static_w) and math.isfinite(wall_s) else np.nan
            rows.append({
                "gpu_profile": profile,
                "gpu": r.get("gpu", ""),
                "category": cat,
                "op": op,
                "dtype": dt,
                "mode": mode,
                "variant": r.get("variant", ""),
                "headline_status": r.get("headline_status", ""),
                "measured_total_energy_j": measured_total,
                "measured_dyn_energy_j": measured_dyn,
                "static_energy_model_j": static_j,
                "compute_energy_model_j": compute_j,
                "memory_energy_model_j": memory_j,
                "l2_energy_model_j": l2_j,
                "nonlinear_energy_model_j": nonlinear_j,
                "model_dyn_energy_j": model_dyn,
                "unmodeled_residual_j": residual,
                "unmodeled_residual_pct": residual_pct,
                "classification": scope,
                "assigned_terms": "+".join(assigned),
                "source_csv": str(csv_path),
            })
    return pd.DataFrame(rows)


def save_coverage_plot(coverage: pd.DataFrame, out_png: Path) -> None:
    if coverage.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = {
        "pass": 4,
        "low_conf": 3,
        "not_headline": 2,
        "not_applicable": 1,
        "missing": 0,
        "fail": -1,
    }
    pivot = coverage.pivot(index="component", columns="gpu_profile", values="status")
    vals = pivot.apply(lambda col: col.map(score).fillna(0)).to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(vals, vmin=-1, vmax=4, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i, comp in enumerate(pivot.index):
        for j, prof in enumerate(pivot.columns):
            txt = str(pivot.loc[comp, prof])
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Component Coverage Matrix")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_model_delta_plots(delta: pd.DataFrame, out_dir: Path) -> None:
    if delta.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for profile, g in delta.groupby("gpu_profile"):
        ax.scatter(g["measured_dyn_energy_j"], g["model_dyn_energy_j"], label=profile, alpha=0.75)
    lo = min(delta["measured_dyn_energy_j"].min(), delta["model_dyn_energy_j"].min())
    hi = max(delta["measured_dyn_energy_j"].max(), delta["model_dyn_energy_j"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured dynamic energy (J)")
    ax.set_ylabel("Model dynamic energy (J)")
    ax.set_title("Model vs measured dynamic energy")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_model_vs_measured_scatter_by_gpu.png", dpi=180)
    plt.close(fig)

    agg = (delta.groupby(["gpu_profile", "category"], as_index=False)
                .agg(delta_pct_median=("delta_pct", "median"),
                     n=("delta_pct", "size")))
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.gpu_profile}\n{r.category}" for r in agg.itertuples()]
    ax.bar(np.arange(len(agg)), agg["delta_pct_median"])
    ax.axhline(10, color="red", linestyle="--", linewidth=1)
    ax.axhline(-10, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(agg)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median delta (%)")
    ax.set_title("Model delta by GPU and workload category")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02_delta_by_gpu_and_workload.png", dpi=180)
    plt.close(fig)


def save_reconstruction_plot(recon: pd.DataFrame, out_png: Path) -> None:
    if recon.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = recon.copy()
    plot_df["unmodeled_residual_pct"] = pd.to_numeric(
        plot_df["unmodeled_residual_pct"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["unmodeled_residual_pct"])]
    if plot_df.empty:
        return
    agg = (plot_df.groupby(["gpu_profile", "category"], as_index=False)
                  .agg(residual_pct_median=("unmodeled_residual_pct", "median"),
                       n=("unmodeled_residual_pct", "size")))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r.gpu_profile}\n{r.category}" for r in agg.itertuples()]
    ax.bar(np.arange(len(agg)), agg["residual_pct_median"])
    ax.axhline(10, color="red", linestyle="--", linewidth=1)
    ax.axhline(-10, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(agg)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median unmodeled residual (%)")
    ax.set_title("Component reconstruction residual by GPU and category")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate gpu_power_bench outputs into multi-GPU component validation tables/plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"),
                    help="Directory to recursively search for gpu_power_bench main CSVs.")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="Optional substrings used to filter benchmark CSV filenames.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Defaults to <reports-dir>/component_validation.")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="Directory for image report PNGs. Defaults to <out-dir>/image_report.")
    ap.add_argument("--capture-runtime-spec", action="store_true",
                    help="Also write runtime GPU specs from torch/nvidia-smi for the current host.")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.reports_dir / "component_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.image_dir or (out_dir / "image_report")
    image_dir.mkdir(parents=True, exist_ok=True)

    csvs = find_benchmark_csvs(args.reports_dir, args.tags)
    if not csvs:
        print(f"[warn] no main gpu_power_bench CSVs found under {args.reports_dir}")

    spec = build_gpu_spec_matrix(csvs)
    spec_path = out_dir / "gpu_spec_matrix.csv"
    spec.to_csv(spec_path, index=False)
    print(f"[save] {spec_path}")

    if args.capture_runtime_spec:
        runtime = collect_runtime_specs()
        runtime_path = out_dir / "gpu_runtime_spec_snapshot.csv"
        runtime.to_csv(runtime_path, index=False)
        print(f"[save] {runtime_path}")

    coeffs = build_coefficient_table(csvs)
    coeff_path = out_dir / "coefficient_confidence_table.csv"
    coeffs.to_csv(coeff_path, index=False)
    print(f"[save] {coeff_path}")

    coverage = build_coverage_matrix(csvs, coeffs)
    coverage_path = out_dir / "component_coverage_matrix.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"[save] {coverage_path}")
    coverage_png = image_dir / "00_component_coverage_matrix.png"
    save_coverage_plot(coverage, coverage_png)
    if coverage_png.exists():
        print(f"[save] {coverage_png}")

    delta = build_model_vs_measured(csvs, coeffs)
    delta_path = out_dir / "model_vs_measured_rows.csv"
    delta.to_csv(delta_path, index=False)
    print(f"[save] {delta_path}")
    save_model_delta_plots(delta, image_dir)
    for name in ("01_model_vs_measured_scatter_by_gpu.png", "02_delta_by_gpu_and_workload.png"):
        if (image_dir / name).exists():
            print(f"[save] {image_dir / name}")

    recon = build_component_reconstruction(csvs, coeffs)
    recon_path = out_dir / "component_reconstruction_rows.csv"
    recon.to_csv(recon_path, index=False)
    print(f"[save] {recon_path}")
    recon_png = image_dir / "03_component_reconstruction_residual.png"
    save_reconstruction_plot(recon, recon_png)
    if recon_png.exists():
        print(f"[save] {recon_png}")

    if not coeffs.empty:
        summary = (coeffs.groupby(["gpu_profile", "status"], as_index=False)
                         .size()
                         .sort_values(["gpu_profile", "status"]))
        print("\n== Coefficient status counts ==")
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
