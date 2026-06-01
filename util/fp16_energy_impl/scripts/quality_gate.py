#!/usr/bin/env python3
"""Apply quality gates to analyzed FP16 energy benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NAN"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_ncu_validation(path: Path | None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if path is None:
        return {}
    rows = read_csv(path)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("kernel", "")), str(row.get("threads", "")))] = row
    return out


def ncu_status(kernel: str, threads: Any, ncu_rows: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[bool, str]:
    thread_text = str(threads)
    if "." in thread_text:
        value = parse_float(thread_text)
        if math.isfinite(value):
            thread_text = str(int(value))
    row = ncu_rows.get((kernel, thread_text)) or ncu_rows.get((kernel, ""))
    if not row:
        return (False, "missing NCU validation row")
    if parse_bool(row.get("validation_pass")):
        return (True, "NCU validation passed")
    reason = str(row.get("fail_reasons", "") or "NCU validation failed")
    return (False, reason)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def min_valid_count(run_count: int) -> int:
    return 1 if run_count <= 1 else max(3, math.ceil(run_count * 0.5))


def source_grade(test_source: str, baseline_source: str, test_samples: int, baseline_samples: int,
                 min_power_samples: int) -> Tuple[str, bool, str]:
    if test_source != baseline_source:
        return ("mixed_or_unavailable", False, "test/baseline energy sources differ")
    if test_source == "nvml_total_energy_counter":
        return ("strict_nvml_counter", True, "NVML total energy counter")
    if test_source == "power_trace_integral":
        ok = test_samples >= min_power_samples and baseline_samples >= min_power_samples
        if ok:
            return ("power_trace_fallback", True, "power trace fallback with enough samples")
        return ("power_trace_undersampled", False, "power trace fallback has too few samples")
    return ("mixed_or_unavailable", False, "energy source unavailable")


def baseline_match_grade(test_kernel: str, baseline_kernel: str) -> Tuple[str, bool, str]:
    expected = {
        "tensor_mma_f16acc": "tensor_baseline_u32",
        "tensor_mma_f32acc": "tensor_baseline_f32",
        "fp16_half2": "baseline_regmove",
    }.get(test_kernel)
    if expected is None:
        return ("not_applicable", True, "no strict baseline mapping for this kernel")
    if baseline_kernel == expected:
        return ("structural_baseline", True, f"matched structural baseline {expected}")
    if test_kernel.startswith("tensor_mma") and baseline_kernel == "baseline_nop":
        return (
            "generic_nop_baseline",
            False,
            f"rerun with {expected}; baseline_nop is only a weak loop baseline for Tensor Core separation",
        )
    if test_kernel == "fp16_half2" and baseline_kernel == "baseline_nop":
        return (
            "generic_nop_baseline",
            False,
            "rerun/use baseline_regmove for the stricter CUDA-core FP16 separation",
        )
    return ("baseline_mismatch", False, f"expected {expected}")


def signal_quality(
    incremental_fraction: Any,
    baseline_fraction: Any,
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str], float, float]:
    inc = parse_float(incremental_fraction)
    base = parse_float(baseline_fraction)
    failed: List[str] = []
    warnings: List[str] = []
    if not math.isfinite(inc) or inc <= 0.0:
        failed.append("incremental energy fraction is missing or nonpositive")
    elif inc < args.min_incremental_energy_fraction:
        failed.append(
            f"incremental energy fraction {inc:.4g} < {args.min_incremental_energy_fraction:.4g}"
        )
    elif inc < args.warn_incremental_energy_fraction:
        warnings.append(
            f"incremental energy fraction {inc:.4g} < warning threshold {args.warn_incremental_energy_fraction:.4g}"
        )
    if not math.isfinite(base) or base < 0.0:
        failed.append("baseline energy fraction is missing or invalid")
    elif base > args.max_baseline_energy_fraction:
        failed.append(
            f"baseline energy fraction {base:.4g} > {args.max_baseline_energy_fraction:.4g}"
        )
    return (not failed, failed, warnings, inc, base)


def pair_gate_rows(
    summary_rows: Iterable[Dict[str, Any]],
    args: argparse.Namespace,
    ncu_rows: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        test_samples = int(parse_float(row.get("test_power_samples"), 0.0))
        baseline_samples = int(parse_float(row.get("baseline_power_samples"), 0.0))
        grade, reliable_source, source_note = source_grade(
            str(row.get("test_energy_source", "")),
            str(row.get("baseline_energy_source", "")),
            test_samples,
            baseline_samples,
            args.min_power_samples,
        )
        baseline_grade, baseline_ok, baseline_note = baseline_match_grade(
            str(row.get("test_kernel", "")),
            str(row.get("baseline_kernel", "")),
        )
        positive = parse_bool(row.get("valid_basic"))
        no_l2 = parse_bool(row.get("valid_no_l2"))
        pure = parse_bool(row.get("pure_fp16_candidate"))
        sources_match = parse_bool(row.get("energy_sources_match"))
        clock_span = parse_float(row.get("clock_span_mhz"))
        clock_stable = math.isfinite(clock_span) and clock_span <= args.max_clock_span_mhz
        common_hmma = not parse_bool(row.get("benchmark_uses_wgmma"))
        sm_util_samples = int(parse_float(row.get("test_sm_util_samples"), 0.0))
        sm_util_available = sm_util_samples >= args.min_sm_util_samples
        signal_ok, signal_failed, signal_warnings, inc_fraction, base_fraction = signal_quality(
            row.get("incremental_energy_fraction"),
            row.get("baseline_energy_fraction"),
            args,
        )
        test_ncu_ok, test_ncu_note = ncu_status(str(row.get("test_kernel", "")), row.get("threads", ""), ncu_rows)
        baseline_ncu_ok, baseline_ncu_note = ncu_status(
            str(row.get("baseline_kernel", "")), row.get("threads", ""), ncu_rows
        )
        ncu_ok = bool(test_ncu_ok and baseline_ncu_ok)

        failed: List[str] = []
        warnings: List[str] = []
        if not positive:
            failed.append("nonpositive incremental power/energy")
        if not no_l2:
            failed.append("expected or invalid L2/global traffic")
        if not pure:
            failed.append("not a pure FP16 candidate")
        if not sources_match:
            failed.append("test/baseline energy source mismatch")
        if not reliable_source:
            failed.append(source_note)
        if not baseline_ok:
            failed.append(baseline_note)
        if not signal_ok:
            failed.extend(signal_failed)
        warnings.extend(signal_warnings)
        if grade == "power_trace_fallback":
            warnings.append("NVML energy counter was unavailable; using power trace fallback")
        if not clock_stable:
            failed.append("SM clock span exceeds threshold or is missing")
        if not common_hmma:
            failed.append("benchmark used WGMMA; cross-GPU HMMA comparison is not apples-to-apples")
        if not sm_util_available:
            warnings.append("SM utilization samples missing or sparse")
        if args.require_ncu and not ncu_ok:
            failed.append(f"NCU validation failed or missing: test={test_ncu_note}; baseline={baseline_ncu_note}")
        elif args.ncu_summary and not ncu_ok:
            warnings.append(f"NCU validation not passing: test={test_ncu_note}; baseline={baseline_ncu_note}")

        out.append(
            {
                "scope": "pair",
                "condition": row.get("condition", ""),
                "repeat_index": row.get("repeat_index", ""),
                "gpu": row.get("gpu", ""),
                "architecture_generation": row.get("architecture_generation", ""),
                "architecture_chip": row.get("architecture_chip", ""),
                "sm_count": row.get("sm_count", ""),
                "test_kernel": row.get("test_kernel", ""),
                "baseline_kernel": row.get("baseline_kernel", ""),
                "threads": row.get("threads", ""),
                "threads_per_sm": row.get("threads_per_sm", ""),
                "blocks_per_sm_requested": row.get("blocks_per_sm_requested", ""),
                "unroll": row.get("unroll", ""),
                "measurement_grade": grade,
                "baseline_match_grade": baseline_grade,
                "quality_pass": not failed,
                "target_pass": False,
                "positive_increment": positive,
                "no_intended_l2": no_l2,
                "pure_fp16_candidate": pure,
                "energy_source_reliable": reliable_source,
                "baseline_structural_match": baseline_ok,
                "energy_signal_reliable": signal_ok,
                "ncu_validation_pass": ncu_ok,
                "ncu_required": bool(args.require_ncu),
                "test_ncu_note": test_ncu_note,
                "baseline_ncu_note": baseline_ncu_note,
                "clock_stable": clock_stable,
                "sm_util_available": sm_util_available,
                "common_hmma_path": common_hmma,
                "tflops": row.get("tflops", ""),
                "tensor_peak_tflops_model": row.get("tensor_peak_tflops_model", ""),
                "achieved_flops_per_sm_cycle": row.get("achieved_flops_per_sm_cycle", ""),
                "tensor_model_utilization_pct": row.get("tensor_model_utilization_pct", ""),
                "avg_sm_util_pct": row.get("avg_sm_util_pct", ""),
                "matmul_input_pj_per_bit": row.get("matmul_input_pj_per_bit", ""),
                "incremental_power_w": row.get("incremental_power_w", ""),
                "incremental_energy_fraction": inc_fraction,
                "baseline_energy_fraction": base_fraction,
                "baseline_power_fraction": row.get("baseline_power_fraction", ""),
                "clock_span_mhz": row.get("clock_span_mhz", ""),
                "test_energy_source": row.get("test_energy_source", ""),
                "baseline_energy_source": row.get("baseline_energy_source", ""),
                "test_power_samples": test_samples,
                "baseline_power_samples": baseline_samples,
                "fail_reasons": "; ".join(failed),
                "warnings": "; ".join(warnings),
            }
        )
    return out


def source_counts_by_thread(summary_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")), str(row.get("threads", "")))
        grouped[key].append(row)

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, rows in grouped.items():
        test_counts = Counter(str(r.get("test_energy_source", "")) for r in rows)
        base_counts = Counter(str(r.get("baseline_energy_source", "")) for r in rows)
        test_samples = [int(parse_float(r.get("test_power_samples"), 0.0)) for r in rows]
        base_samples = [int(parse_float(r.get("baseline_power_samples"), 0.0)) for r in rows]
        out[key] = {
            "test_energy_source_counts": ",".join(f"{k}:{v}" for k, v in sorted(test_counts.items())),
            "baseline_energy_source_counts": ",".join(f"{k}:{v}" for k, v in sorted(base_counts.items())),
            "min_test_power_samples": min(test_samples) if test_samples else 0,
            "min_baseline_power_samples": min(base_samples) if base_samples else 0,
            "all_nvml": set(test_counts) == {"nvml_total_energy_counter"} and set(base_counts) == {"nvml_total_energy_counter"},
            "any_power_trace": "power_trace_integral" in test_counts or "power_trace_integral" in base_counts,
            "any_unavailable": "unavailable" in test_counts or "unavailable" in base_counts,
        }
    return out


def util_value(row: Dict[str, Any]) -> float:
    util = parse_float(row.get("avg_sm_util_pct_mean"))
    if not math.isfinite(util):
        util = parse_float(row.get("avg_gpu_util_pct_mean"))
    return util


def thread_gate_rows(
    thread_rows: List[Dict[str, Any]],
    summary_rows: Iterable[Dict[str, Any]],
    args: argparse.Namespace,
    ncu_rows: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    source_by_thread = source_counts_by_thread(summary_rows)
    max_util_by_kernel: Dict[Tuple[str, str, str], float] = {}
    for row in thread_rows:
        key = (str(row.get("fp16_path", "")), str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")))
        util = util_value(row)
        if math.isfinite(util):
            max_util_by_kernel[key] = max(util, max_util_by_kernel.get(key, -math.inf))

    out: List[Dict[str, Any]] = []
    for row in thread_rows:
        run_count = int(parse_float(row.get("run_count"), 0.0))
        valid_no_l2 = int(parse_float(row.get("valid_no_l2_count"), 0.0))
        pure_count = int(parse_float(row.get("pure_fp16_candidate_count"), 0.0))
        required = min_valid_count(run_count)
        enough_no_l2 = valid_no_l2 >= required
        enough_pure = pure_count >= required
        pjbit = parse_float(row.get("matmul_input_pj_per_bit_mean"))
        pjbit_positive = math.isfinite(pjbit) and pjbit > 0.0
        signal_ok, signal_failed, signal_warnings, inc_fraction, base_fraction = signal_quality(
            row.get("incremental_energy_fraction_mean"),
            row.get("baseline_energy_fraction_mean"),
            args,
        )
        clock_span = parse_float(row.get("clock_span_mhz_mean"))
        clock_stable = math.isfinite(clock_span) and clock_span <= args.max_clock_span_mhz
        util = util_value(row)
        sm_util_observed = math.isfinite(util)
        key = (str(row.get("fp16_path", "")), str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")))
        max_util = max_util_by_kernel.get(key, math.nan)
        util_saturated = sm_util_observed and math.isfinite(max_util) and util >= max_util - args.util_tolerance_pct
        selected = parse_bool(row.get("selected_optimal"))

        source_key = (str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")), str(row.get("threads", "")))
        source_info = source_by_thread.get(source_key, {})
        baseline_grade, baseline_ok, baseline_note = baseline_match_grade(
            str(row.get("test_kernel", "")),
            str(row.get("baseline_kernel", "")),
        )
        test_ncu_ok, test_ncu_note = ncu_status(str(row.get("test_kernel", "")), row.get("threads", ""), ncu_rows)
        baseline_ncu_ok, baseline_ncu_note = ncu_status(
            str(row.get("baseline_kernel", "")), row.get("threads", ""), ncu_rows
        )
        ncu_ok = bool(test_ncu_ok and baseline_ncu_ok)
        if source_info.get("all_nvml"):
            grade = "strict_nvml_counter"
            source_ok = True
        elif source_info.get("any_power_trace") and not source_info.get("any_unavailable"):
            grade = "power_trace_fallback"
            source_ok = (
                int(source_info.get("min_test_power_samples", 0)) >= args.min_power_samples
                and int(source_info.get("min_baseline_power_samples", 0)) >= args.min_power_samples
            )
        else:
            grade = "mixed_or_unavailable"
            source_ok = False

        failed: List[str] = []
        warnings: List[str] = []
        if not enough_no_l2:
            failed.append(f"valid_no_l2_count {valid_no_l2} < required {required}")
        if not enough_pure:
            failed.append(f"pure_fp16_candidate_count {pure_count} < required {required}")
        if not pjbit_positive:
            failed.append("matmul_input_pj_per_bit is not positive/finite")
        if not clock_stable:
            failed.append("SM clock span exceeds threshold or is missing")
        if not sm_util_observed:
            failed.append("SM/GPU utilization missing")
        if not source_ok:
            failed.append("energy source is unavailable or undersampled")
        if not baseline_ok:
            failed.append(baseline_note)
        if not signal_ok:
            failed.extend(signal_failed)
        warnings.extend(signal_warnings)
        if args.require_ncu and not ncu_ok:
            failed.append(f"NCU validation failed or missing: test={test_ncu_note}; baseline={baseline_ncu_note}")
        if grade == "power_trace_fallback":
            warnings.append("NVML energy counter was unavailable; using power trace fallback")
        if args.ncu_summary and not args.require_ncu and not ncu_ok:
            warnings.append(f"NCU validation not passing: test={test_ncu_note}; baseline={baseline_ncu_note}")

        quality_pass = not failed
        target_pass = quality_pass and selected and util_saturated

        out.append(
            {
                "scope": "thread_sweep",
                "condition": row.get("condition", ""),
                "gpu": row.get("gpu", ""),
                "architecture_generation": row.get("architecture_generation", ""),
                "architecture_chip": row.get("architecture_chip", ""),
                "sm_count": row.get("sm_count", ""),
                "test_kernel": row.get("test_kernel", ""),
                "baseline_kernel": row.get("baseline_kernel", ""),
                "threads": row.get("threads", ""),
                "threads_per_sm": row.get("threads_per_sm", ""),
                "blocks_per_sm_requested": row.get("blocks_per_sm_requested", ""),
                "unroll": row.get("unroll", ""),
                "run_count": run_count,
                "required_valid_count": required,
                "valid_no_l2_count": valid_no_l2,
                "pure_fp16_candidate_count": pure_count,
                "measurement_grade": grade,
                "baseline_match_grade": baseline_grade,
                "quality_pass": quality_pass,
                "target_pass": target_pass,
                "selected_optimal": selected,
                "util_saturated": util_saturated,
                "no_intended_l2": enough_no_l2,
                "pure_fp16_candidate": enough_pure,
                "energy_source_reliable": source_ok,
                "baseline_structural_match": baseline_ok,
                "energy_signal_reliable": signal_ok,
                "ncu_validation_pass": ncu_ok,
                "ncu_required": bool(args.require_ncu),
                "test_ncu_note": test_ncu_note,
                "baseline_ncu_note": baseline_ncu_note,
                "clock_stable": clock_stable,
                "sm_util_available": sm_util_observed,
                "avg_sm_util_pct_mean": row.get("avg_sm_util_pct_mean", ""),
                "avg_gpu_util_pct_mean": row.get("avg_gpu_util_pct_mean", ""),
                "tflops_mean": row.get("tflops_mean", ""),
                "tensor_peak_tflops_model_mean": row.get("tensor_peak_tflops_model_mean", ""),
                "achieved_flops_per_sm_cycle_mean": row.get("achieved_flops_per_sm_cycle_mean", ""),
                "tensor_model_utilization_pct_mean": row.get("tensor_model_utilization_pct_mean", ""),
                "matmul_input_pj_per_bit_mean": row.get("matmul_input_pj_per_bit_mean", ""),
                "matmul_input_pj_per_bit_ci95": row.get("matmul_input_pj_per_bit_ci95", ""),
                "incremental_power_w_mean": row.get("incremental_power_w_mean", ""),
                "incremental_energy_fraction_mean": inc_fraction,
                "baseline_energy_fraction_mean": base_fraction,
                "baseline_power_fraction_mean": row.get("baseline_power_fraction_mean", ""),
                "clock_span_mhz_mean": row.get("clock_span_mhz_mean", ""),
                "stats_scope": row.get("stats_scope", ""),
                "test_energy_source_counts": source_info.get("test_energy_source_counts", ""),
                "baseline_energy_source_counts": source_info.get("baseline_energy_source_counts", ""),
                "fail_reasons": "; ".join(failed),
                "warnings": "; ".join(warnings),
            }
        )
    return out


def plot_thread_quality(rows: List[Dict[str, Any]], figdir: Path) -> None:
    thread_rows = [r for r in rows if r.get("scope") == "thread_sweep"]
    if not thread_rows:
        return

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in thread_rows:
        grouped[(str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")))].append(row)

    figdir.mkdir(parents=True, exist_ok=True)
    for (test_kernel, baseline_kernel), group in grouped.items():
        group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"), 0.0)))
        xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
        util = []
        for r in group:
            value = parse_float(r.get("avg_sm_util_pct_mean"))
            if not math.isfinite(value):
                value = parse_float(r.get("avg_gpu_util_pct_mean"))
            util.append(value)
        pjbit = [parse_float(r.get("matmul_input_pj_per_bit_mean")) for r in group]

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        colors = [
            "tab:green" if parse_bool(r.get("target_pass")) else
            ("tab:blue" if parse_bool(r.get("quality_pass")) else "tab:red")
            for r in group
        ]
        ax.scatter(xs, util, c=colors, s=48, zorder=3)
        ax.plot(xs, util, color="0.45", linewidth=1.0, alpha=0.8, zorder=2)
        finite_util = [v for v in util if math.isfinite(v)]
        top_util = max(finite_util) if finite_util else math.nan
        for x, y, pj, r in zip(xs, util, pjbit, group):
            if not math.isfinite(y):
                continue
            label = str(r.get("threads", ""))
            if math.isfinite(pj):
                label += f"\n{pj:.3g} pJ/b"
            near_top = y >= 99.98 or (math.isfinite(top_util) and y >= top_util - 0.01)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, -24 if near_top else 7),
                ha="center",
                va="top" if near_top else "bottom",
                fontsize=8,
            )
        selected = [r for r in group if parse_bool(r.get("selected_optimal"))]
        if selected:
            sx = parse_float(selected[0].get("threads_per_sm"), parse_float(selected[0].get("threads")))
            if math.isfinite(sx):
                ax.axvline(sx, color="tab:green", linestyle="--", linewidth=1.2, label="selected")
        ax.set_xlabel("Launched threads per SM")
        ax.set_ylabel("Avg SM utilization (%)")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"Quality-gated thread sweep: {test_kernel} vs {baseline_kernel}", pad=12)
        ax.legend(loc="best")
        fig.tight_layout()
        safe = f"quality_gate_thread_sweep_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
        fig.savefig(figdir / safe, dpi=160)
        plt.close(fig)


def write_summary(input_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    targets = [r for r in rows if r.get("scope") == "thread_sweep" and parse_bool(r.get("target_pass"))]
    selected_diagnostics = [
        r for r in rows
        if r.get("scope") == "thread_sweep" and parse_bool(r.get("selected_optimal")) and not parse_bool(r.get("target_pass"))
    ]
    payload = {
        "input": str(input_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "max_clock_span_mhz": args.max_clock_span_mhz,
            "min_power_samples": args.min_power_samples,
            "min_sm_util_samples": args.min_sm_util_samples,
            "util_tolerance_pct": args.util_tolerance_pct,
            "min_incremental_energy_fraction": args.min_incremental_energy_fraction,
            "warn_incremental_energy_fraction": args.warn_incremental_energy_fraction,
            "max_baseline_energy_fraction": args.max_baseline_energy_fraction,
            "require_ncu": bool(args.require_ncu),
            "ncu_summary": str(args.ncu_summary) if args.ncu_summary else "",
        },
        "counts": {
            "rows": len(rows),
            "pair_rows": sum(1 for r in rows if r.get("scope") == "pair"),
            "thread_rows": sum(1 for r in rows if r.get("scope") == "thread_sweep"),
            "quality_pass": sum(1 for r in rows if parse_bool(r.get("quality_pass"))),
            "target_pass": sum(1 for r in rows if parse_bool(r.get("target_pass"))),
        },
        "selected_targets": targets,
        "selected_diagnostics": selected_diagnostics,
        "notes": [
            "valid_no_l2 means valid_basic=True and the benchmark metadata does not expect global/L2 traffic.",
            "It is not a physical proof of zero L2 traffic; Nsight Compute memory counters are still required.",
            "strict_nvml_counter is preferred for H100/A100/RTX3090 comparison; power_trace_fallback is diagnostic.",
            "Tensor Core final candidates must use tensor_baseline_u32/f32, not the legacy baseline_nop.",
            "energy_signal_reliable requires incremental energy to be a configurable minimum fraction of test energy.",
            "For final claims, run quality_gate.py with --require-ncu and a validated ncu_validation_summary.csv.",
        ],
    }
    with (input_dir / "quality_gate_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality-gate analyzed FP16 energy benchmark results")
    parser.add_argument("--input", type=Path, required=True, help="Analyzed result directory")
    parser.add_argument("--max-clock-span-mhz", type=float, default=60.0)
    parser.add_argument("--min-power-samples", type=int, default=3)
    parser.add_argument("--min-sm-util-samples", type=int, default=1)
    parser.add_argument("--util-tolerance-pct", type=float, default=0.1)
    parser.add_argument("--min-incremental-energy-fraction", type=float, default=0.01)
    parser.add_argument("--warn-incremental-energy-fraction", type=float, default=0.05)
    parser.add_argument("--max-baseline-energy-fraction", type=float, default=0.99)
    parser.add_argument("--ncu-summary", type=Path, default=None, help="ncu_validation_summary.csv from validate_ncu_reports.py")
    parser.add_argument("--require-ncu", action="store_true", help="Require passing NCU validation for quality_pass")
    args = parser.parse_args()

    summary_rows = read_csv(args.input / "summary.csv")
    if not summary_rows:
        raise SystemExit(f"{args.input / 'summary.csv'} not found or empty; run analyze_results.py first")
    thread_rows = read_csv(args.input / "thread_sweep_summary.csv")
    ncu_rows = load_ncu_validation(args.ncu_summary)

    rows = pair_gate_rows(summary_rows, args, ncu_rows)
    rows.extend(thread_gate_rows(thread_rows, summary_rows, args, ncu_rows))
    write_csv(args.input / "quality_gates.csv", rows)
    write_summary(args.input, rows, args)
    plot_thread_quality(rows, args.input / "figures")

    print(f"Wrote: {args.input / 'quality_gates.csv'}")
    print(f"Wrote: {args.input / 'quality_gate_summary.json'}")
    if thread_rows:
        print(f"Wrote quality gate figures under: {args.input / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
