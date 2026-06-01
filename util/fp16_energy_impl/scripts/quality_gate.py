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
    row = ncu_row(kernel, threads, ncu_rows)
    if not row:
        return (False, "missing NCU validation row")
    if parse_bool(row.get("validation_pass")):
        return (True, "NCU validation passed")
    reason = str(row.get("fail_reasons", "") or "NCU validation failed")
    return (False, reason)


def ncu_row(kernel: str, threads: Any, ncu_rows: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    thread_text = str(threads)
    if "." in thread_text:
        value = parse_float(thread_text)
        if math.isfinite(value):
            thread_text = str(int(value))
    return ncu_rows.get((kernel, thread_text)) or ncu_rows.get((kernel, "")) or {}


def ncu_context_status(
    result_row: Dict[str, Any],
    ncu: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str]]:
    if not ncu:
        return (False, ["missing NCU validation row"], [])

    failed: List[str] = []
    warnings: List[str] = []

    comparisons = [
        ("blocks_per_sm_requested", "validation_blocks_per_sm", "blocks_per_sm"),
        ("unroll", "validation_unroll", "unroll"),
    ]
    for result_key, ncu_key, label in comparisons:
        expected = parse_float(result_row.get(result_key))
        observed = parse_float(ncu.get(ncu_key))
        if not math.isfinite(observed):
            msg = f"NCU validation {label} context is missing"
            if args.require_ncu:
                failed.append(msg)
            else:
                warnings.append(msg)
            continue
        if math.isfinite(expected) and int(round(expected)) != int(round(observed)):
            failed.append(
                f"NCU validation {label} {int(round(observed))} != measurement {int(round(expected))}"
            )

    if "validation_suppress_output_store" in ncu and str(ncu.get("validation_suppress_output_store", "")).strip():
        expected_store = parse_bool(result_row.get("suppress_output_store"))
        observed_store = parse_bool(ncu.get("validation_suppress_output_store"))
        if expected_store != observed_store:
            failed.append(
                "NCU validation suppress_output_store "
                f"{observed_store} != measurement {expected_store}"
            )
    elif args.require_ncu:
        failed.append("NCU validation suppress_output_store context is missing")
    else:
        warnings.append("NCU validation suppress_output_store context is missing")

    return (not failed, failed, warnings)


def ncu_tensor_activity_status(kernel: str, ncu: Dict[str, Any]) -> Tuple[bool, str]:
    if not kernel.startswith("tensor_mma_"):
        return (True, "not a tensor_mma test kernel")
    if not ncu:
        return (False, "missing NCU validation row for tensor activity")
    if parse_bool(ncu.get("tensor_activity_observed")):
        pct = parse_float(ncu.get("tensor_activity_pct"))
        if math.isfinite(pct):
            return (True, f"NCU tensor activity observed at {pct:.4g}%")
        return (True, "NCU tensor activity observed")
    pct = parse_float(ncu.get("tensor_activity_pct"))
    if math.isfinite(pct):
        return (False, f"NCU tensor activity not observed; tensor_activity_pct={pct:.4g}%")
    return (False, "NCU tensor activity is missing")


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
        "tensor_mma_f16acc": "tensor_baseline_mov",
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
    if test_kernel == "tensor_mma_f16acc" and baseline_kernel == "tensor_baseline_u32":
        return (
            "structural_alu_baseline",
            False,
            "tensor_baseline_u32 is an ALU-heavy diagnostic baseline; rerun with tensor_baseline_mov",
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


def resolution_quality(
    elapsed_s: Any,
    baseline_elapsed_s: Any,
    test_energy_j: Any,
    incremental_energy_j: Any,
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str], float, float, float, float]:
    elapsed = parse_float(elapsed_s)
    baseline_elapsed = parse_float(baseline_elapsed_s)
    test_energy = parse_float(test_energy_j)
    incremental_energy = parse_float(incremental_energy_j)
    failed: List[str] = []
    warnings: List[str] = []
    if not math.isfinite(elapsed) or elapsed < args.min_test_elapsed_s:
        failed.append(f"test elapsed_s is below {args.min_test_elapsed_s:g}s or missing")
    elif elapsed < args.warn_test_elapsed_s:
        warnings.append(f"test elapsed_s {elapsed:.4g}s < warning threshold {args.warn_test_elapsed_s:g}s")
    if math.isfinite(baseline_elapsed) and baseline_elapsed > 0.0:
        if baseline_elapsed < args.min_baseline_elapsed_s:
            failed.append(f"baseline elapsed_s is below {args.min_baseline_elapsed_s:g}s")
        elif baseline_elapsed < args.warn_baseline_elapsed_s:
            warnings.append(
                f"baseline elapsed_s {baseline_elapsed:.4g}s < warning threshold {args.warn_baseline_elapsed_s:g}s"
            )
    elif args.require_baseline_elapsed:
        failed.append("baseline elapsed_s is missing")
    if not math.isfinite(test_energy) or test_energy < args.min_test_energy_j:
        failed.append(f"test energy is below {args.min_test_energy_j:g} J or missing")
    if not math.isfinite(incremental_energy) or incremental_energy < args.min_incremental_energy_j:
        failed.append(f"incremental energy is below {args.min_incremental_energy_j:g} J or missing")
    return (not failed, failed, warnings, elapsed, baseline_elapsed, test_energy, incremental_energy)


def counter_trace_crosscheck(
    test_source: str,
    baseline_source: str,
    test_ratio_value: Any,
    baseline_ratio_value: Any,
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str], float, float]:
    """Check NVML total-energy counter against nvidia-smi power trace integration.

    The NVML counter remains the primary energy source. This check is a telemetry
    sanity warning by default because H100/Ampere power.draw values may be averaged
    over a different window than the timed kernel interval.
    """
    test_ratio = parse_float(test_ratio_value)
    baseline_ratio = parse_float(baseline_ratio_value)
    failed: List[str] = []
    warnings: List[str] = []

    ratios: List[Tuple[str, float]] = []
    for role, source, ratio in (
        ("test", test_source, test_ratio),
        ("baseline", baseline_source, baseline_ratio),
    ):
        if source != "nvml_total_energy_counter":
            continue
        if math.isfinite(ratio) and ratio > 0.0:
            ratios.append((role, ratio))
        else:
            msg = f"{role} NVML-counter/power-trace ratio is missing"
            if args.require_counter_trace_agreement:
                failed.append(msg)
            else:
                warnings.append(msg)

    for role, ratio in ratios:
        if ratio < args.warn_counter_trace_ratio_low or ratio > args.warn_counter_trace_ratio_high:
            msg = (
                f"{role} NVML-counter/power-trace ratio {ratio:.4g} outside "
                f"[{args.warn_counter_trace_ratio_low:.4g}, {args.warn_counter_trace_ratio_high:.4g}]"
            )
            if args.require_counter_trace_agreement:
                failed.append(msg)
            else:
                warnings.append(msg)

    return (not failed and not warnings, failed, warnings, test_ratio, baseline_ratio)


def row_value_or_mean(row: Dict[str, Any], key: str) -> Any:
    if key in row and str(row.get(key, "")).strip():
        return row.get(key)
    mean_key = f"{key}_mean"
    return row.get(mean_key, "")


def feature_set(text: Any) -> set[str]:
    return {item.strip() for item in str(text or "").replace(";", ",").split(",") if item.strip()}


def benchmark_schema_quality(
    row: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    expected = str(args.expected_benchmark_schema_version)
    failed: List[str] = []
    warnings: List[str] = []

    if "test_benchmark_schema_versions" in row or "baseline_benchmark_schema_versions" in row:
        test_versions = feature_set(row.get("test_benchmark_schema_versions"))
        baseline_versions = feature_set(row.get("baseline_benchmark_schema_versions"))
        current_schema = parse_bool(row.get("benchmark_schema_v2_all"))
        features_required_all = parse_bool(row.get("benchmark_schema_features_required_all"))
    else:
        test_versions = feature_set(row.get("test_benchmark_schema_version"))
        baseline_versions = feature_set(row.get("baseline_benchmark_schema_version"))
        current_schema = test_versions == {expected} and baseline_versions == {expected}
        features_required_all = True

    test_features = feature_set(row.get("test_benchmark_schema_features"))
    baseline_features = feature_set(row.get("baseline_benchmark_schema_features"))
    required_features = {
        "nvml_timed_energy_counter",
        "explicit_m16n16k16_denominator",
        "strict_denominator_provenance",
    }

    if not current_schema:
        failed.append(
            f"benchmark schema is not uniformly {expected}: "
            f"test={','.join(sorted(test_versions)) or 'missing'}, "
            f"baseline={','.join(sorted(baseline_versions)) or 'missing'}"
        )
    if not features_required_all:
        failed.append("not all thread-group rows include required benchmark schema features")
    missing_test = sorted(required_features - test_features)
    missing_baseline = sorted(required_features - baseline_features)
    if missing_test:
        failed.append("test benchmark schema features missing: " + ",".join(missing_test))
    if missing_baseline:
        failed.append("baseline benchmark schema features missing: " + ",".join(missing_baseline))

    return (
        not failed,
        failed,
        warnings,
        {
            "current_schema": current_schema,
            "test_versions": ",".join(sorted(test_versions)),
            "baseline_versions": ",".join(sorted(baseline_versions)),
            "test_features": ",".join(sorted(test_features)),
            "baseline_features": ",".join(sorted(baseline_features)),
        },
    )


def matmul_denominator_quality(
    row: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    kernel = str(row.get("test_kernel", ""))
    if kernel not in {"tensor_mma_f16acc", "tensor_mma_f32acc"}:
        return (
            True,
            [],
            [],
            {
                "valid": True,
                "note": "not_tensor_matmul_kernel",
                "input_bits_per_logical_mma": math.nan,
                "flops_per_logical_mma": math.nan,
                "logical_mma_count": math.nan,
                "metadata_complete": False,
                "source": "not_applicable",
            },
        )

    valid_token = row.get("matmul_denominator_valid", "")
    if not str(valid_token).strip():
        valid_token = row.get("matmul_denominator_valid_all", "")
    metadata_valid = parse_bool(valid_token)
    complete_token = row.get("matmul_denominator_metadata_complete", "")
    if not str(complete_token).strip():
        complete_token = row.get("matmul_denominator_metadata_complete_all", "")
    metadata_complete = parse_bool(complete_token)
    source = str(row.get("matmul_denominator_source", "") or "")
    input_bits = parse_float(row_value_or_mean(row, "matmul_input_bits_per_logical_mma"))
    flops = parse_float(row_value_or_mean(row, "matmul_flops_per_logical_mma"))
    mma_count = parse_float(row_value_or_mean(row, "matmul_logical_mma_count"))
    note = str(row.get("matmul_denominator_note", "") or "")

    failed: List[str] = []
    warnings: List[str] = []
    if not metadata_complete:
        failed.append("matmul denominator metadata is not complete in benchmark JSON")
    if source != "bench_json_metadata":
        failed.append(f"matmul denominator source is {source or 'missing'}, not bench_json_metadata")
    if not metadata_valid:
        failed.append("matmul logical denominator metadata is missing or invalid")
    if not math.isfinite(input_bits) or abs(input_bits - args.expected_matmul_input_bits_per_logical_mma) > 1e-6:
        failed.append(
            "matmul input-bit denominator "
            f"{input_bits:g} != {args.expected_matmul_input_bits_per_logical_mma:g}"
        )
    if not math.isfinite(flops) or abs(flops - args.expected_mma_flops_per_logical_mma) > 1e-6:
        failed.append(
            f"mma FLOP denominator {flops:g} != {args.expected_mma_flops_per_logical_mma:g}"
        )
    if not math.isfinite(mma_count) or mma_count <= 0.0:
        failed.append("logical MMA count is missing or nonpositive")
    if note and metadata_valid and "m16n16k16" not in note:
        warnings.append(note)

    return (
        not failed,
        failed,
        warnings,
        {
            "valid": not failed,
            "note": note or ("logical_m16n16k16_input_bits_8192" if not failed else ""),
            "input_bits_per_logical_mma": input_bits,
            "flops_per_logical_mma": flops,
            "logical_mma_count": mma_count,
            "metadata_complete": metadata_complete,
            "source": source,
        },
    )


def pair_gate_rows(
    summary_rows: Iterable[Dict[str, Any]],
    args: argparse.Namespace,
    ncu_rows: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        test_source = str(row.get("test_energy_source", ""))
        baseline_source = str(row.get("baseline_energy_source", ""))
        test_samples = int(parse_float(row.get("test_power_samples"), 0.0))
        baseline_samples = int(parse_float(row.get("baseline_power_samples"), 0.0))
        grade, reliable_source, source_note = source_grade(
            test_source,
            baseline_source,
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
        resolution_ok, resolution_failed, resolution_warnings, elapsed_s, baseline_elapsed_s, test_energy_j, inc_energy_j = (
            resolution_quality(
                row.get("elapsed_s"),
                row.get("baseline_elapsed_s"),
                row.get("test_energy_j"),
                row.get("incremental_energy_j"),
                args,
            )
        )
        trace_ok, trace_failed, trace_warnings, test_trace_ratio, baseline_trace_ratio = counter_trace_crosscheck(
            test_source,
            baseline_source,
            row.get("test_energy_counter_vs_trace_ratio"),
            row.get("baseline_energy_counter_vs_trace_ratio"),
            args,
        )
        schema_ok, schema_failed, schema_warnings, schema_info = benchmark_schema_quality(row, args)
        denom_ok, denom_failed, denom_warnings, denom_info = matmul_denominator_quality(row, args)
        test_ncu_ok, test_ncu_note = ncu_status(str(row.get("test_kernel", "")), row.get("threads", ""), ncu_rows)
        test_ncu = ncu_row(str(row.get("test_kernel", "")), row.get("threads", ""), ncu_rows)
        baseline_ncu_ok, baseline_ncu_note = ncu_status(
            str(row.get("baseline_kernel", "")), row.get("threads", ""), ncu_rows
        )
        baseline_ncu = ncu_row(str(row.get("baseline_kernel", "")), row.get("threads", ""), ncu_rows)
        ncu_ok = bool(test_ncu_ok and baseline_ncu_ok)
        test_context_ok, test_context_failed, test_context_warnings = ncu_context_status(row, test_ncu, args)
        baseline_context_ok, baseline_context_failed, baseline_context_warnings = ncu_context_status(
            row,
            baseline_ncu,
            args,
        )
        ncu_context_ok = bool(test_context_ok and baseline_context_ok)
        test_tensor_activity_ok, test_tensor_activity_note = ncu_tensor_activity_status(
            str(row.get("test_kernel", "")),
            test_ncu,
        )

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
        if not schema_ok:
            failed.extend(schema_failed)
        warnings.extend(schema_warnings)
        if not denom_ok:
            failed.extend(denom_failed)
        warnings.extend(denom_warnings)
        if not signal_ok:
            failed.extend(signal_failed)
        warnings.extend(signal_warnings)
        if not resolution_ok:
            failed.extend(resolution_failed)
        warnings.extend(resolution_warnings)
        if args.require_counter_trace_agreement and not trace_ok:
            failed.extend(trace_failed)
        warnings.extend(trace_warnings)
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
        if args.require_ncu and not ncu_context_ok:
            failed.extend([f"test NCU context: {msg}" for msg in test_context_failed])
            failed.extend([f"baseline NCU context: {msg}" for msg in baseline_context_failed])
        elif args.ncu_summary and not ncu_context_ok:
            warnings.extend([f"test NCU context: {msg}" for msg in test_context_failed + test_context_warnings])
            warnings.extend(
                f"baseline NCU context: {msg}"
                for msg in baseline_context_failed + baseline_context_warnings
            )
        if args.require_ncu and args.require_ncu_tensor_activity and not test_tensor_activity_ok:
            failed.append(f"test NCU tensor activity: {test_tensor_activity_note}")
        elif args.ncu_summary and not test_tensor_activity_ok:
            warnings.append(f"test NCU tensor activity: {test_tensor_activity_note}")

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
                "energy_trace_crosscheck_pass": trace_ok,
                "baseline_structural_match": baseline_ok,
                "benchmark_schema_current": schema_ok,
                "test_benchmark_schema_version": schema_info["test_versions"],
                "baseline_benchmark_schema_version": schema_info["baseline_versions"],
                "matmul_denominator_valid": denom_ok,
                "matmul_denominator_note": denom_info["note"],
                "matmul_denominator_metadata_complete": denom_info["metadata_complete"],
                "matmul_denominator_source": denom_info["source"],
                "energy_signal_reliable": signal_ok,
                "measurement_resolution_reliable": resolution_ok,
                "ncu_validation_pass": ncu_ok,
                "ncu_validation_context_match": ncu_context_ok,
                "ncu_required": bool(args.require_ncu),
                "ncu_tensor_activity_required": bool(args.require_ncu_tensor_activity),
                "test_ncu_note": test_ncu_note,
                "baseline_ncu_note": baseline_ncu_note,
                "test_ncu_tensor_activity_note": test_tensor_activity_note,
                "test_ncu_validation_blocks_per_sm": test_ncu.get("validation_blocks_per_sm", ""),
                "baseline_ncu_validation_blocks_per_sm": baseline_ncu.get("validation_blocks_per_sm", ""),
                "test_ncu_validation_unroll": test_ncu.get("validation_unroll", ""),
                "baseline_ncu_validation_unroll": baseline_ncu.get("validation_unroll", ""),
                "test_ncu_validation_suppress_output_store": test_ncu.get("validation_suppress_output_store", ""),
                "baseline_ncu_validation_suppress_output_store": baseline_ncu.get(
                    "validation_suppress_output_store",
                    "",
                ),
                "test_ncu_tensor_activity_pct": test_ncu.get("tensor_activity_pct", ""),
                "baseline_ncu_tensor_activity_pct": baseline_ncu.get("tensor_activity_pct", ""),
                "test_ncu_sm_activity_pct": test_ncu.get("sm_activity_pct", ""),
                "baseline_ncu_sm_activity_pct": baseline_ncu.get("sm_activity_pct", ""),
                "test_ncu_tensor_activity_observed": test_ncu.get("tensor_activity_observed", ""),
                "baseline_ncu_tensor_activity_observed": baseline_ncu.get("tensor_activity_observed", ""),
                "clock_stable": clock_stable,
                "sm_util_available": sm_util_available,
                "common_hmma_path": common_hmma,
                "tflops": row.get("tflops", ""),
                "elapsed_s": elapsed_s,
                "baseline_elapsed_s": baseline_elapsed_s,
                "tensor_peak_tflops_model": row.get("tensor_peak_tflops_model", ""),
                "achieved_flops_per_sm_cycle": row.get("achieved_flops_per_sm_cycle", ""),
                "tensor_model_utilization_pct": row.get("tensor_model_utilization_pct", ""),
                "avg_sm_util_pct": row.get("avg_sm_util_pct", ""),
                "matmul_logical_mma_count": denom_info["logical_mma_count"],
                "matmul_flops_per_logical_mma": denom_info["flops_per_logical_mma"],
                "matmul_input_bits_per_logical_mma": denom_info["input_bits_per_logical_mma"],
                "matmul_input_pj_per_bit": row.get("matmul_input_pj_per_bit", ""),
                "incremental_power_w": row.get("incremental_power_w", ""),
                "test_energy_j": test_energy_j,
                "incremental_energy_j": inc_energy_j,
                "incremental_energy_fraction": inc_fraction,
                "baseline_energy_fraction": base_fraction,
                "baseline_power_fraction": row.get("baseline_power_fraction", ""),
                "clock_span_mhz": row.get("clock_span_mhz", ""),
                "test_energy_source": row.get("test_energy_source", ""),
                "baseline_energy_source": row.get("baseline_energy_source", ""),
                "test_energy_counter_vs_trace_ratio": test_trace_ratio,
                "baseline_energy_counter_vs_trace_ratio": baseline_trace_ratio,
                "test_energy_counter_vs_trace_delta_j": row.get("test_energy_counter_vs_trace_delta_j", ""),
                "baseline_energy_counter_vs_trace_delta_j": row.get("baseline_energy_counter_vs_trace_delta_j", ""),
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


def util_metric_source(row: Dict[str, Any]) -> str:
    if math.isfinite(parse_float(row.get("avg_sm_util_pct_mean"))):
        return "avg_sm_util_pct_mean"
    if math.isfinite(parse_float(row.get("avg_gpu_util_pct_mean"))):
        return "avg_gpu_util_pct_mean"
    return ""


def thread_gate_rows(
    thread_rows: List[Dict[str, Any]],
    summary_rows: Iterable[Dict[str, Any]],
    args: argparse.Namespace,
    ncu_rows: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    source_by_thread = source_counts_by_thread(summary_rows)
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
        resolution_ok, resolution_failed, resolution_warnings, elapsed_s, baseline_elapsed_s, test_energy_j, inc_energy_j = (
            resolution_quality(
                row.get("elapsed_s_mean"),
                row.get("baseline_elapsed_s_mean"),
                row.get("test_energy_j_mean"),
                row.get("incremental_energy_j_mean"),
                args,
            )
        )
        clock_span = parse_float(row.get("clock_span_mhz_mean"))
        clock_stable = math.isfinite(clock_span) and clock_span <= args.max_clock_span_mhz
        util = util_value(row)
        util_source = util_metric_source(row)
        sm_util_observed = math.isfinite(util)
        selected = parse_bool(row.get("selected_optimal"))

        source_key = (str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")), str(row.get("threads", "")))
        source_info = source_by_thread.get(source_key, {})
        baseline_grade, baseline_ok, baseline_note = baseline_match_grade(
            str(row.get("test_kernel", "")),
            str(row.get("baseline_kernel", "")),
        )
        test_ncu_ok, test_ncu_note = ncu_status(str(row.get("test_kernel", "")), row.get("threads", ""), ncu_rows)
        test_ncu = ncu_row(str(row.get("test_kernel", "")), row.get("threads", ""), ncu_rows)
        baseline_ncu_ok, baseline_ncu_note = ncu_status(
            str(row.get("baseline_kernel", "")), row.get("threads", ""), ncu_rows
        )
        baseline_ncu = ncu_row(str(row.get("baseline_kernel", "")), row.get("threads", ""), ncu_rows)
        ncu_ok = bool(test_ncu_ok and baseline_ncu_ok)
        test_context_ok, test_context_failed, test_context_warnings = ncu_context_status(row, test_ncu, args)
        baseline_context_ok, baseline_context_failed, baseline_context_warnings = ncu_context_status(
            row,
            baseline_ncu,
            args,
        )
        ncu_context_ok = bool(test_context_ok and baseline_context_ok)
        test_tensor_activity_ok, test_tensor_activity_note = ncu_tensor_activity_status(
            str(row.get("test_kernel", "")),
            test_ncu,
        )
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
        thread_test_source = "nvml_total_energy_counter" if source_info.get("all_nvml") else ""
        thread_baseline_source = "nvml_total_energy_counter" if source_info.get("all_nvml") else ""
        if grade == "power_trace_fallback":
            thread_test_source = "power_trace_integral"
            thread_baseline_source = "power_trace_integral"
        trace_ok, trace_failed, trace_warnings, test_trace_ratio, baseline_trace_ratio = counter_trace_crosscheck(
            thread_test_source,
            thread_baseline_source,
            row.get("test_energy_counter_vs_trace_ratio_mean"),
            row.get("baseline_energy_counter_vs_trace_ratio_mean"),
            args,
        )
        schema_ok, schema_failed, schema_warnings, schema_info = benchmark_schema_quality(row, args)
        denom_ok, denom_failed, denom_warnings, denom_info = matmul_denominator_quality(row, args)

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
        if not schema_ok:
            failed.extend(schema_failed)
        warnings.extend(schema_warnings)
        if not denom_ok:
            failed.extend(denom_failed)
        warnings.extend(denom_warnings)
        if not signal_ok:
            failed.extend(signal_failed)
        warnings.extend(signal_warnings)
        if not resolution_ok:
            failed.extend(resolution_failed)
        warnings.extend(resolution_warnings)
        if args.require_counter_trace_agreement and not trace_ok:
            failed.extend(trace_failed)
        warnings.extend(trace_warnings)
        if args.require_ncu and not ncu_ok:
            failed.append(f"NCU validation failed or missing: test={test_ncu_note}; baseline={baseline_ncu_note}")
        if args.require_ncu and not ncu_context_ok:
            failed.extend([f"test NCU context: {msg}" for msg in test_context_failed])
            failed.extend([f"baseline NCU context: {msg}" for msg in baseline_context_failed])
        if args.require_ncu and args.require_ncu_tensor_activity and not test_tensor_activity_ok:
            failed.append(f"test NCU tensor activity: {test_tensor_activity_note}")
        if grade == "power_trace_fallback":
            warnings.append("NVML energy counter was unavailable; using power trace fallback")
        if args.ncu_summary and not args.require_ncu and not ncu_ok:
            warnings.append(f"NCU validation not passing: test={test_ncu_note}; baseline={baseline_ncu_note}")
        if args.ncu_summary and not args.require_ncu and not ncu_context_ok:
            warnings.extend([f"test NCU context: {msg}" for msg in test_context_failed + test_context_warnings])
            warnings.extend(
                f"baseline NCU context: {msg}"
                for msg in baseline_context_failed + baseline_context_warnings
            )
        if args.ncu_summary and not (args.require_ncu and args.require_ncu_tensor_activity) and not test_tensor_activity_ok:
            warnings.append(f"test NCU tensor activity: {test_tensor_activity_note}")

        quality_pass = not failed

        out.append(
            {
                "scope": "thread_sweep",
                "condition": row.get("condition", ""),
                "gpu": row.get("gpu", ""),
                "architecture_generation": row.get("architecture_generation", ""),
                "architecture_chip": row.get("architecture_chip", ""),
                "sm_count": row.get("sm_count", ""),
                "fp16_path": row.get("fp16_path", ""),
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
                "target_pass": False,
                "selected_optimal": selected,
                "quality_gate_selected_target": False,
                "util_saturated": False,
                "util_reference_scope": "pending_quality_pass_selection",
                "util_reference_max_pct": math.nan,
                "util_metric_source": util_source,
                "target_selection_note": "",
                "no_intended_l2": enough_no_l2,
                "pure_fp16_candidate": enough_pure,
                "energy_source_reliable": source_ok,
                "energy_trace_crosscheck_pass": trace_ok,
                "baseline_structural_match": baseline_ok,
                "benchmark_schema_current": schema_ok,
                "test_benchmark_schema_versions": schema_info["test_versions"],
                "baseline_benchmark_schema_versions": schema_info["baseline_versions"],
                "matmul_denominator_valid": denom_ok,
                "matmul_denominator_note": denom_info["note"],
                "matmul_denominator_metadata_complete": denom_info["metadata_complete"],
                "matmul_denominator_source": denom_info["source"],
                "energy_signal_reliable": signal_ok,
                "measurement_resolution_reliable": resolution_ok,
                "ncu_validation_pass": ncu_ok,
                "ncu_validation_context_match": ncu_context_ok,
                "ncu_required": bool(args.require_ncu),
                "ncu_tensor_activity_required": bool(args.require_ncu_tensor_activity),
                "test_ncu_note": test_ncu_note,
                "baseline_ncu_note": baseline_ncu_note,
                "test_ncu_tensor_activity_note": test_tensor_activity_note,
                "test_ncu_validation_blocks_per_sm": test_ncu.get("validation_blocks_per_sm", ""),
                "baseline_ncu_validation_blocks_per_sm": baseline_ncu.get("validation_blocks_per_sm", ""),
                "test_ncu_validation_unroll": test_ncu.get("validation_unroll", ""),
                "baseline_ncu_validation_unroll": baseline_ncu.get("validation_unroll", ""),
                "test_ncu_validation_suppress_output_store": test_ncu.get("validation_suppress_output_store", ""),
                "baseline_ncu_validation_suppress_output_store": baseline_ncu.get(
                    "validation_suppress_output_store",
                    "",
                ),
                "test_ncu_tensor_activity_pct": test_ncu.get("tensor_activity_pct", ""),
                "baseline_ncu_tensor_activity_pct": baseline_ncu.get("tensor_activity_pct", ""),
                "test_ncu_sm_activity_pct": test_ncu.get("sm_activity_pct", ""),
                "baseline_ncu_sm_activity_pct": baseline_ncu.get("sm_activity_pct", ""),
                "test_ncu_tensor_activity_observed": test_ncu.get("tensor_activity_observed", ""),
                "baseline_ncu_tensor_activity_observed": baseline_ncu.get("tensor_activity_observed", ""),
                "clock_stable": clock_stable,
                "sm_util_available": sm_util_observed,
                "avg_sm_util_pct_mean": row.get("avg_sm_util_pct_mean", ""),
                "avg_gpu_util_pct_mean": row.get("avg_gpu_util_pct_mean", ""),
                "tflops_mean": row.get("tflops_mean", ""),
                "elapsed_s_mean": elapsed_s,
                "baseline_elapsed_s_mean": baseline_elapsed_s,
                "tensor_peak_tflops_model_mean": row.get("tensor_peak_tflops_model_mean", ""),
                "achieved_flops_per_sm_cycle_mean": row.get("achieved_flops_per_sm_cycle_mean", ""),
                "tensor_model_utilization_pct_mean": row.get("tensor_model_utilization_pct_mean", ""),
                "matmul_logical_mma_count_mean": denom_info["logical_mma_count"],
                "matmul_flops_per_logical_mma": denom_info["flops_per_logical_mma"],
                "matmul_input_bits_per_logical_mma": denom_info["input_bits_per_logical_mma"],
                "matmul_input_pj_per_bit_mean": row.get("matmul_input_pj_per_bit_mean", ""),
                "matmul_input_pj_per_bit_ci95": row.get("matmul_input_pj_per_bit_ci95", ""),
                "incremental_power_w_mean": row.get("incremental_power_w_mean", ""),
                "test_energy_j_mean": test_energy_j,
                "incremental_energy_j_mean": inc_energy_j,
                "incremental_energy_fraction_mean": inc_fraction,
                "baseline_energy_fraction_mean": base_fraction,
                "baseline_power_fraction_mean": row.get("baseline_power_fraction_mean", ""),
                "clock_span_mhz_mean": row.get("clock_span_mhz_mean", ""),
                "stats_scope": row.get("stats_scope", ""),
                "test_energy_source_counts": source_info.get("test_energy_source_counts", ""),
                "baseline_energy_source_counts": source_info.get("baseline_energy_source_counts", ""),
                "test_energy_counter_vs_trace_ratio_mean": test_trace_ratio,
                "baseline_energy_counter_vs_trace_ratio_mean": baseline_trace_ratio,
                "test_energy_counter_vs_trace_delta_j_mean": row.get("test_energy_counter_vs_trace_delta_j_mean", ""),
                "baseline_energy_counter_vs_trace_delta_j_mean": row.get(
                    "baseline_energy_counter_vs_trace_delta_j_mean",
                    "",
                ),
                "fail_reasons": "; ".join(failed),
                "warnings": "; ".join(warnings),
            }
        )
    by_target_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in out:
        key = (
            str(row.get("fp16_path", "")),
            str(row.get("test_kernel", "")),
            str(row.get("baseline_kernel", "")),
        )
        by_target_key[key].append(row)

    for group in by_target_key.values():
        quality_rows = [
            row
            for row in group
            if parse_bool(row.get("quality_pass")) and math.isfinite(util_value(row))
        ]
        if not quality_rows:
            for row in group:
                row["util_reference_scope"] = "no_quality_pass"
                row["target_selection_note"] = (
                    "no_quality_pass_candidates"
                    if parse_bool(row.get("quality_pass"))
                    else "not_quality_pass"
                )
            continue

        target_pool = quality_rows
        if not args.allow_power_trace_target:
            target_pool = [
                row for row in quality_rows
                if str(row.get("measurement_grade", "")) == "strict_nvml_counter"
            ]

        if not target_pool:
            max_quality_util = max(util_value(row) for row in quality_rows)
            for row in group:
                row["util_reference_scope"] = "quality_pass_no_strict_nvml_counter"
                row["util_reference_max_pct"] = max_quality_util
                row["util_saturated"] = False
                row["quality_gate_selected_target"] = False
                row["target_pass"] = False
                row["target_selection_note"] = (
                    "quality_pass_non_strict_energy_source_diagnostic"
                    if parse_bool(row.get("quality_pass"))
                    else "not_quality_pass"
                )
            continue

        max_quality_util = max(util_value(row) for row in target_pool)
        saturated = [
            row
            for row in target_pool
            if util_value(row) >= max_quality_util - args.util_tolerance_pct
        ]

        def target_score(row: Dict[str, Any]) -> Tuple[float, float, float]:
            threads_per_sm = parse_float(row.get("threads_per_sm"), math.inf)
            if not math.isfinite(threads_per_sm):
                threads_per_sm = parse_float(row.get("threads"), math.inf)
            tflops = parse_float(row.get("tflops_mean"), -math.inf)
            if not math.isfinite(tflops):
                tflops = -math.inf
            clock_span = parse_float(row.get("clock_span_mhz_mean"), math.inf)
            if not math.isfinite(clock_span):
                clock_span = math.inf
            return (threads_per_sm, -tflops, clock_span)

        target = min(saturated, key=target_score) if saturated else None
        saturated_ids = {id(row) for row in saturated}
        target_id = id(target) if target is not None else None
        target_pool_ids = {id(row) for row in target_pool}
        for row in group:
            row["util_reference_scope"] = "quality_pass"
            row["util_reference_max_pct"] = max_quality_util
            row["util_saturated"] = id(row) in saturated_ids
            row["quality_gate_selected_target"] = id(row) == target_id
            row["target_pass"] = id(row) == target_id
            if id(row) == target_id:
                row["target_selection_note"] = "quality_gate_first_saturation_point"
            elif not parse_bool(row.get("quality_pass")):
                row["target_selection_note"] = "not_quality_pass"
            elif id(row) not in target_pool_ids:
                row["target_selection_note"] = "quality_pass_non_strict_energy_source_diagnostic"
            elif id(row) in saturated_ids:
                row["target_selection_note"] = "quality_pass_saturated_tie_loser"
            else:
                row["target_selection_note"] = "quality_pass_below_saturation_band"
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
        targets = [r for r in group if parse_bool(r.get("target_pass"))]
        if targets:
            sx = parse_float(targets[0].get("threads_per_sm"), parse_float(targets[0].get("threads")))
            if math.isfinite(sx):
                ax.axvline(sx, color="tab:green", linestyle="--", linewidth=1.2, label="target_pass")
        else:
            selected = [r for r in group if parse_bool(r.get("selected_optimal"))]
            if selected:
                sx = parse_float(selected[0].get("threads_per_sm"), parse_float(selected[0].get("threads")))
                if math.isfinite(sx):
                    ax.axvline(
                        sx,
                        color="0.35",
                        linestyle=":",
                        linewidth=1.0,
                        label="analyzer selected diagnostic",
                    )
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
            "min_test_elapsed_s": args.min_test_elapsed_s,
            "min_baseline_elapsed_s": args.min_baseline_elapsed_s,
            "min_test_energy_j": args.min_test_energy_j,
            "min_incremental_energy_j": args.min_incremental_energy_j,
            "expected_benchmark_schema_version": args.expected_benchmark_schema_version,
            "expected_matmul_input_bits_per_logical_mma": args.expected_matmul_input_bits_per_logical_mma,
            "expected_mma_flops_per_logical_mma": args.expected_mma_flops_per_logical_mma,
            "warn_counter_trace_ratio_low": args.warn_counter_trace_ratio_low,
            "warn_counter_trace_ratio_high": args.warn_counter_trace_ratio_high,
            "require_counter_trace_agreement": bool(args.require_counter_trace_agreement),
            "require_ncu": bool(args.require_ncu),
            "require_ncu_tensor_activity": bool(args.require_ncu_tensor_activity),
            "allow_power_trace_target": bool(args.allow_power_trace_target),
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
            "strict_nvml_counter is required for default target_pass selection; "
            "power_trace_fallback stays diagnostic unless --allow-power-trace-target is used.",
            "Tensor Core final candidates must use tensor_baseline_mov/f32, not the legacy baseline_nop.",
            "energy_signal_reliable requires incremental energy to be a configurable minimum fraction of test energy.",
            "measurement_resolution_reliable requires enough elapsed time and energy magnitude for stable measurement.",
            "benchmark_schema_current requires test and baseline JSON from the current explicit-denominator "
            "schema before a strict target can pass.",
            "matmul_denominator_valid requires the Tensor Core logical m16n16k16 denominator: "
            "8192 FP16 input bits and 8192 FLOP per logical MMA.",
            "strict denominator gates require these values to come from complete benchmark JSON metadata, "
            "not analyzer-only legacy fallback formulas.",
            "energy_trace_crosscheck_pass compares NVML total-energy delta with nvidia-smi power trace integration; "
            "it is a warning by default because power.draw may be averaged over a different window.",
            "For final claims, run quality_gate.py with --require-ncu and a validated ncu_validation_summary.csv.",
            "For Tensor Core final claims, --require-ncu-tensor-activity should also be enabled so "
            "selected tensor_mma rows have profiler-side Tensor pipe activity evidence.",
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
    parser.add_argument("--min-test-elapsed-s", type=float, default=0.25)
    parser.add_argument("--warn-test-elapsed-s", type=float, default=1.0)
    parser.add_argument("--min-baseline-elapsed-s", type=float, default=0.25)
    parser.add_argument("--warn-baseline-elapsed-s", type=float, default=1.0)
    parser.add_argument("--min-test-energy-j", type=float, default=1.0)
    parser.add_argument("--min-incremental-energy-j", type=float, default=0.1)
    parser.add_argument("--expected-benchmark-schema-version", default="fp16-energy-bench-v2")
    parser.add_argument("--expected-matmul-input-bits-per-logical-mma", type=float, default=8192.0)
    parser.add_argument("--expected-mma-flops-per-logical-mma", type=float, default=8192.0)
    parser.add_argument("--require-baseline-elapsed", action="store_true")
    parser.add_argument("--warn-counter-trace-ratio-low", type=float, default=0.5)
    parser.add_argument("--warn-counter-trace-ratio-high", type=float, default=1.5)
    parser.add_argument(
        "--require-counter-trace-agreement",
        action="store_true",
        help="Fail quality gates when NVML-counter/power-trace ratio is missing or outside the warning band",
    )
    parser.add_argument("--ncu-summary", type=Path, default=None, help="ncu_validation_summary.csv from validate_ncu_reports.py")
    parser.add_argument("--require-ncu", action="store_true", help="Require passing NCU validation for quality_pass")
    parser.add_argument(
        "--require-ncu-tensor-activity",
        action="store_true",
        help="Require selected tensor_mma rows to have NCU Tensor pipe activity evidence",
    )
    parser.add_argument(
        "--allow-power-trace-target",
        action="store_true",
        help=(
            "Diagnostic mode: allow power_trace_fallback quality_pass rows to become target_pass. "
            "Default target selection requires strict_nvml_counter."
        ),
    )
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
