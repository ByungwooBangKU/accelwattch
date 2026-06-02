#!/usr/bin/env python3
"""Audit strict FP16 result directories before final architecture comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return int(parsed)
    return default


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def artifact_exists(info: Any) -> bool:
    return isinstance(info, dict) and parse_bool(info.get("exists"))


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


def normalize_thread(value: Any) -> str:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return str(int(parsed))
    return str(value or "")


def find_ncu_row(rows: List[Dict[str, Any]], kernel: str, threads: str, blocks_per_sm: str) -> Dict[str, Any]:
    blocks_per_sm = normalize_thread(blocks_per_sm)
    for row in rows:
        if (
            str(row.get("kernel", "")) == kernel
            and normalize_thread(row.get("threads", "")) == threads
            and normalize_thread(row.get("validation_blocks_per_sm", "")) == blocks_per_sm
        ):
            return row
    for row in rows:
        if str(row.get("kernel", "")) == kernel and normalize_thread(row.get("threads", "")) == threads:
            return row
    for row in rows:
        if (
            str(row.get("kernel", "")) == kernel
            and not str(row.get("threads", ""))
            and normalize_thread(row.get("validation_blocks_per_sm", "")) == blocks_per_sm
        ):
            return row
    for row in rows:
        if str(row.get("kernel", "")) == kernel and not str(row.get("threads", "")):
            return row
    return {}


def find_resource_row(
    rows: List[Dict[str, Any]],
    role: str,
    kernel: str,
    threads: str,
    unroll: str,
    blocks_per_sm: str,
) -> Dict[str, Any]:
    blocks_per_sm = normalize_thread(blocks_per_sm)
    for row in rows:
        if (
            str(row.get("role", "")) == role
            and str(row.get("kernel", "")) == kernel
            and normalize_thread(row.get("threads", "")) == threads
            and (not unroll or normalize_thread(row.get("unroll", "")) == normalize_thread(unroll))
            and normalize_thread(row.get("blocks_per_sm_requested", "")) == blocks_per_sm
        ):
            return row
    for row in rows:
        if (
            str(row.get("role", "")) == role
            and str(row.get("kernel", "")) == kernel
            and normalize_thread(row.get("threads", "")) == threads
            and normalize_thread(row.get("blocks_per_sm_requested", "")) == blocks_per_sm
        ):
            return row
    for row in rows:
        if (
            str(row.get("role", "")) == role
            and str(row.get("kernel", "")) == kernel
            and normalize_thread(row.get("threads", "")) == threads
            and (not unroll or normalize_thread(row.get("unroll", "")) == normalize_thread(unroll))
        ):
            return row
    for row in rows:
        if (
            str(row.get("role", "")) == role
            and str(row.get("kernel", "")) == kernel
            and normalize_thread(row.get("threads", "")) == threads
        ):
            return row
    return {}


def select_target(
    targets: Any,
    quality_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    target_rows = [dict(row) for row in targets if isinstance(row, dict)] if isinstance(targets, list) else []
    matching_targets = [
        row
        for row in target_rows
        if str(row.get("test_kernel", "")) == args.require_kernel
        and str(row.get("baseline_kernel", "")) == args.require_baseline
    ]
    if matching_targets:
        return (
            dict(matching_targets[0]),
            {
                "target_selection_source": "selected_targets_required_kernel_baseline",
                "selected_target_count": len(target_rows),
                "matching_selected_target_count": len(matching_targets),
            },
        )
    if target_rows:
        return (
            dict(target_rows[0]),
            {
                "target_selection_source": "selected_targets_no_required_match_first_fallback",
                "selected_target_count": len(target_rows),
                "matching_selected_target_count": 0,
            },
        )

    # Keep useful context in failed audits.
    thread_rows = [r for r in quality_rows if str(r.get("scope", "")) == "thread_sweep"]
    matching_quality = [
        row
        for row in thread_rows
        if str(row.get("test_kernel", "")) == args.require_kernel
        and str(row.get("baseline_kernel", "")) == args.require_baseline
    ]
    if matching_quality:
        return (
            dict(matching_quality[0]),
            {
                "target_selection_source": "quality_rows_required_kernel_baseline_fallback",
                "selected_target_count": 0,
                "matching_selected_target_count": 0,
            },
        )
    if thread_rows:
        return (
            dict(thread_rows[0]),
            {
                "target_selection_source": "quality_rows_first_thread_fallback",
                "selected_target_count": 0,
                "matching_selected_target_count": 0,
            },
        )
    if quality_rows:
        return (
            dict(quality_rows[0]),
            {
                "target_selection_source": "quality_rows_first_fallback",
                "selected_target_count": 0,
                "matching_selected_target_count": 0,
            },
        )
    return (
        {},
        {
            "target_selection_source": "missing",
            "selected_target_count": 0,
            "matching_selected_target_count": 0,
        },
    )


def audit_dir(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    summary = read_json(path / "quality_gate_summary.json")
    manifest = read_json(path / "strict_pipeline_manifest.json")
    pipeline_preflight = read_json(path / "strict_pipeline_preflight.json")
    ncu_permission_probe = read_json(path / "ncu_permission_probe" / "ncu_permission_probe.json")
    quality_rows = read_csv(path / "quality_gates.csv")
    ncu_rows = read_csv(path / "ncu_no_l2_thread_sweep" / "ncu_validation_summary.csv")
    resource_rows = read_csv(path / "resource_audit" / "thread_resource_occupancy.csv")
    targets = summary.get("selected_targets") or []
    target, target_selection = select_target(targets, quality_rows, args)

    chip = str(target.get("architecture_chip", "") or "unknown")
    generation = str(target.get("architecture_generation", "") or "unknown")
    gpu = str(target.get("gpu", "") or "")
    test_kernel = str(target.get("test_kernel", "") or "")
    baseline_kernel = str(target.get("baseline_kernel", "") or "")
    threads = normalize_thread(target.get("threads", ""))
    blocks_per_sm = normalize_thread(target.get("blocks_per_sm_requested", ""))
    unroll = normalize_thread(target.get("unroll", ""))
    test_ncu = find_ncu_row(ncu_rows, test_kernel, threads, blocks_per_sm)
    baseline_ncu = find_ncu_row(ncu_rows, baseline_kernel, threads, blocks_per_sm)
    test_resource = find_resource_row(resource_rows, "test", test_kernel, threads, unroll, blocks_per_sm)
    baseline_resource = find_resource_row(
        resource_rows,
        "baseline",
        baseline_kernel,
        threads,
        unroll,
        blocks_per_sm,
    )

    failed: List[str] = []
    warnings: List[str] = []
    manifest_schema = str(manifest.get("manifest_schema", "") or "")
    manifest_status = str(manifest.get("status", "") or "")
    manifest_params = manifest.get("parameters") if isinstance(manifest.get("parameters"), dict) else {}
    manifest_git = manifest.get("git") if isinstance(manifest.get("git"), dict) else {}
    manifest_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    manifest_head = ""
    if isinstance(manifest_git.get("head"), dict):
        manifest_head = str(manifest_git["head"].get("stdout", "") or "")
    manifest_binary = manifest_artifacts.get("binary") if isinstance(manifest_artifacts.get("binary"), dict) else {}
    manifest_quality_summary = (
        manifest_artifacts.get("quality_gate_summary")
        if isinstance(manifest_artifacts.get("quality_gate_summary"), dict)
        else {}
    )
    manifest_pipeline_preflight_json = (
        manifest_artifacts.get("pipeline_preflight_json")
        if isinstance(manifest_artifacts.get("pipeline_preflight_json"), dict)
        else {}
    )
    manifest_pipeline_preflight_csv = (
        manifest_artifacts.get("pipeline_preflight_csv")
        if isinstance(manifest_artifacts.get("pipeline_preflight_csv"), dict)
        else {}
    )
    manifest_ncu_summary = (
        manifest_artifacts.get("ncu_validation_summary")
        if isinstance(manifest_artifacts.get("ncu_validation_summary"), dict)
        else {}
    )
    manifest_ncu_permission_probe_json = (
        manifest_artifacts.get("ncu_permission_probe_json")
        if isinstance(manifest_artifacts.get("ncu_permission_probe_json"), dict)
        else {}
    )
    manifest_ncu_permission_probe_log = (
        manifest_artifacts.get("ncu_permission_probe_log")
        if isinstance(manifest_artifacts.get("ncu_permission_probe_log"), dict)
        else {}
    )
    manifest_resource_audit = (
        manifest_artifacts.get("resource_audit")
        if isinstance(manifest_artifacts.get("resource_audit"), dict)
        else {}
    )

    if not manifest:
        failed.append("strict_pipeline_manifest.json is missing or empty")
    if manifest_schema != "fp16-strict-pipeline-manifest-v1":
        failed.append("strict_pipeline_manifest.json schema is not fp16-strict-pipeline-manifest-v1")
    if manifest_status != "completed":
        failed.append("strict_pipeline_manifest.json status is not completed")
    if parse_bool(manifest_params.get("skip_preflight")):
        failed.append("strict pipeline manifest indicates skip_preflight=1")
    if parse_bool(manifest_params.get("allow_compute_apps")):
        failed.append("strict pipeline manifest indicates allow_compute_apps=1")
    if parse_bool(manifest_params.get("diagnostic_no_ncu")):
        failed.append("strict pipeline manifest indicates diagnostic_no_ncu=1")
    if not manifest_head:
        failed.append("strict pipeline manifest git head is missing")
    if not str(manifest_binary.get("sha256", "") or ""):
        failed.append("strict pipeline manifest binary sha256 is missing")
    if not parse_bool(manifest_quality_summary.get("exists")):
        failed.append("strict pipeline manifest did not record quality_gate_summary.json")
    if not artifact_exists(manifest_pipeline_preflight_json):
        failed.append("strict pipeline manifest did not record strict_pipeline_preflight.json")
    if not artifact_exists(manifest_pipeline_preflight_csv):
        failed.append("strict pipeline manifest did not record strict_pipeline_preflight.csv")
    if not parse_bool(manifest_ncu_summary.get("exists")):
        failed.append("strict pipeline manifest did not record NCU validation summary")
    if not artifact_exists(manifest_ncu_permission_probe_json):
        failed.append("strict pipeline manifest did not record NCU permission probe JSON")
    if not artifact_exists(manifest_ncu_permission_probe_log):
        failed.append("strict pipeline manifest did not record NCU permission probe log")
    if not parse_bool(manifest_resource_audit.get("exists")):
        failed.append("strict pipeline manifest did not record resource audit")

    if not ncu_permission_probe:
        failed.append("ncu_permission_probe/ncu_permission_probe.json is missing or empty")
    if ncu_permission_probe and not parse_bool(ncu_permission_probe.get("permission_probe_pass")):
        failed.append("NCU permission probe did not pass")
    if ncu_permission_probe and parse_bool(ncu_permission_probe.get("permission_denied")):
        failed.append("NCU permission probe recorded ERR_NVGPUCTRPERM")

    pipeline_preflight_schema = str(pipeline_preflight.get("preflight_schema", "") or "")
    pipeline_preflight_rows = (
        pipeline_preflight.get("rows") if isinstance(pipeline_preflight.get("rows"), list) else []
    )
    pipeline_preflight_matching_rows = [
        row
        for row in pipeline_preflight_rows
        if isinstance(row, dict)
        and str(row.get("cuda_arch", "") or "") == str(manifest_params.get("cuda_arch", "") or "")
        and str(row.get("nvidia_smi_id", "") or "") == str(manifest_params.get("nvidia_smi_id", "") or "")
    ]
    pipeline_toolchain = (
        pipeline_preflight.get("cuda_toolchain_compatibility")
        if isinstance(pipeline_preflight.get("cuda_toolchain_compatibility"), dict)
        else {}
    )
    if not pipeline_preflight:
        failed.append("strict_pipeline_preflight.json is missing or empty")
    if not (path / "strict_pipeline_preflight.csv").exists():
        failed.append("strict_pipeline_preflight.csv is missing")
    if pipeline_preflight_schema != "fp16-strict-architecture-suite-preflight-v1":
        failed.append("strict_pipeline_preflight.json schema is not fp16-strict-architecture-suite-preflight-v1")
    if parse_bool(pipeline_preflight.get("dry_run")):
        failed.append("strict_pipeline_preflight.json has dry_run=true")
    if not parse_bool(pipeline_preflight.get("overall_pass")):
        failed.append("strict_pipeline_preflight.json overall_pass is not true")
    if not parse_bool(pipeline_preflight.get("required_tools_pass")):
        failed.append("strict_pipeline_preflight.json required_tools_pass is not true")
    if pipeline_toolchain and not parse_bool(pipeline_toolchain.get("pass")):
        failed.append("strict_pipeline_preflight.json CUDA toolchain compatibility did not pass")
    if not pipeline_preflight_matching_rows:
        failed.append("strict_pipeline_preflight.json has no row matching manifest cuda_arch/nvidia_smi_id")

    if not targets:
        failed.append("quality_gate_summary selected_targets is empty")
    if int(target_selection.get("matching_selected_target_count", 0)) <= 0:
        failed.append(
            "quality_gate_summary has no selected target matching "
            f"{args.require_kernel}/{args.require_baseline}"
        )
    if int(target_selection.get("matching_selected_target_count", 0)) > 1:
        warnings.append(
            "quality_gate_summary has multiple selected targets matching "
            f"{args.require_kernel}/{args.require_baseline}; using the first"
        )
    if not parse_bool(target.get("target_pass")):
        failed.append("target_pass is not true")
    if not parse_bool(target.get("quality_pass")):
        failed.append("quality_pass is not true")
    if not parse_bool(target.get("quality_gate_selected_target")):
        failed.append("quality_gate_selected_target is not true; rerun quality_gate.py with current target selection")
    if not parse_bool(target.get("util_saturated")):
        failed.append("util_saturated is not true for the quality-gated target")
    if str(target.get("util_reference_scope", "") or "") != "quality_pass":
        failed.append("util_reference_scope is not quality_pass")
    if not math.isfinite(parse_float(target.get("util_reference_max_pct"))):
        failed.append("util_reference_max_pct is missing")
    if str(target.get("target_selection_note", "") or "") != "quality_gate_first_saturation_point":
        failed.append("target_selection_note does not identify the quality-gated first saturation point")
    if str(target.get("measurement_grade", "")) != "strict_nvml_counter":
        failed.append("measurement_grade is not strict_nvml_counter")
    if str(target.get("measurement_grade", "")) == "strict_nvml_counter" and not parse_bool(
        target.get("energy_trace_crosscheck_pass")
    ):
        msg = "NVML-counter/power-trace cross-check is missing or outside the warning band"
        if args.require_counter_trace_agreement:
            failed.append(msg)
        else:
            warnings.append(msg)
    if str(target.get("baseline_match_grade", "")) != "structural_baseline":
        failed.append("baseline_match_grade is not structural_baseline")
    if not parse_bool(target.get("energy_source_reliable")):
        failed.append("energy_source_reliable is not true")
    if not parse_bool(target.get("baseline_structural_match")):
        failed.append("baseline_structural_match is not true")
    if not parse_bool(target.get("benchmark_schema_current")):
        failed.append("benchmark_schema_current is not true")
    if not parse_bool(target.get("ncu_required")):
        failed.append("ncu_required is not true")
    if not parse_bool(target.get("ncu_validation_pass")):
        failed.append("ncu_validation_pass is not true")
    if not parse_bool(target.get("ncu_validation_context_match")):
        failed.append("ncu_validation_context_match is not true")
    if args.require_kernel and test_kernel != args.require_kernel:
        failed.append(f"test_kernel is not {args.require_kernel}")
    if args.require_baseline and baseline_kernel != args.require_baseline:
        failed.append(f"baseline_kernel is not {args.require_baseline}")

    required_valid = parse_int(target.get("required_valid_count"), 0)
    valid_no_l2 = parse_int(target.get("valid_no_l2_count"), 0)
    if required_valid <= 0 or valid_no_l2 < required_valid:
        failed.append("valid_no_l2_count does not meet required_valid_count")
    pjbit = parse_float(target.get("matmul_input_pj_per_bit_mean"))
    if not math.isfinite(pjbit) or pjbit <= 0.0:
        failed.append("matmul_input_pj_per_bit_mean is not positive/finite")
    matmul_denominator_valid = parse_bool(target.get("matmul_denominator_valid"))
    matmul_denominator_metadata_complete = parse_bool(target.get("matmul_denominator_metadata_complete"))
    matmul_denominator_source = str(target.get("matmul_denominator_source", "") or "")
    matmul_input_bits_per_mma = parse_float(target.get("matmul_input_bits_per_logical_mma"))
    matmul_flops_per_mma = parse_float(target.get("matmul_flops_per_logical_mma"))
    if not matmul_denominator_valid:
        failed.append("matmul_denominator_valid is not true")
    if not matmul_denominator_metadata_complete:
        failed.append("matmul_denominator_metadata_complete is not true")
    if matmul_denominator_source != "bench_json_metadata":
        failed.append(
            f"matmul_denominator_source is {matmul_denominator_source or 'missing'}, not bench_json_metadata"
        )
    if (
        not math.isfinite(matmul_input_bits_per_mma)
        or abs(matmul_input_bits_per_mma - args.expected_matmul_input_bits_per_logical_mma) > 1e-6
    ):
        failed.append(
            "matmul_input_bits_per_logical_mma "
            f"{matmul_input_bits_per_mma:g} != {args.expected_matmul_input_bits_per_logical_mma:g}"
        )
    if (
        not math.isfinite(matmul_flops_per_mma)
        or abs(matmul_flops_per_mma - args.expected_mma_flops_per_logical_mma) > 1e-6
    ):
        failed.append(
            f"matmul_flops_per_logical_mma {matmul_flops_per_mma:g} "
            f"!= {args.expected_mma_flops_per_logical_mma:g}"
        )
    sm_util = parse_float(target.get("avg_sm_util_pct_mean"))
    if not math.isfinite(sm_util):
        warnings.append("avg_sm_util_pct_mean is missing")
    elapsed_s = parse_float(target.get("elapsed_s_mean"))
    baseline_elapsed_s = parse_float(target.get("baseline_elapsed_s_mean"))
    test_energy_j = parse_float(target.get("test_energy_j_mean"))
    incremental_energy_j = parse_float(target.get("incremental_energy_j_mean"))
    if not math.isfinite(elapsed_s) or elapsed_s < args.min_test_elapsed_s:
        failed.append(f"elapsed_s_mean is below {args.min_test_elapsed_s:g}s or missing")
    elif elapsed_s < args.warn_test_elapsed_s:
        warnings.append(f"elapsed_s_mean {elapsed_s:.4g}s < warning threshold {args.warn_test_elapsed_s:g}s")
    if math.isfinite(baseline_elapsed_s) and baseline_elapsed_s > 0.0:
        if baseline_elapsed_s < args.min_baseline_elapsed_s:
            failed.append(f"baseline_elapsed_s_mean is below {args.min_baseline_elapsed_s:g}s")
        elif baseline_elapsed_s < args.warn_baseline_elapsed_s:
            warnings.append(
                f"baseline_elapsed_s_mean {baseline_elapsed_s:.4g}s < warning threshold "
                f"{args.warn_baseline_elapsed_s:g}s"
            )
    elif args.require_baseline_elapsed:
        failed.append("baseline_elapsed_s_mean is missing")
    if not math.isfinite(test_energy_j) or test_energy_j < args.min_test_energy_j:
        failed.append(f"test_energy_j_mean is below {args.min_test_energy_j:g} J or missing")
    if not math.isfinite(incremental_energy_j) or incremental_energy_j < args.min_incremental_energy_j:
        failed.append(f"incremental_energy_j_mean is below {args.min_incremental_energy_j:g} J or missing")
    inc_fraction = parse_float(target.get("incremental_energy_fraction_mean"))
    base_fraction = parse_float(target.get("baseline_energy_fraction_mean"))
    if not math.isfinite(inc_fraction) or inc_fraction <= 0.0:
        failed.append("incremental_energy_fraction_mean is missing or nonpositive")
    elif inc_fraction < args.min_incremental_energy_fraction:
        failed.append(
            f"incremental_energy_fraction_mean {inc_fraction:.4g} < {args.min_incremental_energy_fraction:.4g}"
        )
    elif inc_fraction < args.warn_incremental_energy_fraction:
        warnings.append(
            f"incremental_energy_fraction_mean {inc_fraction:.4g} < warning threshold "
            f"{args.warn_incremental_energy_fraction:.4g}"
        )
    if not math.isfinite(base_fraction) or base_fraction < 0.0:
        failed.append("baseline_energy_fraction_mean is missing or invalid")
    elif base_fraction > args.max_baseline_energy_fraction:
        failed.append(
            f"baseline_energy_fraction_mean {base_fraction:.4g} > {args.max_baseline_energy_fraction:.4g}"
        )
    model_util = parse_float(target.get("tensor_model_utilization_pct_mean"))
    if not math.isfinite(model_util) or model_util <= 0.0:
        failed.append("tensor_model_utilization_pct_mean is not positive/finite")
    elif model_util > args.max_tensor_model_util_pct:
        failed.append(
            f"tensor_model_utilization_pct_mean exceeds {args.max_tensor_model_util_pct}; "
            "check architecture model, clock telemetry, and FLOP estimate"
        )
    elif model_util < args.warn_tensor_model_util_pct:
        warnings.append(
            f"tensor_model_utilization_pct_mean is below {args.warn_tensor_model_util_pct}; "
            "selected point may not be Tensor Core throughput saturated"
        )
    if not ncu_rows:
        failed.append("ncu_validation_summary.csv is missing or empty")
    if ncu_rows and not test_ncu:
        failed.append("missing NCU validation row for selected test kernel/thread/blocks_per_sm")
    if ncu_rows and not baseline_ncu:
        failed.append("missing NCU validation row for selected baseline kernel/thread/blocks_per_sm")
    if test_ncu and not parse_bool(test_ncu.get("validation_pass")):
        failed.append("selected test NCU validation did not pass")
    if baseline_ncu and not parse_bool(baseline_ncu.get("validation_pass")):
        failed.append("selected baseline NCU validation did not pass")
    if test_ncu and not parse_bool(test_ncu.get("tensor_activity_observed")):
        msg = "selected test NCU tensor activity is missing or below threshold"
        if args.require_ncu_tensor_activity:
            failed.append(msg)
        else:
            warnings.append(msg)
    if not resource_rows:
        failed.append("resource_audit/thread_resource_occupancy.csv is missing or empty")
    if resource_rows and not test_resource:
        failed.append("missing resource audit row for selected test kernel/thread/blocks_per_sm")
    if resource_rows and not baseline_resource:
        failed.append("missing resource audit row for selected baseline kernel/thread/blocks_per_sm")
    if test_resource and parse_bool(test_resource.get("has_spills")):
        failed.append("selected test kernel has ptxas stack/spill usage")
    if baseline_resource and parse_bool(baseline_resource.get("has_spills")):
        failed.append("selected baseline kernel has ptxas stack/spill usage")

    return {
        "input_dir": str(path),
        "audit_pass": not failed,
        "gpu": gpu,
        "architecture_generation": generation,
        "architecture_chip": chip,
        "test_kernel": test_kernel,
        "baseline_kernel": baseline_kernel,
        "threads": threads,
        "blocks_per_sm_requested": blocks_per_sm,
        "unroll": unroll,
        "threads_per_sm": target.get("threads_per_sm", ""),
        "measurement_grade": target.get("measurement_grade", ""),
        "baseline_match_grade": target.get("baseline_match_grade", ""),
        "target_selection_source": target_selection.get("target_selection_source", ""),
        "selected_target_count": target_selection.get("selected_target_count", ""),
        "matching_selected_target_count": target_selection.get("matching_selected_target_count", ""),
        "benchmark_schema_current": target.get("benchmark_schema_current", ""),
        "test_benchmark_schema_versions": target.get(
            "test_benchmark_schema_versions",
            target.get("test_benchmark_schema_version", ""),
        ),
        "baseline_benchmark_schema_versions": target.get(
            "baseline_benchmark_schema_versions",
            target.get("baseline_benchmark_schema_version", ""),
        ),
        "pipeline_manifest_present": bool(manifest),
        "pipeline_manifest_schema": manifest_schema,
        "pipeline_status": manifest_status,
        "pipeline_cuda_arch": manifest_params.get("cuda_arch", "") if manifest_params else "",
        "pipeline_nvidia_smi_id": manifest_params.get("nvidia_smi_id", "") if manifest_params else "",
        "pipeline_threads": manifest_params.get("threads", "") if manifest_params else "",
        "pipeline_ncu_blocks_per_sm_csv": (
            manifest_params.get("ncu_blocks_per_sm_csv", "") if manifest_params else ""
        ),
        "pipeline_skip_preflight": manifest_params.get("skip_preflight", "") if manifest_params else "",
        "pipeline_allow_compute_apps": manifest_params.get("allow_compute_apps", "") if manifest_params else "",
        "pipeline_diagnostic_no_ncu": manifest_params.get("diagnostic_no_ncu", "") if manifest_params else "",
        "pipeline_git_head": manifest_head,
        "pipeline_binary_sha256": manifest_binary.get("sha256", "") if manifest_binary else "",
        "pipeline_preflight_schema": pipeline_preflight_schema,
        "pipeline_preflight_overall_pass": pipeline_preflight.get("overall_pass", "") if pipeline_preflight else "",
        "pipeline_preflight_required_tools_pass": (
            pipeline_preflight.get("required_tools_pass", "") if pipeline_preflight else ""
        ),
        "pipeline_preflight_dry_run": pipeline_preflight.get("dry_run", "") if pipeline_preflight else "",
        "pipeline_preflight_toolchain_pass": (
            pipeline_toolchain.get("pass", "") if pipeline_toolchain else ""
        ),
        "pipeline_preflight_nvcc_release": (
            pipeline_toolchain.get("nvcc_release", "") if pipeline_toolchain else ""
        ),
        "pipeline_preflight_driver_cuda_version": (
            pipeline_toolchain.get("driver_cuda_version", "") if pipeline_toolchain else ""
        ),
        "pipeline_preflight_matching_row_count": len(pipeline_preflight_matching_rows),
        "pipeline_ncu_permission_probe_recorded": artifact_exists(manifest_ncu_permission_probe_json),
        "pipeline_ncu_permission_probe_pass": (
            ncu_permission_probe.get("permission_probe_pass", "") if ncu_permission_probe else ""
        ),
        "pipeline_ncu_permission_probe_status": (
            ncu_permission_probe.get("status", "") if ncu_permission_probe else ""
        ),
        "pipeline_ncu_permission_probe_permission_denied": (
            ncu_permission_probe.get("permission_denied", "") if ncu_permission_probe else ""
        ),
        "pipeline_ncu_permission_probe_fail_reasons": (
            "; ".join(ncu_permission_probe.get("fail_reasons", []))
            if isinstance(ncu_permission_probe.get("fail_reasons"), list)
            else ncu_permission_probe.get("fail_reasons", "")
        ) if ncu_permission_probe else "",
        "pipeline_ncu_permission_probe_log": (
            ncu_permission_probe.get("log_file", "") if ncu_permission_probe else ""
        ),
        "pipeline_invocation": manifest.get("invocation", ""),
        "energy_trace_crosscheck_pass": target.get("energy_trace_crosscheck_pass", ""),
        "target_pass": target.get("target_pass", ""),
        "quality_pass": target.get("quality_pass", ""),
        "quality_gate_selected_target": target.get("quality_gate_selected_target", ""),
        "util_saturated": target.get("util_saturated", ""),
        "util_reference_scope": target.get("util_reference_scope", ""),
        "util_reference_max_pct": target.get("util_reference_max_pct", ""),
        "util_metric_source": target.get("util_metric_source", ""),
        "target_selection_note": target.get("target_selection_note", ""),
        "ncu_required": target.get("ncu_required", ""),
        "ncu_validation_pass": target.get("ncu_validation_pass", ""),
        "ncu_validation_context_match": target.get("ncu_validation_context_match", ""),
        "valid_no_l2_count": target.get("valid_no_l2_count", ""),
        "required_valid_count": target.get("required_valid_count", ""),
        "avg_sm_util_pct_mean": target.get("avg_sm_util_pct_mean", ""),
        "tflops_mean": target.get("tflops_mean", ""),
        "elapsed_s_mean": target.get("elapsed_s_mean", ""),
        "baseline_elapsed_s_mean": target.get("baseline_elapsed_s_mean", ""),
        "tensor_peak_tflops_model_mean": target.get("tensor_peak_tflops_model_mean", ""),
        "achieved_flops_per_sm_cycle_mean": target.get("achieved_flops_per_sm_cycle_mean", ""),
        "tensor_model_utilization_pct_mean": target.get("tensor_model_utilization_pct_mean", ""),
        "matmul_denominator_valid": target.get("matmul_denominator_valid", ""),
        "matmul_denominator_note": target.get("matmul_denominator_note", ""),
        "matmul_denominator_metadata_complete": target.get("matmul_denominator_metadata_complete", ""),
        "matmul_denominator_source": target.get("matmul_denominator_source", ""),
        "matmul_input_bits_per_logical_mma": target.get("matmul_input_bits_per_logical_mma", ""),
        "matmul_flops_per_logical_mma": target.get("matmul_flops_per_logical_mma", ""),
        "matmul_logical_mma_count_mean": target.get("matmul_logical_mma_count_mean", ""),
        "matmul_input_pj_per_bit_mean": target.get("matmul_input_pj_per_bit_mean", ""),
        "incremental_power_w_mean": target.get("incremental_power_w_mean", ""),
        "test_energy_j_mean": target.get("test_energy_j_mean", ""),
        "incremental_energy_j_mean": target.get("incremental_energy_j_mean", ""),
        "test_energy_counter_vs_trace_ratio_mean": target.get("test_energy_counter_vs_trace_ratio_mean", ""),
        "baseline_energy_counter_vs_trace_ratio_mean": target.get(
            "baseline_energy_counter_vs_trace_ratio_mean",
            "",
        ),
        "test_energy_counter_vs_trace_delta_j_mean": target.get(
            "test_energy_counter_vs_trace_delta_j_mean",
            "",
        ),
        "baseline_energy_counter_vs_trace_delta_j_mean": target.get(
            "baseline_energy_counter_vs_trace_delta_j_mean",
            "",
        ),
        "incremental_energy_fraction_mean": target.get("incremental_energy_fraction_mean", ""),
        "baseline_energy_fraction_mean": target.get("baseline_energy_fraction_mean", ""),
        "baseline_power_fraction_mean": target.get("baseline_power_fraction_mean", ""),
        "test_ncu_pass": test_ncu.get("validation_pass", "") if test_ncu else "",
        "baseline_ncu_pass": baseline_ncu.get("validation_pass", "") if baseline_ncu else "",
        "test_ncu_memory_counter_classes_complete": test_ncu.get("memory_counter_classes_complete", "") if test_ncu else "",
        "baseline_ncu_memory_counter_classes_complete": baseline_ncu.get("memory_counter_classes_complete", "") if baseline_ncu else "",
        "test_ncu_validation_blocks_per_sm": test_ncu.get("validation_blocks_per_sm", "") if test_ncu else "",
        "baseline_ncu_validation_blocks_per_sm": baseline_ncu.get("validation_blocks_per_sm", "") if baseline_ncu else "",
        "test_ncu_validation_unroll": test_ncu.get("validation_unroll", "") if test_ncu else "",
        "baseline_ncu_validation_unroll": baseline_ncu.get("validation_unroll", "") if baseline_ncu else "",
        "test_ncu_validation_suppress_output_store": (
            test_ncu.get("validation_suppress_output_store", "") if test_ncu else ""
        ),
        "baseline_ncu_validation_suppress_output_store": (
            baseline_ncu.get("validation_suppress_output_store", "") if baseline_ncu else ""
        ),
        "test_ncu_l2_counter_total": test_ncu.get("l2_counter_total", "") if test_ncu else "",
        "baseline_ncu_l2_counter_total": baseline_ncu.get("l2_counter_total", "") if baseline_ncu else "",
        "test_ncu_dram_counter_total": test_ncu.get("dram_counter_total", "") if test_ncu else "",
        "baseline_ncu_dram_counter_total": baseline_ncu.get("dram_counter_total", "") if baseline_ncu else "",
        "test_ncu_local_counter_total": test_ncu.get("local_counter_total", "") if test_ncu else "",
        "baseline_ncu_local_counter_total": baseline_ncu.get("local_counter_total", "") if baseline_ncu else "",
        "test_ncu_tensor_activity_pct": test_ncu.get("tensor_activity_pct", "") if test_ncu else "",
        "baseline_ncu_tensor_activity_pct": baseline_ncu.get("tensor_activity_pct", "") if baseline_ncu else "",
        "test_ncu_sm_activity_pct": test_ncu.get("sm_activity_pct", "") if test_ncu else "",
        "baseline_ncu_sm_activity_pct": baseline_ncu.get("sm_activity_pct", "") if baseline_ncu else "",
        "test_ncu_tensor_activity_observed": test_ncu.get("tensor_activity_observed", "") if test_ncu else "",
        "baseline_ncu_tensor_activity_observed": (
            baseline_ncu.get("tensor_activity_observed", "") if baseline_ncu else ""
        ),
        "test_registers_per_thread": test_resource.get("registers_per_thread", "") if test_resource else "",
        "baseline_registers_per_thread": baseline_resource.get("registers_per_thread", "") if baseline_resource else "",
        "test_thread_occupancy_pct_model": test_resource.get("thread_occupancy_pct_model", "") if test_resource else "",
        "baseline_thread_occupancy_pct_model": baseline_resource.get("thread_occupancy_pct_model", "") if baseline_resource else "",
        "test_resource_has_spills": test_resource.get("has_spills", "") if test_resource else "",
        "baseline_resource_has_spills": baseline_resource.get("has_spills", "") if baseline_resource else "",
        "fail_reasons": "; ".join(failed),
        "warnings": "; ".join(warnings),
    }


def plot_audit(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [str(r.get("architecture_chip") or Path(str(r.get("input_dir", ""))).name) for r in rows]
    vals = [1.0 if parse_bool(r.get("audit_pass")) else 0.0 for r in rows]
    colors = ["tab:green" if v == 1.0 else "tab:red" for v in vals]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(rows)), 4.2))
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["fail", "pass"])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Strict FP16 result audit")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "strict_result_audit.png", dpi=160)
    plt.close(fig)


def plot_metric(rows: List[Dict[str, Any]], outdir: Path, metric: str, ylabel: str, title: str, filename: str) -> None:
    values = [parse_float(r.get(metric)) for r in rows]
    if not any(math.isfinite(v) for v in values):
        return
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [str(r.get("architecture_chip") or Path(str(r.get("input_dir", ""))).name) for r in rows]
    colors = ["tab:green" if parse_bool(r.get("audit_pass")) else "tab:red" for r in rows]
    plot_values = [v if math.isfinite(v) else 0.0 for v in values]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(rows)), 4.4))
    ax.bar(range(len(rows)), plot_values, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        if math.isfinite(value):
            ax.annotate(f"{value:.3g}", (idx, plot_values[idx]), textcoords="offset points", xytext=(0, 5), ha="center")
        else:
            ax.annotate("missing", (idx, 0.0), textcoords="offset points", xytext=(0, 5), ha="center")
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=160)
    plt.close(fig)


def plot_audit_metrics(rows: List[Dict[str, Any]], outdir: Path) -> None:
    plot_audit(rows, outdir)
    plot_metric(
        rows,
        outdir,
        "matmul_input_pj_per_bit_mean",
        "pJ/logical input bit",
        "Strict FP16 selected pJ/bit",
        "strict_result_matmul_input_pj_per_bit.png",
    )
    plot_metric(
        rows,
        outdir,
        "tflops_mean",
        "TFLOPS",
        "Strict FP16 selected throughput",
        "strict_result_tflops.png",
    )
    plot_metric(
        rows,
        outdir,
        "elapsed_s_mean",
        "CUDA event elapsed time (s)",
        "Strict FP16 selected test duration",
        "strict_result_elapsed_s.png",
    )
    plot_metric(
        rows,
        outdir,
        "avg_sm_util_pct_mean",
        "Avg SM utilization (%)",
        "Strict FP16 selected SM utilization",
        "strict_result_sm_utilization.png",
    )
    plot_metric(
        rows,
        outdir,
        "tensor_model_utilization_pct_mean",
        "Dense Tensor Core model utilization (%)",
        "Strict FP16 selected Tensor Core model utilization",
        "strict_result_tensor_model_utilization.png",
    )
    plot_metric(
        rows,
        outdir,
        "test_ncu_tensor_activity_pct",
        "NCU tensor activity (%)",
        "Strict FP16 selected NCU tensor activity",
        "strict_result_ncu_tensor_activity.png",
    )
    plot_metric(
        rows,
        outdir,
        "incremental_energy_fraction_mean",
        "Incremental energy / test energy",
        "Strict FP16 selected incremental energy signal",
        "strict_result_incremental_energy_fraction.png",
    )
    plot_metric(
        rows,
        outdir,
        "incremental_energy_j_mean",
        "Incremental energy (J)",
        "Strict FP16 selected incremental energy magnitude",
        "strict_result_incremental_energy_j.png",
    )
    plot_metric(
        rows,
        outdir,
        "test_energy_counter_vs_trace_ratio_mean",
        "NVML energy / power-trace energy",
        "Strict FP16 selected energy counter cross-check",
        "strict_result_counter_trace_ratio.png",
    )
    plot_metric(
        rows,
        outdir,
        "baseline_energy_fraction_mean",
        "Baseline-scaled energy / test energy",
        "Strict FP16 selected baseline energy fraction",
        "strict_result_baseline_energy_fraction.png",
    )


def write_json(path: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    required = [x.strip() for x in args.require_architectures.split(",") if x.strip()]
    passed = {str(r.get("architecture_chip", "")) for r in rows if parse_bool(r.get("audit_pass"))}
    missing = [chip for chip in required if chip not in passed]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dirs": [str(p) for p in args.input],
        "required_architectures": required,
        "missing_required_architectures": missing,
        "tensor_model_thresholds": {
            "max_tensor_model_util_pct": args.max_tensor_model_util_pct,
            "warn_tensor_model_util_pct": args.warn_tensor_model_util_pct,
        },
        "energy_signal_thresholds": {
            "min_incremental_energy_fraction": args.min_incremental_energy_fraction,
            "warn_incremental_energy_fraction": args.warn_incremental_energy_fraction,
            "max_baseline_energy_fraction": args.max_baseline_energy_fraction,
        },
        "measurement_resolution_thresholds": {
            "min_test_elapsed_s": args.min_test_elapsed_s,
            "warn_test_elapsed_s": args.warn_test_elapsed_s,
            "min_baseline_elapsed_s": args.min_baseline_elapsed_s,
            "warn_baseline_elapsed_s": args.warn_baseline_elapsed_s,
            "min_test_energy_j": args.min_test_energy_j,
            "min_incremental_energy_j": args.min_incremental_energy_j,
        },
        "benchmark_schema_thresholds": {
            "expected_benchmark_schema_version": "fp16-energy-bench-v2",
        },
        "pipeline_manifest_thresholds": {
            "expected_manifest_schema": "fp16-strict-pipeline-manifest-v1",
            "expected_status": "completed",
            "expected_preflight_schema": "fp16-strict-architecture-suite-preflight-v1",
            "require_pipeline_preflight_overall_pass": True,
            "require_pipeline_toolchain_compatibility": True,
        },
        "matmul_denominator_thresholds": {
            "expected_matmul_input_bits_per_logical_mma": args.expected_matmul_input_bits_per_logical_mma,
            "expected_mma_flops_per_logical_mma": args.expected_mma_flops_per_logical_mma,
        },
        "counter_trace_crosscheck_thresholds": {
            "warn_counter_trace_ratio_low": args.warn_counter_trace_ratio_low,
            "warn_counter_trace_ratio_high": args.warn_counter_trace_ratio_high,
            "require_counter_trace_agreement": bool(args.require_counter_trace_agreement),
        },
        "ncu_activity_thresholds": {
            "require_ncu_tensor_activity": bool(args.require_ncu_tensor_activity),
        },
        "counts": {
            "rows": len(rows),
            "audit_pass": sum(1 for r in rows if parse_bool(r.get("audit_pass"))),
            "audit_fail": sum(1 for r in rows if not parse_bool(r.get("audit_pass"))),
        },
        "overall_pass": bool(rows) and not missing and all(parse_bool(r.get("audit_pass")) for r in rows),
        "rows": rows,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit strict FP16 result directories")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Strict result directories")
    parser.add_argument("--outdir", type=Path, required=True, help="Audit output directory")
    parser.add_argument("--require-architectures", default="ga100,gh100,ga102")
    parser.add_argument("--require-kernel", default="tensor_mma_f16acc")
    parser.add_argument("--require-baseline", default="tensor_baseline_mov")
    parser.add_argument("--max-tensor-model-util-pct", type=float, default=105.0)
    parser.add_argument("--warn-tensor-model-util-pct", type=float, default=50.0)
    parser.add_argument("--min-incremental-energy-fraction", type=float, default=0.01)
    parser.add_argument("--warn-incremental-energy-fraction", type=float, default=0.05)
    parser.add_argument("--max-baseline-energy-fraction", type=float, default=0.99)
    parser.add_argument("--min-test-elapsed-s", type=float, default=0.25)
    parser.add_argument("--warn-test-elapsed-s", type=float, default=1.0)
    parser.add_argument("--min-baseline-elapsed-s", type=float, default=0.25)
    parser.add_argument("--warn-baseline-elapsed-s", type=float, default=1.0)
    parser.add_argument("--min-test-energy-j", type=float, default=1.0)
    parser.add_argument("--min-incremental-energy-j", type=float, default=0.1)
    parser.add_argument("--expected-matmul-input-bits-per-logical-mma", type=float, default=8192.0)
    parser.add_argument("--expected-mma-flops-per-logical-mma", type=float, default=8192.0)
    parser.add_argument("--require-baseline-elapsed", action="store_true")
    parser.add_argument("--warn-counter-trace-ratio-low", type=float, default=0.5)
    parser.add_argument("--warn-counter-trace-ratio-high", type=float, default=1.5)
    parser.add_argument(
        "--require-counter-trace-agreement",
        action="store_true",
        help="Fail audit when selected NVML-counter/power-trace ratio is missing or outside the warning band",
    )
    parser.add_argument(
        "--require-ncu-tensor-activity",
        action="store_true",
        help="Fail audit when selected tensor_mma test row lacks positive NCU tensor activity evidence",
    )
    parser.add_argument("--no-fail", action="store_true", help="Write audit files but return success even if audit fails")
    args = parser.parse_args()

    rows = [audit_dir(path, args) for path in args.input]
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "strict_result_audit.csv", rows)
    write_json(args.outdir / "strict_result_audit.json", rows, args)
    plot_audit_metrics(rows, args.outdir / "figures")

    required = [x.strip() for x in args.require_architectures.split(",") if x.strip()]
    passed = {str(r.get("architecture_chip", "")) for r in rows if parse_bool(r.get("audit_pass"))}
    missing = [chip for chip in required if chip not in passed]
    overall_pass = bool(rows) and not missing and all(parse_bool(r.get("audit_pass")) for r in rows)

    print(f"Wrote: {args.outdir / 'strict_result_audit.csv'}")
    print(f"Wrote: {args.outdir / 'strict_result_audit.json'}")
    print(f"Wrote figures under: {args.outdir / 'figures'}")
    if missing:
        print("Missing required passing architectures: " + ", ".join(missing), file=sys.stderr)
    if not overall_pass:
        print("Strict FP16 audit failed", file=sys.stderr)
    return 0 if overall_pass or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
