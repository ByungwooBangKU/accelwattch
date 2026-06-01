#!/usr/bin/env python3
"""GPU-free smoke tests for strict FP16 audit/report invariants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not row_list:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in row_list:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_list)


def read_single_csv_row(path: Path) -> Dict[str, str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise AssertionError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def run(cmd: List[str], *, cwd: Path, env: Dict[str, str], expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if expect_success and cp.returncode != 0:
        raise AssertionError(
            f"Command failed with code {cp.returncode}: {' '.join(cmd)}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    if not expect_success and cp.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {' '.join(cmd)}\nSTDOUT:\n{cp.stdout}")
    return cp


def target_row(test_kernel: str, baseline_kernel: str, threads: int) -> Dict[str, Any]:
    return {
        "scope": "thread_sweep",
        "gpu": "Synthetic H100",
        "architecture_generation": "Hopper",
        "architecture_chip": "gh100",
        "test_kernel": test_kernel,
        "baseline_kernel": baseline_kernel,
        "threads": threads,
        "unroll": 16,
        "threads_per_sm": 1024,
        "measurement_grade": "strict_nvml_counter",
        "baseline_match_grade": "structural_baseline",
        "target_pass": "true",
        "quality_pass": "true",
        "quality_gate_selected_target": "true",
        "util_saturated": "true",
        "util_reference_scope": "quality_pass",
        "util_reference_max_pct": 97.5,
        "util_metric_source": "avg_sm_util_pct_mean",
        "target_selection_note": "quality_gate_first_saturation_point",
        "energy_trace_crosscheck_pass": "true",
        "energy_source_reliable": "true",
        "baseline_structural_match": "true",
        "benchmark_schema_current": "true",
        "test_benchmark_schema_versions": "fp16-energy-bench-v2",
        "baseline_benchmark_schema_versions": "fp16-energy-bench-v2",
        "ncu_required": "true",
        "ncu_validation_pass": "true",
        "ncu_validation_context_match": "true",
        "valid_no_l2_count": 3,
        "required_valid_count": 3,
        "avg_sm_util_pct_mean": 97.5,
        "tflops_mean": 820.0,
        "elapsed_s_mean": 1.25,
        "baseline_elapsed_s_mean": 1.18,
        "tensor_peak_tflops_model_mean": 1000.0,
        "achieved_flops_per_sm_cycle_mean": 4100.0,
        "tensor_model_utilization_pct_mean": 82.0,
        "matmul_denominator_valid": "true",
        "matmul_denominator_metadata_complete": "true",
        "matmul_denominator_source": "bench_json_metadata",
        "matmul_input_bits_per_logical_mma": 8192,
        "matmul_flops_per_logical_mma": 8192,
        "matmul_logical_mma_count_mean": 2500000,
        "matmul_input_pj_per_bit_mean": 0.123,
        "incremental_power_w_mean": 8.0,
        "test_energy_j_mean": 120.0,
        "incremental_energy_j_mean": 7.2,
        "incremental_energy_fraction_mean": 0.06,
        "baseline_energy_fraction_mean": 0.94,
        "baseline_power_fraction_mean": 0.94,
        "test_energy_counter_vs_trace_ratio_mean": 1.02,
        "baseline_energy_counter_vs_trace_ratio_mean": 0.98,
        "test_energy_counter_vs_trace_delta_j_mean": 2.0,
        "baseline_energy_counter_vs_trace_delta_j_mean": -1.5,
    }


def ncu_row(kernel: str, threads: int) -> Dict[str, Any]:
    return {
        "kernel": kernel,
        "threads": threads,
        "validation_pass": "true",
        "memory_counter_classes_complete": "true",
        "validation_blocks_per_sm": 8,
        "validation_unroll": 16,
        "validation_suppress_output_store": "true",
        "l2_counter_total": 0,
        "dram_counter_total": 0,
        "local_counter_total": 0,
        "tensor_activity_observed": "true",
        "tensor_activity_pct": 81.0 if kernel == "tensor_mma_f16acc" else 0.0,
        "sm_activity_pct": 92.0,
    }


def resource_row(role: str, kernel: str, threads: int) -> Dict[str, Any]:
    return {
        "role": role,
        "kernel": kernel,
        "threads": threads,
        "unroll": 16,
        "registers_per_thread": 48 if role == "test" else 24,
        "thread_occupancy_pct_model": 50.0,
        "has_spills": "false",
    }


def write_result_dir(path: Path, *, include_required_target: bool) -> None:
    wrong = target_row("fp16_half2", "baseline_regmove", 64)
    required = target_row("tensor_mma_f16acc", "tensor_baseline_u32", 128)
    selected_targets = [wrong, required] if include_required_target else [wrong]
    write_json(path / "quality_gate_summary.json", {"selected_targets": selected_targets})
    write_csv(path / "quality_gates.csv", [wrong, required])
    write_csv(
        path / "ncu_no_l2_thread_sweep" / "ncu_validation_summary.csv",
        [
            ncu_row("fp16_half2", 64),
            ncu_row("baseline_regmove", 64),
            ncu_row("tensor_mma_f16acc", 128),
            ncu_row("tensor_baseline_u32", 128),
        ],
    )
    write_csv(
        path / "resource_audit" / "thread_resource_occupancy.csv",
        [
            resource_row("test", "fp16_half2", 64),
            resource_row("baseline", "baseline_regmove", 64),
            resource_row("test", "tensor_mma_f16acc", 128),
            resource_row("baseline", "tensor_baseline_u32", 128),
        ],
    )
    write_json(
        path / "strict_pipeline_manifest.json",
        {
            "manifest_schema": "fp16-strict-pipeline-manifest-v1",
            "status": "completed",
            "parameters": {"cuda_arch": "90", "threads": "", "diagnostic_no_ncu": False},
            "git": {"head": {"stdout": "synthetic-smoke"}},
            "artifacts": {
                "binary": {"sha256": "0" * 64},
                "quality_gate_summary": {"exists": True},
                "ncu_validation_summary": {"exists": True},
                "resource_audit": {"exists": True},
            },
            "invocation": "synthetic strict smoke",
        },
    )


def write_model_dir(path: Path) -> None:
    write_csv(
        path / "architecture_model_summary.csv",
        [
            {
                "architecture_chip": "gh100",
                "recommended_cuda_arch": "90",
                "dense_tensor_fp16_flop_per_sm_cycle": 4096,
                "reference_sm_count": 132,
                "reference_boost_clock_mhz": 1830,
                "reference_dense_tensor_fp16_tflops": 989,
                "derived_dense_tensor_fp16_tflops": 989,
                "reference_error_pct": 0.0,
            }
        ],
    )


def write_preflight(path: Path) -> None:
    write_json(
        path,
        {
            "preflight_schema": "fp16-strict-architecture-suite-preflight-v1",
            "overall_pass": True,
            "dry_run": False,
        },
    )


def assert_requirement_pass(path: Path, requirement: str) -> None:
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("requirement") == requirement:
            if row.get("status") != "pass":
                raise AssertionError(f"Requirement {requirement!r} did not pass: {row}")
            return
    raise AssertionError(f"Requirement {requirement!r} not found in {path}")


def assert_requirement_fail(path: Path, requirement: str) -> None:
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("requirement") == requirement:
            if row.get("status") == "pass":
                raise AssertionError(f"Requirement {requirement!r} unexpectedly passed: {row}")
            return
    raise AssertionError(f"Requirement {requirement!r} not found in {path}")


def smoke(base: Path, env: Dict[str, str]) -> None:
    good = base / "good"
    no_required = base / "no_required"
    model_dir = base / "models"
    preflight = base / "strict_architecture_suite_preflight.json"
    write_result_dir(good, include_required_target=True)
    write_result_dir(no_required, include_required_target=False)
    write_model_dir(model_dir)
    write_preflight(preflight)

    audit_good = base / "audit_good"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "audit_strict_results.py"),
            "--input",
            str(good),
            "--outdir",
            str(audit_good),
            "--require-architectures",
            "gh100",
        ],
        cwd=ROOT,
        env=env,
    )
    good_row = read_single_csv_row(audit_good / "strict_result_audit.csv")
    if good_row.get("audit_pass") != "True":
        raise AssertionError(f"Positive strict audit did not pass: {good_row}")
    if good_row.get("test_kernel") != "tensor_mma_f16acc":
        raise AssertionError(f"Strict audit selected the wrong test kernel: {good_row}")
    if good_row.get("baseline_kernel") != "tensor_baseline_u32":
        raise AssertionError(f"Strict audit selected the wrong baseline kernel: {good_row}")
    if good_row.get("target_selection_source") != "selected_targets_required_kernel_baseline":
        raise AssertionError(f"Strict audit did not record required target selection: {good_row}")
    if good_row.get("selected_target_count") != "2" or good_row.get("matching_selected_target_count") != "1":
        raise AssertionError(f"Strict audit target-selection counts are wrong: {good_row}")

    audit_bad = base / "audit_no_required"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "audit_strict_results.py"),
            "--input",
            str(no_required),
            "--outdir",
            str(audit_bad),
            "--require-architectures",
            "gh100",
            "--no-fail",
        ],
        cwd=ROOT,
        env=env,
    )
    bad_row = read_single_csv_row(audit_bad / "strict_result_audit.csv")
    if bad_row.get("audit_pass") != "False":
        raise AssertionError(f"Negative strict audit unexpectedly passed: {bad_row}")
    expected_reason = "quality_gate_summary has no selected target matching tensor_mma_f16acc/tensor_baseline_u32"
    if expected_reason not in bad_row.get("fail_reasons", ""):
        raise AssertionError(f"Negative strict audit missed required-target failure: {bad_row}")

    report_dir = base / "report_good"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "report_strict_results.py"),
            "--audit-dir",
            str(audit_good),
            "--architecture-model-dir",
            str(model_dir),
            "--suite-preflight-json",
            str(preflight),
            "--outdir",
            str(report_dir),
            "--require-architectures",
            "gh100",
            "--fail-on-missing-requirements",
        ],
        cwd=ROOT,
        env=env,
    )
    assert_requirement_pass(report_dir / "fp16_strict_report_requirements.csv", "required kernel target selected")
    assert_requirement_pass(report_dir / "fp16_strict_report_requirements.csv", "suite preflight passed")
    report_row = read_single_csv_row(report_dir / "fp16_strict_report_summary.csv")
    if report_row.get("target_selection_source") != "selected_targets_required_kernel_baseline":
        raise AssertionError(f"Report did not carry target selection source: {report_row}")

    report_bad = base / "report_no_required"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "report_strict_results.py"),
            "--audit-dir",
            str(audit_bad),
            "--architecture-model-dir",
            str(model_dir),
            "--suite-preflight-json",
            str(preflight),
            "--outdir",
            str(report_bad),
            "--require-architectures",
            "gh100",
            "--fail-on-missing-requirements",
        ],
        cwd=ROOT,
        env=env,
        expect_success=False,
    )
    assert_requirement_fail(report_bad / "fp16_strict_report_requirements.csv", "strict audit overall pass")
    assert_requirement_fail(report_bad / "fp16_strict_report_requirements.csv", "required kernel target selected")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU-free strict FP16 pipeline smoke tests")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the generated temporary smoke directory")
    parser.add_argument("--workdir", type=Path, default=None, help="Use an explicit smoke working directory")
    args = parser.parse_args()

    temp: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        base = args.workdir.resolve()
        base.mkdir(parents=True, exist_ok=True)
        cleanup = False
    elif args.keep_temp:
        base = Path(tempfile.mkdtemp(prefix="fp16_strict_smoke_"))
        cleanup = False
    else:
        temp = tempfile.TemporaryDirectory(prefix="fp16_strict_smoke_")
        base = Path(temp.name)
        cleanup = True

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(base / "mplconfig")
    try:
        smoke(base, env)
        print(f"Strict FP16 smoke passed: {base}")
    finally:
        if args.workdir:
            pass
        elif cleanup and temp is not None:
            temp.cleanup()
        else:
            print(f"Kept smoke directory: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
