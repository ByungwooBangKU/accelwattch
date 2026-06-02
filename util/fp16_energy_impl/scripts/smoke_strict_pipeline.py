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


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


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
        "blocks_per_sm_requested": 8,
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


def ncu_row(kernel: str, threads: int, *, tensor_activity_observed: bool | None = None) -> Dict[str, Any]:
    if tensor_activity_observed is None:
        tensor_activity_observed = kernel == "tensor_mma_f16acc"
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
        "tensor_activity_observed": "true" if tensor_activity_observed else "false",
        "tensor_activity_pct": 81.0 if kernel == "tensor_mma_f16acc" else 0.0,
        "sm_activity_pct": 92.0,
    }


def resource_row(role: str, kernel: str, threads: int) -> Dict[str, Any]:
    return {
        "role": role,
        "kernel": kernel,
        "threads": threads,
        "blocks_per_sm_requested": 8,
        "unroll": 16,
        "registers_per_thread": 48 if role == "test" else 24,
        "thread_occupancy_pct_model": 50.0,
        "has_spills": "false",
    }


def compare_thread_row(
    threads: int,
    threads_per_sm: int,
    sm_util: float,
    pjbit: float,
    *,
    target: bool,
    blocks_per_sm: int = 8,
    measurement_grade: str = "strict_nvml_counter",
) -> Dict[str, Any]:
    row = target_row("tensor_mma_f16acc", "tensor_baseline_mov", threads)
    incremental_fraction = 0.03 if threads < 64 else (0.06 if target else 0.07)
    row.update(
        {
            "measurement_grade": measurement_grade,
            "threads_per_sm": threads_per_sm,
            "blocks_per_sm_requested": blocks_per_sm,
            "avg_sm_util_pct_mean": sm_util,
            "matmul_input_pj_per_bit_mean": pjbit,
            "tflops_mean": 600.0 + sm_util,
            "tensor_model_utilization_pct_mean": sm_util * 0.8,
            "incremental_energy_fraction_mean": incremental_fraction,
            "baseline_energy_fraction_mean": 1.0 - incremental_fraction,
            "baseline_power_fraction_mean": 1.0 - incremental_fraction,
            "pure_fp16_candidate_count": 3,
            "valid_no_l2_count": 3,
            "valid_count": 3,
            "selected_optimal": "true" if target else "false",
            "target_pass": "true" if target else "false",
            "quality_gate_selected_target": "true" if target else "false",
            "util_saturated": "true" if target else "false",
            "target_selection_note": (
                "quality_gate_first_saturation_point" if target else "quality_pass_below_saturation_band"
            ),
        }
    )
    return row


def write_compare_dir(path: Path, *, measurement_grade: str = "strict_nvml_counter") -> None:
    rows = [
        compare_thread_row(32, 256, 60.0, 0.22, target=False, measurement_grade=measurement_grade),
        compare_thread_row(64, 256, 95.8, 0.18, target=True, blocks_per_sm=4, measurement_grade=measurement_grade),
        compare_thread_row(64, 512, 96.1, 0.17, target=False, blocks_per_sm=8, measurement_grade=measurement_grade),
        compare_thread_row(96, 768, 96.05, 0.17, target=False, measurement_grade=measurement_grade),
    ]
    seed = {
        "gpu": "Synthetic H100",
        "device_name": "Synthetic H100",
        "compute_capability": "9.0",
        "architecture_generation": "hopper",
        "architecture_chip": "gh100",
        "gpu_product_class": "datacenter",
        "recommended_cuda_arch": "90",
    }
    condition = {**seed, **rows[1]}
    write_csv(path / "condition_summary.csv", [condition])
    write_csv(path / "summary.csv", [condition])
    write_csv(path / "thread_sweep_summary.csv", rows)
    write_csv(path / "quality_gates.csv", rows)
    write_csv(
        path / "resource_audit" / "thread_resource_occupancy.csv",
        [
            {**seed, **resource_row("test", "tensor_mma_f16acc", 32), "threads_per_sm": 256},
            {**seed, **resource_row("test", "tensor_mma_f16acc", 64), "threads_per_sm": 512},
            {**seed, **resource_row("test", "tensor_mma_f16acc", 96), "threads_per_sm": 768},
        ],
    )


def quality_gate_summary_row(*, energy_source: str = "nvml_total_energy_counter") -> Dict[str, Any]:
    features = "nvml_timed_energy_counter,explicit_m16n16k16_denominator,strict_denominator_provenance"
    row = target_row("tensor_mma_f16acc", "tensor_baseline_mov", 128)
    row.update(
        {
            "condition": "synthetic_quality_gate",
            "repeat_index": 0,
            "fp16_path": "tensor_mma_f16acc_vs_tensor_baseline_mov",
            "blocks_per_sm_requested": 8,
            "suppress_output_store": "true",
            "test_energy_source": energy_source,
            "baseline_energy_source": energy_source,
            "test_power_samples": 10,
            "baseline_power_samples": 10,
            "valid_basic": "true",
            "valid_no_l2": "true",
            "pure_fp16_candidate": "true",
            "energy_sources_match": "true",
            "clock_span_mhz": 5.0,
            "benchmark_uses_wgmma": "false",
            "test_sm_util_samples": 2,
            "incremental_energy_fraction": 0.06,
            "baseline_energy_fraction": 0.94,
            "elapsed_s": 1.25,
            "baseline_elapsed_s": 1.18,
            "test_energy_j": 120.0,
            "incremental_energy_j": 7.2,
            "test_benchmark_schema_version": "fp16-energy-bench-v2",
            "baseline_benchmark_schema_version": "fp16-energy-bench-v2",
            "test_benchmark_schema_features": features,
            "baseline_benchmark_schema_features": features,
            "matmul_logical_mma_count": 2500000,
            "clock_span_mhz_mean": 5.0,
        }
    )
    return row


def quality_gate_thread_row(*, energy_source: str = "nvml_total_energy_counter") -> Dict[str, Any]:
    features = "nvml_timed_energy_counter,explicit_m16n16k16_denominator,strict_denominator_provenance"
    row = quality_gate_summary_row(energy_source=energy_source)
    row.update(
        {
            "run_count": 3,
            "valid_count": 3,
            "valid_no_l2_count": 3,
            "pure_fp16_candidate_count": 3,
            "stats_scope": "valid_no_l2",
            "selected_optimal": "true",
            "benchmark_schema_v2_all": "true",
            "benchmark_schema_features_required_all": "true",
            "test_benchmark_schema_versions": "fp16-energy-bench-v2",
            "baseline_benchmark_schema_versions": "fp16-energy-bench-v2",
            "test_benchmark_schema_features": features,
            "baseline_benchmark_schema_features": features,
        }
    )
    return row


def write_quality_gate_input(
    path: Path,
    *,
    tensor_activity_observed: bool,
    energy_source: str = "nvml_total_energy_counter",
) -> None:
    write_csv(path / "summary.csv", [quality_gate_summary_row(energy_source=energy_source)])
    write_csv(path / "thread_sweep_summary.csv", [quality_gate_thread_row(energy_source=energy_source)])
    write_csv(
        path / "ncu_validation_summary.csv",
        [
            ncu_row("tensor_mma_f16acc", 128, tensor_activity_observed=tensor_activity_observed),
            ncu_row("tensor_baseline_mov", 128),
        ],
    )


def write_quality_gate_model_util_input(path: Path) -> None:
    summary = quality_gate_summary_row()
    thread = quality_gate_thread_row()
    for row in (summary, thread):
        row.update(
            {
                "avg_sm_util_pct_mean": "nan",
                "avg_gpu_util_pct_mean": "nan",
                "avg_sm_util_pct": "nan",
                "avg_gpu_util_pct": "nan",
                "test_sm_util_samples": 0,
                "tensor_model_utilization_pct_mean": 97.5,
            }
        )
    write_csv(path / "summary.csv", [summary])
    write_csv(path / "thread_sweep_summary.csv", [thread])
    write_csv(
        path / "ncu_validation_summary.csv",
        [
            ncu_row("tensor_mma_f16acc", 128, tensor_activity_observed=True),
            ncu_row("tensor_baseline_mov", 128),
        ],
    )


def write_result_dir(path: Path, *, include_required_target: bool, required_tensor_activity: bool = True) -> None:
    wrong = target_row("fp16_half2", "baseline_regmove", 64)
    required = target_row("tensor_mma_f16acc", "tensor_baseline_mov", 128)
    selected_targets = [wrong, required] if include_required_target else [wrong]
    write_preflight(path / "strict_pipeline_preflight.json")
    write_json(
        path / "ncu_permission_probe" / "ncu_permission_probe.json",
        {
            "status": "pass",
            "permission_probe_pass": True,
            "permission_denied": False,
            "profiler_errors": [],
            "fail_reasons": [],
            "returncode": 0,
            "log_file": str(path / "ncu_permission_probe" / "ncu_permission_probe.ncu.txt"),
        },
    )
    write_csv(
        path / "ncu_permission_probe" / "ncu_permission_probe.csv",
        [
            {
                "status": "pass",
                "permission_probe_pass": "true",
                "permission_denied": "false",
                "returncode": 0,
                "log_file": str(path / "ncu_permission_probe" / "ncu_permission_probe.ncu.txt"),
            }
        ],
    )
    (path / "ncu_permission_probe" / "ncu_permission_probe.ncu.txt").write_text(
        "==PROF== Synthetic NCU permission probe passed\n"
    )
    write_csv(
        path / "strict_pipeline_preflight.csv",
        [
            {
                "label": "synthetic",
                "gpu": "0",
                "cuda_arch": "90",
                "nvidia_smi_id": "GPU-synthetic",
                "preflight_pass": "True",
                "required_tools_pass": "True",
                "overall_preflight_pass": "True",
                "publishable_preflight_pass": "True",
            }
        ],
    )
    write_json(path / "quality_gate_summary.json", {"selected_targets": selected_targets})
    write_csv(path / "quality_gates.csv", [wrong, required])
    write_csv(
        path / "ncu_no_l2_thread_sweep" / "ncu_validation_summary.csv",
        [
            ncu_row("fp16_half2", 64),
            ncu_row("baseline_regmove", 64),
            ncu_row("tensor_mma_f16acc", 128, tensor_activity_observed=required_tensor_activity),
            ncu_row("tensor_baseline_mov", 128),
        ],
    )
    write_csv(
        path / "resource_audit" / "thread_resource_occupancy.csv",
        [
            resource_row("test", "fp16_half2", 64),
            resource_row("baseline", "baseline_regmove", 64),
            resource_row("test", "tensor_mma_f16acc", 128),
            resource_row("baseline", "tensor_baseline_mov", 128),
        ],
    )
    write_json(
        path / "strict_pipeline_manifest.json",
        {
            "manifest_schema": "fp16-strict-pipeline-manifest-v1",
            "status": "completed",
            "parameters": {
                "cuda_arch": "90",
                "nvidia_smi_id": "GPU-synthetic",
                "threads": "",
                "ncu_blocks_per_sm_csv": "8",
                "skip_preflight": False,
                "allow_compute_apps": False,
                "diagnostic_no_ncu": False,
            },
            "git": {"head": {"stdout": "synthetic-smoke"}},
            "artifacts": {
                "binary": {"sha256": "0" * 64},
                "pipeline_preflight_json": {"exists": True},
                "pipeline_preflight_csv": {"exists": True},
                "quality_gate_summary": {"exists": True},
                "ncu_permission_probe_json": {"exists": True},
                "ncu_permission_probe_csv": {"exists": True},
                "ncu_permission_probe_log": {"exists": True},
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
                "reference_sparse_tensor_fp16_tflops": 1978,
                "sparse_to_dense_ratio": 2.0,
                "sparsity_mode": "dense_no_sparsity",
                "uses_wgmma_model": "false",
                "reference_source_url": "https://www.nvidia.com/en-us/data-center/h100/",
                "tensor_core_architecture_source_url": (
                    "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"
                ),
                "reference_note": "Synthetic GH100 dense FP16 Tensor Core peak, no sparsity",
                "dense_reference_formula": (
                    "dense_tensor_fp16_flop_per_sm_cycle * reference_sm_count * "
                    "reference_boost_clock_mhz * 1e6 / 1e12"
                ),
                "common_tensor_instruction_path": (
                    "warp-level HMMA mma.sync.aligned.m16n8k16 pair -> logical m16n16k16"
                ),
                "normalization_scope": "dense FP16 Tensor Core HMMA normalization only; no sparsity, no WGMMA",
                "normalization_note": (
                    "Dense Tensor Core peak is used only to normalize measured FP16 HMMA throughput; "
                    "it is not an energy source and does not imply H100 WGMMA was benchmarked."
                ),
            }
        ],
    )


def write_preflight(path: Path) -> None:
    write_json(
        path,
        {
            "preflight_schema": "fp16-strict-architecture-suite-preflight-v1",
            "overall_pass": True,
            "required_tools_pass": True,
            "dry_run": False,
            "cuda_toolchain_compatibility": {
                "checked": True,
                "pass": True,
                "nvcc_release": "12.4",
                "driver_cuda_version": "12.4",
            },
            "rows": [
                {
                    "label": "synthetic",
                    "gpu": "0",
                    "cuda_arch": "90",
                    "nvidia_smi_id": "GPU-synthetic",
                    "preflight_pass": True,
                }
            ],
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
    import analyze_results
    import quality_gate as quality_gate_module

    good = base / "good"
    no_required = base / "no_required"
    no_tensor_activity = base / "no_tensor_activity"
    model_dir = base / "models"
    preflight = base / "strict_architecture_suite_preflight.json"
    write_result_dir(good, include_required_target=True)
    write_result_dir(no_required, include_required_target=False)
    write_result_dir(no_tensor_activity, include_required_target=True, required_tensor_activity=False)

    quality_gate_ok = base / "quality_gate_tensor_activity_ok"
    quality_gate_bad = base / "quality_gate_tensor_activity_bad"
    quality_gate_model_util = base / "quality_gate_model_util_fallback"
    write_quality_gate_input(quality_gate_ok, tensor_activity_observed=True)
    write_quality_gate_input(quality_gate_bad, tensor_activity_observed=False)
    write_quality_gate_model_util_input(quality_gate_model_util)
    write_model_dir(model_dir)
    write_preflight(preflight)

    bad_preflight_json = base / "bad_toolchain_preflight.json"
    bad_preflight_csv = base / "bad_toolchain_preflight.csv"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "preflight_strict_architecture_suite.py"),
            "--spec",
            "rtx3090:0:86",
            "--out-json",
            str(bad_preflight_json),
            "--out-csv",
            str(bad_preflight_csv),
            "--cmake-bin",
            str(base / "missing-cmake"),
            "--nvcc-bin",
            "/bin/true",
            "--ncu-bin",
            "/bin/true",
            "--nvidia-smi-bin",
            "/bin/true",
            "--require-ncu",
            "--no-fail",
        ],
        cwd=ROOT,
        env=env,
    )
    bad_preflight_row = read_single_csv_row(bad_preflight_csv)
    if bad_preflight_row.get("required_tools_pass") != "False":
        raise AssertionError(f"Preflight CSV did not expose toolchain failure: {bad_preflight_row}")
    if bad_preflight_row.get("overall_preflight_pass") != "False":
        raise AssertionError(f"Preflight CSV did not expose overall failure: {bad_preflight_row}")
    if "cmake not found on PATH" not in bad_preflight_row.get("required_tool_fail_reasons", ""):
        raise AssertionError(f"Preflight CSV missed required tool failure reason: {bad_preflight_row}")
    if "nvidia-smi GPU metadata query returned incomplete output" not in bad_preflight_row.get("fail_reasons", ""):
        raise AssertionError(f"Preflight CSV missed malformed GPU metadata failure: {bad_preflight_row}")

    fake_ncu_denied = base / "fake_ncu_denied"
    write_executable(
        fake_ncu_denied,
        """#!/usr/bin/env bash
log_file=""
prev=""
for arg in "$@"; do
  if [[ "${prev}" == "--log-file" ]]; then
    log_file="${arg}"
  fi
  prev="${arg}"
done
if [[ -n "${log_file}" ]]; then
  mkdir -p "$(dirname "${log_file}")"
  cat > "${log_file}" <<'EOF'
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
==WARNING== No kernels were profiled.
EOF
fi
exit 1
""",
    )
    denied_probe_dir = base / "ncu_probe_denied"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "probe_ncu_permissions.py"),
            "--binary",
            "/bin/true",
            "--outdir",
            str(denied_probe_dir),
            "--ncu-bin",
            str(fake_ncu_denied),
        ],
        cwd=ROOT,
        env=env,
        expect_success=False,
    )
    denied_probe = json.loads((denied_probe_dir / "ncu_permission_probe.json").read_text())
    if denied_probe.get("status") != "permission_denied" or not denied_probe.get("permission_denied"):
        raise AssertionError(f"NCU permission probe missed ERR_NVGPUCTRPERM: {denied_probe}")

    fake_ncu_ok = base / "fake_ncu_ok"
    write_executable(
        fake_ncu_ok,
        """#!/usr/bin/env bash
log_file=""
prev=""
for arg in "$@"; do
  if [[ "${prev}" == "--log-file" ]]; then
    log_file="${arg}"
  fi
  prev="${arg}"
done
if [[ -n "${log_file}" ]]; then
  mkdir -p "$(dirname "${log_file}")"
  echo "==PROF== Disconnected from process 1" > "${log_file}"
fi
exit 0
""",
    )
    ok_probe_dir = base / "ncu_probe_ok"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "probe_ncu_permissions.py"),
            "--binary",
            "/bin/true",
            "--outdir",
            str(ok_probe_dir),
            "--ncu-bin",
            str(fake_ncu_ok),
        ],
        cwd=ROOT,
        env=env,
    )
    ok_probe = json.loads((ok_probe_dir / "ncu_permission_probe.json").read_text())
    if ok_probe.get("status") != "pass" or not ok_probe.get("permission_probe_pass"):
        raise AssertionError(f"NCU permission probe did not accept successful profiling command: {ok_probe}")

    fake_nvcc = base / "fake_nvcc_13_2"
    fake_nvidia_smi = base / "fake_nvidia_smi_cuda_13_1"
    write_executable(
        fake_nvcc,
        """#!/usr/bin/env bash
cat <<'EOF'
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 13.2, V13.2.78
EOF
""",
    )
    write_executable(
        fake_nvidia_smi,
        """#!/usr/bin/env bash
case "$*" in
  *--version*)
    echo "NVIDIA-SMI 570.00 Driver Version: 570.00 CUDA Version: 13.1"
    exit 0
    ;;
  *--query-compute-apps*)
    exit 0
    ;;
  *--query-gpu*)
    echo "0, GPU-synthetic-3090, 00000000:01:00.0, NVIDIA GeForce RTX 3090, 570.00, 350.00, 25.0, P8, 210, 9751, 45"
    exit 0
    ;;
esac
echo "unexpected fake nvidia-smi args: $*" >&2
exit 1
""",
    )
    compat_preflight_json = base / "bad_compat_preflight.json"
    compat_preflight_csv = base / "bad_compat_preflight.csv"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "preflight_strict_architecture_suite.py"),
            "--spec",
            "rtx3090:0:86",
            "--out-json",
            str(compat_preflight_json),
            "--out-csv",
            str(compat_preflight_csv),
            "--cmake-bin",
            "/bin/true",
            "--nvcc-bin",
            str(fake_nvcc),
            "--ncu-bin",
            "/bin/true",
            "--nvidia-smi-bin",
            str(fake_nvidia_smi),
            "--require-ncu",
            "--no-fail",
        ],
        cwd=ROOT,
        env=env,
    )
    compat_row = read_single_csv_row(compat_preflight_csv)
    if compat_row.get("required_tools_pass") != "False":
        raise AssertionError(f"Preflight CSV did not fail incompatible nvcc: {compat_row}")
    if compat_row.get("toolchain_compatibility_pass") != "False":
        raise AssertionError(f"Preflight CSV missed toolchain compatibility failure: {compat_row}")
    if compat_row.get("toolchain_nvcc_release") != "13.2":
        raise AssertionError(f"Preflight CSV missed nvcc release: {compat_row}")
    if compat_row.get("toolchain_driver_cuda_version") != "13.1":
        raise AssertionError(f"Preflight CSV missed driver CUDA version: {compat_row}")
    recovery = compat_row.get("toolchain_recovery_commands", "")
    if "--gpu-kind rtx3090 --cuda-version 12.1" not in recovery:
        raise AssertionError(f"Preflight CSV missed RTX3090 recovery command: {compat_row}")
    compat_payload = json.loads(compat_preflight_json.read_text())
    compat = compat_payload.get("cuda_toolchain_compatibility", {})
    if not compat.get("needs_compatible_toolchain"):
        raise AssertionError(f"Preflight JSON missed needs_compatible_toolchain: {compat}")
    if "Recommended strict-suite toolkit: CUDA 12.1" not in "; ".join(compat.get("fail_reasons", [])):
        raise AssertionError(f"Preflight JSON missed recommended toolkit reason: {compat}")

    architecture_model_smoke = base / "architecture_model_smoke"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "architecture_models.py"),
            "--outdir",
            str(architecture_model_smoke),
            "--fail-on-model-error-pct",
            "1.0",
            "--fail-on-missing-metadata",
        ],
        cwd=ROOT,
        env=env,
    )
    for artifact in (
        "architecture_model_summary.csv",
        "architecture_model_dense_peak.png",
        "architecture_model_per_sm_capacity.png",
        "architecture_model_resource_limits.png",
    ):
        if not (architecture_model_smoke / artifact).exists():
            raise AssertionError(f"Architecture model smoke did not write {artifact}")

    launch_shape_rows = [
        {
            "fp16_path": "tensor_mma_f16acc_vs_tensor_baseline_mov",
            "test_kernel": "tensor_mma_f16acc",
            "baseline_kernel": "tensor_baseline_mov",
            "threads": threads,
            "blocks_per_sm_requested": blocks_per_sm,
            "threads_per_sm": threads * blocks_per_sm,
            "valid_basic": True,
            "expected_l2_touch": False,
            "test_energy_source": "nvml_total_energy_counter",
            "baseline_energy_source": "nvml_total_energy_counter",
            "test_power_samples": 4,
            "baseline_power_samples": 4,
        }
        for threads, blocks_per_sm in ((64, 4), (64, 8), (128, 4))
    ]
    shape_summary = analyze_results.aggregate_thread_sweep(launch_shape_rows)
    shape_keys = {(str(row.get("threads")), str(row.get("blocks_per_sm_requested"))) for row in shape_summary}
    if ("64", "4") not in shape_keys or ("64", "8") not in shape_keys:
        raise AssertionError(f"Thread sweep aggregation collapsed distinct launch shapes: {shape_summary}")
    source_keys = set(quality_gate_module.source_counts_by_thread(launch_shape_rows))
    if (
        ("tensor_mma_f16acc", "tensor_baseline_mov", "64", "4") not in source_keys
        or ("tensor_mma_f16acc", "tensor_baseline_mov", "64", "8") not in source_keys
    ):
        raise AssertionError(f"Quality gate source grouping collapsed distinct launch shapes: {source_keys}")

    compare_input = base / "compare_input"
    compare_out = base / "compare_out"
    compare_power_trace_input = base / "compare_power_trace_input"
    compare_power_trace_out = base / "compare_power_trace_out"
    write_compare_dir(compare_input)
    write_compare_dir(compare_power_trace_input, measurement_grade="power_trace_fallback")
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "compare_architectures.py"),
            "--input",
            str(compare_input),
            "--outdir",
            str(compare_out),
        ],
        cwd=ROOT,
        env=env,
    )
    for artifact in (
        "architecture_best_fp16.csv",
        "architecture_strict_coverage.csv",
        "architecture_strict_coverage.png",
        "architecture_thread_sweep_util_tensor_mma_f16acc_vs_tensor_baseline_mov.png",
        "architecture_thread_sweep_pjbit_tensor_mma_f16acc_vs_tensor_baseline_mov.png",
        "architecture_thread_sweep_energy_fraction_tensor_mma_f16acc_vs_tensor_baseline_mov.png",
    ):
        if not (compare_out / artifact).exists():
            raise AssertionError(f"Architecture compare smoke did not write {artifact}")
    coverage = {row.get("architecture_chip"): row for row in read_csv_rows(compare_out / "architecture_strict_coverage.csv")}
    if coverage.get("gh100", {}).get("coverage_status") != "strict_pass":
        raise AssertionError(f"Strict coverage did not mark GH100 as strict_pass: {coverage}")
    for missing_chip in ("ga100", "ga102"):
        if coverage.get(missing_chip, {}).get("coverage_status") != "missing_result":
            raise AssertionError(f"Strict coverage did not mark {missing_chip} as missing_result: {coverage}")
    compare_threads = read_csv_rows(compare_out / "architecture_thread_sweep_summary.csv")
    duplicate_launch_shapes = {
        row.get("blocks_per_sm_requested"): row
        for row in compare_threads
        if row.get("threads") == "64"
    }
    if set(duplicate_launch_shapes) != {"4", "8"}:
        raise AssertionError(f"Architecture compare collapsed launch shapes: {duplicate_launch_shapes}")
    if duplicate_launch_shapes["4"].get("target_pass") != "true":
        raise AssertionError(f"Architecture compare lost target_pass for blocks/SM=4: {duplicate_launch_shapes}")
    if duplicate_launch_shapes["8"].get("target_pass") != "false":
        raise AssertionError(f"Architecture compare copied target_pass across blocks/SM=8: {duplicate_launch_shapes}")

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "compare_architectures.py"),
            "--input",
            str(compare_power_trace_input),
            "--outdir",
            str(compare_power_trace_out),
        ],
        cwd=ROOT,
        env=env,
    )
    compare_power_row = read_single_csv_row(compare_power_trace_out / "architecture_best_fp16.csv")
    if compare_power_row.get("quality_rejected") != "True":
        raise AssertionError(f"Power-trace compare row was not rejected by default: {compare_power_row}")
    if compare_power_row.get("selection_note") != "quality_gate_target_pass_without_strict_nvml_counter":
        raise AssertionError(f"Unexpected power-trace compare rejection note: {compare_power_row}")
    power_coverage = {
        row.get("architecture_chip"): row
        for row in read_csv_rows(compare_power_trace_out / "architecture_strict_coverage.csv")
    }
    if power_coverage.get("gh100", {}).get("coverage_status") != "diagnostic_or_rejected_only":
        raise AssertionError(f"Power-trace coverage did not stay diagnostic: {power_coverage}")

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "quality_gate.py"),
            "--input",
            str(quality_gate_ok),
            "--ncu-summary",
            str(quality_gate_ok / "ncu_validation_summary.csv"),
            "--require-ncu",
            "--require-ncu-tensor-activity",
        ],
        cwd=ROOT,
        env=env,
    )
    qg_ok_summary = json.loads((quality_gate_ok / "quality_gate_summary.json").read_text())
    if len(qg_ok_summary.get("selected_targets", [])) != 1:
        raise AssertionError(f"Tensor activity quality gate did not select a target: {qg_ok_summary}")

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "quality_gate.py"),
            "--input",
            str(quality_gate_bad),
            "--ncu-summary",
            str(quality_gate_bad / "ncu_validation_summary.csv"),
            "--require-ncu",
            "--require-ncu-tensor-activity",
        ],
        cwd=ROOT,
        env=env,
    )
    qg_bad_summary = json.loads((quality_gate_bad / "quality_gate_summary.json").read_text())
    if qg_bad_summary.get("selected_targets"):
        raise AssertionError(f"Missing Tensor activity quality gate selected a target: {qg_bad_summary}")
    qg_bad_rows = read_csv_rows(quality_gate_bad / "quality_gates.csv")
    if not any("test NCU tensor activity" in row.get("fail_reasons", "") for row in qg_bad_rows):
        raise AssertionError(f"Missing Tensor activity quality gate did not record the failure: {qg_bad_rows}")

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "quality_gate.py"),
            "--input",
            str(quality_gate_model_util),
            "--ncu-summary",
            str(quality_gate_model_util / "ncu_validation_summary.csv"),
            "--require-ncu",
            "--require-ncu-tensor-activity",
        ],
        cwd=ROOT,
        env=env,
    )
    qg_model_summary = json.loads((quality_gate_model_util / "quality_gate_summary.json").read_text())
    if len(qg_model_summary.get("selected_targets", [])) != 1:
        raise AssertionError(f"Model-util fallback quality gate did not select a target: {qg_model_summary}")
    qg_model_rows = [
        row for row in read_csv_rows(quality_gate_model_util / "quality_gates.csv")
        if row.get("scope") == "thread_sweep"
    ]
    if len(qg_model_rows) != 1:
        raise AssertionError(f"Expected one model-util thread row: {qg_model_rows}")
    qg_model_row = qg_model_rows[0]
    if qg_model_row.get("target_pass") != "True":
        raise AssertionError(f"Model-util fallback row was not selected: {qg_model_row}")
    if qg_model_row.get("util_metric_source") != "tensor_model_utilization_pct_mean":
        raise AssertionError(f"Model-util fallback used the wrong target metric: {qg_model_row}")
    if qg_model_row.get("sm_util_available") != "False":
        raise AssertionError(f"Model-util fallback should record missing SM/GPU telemetry: {qg_model_row}")
    if qg_model_row.get("target_util_available") != "True":
        raise AssertionError(f"Model-util fallback did not record target utilization availability: {qg_model_row}")
    if "SM/GPU utilization missing" not in qg_model_row.get("warnings", ""):
        raise AssertionError(f"Model-util fallback did not warn about telemetry fallback: {qg_model_row}")

    quality_gate_power_trace = base / "quality_gate_power_trace"
    write_quality_gate_input(
        quality_gate_power_trace,
        tensor_activity_observed=True,
        energy_source="power_trace_integral",
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "quality_gate.py"),
            "--input",
            str(quality_gate_power_trace),
            "--ncu-summary",
            str(quality_gate_power_trace / "ncu_validation_summary.csv"),
            "--require-ncu",
            "--require-ncu-tensor-activity",
        ],
        cwd=ROOT,
        env=env,
    )
    qg_power_summary = json.loads((quality_gate_power_trace / "quality_gate_summary.json").read_text())
    if qg_power_summary.get("selected_targets"):
        raise AssertionError(f"Power-trace fallback quality gate selected a target by default: {qg_power_summary}")
    qg_power_rows = read_csv_rows(quality_gate_power_trace / "quality_gates.csv")
    if not any(
        row.get("target_selection_note") == "quality_pass_non_strict_energy_source_diagnostic"
        for row in qg_power_rows
    ):
        raise AssertionError(f"Power-trace fallback diagnostic note was not recorded: {qg_power_rows}")

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
    if good_row.get("baseline_kernel") != "tensor_baseline_mov":
        raise AssertionError(f"Strict audit selected the wrong baseline kernel: {good_row}")
    if good_row.get("target_selection_source") != "selected_targets_required_kernel_baseline":
        raise AssertionError(f"Strict audit did not record required target selection: {good_row}")
    if good_row.get("selected_target_count") != "2" or good_row.get("matching_selected_target_count") != "1":
        raise AssertionError(f"Strict audit target-selection counts are wrong: {good_row}")
    if good_row.get("pipeline_preflight_overall_pass") != "True":
        raise AssertionError(f"Strict audit did not carry passing pipeline preflight evidence: {good_row}")
    if good_row.get("pipeline_preflight_toolchain_pass") != "True":
        raise AssertionError(f"Strict audit did not carry passing toolchain compatibility evidence: {good_row}")
    if good_row.get("pipeline_ncu_permission_probe_recorded") != "True":
        raise AssertionError(f"Strict audit did not carry NCU permission probe artifact evidence: {good_row}")
    if good_row.get("pipeline_ncu_permission_probe_pass") != "True":
        raise AssertionError(f"Strict audit did not carry passing NCU permission probe evidence: {good_row}")
    if good_row.get("pipeline_ncu_permission_probe_permission_denied") != "False":
        raise AssertionError(f"Strict audit did not carry NCU permission-denied evidence: {good_row}")

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
    expected_reason = "quality_gate_summary has no selected target matching tensor_mma_f16acc/tensor_baseline_mov"
    if expected_reason not in bad_row.get("fail_reasons", ""):
        raise AssertionError(f"Negative strict audit missed required-target failure: {bad_row}")

    audit_no_tensor = base / "audit_no_tensor_activity"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "audit_strict_results.py"),
            "--input",
            str(no_tensor_activity),
            "--outdir",
            str(audit_no_tensor),
            "--require-architectures",
            "gh100",
            "--require-ncu-tensor-activity",
            "--no-fail",
        ],
        cwd=ROOT,
        env=env,
    )
    no_tensor_row = read_single_csv_row(audit_no_tensor / "strict_result_audit.csv")
    if no_tensor_row.get("audit_pass") != "False":
        raise AssertionError(f"Missing Tensor activity strict audit unexpectedly passed: {no_tensor_row}")
    expected_tensor_reason = "selected test NCU tensor activity is missing or below threshold"
    if expected_tensor_reason not in no_tensor_row.get("fail_reasons", ""):
        raise AssertionError(f"Missing Tensor activity audit missed expected failure: {no_tensor_row}")

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
    assert_requirement_pass(report_dir / "fp16_strict_report_requirements.csv", "strict pipeline preflight passed")
    assert_requirement_pass(
        report_dir / "fp16_strict_report_requirements.csv",
        "NCU performance-counter permission probe passed",
    )
    assert_requirement_pass(report_dir / "fp16_strict_report_requirements.csv", "architecture model metadata")
    assert_requirement_pass(report_dir / "fp16_strict_report_requirements.csv", "NCU Tensor activity observed")
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
