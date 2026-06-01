#!/usr/bin/env python3
"""Analyze FP16 energy benchmark runs and generate summary tables/figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def read_runs(path: Path) -> List[Dict[str, Any]]:
    runs_file = path / "runs.jsonl"
    rows: List[Dict[str, Any]] = []
    with runs_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_power_csv(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ns = int(r["sample_unix_ns"])
            except Exception:
                continue
            rows.append(
                {
                    "sample_unix_ns": ns,
                    "t_s": ns / 1e9,
                    "power_w": parse_float(r.get("power_w")),
                    "power_draw_w": parse_float(r.get("power_draw_w")),
                    "power_draw_average_w": parse_float(r.get("power_draw_average_w")),
                    "power_draw_instant_w": parse_float(r.get("power_draw_instant_w")),
                    "power_limit_w": parse_float(r.get("power_limit_w")),
                    "sm_clock_mhz": parse_float(r.get("sm_clock_mhz")),
                    "mem_clock_mhz": parse_float(r.get("mem_clock_mhz")),
                    "temp_c": parse_float(r.get("temp_c")),
                    "pstate": r.get("pstate", ""),
                    "util_gpu_pct": parse_float(r.get("util_gpu_pct")),
                    "query_mode": r.get("query_mode", ""),
                }
            )
    return rows


def read_sm_util_csv(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ns = int(r["sample_unix_ns"])
            except Exception:
                continue
            rows.append(
                {
                    "sample_unix_ns": ns,
                    "t_s": ns / 1e9,
                    "sm_util_pct": parse_float(r.get("sm_util_pct")),
                    "mem_util_pct": parse_float(r.get("mem_util_pct")),
                }
            )
    return rows


def integrate_power(samples: List[Dict[str, Any]], start_ns: int, end_ns: int) -> Tuple[float, float, int]:
    """Return (energy_j, avg_power_w, sample_count) inside [start_ns, end_ns].

    Uses trapezoidal integration over samples inside the benchmark host interval.
    If too few samples are available, falls back to arithmetic mean * duration.
    """
    window = [s for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["power_w"] is not None]
    duration_s = max((end_ns - start_ns) / 1e9, 0.0)
    if not window or duration_s <= 0:
        return (math.nan, math.nan, len(window))
    powers = [float(s["power_w"]) for s in window]
    if len(window) >= 2:
        energy = 0.0
        for a, b in zip(window[:-1], window[1:]):
            dt = (b["sample_unix_ns"] - a["sample_unix_ns"]) / 1e9
            energy += 0.5 * (float(a["power_w"]) + float(b["power_w"])) * dt
        # Edge intervals are usually shorter than the sampling period. Use mean-power extension
        # rather than extrapolating the first/last sample aggressively.
        avg = energy / max((window[-1]["sample_unix_ns"] - window[0]["sample_unix_ns"]) / 1e9, 1e-12)
        return (avg * duration_s, avg, len(window))
    avg = sum(powers) / len(powers)
    return (avg * duration_s, avg, len(window))


def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    arch = classify_architecture(run)
    samples = read_power_csv(run.get("power_csv", ""))
    sm_util_samples = read_sm_util_csv(run.get("sm_util_csv", ""))
    start_ns = int(run["host_start_unix_ns"])
    end_ns = int(run["host_end_unix_ns"])
    duration_s = max((end_ns - start_ns) / 1e9, 0.0)
    trace_energy_j, trace_avg_power_w, sample_count = integrate_power(samples, start_ns, end_ns)
    nvml_energy_delta_j = finite_float(run.get("nvml_energy_delta_j"))
    use_nvml_counter = (
        bool(run.get("nvml_energy_supported", False))
        and math.isfinite(nvml_energy_delta_j)
        and nvml_energy_delta_j > 0.0
    )
    if use_nvml_counter:
        energy_j = nvml_energy_delta_j
        avg_power_w = nvml_energy_delta_j / duration_s if duration_s > 0.0 else math.nan
        energy_source = "nvml_total_energy_counter"
    else:
        energy_j = trace_energy_j
        avg_power_w = trace_avg_power_w
        energy_source = "power_trace_integral" if math.isfinite(trace_energy_j) else "unavailable"

    energy_counter_vs_trace_delta_j = (
        nvml_energy_delta_j - trace_energy_j
        if math.isfinite(nvml_energy_delta_j) and math.isfinite(trace_energy_j)
        else math.nan
    )
    energy_counter_vs_trace_ratio = (
        nvml_energy_delta_j / trace_energy_j
        if math.isfinite(nvml_energy_delta_j) and math.isfinite(trace_energy_j) and trace_energy_j > 0.0
        else math.nan
    )
    clocks = [s["sm_clock_mhz"] for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["sm_clock_mhz"] is not None]
    temps = [s["temp_c"] for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["temp_c"] is not None]
    utils = [s["util_gpu_pct"] for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["util_gpu_pct"] is not None]
    power_draws = [s["power_draw_w"] for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["power_draw_w"] is not None]
    power_averages = [
        s["power_draw_average_w"]
        for s in samples
        if start_ns <= s["sample_unix_ns"] <= end_ns and s["power_draw_average_w"] is not None
    ]
    power_instants = [
        s["power_draw_instant_w"]
        for s in samples
        if start_ns <= s["sample_unix_ns"] <= end_ns and s["power_draw_instant_w"] is not None
    ]
    power_limits = [
        s["power_limit_w"]
        for s in samples
        if start_ns <= s["sample_unix_ns"] <= end_ns and s["power_limit_w"] is not None
    ]
    query_modes = sorted({str(s.get("query_mode", "")) for s in samples if str(s.get("query_mode", ""))})
    sm_utils = [
        s["sm_util_pct"]
        for s in sm_util_samples
        if start_ns <= s["sample_unix_ns"] <= end_ns and s["sm_util_pct"] is not None
    ]
    return {
        **run,
        **arch,
        "power_energy_j": energy_j,
        "avg_power_w": avg_power_w,
        "energy_source": energy_source,
        "power_trace_energy_j": trace_energy_j,
        "power_trace_avg_power_w": trace_avg_power_w,
        "power_trace_query_modes": ",".join(query_modes),
        "avg_power_draw_w": sum(power_draws) / len(power_draws) if power_draws else math.nan,
        "avg_power_draw_average_w": sum(power_averages) / len(power_averages) if power_averages else math.nan,
        "avg_power_draw_instant_w": sum(power_instants) / len(power_instants) if power_instants else math.nan,
        "avg_power_limit_w": sum(power_limits) / len(power_limits) if power_limits else math.nan,
        "nvml_energy_delta_j": nvml_energy_delta_j,
        "nvml_energy_counter_supported": bool(run.get("nvml_energy_supported", False)),
        "nvml_energy_note": run.get("nvml_energy_note", ""),
        "energy_counter_vs_trace_delta_j": energy_counter_vs_trace_delta_j,
        "energy_counter_vs_trace_ratio": energy_counter_vs_trace_ratio,
        "power_sample_count": sample_count,
        "sm_util_sample_count": len(sm_utils),
        "avg_sm_clock_mhz": sum(clocks) / len(clocks) if clocks else math.nan,
        "min_sm_clock_mhz": min(clocks) if clocks else math.nan,
        "max_sm_clock_mhz": max(clocks) if clocks else math.nan,
        "avg_temp_c": sum(temps) / len(temps) if temps else math.nan,
        "max_temp_c": max(temps) if temps else math.nan,
        "avg_gpu_util_pct": sum(utils) / len(utils) if utils else math.nan,
        "max_gpu_util_pct": max(utils) if utils else math.nan,
        "avg_sm_util_pct": sum(sm_utils) / len(sm_utils) if sm_utils else math.nan,
        "max_sm_util_pct": max(sm_utils) if sm_utils else math.nan,
    }


def finite_float(x: Any, default: float = math.nan) -> float:
    value = parse_float(x)
    return value if value is not None else default


def classify_architecture(run: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized architecture tags for new and legacy benchmark JSON."""
    generation = str(run.get("architecture_generation", "") or "")
    chip = str(run.get("architecture_chip", "") or "")
    product_class = str(run.get("gpu_product_class", "") or "")
    cuda_arch = str(run.get("recommended_cuda_arch", "") or "")
    path = str(
        run.get(
            "fp16_tensor_instruction_path",
            "benchmark uses warp-level HMMA mma.sync m16n8k16 pairs",
        )
        or ""
    )
    note = str(run.get("architecture_measurement_note", "") or "")
    wgmma_supported = bool(run.get("wgmma_supported", False))
    benchmark_uses_wgmma = bool(run.get("benchmark_uses_wgmma", False))

    if generation and chip and cuda_arch:
        return {
            "architecture_generation": generation,
            "architecture_chip": chip,
            "gpu_product_class": product_class or "unknown",
            "recommended_cuda_arch": cuda_arch,
            "fp16_tensor_instruction_path": path,
            "wgmma_supported": wgmma_supported,
            "benchmark_uses_wgmma": benchmark_uses_wgmma,
            "architecture_measurement_note": note,
        }

    name = str(run.get("device_name", "") or "")
    cc = str(run.get("compute_capability", "") or "")
    generation = "unknown"
    chip = "unknown"
    product_class = "unknown"
    cuda_arch = cc.replace(".", "") if cc else ""
    note = "unknown GPU architecture; inspect compute capability and validation counters before comparison"

    if cc.startswith("9."):
        generation = "hopper"
        chip = "gh100" if "H100" in name else "hopper_sm90"
        product_class = "datacenter"
        cuda_arch = "90"
        wgmma_supported = True
        note = (
            "H100/Hopper supports WGMMA, but this benchmark uses the same warp-level HMMA "
            "m16n8k16 pair path as Ampere for cross-GPU comparison"
        )
    elif cc == "8.0":
        generation = "ampere"
        chip = "ga100" if "A100" in name else "ampere_sm80"
        product_class = "datacenter"
        cuda_arch = "80"
        note = "A100/GA100-class HMMA path; compare with the same workload, clocks, and baseline subtraction"
    elif cc == "8.6":
        generation = "ampere"
        chip = "ga102" if "3090" in name else "ampere_sm86"
        product_class = "consumer" if "RTX" in name else "workstation_or_consumer"
        cuda_arch = "86"
        note = "RTX/GA10x-class HMMA path; validate clock stability and no-L2 behavior before using pJ/bit"
    elif cc.startswith("8."):
        generation = "ampere"
        chip = "ampere_sm8x"
        product_class = "unknown"

    return {
        "architecture_generation": generation,
        "architecture_chip": chip,
        "gpu_product_class": product_class,
        "recommended_cuda_arch": cuda_arch,
        "fp16_tensor_instruction_path": path,
        "wgmma_supported": wgmma_supported,
        "benchmark_uses_wgmma": benchmark_uses_wgmma,
        "architecture_measurement_note": note,
    }


def estimate_threads_per_sm(run: Dict[str, Any]) -> float:
    threads = finite_float(run.get("threads"))
    blocks = finite_float(run.get("blocks"))
    sm_count = finite_float(run.get("sm_count"))
    if math.isfinite(threads) and math.isfinite(blocks) and math.isfinite(sm_count) and sm_count > 0:
        return threads * blocks / sm_count
    return math.nan


def sort_key(run: Dict[str, Any]) -> Tuple[int, str]:
    return (int(run.get("runner_wall_start_unix_ns", run.get("host_start_unix_ns", 0))), run.get("run_id", ""))


def has_reliable_energy(run: Dict[str, Any]) -> bool:
    energy = finite_float(run.get("power_energy_j"))
    if not math.isfinite(energy) or energy <= 0.0:
        return False
    if run.get("energy_source") == "nvml_total_energy_counter":
        return True
    return int(run.get("power_sample_count", 0)) >= 3


def matmul_bit_estimates(run: Dict[str, Any], ops: float) -> Dict[str, float]:
    """Return logical Tensor Core matmul bit counts, not memory traffic bits."""
    kernel = str(run.get("kernel", ""))
    if kernel not in {"tensor_mma_f16acc", "tensor_mma_f32acc"} or ops <= 0:
        return {
            "matmul_input_bits": 0.0,
            "matmul_accumulator_read_bits": 0.0,
            "matmul_output_bits": 0.0,
            "matmul_arithmetic_read_bits": 0.0,
            "matmul_register_read_write_bits": 0.0,
        }

    mma_m = finite_float(run.get("mma_m"), 16.0)
    mma_n = finite_float(run.get("mma_n"), 16.0)
    mma_k = finite_float(run.get("mma_k"), 16.0)
    flops_per_logical_mma = finite_float(run.get("mma_flops_per_logical_mma"), 2.0 * mma_m * mma_n * mma_k)
    if not math.isfinite(flops_per_logical_mma) or flops_per_logical_mma <= 0:
        flops_per_logical_mma = 2.0 * 16.0 * 16.0 * 16.0

    mma_count = ops / flops_per_logical_mma
    input_bits = mma_count * (mma_m * mma_k + mma_k * mma_n) * 16
    acc_bits = 16 if kernel == "tensor_mma_f16acc" else 32
    accumulator_read_bits = mma_count * mma_m * mma_n * acc_bits
    output_bits = accumulator_read_bits
    arithmetic_read_bits = input_bits + accumulator_read_bits
    register_read_write_bits = arithmetic_read_bits + output_bits
    return {
        "matmul_input_bits": input_bits,
        "matmul_accumulator_read_bits": accumulator_read_bits,
        "matmul_output_bits": output_bits,
        "matmul_arithmetic_read_bits": arithmetic_read_bits,
        "matmul_register_read_write_bits": register_read_write_bits,
    }


def group_pairs(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cond: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    singles: List[Dict[str, Any]] = []
    for r in rows:
        if r["role"] == "single":
            singles.append(r)
        else:
            by_cond[r["condition"]][r["role"]].append(r)

    summaries: List[Dict[str, Any]] = []
    for cond, roles in by_cond.items():
        if "test" not in roles or "baseline" not in roles:
            continue
        tests = sorted(roles["test"], key=sort_key)
        baselines = sorted(roles["baseline"], key=sort_key)
        pair_count = min(len(tests), len(baselines))
        for pair_index in range(pair_count):
            t = tests[pair_index]
            b = baselines[pair_index]
            summaries.append(summarize_pair(cond, pair_index, t, b))
    for pair_index, s in enumerate(sorted(singles, key=sort_key)):
        ops = finite_float(s.get("fp16_ops_estimate", 0.0), 0.0)
        mem_bytes = finite_float(s.get("memory_bytes_estimate", 0.0), 0.0)
        mem_bits = mem_bytes * 8.0
        matmul_bits = matmul_bit_estimates(s, ops)
        elapsed_s = finite_float(s.get("cuda_elapsed_ms", 0.0), 0.0) / 1000.0
        total_energy = finite_float(s.get("power_energy_j"))
        total_pj_per_bit = total_energy / mem_bits * 1e12 if mem_bits > 0 and math.isfinite(total_energy) else math.nan
        memory_gbps = mem_bytes / elapsed_s / 1e9 if mem_bytes > 0 and elapsed_s > 0 else math.nan
        summaries.append(
            {
                "condition": s["condition"],
                "pair_index": pair_index,
                "repeat_index": s.get("repeat_index", pair_index),
                "gpu": s.get("device_name", ""),
                "architecture_generation": s.get("architecture_generation", ""),
                "architecture_chip": s.get("architecture_chip", ""),
                "gpu_product_class": s.get("gpu_product_class", ""),
                "recommended_cuda_arch": s.get("recommended_cuda_arch", ""),
                "fp16_tensor_instruction_path": s.get("fp16_tensor_instruction_path", ""),
                "wgmma_supported": bool(s.get("wgmma_supported", False)),
                "benchmark_uses_wgmma": bool(s.get("benchmark_uses_wgmma", False)),
                "fp16_path": s.get("fp16_path", ""),
                "test_kernel": s.get("kernel", ""),
                "baseline_kernel": "",
                "blocks": s.get("blocks", ""),
                "threads": s.get("threads", ""),
                "threads_per_sm": estimate_threads_per_sm(s),
                "iters": s.get("iters", ""),
                "unroll": s.get("unroll", ""),
                "elapsed_s": elapsed_s,
                "baseline_elapsed_s": math.nan,
                "fp16_ops": ops,
                "memory_bytes": mem_bytes,
                "memory_bits": mem_bits,
                **matmul_bits,
                "suppress_output_store": bool(s.get("suppress_output_store", False)),
                "expected_l2_touch": bool(mem_bytes > 0 or not s.get("suppress_output_store", False)),
                "tflops": finite_float(s.get("estimated_tflops", math.nan)),
                "memory_gbps": memory_gbps,
                "test_avg_power_w": s.get("avg_power_w", math.nan),
                "baseline_avg_power_w": math.nan,
                "test_energy_j": total_energy,
                "baseline_energy_j": math.nan,
                "test_energy_source": s.get("energy_source", ""),
                "baseline_energy_source": "",
                "test_power_trace_energy_j": s.get("power_trace_energy_j", math.nan),
                "baseline_power_trace_energy_j": math.nan,
                "test_power_trace_avg_power_w": s.get("power_trace_avg_power_w", math.nan),
                "baseline_power_trace_avg_power_w": math.nan,
                "test_power_trace_query_modes": s.get("power_trace_query_modes", ""),
                "baseline_power_trace_query_modes": "",
                "test_avg_power_draw_average_w": s.get("avg_power_draw_average_w", math.nan),
                "baseline_avg_power_draw_average_w": math.nan,
                "test_avg_power_draw_instant_w": s.get("avg_power_draw_instant_w", math.nan),
                "baseline_avg_power_draw_instant_w": math.nan,
                "test_avg_power_limit_w": s.get("avg_power_limit_w", math.nan),
                "baseline_avg_power_limit_w": math.nan,
                "test_nvml_energy_delta_j": s.get("nvml_energy_delta_j", math.nan),
                "baseline_nvml_energy_delta_j": math.nan,
                "test_energy_counter_vs_trace_delta_j": s.get("energy_counter_vs_trace_delta_j", math.nan),
                "baseline_energy_counter_vs_trace_delta_j": math.nan,
                "test_energy_counter_vs_trace_ratio": s.get("energy_counter_vs_trace_ratio", math.nan),
                "baseline_energy_counter_vs_trace_ratio": math.nan,
                "baseline_scaled_energy_j": math.nan,
                "baseline_energy_fraction": math.nan,
                "incremental_energy_fraction": math.nan,
                "baseline_power_fraction": math.nan,
                "energy_sources_match": False,
                "incremental_power_w": math.nan,
                "incremental_energy_j": math.nan,
                "pj_per_flop": math.nan,
                "w_per_tflops": math.nan,
                "incremental_pj_per_bit": math.nan,
                "total_pj_per_bit": total_pj_per_bit,
                "matmul_input_pj_per_bit": math.nan,
                "matmul_arithmetic_read_pj_per_bit": math.nan,
                "matmul_register_read_write_pj_per_bit": math.nan,
                "test_power_samples": s.get("power_sample_count", 0),
                "baseline_power_samples": 0,
                "test_sm_util_samples": s.get("sm_util_sample_count", 0),
                "baseline_sm_util_samples": 0,
                "avg_sm_clock_mhz": s.get("avg_sm_clock_mhz", math.nan),
                "clock_span_mhz": finite_float(s.get("max_sm_clock_mhz")) - finite_float(s.get("min_sm_clock_mhz")),
                "max_temp_c": s.get("max_temp_c", math.nan),
                "avg_gpu_util_pct": s.get("avg_gpu_util_pct", math.nan),
                "max_gpu_util_pct": s.get("max_gpu_util_pct", math.nan),
                "avg_sm_util_pct": s.get("avg_sm_util_pct", math.nan),
                "max_sm_util_pct": s.get("max_sm_util_pct", math.nan),
                "valid_basic": False,
                "valid_no_l2": False,
                "pure_fp16_candidate": False,
                "separation_quality": "single_run_no_baseline",
                "test_run_id": s.get("run_id", ""),
                "baseline_run_id": "",
                "test_power_csv": s.get("power_csv", ""),
                "baseline_power_csv": "",
                "test_sm_util_csv": s.get("sm_util_csv", ""),
                "baseline_sm_util_csv": "",
            }
        )
    return summaries


def summarize_pair(cond: str, pair_index: int, t: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ops = finite_float(t.get("fp16_ops_estimate", 0.0), 0.0)
    mem_bytes = finite_float(t.get("memory_bytes_estimate", 0.0), 0.0)
    mem_bits = mem_bytes * 8.0
    matmul_bits = matmul_bit_estimates(t, ops)
    elapsed_s = finite_float(t.get("cuda_elapsed_ms", 0.0), 0.0) / 1000.0
    baseline_elapsed_s = finite_float(b.get("cuda_elapsed_ms", 0.0), 0.0) / 1000.0
    test_energy = finite_float(t.get("power_energy_j"))
    baseline_energy = finite_float(b.get("power_energy_j"))
    base_avg_power = finite_float(b.get("avg_power_w"))
    test_avg_power = finite_float(t.get("avg_power_w"))
    inc_power = test_avg_power - base_avg_power
    baseline_scaled_energy = base_avg_power * elapsed_s if math.isfinite(base_avg_power) else math.nan
    inc_energy = (
        test_energy - baseline_scaled_energy
        if math.isfinite(test_energy) and math.isfinite(baseline_scaled_energy)
        else math.nan
    )
    baseline_energy_fraction = (
        baseline_scaled_energy / test_energy
        if math.isfinite(baseline_scaled_energy) and math.isfinite(test_energy) and test_energy > 0.0
        else math.nan
    )
    incremental_energy_fraction = (
        inc_energy / test_energy
        if math.isfinite(inc_energy) and math.isfinite(test_energy) and test_energy > 0.0
        else math.nan
    )
    baseline_power_fraction = (
        base_avg_power / test_avg_power
        if math.isfinite(base_avg_power) and math.isfinite(test_avg_power) and test_avg_power > 0.0
        else math.nan
    )
    tflops = ops / elapsed_s / 1e12 if ops > 0 and elapsed_s > 0 else math.nan
    gbps = mem_bytes / elapsed_s / 1e9 if mem_bytes > 0 and elapsed_s > 0 else math.nan
    pj_per_flop = inc_energy / ops * 1e12 if ops > 0 and math.isfinite(inc_energy) else math.nan
    w_per_tflops = inc_power / tflops if math.isfinite(inc_power) and math.isfinite(tflops) and tflops > 0 else math.nan
    incremental_pj_per_bit = inc_energy / mem_bits * 1e12 if mem_bits > 0 and math.isfinite(inc_energy) else math.nan
    total_pj_per_bit = test_energy / mem_bits * 1e12 if mem_bits > 0 and math.isfinite(test_energy) else math.nan
    matmul_input_pj_per_bit = (
        inc_energy / matmul_bits["matmul_input_bits"] * 1e12
        if matmul_bits["matmul_input_bits"] > 0 and math.isfinite(inc_energy)
        else math.nan
    )
    matmul_arithmetic_read_pj_per_bit = (
        inc_energy / matmul_bits["matmul_arithmetic_read_bits"] * 1e12
        if matmul_bits["matmul_arithmetic_read_bits"] > 0 and math.isfinite(inc_energy)
        else math.nan
    )
    matmul_register_read_write_pj_per_bit = (
        inc_energy / matmul_bits["matmul_register_read_write_bits"] * 1e12
        if matmul_bits["matmul_register_read_write_bits"] > 0 and math.isfinite(inc_energy)
        else math.nan
    )
    has_work = ops > 0 or mem_bytes > 0
    reliable_energy = has_reliable_energy(t) and has_reliable_energy(b)
    valid_basic = bool(
        has_work
        and reliable_energy
        and math.isfinite(inc_power)
        and inc_power > 0
        and math.isfinite(inc_energy)
        and inc_energy > 0
    )
    expected_l2_touch = bool(mem_bytes > 0 or not t.get("suppress_output_store", False))
    valid_no_l2 = bool(valid_basic and not expected_l2_touch)
    pure_fp16_candidate = bool(
        valid_no_l2
        and str(t.get("kernel", "")) in {"tensor_mma_f16acc", "tensor_mma_f32acc", "fp16_half2"}
    )
    if pure_fp16_candidate:
        separation_quality = "pure_fp16_candidate_no_l2"
    elif valid_basic and expected_l2_touch:
        separation_quality = "valid_but_expected_l2_touch"
    elif valid_basic:
        separation_quality = "valid_non_fp16_or_memory"
    else:
        separation_quality = "invalid_or_nonpositive_increment"
    return {
        "condition": cond,
        "pair_index": pair_index,
        "repeat_index": t.get("repeat_index", pair_index),
        "gpu": t.get("device_name", ""),
        "architecture_generation": t.get("architecture_generation", ""),
        "architecture_chip": t.get("architecture_chip", ""),
        "gpu_product_class": t.get("gpu_product_class", ""),
        "recommended_cuda_arch": t.get("recommended_cuda_arch", ""),
        "fp16_tensor_instruction_path": t.get("fp16_tensor_instruction_path", ""),
        "wgmma_supported": bool(t.get("wgmma_supported", False)),
        "benchmark_uses_wgmma": bool(t.get("benchmark_uses_wgmma", False)),
        "fp16_path": t.get("fp16_path", ""),
        "test_kernel": t.get("kernel", ""),
        "baseline_kernel": b.get("kernel", ""),
        "blocks": t.get("blocks", ""),
        "threads": t.get("threads", ""),
        "threads_per_sm": estimate_threads_per_sm(t),
        "iters": t.get("iters", ""),
        "unroll": t.get("unroll", ""),
        "elapsed_s": elapsed_s,
        "baseline_elapsed_s": baseline_elapsed_s,
        "fp16_ops": ops,
        "memory_bytes": mem_bytes,
        "memory_bits": mem_bits,
        **matmul_bits,
        "suppress_output_store": bool(t.get("suppress_output_store", False)),
        "expected_l2_touch": expected_l2_touch,
        "tflops": tflops,
        "memory_gbps": gbps,
        "test_avg_power_w": test_avg_power,
        "baseline_avg_power_w": base_avg_power,
        "test_energy_j": test_energy,
        "baseline_energy_j": baseline_energy,
        "test_energy_source": t.get("energy_source", ""),
        "baseline_energy_source": b.get("energy_source", ""),
        "test_power_trace_energy_j": t.get("power_trace_energy_j", math.nan),
        "baseline_power_trace_energy_j": b.get("power_trace_energy_j", math.nan),
        "test_power_trace_avg_power_w": t.get("power_trace_avg_power_w", math.nan),
        "baseline_power_trace_avg_power_w": b.get("power_trace_avg_power_w", math.nan),
        "test_power_trace_query_modes": t.get("power_trace_query_modes", ""),
        "baseline_power_trace_query_modes": b.get("power_trace_query_modes", ""),
        "test_avg_power_draw_average_w": t.get("avg_power_draw_average_w", math.nan),
        "baseline_avg_power_draw_average_w": b.get("avg_power_draw_average_w", math.nan),
        "test_avg_power_draw_instant_w": t.get("avg_power_draw_instant_w", math.nan),
        "baseline_avg_power_draw_instant_w": b.get("avg_power_draw_instant_w", math.nan),
        "test_avg_power_limit_w": t.get("avg_power_limit_w", math.nan),
        "baseline_avg_power_limit_w": b.get("avg_power_limit_w", math.nan),
        "test_nvml_energy_delta_j": t.get("nvml_energy_delta_j", math.nan),
        "baseline_nvml_energy_delta_j": b.get("nvml_energy_delta_j", math.nan),
        "test_energy_counter_vs_trace_delta_j": t.get("energy_counter_vs_trace_delta_j", math.nan),
        "baseline_energy_counter_vs_trace_delta_j": b.get("energy_counter_vs_trace_delta_j", math.nan),
        "test_energy_counter_vs_trace_ratio": t.get("energy_counter_vs_trace_ratio", math.nan),
        "baseline_energy_counter_vs_trace_ratio": b.get("energy_counter_vs_trace_ratio", math.nan),
        "baseline_scaled_energy_j": baseline_scaled_energy,
        "baseline_energy_fraction": baseline_energy_fraction,
        "incremental_energy_fraction": incremental_energy_fraction,
        "baseline_power_fraction": baseline_power_fraction,
        "energy_sources_match": t.get("energy_source", "") == b.get("energy_source", ""),
        "incremental_power_w": inc_power,
        "incremental_energy_j": inc_energy,
        "pj_per_flop": pj_per_flop,
        "w_per_tflops": w_per_tflops,
        "incremental_pj_per_bit": incremental_pj_per_bit,
        "total_pj_per_bit": total_pj_per_bit,
        "matmul_input_pj_per_bit": matmul_input_pj_per_bit,
        "matmul_arithmetic_read_pj_per_bit": matmul_arithmetic_read_pj_per_bit,
        "matmul_register_read_write_pj_per_bit": matmul_register_read_write_pj_per_bit,
        "test_power_samples": t.get("power_sample_count", 0),
        "baseline_power_samples": b.get("power_sample_count", 0),
        "test_sm_util_samples": t.get("sm_util_sample_count", 0),
        "baseline_sm_util_samples": b.get("sm_util_sample_count", 0),
        "avg_sm_clock_mhz": t.get("avg_sm_clock_mhz", math.nan),
        "clock_span_mhz": finite_float(t.get("max_sm_clock_mhz")) - finite_float(t.get("min_sm_clock_mhz")),
        "max_temp_c": t.get("max_temp_c", math.nan),
        "avg_gpu_util_pct": t.get("avg_gpu_util_pct", math.nan),
        "max_gpu_util_pct": t.get("max_gpu_util_pct", math.nan),
        "avg_sm_util_pct": t.get("avg_sm_util_pct", math.nan),
        "max_sm_util_pct": t.get("max_sm_util_pct", math.nan),
        "valid_basic": valid_basic,
        "valid_no_l2": valid_no_l2,
        "pure_fp16_candidate": pure_fp16_candidate,
        "separation_quality": separation_quality,
        "test_run_id": t.get("run_id", ""),
        "baseline_run_id": b.get("run_id", ""),
        "test_power_csv": t.get("power_csv", ""),
        "baseline_power_csv": b.get("power_csv", ""),
        "test_sm_util_csv": t.get("sm_util_csv", ""),
        "baseline_sm_util_csv": b.get("sm_util_csv", ""),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    seen = set(keys)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_stats(values: List[float]) -> Dict[str, Any]:
    clean = [v for v in values if math.isfinite(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "ci95": math.nan}
    mean = sum(clean) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in clean) / (n - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(n)
    else:
        std = 0.0
        ci95 = 0.0
    return {"n": n, "mean": mean, "std": std, "min": min(clean), "max": max(clean), "ci95": ci95}


def aggregate_conditions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)

    metrics = [
        "pj_per_flop",
        "incremental_pj_per_bit",
        "total_pj_per_bit",
        "matmul_input_pj_per_bit",
        "matmul_arithmetic_read_pj_per_bit",
        "matmul_register_read_write_pj_per_bit",
        "tflops",
        "memory_gbps",
        "incremental_power_w",
        "incremental_energy_j",
        "baseline_energy_fraction",
        "incremental_energy_fraction",
        "baseline_power_fraction",
        "w_per_tflops",
        "avg_sm_clock_mhz",
        "clock_span_mhz",
        "max_temp_c",
        "avg_gpu_util_pct",
        "max_gpu_util_pct",
        "avg_sm_util_pct",
        "max_sm_util_pct",
    ]
    out: List[Dict[str, Any]] = []
    for cond, group in by_cond.items():
        valid = [r for r in group if bool(r.get("valid_basic", False))]
        stats_source = valid if valid else group
        first = group[0]
        row: Dict[str, Any] = {
            "condition": cond,
            "gpu": first.get("gpu", ""),
            "architecture_generation": first.get("architecture_generation", ""),
            "architecture_chip": first.get("architecture_chip", ""),
            "gpu_product_class": first.get("gpu_product_class", ""),
            "recommended_cuda_arch": first.get("recommended_cuda_arch", ""),
            "fp16_path": first.get("fp16_path", ""),
            "test_kernel": first.get("test_kernel", ""),
            "baseline_kernel": first.get("baseline_kernel", ""),
            "run_count": len(group),
            "valid_count": len(valid),
            "valid_no_l2_count": sum(1 for r in group if bool(r.get("valid_no_l2", False))),
            "pure_fp16_candidate_count": sum(1 for r in group if bool(r.get("pure_fp16_candidate", False))),
            "invalid_count": len(group) - len(valid),
            "stats_scope": "valid_basic" if valid else "all_runs_no_valid_basic",
        }
        for metric in metrics:
            stats = metric_stats([finite_float(r.get(metric)) for r in stats_source])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        out.append(row)
    return out


def grouped_metric_stats(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    out: List[Dict[str, Any]] = []
    for cond, group in by_cond.items():
        valid = [r for r in group if bool(r.get("valid_basic", False))]
        stats_source = valid if valid else group
        stats = metric_stats([finite_float(r.get(metric)) for r in stats_source])
        if stats["n"] > 0:
            out.append({"condition": cond, **stats})
    return out


def aggregate_thread_sweep(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    thread_values = {int(r["threads"]) for r in rows if str(r.get("threads", "")).isdigit()}
    if len(thread_values) < 2:
        return []

    by_variant: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            threads = int(row["threads"])
        except Exception:
            continue
        key = (
            str(row.get("fp16_path", "")),
            str(row.get("test_kernel", "")),
            str(row.get("baseline_kernel", "")),
            threads,
        )
        by_variant[key].append(row)

    metrics = [
        "avg_gpu_util_pct",
        "max_gpu_util_pct",
        "avg_sm_util_pct",
        "max_sm_util_pct",
        "tflops",
        "matmul_input_pj_per_bit",
        "pj_per_flop",
        "incremental_power_w",
        "avg_sm_clock_mhz",
        "clock_span_mhz",
        "max_temp_c",
    ]
    out: List[Dict[str, Any]] = []
    for (fp16_path, test_kernel, baseline_kernel, threads), group in by_variant.items():
        valid_no_l2 = [
            r for r in group
            if bool(r.get("valid_basic", False)) and not bool(r.get("expected_l2_touch", True))
        ]
        valid = [r for r in group if bool(r.get("valid_basic", False))]
        stats_source = valid_no_l2 or valid or group
        row: Dict[str, Any] = {
            "gpu": group[0].get("gpu", ""),
            "architecture_generation": group[0].get("architecture_generation", ""),
            "architecture_chip": group[0].get("architecture_chip", ""),
            "gpu_product_class": group[0].get("gpu_product_class", ""),
            "recommended_cuda_arch": group[0].get("recommended_cuda_arch", ""),
            "fp16_path": fp16_path,
            "test_kernel": test_kernel,
            "baseline_kernel": baseline_kernel,
            "threads": threads,
            "threads_per_sm": finite_float(group[0].get("threads_per_sm")),
            "run_count": len(group),
            "valid_count": len(valid),
            "valid_no_l2_count": len(valid_no_l2),
            "pure_fp16_candidate_count": sum(1 for r in group if bool(r.get("pure_fp16_candidate", False))),
            "stats_scope": "valid_no_l2" if valid_no_l2 else ("valid_basic" if valid else "all_runs_no_valid"),
            "expected_l2_touch": any(bool(r.get("expected_l2_touch", True)) for r in stats_source),
        }
        for metric in metrics:
            stats = metric_stats([finite_float(r.get(metric)) for r in stats_source])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        row["selected_optimal"] = False
        out.append(row)

    by_kernel: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in out:
        by_kernel[(row["fp16_path"], row["test_kernel"], row["baseline_kernel"])].append(row)

    for group in by_kernel.values():
        def has_enough_valid(row: Dict[str, Any], key: str) -> bool:
            run_count = int(row.get("run_count", 0))
            valid_count = int(row.get(key, 0))
            min_count = 1 if run_count <= 1 else max(3, math.ceil(run_count * 0.5))
            return valid_count >= min_count

        eligible = [r for r in group if has_enough_valid(r, "valid_no_l2_count")]
        if not eligible:
            eligible = [r for r in group if int(r.get("valid_no_l2_count", 0)) > 0]
        if not eligible:
            eligible = [r for r in group if has_enough_valid(r, "valid_count")]
        if not eligible:
            eligible = [r for r in group if int(r.get("valid_count", 0)) > 0]
        if not eligible:
            eligible = group

        def util_score(row: Dict[str, Any]) -> float:
            util = finite_float(row.get("avg_sm_util_pct_mean"), -math.inf)
            if not math.isfinite(util):
                util = finite_float(row.get("avg_gpu_util_pct_mean"), -math.inf)
            return util if math.isfinite(util) else -math.inf

        max_util = max(util_score(row) for row in eligible)
        saturated = [row for row in eligible if util_score(row) >= max_util - 0.1]
        target_pool = saturated if saturated else eligible

        def target_score(row: Dict[str, Any]) -> Tuple[float, float, float]:
            threads_per_sm = finite_float(row.get("threads_per_sm"), math.inf)
            if not math.isfinite(threads_per_sm):
                threads_per_sm = finite_float(row.get("threads"), math.inf)
            tflops = finite_float(row.get("tflops_mean"), -math.inf)
            if not math.isfinite(tflops):
                tflops = -math.inf
            clock_span = finite_float(row.get("clock_span_mhz_mean"), math.inf)
            if not math.isfinite(clock_span):
                clock_span = math.inf
            # Select the first utilization-saturation point; use throughput only as a tie-break.
            return (threads_per_sm, -tflops, clock_span)

        best = min(target_pool, key=target_score)
        best["selected_optimal"] = True

    return sorted(out, key=lambda r: (str(r["test_kernel"]), int(r["threads"])))


def plot_power_trace(summary: Dict[str, Any], figdir: Path) -> None:
    paths = []
    if summary.get("baseline_power_csv"):
        paths.append(("baseline", summary["baseline_power_csv"]))
    if summary.get("test_power_csv"):
        paths.append(("test", summary["test_power_csv"]))
    if not paths:
        return
    plt.figure(figsize=(9, 4.8))
    for label, p in paths:
        rows = read_power_csv(p)
        if not rows:
            continue
        t0 = rows[0]["sample_unix_ns"]
        xs = [(r["sample_unix_ns"] - t0) / 1e9 for r in rows if r["power_w"] is not None]
        ys = [r["power_w"] for r in rows if r["power_w"] is not None]
        if xs and ys:
            plt.plot(xs, ys, label=label)
    plt.xlabel("Time since logger start (s)")
    plt.ylabel("Power draw (W)")
    plt.title(f"Power trace: {summary['condition']}")
    plt.legend()
    plt.tight_layout()
    pair_index = int(summary.get("pair_index", 0))
    plt.savefig(figdir / f"power_trace_{summary['condition']}_pair{pair_index:03d}.png", dpi=160)
    plt.close()


def plot_bar(summary_rows: List[Dict[str, Any]], figdir: Path) -> None:
    rows = grouped_metric_stats(summary_rows, "pj_per_flop")
    if not rows:
        return
    plt.figure(figsize=(10, 5))
    labels = [r["condition"] for r in rows]
    vals = [float(r["mean"]) for r in rows]
    yerr = [float(r["std"]) for r in rows]
    plt.bar(labels, vals, yerr=yerr, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("pJ/FLOP")
    plt.title("Baseline-subtracted FP16 compute energy estimate")
    plt.tight_layout()
    plt.savefig(figdir / "pj_per_flop_bar.png", dpi=160)
    plt.close()


def plot_pj_per_bit_bar(summary_rows: List[Dict[str, Any]], figdir: Path) -> None:
    metric_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        value = finite_float(row.get("incremental_pj_per_bit"))
        if not math.isfinite(value):
            value = finite_float(row.get("total_pj_per_bit"))
        metric_rows.append({**row, "pj_per_bit_for_plot": value})
    rows = grouped_metric_stats(metric_rows, "pj_per_bit_for_plot")
    if not rows:
        return
    plt.figure(figsize=(10, 5))
    labels = [r["condition"] for r in rows]
    vals = [float(r["mean"]) for r in rows]
    yerr = [float(r["std"]) for r in rows]
    plt.bar(labels, vals, yerr=yerr, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("pJ/bit")
    plt.title("Memory energy estimate per transferred bit")
    plt.tight_layout()
    plt.savefig(figdir / "pj_per_bit_bar.png", dpi=160)
    plt.close()


def plot_matmul_pj_per_bit_bar(summary_rows: List[Dict[str, Any]], figdir: Path) -> None:
    rows = grouped_metric_stats(summary_rows, "matmul_input_pj_per_bit")
    if not rows:
        return
    plt.figure(figsize=(10, 5))
    labels = [r["condition"] for r in rows]
    vals = [float(r["mean"]) for r in rows]
    yerr = [float(r["std"]) for r in rows]
    plt.bar(labels, vals, yerr=yerr, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("pJ/logical input bit")
    plt.title("FP16 Tensor Core matmul energy estimate")
    plt.tight_layout()
    plt.savefig(figdir / "matmul_input_pj_per_bit_bar.png", dpi=160)
    plt.close()


def plot_energy_separation(summary_rows: List[Dict[str, Any]], figdir: Path) -> None:
    rows = [
        r for r in summary_rows
        if math.isfinite(finite_float(r.get("baseline_scaled_energy_j")))
        and math.isfinite(finite_float(r.get("incremental_energy_j")))
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (str(r.get("condition", "")), int(r.get("pair_index", 0))))[:40]
    labels = [f"{r['condition']}#{int(r.get('pair_index', 0))}" for r in rows]
    baseline = [finite_float(r.get("baseline_scaled_energy_j"), 0.0) for r in rows]
    incremental = [finite_float(r.get("incremental_energy_j"), 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(rows)), 5.2))
    xs = list(range(len(rows)))
    ax.bar(xs, baseline, label="baseline-scaled energy")
    ax.bar(xs, incremental, bottom=baseline, label="FP16 incremental energy")
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Energy in test interval (J)")
    ax.set_title("Baseline separation for FP16 incremental estimate")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figdir / "fp16_energy_separation_stack.png", dpi=160)
    plt.close(fig)


def plot_thread_sweep(thread_rows: List[Dict[str, Any]], figdir: Path) -> None:
    if not thread_rows:
        return
    by_kernel: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in thread_rows:
        by_kernel[(str(row["test_kernel"]), str(row["baseline_kernel"]))].append(row)

    for (test_kernel, baseline_kernel), rows in by_kernel.items():
        rows = sorted(rows, key=lambda r: int(r["threads"]))
        xs = []
        for r in rows:
            threads_per_sm = finite_float(r.get("threads_per_sm"))
            xs.append(threads_per_sm if math.isfinite(threads_per_sm) else float(r["threads"]))
        sm_util = [finite_float(r.get("avg_sm_util_pct_mean")) for r in rows]
        has_sm_util = any(math.isfinite(v) for v in sm_util)
        util = sm_util if has_sm_util else [finite_float(r.get("avg_gpu_util_pct_mean")) for r in rows]
        tflops = [finite_float(r.get("tflops_mean")) for r in rows]
        selected = [r for r in rows if bool(r.get("selected_optimal", False))]

        fig, ax1 = plt.subplots(figsize=(8, 4.8))
        util_label = "avg SM util" if has_sm_util else "avg GPU util"
        ylabel = "Avg SM utilization (%)" if has_sm_util else "Avg GPU utilization (%)"
        ax1.plot(xs, util, marker="o", label=util_label)
        ax1.set_xlabel("Launched threads per SM")
        ax1.set_ylabel(ylabel)
        ax1.set_xticks(xs)
        ax1.get_xaxis().set_major_formatter(ScalarFormatter())
        ax1.grid(True, axis="y", alpha=0.3)
        for x, y, row in zip(xs, util, rows):
            if math.isfinite(y):
                ax1.annotate(
                    str(row["threads"]),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )

        ax2 = ax1.twinx()
        ax2.plot(xs, tflops, marker="s", color="tab:orange", label="TFLOPS")
        ax2.set_ylabel("TFLOPS")

        if selected:
            sx = finite_float(selected[0].get("threads_per_sm"))
            if not math.isfinite(sx):
                sx = float(selected[0]["threads"])
            ax1.axvline(sx, color="tab:green", linestyle="--", linewidth=1.2, label="selected")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        plt.title(f"Thread sweep: {test_kernel} vs {baseline_kernel} (labels: threads/block)")
        plt.tight_layout()
        safe_name = f"thread_sweep_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
        plt.savefig(figdir / safe_name, dpi=160)
        plt.close(fig)

        pjbit = [finite_float(r.get("matmul_input_pj_per_bit_mean")) for r in rows]
        if any(math.isfinite(v) for v in pjbit):
            pjbit_ci = []
            for r in rows:
                ci = finite_float(r.get("matmul_input_pj_per_bit_ci95"), 0.0)
                pjbit_ci.append(ci if math.isfinite(ci) and ci >= 0.0 else 0.0)

            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.errorbar(xs, pjbit, yerr=pjbit_ci, marker="D", capsize=3, color="tab:purple", label="pJ/bit")
            ax.axhline(0.0, color="0.3", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("pJ/logical input bit")
            ax.set_xticks(xs)
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.3)
            for x, y, row in zip(xs, pjbit, rows):
                if math.isfinite(y):
                    ax.annotate(
                        f"{row['threads']}\n{y:.3g}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 7 if y >= 0 else -18),
                        ha="center",
                        fontsize=8,
                    )
            if selected:
                sx = finite_float(selected[0].get("threads_per_sm"))
                if not math.isfinite(sx):
                    sx = float(selected[0]["threads"])
                ax.axvline(sx, color="tab:green", linestyle="--", linewidth=1.2, label="selected")
            ax.legend(loc="best")
            plt.title(f"Thread sweep pJ/bit: {test_kernel} vs {baseline_kernel}")
            plt.tight_layout()
            safe_name = f"thread_sweep_pjbit_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
            plt.savefig(figdir / safe_name, dpi=160)
            plt.close(fig)


def plot_scatter(summary_rows: List[Dict[str, Any]], figdir: Path) -> None:
    rows = [r for r in summary_rows if r.get("pj_per_flop") == r.get("pj_per_flop") and r.get("tflops") == r.get("tflops")]
    if not rows:
        return
    plt.figure(figsize=(7, 5))
    xs = [float(r["tflops"]) for r in rows]
    ys = [float(r["pj_per_flop"]) for r in rows]
    plt.scatter(xs, ys)
    for r, x, y in zip(rows, xs, ys):
        plt.annotate(r["condition"], (x, y), fontsize=8)
    plt.xlabel("Achieved TFLOPS")
    plt.ylabel("pJ/FLOP")
    plt.title("Throughput vs energy estimate")
    plt.tight_layout()
    plt.savefig(figdir / "tflops_vs_pj_per_flop.png", dpi=160)
    plt.close()


def plot_clock_temp(rows: List[Dict[str, Any]], figdir: Path) -> None:
    # Basic per-run clock/temperature timeline from power CSV files.
    for r in rows:
        p = r.get("power_csv")
        if not p:
            continue
        samples = read_power_csv(p)
        if not samples:
            continue
        t0 = samples[0]["sample_unix_ns"]
        xs_clock = [(s["sample_unix_ns"] - t0) / 1e9 for s in samples if s["sm_clock_mhz"] is not None]
        ys_clock = [s["sm_clock_mhz"] for s in samples if s["sm_clock_mhz"] is not None]
        xs_temp = [(s["sample_unix_ns"] - t0) / 1e9 for s in samples if s["temp_c"] is not None]
        ys_temp = [s["temp_c"] for s in samples if s["temp_c"] is not None]
        if xs_clock:
            plt.figure(figsize=(8, 4.5))
            plt.plot(xs_clock, ys_clock)
            plt.xlabel("Time since logger start (s)")
            plt.ylabel("SM clock (MHz)")
            plt.title(f"SM clock: {r['condition']} / {r['role']}")
            plt.tight_layout()
            plt.savefig(figdir / f"clock_{r['run_id']}.png", dpi=140)
            plt.close()
        if xs_temp:
            plt.figure(figsize=(8, 4.5))
            plt.plot(xs_temp, ys_temp)
            plt.xlabel("Time since logger start (s)")
            plt.ylabel("Temperature (C)")
            plt.title(f"Temperature: {r['condition']} / {r['role']}")
            plt.tight_layout()
            plt.savefig(figdir / f"temperature_{r['run_id']}.png", dpi=140)
            plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze FP16 energy experiment output")
    parser.add_argument("--input", type=Path, required=True, help="Run directory containing runs.jsonl")
    args = parser.parse_args()

    raw_runs = read_runs(args.input)
    enriched = [summarize_run(r) for r in raw_runs]
    summary = group_pairs(enriched)
    condition_summary = aggregate_conditions(summary)
    thread_sweep_summary = aggregate_thread_sweep(summary)

    write_csv(args.input / "run_level_summary.csv", enriched)
    write_csv(args.input / "summary.csv", summary)
    write_csv(args.input / "condition_summary.csv", condition_summary)
    if thread_sweep_summary:
        write_csv(args.input / "thread_sweep_summary.csv", thread_sweep_summary)

    figdir = args.input / "figures"
    figdir.mkdir(exist_ok=True)
    plot_bar(summary, figdir)
    plot_pj_per_bit_bar(summary, figdir)
    plot_matmul_pj_per_bit_bar(summary, figdir)
    plot_energy_separation(summary, figdir)
    plot_thread_sweep(thread_sweep_summary, figdir)
    plot_scatter(summary, figdir)
    for s in summary:
        plot_power_trace(s, figdir)
    plot_clock_temp(enriched, figdir)

    print(f"Wrote: {args.input / 'summary.csv'}")
    print(f"Wrote: {args.input / 'condition_summary.csv'}")
    if thread_sweep_summary:
        print(f"Wrote: {args.input / 'thread_sweep_summary.csv'}")
    print(f"Wrote figures under: {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
