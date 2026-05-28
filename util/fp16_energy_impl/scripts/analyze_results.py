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
                    "sm_clock_mhz": parse_float(r.get("sm_clock_mhz")),
                    "mem_clock_mhz": parse_float(r.get("mem_clock_mhz")),
                    "temp_c": parse_float(r.get("temp_c")),
                    "pstate": r.get("pstate", ""),
                    "util_gpu_pct": parse_float(r.get("util_gpu_pct")),
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
    samples = read_power_csv(run.get("power_csv", ""))
    start_ns = int(run["host_start_unix_ns"])
    end_ns = int(run["host_end_unix_ns"])
    energy_j, avg_power_w, sample_count = integrate_power(samples, start_ns, end_ns)
    clocks = [s["sm_clock_mhz"] for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["sm_clock_mhz"] is not None]
    temps = [s["temp_c"] for s in samples if start_ns <= s["sample_unix_ns"] <= end_ns and s["temp_c"] is not None]
    return {
        **run,
        "power_energy_j": energy_j,
        "avg_power_w": avg_power_w,
        "power_sample_count": sample_count,
        "avg_sm_clock_mhz": sum(clocks) / len(clocks) if clocks else math.nan,
        "min_sm_clock_mhz": min(clocks) if clocks else math.nan,
        "max_sm_clock_mhz": max(clocks) if clocks else math.nan,
        "avg_temp_c": sum(temps) / len(temps) if temps else math.nan,
        "max_temp_c": max(temps) if temps else math.nan,
    }


def finite_float(x: Any, default: float = math.nan) -> float:
    value = parse_float(x)
    return value if value is not None else default


def sort_key(run: Dict[str, Any]) -> Tuple[int, str]:
    return (int(run.get("runner_wall_start_unix_ns", run.get("host_start_unix_ns", 0))), run.get("run_id", ""))


def matmul_bit_estimates(run: Dict[str, Any], ops: float) -> Dict[str, float]:
    """Return logical m16n8k16 matmul bit counts, not memory traffic bits."""
    kernel = str(run.get("kernel", ""))
    if kernel not in {"tensor_mma_f16acc", "tensor_mma_f32acc"} or ops <= 0:
        return {
            "matmul_input_bits": 0.0,
            "matmul_accumulator_read_bits": 0.0,
            "matmul_output_bits": 0.0,
            "matmul_arithmetic_read_bits": 0.0,
            "matmul_register_read_write_bits": 0.0,
        }

    mma_count = ops / 4096.0
    input_bits = mma_count * (16 * 16 + 16 * 8) * 16
    acc_bits = 16 if kernel == "tensor_mma_f16acc" else 32
    accumulator_read_bits = mma_count * 16 * 8 * acc_bits
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
                "fp16_path": s.get("fp16_path", ""),
                "test_kernel": s.get("kernel", ""),
                "baseline_kernel": "",
                "blocks": s.get("blocks", ""),
                "threads": s.get("threads", ""),
                "iters": s.get("iters", ""),
                "unroll": s.get("unroll", ""),
                "elapsed_s": elapsed_s,
                "baseline_elapsed_s": math.nan,
                "fp16_ops": ops,
                "memory_bytes": mem_bytes,
                "memory_bits": mem_bits,
                **matmul_bits,
                "tflops": finite_float(s.get("estimated_tflops", math.nan)),
                "memory_gbps": memory_gbps,
                "test_avg_power_w": s.get("avg_power_w", math.nan),
                "baseline_avg_power_w": math.nan,
                "test_energy_j": total_energy,
                "baseline_energy_j": math.nan,
                "baseline_scaled_energy_j": math.nan,
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
                "avg_sm_clock_mhz": s.get("avg_sm_clock_mhz", math.nan),
                "clock_span_mhz": finite_float(s.get("max_sm_clock_mhz")) - finite_float(s.get("min_sm_clock_mhz")),
                "max_temp_c": s.get("max_temp_c", math.nan),
                "valid_basic": False,
                "test_run_id": s.get("run_id", ""),
                "baseline_run_id": "",
                "test_power_csv": s.get("power_csv", ""),
                "baseline_power_csv": "",
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
    enough_samples = int(t.get("power_sample_count", 0)) >= 3 and int(b.get("power_sample_count", 0)) >= 3
    valid_basic = bool(has_work and enough_samples and math.isfinite(inc_power) and inc_power > 0)
    return {
        "condition": cond,
        "pair_index": pair_index,
        "repeat_index": t.get("repeat_index", pair_index),
        "gpu": t.get("device_name", ""),
        "fp16_path": t.get("fp16_path", ""),
        "test_kernel": t.get("kernel", ""),
        "baseline_kernel": b.get("kernel", ""),
        "blocks": t.get("blocks", ""),
        "threads": t.get("threads", ""),
        "iters": t.get("iters", ""),
        "unroll": t.get("unroll", ""),
        "elapsed_s": elapsed_s,
        "baseline_elapsed_s": baseline_elapsed_s,
        "fp16_ops": ops,
        "memory_bytes": mem_bytes,
        "memory_bits": mem_bits,
        **matmul_bits,
        "tflops": tflops,
        "memory_gbps": gbps,
        "test_avg_power_w": test_avg_power,
        "baseline_avg_power_w": base_avg_power,
        "test_energy_j": test_energy,
        "baseline_energy_j": baseline_energy,
        "baseline_scaled_energy_j": baseline_scaled_energy,
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
        "avg_sm_clock_mhz": t.get("avg_sm_clock_mhz", math.nan),
        "clock_span_mhz": finite_float(t.get("max_sm_clock_mhz")) - finite_float(t.get("min_sm_clock_mhz")),
        "max_temp_c": t.get("max_temp_c", math.nan),
        "valid_basic": valid_basic,
        "test_run_id": t.get("run_id", ""),
        "baseline_run_id": b.get("run_id", ""),
        "test_power_csv": t.get("power_csv", ""),
        "baseline_power_csv": b.get("power_csv", ""),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
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
        "w_per_tflops",
        "avg_sm_clock_mhz",
        "clock_span_mhz",
        "max_temp_c",
    ]
    out: List[Dict[str, Any]] = []
    for cond, group in by_cond.items():
        valid = [r for r in group if bool(r.get("valid_basic", False))]
        stats_source = valid if valid else group
        first = group[0]
        row: Dict[str, Any] = {
            "condition": cond,
            "gpu": first.get("gpu", ""),
            "fp16_path": first.get("fp16_path", ""),
            "test_kernel": first.get("test_kernel", ""),
            "baseline_kernel": first.get("baseline_kernel", ""),
            "run_count": len(group),
            "valid_count": len(valid),
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

    write_csv(args.input / "run_level_summary.csv", enriched)
    write_csv(args.input / "summary.csv", summary)
    write_csv(args.input / "condition_summary.csv", condition_summary)

    figdir = args.input / "figures"
    figdir.mkdir(exist_ok=True)
    plot_bar(summary, figdir)
    plot_pj_per_bit_bar(summary, figdir)
    plot_matmul_pj_per_bit_bar(summary, figdir)
    plot_scatter(summary, figdir)
    for s in summary:
        plot_power_trace(s, figdir)
    plot_clock_temp(enriched, figdir)

    print(f"Wrote: {args.input / 'summary.csv'}")
    print(f"Wrote: {args.input / 'condition_summary.csv'}")
    print(f"Wrote figures under: {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
