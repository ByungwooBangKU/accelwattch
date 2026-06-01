#!/usr/bin/env python3
"""Compare FP16 energy results across GPU architectures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_float(x: Any, default: float = math.nan) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s.upper() in {"N/A", "[N/A]"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    seen = set(keys)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_from_row(row: Dict[str, Any]) -> Dict[str, str]:
    gpu = str(row.get("gpu", "") or row.get("device_name", "") or "")
    cc = str(row.get("compute_capability", "") or "")
    generation = str(row.get("architecture_generation", "") or "")
    chip = str(row.get("architecture_chip", "") or "")
    product_class = str(row.get("gpu_product_class", "") or "")
    cuda_arch = str(row.get("recommended_cuda_arch", "") or "")

    if generation and chip:
        return {
            "gpu": gpu,
            "architecture_generation": generation,
            "architecture_chip": chip,
            "gpu_product_class": product_class,
            "recommended_cuda_arch": cuda_arch,
        }

    if "H100" in gpu or cc.startswith("9."):
        return {
            "gpu": gpu,
            "architecture_generation": "hopper",
            "architecture_chip": "gh100" if "H100" in gpu else "hopper_sm90",
            "gpu_product_class": "datacenter",
            "recommended_cuda_arch": "90",
        }
    if "A100" in gpu or cc == "8.0":
        return {
            "gpu": gpu,
            "architecture_generation": "ampere",
            "architecture_chip": "ga100" if "A100" in gpu else "ampere_sm80",
            "gpu_product_class": "datacenter",
            "recommended_cuda_arch": "80",
        }
    if "3090" in gpu or cc == "8.6":
        return {
            "gpu": gpu,
            "architecture_generation": "ampere",
            "architecture_chip": "ga102" if "3090" in gpu else "ampere_sm86",
            "gpu_product_class": "consumer" if "RTX" in gpu else "workstation_or_consumer",
            "recommended_cuda_arch": "86",
        }
    return {
        "gpu": gpu,
        "architecture_generation": generation or "unknown",
        "architecture_chip": chip or "unknown",
        "gpu_product_class": product_class or "unknown",
        "recommended_cuda_arch": cuda_arch,
    }


def arch_label(row: Dict[str, Any], fallback: str) -> str:
    arch = str(row.get("architecture_chip", "") or "unknown")
    gpu = str(row.get("gpu", "") or "").replace("NVIDIA ", "")
    if gpu:
        return f"{arch}\n{gpu}"
    return f"{arch}\n{fallback}"


def result_label(path: Path, row: Optional[Dict[str, Any]] = None) -> str:
    if row:
        label = arch_label(row, path.name)
        if "unknown" not in label:
            return label
    return path.name


def load_result_dir(
    path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    condition_rows = read_csv(path / "condition_summary.csv")
    summary_rows = read_csv(path / "summary.csv")
    thread_rows = read_csv(path / "thread_sweep_summary.csv")
    quality_rows = read_csv(path / "quality_gates.csv")
    seed_row = (condition_rows or summary_rows or [{}])[0]
    arch = classify_from_row(seed_row)
    label = result_label(path, arch)

    quality_thread: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in quality_rows:
        if str(row.get("scope", "")) != "thread_sweep":
            continue
        key = (str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")), str(row.get("threads", "")))
        quality_thread[key] = row

    def enrich(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            row_arch = classify_from_row({**arch, **row})
            enriched = {
                **row,
                **row_arch,
                "input_dir": str(path),
                "architecture_label": label,
            }
            if "threads" in row:
                key = (
                    str(row.get("test_kernel", "")),
                    str(row.get("baseline_kernel", "")),
                    str(row.get("threads", "")),
                )
                q = quality_thread.get(key)
                if q:
                    for qkey in (
                        "measurement_grade",
                        "baseline_match_grade",
                        "quality_pass",
                        "target_pass",
                        "baseline_structural_match",
                        "energy_source_reliable",
                        "ncu_validation_pass",
                        "ncu_required",
                        "test_ncu_note",
                        "baseline_ncu_note",
                        "fail_reasons",
                        "warnings",
                    ):
                        enriched[qkey] = q.get(qkey, "")
            out.append(enriched)
        return out

    return enrich(condition_rows), enrich(summary_rows), enrich(thread_rows), enrich(quality_rows)


def is_fp16_candidate(row: Dict[str, Any]) -> bool:
    test_kernel = str(row.get("test_kernel", ""))
    if test_kernel not in {"tensor_mma_f16acc", "tensor_mma_f32acc", "fp16_half2"}:
        return False
    pure_count = int(parse_float(row.get("pure_fp16_candidate_count"), 0.0))
    valid_no_l2 = int(parse_float(row.get("valid_no_l2_count"), 0.0))
    valid = int(parse_float(row.get("valid_count"), 0.0))
    return pure_count > 0 or valid_no_l2 > 0 or valid > 0


def select_best_fp16(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_input: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if is_fp16_candidate(row):
            by_input.setdefault(str(row.get("input_dir", "")), []).append(row)

    selected: List[Dict[str, Any]] = []
    for group in by_input.values():
        strict_targets = [row for row in group if parse_bool(row.get("target_pass"))]
        strict_quality = [row for row in group if parse_bool(row.get("quality_pass"))]
        has_quality_info = any("quality_pass" in row for row in group)
        if strict_targets:
            pool = strict_targets
            selection_note = "quality_gate_target_pass"
        elif strict_quality:
            pool = strict_quality
            selection_note = "quality_gate_quality_pass_no_target"
        elif has_quality_info:
            rejected = dict(max(group, key=lambda row: (
                int(parse_float(row.get("pure_fp16_candidate_count"), 0.0)),
                int(parse_float(row.get("valid_no_l2_count"), 0.0)),
                int(parse_float(row.get("valid_count"), 0.0)),
            )))
            rejected["selection_note"] = "quality_gate_failed_no_best"
            rejected["quality_rejected"] = True
            rejected["matmul_input_pj_per_bit_mean"] = math.nan
            rejected["tflops_mean"] = math.nan
            rejected["incremental_power_w_mean"] = math.nan
            selected.append(rejected)
            continue
        else:
            pool = group
            selection_note = "legacy_or_no_quality_gate"

        def score(row: Dict[str, Any]) -> Tuple[int, int, int, float, float]:
            pure_count = int(parse_float(row.get("pure_fp16_candidate_count"), 0.0))
            valid_no_l2 = int(parse_float(row.get("valid_no_l2_count"), 0.0))
            valid = int(parse_float(row.get("valid_count"), 0.0))
            util = parse_float(row.get("avg_sm_util_pct_mean"), -math.inf)
            if not math.isfinite(util):
                util = parse_float(row.get("avg_gpu_util_pct_mean"), -math.inf)
            tflops = parse_float(row.get("tflops_mean"), -math.inf)
            return (pure_count, valid_no_l2, valid, util, tflops)

        best = dict(max(pool, key=score))
        best["selection_note"] = selection_note
        best["quality_rejected"] = False
        selected.append(best)
    return selected


def plot_bar(rows: List[Dict[str, Any]], metric: str, ylabel: str, title: str, path: Path) -> None:
    clean = [r for r in rows if math.isfinite(parse_float(r.get(metric)))]
    if not clean:
        return
    labels = [str(r.get("architecture_label", r.get("input_dir", ""))) for r in clean]
    vals = [parse_float(r.get(metric)) for r in clean]
    yerr = []
    for r in clean:
        std_key = metric.replace("_mean", "_std") if metric.endswith("_mean") else ""
        v = parse_float(r.get(std_key), 0.0) if std_key else 0.0
        yerr.append(v if math.isfinite(v) and v >= 0.0 else 0.0)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.7 * len(clean)), 5.0))
    ax.bar(range(len(clean)), vals, yerr=yerr, capsize=4)
    ax.set_xticks(range(len(clean)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_thread_compare(thread_rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not thread_rows:
        return
    by_kernel: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in thread_rows:
        key = (str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")))
        by_kernel.setdefault(key, []).append(row)

    for (test_kernel, baseline_kernel), rows in by_kernel.items():
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for label in sorted({str(r.get("architecture_label", "")) for r in rows}):
            group = [r for r in rows if str(r.get("architecture_label", "")) == label]
            group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
            xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
            ys = [parse_float(r.get("avg_sm_util_pct_mean")) for r in group]
            if not any(math.isfinite(y) for y in ys):
                ys = [parse_float(r.get("avg_gpu_util_pct_mean")) for r in group]
            if any(math.isfinite(y) for y in ys):
                ax.plot(xs, ys, marker="o", label=label)
            for r, x, y in zip(group, xs, ys):
                if bool(str(r.get("selected_optimal", "")).lower() == "true") and math.isfinite(x):
                    ax.axvline(x, color="0.35", linestyle="--", linewidth=0.9, alpha=0.5)
                    if math.isfinite(y):
                        ax.annotate("selected", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)

        ax.set_xlabel("Launched threads per SM")
        ax.set_ylabel("Avg SM utilization (%)")
        ax.set_title(f"Architecture thread sweep utilization: {test_kernel} vs {baseline_kernel}")
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        safe = f"architecture_thread_sweep_util_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
        fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plotted = False
        for label in sorted({str(r.get("architecture_label", "")) for r in rows}):
            group = [r for r in rows if str(r.get("architecture_label", "")) == label]
            group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
            xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
            ys = [parse_float(r.get("matmul_input_pj_per_bit_mean")) for r in group]
            if any(math.isfinite(y) for y in ys):
                ax.plot(xs, ys, marker="D", label=label)
                plotted = True
        if plotted:
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("pJ/logical input bit")
            ax.set_title(f"Architecture thread sweep pJ/bit: {test_kernel} vs {baseline_kernel}")
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            safe = f"architecture_thread_sweep_pjbit_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
            fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare analyzed FP16 energy result directories")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more analyzed result dirs")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for comparison CSVs/figures")
    args = parser.parse_args()

    all_conditions: List[Dict[str, Any]] = []
    all_summary: List[Dict[str, Any]] = []
    all_threads: List[Dict[str, Any]] = []
    all_quality: List[Dict[str, Any]] = []
    for path in args.input:
        conditions, summary, threads, quality = load_result_dir(path)
        if not conditions and not summary:
            raise SystemExit(f"{path} has no summary.csv or condition_summary.csv; run analyze_results.py first")
        all_conditions.extend(conditions)
        all_summary.extend(summary)
        all_threads.extend(threads)
        all_quality.extend(quality)

    args.outdir.mkdir(parents=True, exist_ok=True)
    best_source = all_threads if all_threads else all_conditions
    best = select_best_fp16(best_source)
    write_csv(args.outdir / "architecture_condition_summary.csv", all_conditions)
    write_csv(args.outdir / "architecture_summary_rows.csv", all_summary)
    write_csv(args.outdir / "architecture_thread_sweep_summary.csv", all_threads)
    write_csv(args.outdir / "architecture_quality_gates.csv", all_quality)
    write_csv(args.outdir / "architecture_best_fp16.csv", best)

    plot_bar(
        best,
        "matmul_input_pj_per_bit_mean",
        "pJ/logical input bit",
        "Best pure-FP16 matmul energy candidate by architecture",
        args.outdir / "architecture_best_matmul_input_pj_per_bit.png",
    )
    plot_bar(
        best,
        "tflops_mean",
        "TFLOPS",
        "Best pure-FP16 throughput candidate by architecture",
        args.outdir / "architecture_best_tflops.png",
    )
    plot_bar(
        best,
        "incremental_power_w_mean",
        "Incremental power (W)",
        "Best pure-FP16 incremental power by architecture",
        args.outdir / "architecture_best_incremental_power.png",
    )
    plot_thread_compare(all_threads, args.outdir)

    print(f"Wrote: {args.outdir / 'architecture_condition_summary.csv'}")
    print(f"Wrote: {args.outdir / 'architecture_best_fp16.csv'}")
    print(f"Wrote figures under: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
