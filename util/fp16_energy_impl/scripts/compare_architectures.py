#!/usr/bin/env python3
"""Compare FP16 energy results across GPU architectures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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


def normalize_int_text(value: Any) -> str:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return str(int(round(parsed)))
    return str(value or "")


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


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

    quality_thread: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in quality_rows:
        if str(row.get("scope", "")) != "thread_sweep":
            continue
        key = (
            str(row.get("test_kernel", "")),
            str(row.get("baseline_kernel", "")),
            normalize_int_text(row.get("threads", "")),
            normalize_int_text(row.get("blocks_per_sm_requested", "")),
        )
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
                    normalize_int_text(row.get("threads", "")),
                    normalize_int_text(row.get("blocks_per_sm_requested", "")),
                )
                legacy_key = (key[0], key[1], key[2], "")
                q = quality_thread.get(key) or quality_thread.get(legacy_key)
                if q:
                    for qkey in (
                        "measurement_grade",
                        "baseline_match_grade",
                        "quality_pass",
                        "target_pass",
                        "quality_gate_selected_target",
                        "baseline_structural_match",
                        "energy_source_reliable",
                        "energy_trace_crosscheck_pass",
                        "energy_signal_reliable",
                        "measurement_resolution_reliable",
                        "benchmark_schema_current",
                        "test_benchmark_schema_versions",
                        "baseline_benchmark_schema_versions",
                        "matmul_denominator_valid",
                        "matmul_denominator_note",
                        "matmul_denominator_metadata_complete",
                        "matmul_denominator_source",
                        "matmul_logical_mma_count_mean",
                        "matmul_flops_per_logical_mma",
                        "matmul_input_bits_per_logical_mma",
                        "ncu_validation_pass",
                        "ncu_validation_context_match",
                        "ncu_required",
                        "ncu_tensor_activity_required",
                        "test_ncu_note",
                        "baseline_ncu_note",
                        "test_ncu_validation_blocks_per_sm",
                        "baseline_ncu_validation_blocks_per_sm",
                        "test_ncu_validation_unroll",
                        "baseline_ncu_validation_unroll",
                        "test_ncu_validation_suppress_output_store",
                        "baseline_ncu_validation_suppress_output_store",
                        "test_ncu_tensor_activity_pct",
                        "baseline_ncu_tensor_activity_pct",
                        "test_ncu_sm_activity_pct",
                        "baseline_ncu_sm_activity_pct",
                        "test_ncu_tensor_activity_observed",
                        "baseline_ncu_tensor_activity_observed",
                        "test_ncu_memory_counter_classes_complete",
                        "baseline_ncu_memory_counter_classes_complete",
                        "test_ncu_common_hmma_seen",
                        "baseline_ncu_common_hmma_seen",
                        "test_ncu_hmma_token_seen",
                        "baseline_ncu_hmma_token_seen",
                        "test_ncu_tensor_inst_seen",
                        "baseline_ncu_tensor_inst_seen",
                        "test_ncu_wgmma_token_seen",
                        "baseline_ncu_wgmma_token_seen",
                        "test_ncu_no_l2",
                        "baseline_ncu_no_l2",
                        "test_ncu_no_dram",
                        "baseline_ncu_no_dram",
                        "test_ncu_no_local_spill",
                        "baseline_ncu_no_local_spill",
                        "util_saturated",
                        "util_reference_scope",
                        "util_reference_max_pct",
                        "util_metric_source",
                        "target_selection_note",
                        "incremental_energy_fraction_mean",
                        "elapsed_s_mean",
                        "test_energy_j_mean",
                        "incremental_energy_j_mean",
                        "test_energy_counter_vs_trace_ratio_mean",
                        "baseline_energy_counter_vs_trace_ratio_mean",
                        "test_energy_counter_vs_trace_delta_j_mean",
                        "baseline_energy_counter_vs_trace_delta_j_mean",
                        "baseline_energy_fraction_mean",
                        "baseline_power_fraction_mean",
                        "tensor_peak_tflops_model_mean",
                        "achieved_flops_per_sm_cycle_mean",
                        "tensor_model_utilization_pct_mean",
                        "fail_reasons",
                        "warnings",
                    ):
                        enriched[qkey] = q.get(qkey, "")
            out.append(enriched)
        return out

    return enrich(condition_rows), enrich(summary_rows), enrich(thread_rows), enrich(quality_rows)


def load_resource_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv(path / "resource_audit" / "thread_resource_occupancy.csv")
    if not rows:
        return []
    seed = rows[0]
    arch = classify_from_row(seed)
    label = result_label(path, arch)
    return [
        {
            **row,
            **classify_from_row({**arch, **row}),
            "input_dir": str(path),
            "architecture_label": label,
        }
        for row in rows
    ]


def load_audit_rows(audit_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if not audit_dir:
        return []
    rows = read_csv(audit_dir / "strict_result_audit.csv")
    out: List[Dict[str, Any]] = []
    for row in rows:
        arch = classify_from_row(row)
        input_dir = Path(str(row.get("input_dir", "") or audit_dir))
        out.append(
            {
                **row,
                **arch,
                "input_dir": str(input_dir),
                "architecture_label": result_label(input_dir, arch),
                "audit_evidence_source": str(audit_dir / "strict_result_audit.csv"),
            }
        )
    return out


def required_pair_matches(row: Dict[str, Any], required_kernel: str, required_baseline: str) -> bool:
    if required_kernel and str(row.get("test_kernel", "")) != required_kernel:
        return False
    if required_baseline and str(row.get("baseline_kernel", "")) != required_baseline:
        return False
    return True


def audit_best_rows(
    audit_rows: List[Dict[str, Any]],
    *,
    required_kernel: str,
    required_baseline: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in audit_rows:
        pair_matches = required_pair_matches(row, required_kernel, required_baseline)
        publishable = parse_bool(row.get("audit_pass")) and pair_matches
        best = dict(row)
        best["selection_note"] = (
            "strict_audit_pass"
            if publishable
            else ("strict_audit_required_kernel_baseline_mismatch" if not pair_matches else "strict_audit_failed")
        )
        best["quality_rejected"] = not publishable
        best["audit_required_for_publishable"] = True
        best["publishable_strict_audit"] = publishable
        best["required_kernel"] = required_kernel
        best["required_baseline"] = required_baseline
        out.append(best)
    return out


def is_fp16_candidate(row: Dict[str, Any]) -> bool:
    test_kernel = str(row.get("test_kernel", ""))
    if test_kernel not in {"tensor_mma_f16acc", "tensor_mma_f32acc", "fp16_half2"}:
        return False
    pure_count = int(parse_float(row.get("pure_fp16_candidate_count"), 0.0))
    valid_no_l2 = int(parse_float(row.get("valid_no_l2_count"), 0.0))
    valid = int(parse_float(row.get("valid_count"), 0.0))
    return pure_count > 0 or valid_no_l2 > 0 or valid > 0


def ncu_tensor_activity_ok(row: Dict[str, Any]) -> bool:
    if not str(row.get("test_kernel", "")).startswith("tensor_mma_"):
        return True
    value = str(row.get("test_ncu_tensor_activity_observed", "")).strip()
    pct = parse_float(row.get("test_ncu_tensor_activity_pct"))
    return parse_bool(value) and math.isfinite(pct) and pct > 0.0


def ncu_common_hmma_path_ok(row: Dict[str, Any]) -> bool:
    if not str(row.get("test_kernel", "")).startswith("tensor_mma_"):
        return True
    return (
        parse_bool(row.get("test_ncu_common_hmma_seen"))
        and not parse_bool(row.get("test_ncu_wgmma_token_seen"))
        and not parse_bool(row.get("baseline_ncu_common_hmma_seen"))
        and not parse_bool(row.get("baseline_ncu_tensor_inst_seen"))
        and not parse_bool(row.get("baseline_ncu_wgmma_token_seen"))
    )


def ncu_no_memory_ok(row: Dict[str, Any]) -> bool:
    return (
        parse_bool(row.get("test_ncu_memory_counter_classes_complete"))
        and parse_bool(row.get("baseline_ncu_memory_counter_classes_complete"))
        and parse_bool(row.get("test_ncu_no_l2"))
        and parse_bool(row.get("baseline_ncu_no_l2"))
        and parse_bool(row.get("test_ncu_no_dram"))
        and parse_bool(row.get("baseline_ncu_no_dram"))
        and parse_bool(row.get("test_ncu_no_local_spill"))
        and parse_bool(row.get("baseline_ncu_no_local_spill"))
    )


def is_strict_publishable_target(row: Dict[str, Any]) -> bool:
    if str(row.get("audit_required_for_publishable", "")).strip() or str(row.get("audit_pass", "")).strip():
        return (
            not parse_bool(row.get("quality_rejected"))
            and parse_bool(row.get("audit_pass"))
            and parse_bool(row.get("target_pass"))
        )
    return (
        parse_bool(row.get("target_pass"))
        and str(row.get("measurement_grade", "")) == "strict_nvml_counter"
        and parse_bool(row.get("ncu_required"))
        and parse_bool(row.get("ncu_validation_pass"))
        and parse_bool(row.get("ncu_validation_context_match"))
        and ncu_tensor_activity_ok(row)
        and ncu_common_hmma_path_ok(row)
        and ncu_no_memory_ok(row)
        and str(row.get("baseline_match_grade", "")) == "structural_baseline"
        and parse_bool(row.get("matmul_denominator_valid"))
        and parse_bool(row.get("matmul_denominator_metadata_complete"))
        and str(row.get("matmul_denominator_source", "")) == "bench_json_metadata"
    )


def reject_best_candidate(group: List[Dict[str, Any]], selection_note: str) -> Dict[str, Any]:
    rejected = dict(max(group, key=lambda row: (
        int(parse_float(row.get("pure_fp16_candidate_count"), 0.0)),
        int(parse_float(row.get("valid_no_l2_count"), 0.0)),
        int(parse_float(row.get("valid_count"), 0.0)),
    )))
    rejected["selection_note"] = selection_note
    rejected["quality_rejected"] = True
    rejected["target_pass"] = False
    rejected["matmul_input_pj_per_bit_mean"] = math.nan
    rejected["tflops_mean"] = math.nan
    rejected["incremental_power_w_mean"] = math.nan
    rejected["tensor_model_utilization_pct_mean"] = math.nan
    return rejected


def tag_required_pair(row: Dict[str, Any], required_kernel: str, required_baseline: str) -> Dict[str, Any]:
    row["required_kernel"] = required_kernel
    row["required_baseline"] = required_baseline
    return row


def select_best_fp16(
    rows: List[Dict[str, Any]],
    *,
    allow_diagnostic_fallback: bool = False,
    allow_legacy_best: bool = False,
    required_kernel: str = "tensor_mma_f16acc",
    required_baseline: str = "tensor_baseline_mov",
) -> List[Dict[str, Any]]:
    by_input: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if is_fp16_candidate(row):
            by_input.setdefault(str(row.get("input_dir", "")), []).append(row)

    selected: List[Dict[str, Any]] = []
    for group in by_input.values():
        required_group = [
            row for row in group
            if required_pair_matches(row, required_kernel, required_baseline)
        ]
        if not required_group:
            selected.append(
                tag_required_pair(
                    reject_best_candidate(group, "missing_required_kernel_baseline"),
                    required_kernel,
                    required_baseline,
                )
            )
            continue
        group = required_group
        target_rows = [row for row in group if parse_bool(row.get("target_pass"))]
        strict_targets = [
            row
            for row in target_rows
            if is_strict_publishable_target(row)
        ]
        strict_quality = [
            row
            for row in group
            if parse_bool(row.get("quality_pass"))
            and str(row.get("measurement_grade", "")) == "strict_nvml_counter"
            and parse_bool(row.get("ncu_required"))
            and parse_bool(row.get("ncu_validation_pass"))
            and parse_bool(row.get("ncu_validation_context_match"))
            and ncu_tensor_activity_ok(row)
            and ncu_common_hmma_path_ok(row)
            and ncu_no_memory_ok(row)
        ]
        diagnostic_quality = [row for row in group if parse_bool(row.get("quality_pass"))]
        has_quality_info = any(
            key in row
            for row in group
            for key in ("quality_pass", "target_pass", "measurement_grade")
        )
        if strict_targets:
            pool = strict_targets
            selection_note = "quality_gate_strict_nvml_target_pass"
        elif diagnostic_quality and allow_diagnostic_fallback:
            pool = diagnostic_quality
            selection_note = "quality_gate_quality_pass_no_target_diagnostic"
        elif has_quality_info:
            strict_energy_targets = [
                row for row in target_rows if str(row.get("measurement_grade", "")) == "strict_nvml_counter"
            ]
            if strict_energy_targets:
                has_ncu_context = any(
                    parse_bool(row.get("ncu_required"))
                    and parse_bool(row.get("ncu_validation_pass"))
                    and parse_bool(row.get("ncu_validation_context_match"))
                    for row in strict_energy_targets
                )
                if has_ncu_context and any(not ncu_tensor_activity_ok(row) for row in strict_energy_targets):
                    note = "quality_gate_target_pass_without_required_tensor_activity_evidence"
                elif has_ncu_context and any(not ncu_common_hmma_path_ok(row) for row in strict_energy_targets):
                    note = "quality_gate_target_pass_without_required_hmma_evidence"
                elif has_ncu_context and any(not ncu_no_memory_ok(row) for row in strict_energy_targets):
                    note = "quality_gate_target_pass_without_required_no_memory_evidence"
                else:
                    note = "quality_gate_target_pass_without_required_ncu_evidence"
            elif target_rows:
                note = "quality_gate_target_pass_without_strict_nvml_counter"
            elif strict_quality:
                note = "quality_gate_no_target_pass"
            elif diagnostic_quality:
                note = "quality_gate_quality_pass_without_strict_nvml_counter"
            else:
                note = "quality_gate_failed_no_best"
            selected.append(tag_required_pair(reject_best_candidate(group, note), required_kernel, required_baseline))
            continue
        elif not allow_legacy_best:
            selected.append(
                tag_required_pair(
                    reject_best_candidate(group, "missing_quality_gate_no_best"),
                    required_kernel,
                    required_baseline,
                )
            )
            continue
        else:
            pool = group
            selection_note = "legacy_or_no_quality_gate_diagnostic"

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
        selected.append(tag_required_pair(best, required_kernel, required_baseline))
    return selected


def plot_bar(rows: List[Dict[str, Any]], metric: str, ylabel: str, title: str, path: Path) -> None:
    clean = [r for r in rows if math.isfinite(parse_float(r.get(metric)))]
    if not clean:
        return
    labels = [str(r.get("architecture_label", r.get("input_dir", ""))) for r in clean]
    vals = [parse_float(r.get(metric)) for r in clean]
    colors = [
        "tab:green" if is_strict_publishable_target(r) else ("tab:red" if parse_bool(r.get("quality_rejected")) else "tab:orange")
        for r in clean
    ]
    yerr = []
    for r in clean:
        std_key = metric.replace("_mean", "_std") if metric.endswith("_mean") else ""
        v = parse_float(r.get(std_key), 0.0) if std_key else 0.0
        yerr.append(v if math.isfinite(v) and v >= 0.0 else 0.0)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.7 * len(clean)), 5.0))
    ax.bar(range(len(clean)), vals, yerr=yerr, capsize=4, color=colors, alpha=0.86)
    ax.set_xticks(range(len(clean)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="tab:green", label="strict audit publishable"),
            Patch(facecolor="tab:red", label="rejected/diagnostic"),
            Patch(facecolor="tab:orange", label="diagnostic best"),
        ],
        loc="best",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def quality_class(row: Dict[str, Any]) -> str:
    has_quality = str(row.get("quality_pass", "")).strip() != ""
    if is_strict_publishable_target(row):
        return "target"
    if parse_bool(row.get("target_pass")):
        return "diagnostic_target"
    if parse_bool(row.get("quality_pass")):
        return "quality"
    if has_quality:
        return "failed"
    return "legacy"


def scatter_quality_point(ax: Any, x: float, y: float, row: Dict[str, Any], color: str) -> None:
    if not math.isfinite(x) or not math.isfinite(y):
        return
    cls = quality_class(row)
    if cls == "target":
        ax.scatter([x], [y], marker="*", s=135, c=[color], edgecolors="black", linewidths=0.8, zorder=5)
    elif cls == "diagnostic_target":
        ax.scatter([x], [y], marker="D", s=70, facecolors="tab:orange", edgecolors="black", linewidths=0.8, zorder=5)
    elif cls == "quality":
        ax.scatter([x], [y], marker="o", s=52, facecolors="white", edgecolors=color, linewidths=1.5, zorder=4)
    elif cls == "failed":
        ax.scatter([x], [y], marker="x", s=58, c="tab:red", linewidths=1.5, zorder=4)
    else:
        ax.scatter([x], [y], marker="s", s=38, facecolors="0.85", edgecolors=color, linewidths=1.0, zorder=4)


def quality_marker_style(row: Dict[str, Any]) -> Dict[str, Any]:
    cls = quality_class(row)
    if cls == "target":
        return {"marker": "*", "s": 150, "linewidths": 0.8}
    if cls == "diagnostic_target":
        return {"marker": "D", "s": 74, "linewidths": 0.8}
    if cls == "quality":
        return {"marker": "o", "s": 58, "linewidths": 1.3}
    if cls == "failed":
        return {"marker": "x", "s": 62, "linewidths": 1.5}
    return {"marker": "s", "s": 42, "linewidths": 1.0}


def scatter_quality_point_with_pjbit(
    ax: Any,
    x: float,
    y: float,
    row: Dict[str, Any],
    *,
    norm: Normalize,
    cmap: str,
    edge_color: str,
) -> bool:
    pjbit = parse_float(row.get("matmul_input_pj_per_bit_mean"))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(pjbit)):
        return False
    style = quality_marker_style(row)
    kwargs = {
        "marker": style["marker"],
        "s": style["s"],
        "linewidths": style["linewidths"],
        "c": [pjbit],
        "cmap": cmap,
        "norm": norm,
        "zorder": 5 if quality_class(row) in {"target", "diagnostic_target"} else 4,
    }
    if style["marker"] != "x":
        kwargs["edgecolors"] = "black" if quality_class(row) in {"target", "diagnostic_target"} else edge_color
    ax.scatter([x], [y], **kwargs)
    return True


def add_quality_legend(ax: Any) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="0.55", markeredgecolor="black",
               label="publishable strict target", markersize=10),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="tab:orange", markeredgecolor="black",
               label="diagnostic target", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="0.35",
               label="quality_pass diagnostic", markersize=7),
        Line2D([0], [0], marker="x", color="tab:red", label="quality fail", markersize=7),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="0.85", markeredgecolor="0.35",
               label="legacy/no gate", markersize=6),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, title="gate state")


def selected_annotation(row: Dict[str, Any]) -> str:
    cls = quality_class(row)
    if cls == "target":
        label = "strict\npublishable"
    elif cls == "diagnostic_target":
        label = "diagnostic\ntarget"
    else:
        label = "selected\nnot target"
    threads = str(row.get("threads", ""))
    if threads:
        blocks = str(row.get("blocks_per_sm_requested", ""))
        label += f"\nt{threads}"
        if blocks:
            label += f"/b{blocks}"
    pjbit = parse_float(row.get("matmul_input_pj_per_bit_mean"))
    if math.isfinite(pjbit):
        label += f"\n{pjbit:.3g} pJ/b"
    return label


def selected_energy_fraction_annotation(row: Dict[str, Any]) -> str:
    label = selected_annotation(row)
    incremental = parse_float(row.get("incremental_energy_fraction_mean"))
    baseline = parse_float(row.get("baseline_energy_fraction_mean"))
    if math.isfinite(incremental):
        label += f"\ninc {100.0 * incremental:.1f}%"
    if math.isfinite(baseline):
        label += f"\nbase {100.0 * baseline:.1f}%"
    return label


def add_energy_fraction_legend(ax: Any) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="0.25", linestyle="-", label="incremental FP16 signal"),
        Line2D([0], [0], color="0.25", linestyle="--", label="baseline-scaled energy"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, title="energy fraction")
    ax.add_artist(legend)


def plot_thread_compare(thread_rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not thread_rows:
        return
    by_kernel: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in thread_rows:
        key = (str(row.get("test_kernel", "")), str(row.get("baseline_kernel", "")))
        by_kernel.setdefault(key, []).append(row)

    for (test_kernel, baseline_kernel), rows in by_kernel.items():
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plotted = False
        for label in sorted({str(r.get("architecture_label", "")) for r in rows}):
            group = [r for r in rows if str(r.get("architecture_label", "")) == label]
            group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
            xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
            ys = [parse_float(r.get("avg_sm_util_pct_mean")) for r in group]
            if not any(math.isfinite(y) for y in ys):
                ys = [parse_float(r.get("avg_gpu_util_pct_mean")) for r in group]
            if any(math.isfinite(y) for y in ys):
                line = ax.plot(xs, ys, linewidth=1.2, alpha=0.75, label=label)[0]
                for r, x, y in zip(group, xs, ys):
                    scatter_quality_point(ax, x, y, r, line.get_color())
                plotted = True
            finite_group_y = [y for y in ys if math.isfinite(y)]
            top_y = max(finite_group_y) if finite_group_y else math.nan
            for r, x, y in zip(group, xs, ys):
                cls = quality_class(r)
                is_target = cls == "target"
                is_diagnostic_target = cls == "diagnostic_target"
                is_analyzer_selected = parse_bool(r.get("selected_optimal"))
                if (is_target or is_diagnostic_target or is_analyzer_selected) and math.isfinite(x):
                    if is_target:
                        color = "tab:green"
                        linestyle = "--"
                    elif is_diagnostic_target:
                        color = "tab:orange"
                        linestyle = "-."
                    else:
                        color = "0.35"
                        linestyle = ":"
                    ax.axvline(x, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.55)
                    if math.isfinite(y):
                        near_top = (math.isfinite(top_y) and y >= top_y - 0.1) or y >= 95.0
                        ax.annotate(
                            selected_annotation(r),
                            (x, y),
                            textcoords="offset points",
                            xytext=(6, -18 if near_top else 6),
                            va="top" if near_top else "bottom",
                            fontsize=8,
                            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
                        )

        if plotted:
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("Avg SM utilization (%)")
            ax.set_title(f"Architecture thread sweep utilization: {test_kernel} vs {baseline_kernel}")
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.25)
            arch_legend = ax.legend(loc="best", title="architecture")
            ax.add_artist(arch_legend)
            add_quality_legend(ax)
            fig.tight_layout()
            safe = f"architecture_thread_sweep_util_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
            fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)

        finite_pjbit_rows = [
            r for r in rows if math.isfinite(parse_float(r.get("matmul_input_pj_per_bit_mean")))
        ]
        if finite_pjbit_rows:
            pj_vals = [parse_float(r.get("matmul_input_pj_per_bit_mean")) for r in finite_pjbit_rows]
            vmin = min(pj_vals)
            vmax = max(pj_vals)
            if vmin == vmax:
                delta = abs(vmin) * 0.05 if vmin else 1.0
                vmin -= delta
                vmax += delta
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = "viridis_r"
            fig, ax = plt.subplots(figsize=(9.4, 5.4))
            plotted = False
            for label in sorted({str(r.get("architecture_label", "")) for r in rows}):
                group = [r for r in rows if str(r.get("architecture_label", "")) == label]
                group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
                xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
                ys = [parse_float(r.get("avg_sm_util_pct_mean")) for r in group]
                if not any(math.isfinite(y) for y in ys):
                    ys = [parse_float(r.get("avg_gpu_util_pct_mean")) for r in group]
                if not any(math.isfinite(y) for y in ys):
                    continue
                line = ax.plot(xs, ys, linewidth=1.0, alpha=0.35, label=label)[0]
                edge_color = line.get_color()
                finite_xs = [x for x in xs if math.isfinite(x)]
                finite_ys = [y for y in ys if math.isfinite(y)]
                x_max = max(finite_xs) if finite_xs else math.nan
                y_max = max(finite_ys) if finite_ys else math.nan
                for r, x, y in zip(group, xs, ys):
                    plotted = scatter_quality_point_with_pjbit(
                        ax,
                        x,
                        y,
                        r,
                        norm=norm,
                        cmap=cmap,
                        edge_color=edge_color,
                    ) or plotted

                label_rows = [
                    r
                    for r in group
                    if quality_class(r) in {"target", "diagnostic_target"} or parse_bool(r.get("selected_optimal"))
                ]
                valid_no_l2 = [
                    r
                    for r in group
                    if parse_float(r.get("matmul_input_pj_per_bit_mean"), math.inf) > 0.0
                    and parse_float(r.get("valid_no_l2_count"), 0.0) > 0.0
                ]
                if valid_no_l2:
                    label_rows.append(
                        min(valid_no_l2, key=lambda r: parse_float(r.get("matmul_input_pj_per_bit_mean"), math.inf))
                    )
                seen_label_ids = set()
                for r in label_rows:
                    row_id = id(r)
                    if row_id in seen_label_ids:
                        continue
                    seen_label_ids.add(row_id)
                    x = parse_float(r.get("threads_per_sm"), parse_float(r.get("threads")))
                    y = parse_float(r.get("avg_sm_util_pct_mean"))
                    if not math.isfinite(y):
                        y = parse_float(r.get("avg_gpu_util_pct_mean"))
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    label_text = selected_annotation(r)
                    if quality_class(r) not in {"target", "diagnostic_target"} and not parse_bool(r.get("selected_optimal")):
                        label_text = "lowest pJ/b\n" + label_text
                    near_right = math.isfinite(x_max) and x >= x_max * 0.78
                    near_top = (math.isfinite(y_max) and y >= y_max - 0.2) or y >= 98.0
                    x_offset = -8 if near_right else 8
                    y_offset = -34 if near_top else 8
                    ax.annotate(
                        label_text,
                        (x, y),
                        textcoords="offset points",
                        xytext=(x_offset, y_offset),
                        ha="right" if near_right else "left",
                        va="top" if near_top else "bottom",
                        fontsize=8,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.76},
                    )

            if plotted:
                ax.set_xlabel("Launched threads per SM")
                ax.set_ylabel("Avg SM utilization (%)")
                ax.set_title(
                    f"Architecture thread sweep utilization colored by pJ/bit: {test_kernel} vs {baseline_kernel}"
                )
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.grid(True, axis="y", alpha=0.25)
                arch_legend = ax.legend(loc="best", title="architecture")
                ax.add_artist(arch_legend)
                add_quality_legend(ax)
                cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
                cbar.set_label("pJ/logical input bit")
                fig.tight_layout()
                safe = f"architecture_thread_sweep_util_pjbit_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
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
                line = ax.plot(xs, ys, linewidth=1.2, alpha=0.75, label=label)[0]
                for r, x, y in zip(group, xs, ys):
                    scatter_quality_point(ax, x, y, r, line.get_color())
                    if quality_class(r) == "target" and math.isfinite(x) and math.isfinite(y):
                        ax.annotate(
                            f"{r.get('threads', '')}\n{y:.3g} pJ/b",
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center",
                            fontsize=8,
                        )
                plotted = True
        if plotted:
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("pJ/logical input bit")
            ax.set_title(f"Architecture thread sweep pJ/bit: {test_kernel} vs {baseline_kernel}")
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.25)
            arch_legend = ax.legend(loc="best", title="architecture")
            ax.add_artist(arch_legend)
            add_quality_legend(ax)
            fig.tight_layout()
            safe = f"architecture_thread_sweep_pjbit_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
            fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plotted = False
        for label in sorted({str(r.get("architecture_label", "")) for r in rows}):
            group = [r for r in rows if str(r.get("architecture_label", "")) == label]
            group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
            xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
            ys = [parse_float(r.get("tensor_model_utilization_pct_mean")) for r in group]
            if any(math.isfinite(y) for y in ys):
                line = ax.plot(xs, ys, linewidth=1.2, alpha=0.75, label=label)[0]
                for r, x, y in zip(group, xs, ys):
                    scatter_quality_point(ax, x, y, r, line.get_color())
                plotted = True
        if plotted:
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("Dense Tensor Core model utilization (%)")
            ax.set_title(f"Architecture thread sweep Tensor Core model utilization: {test_kernel} vs {baseline_kernel}")
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.25)
            arch_legend = ax.legend(loc="best", title="architecture")
            ax.add_artist(arch_legend)
            add_quality_legend(ax)
            fig.tight_layout()
            safe = f"architecture_thread_sweep_model_util_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
            fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plotted = False
        arch_handles = []
        for label in sorted({str(r.get("architecture_label", "")) for r in rows}):
            group = [r for r in rows if str(r.get("architecture_label", "")) == label]
            group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
            xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
            incremental = [parse_float(r.get("incremental_energy_fraction_mean")) for r in group]
            baseline = [parse_float(r.get("baseline_energy_fraction_mean")) for r in group]
            if any(math.isfinite(y) for y in incremental) or any(math.isfinite(y) for y in baseline):
                line = ax.plot(xs, incremental, linewidth=1.3, alpha=0.85, label=label)[0]
                ax.plot(xs, baseline, linewidth=1.1, alpha=0.65, linestyle="--", color=line.get_color())
                arch_handles.append(line)
                for r, x, y in zip(group, xs, incremental):
                    scatter_quality_point(ax, x, y, r, line.get_color())
                    if quality_class(r) == "target" and math.isfinite(x) and math.isfinite(y):
                        ax.axvline(x, color="tab:green", linestyle="--", linewidth=0.9, alpha=0.55)
                        ax.annotate(
                            selected_energy_fraction_annotation(r),
                            (x, y),
                            textcoords="offset points",
                            xytext=(8, 8),
                            fontsize=8,
                            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.78},
                        )
                plotted = True
        if plotted:
            ax.axhline(0.01, color="tab:red", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.axhline(0.05, color="tab:orange", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.axhline(0.99, color="tab:red", linestyle=":", linewidth=0.9, alpha=0.6)
            threshold_label_box = {"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.1}
            ax.text(
                0.01,
                0.01,
                "min 1%",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=8,
                bbox=threshold_label_box,
            )
            ax.text(
                0.01,
                0.05,
                "warn 5%",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=8,
                bbox=threshold_label_box,
            )
            ax.text(
                0.48,
                0.99,
                "baseline max 99%",
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                bbox=threshold_label_box,
            )
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("Fraction of test energy")
            ax.set_title(f"Architecture thread sweep energy separation: {test_kernel} vs {baseline_kernel}")
            ax.set_ylim(-0.02, 1.05)
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.25)
            if arch_handles:
                arch_legend = ax.legend(handles=arch_handles, loc="upper right", title="architecture")
                ax.add_artist(arch_legend)
            add_energy_fraction_legend(ax)
            add_quality_legend(ax)
            fig.tight_layout()
            safe = f"architecture_thread_sweep_energy_fraction_{test_kernel}_vs_{baseline_kernel}.png".replace("/", "_")
            fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)


def plot_resource_compare(resource_rows: List[Dict[str, Any]], outdir: Path) -> None:
    rows = [r for r in resource_rows if str(r.get("role", "")) == "test"]
    if not rows:
        return
    by_kernel: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_kernel.setdefault(str(row.get("kernel", "")), []).append(row)
    for kernel, group_rows in by_kernel.items():
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plotted = False
        for label in sorted({str(r.get("architecture_label", "")) for r in group_rows}):
            group = [r for r in group_rows if str(r.get("architecture_label", "")) == label]
            group = sorted(group, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
            xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in group]
            ys = [parse_float(r.get("thread_occupancy_pct_model")) for r in group]
            if any(math.isfinite(y) for y in ys):
                ax.plot(xs, ys, marker="o", label=label)
                plotted = True
        if plotted:
            ax.set_xlabel("Launched threads per SM")
            ax.set_ylabel("Resource occupancy model (%)")
            ax.set_title(f"Architecture resource occupancy model: {kernel}")
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            safe = f"architecture_resource_occupancy_{kernel}.png".replace("/", "_")
            fig.savefig(outdir / safe, dpi=160)
        plt.close(fig)


def coverage_rows(
    best_rows: List[Dict[str, Any]],
    thread_rows: List[Dict[str, Any]],
    condition_rows: List[Dict[str, Any]],
    required_architectures: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    chips = sorted(
        set(required_architectures)
        | {str(row.get("architecture_chip", "")) for row in condition_rows if str(row.get("architecture_chip", ""))}
        | {str(row.get("architecture_chip", "")) for row in thread_rows if str(row.get("architecture_chip", ""))}
        | {str(row.get("architecture_chip", "")) for row in best_rows if str(row.get("architecture_chip", ""))}
    )
    for chip in chips:
        chip_conditions = [row for row in condition_rows if str(row.get("architecture_chip", "")) == chip]
        chip_threads = [row for row in thread_rows if str(row.get("architecture_chip", "")) == chip]
        chip_best = [row for row in best_rows if str(row.get("architecture_chip", "")) == chip]
        strict_thread_targets = [
            row for row in chip_threads if is_strict_publishable_target(row)
        ]
        strict_best = [
            row for row in chip_best
            if not parse_bool(row.get("quality_rejected"))
            and is_strict_publishable_target(row)
        ]
        strict_targets = [*strict_thread_targets, *strict_best]
        diagnostic_targets = [
            row for row in chip_threads
            if parse_bool(row.get("target_pass"))
            and str(row.get("measurement_grade", "")) != "strict_nvml_counter"
        ]
        quality_pass = [row for row in chip_threads if parse_bool(row.get("quality_pass"))]
        if strict_best:
            status = "strict_pass"
            publishable = True
        elif chip_conditions or chip_threads or chip_best:
            status = "diagnostic_or_rejected_only"
            publishable = False
        else:
            status = "missing_result"
            publishable = False

        def best_score(row: Dict[str, Any]) -> Tuple[float, float]:
            pjbit = parse_float(row.get("matmul_input_pj_per_bit_mean"), math.inf)
            util = parse_float(row.get("avg_sm_util_pct_mean"), -math.inf)
            return (pjbit if math.isfinite(pjbit) else math.inf, -util if math.isfinite(util) else math.inf)

        selected = min(strict_best, key=best_score) if strict_best else (chip_best[0] if chip_best else {})
        input_dirs = sorted({
            str(row.get("input_dir", ""))
            for row in [*chip_conditions, *chip_threads, *chip_best]
            if str(row.get("input_dir", "")).strip()
        })
        rows.append(
            {
                "architecture_chip": chip,
                "required": chip in required_architectures,
                "coverage_status": status,
                "publishable": publishable,
                "input_dir_count": len(input_dirs),
                "input_dirs": ";".join(input_dirs),
                "thread_row_count": len(chip_threads),
                "strict_target_count": len(strict_targets),
                "diagnostic_target_count": len(diagnostic_targets),
                "quality_pass_count": len(quality_pass),
                "best_selection_note": selected.get("selection_note", ""),
                "best_quality_rejected": selected.get("quality_rejected", ""),
                "best_gpu": selected.get("gpu", ""),
                "best_threads": selected.get("threads", ""),
                "best_threads_per_sm": selected.get("threads_per_sm", ""),
                "best_matmul_input_pj_per_bit_mean": selected.get("matmul_input_pj_per_bit_mean", ""),
                "best_tflops_mean": selected.get("tflops_mean", ""),
                "best_tensor_model_utilization_pct_mean": selected.get("tensor_model_utilization_pct_mean", ""),
                "best_measurement_grade": selected.get("measurement_grade", ""),
                "best_target_pass": selected.get("target_pass", ""),
                "best_baseline_match_grade": selected.get("baseline_match_grade", ""),
            }
        )
    return rows


def plot_coverage(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    required = [row for row in rows if parse_bool(row.get("required"))]
    plot_rows = required if required else rows
    labels = [str(row.get("architecture_chip", "")) for row in plot_rows]
    values = [
        2 if str(row.get("coverage_status", "")) == "strict_pass"
        else (1 if str(row.get("coverage_status", "")) == "diagnostic_or_rejected_only" else 0)
        for row in plot_rows
    ]
    colors = [
        "tab:green" if value == 2 else ("tab:orange" if value == 1 else "tab:red")
        for value in values
    ]
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.65 * len(plot_rows) + 1.8)))
    y = list(range(len(plot_rows)))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 2.35)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["missing", "diagnostic", "strict"])
    ax.set_xlabel("Strict comparison coverage")
    ax.set_title("Required architecture coverage for FP16 pJ/bit comparison")
    ax.grid(True, axis="x", alpha=0.25)
    for idx, row in enumerate(plot_rows):
        status = str(row.get("coverage_status", ""))
        pjbit = parse_float(row.get("best_matmul_input_pj_per_bit_mean"))
        detail = status
        if math.isfinite(pjbit):
            detail += f"\n{pjbit:.3g} pJ/b"
        elif status == "missing_result":
            detail += "\nno result dir"
        else:
            detail += "\nno strict best"
        ax.annotate(
            detail,
            (values[idx], idx),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(outdir / "architecture_strict_coverage.png", dpi=160)
    plt.close(fig)


def comparison_summary(
    coverage: List[Dict[str, Any]],
    *,
    required_architectures: List[str],
    required_kernel: str,
    required_baseline: str,
    allow_diagnostic_best: bool,
    allow_legacy_best: bool,
    audit_dir: Optional[Path],
) -> Dict[str, Any]:
    required_rows = [row for row in coverage if parse_bool(row.get("required"))]
    strict_pass = [
        row for row in required_rows
        if str(row.get("coverage_status", "")) == "strict_pass" and parse_bool(row.get("publishable"))
    ]
    missing_or_nonpublishable = [
        str(row.get("architecture_chip", ""))
        for row in required_rows
        if not parse_bool(row.get("publishable"))
    ]
    status_counts: Dict[str, int] = {}
    for row in coverage:
        status = str(row.get("coverage_status", "") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    diagnostic_only = [
        str(row.get("architecture_chip", ""))
        for row in required_rows
        if str(row.get("coverage_status", "")) == "diagnostic_or_rejected_only"
    ]
    missing = [
        str(row.get("architecture_chip", ""))
        for row in required_rows
        if str(row.get("coverage_status", "")) == "missing_result"
    ]
    publishable = bool(required_rows) and len(strict_pass) == len(required_rows)
    return {
        "summary_schema": "fp16-architecture-comparison-summary-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "publishable": publishable,
        "comparison_status": "publishable_strict" if publishable else "not_publishable_incomplete_coverage",
        "required_architectures": required_architectures,
        "required_kernel": required_kernel,
        "required_baseline": required_baseline,
        "required_architecture_count": len(required_rows),
        "required_strict_pass_count": len(strict_pass),
        "required_missing_or_nonpublishable_architectures": missing_or_nonpublishable,
        "required_missing_architectures": missing,
        "required_diagnostic_only_architectures": diagnostic_only,
        "coverage_status_counts": status_counts,
        "allow_diagnostic_best": allow_diagnostic_best,
        "allow_legacy_best": allow_legacy_best,
        "audit_dir": str(audit_dir or ""),
        "audit_required_for_publishable": bool(audit_dir),
        "final_value_policy": (
            "When --audit-dir is supplied, only audit_pass=true rows are publishable FP16 pJ/bit "
            "comparison points. Without --audit-dir, strict quality-gate targets are diagnostic "
            "readiness evidence and should still be audited before final claims."
        ),
    }


def plot_comparison_readiness(summary: Dict[str, Any], coverage: List[Dict[str, Any]], outdir: Path) -> None:
    required_rows = [row for row in coverage if parse_bool(row.get("required"))]
    if not required_rows:
        return
    counts = {
        "missing": sum(1 for row in required_rows if str(row.get("coverage_status", "")) == "missing_result"),
        "diagnostic": sum(
            1 for row in required_rows if str(row.get("coverage_status", "")) == "diagnostic_or_rejected_only"
        ),
        "strict": sum(1 for row in required_rows if str(row.get("coverage_status", "")) == "strict_pass"),
    }
    labels = ["missing", "diagnostic", "strict"]
    values = [counts[label] for label in labels]
    colors = ["tab:red", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(labels, values, color=colors, alpha=0.88)
    ax.set_ylabel("Required architecture count")
    ax.set_title("FP16 architecture comparison readiness")
    ax.set_ylim(0, max(values + [1]) + 0.6)
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.annotate(str(value), (idx, value), textcoords="offset points", xytext=(0, 5), ha="center")
    if not parse_bool(summary.get("publishable")):
        missing = ",".join(summary.get("required_missing_or_nonpublishable_architectures", []))
        ax.text(
            0.5,
            0.92,
            f"NOT PUBLISHABLE: missing strict coverage for {missing}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="tab:red",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "tab:red", "alpha": 0.9},
        )
    else:
        ax.text(
            0.5,
            0.92,
            "PUBLISHABLE STRICT COVERAGE",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="tab:green",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "tab:green", "alpha": 0.9},
        )
    fig.tight_layout()
    fig.savefig(outdir / "architecture_comparison_readiness.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare analyzed FP16 energy result directories")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more analyzed result dirs")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for comparison CSVs/figures")
    parser.add_argument("--audit-dir", type=Path, default=None, help="Directory from audit_strict_results.py")
    parser.add_argument(
        "--require-architectures",
        default="ga100,gh100,ga102",
        help="Required architecture chips for coverage CSV/figure [ga100,gh100,ga102]",
    )
    parser.add_argument("--require-kernel", default="tensor_mma_f16acc", help="Required test kernel for best/coverage")
    parser.add_argument(
        "--require-baseline",
        default="tensor_baseline_mov",
        help="Required baseline kernel for best/coverage",
    )
    parser.add_argument(
        "--fail-on-missing-required-architectures",
        action="store_true",
        help="Return nonzero when any required architecture lacks a strict NVML-counter target",
    )
    parser.add_argument(
        "--allow-diagnostic-best",
        action="store_true",
        help=(
            "Permit quality_pass rows without a strict NVML-counter target_pass to populate "
            "architecture_best_fp16.csv. By default, quality-gated result directories require "
            "target_pass and measurement_grade=strict_nvml_counter."
        ),
    )
    parser.add_argument(
        "--allow-legacy-best",
        action="store_true",
        help=(
            "Diagnostic mode: permit result directories without quality gate metadata to populate "
            "architecture_best_fp16.csv. Default strict comparison rejects legacy/no-gate inputs."
        ),
    )
    args = parser.parse_args()

    all_conditions: List[Dict[str, Any]] = []
    all_summary: List[Dict[str, Any]] = []
    all_threads: List[Dict[str, Any]] = []
    all_quality: List[Dict[str, Any]] = []
    all_resources: List[Dict[str, Any]] = []
    audit_rows = load_audit_rows(args.audit_dir)
    for path in args.input:
        conditions, summary, threads, quality = load_result_dir(path)
        if not conditions and not summary:
            raise SystemExit(f"{path} has no summary.csv or condition_summary.csv; run analyze_results.py first")
        all_conditions.extend(conditions)
        all_summary.extend(summary)
        all_threads.extend(threads)
        all_quality.extend(quality)
        all_resources.extend(load_resource_rows(path))

    args.outdir.mkdir(parents=True, exist_ok=True)
    if audit_rows:
        best = audit_best_rows(
            audit_rows,
            required_kernel=args.require_kernel,
            required_baseline=args.require_baseline,
        )
    else:
        best_source = all_threads if all_threads else all_conditions
        best = select_best_fp16(
            best_source,
            allow_diagnostic_fallback=args.allow_diagnostic_best,
            allow_legacy_best=args.allow_legacy_best,
            required_kernel=args.require_kernel,
            required_baseline=args.require_baseline,
        )
    required_architectures = parse_csv_list(args.require_architectures)
    coverage = coverage_rows(best, all_threads, all_conditions, required_architectures)
    summary = comparison_summary(
        coverage,
        required_architectures=required_architectures,
        required_kernel=args.require_kernel,
        required_baseline=args.require_baseline,
        allow_diagnostic_best=args.allow_diagnostic_best,
        allow_legacy_best=args.allow_legacy_best,
        audit_dir=args.audit_dir,
    )
    write_csv(args.outdir / "architecture_condition_summary.csv", all_conditions)
    write_csv(args.outdir / "architecture_summary_rows.csv", all_summary)
    write_csv(args.outdir / "architecture_thread_sweep_summary.csv", all_threads)
    write_csv(args.outdir / "architecture_quality_gates.csv", all_quality)
    write_csv(args.outdir / "architecture_resource_occupancy.csv", all_resources)
    write_csv(args.outdir / "architecture_audit_strict.csv", audit_rows)
    write_csv(args.outdir / "architecture_best_fp16.csv", best)
    write_csv(args.outdir / "architecture_strict_coverage.csv", coverage)
    write_csv(args.outdir / "architecture_comparison_summary.csv", [summary])
    write_json(args.outdir / "architecture_comparison_summary.json", summary)

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
        "tensor_model_utilization_pct_mean",
        "Dense Tensor Core model utilization (%)",
        "Best pure-FP16 Tensor Core model utilization by architecture",
        args.outdir / "architecture_best_tensor_model_utilization.png",
    )
    plot_bar(
        best,
        "incremental_power_w_mean",
        "Incremental power (W)",
        "Best pure-FP16 incremental power by architecture",
        args.outdir / "architecture_best_incremental_power.png",
    )
    plot_bar(
        best,
        "incremental_energy_fraction_mean",
        "Incremental energy / test energy",
        "Best pure-FP16 energy signal fraction by architecture",
        args.outdir / "architecture_best_incremental_energy_fraction.png",
    )
    plot_thread_compare(all_threads, args.outdir)
    plot_resource_compare(all_resources, args.outdir)
    plot_coverage(coverage, args.outdir)
    plot_comparison_readiness(summary, coverage, args.outdir)

    print(f"Wrote: {args.outdir / 'architecture_condition_summary.csv'}")
    print(f"Wrote: {args.outdir / 'architecture_best_fp16.csv'}")
    print(f"Wrote: {args.outdir / 'architecture_strict_coverage.csv'}")
    print(f"Wrote: {args.outdir / 'architecture_comparison_summary.json'}")
    print(f"Wrote figures under: {args.outdir}")
    missing = [
        row for row in coverage
        if parse_bool(row.get("required")) and not parse_bool(row.get("publishable"))
    ]
    if args.fail_on_missing_required_architectures and missing:
        missing_chips = ",".join(str(row.get("architecture_chip", "")) for row in missing)
        raise SystemExit(f"Missing strict publishable architecture coverage: {missing_chips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
