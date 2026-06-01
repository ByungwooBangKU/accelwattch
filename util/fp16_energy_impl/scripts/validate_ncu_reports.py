#!/usr/bin/env python3
"""Validate Nsight Compute reports for FP16 no-L2/HMMA evidence.

The validator is intentionally conservative. A final FP16 pJ/bit result should
not pass only because the benchmark metadata says the kernel avoids global
memory. It should also have profiler evidence for:

* expected HMMA/Tensor Core instruction path on tensor_mma_* kernels
* no unexpected HMMA in structural Tensor baselines
* no material L2/DRAM/local-memory traffic inside the profiled kernel

Nsight Compute report formats vary across versions. This script accepts plain
text logs produced by the local ncu_validate*.sh helpers and looks for common
metric IDs/labels. If any required DRAM/L2/local counter class is missing,
strict mode fails the row and reports exactly what evidence is missing. Sector
counters are converted to normalized bytes before thresholding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


METRIC_ALIASES = {
    "hmma_inst": [
        "smsp__inst_executed_pipe_tensor_op_hmma.sum",
        "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
        "sm__inst_executed_pipe_tensor_op_hmma.sum",
    ],
    "tensor_inst": [
        "smsp__inst_executed_pipe_tensor.sum",
        "sm__inst_executed_pipe_tensor.sum",
    ],
    "dram_bytes_read": [
        "dram__bytes_read.sum",
        "dram__sectors_read.sum",
    ],
    "dram_bytes_write": [
        "dram__bytes_write.sum",
        "dram__sectors_write.sum",
    ],
    "l2_bytes_read": [
        "lts__t_bytes_read.sum",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_op_read.sum",
    ],
    "l2_bytes_write": [
        "lts__t_bytes_write.sum",
        "lts__t_sectors_srcunit_tex_op_write.sum",
        "lts__t_sectors_op_write.sum",
    ],
    "local_load": [
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    ],
    "local_store": [
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    ],
}

MEMORY_COUNTER_CLASSES = {
    "dram": ["dram_bytes_read", "dram_bytes_write"],
    "l2": ["l2_bytes_read", "l2_bytes_write"],
    "local": ["local_load", "local_store"],
}

MEMORY_METRICS = [
    "dram_bytes_read",
    "dram_bytes_write",
    "l2_bytes_read",
    "l2_bytes_write",
    "local_load",
    "local_store",
]

PROFILER_ERROR_PATTERNS = [
    "ERR_NVGPUCTRPERM",
    "No kernels were profiled",
    "Permission denied",
    "Failed to prepare kernel",
    "LaunchFailed",
    "not found",
]

GLOBAL_MEMORY_TOKENS = re.compile(r"\b(LDG|STG|LD\.GLOBAL|ST\.GLOBAL|ATOM\.GLOBAL)\b", re.IGNORECASE)
LOCAL_MEMORY_TOKENS = re.compile(r"\b(LDL|STL|LOCAL)\b", re.IGNORECASE)
HMMA_TOKENS = re.compile(r"\b(HMMA|MMA\.SYNC)\b", re.IGNORECASE)


def parse_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    scale = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "G", "T"}:
        scale = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[suffix]
        text = text[:-1]
    try:
        return float(text) * scale
    except ValueError:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_report_name(path: Path) -> Tuple[str, str]:
    stem = path.name
    for suffix in (".ncu.txt", ".txt", ".log", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = re.match(r"(?P<kernel>.+)_t(?P<threads>\d+)$", stem)
    if match:
        return match.group("kernel"), match.group("threads")
    return stem, ""


def numeric_values_from_line(line: str) -> List[float]:
    # Nsight Compute text often contains commas or units. Use the last numeric
    # value on a metric line as the metric value.
    values: List[float] = []
    for token in re.findall(r"[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?[KMGT]?", line):
        token = token.strip()
        if token in {"", "+", "-"}:
            continue
        value = parse_float(token)
        if math.isfinite(value):
            values.append(value)
    return values


def extract_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    for line in text.splitlines():
        lowered = line.lower()
        for name, aliases in METRIC_ALIASES.items():
            matched_alias = next((alias for alias in aliases if alias.lower() in lowered), "")
            if matched_alias:
                values = numeric_values_from_line(line)
                if values:
                    metrics[name] = values[-1]
                    sources[name] = matched_alias
    metrics["_sources"] = sources  # type: ignore[assignment]
    return metrics


def metric_sources(metrics: Dict[str, Any]) -> Dict[str, str]:
    raw = metrics.get("_sources", {})
    return raw if isinstance(raw, dict) else {}


def counter_to_bytes(metric: str, value: float, sources: Dict[str, str], args: argparse.Namespace) -> float:
    if not math.isfinite(value):
        return math.nan
    source = sources.get(metric, "").lower()
    if "sectors" not in source:
        return value
    if metric.startswith("dram"):
        return value * args.dram_sector_bytes
    if metric.startswith("l2"):
        return value * args.l2_sector_bytes
    if metric.startswith("local"):
        return value * args.local_sector_bytes
    return value


def class_present(metrics: Dict[str, Any], cls: str) -> bool:
    return any(name in metrics for name in MEMORY_COUNTER_CLASSES[cls])


def class_total_bytes(metrics: Dict[str, Any], sources: Dict[str, str], cls: str, args: argparse.Namespace) -> float:
    total = 0.0
    seen = False
    for name in MEMORY_COUNTER_CLASSES[cls]:
        raw = metrics.get(name, math.nan)
        if math.isfinite(raw):
            total += counter_to_bytes(name, raw, sources, args)
            seen = True
    return total if seen else math.nan


def class_source_summary(sources: Dict[str, str], cls: str) -> str:
    return ",".join(
        f"{name}:{sources[name]}"
        for name in MEMORY_COUNTER_CLASSES[cls]
        if name in sources
    )


def profiler_errors(text: str) -> List[str]:
    errors = []
    for pattern in PROFILER_ERROR_PATTERNS:
        if pattern.lower() in text.lower():
            errors.append(pattern)
    return errors


def validate_report(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    text = path.read_text(errors="replace")
    kernel, threads = parse_report_name(path)
    metrics = extract_metrics(text)
    sources = metric_sources(metrics)
    errors = profiler_errors(text)

    expected_hmma = kernel.startswith("tensor_mma_")
    baseline_tensor = kernel.startswith("tensor_baseline_")
    hmma_value = metrics.get("hmma_inst", metrics.get("tensor_inst", math.nan))
    hmma_metric_present = "hmma_inst" in metrics or "tensor_inst" in metrics
    hmma_token_seen = bool(HMMA_TOKENS.search(text))
    hmma_seen = (
        (math.isfinite(hmma_value) and hmma_value > 0.0)
        or (not hmma_metric_present and hmma_token_seen)
    )
    hmma_ok = (hmma_seen if expected_hmma else (not hmma_seen if baseline_tensor else True))

    dram_present = class_present(metrics, "dram")
    l2_present = class_present(metrics, "l2")
    local_present = class_present(metrics, "local")
    memory_present = any((dram_present, l2_present, local_present))
    memory_complete = all((dram_present, l2_present, local_present))
    dram_total = class_total_bytes(metrics, sources, "dram", args)
    l2_total = class_total_bytes(metrics, sources, "l2", args)
    local_total = class_total_bytes(metrics, sources, "local", args)

    has_global_tokens = bool(GLOBAL_MEMORY_TOKENS.search(text))
    has_local_tokens = bool(LOCAL_MEMORY_TOKENS.search(text))

    if memory_complete:
        no_dram = dram_total <= args.max_dram_counter
        no_l2 = l2_total <= args.max_l2_counter
        no_local = local_total <= args.max_local_counter
        memory_note = "counter_based"
    else:
        no_dram = (
            dram_total <= args.max_dram_counter
            if dram_present and math.isfinite(dram_total)
            else args.allow_missing_counters and not has_global_tokens
        )
        no_l2 = (
            l2_total <= args.max_l2_counter
            if l2_present and math.isfinite(l2_total)
            else args.allow_missing_counters and not has_global_tokens
        )
        no_local = (
            local_total <= args.max_local_counter
            if local_present and math.isfinite(local_total)
            else args.allow_missing_counters and not has_local_tokens
        )
        memory_note = (
            "partial_counters_token_fallback"
            if args.allow_missing_counters and memory_present
            else ("missing_counters_token_fallback" if args.allow_missing_counters else "missing_counters")
        )

    fail_reasons: List[str] = []
    warnings: List[str] = []
    if errors:
        fail_reasons.append("profiler errors: " + ",".join(errors))
    if expected_hmma and not hmma_ok:
        fail_reasons.append("missing HMMA/Tensor Core instruction evidence")
    if baseline_tensor and not hmma_ok:
        fail_reasons.append("baseline shows unexpected HMMA/Tensor Core evidence")
    if not memory_complete and not args.allow_missing_counters:
        missing = [
            name
            for name, present in (("DRAM", dram_present), ("L2", l2_present), ("local", local_present))
            if not present
        ]
        fail_reasons.append("missing explicit counter class(es): " + ",".join(missing))
    if not no_l2:
        fail_reasons.append("L2/global traffic counter or token evidence exceeds threshold")
    if not no_dram:
        fail_reasons.append("DRAM traffic counter or token evidence exceeds threshold")
    if not no_local:
        fail_reasons.append("local memory/spill counter or token evidence exceeds threshold")
    if args.allow_missing_counters and not memory_present:
        warnings.append("used token fallback because explicit memory counters were missing")
    elif args.allow_missing_counters and memory_present and not memory_complete:
        warnings.append("used token fallback for missing memory counter classes")

    validation_pass = not fail_reasons
    return {
        "report": str(path),
        "kernel": kernel,
        "threads": threads,
        "expected_hmma": expected_hmma,
        "baseline_tensor": baseline_tensor,
        "validation_pass": validation_pass,
        "hmma_ok": hmma_ok,
        "hmma_seen": hmma_seen,
        "hmma_metric_present": hmma_metric_present,
        "hmma_token_seen": hmma_token_seen,
        "memory_counters_present": memory_present,
        "dram_counters_present": dram_present,
        "l2_counters_present": l2_present,
        "local_counters_present": local_present,
        "memory_counter_classes_complete": memory_complete,
        "memory_note": memory_note,
        "no_l2": no_l2,
        "no_dram": no_dram,
        "no_local_spill": no_local,
        "hmma_inst": metrics.get("hmma_inst", math.nan),
        "tensor_inst": metrics.get("tensor_inst", math.nan),
        "dram_counter_total": dram_total if math.isfinite(dram_total) else math.nan,
        "l2_counter_total": l2_total if math.isfinite(l2_total) else math.nan,
        "local_counter_total": local_total if math.isfinite(local_total) else math.nan,
        "dram_counter_sources": class_source_summary(sources, "dram"),
        "l2_counter_sources": class_source_summary(sources, "l2"),
        "local_counter_sources": class_source_summary(sources, "local"),
        "has_global_memory_tokens": has_global_tokens,
        "has_local_memory_tokens": has_local_tokens,
        "fail_reasons": "; ".join(fail_reasons),
        "warnings": "; ".join(warnings),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "report",
        "kernel",
        "threads",
        "expected_hmma",
        "baseline_tensor",
        "validation_pass",
        "hmma_ok",
        "hmma_seen",
        "hmma_metric_present",
        "hmma_token_seen",
        "memory_counters_present",
        "dram_counters_present",
        "l2_counters_present",
        "local_counters_present",
        "memory_counter_classes_complete",
        "memory_note",
        "no_l2",
        "no_dram",
        "no_local_spill",
        "hmma_inst",
        "tensor_inst",
        "dram_counter_total",
        "l2_counter_total",
        "local_counter_total",
        "dram_counter_sources",
        "l2_counter_sources",
        "local_counter_sources",
        "has_global_memory_tokens",
        "has_local_memory_tokens",
        "fail_reasons",
        "warnings",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (str(r.get("kernel", "")), parse_float(r.get("threads"), -1.0)))
    labels = [
        f"{r['kernel']}\n{r['threads'] or '-'}"
        for r in ordered
    ]
    values = [1.0 if parse_bool(r.get("validation_pass")) else 0.0 for r in ordered]
    colors = ["tab:green" if v == 1.0 else "tab:red" for v in values]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(ordered)), 4.8))
    ax.bar(range(len(ordered)), values, color=colors)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["fail", "pass"])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("NCU validation")
    ax.set_title("Nsight Compute FP16 no-L2/HMMA validation")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "ncu_validation_summary.png", dpi=160)
    plt.close(fig)


def plot_memory_counters(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda r: (str(r.get("kernel", "")), parse_float(r.get("threads"), -1.0)))
    labels = [f"{r['kernel']}\n{r['threads'] or '-'}" for r in ordered]
    dram = [parse_float(r.get("dram_counter_total"), 0.0) for r in ordered]
    l2 = [parse_float(r.get("l2_counter_total"), 0.0) for r in ordered]
    local = [parse_float(r.get("local_counter_total"), 0.0) for r in ordered]
    if not any(math.isfinite(v) and v > 0.0 for v in dram + l2 + local):
        # Still create the plot when explicit counters are present and all are zero.
        if not any(parse_bool(r.get("memory_counters_present")) for r in ordered):
            return

    outdir.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(ordered)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(ordered)), 4.8))
    ax.bar([x - width for x in xs], [v if math.isfinite(v) else 0.0 for v in dram], width=width, label="DRAM")
    ax.bar(xs, [v if math.isfinite(v) else 0.0 for v in l2], width=width, label="L2")
    ax.bar([x + width for x in xs], [v if math.isfinite(v) else 0.0 for v in local], width=width, label="local")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Normalized bytes")
    ax.set_title("Nsight Compute memory counter evidence")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "ncu_memory_counter_bytes.png", dpi=160)
    plt.close(fig)


def summarize(rows: List[Dict[str, Any]], input_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    counts = Counter("pass" if parse_bool(r.get("validation_pass")) else "fail" for r in rows)
    return {
        "input": str(input_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "max_l2_counter": args.max_l2_counter,
            "max_dram_counter": args.max_dram_counter,
            "max_local_counter": args.max_local_counter,
            "l2_sector_bytes": args.l2_sector_bytes,
            "dram_sector_bytes": args.dram_sector_bytes,
            "local_sector_bytes": args.local_sector_bytes,
            "allow_missing_counters": args.allow_missing_counters,
        },
        "counts": {
            "reports": len(rows),
            "pass": counts.get("pass", 0),
            "fail": counts.get("fail", 0),
        },
        "notes": [
            "Strict mode requires explicit L2/DRAM/local counter metrics.",
            "Counters with 'sectors' in the metric ID are normalized to bytes before thresholding.",
            "Token fallback is diagnostic only; use explicit counters for final pJ/bit claims.",
            "Rows with validation_pass=false must not be used as final pure-FP16 evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Nsight Compute text reports for FP16 evidence")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing *.ncu.txt reports")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory; defaults to --input")
    parser.add_argument("--max-l2-counter", type=float, default=0.0)
    parser.add_argument("--max-dram-counter", type=float, default=0.0)
    parser.add_argument("--max-local-counter", type=float, default=0.0)
    parser.add_argument("--l2-sector-bytes", type=float, default=32.0)
    parser.add_argument("--dram-sector-bytes", type=float, default=32.0)
    parser.add_argument("--local-sector-bytes", type=float, default=32.0)
    parser.add_argument(
        "--allow-missing-counters",
        action="store_true",
        help="Diagnostic fallback: use token search if explicit memory counters are missing",
    )
    args = parser.parse_args()

    outdir = args.outdir or args.input
    reports = sorted(args.input.glob("*.ncu.txt"))
    rows = [validate_report(path, args) for path in reports]
    write_csv(outdir / "ncu_validation_summary.csv", rows)
    with (outdir / "ncu_validation_summary.json").open("w") as f:
        json.dump(summarize(rows, args.input, args), f, indent=2)
    plot_summary(rows, outdir / "figures")
    plot_memory_counters(rows, outdir / "figures")

    print(f"Wrote: {outdir / 'ncu_validation_summary.csv'}")
    print(f"Wrote: {outdir / 'ncu_validation_summary.json'}")
    if rows:
        print(f"Wrote figures under: {outdir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
