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


def find_ncu_row(rows: List[Dict[str, Any]], kernel: str, threads: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("kernel", "")) == kernel and normalize_thread(row.get("threads", "")) == threads:
            return row
    for row in rows:
        if str(row.get("kernel", "")) == kernel and not str(row.get("threads", "")):
            return row
    return {}


def audit_dir(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    summary = read_json(path / "quality_gate_summary.json")
    quality_rows = read_csv(path / "quality_gates.csv")
    ncu_rows = read_csv(path / "ncu_no_l2_thread_sweep" / "ncu_validation_summary.csv")
    targets = summary.get("selected_targets") or []
    target = dict(targets[0]) if targets else {}

    if not target and quality_rows:
        # Keep useful context in failed audits.
        thread_rows = [r for r in quality_rows if str(r.get("scope", "")) == "thread_sweep"]
        target = dict(thread_rows[0]) if thread_rows else dict(quality_rows[0])

    chip = str(target.get("architecture_chip", "") or "unknown")
    generation = str(target.get("architecture_generation", "") or "unknown")
    gpu = str(target.get("gpu", "") or "")
    test_kernel = str(target.get("test_kernel", "") or "")
    baseline_kernel = str(target.get("baseline_kernel", "") or "")
    threads = normalize_thread(target.get("threads", ""))
    test_ncu = find_ncu_row(ncu_rows, test_kernel, threads)
    baseline_ncu = find_ncu_row(ncu_rows, baseline_kernel, threads)

    failed: List[str] = []
    warnings: List[str] = []
    if not targets:
        failed.append("quality_gate_summary selected_targets is empty")
    if not parse_bool(target.get("target_pass")):
        failed.append("target_pass is not true")
    if not parse_bool(target.get("quality_pass")):
        failed.append("quality_pass is not true")
    if str(target.get("measurement_grade", "")) != "strict_nvml_counter":
        failed.append("measurement_grade is not strict_nvml_counter")
    if str(target.get("baseline_match_grade", "")) != "structural_baseline":
        failed.append("baseline_match_grade is not structural_baseline")
    if not parse_bool(target.get("energy_source_reliable")):
        failed.append("energy_source_reliable is not true")
    if not parse_bool(target.get("baseline_structural_match")):
        failed.append("baseline_structural_match is not true")
    if not parse_bool(target.get("ncu_required")):
        failed.append("ncu_required is not true")
    if not parse_bool(target.get("ncu_validation_pass")):
        failed.append("ncu_validation_pass is not true")
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
    sm_util = parse_float(target.get("avg_sm_util_pct_mean"))
    if not math.isfinite(sm_util):
        warnings.append("avg_sm_util_pct_mean is missing")
    if not ncu_rows:
        failed.append("ncu_validation_summary.csv is missing or empty")
    if ncu_rows and not test_ncu:
        failed.append("missing NCU validation row for selected test kernel/thread")
    if ncu_rows and not baseline_ncu:
        failed.append("missing NCU validation row for selected baseline kernel/thread")
    if test_ncu and not parse_bool(test_ncu.get("validation_pass")):
        failed.append("selected test NCU validation did not pass")
    if baseline_ncu and not parse_bool(baseline_ncu.get("validation_pass")):
        failed.append("selected baseline NCU validation did not pass")

    return {
        "input_dir": str(path),
        "audit_pass": not failed,
        "gpu": gpu,
        "architecture_generation": generation,
        "architecture_chip": chip,
        "test_kernel": test_kernel,
        "baseline_kernel": baseline_kernel,
        "threads": threads,
        "threads_per_sm": target.get("threads_per_sm", ""),
        "measurement_grade": target.get("measurement_grade", ""),
        "baseline_match_grade": target.get("baseline_match_grade", ""),
        "target_pass": target.get("target_pass", ""),
        "quality_pass": target.get("quality_pass", ""),
        "ncu_required": target.get("ncu_required", ""),
        "ncu_validation_pass": target.get("ncu_validation_pass", ""),
        "valid_no_l2_count": target.get("valid_no_l2_count", ""),
        "required_valid_count": target.get("required_valid_count", ""),
        "avg_sm_util_pct_mean": target.get("avg_sm_util_pct_mean", ""),
        "tflops_mean": target.get("tflops_mean", ""),
        "matmul_input_pj_per_bit_mean": target.get("matmul_input_pj_per_bit_mean", ""),
        "incremental_power_w_mean": target.get("incremental_power_w_mean", ""),
        "test_ncu_pass": test_ncu.get("validation_pass", "") if test_ncu else "",
        "baseline_ncu_pass": baseline_ncu.get("validation_pass", "") if baseline_ncu else "",
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


def write_json(path: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    required = [x.strip() for x in args.require_architectures.split(",") if x.strip()]
    passed = {str(r.get("architecture_chip", "")) for r in rows if parse_bool(r.get("audit_pass"))}
    missing = [chip for chip in required if chip not in passed]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dirs": [str(p) for p in args.input],
        "required_architectures": required,
        "missing_required_architectures": missing,
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
    parser.add_argument("--require-baseline", default="tensor_baseline_u32")
    parser.add_argument("--no-fail", action="store_true", help="Write audit files but return success even if audit fails")
    args = parser.parse_args()

    rows = [audit_dir(path, args) for path in args.input]
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "strict_result_audit.csv", rows)
    write_json(args.outdir / "strict_result_audit.json", rows, args)
    plot_audit(rows, args.outdir / "figures")

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
