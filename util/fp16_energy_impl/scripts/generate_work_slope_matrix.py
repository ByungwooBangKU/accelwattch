#!/usr/bin/env python3
"""Generate a work-slope matrix for the quality-gated FP16 target."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def parse_csv_ints(text: str, label: str) -> List[int]:
    values: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit(f"{label} contains a non-integer value: {item!r}") from exc
        if value <= 0:
            raise SystemExit(f"{label} values must be positive: {value}")
        values.append(value)
    if not values:
        raise SystemExit(f"{label} must contain at least one value")
    return values


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def normalize_int(value: Any, label: str) -> int:
    parsed = parse_float(value)
    if not math.isfinite(parsed):
        raise SystemExit(f"Selected target is missing {label}")
    rounded = int(round(parsed))
    if abs(parsed - rounded) > 1e-6 or rounded <= 0:
        raise SystemExit(f"Selected target has invalid {label}: {value!r}")
    return rounded


def selected_target(summary: Dict[str, Any], require_kernel: str, require_baseline: str) -> Dict[str, Any]:
    targets = summary.get("selected_targets")
    if not isinstance(targets, list) or not targets:
        raise SystemExit("quality_gate_summary.json has no selected_targets")
    matches = [
        row
        for row in targets
        if isinstance(row, dict)
        and str(row.get("test_kernel", "")) == require_kernel
        and str(row.get("baseline_kernel", "")) == require_baseline
    ]
    if not matches:
        raise SystemExit(
            "quality_gate_summary.json has no selected target matching "
            f"{require_kernel}/{require_baseline}"
        )
    target = matches[0]
    if not parse_bool(target.get("target_pass")):
        raise SystemExit("Selected target does not have target_pass=true")
    return target


def build_matrix(target: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    threads = normalize_int(target.get("threads"), "threads")
    blocks_per_sm = normalize_int(target.get("blocks_per_sm_requested"), "blocks_per_sm_requested")
    unrolls = parse_csv_ints(args.unrolls, "--unrolls")
    test_kernel = str(target.get("test_kernel", ""))
    baseline_kernel = str(target.get("baseline_kernel", ""))

    defaults = {
        "blocks": args.blocks,
        "blocks_per_sm": blocks_per_sm,
        "warmup": args.warmup,
        "repeats": args.test_repeats,
        "iters": args.iters,
        "suppress_output_store": True,
    }
    if not args.suppress_output_store:
        defaults["suppress_output_store"] = False

    conditions: List[Dict[str, Any]] = []
    for unroll in unrolls:
        conditions.append(
            {
                "name": f"work_slope_{test_kernel}_t{threads:03d}_b{blocks_per_sm:02d}_u{unroll:02d}",
                "args": {
                    "threads": threads,
                    "unroll": unroll,
                },
                "baseline": {
                    "kernel": baseline_kernel,
                    "repeats": args.baseline_repeats,
                },
                "test": {
                    "kernel": test_kernel,
                },
            }
        )

    return {
        "description": (
            "Auto-generated work-amount slope sweep for the quality-gated FP16 target. "
            "Run without matrix calibration so unroll changes remain distinct work points."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_quality_gate_summary": str(args.quality_gate_summary),
        "target_context": {
            "gpu": target.get("gpu", ""),
            "architecture_generation": target.get("architecture_generation", ""),
            "architecture_chip": target.get("architecture_chip", ""),
            "test_kernel": test_kernel,
            "baseline_kernel": baseline_kernel,
            "threads": threads,
            "blocks_per_sm_requested": blocks_per_sm,
            "selected_target_unroll": target.get("unroll", ""),
            "threads_per_sm": target.get("threads_per_sm", ""),
            "matmul_input_pj_per_bit_mean": target.get("matmul_input_pj_per_bit_mean", ""),
            "target_selection_note": target.get("target_selection_note", ""),
        },
        "work_slope_expectation": {
            "fit_scope": "valid_no_l2",
            "min_distinct_work_points": 3,
            "min_slope_r2": args.min_expected_r2,
            "positive_slope_required": True,
            "strict_audit_flag": "--require-work-slope",
        },
        "defaults": defaults,
        "conditions": conditions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a selected-target FP16 work-slope matrix")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Result directory containing quality_gate_summary.json",
    )
    parser.add_argument(
        "--quality-gate-summary",
        type=Path,
        default=None,
        help="Explicit quality_gate_summary.json path. Overrides --result-dir.",
    )
    parser.add_argument("--out-matrix", type=Path, required=True)
    parser.add_argument("--require-kernel", default="tensor_mma_f16acc")
    parser.add_argument("--require-baseline", default="tensor_baseline_mov")
    parser.add_argument("--unrolls", default="1,2,4,8,16")
    parser.add_argument("--iters", type=int, default=2_000_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--test-repeats", type=int, default=2)
    parser.add_argument("--baseline-repeats", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--min-expected-r2", type=float, default=0.80)
    parser.add_argument(
        "--no-suppress-output-store",
        dest="suppress_output_store",
        action="store_false",
        help="Diagnostic override. Strict work-slope evidence should keep output stores suppressed.",
    )
    parser.set_defaults(suppress_output_store=True)
    args = parser.parse_args()

    if args.quality_gate_summary is None:
        if args.result_dir is None:
            raise SystemExit("Provide --result-dir or --quality-gate-summary")
        args.quality_gate_summary = args.result_dir / "quality_gate_summary.json"
    if not args.quality_gate_summary.exists():
        raise SystemExit(f"Missing quality gate summary: {args.quality_gate_summary}")
    if args.iters <= 0 or args.warmup < 0 or args.test_repeats <= 0 or args.baseline_repeats <= 0:
        raise SystemExit("iters/repeats must be positive and warmup must be non-negative")

    summary = read_json(args.quality_gate_summary)
    target = selected_target(summary, args.require_kernel, args.require_baseline)
    matrix = build_matrix(target, args)

    args.out_matrix.parent.mkdir(parents=True, exist_ok=True)
    with args.out_matrix.open("w") as f:
        json.dump(matrix, f, indent=2)
        f.write("\n")
    print(f"Wrote: {args.out_matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
