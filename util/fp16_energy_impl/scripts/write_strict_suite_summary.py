#!/usr/bin/env python3
"""Write a suite-level summary for strict FP16 multi-architecture runs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_json(path: Path | None) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path | None) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_spec(text: str) -> Dict[str, str]:
    parts = text.split(":")
    return {
        "raw": text,
        "label": parts[0] if len(parts) > 0 else "",
        "gpu": parts[1] if len(parts) > 1 else "",
        "cuda_arch": parts[2] if len(parts) > 2 else "",
        "nvidia_smi_id": parts[3] if len(parts) > 3 else "",
    }


def artifact_info(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write strict FP16 architecture suite summary")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--run-status-csv", type=Path, required=True)
    parser.add_argument("--postprocess-dir", type=Path, default=None)
    parser.add_argument("--postprocess-exit-code", default="")
    parser.add_argument("--postprocess-skipped", action="store_true")
    parser.add_argument("--suite-failed", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--require-architectures", default="")
    parser.add_argument("--run-work-slope", action="store_true")
    parser.add_argument("--require-work-slope", action="store_true")
    args = parser.parse_args()

    preflight = read_json(args.preflight_json)
    runs = read_csv(args.run_status_csv)
    audit_json = read_json(args.postprocess_dir / "strict_fp16_audit" / "strict_result_audit.json" if args.postprocess_dir else None)
    report_requirements = read_csv(
        args.postprocess_dir / "strict_fp16_report" / "fp16_strict_report_requirements.csv"
        if args.postprocess_dir
        else None
    )

    run_status_counts: Dict[str, int] = {}
    for row in runs:
        key = str(row.get("status", "") or "unknown")
        run_status_counts[key] = run_status_counts.get(key, 0) + 1

    preflight_pass = (
        args.skip_preflight
        or (
            bool(preflight)
            and str(preflight.get("preflight_schema", "")) == "fp16-strict-architecture-suite-preflight-v1"
            and parse_bool(preflight.get("overall_pass"))
            and not parse_bool(preflight.get("dry_run"))
        )
    )
    audit_pass = parse_bool(audit_json.get("overall_pass"))
    report_requirements_pass = bool(report_requirements) and all(
        str(row.get("status", "")) == "pass" for row in report_requirements
    )
    postprocess_pass = args.no_postprocess or (
        not args.postprocess_skipped
        and str(args.postprocess_exit_code) == "0"
        and audit_pass
        and report_requirements_pass
    )
    all_runs_completed = bool(runs) and all(str(row.get("status", "")) == "completed" for row in runs)
    diagnostic_only = args.dry_run or args.skip_preflight or args.no_postprocess or not preflight_pass
    publishable_pass = (
        not diagnostic_only
        and not parse_bool(args.suite_failed)
        and all_runs_completed
        and postprocess_pass
    )

    payload = {
        "summary_schema": "fp16-strict-architecture-suite-summary-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(args.outdir),
        "required_architectures": [item.strip() for item in args.require_architectures.split(",") if item.strip()],
        "specs": [parse_spec(spec) for spec in args.spec],
        "energy_policy": {
            "final_energy_source": "nvml_total_energy_counter",
            "final_measurement_grade": "strict_nvml_counter",
            "trace_role": "fallback_or_counter_trace_sanity_check",
            "logical_matmul_shape": "m16n16k16",
            "logical_input_bits_per_mma": 8192,
            "work_slope_required": bool(args.require_work_slope),
            "note": (
                "Final FP16 pJ/bit claims require benchmark timed-loop "
                "nvmlDeviceGetTotalEnergyConsumption() deltas; nvidia-smi power traces are "
                "diagnostic unless explicitly downgraded to fallback-grade results. When "
                "work_slope_required=true, each selected target must also have matching "
                "positive work-energy slope evidence."
            ),
        },
        "flags": {
            "dry_run": bool(args.dry_run),
            "skip_preflight": bool(args.skip_preflight),
            "no_postprocess": bool(args.no_postprocess),
            "postprocess_skipped": bool(args.postprocess_skipped),
            "suite_failed": parse_bool(args.suite_failed),
            "run_work_slope": bool(args.run_work_slope),
            "require_work_slope": bool(args.require_work_slope),
        },
        "artifacts": {
            "preflight_json": artifact_info(args.preflight_json),
            "run_status_csv": artifact_info(args.run_status_csv),
            "postprocess_dir": artifact_info(args.postprocess_dir),
        },
        "checks": {
            "preflight_pass": preflight_pass,
            "all_runs_completed": all_runs_completed,
            "audit_pass": audit_pass,
            "report_requirements_pass": report_requirements_pass,
            "postprocess_pass": postprocess_pass,
            "diagnostic_only": diagnostic_only,
            "publishable_pass": publishable_pass,
        },
        "counts": {
            "specs": len(args.spec),
            "runs": len(runs),
            "run_status_counts": run_status_counts,
            "report_requirements": len(report_requirements),
            "report_requirements_failed": sum(
                1 for row in report_requirements if str(row.get("status", "")) != "pass"
            ),
        },
        "postprocess_exit_code": args.postprocess_exit_code,
        "preflight_overall_pass": preflight.get("overall_pass", ""),
        "preflight_dry_run": preflight.get("dry_run", ""),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
