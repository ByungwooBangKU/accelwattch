#!/usr/bin/env python3
"""Fail-fast Nsight Compute performance-counter permission probe."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROFILER_ERROR_PATTERNS = [
    "ERR_NVGPUCTRPERM",
    "No kernels were profiled",
    "Permission denied",
    "Failed to prepare kernel",
    "LaunchFailed",
]


DEFAULT_METRICS = ",".join(
    [
        "smsp__inst_executed_pipe_tensor_op_hmma.sum",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_bytes_read.sum",
        "lts__t_bytes_write.sum",
    ]
)


def profiler_errors(text: str) -> List[str]:
    lowered = text.lower()
    return [pattern for pattern in PROFILER_ERROR_PATTERNS if pattern.lower() in lowered]


def run_probe(args: argparse.Namespace, log_file: Path) -> Dict[str, Any]:
    command = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--kernel-name",
        "regex:.*tensor_mma_f16acc.*",
        "--section",
        "LaunchStats",
        "--section",
        "ComputeWorkloadAnalysis",
        "--metrics",
        args.metrics,
        "--print-summary",
        "per-kernel",
        "--log-file",
        str(log_file),
        args.binary,
        "--device",
        str(args.gpu),
        "--blocks",
        "0",
        "--blocks-per-sm",
        str(args.blocks_per_sm),
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--unroll",
        str(args.unroll),
        "--suppress-output-store",
        "--kernel",
        "tensor_mma_f16acc",
        "--threads",
        str(args.threads),
        "--iters",
        str(args.iters),
    ]
    cp = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_text = log_file.read_text(errors="replace") if log_file.exists() else ""
    combined = "\n".join([cp.stdout, cp.stderr, log_text])
    errors = profiler_errors(combined)
    permission_denied = "ERR_NVGPUCTRPERM" in errors
    no_kernels = "No kernels were profiled" in errors
    pass_ = cp.returncode == 0 and not permission_denied and not no_kernels
    status = "pass" if pass_ else ("permission_denied" if permission_denied else "failed")
    fail_reasons: List[str] = []
    if cp.returncode != 0:
        fail_reasons.append(f"ncu returned {cp.returncode}")
    if permission_denied:
        fail_reasons.append("Nsight Compute performance counters are blocked by ERR_NVGPUCTRPERM")
    if no_kernels:
        fail_reasons.append("Nsight Compute did not profile any kernels")
    for error in errors:
        if error not in {"ERR_NVGPUCTRPERM", "No kernels were profiled"}:
            fail_reasons.append(f"profiler error: {error}")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "permission_probe_pass": pass_,
        "permission_denied": permission_denied,
        "profiler_errors": errors,
        "fail_reasons": fail_reasons,
        "returncode": cp.returncode,
        "command": command,
        "log_file": str(log_file),
        "stdout": cp.stdout.strip(),
        "stderr": cp.stderr.strip(),
        "help": (
            "ERR_NVGPUCTRPERM is an NVIDIA driver/admin policy issue, not a missing Python package. "
            "Run the strict NCU validation in a job/session with GPU performance counter access, "
            "or use --diagnostic-no-ncu only for non-final diagnostic runs."
        ),
    }


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "status",
        "permission_probe_pass",
        "permission_denied",
        "returncode",
        "log_file",
        "profiler_errors",
        "fail_reasons",
        "command",
    ]
    row = dict(payload)
    row["profiler_errors"] = "; ".join(payload.get("profiler_errors", []))
    row["fail_reasons"] = "; ".join(payload.get("fail_reasons", []))
    row["command"] = " ".join(str(part) for part in payload.get("command", []))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in keys})


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Nsight Compute counter permissions with a short HMMA run")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--blocks-per-sm", type=int, default=1)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_file = args.outdir / "ncu_permission_probe.ncu.txt"
    payload = run_probe(args, log_file)

    json_path = args.outdir / "ncu_permission_probe.json"
    csv_path = args.outdir / "ncu_permission_probe.csv"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    write_csv(csv_path, payload)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {log_file}")
    if not payload["permission_probe_pass"]:
        for reason in payload["fail_reasons"]:
            print("ncu permission probe failed: " + reason)
        return 0 if args.no_fail else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
