#!/usr/bin/env python3
"""Write a provenance manifest for a strict FP16 pipeline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def run_command(cmd: List[str], cwd: Path | None = None, timeout_s: float = 8.0) -> Dict[str, Any]:
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        return {
            "command": cmd,
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001 - provenance should not crash on missing optional tools
        return {"command": cmd, "error": str(exc)}


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_info(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write strict FP16 pipeline provenance manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--status", required=True, choices=["started", "completed", "failed"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--pipeline-preflight-json", type=Path, required=True)
    parser.add_argument("--pipeline-preflight-csv", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--ncu-dir", type=Path, required=True)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--nvidia-smi-id", required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--sample-ms", required=True)
    parser.add_argument("--threads", required=True)
    parser.add_argument("--ncu-blocks-per-sm-csv", required=True)
    parser.add_argument("--skip-build", required=True)
    parser.add_argument("--skip-preflight", required=True)
    parser.add_argument("--allow-compute-apps", required=True)
    parser.add_argument("--diagnostic-no-ncu", required=True)
    parser.add_argument("--calibrate-matrix", required=True)
    parser.add_argument("--target-test-s", required=True)
    parser.add_argument("--target-baseline-s", required=True)
    parser.add_argument("--max-calibrated-repeats", required=True)
    parser.add_argument("--require-kernel", required=True)
    parser.add_argument("--require-baseline", required=True)
    parser.add_argument("--run-work-slope", required=True)
    parser.add_argument("--work-slope-matrix", type=Path, required=True)
    parser.add_argument("--work-slope-dir", type=Path, required=True)
    parser.add_argument("--work-slope-unrolls", required=True)
    parser.add_argument("--work-slope-iters", required=True)
    parser.add_argument("--work-slope-warmup", required=True)
    parser.add_argument("--work-slope-test-repeats", required=True)
    parser.add_argument("--work-slope-baseline-repeats", required=True)
    parser.add_argument("--work-slope-repeat", required=True)
    parser.add_argument("--invocation", default="")
    parser.add_argument("--cmake-bin", default="cmake")
    parser.add_argument("--nvcc-bin", default="nvcc")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--nvidia-smi-bin", default="nvidia-smi")
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir.resolve()
    payload = {
        "manifest_schema": "fp16-strict-pipeline-manifest-v1",
        "status": args.status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_unix_ns": time.time_ns(),
        "invocation": args.invocation,
        "parameters": {
            "gpu": args.gpu,
            "nvidia_smi_id": args.nvidia_smi_id,
            "cuda_arch": args.cuda_arch,
            "repeat": args.repeat,
            "sample_ms": args.sample_ms,
            "threads": args.threads,
            "ncu_blocks_per_sm_csv": args.ncu_blocks_per_sm_csv,
            "build_dir": args.build_dir,
            "skip_build": args.skip_build,
            "skip_preflight": args.skip_preflight,
            "allow_compute_apps": args.allow_compute_apps,
            "diagnostic_no_ncu": args.diagnostic_no_ncu,
            "calibrate_matrix": args.calibrate_matrix,
            "target_test_s": args.target_test_s,
            "target_baseline_s": args.target_baseline_s,
            "max_calibrated_repeats": args.max_calibrated_repeats,
            "require_kernel": args.require_kernel,
            "require_baseline": args.require_baseline,
            "run_work_slope": args.run_work_slope,
            "work_slope_unrolls": args.work_slope_unrolls,
            "work_slope_iters": args.work_slope_iters,
            "work_slope_warmup": args.work_slope_warmup,
            "work_slope_test_repeats": args.work_slope_test_repeats,
            "work_slope_baseline_repeats": args.work_slope_baseline_repeats,
            "work_slope_repeat": args.work_slope_repeat,
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "CMAKE_CUDA_FLAGS",
                "MPLCONFIGDIR",
                "NCU_METRICS",
                "NCU_BLOCKS_PER_SM_CSV",
                "NCU_BIN",
                "NVCC_BIN",
                "PATH",
                "LD_LIBRARY_PATH",
            ]
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"], cwd=root),
            "branch": run_command(["git", "branch", "--show-current"], cwd=root),
            "status_short": run_command(["git", "status", "--short"], cwd=root),
        },
        "tool_versions": {
            "python": run_command([args.python_bin, "--version"]),
            "cmake": run_command([args.cmake_bin, "--version"]),
            "nvidia_smi": run_command([args.nvidia_smi_bin, "--version"]),
            "nvcc": run_command([args.nvcc_bin, "--version"]),
            "ncu": run_command([args.ncu_bin, "--version"]),
        },
        "artifacts": {
            "root": str(root),
            "outdir": str(outdir),
            "binary": path_info(args.binary.resolve()),
            "build_log": path_info(args.build_log.resolve()),
            "pipeline_preflight_json": path_info(args.pipeline_preflight_json.resolve()),
            "pipeline_preflight_csv": path_info(args.pipeline_preflight_csv.resolve()),
            "base_matrix": path_info(args.base_matrix.resolve()),
            "matrix": path_info(args.matrix.resolve()),
            "quality_gates": path_info((outdir / "quality_gates.csv").resolve()),
            "quality_gate_summary": path_info((outdir / "quality_gate_summary.json").resolve()),
            "ncu_permission_probe_json": path_info(
                (outdir / "ncu_permission_probe" / "ncu_permission_probe.json").resolve()
            ),
            "ncu_permission_probe_csv": path_info(
                (outdir / "ncu_permission_probe" / "ncu_permission_probe.csv").resolve()
            ),
            "ncu_permission_probe_log": path_info(
                (outdir / "ncu_permission_probe" / "ncu_permission_probe.ncu.txt").resolve()
            ),
            "ncu_validation_summary": path_info((args.ncu_dir / "ncu_validation_summary.csv").resolve()),
            "resource_audit": path_info((args.resource_dir / "thread_resource_occupancy.csv").resolve()),
            "work_slope_matrix": path_info(args.work_slope_matrix.resolve()),
            "work_slope_summary": path_info((outdir / "work_slope_summary.csv").resolve()),
            "work_slope_nested_summary": path_info((args.work_slope_dir / "work_slope_summary.csv").resolve()),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
