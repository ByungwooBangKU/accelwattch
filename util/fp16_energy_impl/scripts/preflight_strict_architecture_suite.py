#!/usr/bin/env python3
"""Preflight checks for the strict multi-architecture FP16 suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_BY_ARCH = {
    "80": {"chip": "ga100", "generation": "ampere", "name_tokens": ["A100"]},
    "86": {"chip": "ga102", "generation": "ampere", "name_tokens": ["3090"]},
    "90": {"chip": "gh100", "generation": "hopper", "name_tokens": ["H100"]},
}


def run_command(cmd: List[str], timeout_s: float = 10.0) -> Dict[str, Any]:
    try:
        cp = subprocess.run(
            cmd,
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
    except Exception as exc:  # noqa: BLE001 - preflight reports missing/broken tools as data
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": "", "error": str(exc)}


def parse_spec(text: str) -> Dict[str, str]:
    parts = text.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(f"invalid spec {text!r}; expected LABEL:GPU:CUDA_ARCH[:NVIDIA_SMI_ID]")
    label, gpu, cuda_arch = parts[:3]
    nvidia_smi_id = parts[3] if len(parts) == 4 else ""
    for name, value in [("label", label), ("gpu", gpu), ("cuda_arch", cuda_arch)]:
        if not value or "," in value or "/" in value:
            raise ValueError(f"invalid {name} in spec {text!r}: {value!r}")
    return {
        "label": label,
        "gpu": gpu,
        "cuda_arch": "".join(ch for ch in cuda_arch if ch.isdigit()) or cuda_arch,
        "nvidia_smi_id": nvidia_smi_id or gpu,
        "raw_spec": text,
    }


def parse_gpu_query(stdout: str) -> Dict[str, str]:
    line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    fields = [item.strip() for item in line.split(",")]
    names = [
        "index",
        "uuid",
        "pci.bus_id",
        "name",
        "driver_version",
        "power.limit",
        "power.draw",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
    ]
    return dict(zip(names, fields)) if len(fields) >= len(names) else {"raw": line}


def parse_compute_apps(stdout: str) -> List[Dict[str, str]]:
    rows = []
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        while len(fields) < 4:
            fields.append("")
        rows.append(
            {
                "pid": fields[0],
                "process_name": fields[1],
                "used_memory": fields[2],
                "gpu_uuid": fields[3],
                "raw": line,
            }
        )
    return rows


def command_available(name: str) -> bool:
    return bool(shutil.which(name))


def tool_result(name: str, cmd: List[str], required: bool, dry_run: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": name,
        "command": cmd,
        "required": required,
        "path": shutil.which(cmd[0]) or "",
        "available": command_available(cmd[0]),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "pass": True,
        "fail_reasons": [],
        "warnings": [],
    }
    if dry_run:
        result["status"] = "dry_run_not_executed"
        result["pass"] = True
        if required and not result["available"]:
            result["warnings"].append(f"{name} not found on PATH during dry-run")
        return result
    if not result["available"]:
        result["pass"] = not required
        if required:
            result["fail_reasons"].append(f"{name} not found on PATH")
        return result
    cp = run_command(cmd)
    result.update(
        {
            "returncode": cp.get("returncode"),
            "stdout": cp.get("stdout", ""),
            "stderr": cp.get("stderr", ""),
        }
    )
    if cp.get("returncode") != 0:
        result["pass"] = not required
        if required:
            result["fail_reasons"].append(f"{name} command failed")
    return result


def check_spec(
    spec: Dict[str, str],
    nvidia_smi_bin: str,
    apps: List[Dict[str, str]],
    allow_compute_apps: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    failed: List[str] = []
    warnings: List[str] = []
    arch = spec["cuda_arch"]
    expected = EXPECTED_BY_ARCH.get(arch, {})
    gpu_query: Dict[str, str] = {}
    matched_apps: List[Dict[str, str]] = []

    if not expected:
        failed.append(f"unsupported CUDA architecture for strict suite: {arch}")

    if dry_run:
        warnings.append("dry-run: GPU metadata and compute-app checks were not executed")
    else:
        cmd = [
            nvidia_smi_bin,
            f"--id={spec['nvidia_smi_id']}",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version,power.limit,power.draw,pstate,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        cp = run_command(cmd)
        if cp.get("returncode") != 0:
            failed.append(cp.get("stderr") or cp.get("stdout") or "nvidia-smi GPU metadata query failed")
        else:
            gpu_query = parse_gpu_query(str(cp.get("stdout", "")))

    gpu_name = gpu_query.get("name", "")
    gpu_uuid = gpu_query.get("uuid", "")
    expected_tokens = expected.get("name_tokens", []) if expected else []
    if gpu_name and expected_tokens and not any(token in gpu_name for token in expected_tokens):
        failed.append(
            f"nvidia-smi target {spec['nvidia_smi_id']} is {gpu_name!r}, "
            f"which does not match expected tokens {expected_tokens} for CUDA arch {arch}"
        )

    if gpu_uuid:
        matched_apps = [row for row in apps if row.get("gpu_uuid") == gpu_uuid]
    elif apps and not dry_run:
        warnings.append("could not map compute-app rows to this GPU because UUID is missing")

    if matched_apps and not allow_compute_apps:
        failed.append("target GPU has active compute processes: " + "; ".join(row["raw"] for row in matched_apps))

    return {
        **spec,
        "expected_chip": expected.get("chip", ""),
        "expected_generation": expected.get("generation", ""),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_index": gpu_query.get("index", ""),
        "gpu_pci_bus_id": gpu_query.get("pci.bus_id", ""),
        "driver_version": gpu_query.get("driver_version", ""),
        "power_limit_w": gpu_query.get("power.limit", ""),
        "active_compute_app_count": len(matched_apps),
        "active_compute_apps": matched_apps,
        "preflight_pass": not failed,
        "fail_reasons": failed,
        "warnings": warnings,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "label",
        "gpu",
        "cuda_arch",
        "nvidia_smi_id",
        "expected_chip",
        "expected_generation",
        "gpu_name",
        "gpu_uuid",
        "gpu_index",
        "gpu_pci_bus_id",
        "driver_version",
        "power_limit_w",
        "active_compute_app_count",
        "preflight_pass",
        "fail_reasons",
        "warnings",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["fail_reasons"] = "; ".join(row.get("fail_reasons", []))
            out["warnings"] = "; ".join(row.get("warnings", []))
            writer.writerow({key: out.get(key, "") for key in keys})


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight strict FP16 architecture suite targets")
    parser.add_argument("--spec", action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--cmake-bin", default="cmake")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--nvidia-smi-bin", default="nvidia-smi")
    parser.add_argument("--require-ncu", action="store_true")
    parser.add_argument("--allow-compute-apps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    failed: List[str] = []
    try:
        specs = [parse_spec(text) for text in args.spec]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tools = [
        tool_result("python", [args.python_bin, "--version"], True, args.dry_run),
        tool_result("cmake", [args.cmake_bin, "--version"], True, args.dry_run),
        tool_result("nvidia-smi", [args.nvidia_smi_bin, "--version"], True, args.dry_run),
        tool_result("nvcc", ["nvcc", "--version"], True, args.dry_run),
        tool_result("ncu", ["ncu", "--version"], args.require_ncu, args.dry_run),
    ]
    for tool in tools:
        failed.extend(tool.get("fail_reasons", []))

    apps: List[Dict[str, str]] = []
    apps_result: Dict[str, Any] = {"skipped": bool(args.dry_run)}
    if not args.dry_run and command_available(args.nvidia_smi_bin):
        apps_result = run_command(
            [
                args.nvidia_smi_bin,
                "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ]
        )
        if apps_result.get("returncode") == 0:
            apps = parse_compute_apps(str(apps_result.get("stdout", "")))

    rows = [
        check_spec(
            spec,
            args.nvidia_smi_bin,
            apps,
            args.allow_compute_apps,
            args.dry_run,
        )
        for spec in specs
    ]
    for row in rows:
        failed.extend(row.get("fail_reasons", []))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_schema": "fp16-strict-architecture-suite-preflight-v1",
        "overall_pass": not failed,
        "dry_run": bool(args.dry_run),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", ""),
            "PATH": os.environ.get("PATH", ""),
        },
        "tool_results": tools,
        "compute_apps_query": apps_result,
        "compute_apps": apps,
        "rows": rows,
        "fail_reasons": failed,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    write_csv(args.out_csv, rows)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_csv}")
    if failed:
        for reason in failed:
            print("preflight failed: " + reason, file=sys.stderr)
        return 0 if args.no_fail else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
