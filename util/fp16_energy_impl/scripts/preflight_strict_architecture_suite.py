#!/usr/bin/env python3
"""Preflight checks for the strict multi-architecture FP16 suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def parse_nvcc_release(text: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"release\s+(\d+)\.(\d+)", text)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def parse_nvidia_smi_cuda_version(text: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"CUDA Version\s*:?\s*(\d+)\.(\d+)", text)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def version_text(version: Optional[Tuple[int, int]]) -> str:
    if version is None:
        return ""
    return f"{version[0]}.{version[1]}"


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


def check_cuda_toolchain_compatibility(tools: List[Dict[str, Any]], dry_run: bool) -> Dict[str, Any]:
    """Check that the selected nvcc runtime is not newer than the installed driver.

    nvidia-smi reports the maximum CUDA runtime version supported by the driver.
    A newer nvcc can compile successfully but produce a binary that fails at
    cudaSetDevice with cudaErrorInsufficientDriver.
    """
    result: Dict[str, Any] = {
        "checked": False,
        "pass": True,
        "nvcc_release": "",
        "driver_cuda_version": "",
        "fail_reasons": [],
        "warnings": [],
    }
    if dry_run:
        result["warnings"].append("dry-run: CUDA toolchain/driver compatibility was not checked")
        return result

    by_name = {str(tool.get("name", "")): tool for tool in tools}
    nvcc = by_name.get("nvcc", {})
    nvidia_smi = by_name.get("nvidia-smi", {})
    if nvcc.get("returncode") != 0 or nvidia_smi.get("returncode") != 0:
        result["warnings"].append("CUDA toolchain/driver compatibility skipped because nvcc or nvidia-smi failed")
        return result

    nvcc_text = "\n".join([str(nvcc.get("stdout", "")), str(nvcc.get("stderr", ""))])
    smi_text = "\n".join([str(nvidia_smi.get("stdout", "")), str(nvidia_smi.get("stderr", ""))])
    nvcc_version = parse_nvcc_release(nvcc_text)
    driver_cuda = parse_nvidia_smi_cuda_version(smi_text)
    result["checked"] = True
    result["nvcc_release"] = version_text(nvcc_version)
    result["driver_cuda_version"] = version_text(driver_cuda)

    if nvcc_version is None:
        result["warnings"].append("could not parse nvcc release version")
    if driver_cuda is None:
        result["warnings"].append("could not parse driver CUDA Version from nvidia-smi --version")
    if nvcc_version is not None and driver_cuda is not None and nvcc_version > driver_cuda:
        result["pass"] = False
        result["fail_reasons"].append(
            f"nvcc release {version_text(nvcc_version)} is newer than driver CUDA Version "
            f"{version_text(driver_cuda)}; use an older NVCC_BIN/toolkit or update the driver"
        )
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
            missing_fields = [
                field
                for field in ("index", "uuid", "pci.bus_id", "name", "driver_version", "power.limit")
                if not str(gpu_query.get(field, "")).strip()
            ]
            if missing_fields:
                failed.append(
                    "nvidia-smi GPU metadata query returned incomplete output; missing "
                    + ",".join(missing_fields)
                )

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
        "required_tools_pass",
        "overall_preflight_pass",
        "publishable_preflight_pass",
        "required_tool_fail_reasons",
        "overall_fail_reasons",
        "fail_reasons",
        "warnings",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["fail_reasons"] = "; ".join(row.get("fail_reasons", []))
            out["required_tool_fail_reasons"] = "; ".join(row.get("required_tool_fail_reasons", []))
            out["overall_fail_reasons"] = "; ".join(row.get("overall_fail_reasons", []))
            out["warnings"] = "; ".join(row.get("warnings", []))
            writer.writerow({key: out.get(key, "") for key in keys})


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight strict FP16 architecture suite targets")
    parser.add_argument("--spec", action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--cmake-bin", default="cmake")
    parser.add_argument("--nvcc-bin", default="nvcc")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--nvidia-smi-bin", default="nvidia-smi")
    parser.add_argument("--require-ncu", action="store_true")
    parser.add_argument("--allow-compute-apps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    failed: List[str] = []
    tool_failures: List[str] = []
    try:
        specs = [parse_spec(text) for text in args.spec]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tools = [
        tool_result("python", [args.python_bin, "--version"], True, args.dry_run),
        tool_result("cmake", [args.cmake_bin, "--version"], True, args.dry_run),
        tool_result("nvidia-smi", [args.nvidia_smi_bin, "--version"], True, args.dry_run),
        tool_result("nvcc", [args.nvcc_bin, "--version"], True, args.dry_run),
        tool_result("ncu", [args.ncu_bin, "--version"], args.require_ncu, args.dry_run),
    ]
    compatibility = check_cuda_toolchain_compatibility(tools, args.dry_run)
    for tool in tools:
        tool_failures.extend(tool.get("fail_reasons", []))
    tool_failures.extend(compatibility.get("fail_reasons", []))
    failed.extend(tool_failures)

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
    overall_pass = not failed
    required_tools_pass = not tool_failures
    for row in rows:
        row_failures = list(row.get("fail_reasons", []))
        row["required_tools_pass"] = required_tools_pass
        row["required_tool_fail_reasons"] = tool_failures
        row["overall_preflight_pass"] = overall_pass
        row["publishable_preflight_pass"] = overall_pass and not args.dry_run
        row["overall_fail_reasons"] = failed
        row["target_fail_reasons"] = row_failures

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_schema": "fp16-strict-architecture-suite-preflight-v1",
        "overall_pass": overall_pass,
        "required_tools_pass": required_tools_pass,
        "required_tool_fail_reasons": tool_failures,
        "dry_run": bool(args.dry_run),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", ""),
            "CMAKE_CUDA_FLAGS": os.environ.get("CMAKE_CUDA_FLAGS", ""),
            "PATH": os.environ.get("PATH", ""),
        },
        "tool_results": tools,
        "cuda_toolchain_compatibility": compatibility,
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
