#!/usr/bin/env python3
"""Verify that a benchmark JSON matches the intended strict GPU architecture."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


STRICT_CHIPS_BY_CUDA_ARCH = {
    "80": ["ga100"],
    "86": ["ga102"],
    "90": ["gh100"],
}

COMPATIBLE_CHIPS_BY_CUDA_ARCH = {
    "80": ["ga100", "ampere_sm80"],
    "86": ["ga102", "ampere_sm86"],
    "90": ["gh100", "hopper_sm90"],
}

EXPECTED_GENERATION_BY_CUDA_ARCH = {
    "80": "ampere",
    "86": "ampere",
    "90": "hopper",
}


def normalize_arch(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def compute_capability_to_arch(value: Any) -> str:
    text = str(value or "").strip()
    match = re.match(r"(\d+)\.(\d+)", text)
    if match:
        return match.group(1) + match.group(2)
    return normalize_arch(text)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def verify(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    requested_arch = normalize_arch(args.cuda_arch)
    recommended_arch = normalize_arch(payload.get("recommended_cuda_arch"))
    cc_arch = compute_capability_to_arch(payload.get("compute_capability"))
    chip = str(payload.get("architecture_chip", "") or "unknown")
    generation = str(payload.get("architecture_generation", "") or "unknown")
    gpu = str(payload.get("device_name", "") or "")
    benchmark_uses_wgmma = parse_bool(payload.get("benchmark_uses_wgmma"))

    failed: List[str] = []
    warnings: List[str] = []

    if not requested_arch:
        failed.append("requested CUDA architecture is empty")
    if recommended_arch and requested_arch and recommended_arch != requested_arch:
        failed.append(f"requested CUDA arch {requested_arch} does not match GPU recommended arch {recommended_arch}")
    if cc_arch and requested_arch and cc_arch != requested_arch:
        failed.append(f"requested CUDA arch {requested_arch} does not match compute capability {cc_arch}")

    expected_generation = args.expect_generation or EXPECTED_GENERATION_BY_CUDA_ARCH.get(requested_arch, "")
    if expected_generation and generation != expected_generation:
        failed.append(f"architecture generation {generation} does not match expected {expected_generation}")

    if args.expect_chip:
        expected_chips = [item.strip() for item in args.expect_chip.split(",") if item.strip()]
    elif args.strict_chip:
        expected_chips = STRICT_CHIPS_BY_CUDA_ARCH.get(requested_arch, [])
    else:
        expected_chips = COMPATIBLE_CHIPS_BY_CUDA_ARCH.get(requested_arch, [])

    if expected_chips and chip not in expected_chips:
        failed.append(f"architecture chip {chip} is not one of expected chips: {','.join(expected_chips)}")
    elif not expected_chips:
        warnings.append(f"no chip allowlist for requested CUDA arch {requested_arch}")

    if args.require_common_hmma and benchmark_uses_wgmma:
        failed.append("benchmark_uses_wgmma=true; strict cross-GPU comparison requires common HMMA path")

    return {
        "architecture_preflight_pass": not failed,
        "requested_cuda_arch": requested_arch,
        "recommended_cuda_arch": recommended_arch,
        "compute_capability_arch": cc_arch,
        "device_name": gpu,
        "architecture_generation": generation,
        "architecture_chip": chip,
        "expected_generation": expected_generation,
        "expected_chips": expected_chips,
        "benchmark_uses_wgmma": benchmark_uses_wgmma,
        "fail_reasons": failed,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict FP16 benchmark architecture metadata")
    parser.add_argument("--input", type=Path, required=True, help="benchmark JSON, usually runtime_preflight.json")
    parser.add_argument("--cuda-arch", required=True, help="Requested CMake CUDA architecture, e.g. 80, 86, 90")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expect-chip", default="", help="Comma-separated allowed architecture_chip values")
    parser.add_argument("--expect-generation", default="")
    parser.add_argument(
        "--strict-chip",
        action="store_true",
        help="Require A100=ga100, RTX3090=ga102, H100=gh100 instead of generic SM-compatible chips",
    )
    parser.add_argument(
        "--require-common-hmma",
        action="store_true",
        help="Fail if benchmark metadata says it used WGMMA",
    )
    args = parser.parse_args()

    result = verify(read_json(args.input), args)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not result["architecture_preflight_pass"]:
        for reason in result["fail_reasons"]:
            print("architecture preflight failed: " + reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
