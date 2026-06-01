#!/usr/bin/env python3
"""Summarize ptxas register/spill evidence and resource-limited occupancy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from architecture_models import ARCH_MODELS

ARCH_LIMITS: Dict[str, Dict[str, Any]] = {
    str(model["recommended_cuda_arch"]): model
    for model in ARCH_MODELS.values()
}

KERNEL_RE = re.compile(
    r"(fp16_half2|baseline_nop|baseline_regmove|tensor_mma_f16acc|tensor_mma_f32acc|"
    r"tensor_baseline_u32|tensor_baseline_f32|memory_policy)_kernelILi(\d+)E(?:Li(\d+)E)?"
)
ENTRY_RE = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
USED_RE = re.compile(r"Used (\d+) registers")


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


def parse_int(value: Any, default: int = 0) -> int:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return int(parsed)
    return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


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


def normalize_kernel(kernel: str, policy: str = "") -> str:
    if kernel == "memory_policy":
        return {
            "0": "memory_default",
            "1": "memory_cg",
            "2": "memory_cs",
        }.get(str(policy), "memory_policy")
    return kernel


def parse_kernel_entry(symbol: str) -> Tuple[str, str, str]:
    match = KERNEL_RE.search(symbol)
    if not match:
        return ("", "", "")
    kernel = normalize_kernel(match.group(1), match.group(3) or "")
    return (kernel, match.group(2), match.group(3) or "")


def parse_ptxas_log(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        entry = ENTRY_RE.search(line)
        if entry:
            kernel, unroll, policy = parse_kernel_entry(entry.group(1))
            current = {
                "source_log": str(path),
                "symbol": entry.group(1),
                "sm": entry.group(2),
                "kernel": kernel,
                "unroll": unroll,
                "memory_policy_id": policy,
            }
            continue
        if not current:
            continue
        spill = SPILL_RE.search(line)
        if spill:
            current["stack_frame_bytes"] = int(spill.group(1))
            current["spill_store_bytes"] = int(spill.group(2))
            current["spill_load_bytes"] = int(spill.group(3))
            continue
        used = USED_RE.search(line)
        if used:
            current["registers_per_thread"] = int(used.group(1))
            current["has_spills"] = bool(
                int(current.get("spill_store_bytes", 0)) > 0
                or int(current.get("spill_load_bytes", 0)) > 0
                or int(current.get("stack_frame_bytes", 0)) > 0
            )
            if current.get("kernel"):
                rows.append(dict(current))
            current = {}
    return dedupe_resource_rows(rows)


def dedupe_resource_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sm", "")), str(row.get("kernel", "")), str(row.get("unroll", "")))
        best[key] = row
    return sorted(best.values(), key=lambda r: (str(r.get("sm", "")), str(r.get("kernel", "")), parse_int(r.get("unroll"))))


def infer_arch_from_rows(thread_rows: List[Dict[str, Any]], resource_rows: List[Dict[str, Any]]) -> str:
    for row in thread_rows:
        arch = str(row.get("recommended_cuda_arch", "") or "")
        if arch:
            return arch
    for row in resource_rows:
        sm = str(row.get("sm", "") or "")
        if sm:
            return sm
    return ""


def find_resource(
    resource_rows: List[Dict[str, Any]],
    kernel: str,
    unroll: str,
    cuda_arch: str,
) -> Dict[str, Any]:
    candidates = [
        row for row in resource_rows
        if str(row.get("kernel", "")) == kernel
        and (not cuda_arch or str(row.get("sm", "")) == cuda_arch)
        and (not unroll or str(row.get("unroll", "")) == unroll)
    ]
    if candidates:
        return candidates[0]
    candidates = [
        row for row in resource_rows
        if str(row.get("kernel", "")) == kernel and (not cuda_arch or str(row.get("sm", "")) == cuda_arch)
    ]
    return candidates[0] if candidates else {}


def limiting_factors(values: Dict[str, int], selected: int) -> str:
    factors = [name for name, value in values.items() if value == selected]
    return ",".join(factors)


def occupancy_row(
    thread_row: Dict[str, Any],
    resource: Dict[str, Any],
    role: str,
    kernel: str,
    cuda_arch: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    limits = dict(ARCH_LIMITS.get(cuda_arch, {}))
    registers_per_sm = int(args.registers_per_sm or limits.get("registers_per_sm", 65536))
    max_threads_per_sm = int(args.max_threads_per_sm or limits.get("max_threads_per_sm", 2048))
    max_blocks_per_sm = int(args.max_blocks_per_sm or limits.get("max_blocks_per_sm", 32))
    max_warps_per_sm = int(args.max_warps_per_sm or limits.get("max_warps_per_sm", max_threads_per_sm // 32))
    threads = parse_int(thread_row.get("threads"), 0)
    threads_per_sm = parse_float(thread_row.get("threads_per_sm"))
    blocks_per_sm_requested = parse_float(thread_row.get("blocks_per_sm_requested"))
    if not math.isfinite(blocks_per_sm_requested) and threads > 0 and math.isfinite(threads_per_sm):
        blocks_per_sm_requested = threads_per_sm / threads
    requested_blocks = int(round(blocks_per_sm_requested)) if math.isfinite(blocks_per_sm_requested) else args.blocks_per_sm
    regs_per_thread = parse_int(resource.get("registers_per_thread"), 0)
    if threads <= 0:
        by_threads = 0
    else:
        by_threads = max_threads_per_sm // threads
    if threads <= 0 or regs_per_thread <= 0:
        by_registers = 0
    else:
        by_registers = registers_per_sm // (regs_per_thread * threads)
    cap_values = {
        "requested_blocks_per_sm": max(requested_blocks, 0),
        "max_blocks_per_sm": max_blocks_per_sm,
        "thread_limit": by_threads,
    }
    if by_registers > 0:
        cap_values["register_limit"] = by_registers
    active_blocks = min(value for value in cap_values.values() if value >= 0) if cap_values else 0
    active_threads = active_blocks * threads
    active_warps = active_threads / 32.0 if threads > 0 else math.nan
    occupancy_pct = active_threads / max_threads_per_sm * 100.0 if max_threads_per_sm > 0 else math.nan
    warp_occupancy_pct = active_warps / max_warps_per_sm * 100.0 if max_warps_per_sm > 0 else math.nan
    spill_load = parse_int(resource.get("spill_load_bytes"), 0)
    spill_store = parse_int(resource.get("spill_store_bytes"), 0)
    stack = parse_int(resource.get("stack_frame_bytes"), 0)
    return {
        "role": role,
        "kernel": kernel,
        "sm": cuda_arch,
        "architecture_chip": limits.get("architecture_chip", thread_row.get("architecture_chip", "")),
        "threads": threads,
        "threads_per_sm": thread_row.get("threads_per_sm", ""),
        "blocks_per_sm_requested": requested_blocks,
        "unroll": thread_row.get("unroll", ""),
        "registers_per_thread": regs_per_thread if regs_per_thread else "",
        "stack_frame_bytes": stack,
        "spill_store_bytes": spill_store,
        "spill_load_bytes": spill_load,
        "has_spills": bool(stack > 0 or spill_store > 0 or spill_load > 0),
        "registers_per_sm_model": registers_per_sm,
        "max_threads_per_sm_model": max_threads_per_sm,
        "max_blocks_per_sm_model": max_blocks_per_sm,
        "active_blocks_per_sm_model": active_blocks,
        "active_threads_per_sm_model": active_threads,
        "active_warps_per_sm_model": active_warps,
        "thread_occupancy_pct_model": occupancy_pct,
        "warp_occupancy_pct_model": warp_occupancy_pct,
        "limiting_factors_model": limiting_factors(cap_values, active_blocks),
        "avg_sm_util_pct_mean": thread_row.get("avg_sm_util_pct_mean", ""),
        "tflops_mean": thread_row.get("tflops_mean", ""),
        "matmul_input_pj_per_bit_mean": thread_row.get("matmul_input_pj_per_bit_mean", ""),
    }


def build_thread_resource_rows(
    thread_rows: List[Dict[str, Any]],
    resource_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not thread_rows:
        return []
    cuda_arch = args.cuda_arch or infer_arch_from_rows(thread_rows, resource_rows)
    out: List[Dict[str, Any]] = []
    for row in thread_rows:
        unroll = str(row.get("unroll", "") or args.default_unroll)
        for role, kernel_key in (("test", "test_kernel"), ("baseline", "baseline_kernel")):
            kernel = str(row.get(kernel_key, "") or "")
            if not kernel:
                continue
            resource = find_resource(resource_rows, kernel, unroll, cuda_arch)
            out.append(occupancy_row({**row, "unroll": unroll}, resource, role, kernel, cuda_arch, args))
    return out


def plot_registers(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r.get("kernel", "")), parse_int(r.get("unroll")), str(r.get("sm", ""))))
    labels = [f"{r.get('kernel')}\nu{r.get('unroll')} sm{r.get('sm')}" for r in rows]
    values = [parse_float(r.get("registers_per_thread")) for r in rows]
    colors = ["tab:red" if parse_bool(r.get("has_spills")) else "tab:blue" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(rows)), 5.0))
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_ylabel("Registers / thread")
    ax.set_title("ptxas kernel register usage")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "kernel_registers_per_thread.png", dpi=160)
    plt.close(fig)


def plot_thread_occupancy(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    test_rows = [r for r in rows if str(r.get("role", "")) == "test"]
    if not test_rows:
        return
    test_rows = sorted(test_rows, key=lambda r: parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))))
    xs = [parse_float(r.get("threads_per_sm"), parse_float(r.get("threads"))) for r in test_rows]
    occ = [parse_float(r.get("thread_occupancy_pct_model")) for r in test_rows]
    util = [parse_float(r.get("avg_sm_util_pct_mean")) for r in test_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if any(math.isfinite(v) for v in occ):
        ax.plot(xs, occ, marker="o", label="resource occupancy model")
    if any(math.isfinite(v) for v in util):
        ax.plot(xs, util, marker="D", label="measured SM utilization")
    for x, y, row in zip(xs, occ, test_rows):
        if math.isfinite(x) and math.isfinite(y):
            ax.annotate(str(row.get("threads", "")), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Launched threads per SM")
    ax.set_ylabel("Percent")
    ax.set_title("Thread sweep resource occupancy vs measured utilization")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "thread_sweep_resource_occupancy.png", dpi=160)
    plt.close(fig)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ptxas kernel resources and occupancy model")
    parser.add_argument("--ptxas-log", type=Path, action="append", default=[], help="Build log containing -Xptxas=-v output")
    parser.add_argument("--result-dir", type=Path, default=None, help="Analyzed result directory with thread_sweep_summary.csv")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cuda-arch", default="", help="CUDA arch without sm_ prefix, e.g. 80, 86, 90")
    parser.add_argument("--default-unroll", default="8")
    parser.add_argument("--blocks-per-sm", type=int, default=8)
    parser.add_argument("--registers-per-sm", type=int, default=0)
    parser.add_argument("--max-threads-per-sm", type=int, default=0)
    parser.add_argument("--max-blocks-per-sm", type=int, default=0)
    parser.add_argument("--max-warps-per-sm", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true", help="Return success even if no ptxas rows are found")
    args = parser.parse_args()

    resource_rows: List[Dict[str, Any]] = []
    for path in args.ptxas_log:
        resource_rows.extend(parse_ptxas_log(path))
    resource_rows = dedupe_resource_rows(resource_rows)
    thread_rows = read_csv(args.result_dir / "thread_sweep_summary.csv") if args.result_dir else []
    occupancy_rows = build_thread_resource_rows(thread_rows, resource_rows, args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "kernel_resource_summary.csv", resource_rows)
    write_csv(args.outdir / "thread_resource_occupancy.csv", occupancy_rows)
    plot_registers(resource_rows, args.outdir / "figures")
    plot_thread_occupancy(occupancy_rows, args.outdir / "figures")

    spill_rows = [r for r in resource_rows if parse_bool(r.get("has_spills"))]
    missing_occupancy = [
        r for r in occupancy_rows
        if str(r.get("role", "")) in {"test", "baseline"} and not str(r.get("registers_per_thread", ""))
    ]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ptxas_logs": [str(p) for p in args.ptxas_log],
        "result_dir": str(args.result_dir) if args.result_dir else "",
        "counts": {
            "kernel_resource_rows": len(resource_rows),
            "thread_resource_rows": len(occupancy_rows),
            "spill_rows": len(spill_rows),
            "missing_thread_resource_rows": len(missing_occupancy),
        },
        "arch_limits": ARCH_LIMITS,
        "notes": [
            "Register and spill counts are ptxas compiler evidence for each instantiated kernel/unroll.",
            "Occupancy is a static resource model, not measured SM utilization.",
            "Use Nsight Compute counters and quality gates for final no-L2/no-spill validation.",
        ],
    }
    write_json(args.outdir / "kernel_resource_summary.json", payload)

    print(f"Wrote: {args.outdir / 'kernel_resource_summary.csv'}")
    if occupancy_rows:
        print(f"Wrote: {args.outdir / 'thread_resource_occupancy.csv'}")
    print(f"Wrote: {args.outdir / 'kernel_resource_summary.json'}")
    if not resource_rows and not args.allow_missing:
        raise SystemExit("No ptxas resource rows found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
