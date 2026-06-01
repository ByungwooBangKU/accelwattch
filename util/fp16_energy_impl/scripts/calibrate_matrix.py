#!/usr/bin/env python3
"""Calibrate FP16 benchmark matrix repeats to a target timed duration.

The strict FP16 pipeline needs test and baseline intervals long enough for stable
NVML total-energy deltas. A fixed matrix can be too short on H100 or too long on
slower/thermal-limited systems, so this tool rewrites per-role repeats from a
short probe run or from an existing analyzed summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


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


def flatten_args(args: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        else:
            out.extend([flag, str(value)])
    return out


def role_items(condition: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    roles: List[Tuple[str, Dict[str, Any]]] = []
    for role in ("baseline", "test", "single"):
        if role in condition:
            roles.append((role, condition[role]))
    return roles


def merged_args(
    defaults: Dict[str, Any],
    condition: Dict[str, Any],
    role_args: Dict[str, Any],
    gpu: Optional[int],
) -> Dict[str, Any]:
    out = dict(defaults)
    out.update(condition.get("args", {}))
    out.update(role_args)
    if gpu is not None:
        out["device"] = gpu
    return out


def role_target_s(role: str, args: argparse.Namespace) -> float:
    if role == "baseline":
        return args.target_baseline_s
    return args.target_test_s


def current_repeats(
    defaults: Dict[str, Any],
    condition: Dict[str, Any],
    role_args: Dict[str, Any],
) -> int:
    merged = dict(defaults)
    merged.update(condition.get("args", {}))
    merged.update(role_args)
    return max(1, int(parse_float(merged.get("repeats", 1), 1.0)))


def calibrated_repeats(
    elapsed_s: float,
    measured_repeats: int,
    existing_repeats: int,
    target_s: float,
    args: argparse.Namespace,
) -> Tuple[int, float, str]:
    if not math.isfinite(elapsed_s) or elapsed_s <= 0.0 or measured_repeats <= 0:
        return (existing_repeats, math.nan, "missing_or_invalid_elapsed")
    elapsed_per_repeat = elapsed_s / measured_repeats
    if elapsed_per_repeat <= 0.0:
        return (existing_repeats, math.nan, "nonpositive_elapsed_per_repeat")

    repeat_target = int(math.ceil(target_s / elapsed_per_repeat))
    repeat_target = max(args.min_repeats, repeat_target)
    repeat_target = min(args.max_repeats, repeat_target)
    if not args.allow_shrink:
        repeat_target = max(existing_repeats, repeat_target)
    return (repeat_target, elapsed_per_repeat, "updated" if repeat_target != existing_repeats else "unchanged")


def set_role_repeats(role_args: Dict[str, Any], repeats: int) -> None:
    role_args["repeats"] = int(repeats)


def run_probe(
    binary: Path,
    bench_args: Dict[str, Any],
    json_path: Path,
) -> Dict[str, Any]:
    cmd = [str(binary), *flatten_args(bench_args), "--json-out", str(json_path)]
    print("PROBE", shlex.join(cmd), flush=True)
    cp = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.stdout:
        print(cp.stdout, end="")
    if cp.stderr:
        print(cp.stderr, end="")
    if cp.returncode != 0:
        raise RuntimeError(f"Probe failed with code {cp.returncode}: {shlex.join(cmd)}")
    with json_path.open() as f:
        return json.load(f)


def probe_elapsed_s(
    *,
    binary: Path,
    outdir: Path,
    condition_name: str,
    role: str,
    bench_args: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[float, int, str]:
    repeats = max(1, args.probe_repeats)
    note = ""
    for attempt in range(args.max_probe_attempts):
        probe_args = dict(bench_args)
        probe_args["repeats"] = repeats
        json_path = outdir / "calibration_raw" / f"{condition_name}_{role}_probe{attempt}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_probe(binary, probe_args, json_path)
        elapsed_s = parse_float(result.get("cuda_elapsed_ms")) / 1000.0
        if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
            return (math.nan, repeats, "probe_elapsed_missing")
        if elapsed_s >= args.min_probe_elapsed_s or attempt + 1 >= args.max_probe_attempts:
            return (elapsed_s, repeats, note or "probe")
        elapsed_per_repeat = elapsed_s / repeats
        next_repeats = int(math.ceil(args.min_probe_elapsed_s / max(elapsed_per_repeat, 1e-12)))
        repeats = max(repeats + 1, min(args.max_probe_repeats, next_repeats))
        note = f"probe_repeated_to_reach_{args.min_probe_elapsed_s:g}s"
    return (math.nan, repeats, "probe_attempts_exhausted")


def summary_elapsed_lookup(summary_csv: Path) -> Dict[Tuple[str, str], Tuple[float, int, str]]:
    rows = read_csv(summary_csv)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition", ""))].append(row)

    out: Dict[Tuple[str, str], Tuple[float, int, str]] = {}
    for condition, group in grouped.items():
        for role, column in (("test", "elapsed_s"), ("baseline", "baseline_elapsed_s"), ("single", "elapsed_s")):
            values = [parse_float(row.get(column)) for row in group]
            clean = [value for value in values if math.isfinite(value) and value > 0.0]
            if clean:
                out[(condition, role)] = (sum(clean) / len(clean), 0, f"from_summary_{column}_mean")
    return out


def calibrate_from_summary(
    matrix: Dict[str, Any],
    args: argparse.Namespace,
    summary_csv: Path,
) -> List[Dict[str, Any]]:
    lookup = summary_elapsed_lookup(summary_csv)
    return calibrate_matrix(matrix, args, lookup=lookup)


def calibrate_from_probe(matrix: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return calibrate_matrix(matrix, args, lookup=None)


def calibrate_matrix(
    matrix: Dict[str, Any],
    args: argparse.Namespace,
    lookup: Optional[Dict[Tuple[str, str], Tuple[float, int, str]]],
) -> List[Dict[str, Any]]:
    defaults = matrix.get("defaults", {})
    conditions = matrix.get("conditions", [])
    summary_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        condition_name = str(condition.get("name", ""))
        for role, role_args in role_items(condition):
            existing = current_repeats(defaults, condition, role_args)
            bench_args = merged_args(defaults, condition, role_args, args.gpu)
            measured_repeats = existing
            note = ""
            if lookup is not None:
                elapsed_s, _, note = lookup.get((condition_name, role), (math.nan, 0, "missing_summary_row"))
            else:
                if args.binary is None:
                    raise SystemExit("--binary is required unless --from-summary is used")
                elapsed_s, measured_repeats, note = probe_elapsed_s(
                    binary=args.binary,
                    outdir=args.outdir,
                    condition_name=condition_name,
                    role=role,
                    bench_args=bench_args,
                    args=args,
                )
            if lookup is None:
                measured_repeats = max(1, measured_repeats)
            else:
                measured_repeats = existing

            target_s = role_target_s(role, args)
            updated, elapsed_per_repeat, action = calibrated_repeats(
                elapsed_s,
                measured_repeats,
                existing,
                target_s,
                args,
            )
            set_role_repeats(role_args, updated)
            summary_rows.append(
                {
                    "condition": condition_name,
                    "role": role,
                    "kernel": role_args.get("kernel", ""),
                    "threads": bench_args.get("threads", ""),
                    "iters": bench_args.get("iters", ""),
                    "unroll": bench_args.get("unroll", ""),
                    "target_elapsed_s": target_s,
                    "observed_elapsed_s": elapsed_s,
                    "measured_repeats": measured_repeats,
                    "elapsed_per_repeat_s": elapsed_per_repeat,
                    "old_repeats": existing,
                    "new_repeats": updated,
                    "action": action,
                    "note": note,
                }
            )
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate matrix repeats for FP16 energy measurements")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out-matrix", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--from-summary", type=Path, default=None, help="Use an analyzed summary.csv instead of binary probes")
    parser.add_argument("--target-test-s", type=float, default=1.0)
    parser.add_argument("--target-baseline-s", type=float, default=1.0)
    parser.add_argument("--min-repeats", type=int, default=1)
    parser.add_argument("--max-repeats", type=int, default=1000)
    parser.add_argument("--allow-shrink", action="store_true", help="Allow repeats to be reduced below the source matrix")
    parser.add_argument("--probe-repeats", type=int, default=1)
    parser.add_argument("--min-probe-elapsed-s", type=float, default=0.02)
    parser.add_argument("--max-probe-repeats", type=int, default=64)
    parser.add_argument("--max-probe-attempts", type=int, default=3)
    args = parser.parse_args()

    if args.target_test_s <= 0.0 or args.target_baseline_s <= 0.0:
        raise SystemExit("target durations must be positive")
    if args.min_repeats <= 0 or args.max_repeats < args.min_repeats:
        raise SystemExit("repeat bounds are invalid")

    matrix = load_json(args.matrix)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.from_summary:
        summary_rows = calibrate_from_summary(matrix, args, args.from_summary)
        source = str(args.from_summary)
    else:
        summary_rows = calibrate_from_probe(matrix, args)
        source = "binary_probe"

    matrix["calibration"] = {
        "generated_by": "scripts/calibrate_matrix.py",
        "source_matrix": str(args.matrix),
        "calibration_source": source,
        "target_test_s": args.target_test_s,
        "target_baseline_s": args.target_baseline_s,
        "min_repeats": args.min_repeats,
        "max_repeats": args.max_repeats,
        "allow_shrink": bool(args.allow_shrink),
    }
    write_json(args.out_matrix, matrix)
    write_csv(args.outdir / "matrix_calibration_summary.csv", summary_rows)
    print(f"Wrote calibrated matrix: {args.out_matrix}")
    print(f"Wrote calibration summary: {args.outdir / 'matrix_calibration_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
