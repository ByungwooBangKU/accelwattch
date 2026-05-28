#!/usr/bin/env python3
"""Run FP16 energy microbenchmarks with synchronized nvidia-smi power logging.

This runner intentionally keeps the primary experiment matrix small. Cache-policy
experiments are supported as P1 memory baselines, not as the default P0 compute sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PowerSample:
    sample_unix_ns: int
    timestamp_text: str
    power_w: Optional[float]
    sm_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]
    temp_c: Optional[float]
    pstate: str
    util_gpu_pct: Optional[float]
    raw: str


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class NvidiaSmiPowerLogger:
    def __init__(self, gpu: int, sample_ms: int, out_csv: Path, nvidia_smi: str = "nvidia-smi") -> None:
        self.gpu = gpu
        self.sample_s = max(sample_ms, 20) / 1000.0
        self.out_csv = out_csv
        self.nvidia_smi = nvidia_smi
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._errors: "queue.Queue[str]" = queue.Queue()

    def start(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[str]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        errors: List[str] = []
        while not self._errors.empty():
            errors.append(self._errors.get())
        return errors

    def _query_once(self) -> Optional[PowerSample]:
        query = (
            "timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,"
            "pstate,utilization.gpu"
        )
        cmd = [
            self.nvidia_smi,
            f"--id={self.gpu}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        sample_ns = time.time_ns()
        try:
            cp = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:  # noqa: BLE001 - logger should not kill the run immediately
            self._errors.put(str(exc))
            return None
        line = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 7:
            self._errors.put(f"Unexpected nvidia-smi output: {line!r}")
            return None
        return PowerSample(
            sample_unix_ns=sample_ns,
            timestamp_text=fields[0],
            power_w=parse_float(fields[1]),
            sm_clock_mhz=parse_float(fields[2]),
            mem_clock_mhz=parse_float(fields[3]),
            temp_c=parse_float(fields[4]),
            pstate=fields[5],
            util_gpu_pct=parse_float(fields[6]),
            raw=line,
        )

    def _run(self) -> None:
        with self.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_unix_ns",
                    "timestamp_text",
                    "power_w",
                    "sm_clock_mhz",
                    "mem_clock_mhz",
                    "temp_c",
                    "pstate",
                    "util_gpu_pct",
                    "raw",
                ],
            )
            writer.writeheader()
            while not self._stop.is_set():
                sample = self._query_once()
                if sample is not None:
                    writer.writerow(sample.__dict__)
                    f.flush()
                time.sleep(self.sample_s)


def load_matrix(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def flatten_args(base: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in base.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def run_benchmark(binary: Path, bench_args: Dict[str, Any], json_path: Path) -> Dict[str, Any]:
    cmd = [str(binary), *flatten_args(bench_args), "--json-out", str(json_path)]
    print("RUN", shlex.join(cmd), flush=True)
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.stdout:
        print(cp.stdout, end="")
    if cp.stderr:
        print(cp.stderr, end="", file=sys.stderr)
    if cp.returncode != 0:
        raise RuntimeError(f"Benchmark failed with code {cp.returncode}: {shlex.join(cmd)}")
    with json_path.open() as f:
        return json.load(f)


def run_one_role(
    *,
    binary: Path,
    outdir: Path,
    gpu: int,
    condition_name: str,
    repeat_index: int,
    role: str,
    role_args: Dict[str, Any],
    default_args: Dict[str, Any],
    sample_ms: int,
    nvidia_smi: str,
    no_power: bool,
) -> Dict[str, Any]:
    run_id = f"{condition_name}_rep{repeat_index:03d}_{role}_{uuid.uuid4().hex[:8]}"
    run_dir = outdir / "raw" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    power_csv = run_dir / "power.csv"
    bench_json_path = run_dir / "bench.json"

    bench_args = dict(default_args)
    bench_args.update(role_args)
    bench_args["device"] = gpu

    logger: Optional[NvidiaSmiPowerLogger] = None
    power_errors: List[str] = []
    if not no_power:
        logger = NvidiaSmiPowerLogger(gpu=gpu, sample_ms=sample_ms, out_csv=power_csv, nvidia_smi=nvidia_smi)
        logger.start()
        # Let the logger capture a pre-kernel sample for plotting context.
        time.sleep(max(sample_ms / 1000.0, 0.05))

    wall_start_ns = time.time_ns()
    try:
        result = run_benchmark(binary, bench_args, bench_json_path)
    finally:
        wall_end_ns = time.time_ns()
        if logger is not None:
            power_errors = logger.stop()

    result.update(
        {
            "run_id": run_id,
            "condition": condition_name,
            "repeat_index": repeat_index,
            "role": role,
            "runner_wall_start_unix_ns": wall_start_ns,
            "runner_wall_end_unix_ns": wall_end_ns,
            "power_csv": str(power_csv) if not no_power else "",
            "bench_json": str(bench_json_path),
            "power_logger_errors": power_errors,
            "validity_note": "unchecked; run Nsight Compute validation separately",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FP16 energy experiment matrix")
    parser.add_argument("--binary", type=Path, default=Path("build/fp16_energy_bench"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/primary_matrix.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results/run"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full matrix N times")
    parser.add_argument("--append", action="store_true", help="Append to an existing runs.jsonl")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--no-power", action="store_true", help="Run benchmarks without nvidia-smi power logging")
    args = parser.parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")

    matrix = load_matrix(args.matrix)
    defaults: Dict[str, Any] = matrix.get("defaults", {})
    conditions: List[Dict[str, Any]] = matrix.get("conditions", [])
    if not conditions:
        raise SystemExit("No conditions found in matrix JSON")

    runs_jsonl = args.outdir / "runs.jsonl"
    if runs_jsonl.exists() and not args.append:
        raise SystemExit(f"{runs_jsonl} already exists; use a new --outdir or pass --append")

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "matrix_used.json").open("w") as f:
        json.dump(matrix, f, indent=2)

    with runs_jsonl.open("a") as log:
        for repeat_index in range(args.repeat):
            if args.repeat > 1:
                print(f"\n=== matrix repeat {repeat_index + 1}/{args.repeat} ===", flush=True)
            for cond in conditions:
                name = cond["name"]
                print(f"\n=== condition: {name} ===", flush=True)
                roles = []
                if "baseline" in cond:
                    roles.append(("baseline", cond["baseline"]))
                if "test" in cond:
                    roles.append(("test", cond["test"]))
                if "single" in cond:
                    roles.append(("single", cond["single"]))
                if not roles:
                    raise RuntimeError(f"Condition {name} has no baseline/test/single role")
                for role, role_args in roles:
                    result = run_one_role(
                        binary=args.binary,
                        outdir=args.outdir,
                        gpu=args.gpu,
                        condition_name=name,
                        repeat_index=repeat_index,
                        role=role,
                        role_args=role_args,
                        default_args=defaults,
                        sample_ms=args.sample_ms,
                        nvidia_smi=args.nvidia_smi,
                        no_power=args.no_power,
                    )
                    log.write(json.dumps(result, sort_keys=True) + "\n")
                    log.flush()

    print(f"\nWrote run metadata: {runs_jsonl}")
    print("Next: python3 scripts/analyze_results.py --input", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
