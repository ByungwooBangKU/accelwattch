#!/usr/bin/env python3
"""Run FP16 energy microbenchmarks with synchronized nvidia-smi power logging.

This runner intentionally keeps the primary experiment matrix small. Cache-policy
experiments are supported as P1 memory baselines, not as the default P0 compute sweep.
The benchmark binary also emits an optional timed-loop NVML total-energy counter
delta. The nvidia-smi trace remains useful for fallback energy integration and
clock/temperature/utilization diagnostics.
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
    power_draw_w: Optional[float]
    power_draw_average_w: Optional[float]
    power_draw_instant_w: Optional[float]
    power_limit_w: Optional[float]
    sm_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]
    temp_c: Optional[float]
    pstate: str
    util_gpu_pct: Optional[float]
    query_mode: str
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
    QUERY_VARIANTS = [
        (
            "average_instant",
            [
                "timestamp",
                "power.draw",
                "power.draw.average",
                "power.draw.instant",
                "power.limit",
                "clocks.sm",
                "clocks.mem",
                "temperature.gpu",
                "pstate",
                "utilization.gpu",
            ],
        ),
        (
            "legacy_with_limit",
            [
                "timestamp",
                "power.draw",
                "power.limit",
                "clocks.sm",
                "clocks.mem",
                "temperature.gpu",
                "pstate",
                "utilization.gpu",
            ],
        ),
        (
            "legacy",
            [
                "timestamp",
                "power.draw",
                "clocks.sm",
                "clocks.mem",
                "temperature.gpu",
                "pstate",
                "utilization.gpu",
            ],
        ),
    ]

    def __init__(self, gpu_id: str, sample_ms: int, out_csv: Path, nvidia_smi: str = "nvidia-smi") -> None:
        self.gpu_id = gpu_id
        self.sample_s = max(sample_ms, 20) / 1000.0
        self.out_csv = out_csv
        self.nvidia_smi = nvidia_smi
        self._variant_index = 0
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
        line = ""
        mode = ""
        field_names: List[str] = []
        sample_ns = time.time_ns()
        while self._variant_index < len(self.QUERY_VARIANTS):
            mode, field_names = self.QUERY_VARIANTS[self._variant_index]
            cmd = [
                self.nvidia_smi,
                f"--id={self.gpu_id}",
                f"--query-gpu={','.join(field_names)}",
                "--format=csv,noheader,nounits",
            ]
            try:
                cp = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as exc:  # noqa: BLE001 - logger should not kill the run immediately
                self._errors.put(str(exc))
                return None
            if cp.returncode == 0:
                line = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
                break
            msg = cp.stderr.strip() or cp.stdout.strip() or f"nvidia-smi failed with code {cp.returncode}"
            if self._variant_index + 1 < len(self.QUERY_VARIANTS):
                self._errors.put(f"Power query mode {mode} unavailable ({msg}); falling back")
                self._variant_index += 1
                continue
            self._errors.put(msg)
            return None

        fields = [f.strip() for f in line.split(",")]
        if len(fields) < len(field_names):
            self._errors.put(f"Unexpected nvidia-smi output for {mode}: {line!r}")
            return None
        values = dict(zip(field_names, fields))
        power_draw_w = parse_float(values.get("power.draw", ""))
        power_average_w = parse_float(values.get("power.draw.average", ""))
        power_instant_w = parse_float(values.get("power.draw.instant", ""))
        return PowerSample(
            sample_unix_ns=sample_ns,
            timestamp_text=values.get("timestamp", ""),
            power_w=power_draw_w if power_draw_w is not None else (power_average_w or power_instant_w),
            power_draw_w=power_draw_w,
            power_draw_average_w=power_average_w,
            power_draw_instant_w=power_instant_w,
            power_limit_w=parse_float(values.get("power.limit", "")),
            sm_clock_mhz=parse_float(values.get("clocks.sm", "")),
            mem_clock_mhz=parse_float(values.get("clocks.mem", "")),
            temp_c=parse_float(values.get("temperature.gpu", "")),
            pstate=values.get("pstate", ""),
            util_gpu_pct=parse_float(values.get("utilization.gpu", "")),
            query_mode=mode,
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
                    "power_draw_w",
                    "power_draw_average_w",
                    "power_draw_instant_w",
                    "power_limit_w",
                    "sm_clock_mhz",
                    "mem_clock_mhz",
                    "temp_c",
                    "pstate",
                    "util_gpu_pct",
                    "query_mode",
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


class NvidiaSmiDmonUtilLogger:
    """Capture dmon utilization counters, including the SM utilization column."""

    def __init__(self, gpu_id: str, out_csv: Path, nvidia_smi: str = "nvidia-smi") -> None:
        self.gpu_id = gpu_id
        self.out_csv = out_csv
        self.nvidia_smi = nvidia_smi
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._errors: "queue.Queue[str]" = queue.Queue()

    def start(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[str]:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=5)
        errors: List[str] = []
        while not self._errors.empty():
            errors.append(self._errors.get())
        return errors

    def _run(self) -> None:
        cmd = [
            self.nvidia_smi,
            "dmon",
            "-i",
            str(self.gpu_id),
            "-s",
            "u",
            "-d",
            "1",
            "--format",
            "csv,nounit",
        ]
        with self.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_unix_ns",
                    "gpu",
                    "sm_util_pct",
                    "mem_util_pct",
                    "enc_util_pct",
                    "dec_util_pct",
                    "jpg_util_pct",
                    "ofa_util_pct",
                    "raw",
                ],
            )
            writer.writeheader()
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                )
            except Exception as exc:  # noqa: BLE001
                self._errors.put(str(exc))
                return
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                sample_ns = time.time_ns()
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                fields = [field.strip() for field in raw.split(",")]
                if len(fields) < 7:
                    self._errors.put(f"Unexpected nvidia-smi dmon output: {raw!r}")
                    continue
                writer.writerow(
                    {
                        "sample_unix_ns": sample_ns,
                        "gpu": fields[0],
                        "sm_util_pct": parse_float(fields[1]),
                        "mem_util_pct": parse_float(fields[2]),
                        "enc_util_pct": parse_float(fields[3]),
                        "dec_util_pct": parse_float(fields[4]),
                        "jpg_util_pct": parse_float(fields[5]),
                        "ofa_util_pct": parse_float(fields[6]),
                        "raw": raw,
                    }
                )
                f.flush()
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read().strip()
                if stderr:
                    self._errors.put(stderr)


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


def resolve_nvidia_smi_id(cuda_gpu: int, override: Optional[str]) -> str:
    if override:
        return override
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"nodevfiles", "void", "none"}:
        parts = [part.strip() for part in visible.split(",")]
        if 0 <= cuda_gpu < len(parts) and parts[cuda_gpu]:
            return parts[cuda_gpu]
    return str(cuda_gpu)


def query_nvidia_smi_metadata(nvidia_smi: str, nvidia_smi_id: str) -> Dict[str, Any]:
    fields = ["index", "uuid", "pci.bus_id", "name", "driver_version", "power.limit"]
    cmd = [
        nvidia_smi,
        f"--id={nvidia_smi_id}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # noqa: BLE001
        return {"nvidia_smi_metadata_error": str(exc)}
    if cp.returncode != 0:
        return {"nvidia_smi_metadata_error": cp.stderr.strip() or cp.stdout.strip()}
    line = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
    values = [part.strip() for part in line.split(",")]
    if len(values) < len(fields):
        return {"nvidia_smi_metadata_error": f"unexpected nvidia-smi metadata output: {line!r}"}
    data = dict(zip(fields, values))
    return {
        "nvidia_smi_index": data.get("index", ""),
        "gpu_uuid": data.get("uuid", ""),
        "pci_bus_id": data.get("pci.bus_id", ""),
        "nvidia_smi_name": data.get("name", ""),
        "driver_version": data.get("driver_version", ""),
        "nvidia_smi_power_limit_w": parse_float(data.get("power.limit", "")),
    }


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
    nvidia_smi_id: str,
    nvidia_smi_metadata: Dict[str, Any],
    no_power: bool,
) -> Dict[str, Any]:
    run_id = f"{condition_name}_rep{repeat_index:03d}_{role}_{uuid.uuid4().hex[:8]}"
    run_dir = outdir / "raw" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    power_csv = run_dir / "power.csv"
    sm_util_csv = run_dir / "sm_util.csv"
    bench_json_path = run_dir / "bench.json"

    bench_args = dict(default_args)
    bench_args.update(role_args)
    bench_args["device"] = gpu

    logger: Optional[NvidiaSmiPowerLogger] = None
    sm_util_logger: Optional[NvidiaSmiDmonUtilLogger] = None
    power_errors: List[str] = []
    sm_util_errors: List[str] = []
    dmon_id = str(nvidia_smi_metadata.get("nvidia_smi_index") or nvidia_smi_id)
    if not no_power:
        logger = NvidiaSmiPowerLogger(gpu_id=nvidia_smi_id, sample_ms=sample_ms, out_csv=power_csv, nvidia_smi=nvidia_smi)
        logger.start()
        sm_util_logger = NvidiaSmiDmonUtilLogger(gpu_id=dmon_id, out_csv=sm_util_csv, nvidia_smi=nvidia_smi)
        sm_util_logger.start()
        # Let the logger capture a pre-kernel sample for plotting context.
        time.sleep(max(sample_ms / 1000.0, 0.10))

    wall_start_ns = time.time_ns()
    try:
        result = run_benchmark(binary, bench_args, bench_json_path)
    finally:
        wall_end_ns = time.time_ns()
        if logger is not None:
            power_errors = logger.stop()
        if sm_util_logger is not None:
            sm_util_errors = sm_util_logger.stop()

    result.update(
        {
            "run_id": run_id,
            "condition": condition_name,
            "repeat_index": repeat_index,
            "role": role,
            "cuda_device_index": gpu,
            "nvidia_smi_id": nvidia_smi_id,
            "nvidia_smi_dmon_id": dmon_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "runner_wall_start_unix_ns": wall_start_ns,
            "runner_wall_end_unix_ns": wall_end_ns,
            "power_csv": str(power_csv) if not no_power else "",
            "sm_util_csv": str(sm_util_csv) if not no_power else "",
            "bench_json": str(bench_json_path),
            "power_logger_errors": power_errors,
            "sm_util_logger_errors": sm_util_errors,
            "validity_note": "unchecked; run Nsight Compute validation separately",
            **nvidia_smi_metadata,
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
    parser.add_argument(
        "--nvidia-smi-id",
        default=None,
        help="Physical nvidia-smi GPU id/UUID for telemetry. Defaults to CUDA_VISIBLE_DEVICES mapping or --gpu.",
    )
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

    nvidia_smi_id = resolve_nvidia_smi_id(args.gpu, args.nvidia_smi_id)
    nvidia_smi_metadata = query_nvidia_smi_metadata(args.nvidia_smi, nvidia_smi_id)
    print(f"Telemetry nvidia-smi id: {nvidia_smi_id}", flush=True)

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
                condition_args = cond.get("args", {})
                default_args_for_condition = dict(defaults)
                default_args_for_condition.update(condition_args)
                for role, role_args in roles:
                    result = run_one_role(
                        binary=args.binary,
                        outdir=args.outdir,
                        gpu=args.gpu,
                        condition_name=name,
                        repeat_index=repeat_index,
                        role=role,
                        role_args=role_args,
                        default_args=default_args_for_condition,
                        sample_ms=args.sample_ms,
                        nvidia_smi=args.nvidia_smi,
                        nvidia_smi_id=nvidia_smi_id,
                        nvidia_smi_metadata=nvidia_smi_metadata,
                        no_power=args.no_power,
                    )
                    log.write(json.dumps(result, sort_keys=True) + "\n")
                    log.flush()

    print(f"\nWrote run metadata: {runs_jsonl}")
    print("Next: python3 scripts/analyze_results.py --input", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
