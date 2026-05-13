#!/usr/bin/env python3
"""Measure DRAM read/write bandwidth, marginal power, and pJ/bit via CuPy.

The benchmark uses streaming DRAM kernels on a buffer much larger than L2 and
NVML power polling.  Reported energy is a *marginal GPU/board dynamic* estimate:

    pJ/bit = (E_total - P_idle * wall_time) / (transferred_bytes * 8) * 1e12

NVIDIA NVML does not expose a DRAM-rail-only power sensor on typical GPUs, so the
reported dynamic power is the workload-attributed GPU/board power above idle.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import statistics
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


def _preload_nvrtc() -> str | None:
    """Preload libnvrtc from pip wheels when CuPy cannot find it by soname."""
    targets = ["libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"]
    roots = [Path(sp) for sp in sys.path if sp]
    extra = [
        Path.home() / ".local/lib/python3.10/site-packages",
        Path.home() / ".local/lib/python3.11/site-packages",
        Path.home() / ".local/lib/python3.12/site-packages",
    ]
    seen = set()
    for base in list(roots) + extra:
        sub = base / "nvidia" / "cuda_nvrtc" / "lib"
        if not sub.exists() or sub in seen:
            continue
        seen.add(sub)
        for name in targets:
            p = sub / name
            if not p.exists():
                continue
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                return str(p)
            except OSError:
                pass
    return None


_nvrtc_path = _preload_nvrtc()

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nvtx

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import pynvml


KERNEL_CODE = r"""
extern "C" __global__
void stream_read(const float4* __restrict__ in,
                 float4* __restrict__ sink,
                 unsigned long long n, int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            float4 v = __ldcg(in + i);  // cache-global load: bypass L1, probe L2/DRAM
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (acc.x == 1.2345e-30f) sink[tid % 1024] = acc;
}

extern "C" __global__
void stream_write(float4* __restrict__ out,
                  float4* __restrict__ sink,
                  unsigned long long n, int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (int p = 0; p < passes; ++p) {
        float x = (float)((p & 255) + 1);
        float4 v = make_float4(x, x + 1.f, x + 2.f, x + 3.f);
        for (unsigned long long i = tid; i < n; i += stride) {
            out[i] = v;
        }
    }
    if (n == 0) sink[0] = make_float4(0.f, 0.f, 0.f, 0.f);
}
"""


@dataclass
class PowerSample:
    t_s: float
    power_w: float
    power_instant_w: float
    power_average_w: float
    power_instant_status: int
    power_average_status: int
    gpu_util_pct: int
    mem_util_pct: int
    sm_clock_mhz: int
    mem_clock_mhz: int
    temp_gpu_c: int
    pstate: int
    throttle_reasons: int
    phase: str


@dataclass
class PhaseResult:
    mode: str
    target_pct: int
    phase: str
    t0_s: float
    t1_s: float
    wall_s: float
    launches: int
    passes_per_launch: int
    bytes_transferred: int
    bandwidth_gbps: float
    total_energy_j: float
    idle_energy_j: float
    dynamic_energy_j: float
    avg_power_w: float
    dynamic_power_w: float
    pj_per_bit: float
    samples: int
    power_std_w: float
    power_min_w: float
    power_p05_w: float
    power_p50_w: float
    power_p95_w: float
    power_max_w: float
    sm_clock_mhz_mean: float
    mem_clock_mhz_mean: float
    temp_gpu_c_mean: float
    pstate_p50: float
    mem_util_pct_mean: float
    gpu_util_pct_mean: float
    avg_power_instant_w: float
    avg_power_average_w: float
    power_instant_samples: int
    power_average_samples: int


@dataclass
class LinearFit:
    slope_w_per_gbps: float
    intercept_power_w: float
    r2: float
    residuals: list[tuple[int, float, float, float]]


class PowerPoller(threading.Thread):
    """Background NVML poller with phase labels and trapezoid integration."""

    def __init__(self, handle, hz: int):
        super().__init__(daemon=True)
        self.handle = handle
        self.interval_s = 1.0 / hz
        self.samples: list[PowerSample] = []
        self._stop_ev = threading.Event()
        self._phase = ""
        self.t0 = 0.0

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def start(self) -> None:
        self.t0 = time.perf_counter()
        super().start()

    def run(self) -> None:
        while not self._stop_ev.is_set():
            now = time.perf_counter() - self.t0
            power_mw = nvml_or(pynvml.nvmlDeviceGetPowerUsage, self.handle, default=-1)
            power_fields = nvml_power_field_values(self.handle)
            util = nvml_or(pynvml.nvmlDeviceGetUtilizationRates, self.handle, default=None)
            sm_clock = nvml_or(
                pynvml.nvmlDeviceGetClockInfo, self.handle, pynvml.NVML_CLOCK_SM, default=-1)
            mem_clock = nvml_or(
                pynvml.nvmlDeviceGetClockInfo, self.handle, pynvml.NVML_CLOCK_MEM, default=-1)
            temp_gpu = nvml_or(
                pynvml.nvmlDeviceGetTemperature, self.handle, pynvml.NVML_TEMPERATURE_GPU,
                default=-1)
            pstate = nvml_or(pynvml.nvmlDeviceGetPerformanceState, self.handle, default=-1)
            throttle = nvml_or(
                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, self.handle, default=-1)
            self.samples.append(PowerSample(
                t_s=now,
                power_w=power_mw / 1000.0 if power_mw >= 0 else -1.0,
                power_instant_w=float(power_fields["instant_w"]),
                power_average_w=float(power_fields["average_w"]),
                power_instant_status=int(power_fields["instant_status"]),
                power_average_status=int(power_fields["average_status"]),
                gpu_util_pct=util.gpu if util is not None else -1,
                mem_util_pct=util.memory if util is not None else -1,
                sm_clock_mhz=sm_clock,
                mem_clock_mhz=mem_clock,
                temp_gpu_c=temp_gpu,
                pstate=pstate,
                throttle_reasons=throttle,
                phase=self._phase,
            ))
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_ev.set()
        self.join(timeout=2.0)

    def slice(self, t0_s: float, t1_s: float) -> list[PowerSample]:
        return [s for s in self.samples if t0_s <= s.t_s <= t1_s and s.power_w >= 0]

    def energy_j(self, t0_s: float, t1_s: float) -> float:
        sl = self.slice(t0_s, t1_s)
        if len(sl) < 2:
            return 0.0
        return sum(0.5 * (a.power_w + b.power_w) * (b.t_s - a.t_s)
                   for a, b in zip(sl[:-1], sl[1:]))


def nvml_or(fn, *args, default=None):
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return default


def nvml_or_name(name: str, *args, default=None):
    fn = getattr(pynvml, name, None)
    if fn is None:
        return default
    return nvml_or(fn, *args, default=default)


def prop(props: dict, key: str):
    v = props[key]
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def mean_or_nan(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def stdev_or_nan(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def percentile_or_nan(vals: list[float], pct: float) -> float:
    return float(np.percentile(vals, pct)) if vals else float("nan")


def decode_text(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    return str(v)


def nvml_watts(fn, handle) -> float | None:
    v = nvml_or(fn, handle, default=None)
    return v / 1000.0 if v is not None and v >= 0 else None


def nvml_watts_name(name: str, handle) -> float | None:
    v = nvml_or_name(name, handle, default=None)
    return v / 1000.0 if v is not None and v >= 0 else None


def nvml_power_field_values(handle) -> dict[str, float | int]:
    """Best-effort query of explicit NVML power fields for cross-validation."""
    out = {
        "instant_w": -1.0,
        "average_w": -1.0,
        "instant_status": -1,
        "average_status": -1,
    }
    get_fields = getattr(pynvml, "nvmlDeviceGetFieldValues", None)
    instant_id = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", None)
    average_id = getattr(pynvml, "NVML_FI_DEV_POWER_AVERAGE", None)
    if get_fields is None or instant_id is None or average_id is None:
        return out

    try:
        values = get_fields(handle, [instant_id, average_id])
    except pynvml.NVMLError:
        return out

    for value in values:
        field_id = getattr(value, "fieldId", None)
        status = int(getattr(value, "nvmlReturn", -1))
        if field_id == instant_id:
            key = "instant"
        elif field_id == average_id:
            key = "average"
        else:
            continue
        out[f"{key}_status"] = status
        if status != getattr(pynvml, "NVML_SUCCESS", 0):
            continue
        raw_mw = getattr(value.value, "uiVal", -1)
        out[f"{key}_w"] = raw_mw / 1000.0 if raw_mw >= 0 else -1.0
    return out


def nvml_snapshot(handle) -> dict:
    """Best-effort device state snapshot for power experiment provenance."""
    mig_mode = nvml_or_name("nvmlDeviceGetMigMode", handle, default=None)
    ecc_mode = nvml_or_name("nvmlDeviceGetEccMode", handle, default=None)
    mem_info = nvml_or(pynvml.nvmlDeviceGetMemoryInfo, handle, default=None)
    util = nvml_or(pynvml.nvmlDeviceGetUtilizationRates, handle, default=None)
    return {
        "name": decode_text(nvml_or(pynvml.nvmlDeviceGetName, handle, default="")),
        "uuid": decode_text(nvml_or(pynvml.nvmlDeviceGetUUID, handle, default="")),
        "driver_version": decode_text(nvml_or(pynvml.nvmlSystemGetDriverVersion, default="")),
        "vbios_version": decode_text(nvml_or(pynvml.nvmlDeviceGetVbiosVersion, handle, default="")),
        "pstate": nvml_or(pynvml.nvmlDeviceGetPerformanceState, handle, default=None),
        "sm_clock_mhz": nvml_or(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM, default=None),
        "mem_clock_mhz": nvml_or(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM, default=None),
        "temperature_gpu_c": nvml_or(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU,
            default=None),
        "power_usage_w": nvml_watts(pynvml.nvmlDeviceGetPowerUsage, handle),
        "power_limit_w": nvml_watts_name("nvmlDeviceGetPowerManagementLimit", handle),
        "power_default_limit_w": nvml_watts_name(
            "nvmlDeviceGetPowerManagementDefaultLimit", handle),
        "enforced_power_limit_w": nvml_watts_name("nvmlDeviceGetEnforcedPowerLimit", handle),
        "throttle_reasons_hex": (
            f"0x{nvml_or(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle, default=0):x}"
        ),
        "persistence_mode": nvml_or(pynvml.nvmlDeviceGetPersistenceMode, handle, default=None),
        "compute_mode": nvml_or(pynvml.nvmlDeviceGetComputeMode, handle, default=None),
        "mig_mode": list(mig_mode) if isinstance(mig_mode, tuple) else mig_mode,
        "ecc_mode": list(ecc_mode) if isinstance(ecc_mode, tuple) else ecc_mode,
        "memory_total_bytes": getattr(mem_info, "total", None),
        "memory_used_bytes": getattr(mem_info, "used", None),
        "gpu_util_pct": getattr(util, "gpu", None),
        "mem_util_pct": getattr(util, "memory", None),
    }


def measure_idle_power(handle, seconds: float, hz: int) -> tuple[float, float, int]:
    poller = PowerPoller(handle, hz)
    poller.start()
    poller.set_phase("idle_baseline")
    time.sleep(seconds)
    poller.stop()
    ps = [s.power_w for s in poller.samples if s.power_w >= 0]
    if not ps:
        return 0.0, 0.0, 0
    return statistics.fmean(ps), statistics.pstdev(ps) if len(ps) > 1 else 0.0, len(ps)


def calibrate_kernel(kernel, stream, blocks: int, threads: int, buf, sink,
                     n_f4: int, cal_passes: int, repeats: int) -> tuple[float, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    best_ms_per_pass = float("inf")
    for _ in range(repeats):
        with stream:
            start.record(stream=stream)
            kernel((blocks,), (threads,),
                   (buf, sink, np.uint64(n_f4), np.int32(cal_passes)))
            end.record(stream=stream)
        end.synchronize()
        best_ms_per_pass = min(best_ms_per_pass,
                               cp.cuda.get_elapsed_time(start, end) / cal_passes)
    return best_ms_per_pass, 1.0 / (best_ms_per_pass * 1e-3)


def warn_window_quantization(targets: list[int], window_ms: float,
                             calibration: dict[str, dict[str, float]]) -> None:
    """Warn when a large buffer makes duty-cycle pass counts too coarse."""
    duty_targets = [t for t in targets if 0 < t < 100]
    if not duty_targets or window_ms <= 0:
        return

    min_target = min(duty_targets)
    for mode, data in calibration.items():
        ms_per_pass = data["ms_per_pass"]
        min_desired_passes = window_ms * min_target / 100.0 / ms_per_pass
        if min_desired_passes < 4.0:
            recommended_ms = ms_per_pass * 4.0 * 100.0 / min_target
            print(
                f"[warn] {mode} duty window may quantize low targets: "
                f"{min_target}% requests only {min_desired_passes:.2f} "
                f"passes/window. Consider --window-ms >= {recommended_ms:.0f} "
                "for cleaner non-100 target separation."
            )

        for target in duty_targets:
            desired_passes = window_ms * target / 100.0 / ms_per_pass
            actual_passes = max(1, int(round(desired_passes)))
            nominal_target = actual_passes * ms_per_pass / window_ms * 100.0
            if abs(nominal_target - target) > 5.0:
                print(
                    f"[warn] {mode} target {target}% is rounded to "
                    f"{actual_passes} pass(es)/window, nominally "
                    f"{nominal_target:.1f}% with --window-ms={window_ms:g}."
                )


def run_phase(mode: str, target: int, kernel, stream, blocks: int, threads: int,
              buf, sink, n_f4: int, buf_bytes: int, peak_passes_per_s: float,
              phase_seconds: float, window_ms: float, poller: PowerPoller,
              idle_power_w: float) -> PhaseResult:
    phase = f"{mode}_{target}"
    if target == 0:
        passes = 0
        window_s = 0.0
    else:
        active_ms = window_ms * target / 100.0
        ms_per_pass = 1000.0 / peak_passes_per_s
        passes = max(1, int(round(active_ms / ms_per_pass)))
        window_s = window_ms / 1000.0
    launches = 0

    with nvtx.annotate(phase, color="blue" if mode == "read" else "purple"):
        poller.set_phase(phase)
        t0_abs = time.perf_counter()
        t0 = t0_abs - poller.t0
        deadline = t0_abs + phase_seconds
        if target == 0:
            time.sleep(max(0.0, phase_seconds))
        else:
            while time.perf_counter() < deadline:
                w0 = time.perf_counter()
                with stream:
                    kernel((blocks,), (threads,),
                           (buf, sink, np.uint64(n_f4), np.int32(passes)))
                stream.synchronize()
                launches += 1
                if target < 100:
                    rest = window_s - (time.perf_counter() - w0)
                    if rest > 2e-4:
                        time.sleep(rest)
        t1 = time.perf_counter() - poller.t0

    wall_s = max(t1 - t0, 1e-12)
    bytes_transferred = int(launches * passes * buf_bytes)
    bandwidth_gbps = bytes_transferred / wall_s / 1e9
    total_energy_j = poller.energy_j(t0, t1)
    idle_energy_j = idle_power_w * wall_s
    dynamic_energy_j = max(0.0, total_energy_j - idle_energy_j)
    dynamic_power_w = dynamic_energy_j / wall_s
    pj_per_bit = (dynamic_energy_j / (bytes_transferred * 8.0) * 1e12
                  if bytes_transferred > 0 else float("nan"))
    sl = poller.slice(t0, t1)
    power_vals = [s.power_w for s in sl if s.power_w >= 0]
    power_instant_vals = [s.power_instant_w for s in sl if s.power_instant_w >= 0]
    power_average_vals = [s.power_average_w for s in sl if s.power_average_w >= 0]
    sm_clock_vals = [s.sm_clock_mhz for s in sl if s.sm_clock_mhz >= 0]
    mem_clock_vals = [s.mem_clock_mhz for s in sl if s.mem_clock_mhz >= 0]
    temp_vals = [s.temp_gpu_c for s in sl if s.temp_gpu_c >= 0]
    pstate_vals = [s.pstate for s in sl if s.pstate >= 0]
    return PhaseResult(
        mode=mode, target_pct=target, phase=phase, t0_s=t0, t1_s=t1,
        wall_s=wall_s, launches=launches, passes_per_launch=passes,
        bytes_transferred=bytes_transferred, bandwidth_gbps=bandwidth_gbps,
        total_energy_j=total_energy_j, idle_energy_j=idle_energy_j,
        dynamic_energy_j=dynamic_energy_j,
        avg_power_w=total_energy_j / wall_s if total_energy_j > 0 else float("nan"),
        dynamic_power_w=dynamic_power_w, pj_per_bit=pj_per_bit, samples=len(sl),
        power_std_w=stdev_or_nan(power_vals),
        power_min_w=min(power_vals) if power_vals else float("nan"),
        power_p05_w=percentile_or_nan(power_vals, 5),
        power_p50_w=percentile_or_nan(power_vals, 50),
        power_p95_w=percentile_or_nan(power_vals, 95),
        power_max_w=max(power_vals) if power_vals else float("nan"),
        sm_clock_mhz_mean=mean_or_nan(sm_clock_vals),
        mem_clock_mhz_mean=mean_or_nan(mem_clock_vals),
        temp_gpu_c_mean=mean_or_nan(temp_vals),
        pstate_p50=percentile_or_nan(pstate_vals, 50),
        mem_util_pct_mean=mean_or_nan([s.mem_util_pct for s in sl if s.mem_util_pct >= 0]),
        gpu_util_pct_mean=mean_or_nan([s.gpu_util_pct for s in sl if s.gpu_util_pct >= 0]),
        avg_power_instant_w=mean_or_nan(power_instant_vals),
        avg_power_average_w=mean_or_nan(power_average_vals),
        power_instant_samples=len(power_instant_vals),
        power_average_samples=len(power_average_vals),
    )


def save_csvs(out_dir: Path, stem: str, results: list[PhaseResult],
              samples: list[PowerSample]) -> tuple[Path, Path]:
    summary_csv = out_dir / f"{stem}_summary.csv"
    trace_csv = out_dir / f"{stem}_trace.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "target_pct", "phase", "wall_s", "launches",
            "passes_per_launch", "bytes_transferred", "bandwidth_gbps",
            "avg_power_w", "dynamic_power_w", "total_energy_j",
            "idle_energy_j", "dynamic_energy_j", "pj_per_bit",
            "power_std_w", "power_min_w", "power_p05_w", "power_p50_w",
            "power_p95_w", "power_max_w",
            "sm_clock_mhz_mean", "mem_clock_mhz_mean", "temp_gpu_c_mean", "pstate_p50",
            "mem_util_pct_mean", "gpu_util_pct_mean",
            "avg_power_instant_w", "avg_power_average_w",
            "power_instant_samples", "power_average_samples", "samples",
        ])
        for r in results:
            w.writerow([
                r.mode, r.target_pct, r.phase, f"{r.wall_s:.6f}", r.launches,
                r.passes_per_launch, r.bytes_transferred,
                f"{r.bandwidth_gbps:.3f}", f"{r.avg_power_w:.3f}",
                f"{r.dynamic_power_w:.3f}", f"{r.total_energy_j:.6f}",
                f"{r.idle_energy_j:.6f}", f"{r.dynamic_energy_j:.6f}",
                f"{r.pj_per_bit:.6f}", f"{r.power_std_w:.3f}",
                f"{r.power_min_w:.3f}", f"{r.power_p05_w:.3f}",
                f"{r.power_p50_w:.3f}", f"{r.power_p95_w:.3f}",
                f"{r.power_max_w:.3f}", f"{r.sm_clock_mhz_mean:.3f}",
                f"{r.mem_clock_mhz_mean:.3f}", f"{r.temp_gpu_c_mean:.3f}",
                f"{r.pstate_p50:.3f}", f"{r.mem_util_pct_mean:.3f}",
                f"{r.gpu_util_pct_mean:.3f}", f"{r.avg_power_instant_w:.3f}",
                f"{r.avg_power_average_w:.3f}", r.power_instant_samples,
                r.power_average_samples, r.samples,
            ])
    with open(trace_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_s", "power_w", "power_instant_w", "power_average_w",
            "power_instant_status", "power_average_status",
            "gpu_util_pct", "mem_util_pct",
            "sm_clock_mhz", "mem_clock_mhz", "temp_gpu_c", "pstate",
            "throttle_reasons_hex", "phase",
        ])
        for s in samples:
            w.writerow([f"{s.t_s:.6f}", f"{s.power_w:.3f}",
                        f"{s.power_instant_w:.3f}", f"{s.power_average_w:.3f}",
                        s.power_instant_status, s.power_average_status,
                        s.gpu_util_pct, s.mem_util_pct,
                        s.sm_clock_mhz, s.mem_clock_mhz, s.temp_gpu_c, s.pstate,
                        f"0x{s.throttle_reasons:x}" if s.throttle_reasons >= 0 else "",
                        s.phase])
    return summary_csv, trace_csv


def save_plot(out_dir: Path, stem: str, gpu_name: str, idle_power_w: float,
              results: list[PhaseResult], samples: list[PowerSample]) -> Path:
    png = out_dir / f"{stem}.png"
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

    t = [s.t_s for s in samples if s.power_w >= 0]
    p = [s.power_w for s in samples if s.power_w >= 0]
    ax0.plot(t, p, lw=0.9, color="#1f77b4", label="NVML total GPU/board power")
    ax0.axhline(idle_power_w, color="black", ls="--", lw=1.0,
                label=f"idle baseline {idle_power_w:.1f} W")
    for r in results:
        ax0.axvspan(r.t0_s, r.t1_s, alpha=0.10,
                    color="#2ca02c" if r.mode == "read" else "#9467bd")
        ax0.text((r.t0_s + r.t1_s) / 2, max(p) if p else idle_power_w,
                 r.phase, ha="center", va="bottom", fontsize=8, rotation=0)
    ax0.set_ylabel("power (W)")
    ax0.set_title(f"DRAM read/write pJ/bit — {gpu_name}")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    labels = [r.phase for r in results]
    x = np.arange(len(results))
    colors = ["#2ca02c" if r.mode == "read" else "#9467bd" for r in results]
    ax1.bar(x, [r.bandwidth_gbps for r in results], color=colors, alpha=0.75)
    ax1.set_ylabel("BW (GB/s)")
    ax1.set_xticks(x, labels, rotation=30, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, [r.pj_per_bit for r in results], color=colors, alpha=0.75)
    ax2.set_ylabel("dynamic pJ/bit")
    ax2.set_xticks(x, labels, rotation=30, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xlabel("phase")

    fig.text(0.01, 0.01,
             "pJ/bit = max(0, integrated NVML power - idle baseline × time) / "
             "(streamed bytes × 8). NVML is not DRAM-rail-only power.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def ordered_modes(results: list[PhaseResult]) -> list[str]:
    modes: list[str] = []
    for r in results:
        if r.mode not in modes:
            modes.append(r.mode)
    return modes


def linear_fit_power_vs_bw(points: list[tuple[int, float, float]]) -> LinearFit | None:
    if len(points) < 2:
        return None
    n = len(points)
    sx = sum(x for _, x, _ in points)
    sy = sum(y for _, _, y in points)
    sxx = sum(x * x for _, x, _ in points)
    sxy = sum(x * y for _, x, y in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    residuals = [(target, x, y, y - (slope * x + intercept)) for target, x, y in points]
    ss_res = sum(resid * resid for _, _, _, resid in residuals)
    y_mean = sy / n
    ss_tot = sum((y - y_mean) * (y - y_mean) for _, _, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return LinearFit(
        slope_w_per_gbps=slope,
        intercept_power_w=intercept,
        r2=r2,
        residuals=residuals,
    )


def make_analysis_rows(results: list[PhaseResult]) -> list[dict[str, str]]:
    by = {(r.mode, r.target_pct): r for r in results}
    rows: list[dict[str, str]] = []
    for mode in ordered_modes(results):
        r0 = by.get((mode, 0))
        r100 = by.get((mode, 100))
        if r0 is not None and r100 is not None and r100.bandwidth_gbps > 0:
            delta_power_w = r100.avg_power_w - r0.avg_power_w
            pj_per_bit = delta_power_w * 1000.0 / (8.0 * r100.bandwidth_gbps)
            rows.append({
                "mode": mode,
                "method": "100_minus_0_avg_power",
                "target_points": "0,100",
                "baseline_power_w": f"{r0.avg_power_w:.6f}",
                "active_power_w": f"{r100.avg_power_w:.6f}",
                "delta_power_w": f"{delta_power_w:.6f}",
                "bandwidth_gbps": f"{r100.bandwidth_gbps:.6f}",
                "slope_w_per_gbps": "",
                "intercept_power_w": "",
                "r2": "",
                "max_abs_residual_w": "",
                "pj_per_bit": f"{pj_per_bit:.6f}",
                "note": "phase-local avg_power delta; not DRAM-rail-only",
            })

        targets = sorted(t for m, t in by if m == mode)
        for lo_idx, lo in enumerate(targets):
            for hi in targets[lo_idx + 1:]:
                rlo = by[(mode, lo)]
                rhi = by[(mode, hi)]
                delta_bw = rhi.bandwidth_gbps - rlo.bandwidth_gbps
                if delta_bw <= 0:
                    continue
                delta_power_w = rhi.avg_power_w - rlo.avg_power_w
                pj_per_bit = delta_power_w * 1000.0 / (8.0 * delta_bw)
                rows.append({
                    "mode": mode,
                    "method": "pair_delta_avg_power",
                    "target_points": f"{hi}-{lo}",
                    "baseline_power_w": f"{rlo.avg_power_w:.6f}",
                    "active_power_w": f"{rhi.avg_power_w:.6f}",
                    "delta_power_w": f"{delta_power_w:.6f}",
                    "bandwidth_gbps": f"{delta_bw:.6f}",
                    "slope_w_per_gbps": "",
                    "intercept_power_w": "",
                    "r2": "",
                    "max_abs_residual_w": "",
                    "pj_per_bit": f"{pj_per_bit:.6f}",
                    "note": "all-pairs phase-local avg_power delta over delta bandwidth",
                })

        slope_targets = (50, 75, 100)
        points = [
            (t, by[(mode, t)].bandwidth_gbps, by[(mode, t)].avg_power_w)
            for t in slope_targets
            if (mode, t) in by and by[(mode, t)].bandwidth_gbps > 0
        ]
        fit = linear_fit_power_vs_bw(points)
        if fit is not None:
            max_abs_resid = max(abs(r[3]) for r in fit.residuals)
            rows.append({
                "mode": mode,
                "method": "slope_avg_power_vs_bw",
                "target_points": ",".join(str(t) for t in slope_targets if (mode, t) in by),
                "baseline_power_w": "",
                "active_power_w": "",
                "delta_power_w": "",
                "bandwidth_gbps": "",
                "slope_w_per_gbps": f"{fit.slope_w_per_gbps:.9f}",
                "intercept_power_w": f"{fit.intercept_power_w:.6f}",
                "r2": f"{fit.r2:.6f}",
                "max_abs_residual_w": f"{max_abs_resid:.6f}",
                "pj_per_bit": f"{fit.slope_w_per_gbps * 1000.0 / 8.0:.6f}",
                "note": "recommended marginal board-energy estimate",
            })
    return rows


def save_analysis_csv(out_dir: Path, stem: str, rows: list[dict[str, str]]) -> Path | None:
    if not rows:
        return None
    path = out_dir / f"{stem}_analysis.csv"
    fields = [
        "mode", "method", "target_points", "baseline_power_w", "active_power_w",
        "delta_power_w", "bandwidth_gbps", "slope_w_per_gbps",
        "intercept_power_w", "r2", "max_abs_residual_w", "pj_per_bit", "note",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def save_metadata_json(out_dir: Path, stem: str, metadata: dict) -> Path:
    path = out_dir / f"{stem}_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return path


def analysis_value(rows: list[dict[str, str]], mode: str, method: str,
                   field: str) -> float:
    for row in rows:
        if row["mode"] == mode and row["method"] == method and row[field]:
            return float(row[field])
    return float("nan")


def save_analysis_plot(out_dir: Path, stem: str, gpu_name: str,
                       results: list[PhaseResult], samples: list[PowerSample],
                       analysis_rows: list[dict[str, str]]) -> Path | None:
    if not results:
        return None

    png = out_dir / f"{stem}_analysis.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax0, ax1, ax2, ax3 = axes.ravel()
    modes = ordered_modes(results)
    colors = {"read": "#2ca02c", "write": "#9467bd"}
    markers = {"read": "o", "write": "s"}
    by = {(r.mode, r.target_pct): r for r in results}

    # Panel 1: average power versus achieved bandwidth, with the 50/75/100 fit.
    for mode in modes:
        mode_results = [r for r in results if r.mode == mode and r.bandwidth_gbps > 0]
        if not mode_results:
            continue
        c = colors.get(mode, "#1f77b4")
        ax0.scatter(
            [r.bandwidth_gbps for r in mode_results],
            [r.avg_power_w for r in mode_results],
            color=c, marker=markers.get(mode, "o"), label=f"{mode} phases", zorder=3)
        for r in mode_results:
            ax0.annotate(str(r.target_pct), (r.bandwidth_gbps, r.avg_power_w),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)

        fit_points = [
            (t, by[(mode, t)].bandwidth_gbps, by[(mode, t)].avg_power_w)
            for t in (50, 75, 100)
            if (mode, t) in by and by[(mode, t)].bandwidth_gbps > 0
        ]
        fit = linear_fit_power_vs_bw(fit_points)
        if fit is not None:
            xs = np.linspace(min(p[1] for p in fit_points),
                             max(p[1] for p in fit_points), 50)
            ys = fit.slope_w_per_gbps * xs + fit.intercept_power_w
            pj = fit.slope_w_per_gbps * 1000.0 / 8.0
            ax0.plot(xs, ys, color=c, lw=1.6,
                     label=f"{mode} slope={pj:.2f} pJ/bit R2={fit.r2:.3f}")
    ax0.set_title("avg power vs bandwidth")
    ax0.set_xlabel("bandwidth (GB/s)")
    ax0.set_ylabel("avg power (W)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)

    # Panel 2: compare phase-local 100%-0% estimate with slope estimate.
    x = np.arange(len(modes))
    width = 0.35
    delta_vals = [
        analysis_value(analysis_rows, mode, "100_minus_0_avg_power", "pj_per_bit")
        for mode in modes
    ]
    slope_vals = [
        analysis_value(analysis_rows, mode, "slope_avg_power_vs_bw", "pj_per_bit")
        for mode in modes
    ]
    ax1.bar(x - width / 2, delta_vals, width, label="100%-0%", color="#4c78a8")
    ax1.bar(x + width / 2, slope_vals, width, label="50/75/100 slope", color="#f58518")
    ax1.set_title("pJ/bit estimator comparison")
    ax1.set_ylabel("pJ/bit")
    ax1.set_xticks(x, modes)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    # Panel 3: residuals from the recommended avg_power~bandwidth fit.
    for mode in modes:
        fit_points = [
            (t, by[(mode, t)].bandwidth_gbps, by[(mode, t)].avg_power_w)
            for t in (50, 75, 100)
            if (mode, t) in by and by[(mode, t)].bandwidth_gbps > 0
        ]
        fit = linear_fit_power_vs_bw(fit_points)
        if fit is None:
            continue
        c = colors.get(mode, "#1f77b4")
        targets = [r[0] for r in fit.residuals]
        residuals = [r[3] for r in fit.residuals]
        ax2.plot(targets, residuals, marker=markers.get(mode, "o"),
                 color=c, label=mode)
    ax2.axhline(0.0, color="black", lw=1.0, ls="--")
    ax2.set_title("fit residuals")
    ax2.set_xlabel("target (%)")
    ax2.set_ylabel("avg power residual (W)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Panel 4: phase-local power distribution from raw NVML samples.
    phase_labels = [r.phase for r in results]
    groups = [
        [s.power_w for s in samples if s.phase == r.phase and s.power_w >= 0]
        for r in results
    ]
    nonempty = [(label, vals) for label, vals in zip(phase_labels, groups) if vals]
    if nonempty:
        labels, vals = zip(*nonempty)
        try:
            ax3.boxplot(vals, tick_labels=labels, showfliers=False)
        except TypeError:
            ax3.boxplot(vals, labels=labels, showfliers=False)
        ax3.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax3.set_title("phase power distribution")
    ax3.set_ylabel("power (W)")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"DRAM pJ/bit analysis — {gpu_name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def print_analysis(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    print()
    print("post-analysis")
    print("-" * 111)
    print(f"{'mode':<6} {'method':<24} {'points':<10} {'P0(W)':>9} {'P100(W)':>9} "
          f"{'dP(W)':>9} {'BW(GB/s)':>10} {'R^2':>8} {'max|res|W':>10} {'pJ/bit':>10}")
    print("-" * 111)
    for row in rows:
        print(f"{row['mode']:<6} {row['method']:<24} {row['target_points']:<10} "
              f"{row['baseline_power_w'] or '-':>9} {row['active_power_w'] or '-':>9} "
              f"{row['delta_power_w'] or '-':>9} {row['bandwidth_gbps'] or '-':>10} "
              f"{row['r2'] or '-':>8} {row['max_abs_residual_w'] or '-':>10} "
              f"{row['pj_per_bit']:>10}")
    print()
    print("[note] 100_minus_0 uses phase-local avg_power delta. "
          "pair_delta_avg_power checks all target pairs. "
          "slope_avg_power_vs_bw uses avg_power~bandwidth fit over available 50/75/100 points.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", choices=["read", "write"],
                    default=["read", "write"], help="traffic types to measure")
    ap.add_argument("--targets", type=int, nargs="+", default=[100],
                    help="duty-cycle targets as percent of calibrated mode peak")
    ap.add_argument("--phase-seconds", type=float, default=5.0)
    ap.add_argument("--idle-seconds", type=float, default=5.0,
                    help="idle baseline duration before active phases")
    ap.add_argument("--window-ms", type=float, default=20.0)
    ap.add_argument("--poll-hz", type=int, default=100)
    ap.add_argument("--buf-bytes", type=int, default=None,
                    help="default: max(1 GiB, 64 * L2)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cal-passes", type=int, default=8)
    ap.add_argument("--cal-repeats", type=int, default=3)
    args = ap.parse_args()
    bad_targets = [t for t in args.targets if t < 0 or t > 100]
    if bad_targets:
        raise SystemExit(f"--targets must be between 0 and 100: {bad_targets}")

    cp.cuda.Device(args.device).use()
    props = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = prop(props, "name")
    sm_count = props["multiProcessorCount"]
    l2_bytes = props["l2CacheSize"]
    if args.buf_bytes is None:
        # Heuristic, not a DRAM constant: keep the streaming reuse distance far
        # beyond L2 while avoiding an unnecessarily huge allocation by default.
        args.buf_bytes = max(1 << 30, l2_bytes * 64)
    n_f4 = args.buf_bytes // 16
    args.buf_bytes = n_f4 * 16

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    metadata_before = nvml_snapshot(handle)

    print(f"[info] GPU={gpu_name} SMs={sm_count} L2={l2_bytes/(1<<20):.1f} MiB "
          f"buf={args.buf_bytes/(1<<30):.2f} GiB modes={args.modes}")
    if _nvrtc_path:
        print(f"[info] preloaded nvrtc: {_nvrtc_path}")

    buf = cp.empty(n_f4 * 4, dtype=cp.float32)
    sink = cp.empty(1024 * 4, dtype=cp.float32)
    buf.fill(1.0)
    module = cp.RawModule(code=KERNEL_CODE, options=("--std=c++14",))
    kernels = {
        "read": module.get_function("stream_read"),
        "write": module.get_function("stream_write"),
    }
    threads = 256
    blocks = sm_count * 32
    stream = cp.cuda.Stream(non_blocking=True)

    # Warm up both kernels so compilation and first-touch costs are outside phases.
    with stream:
        kernels["read"]((blocks,), (threads,),
                        (buf, sink, np.uint64(n_f4), np.int32(1)))
        kernels["write"]((blocks,), (threads,),
                         (buf, sink, np.uint64(n_f4), np.int32(1)))
    stream.synchronize()

    peak_passes_per_s: dict[str, float] = {}
    calibration: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        ms_per_pass, passes_per_s = calibrate_kernel(
            kernels[mode], stream, blocks, threads, buf, sink, n_f4,
            args.cal_passes, args.cal_repeats)
        peak_passes_per_s[mode] = passes_per_s
        peak_bw = args.buf_bytes * passes_per_s / 1e9
        calibration[mode] = {
            "ms_per_pass": ms_per_pass,
            "passes_per_second": passes_per_s,
            "peak_bandwidth_gbps": peak_bw,
        }
        print(f"[calib] {mode:<5} {ms_per_pass:.3f} ms/pass  "
              f"~{peak_bw:.1f} GB/s effective user-data BW")

    warn_window_quantization(args.targets, args.window_ms, calibration)

    cp.cuda.runtime.deviceSynchronize()
    print(f"[idle] measuring baseline for {args.idle_seconds:.1f} s ...")
    idle_power_w, idle_std_w, idle_n = measure_idle_power(
        handle, args.idle_seconds, args.poll_hz)
    print(f"[idle] {idle_power_w:.2f} W ± {idle_std_w:.2f} W  n={idle_n}")

    poller = PowerPoller(handle, args.poll_hz)
    results: list[PhaseResult] = []
    metadata_after: dict = {}
    poller.start()
    try:
        for mode in args.modes:
            for target in args.targets:
                print(f"[phase] {mode}_{target} start")
                result = run_phase(
                    mode, target, kernels[mode], stream, blocks, threads,
                    buf, sink, n_f4, args.buf_bytes, peak_passes_per_s[mode],
                    args.phase_seconds, args.window_ms, poller, idle_power_w)
                results.append(result)
                print(f"[phase] {result.phase:<9} BW={result.bandwidth_gbps:.1f} GB/s  "
                      f"Pdyn={result.dynamic_power_w:.1f} W  "
                      f"E={result.dynamic_energy_j:.3f} J  "
                      f"pJ/bit={result.pj_per_bit:.3f}")
                with nvtx.annotate("gap", color="gray"):
                    poller.set_phase("gap")
                    time.sleep(0.5)
    finally:
        poller.stop()
        metadata_after = nvml_snapshot(handle)
        pynvml.nvmlShutdown()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    safe_gpu = "".join(c.lower() if c.isalnum() else "_" for c in gpu_name).strip("_")
    stem = f"pjbit_cupy_{safe_gpu}_{stamp}{suffix}"
    summary_csv, trace_csv = save_csvs(out_dir, stem, results, poller.samples)
    analysis_rows = make_analysis_rows(results)
    analysis_csv = save_analysis_csv(out_dir, stem, analysis_rows)
    metadata = {
        "args": vars(args),
        "cuda": {
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
        },
        "device": {
            "name": gpu_name,
            "sm_count": sm_count,
            "l2_bytes": l2_bytes,
            "buffer_bytes": args.buf_bytes,
            "n_float4": n_f4,
            "blocks": blocks,
            "threads_per_block": threads,
        },
        "nvml_before": metadata_before,
        "nvml_after": metadata_after,
        "calibration": calibration,
        "idle": {
            "power_w_mean": idle_power_w,
            "power_w_std": idle_std_w,
            "samples": idle_n,
        },
        "notes": [
            "NVML power is GPU/board plus associated circuitry, not DRAM-rail-only.",
            "On GA10x Ampere, nvmlDeviceGetPowerUsage is documented as a 1-second average.",
            "Trace CSV includes NVML_FI_DEV_POWER_INSTANT/AVERAGE when supported.",
            "Use slope_avg_power_vs_bw as the preferred marginal board-energy estimate.",
        ],
    }
    metadata_json = save_metadata_json(out_dir, stem, metadata)
    png = save_plot(out_dir, stem, gpu_name, idle_power_w, results, poller.samples)
    analysis_png = save_analysis_plot(
        out_dir, stem, gpu_name, results, poller.samples, analysis_rows)

    print(f"[save] {summary_csv}")
    print(f"[save] {trace_csv}")
    if analysis_csv is not None:
        print(f"[save] {analysis_csv}")
    print(f"[save] {metadata_json}")
    print(f"[save] {png}")
    if analysis_png is not None:
        print(f"[save] {analysis_png}")
    print()
    print(f"{'phase':<10} {'BW(GB/s)':>10} {'Pavg(W)':>10} {'Pdyn(W)':>10} "
          f"{'Edyn(J)':>10} {'pJ/bit':>10}")
    print("-" * 66)
    for r in results:
        print(f"{r.phase:<10} {r.bandwidth_gbps:>10.1f} {r.avg_power_w:>10.1f} "
              f"{r.dynamic_power_w:>10.1f} {r.dynamic_energy_j:>10.3f} "
              f"{r.pj_per_bit:>10.3f}")
    print_analysis(analysis_rows)


if __name__ == "__main__":
    main()
