#!/usr/bin/env python3
"""NVML power/temperature polling with energy integration.

Provides three facilities the main benchmark driver needs:

  1. `PowerSampler` — background thread polling instantaneous power (mW),
     GPU temperature (°C), SM/MEM clocks (MHz), and SM utilization (%).
     Exposes `.samples` (list of PowerSample) and `.energy_joules(t0, t1)`
     computed by trapezoidal integration — this is what gives us Joules.

  2. `measure_static_power()` — sits idle for a few seconds and reports the
     average power draw. Used as the static / baseline power to subtract out
     when reporting *dynamic* Joules-per-operation.

  3. `wait_for_cooldown()` — blocks until GPU temperature falls below a
     threshold (or a max timeout elapses). Keeps back-to-back experiments
     from bleeding thermal state into each other.

NVML power is sampled internally by the driver at ~20 Hz on most SKUs, so
polling faster than that just duplicates samples. We poll at 100 Hz by default
which is a reasonable compromise.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import pynvml


@dataclass
class PowerSample:
    t: float          # seconds since sampler start
    power_w: float    # instantaneous power draw (W)
    temp_c: int       # GPU core temperature (°C)
    sm_mhz: int       # SM clock (MHz), -1 if unavailable
    mem_mhz: int      # memory clock (MHz), -1 if unavailable
    gpu_util: int     # SM utilization (%), -1 if unavailable
    mem_util: int     # memory-controller utilization (%), -1 if unavailable
    phase: str = ""   # user-set label ("idle", "fp16_mul_N=1M", ...)


def _nvml_or(fn, *args, default=-1):
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return default


class PowerSampler(threading.Thread):
    """Background NVML poller. Call start(), then set_phase(...) around work."""

    def __init__(self, handle, hz: int = 100):
        super().__init__(daemon=True)
        self.h = handle
        self.interval = 1.0 / hz
        self._stop_event = threading.Event()
        self._phase = ""
        self.t0 = 0.0
        self.samples: list[PowerSample] = []

    def set_phase(self, name: str) -> None:
        self._phase = name

    def start(self) -> None:
        # Pin t0 before the thread actually runs so callers can subtract it
        # immediately. Otherwise there's a race between `start()` and the
        # first line of `run()`.
        self.t0 = time.perf_counter()
        super().start()

    def run(self) -> None:
        while not self._stop_event.is_set():
            now = time.perf_counter() - self.t0
            p_mw = _nvml_or(pynvml.nvmlDeviceGetPowerUsage, self.h, default=-1)
            temp = _nvml_or(pynvml.nvmlDeviceGetTemperature, self.h,
                            pynvml.NVML_TEMPERATURE_GPU, default=-1)
            sm = _nvml_or(pynvml.nvmlDeviceGetClockInfo, self.h,
                          pynvml.NVML_CLOCK_SM, default=-1)
            mem = _nvml_or(pynvml.nvmlDeviceGetClockInfo, self.h,
                           pynvml.NVML_CLOCK_MEM, default=-1)
            util = _nvml_or(pynvml.nvmlDeviceGetUtilizationRates, self.h, default=None)
            gpu_u = util.gpu if util is not None else -1
            mem_u = util.memory if util is not None else -1
            self.samples.append(PowerSample(
                t=now,
                power_w=p_mw / 1000.0 if p_mw >= 0 else -1.0,
                temp_c=temp, sm_mhz=sm, mem_mhz=mem,
                gpu_util=gpu_u, mem_util=mem_u,
                phase=self._phase,
            ))
            # sleep to target rate; perf_counter-based to avoid drift
            nxt = self.t0 + len(self.samples) * self.interval
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)

    # ----- analysis helpers over the in-memory sample buffer ------------------

    def _slice(self, t0: float, t1: float) -> list[PowerSample]:
        return [s for s in self.samples if t0 <= s.t <= t1 and s.power_w >= 0]

    def energy_joules(self, t0: float, t1: float) -> float:
        """Trapezoidal integration of power(t) over [t0, t1] → Joules."""
        sl = self._slice(t0, t1)
        if len(sl) < 2:
            return 0.0
        e = 0.0
        for a, b in zip(sl[:-1], sl[1:]):
            e += 0.5 * (a.power_w + b.power_w) * (b.t - a.t)
        return e

    def avg_power(self, t0: float, t1: float) -> float:
        dur = t1 - t0
        if dur <= 0:
            return 0.0
        return self.energy_joules(t0, t1) / dur

    def avg_temp(self, t0: float, t1: float) -> float:
        sl = self._slice(t0, t1)
        temps = [s.temp_c for s in sl if s.temp_c >= 0]
        return sum(temps) / len(temps) if temps else -1.0

    def peak_temp(self, t0: float, t1: float) -> int:
        sl = self._slice(t0, t1)
        temps = [s.temp_c for s in sl if s.temp_c >= 0]
        return max(temps) if temps else -1


# -------------------------------- utilities ----------------------------------

def measure_static_power(handle, seconds: float = 5.0, hz: int = 100) -> dict:
    """Sit idle for `seconds` and report mean/stdev/min/max of power & temp.

    The returned mean power is the "static" (idle) power that we subtract
    from measurements to isolate the *dynamic* energy of the workload.
    Caller should drop any GPU allocations and synchronize before calling.

    The raw per-sample trace is returned under the "samples" key so that
    `gpu_power_bench.py` can persist it as a sidecar CSV — the idle trace
    is what lets `analyze.py` draw a P_static(t) plot and verify that the
    baseline was actually flat (no background kernel, no clock ramp).
    """
    sampler = PowerSampler(handle, hz=hz)
    sampler.start()
    sampler.set_phase("idle_baseline")
    time.sleep(seconds)
    sampler.stop()

    ps = [s.power_w for s in sampler.samples if s.power_w >= 0]
    ts = [s.temp_c for s in sampler.samples if s.temp_c >= 0]
    trace = [(s.t, s.power_w, s.temp_c) for s in sampler.samples
             if s.power_w >= 0]
    if not ps:
        return {"power_w_mean": -1.0, "power_w_std": -1.0, "power_w_min": -1.0,
                "power_w_max": -1.0, "temp_c_mean": -1.0, "n": 0,
                "samples": []}
    import statistics as st
    return {
        "power_w_mean": st.fmean(ps),
        "power_w_std":  st.pstdev(ps) if len(ps) > 1 else 0.0,
        "power_w_min":  min(ps),
        "power_w_max":  max(ps),
        "temp_c_mean":  (sum(ts) / len(ts)) if ts else -1.0,
        "temp_c_min":   min(ts) if ts else -1,
        "temp_c_max":   max(ts) if ts else -1,
        "duration_s":   seconds,
        "hz":           hz,
        "n":            len(ps),
        "samples":      trace,
    }


def wait_for_cooldown(handle, target_c: int = 45, timeout_s: float = 120.0,
                      poll_s: float = 1.0, verbose: bool = True) -> dict:
    """Block until GPU temp falls at/below `target_c` or `timeout_s` elapses.

    Returns a dict with the final temperature and elapsed wait time so the
    driver can log per-experiment thermal context.
    """
    t0 = time.perf_counter()
    last_print = 0.0
    cur: Optional[int] = None
    while True:
        cur = _nvml_or(pynvml.nvmlDeviceGetTemperature, handle,
                       pynvml.NVML_TEMPERATURE_GPU, default=-1)
        elapsed = time.perf_counter() - t0
        if cur < 0:
            # No temp sensor accessible → don't loop forever.
            return {"final_temp_c": -1, "elapsed_s": elapsed, "reached": False}
        if cur <= target_c:
            if verbose:
                print(f"[cooldown] {cur}°C ≤ {target_c}°C reached in {elapsed:.1f}s")
            return {"final_temp_c": cur, "elapsed_s": elapsed, "reached": True}
        if elapsed >= timeout_s:
            if verbose:
                print(f"[cooldown] timeout after {elapsed:.1f}s at {cur}°C "
                      f"(target {target_c}°C)")
            return {"final_temp_c": cur, "elapsed_s": elapsed, "reached": False}
        if verbose and (elapsed - last_print) >= 5.0:
            print(f"[cooldown] waiting… {cur}°C → {target_c}°C ({elapsed:.0f}s)")
            last_print = elapsed
        time.sleep(poll_s)
