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

import os
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


def resolve_nvml_handle(device_idx: int) -> tuple[object, str]:
    """Return (NVML handle, resolution_label) for a CUDA device, robust to
    PyTorch version differences and CUDA_VISIBLE_DEVICES scenarios.

    Resolution order :
      1. PCI bus id as a properly-formatted string  (canonical NVML path)
      2. PCI bus id as int → format `{domain:04x}:{bus:02x}:{device:02x}.0`
         (some PyTorch versions return `pci_bus_id` as int instead of str)
      3. UUID-based lookup via `nvmlDeviceGetHandleByUUID`
      4. Fallback : `nvmlDeviceGetHandleByIndex(device_idx)`

    The fallback is correct on simple machines but READS THE WRONG GPU when
    `CUDA_VISIBLE_DEVICES` re-orders devices — caller should warn loudly
    when (a) the resolution label says "by index" AND (b) CUDA_VISIBLE_DEVICES
    is set.
    """
    import torch
    props = torch.cuda.get_device_properties(device_idx)

    # --- attempt 1 : pci_bus_id as str -------------------------------------
    pci_id = getattr(props, "pci_bus_id", None)
    if isinstance(pci_id, str) and pci_id:
        try:
            h = pynvml.nvmlDeviceGetHandleByPciBusId(pci_id.encode())
            return h, f"by PCI bus id {pci_id}"
        except (pynvml.NVMLError, AttributeError, TypeError):
            pass

    # --- attempt 2 : pci_bus_id as int → constructed BDF string ------------
    if isinstance(pci_id, int):
        domain = getattr(props, "pci_domain_id", 0) or 0
        bus = pci_id
        dev = getattr(props, "pci_device_id", 0) or 0
        constructed = f"{domain:04x}:{bus:02x}:{dev:02x}.0"
        try:
            h = pynvml.nvmlDeviceGetHandleByPciBusId(constructed.encode())
            return h, f"by PCI bus id {constructed} (constructed from int)"
        except (pynvml.NVMLError, AttributeError, TypeError):
            pass

    # --- attempt 3 : UUID --------------------------------------------------
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        uuid_str = str(uuid) if not isinstance(uuid, str) else uuid
        try:
            h = pynvml.nvmlDeviceGetHandleByUUID(uuid_str.encode())
            return h, f"by UUID {uuid_str}"
        except (pynvml.NVMLError, AttributeError, TypeError):
            pass

    # --- attempt 4 : fall back to index ------------------------------------
    h = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    return h, f"by index {device_idx}"


# ---------------------------------------------------------------------------
# Power-reader resolution
#
# Three power sources NVML exposes :
#
#   "legacy"  — nvmlDeviceGetPowerUsage. ~50 ms internal averaging.
#               THIS IS WHAT nvidia-smi REPORTS. Use this if you want
#               readings to match what the user sees in nvidia-smi /
#               dmon. Default since the Hopper-default-instant change
#               in PR #36 led to ~+40 W reported idle vs nvidia-smi.
#
#   "average" — NVML_FI_DEV_POWER_AVERAGE field. Running average over a
#               longer window than legacy. Slightly smoother for steady-
#               state measurements but biases peak captures downward.
#
#   "instant" — NVML_FI_DEV_POWER_INSTANT field. ~1 ms cadence reading
#               from the on-chip power IC. Captures fast transients
#               (clock ramps, leakage decay tails) that the averaged
#               sources smooth away — but on idle / lightly-loaded
#               GPUs this picks up real instantaneous spikes (DMA,
#               telemetry, NVLink heartbeats) that DON'T appear in the
#               averaged reading, so the *mean* of instant samples can
#               run noticeably HIGHER than nvidia-smi's idle number.
#               Useful for SoC-bench leakage transients; misleading
#               for static-power baselining vs nvidia-smi.
#
# Default `prefer="legacy"` restores nvidia-smi-matching semantics
# everywhere. SoC bench / future transient analyses can opt in via
# `--power-source instant`. Field IDs only landed in CUDA 12.x pynvml;
# missing constants fall back to numeric IDs from the NVML headers.
# ---------------------------------------------------------------------------
try:
    _POWER_INSTANT_FIELD = pynvml.NVML_FI_DEV_POWER_INSTANT
except AttributeError:
    _POWER_INSTANT_FIELD = 186
try:
    _POWER_AVERAGE_FIELD = pynvml.NVML_FI_DEV_POWER_AVERAGE
except AttributeError:
    _POWER_AVERAGE_FIELD = 187


def _legacy_reader(handle):
    def read_legacy() -> int:
        return _nvml_or(pynvml.nvmlDeviceGetPowerUsage, handle, default=-1)
    return read_legacy, "nvmlDeviceGetPowerUsage (legacy, matches nvidia-smi)"


def _field_reader(handle, field_id: int, label: str):
    """Build a reader that polls one NVML field-values entry. Returns
    (reader, label) or None if the field isn't supported on this GPU.
    """
    try:
        fv = pynvml.c_nvmlFieldValue_t()
        fv.fieldId = field_id
        fv.scopeId = 0
        pynvml.nvmlDeviceGetFieldValues(handle, [fv])
        if fv.nvmlReturn != pynvml.NVML_SUCCESS or fv.value.uiVal <= 0:
            return None
        buf = pynvml.c_nvmlFieldValue_t()
        buf.fieldId = field_id
        buf.scopeId = 0

        def read() -> int:
            try:
                pynvml.nvmlDeviceGetFieldValues(handle, [buf])
                if buf.nvmlReturn == pynvml.NVML_SUCCESS:
                    return int(buf.value.uiVal)
                return -1
            except Exception:
                return -1
        return read, label
    except (AttributeError, pynvml.NVMLError, Exception):
        return None


def _make_power_reader(handle, prefer: str = "legacy"):
    """Return (reader_fn, label). reader_fn() → power in mW (or -1 on error).

    `prefer` ∈ {"legacy", "instant", "average"}. Falls back to legacy
    nvmlDeviceGetPowerUsage if the requested field-values path isn't
    available on this GPU / driver / pynvml combination.
    """
    if prefer == "instant":
        r = _field_reader(handle, _POWER_INSTANT_FIELD,
                          "NVML_FI_DEV_POWER_INSTANT (~1ms, transient-aware)")
        if r is not None:
            return r
    elif prefer == "average":
        r = _field_reader(handle, _POWER_AVERAGE_FIELD,
                          "NVML_FI_DEV_POWER_AVERAGE (running average)")
        if r is not None:
            return r
    elif prefer != "legacy":
        # Unknown prefer value — log and fall through.
        pass
    return _legacy_reader(handle)


class PowerSampler(threading.Thread):
    """Background NVML poller. Call start(), then set_phase(...) around work."""

    def __init__(self, handle, hz: int = 100, power_source: str = "legacy"):
        super().__init__(daemon=True)
        self.h = handle
        self.interval = 1.0 / hz
        self._stop_event = threading.Event()
        self._phase = ""
        self.t0 = 0.0
        self.samples: list[PowerSample] = []
        # `power_source` ∈ {"legacy", "instant", "average"}. Default
        # "legacy" matches nvidia-smi (nvmlDeviceGetPowerUsage). Opt in
        # to "instant" only when you actually want sub-50ms transients
        # — see _make_power_reader docstring for the trade-offs.
        self._read_power_mw, self.power_source = _make_power_reader(
            handle, prefer=power_source)

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
            p_mw = self._read_power_mw()
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

def measure_static_power(handle, seconds: float = 5.0, hz: int = 100,
                          power_source: str = "legacy") -> dict:
    """Sit idle for `seconds` and report mean/stdev/min/max of power & temp.

    The returned mean power is the "static" (idle) power that we subtract
    from measurements to isolate the *dynamic* energy of the workload.
    Caller should drop any GPU allocations and synchronize before calling.

    `power_source` matches PowerSampler — default "legacy" matches
    nvidia-smi. Use "instant" only when you accept the +5..40 W noise
    floor that comes with sub-ms sampling on idle GPUs.

    The raw per-sample trace is returned under the "samples" key so that
    `gpu_power_bench.py` can persist it as a sidecar CSV — the idle trace
    is what lets `analyze.py` draw a P_static(t) plot and verify that the
    baseline was actually flat (no background kernel, no clock ramp).
    """
    sampler = PowerSampler(handle, hz=hz, power_source=power_source)
    sampler.start()
    sampler.set_phase("idle_baseline")
    time.sleep(seconds)
    sampler.stop()

    # ---- P-state filtering ----------------------------------------------
    # NVIDIA driver hysteresis : after recent kernel activity the GPU
    # stays in P0 (boost clocks) for tens of seconds even with zero
    # utilization. nvidia-smi's "idle" reading is P8 (true idle, SM clock
    # at the floor — H100 ≈ 210 MHz, A100 ≈ 210 MHz). When the script
    # measures static between phases (or right after CUDA context init
    # / build_matmul warmup), the early samples are still in P0 and
    # report 30-50 W higher than the actual cold-idle power.
    #
    # Strategy : keep only samples whose sm_mhz indicates the chip has
    # actually settled into idle clocks. Threshold defaults to 500 MHz
    # — comfortably above any P8 (~210) and below any meaningful active
    # state. Override with PSTATE_IDLE_CLOCK_THRESHOLD_MHZ env var.
    #
    # Fallbacks :
    #   * sm_mhz unavailable (older driver / non-CUDA context) → use all
    #     samples, no filtering possible
    #   * fewer than 30 % of samples reach P8 within the window → use
    #     all samples + emit a warning so the operator knows the
    #     measurement may still be P0-inflated (typical when seconds is
    #     too short to ride out hysteresis; bump --static-seconds to 30+)
    p8_threshold = int(os.environ.get("PSTATE_IDLE_CLOCK_THRESHOLD_MHZ", "500"))
    all_valid = [s for s in sampler.samples if s.power_w >= 0]
    sm_known  = [s for s in all_valid if s.sm_mhz >= 0]
    sm_idle   = [s for s in sm_known  if s.sm_mhz < p8_threshold]
    pstate_filter_note = ""
    if not all_valid:
        use_samples = []
        pstate_filter_note = "no valid samples"
    elif len(sm_known) < 0.5 * len(all_valid):
        # Driver isn't reporting sm_mhz reliably — disable the filter.
        use_samples = all_valid
        pstate_filter_note = "sm_mhz unavailable; P-state filter disabled"
    elif len(sm_idle) >= 0.30 * len(sm_known):
        use_samples = sm_idle
        pstate_filter_note = (
            f"P-state filter: kept {len(sm_idle)}/{len(sm_known)} samples "
            f"with sm_mhz < {p8_threshold} (P8 idle)")
    else:
        use_samples = all_valid
        pstate_filter_note = (
            f"WARN: only {len(sm_idle)}/{len(sm_known)} samples reached "
            f"P8 idle (sm_mhz < {p8_threshold}); GPU likely still in P0 "
            f"due to recent activity. Consider --static-seconds 30+ or "
            f"`nvidia-smi -rgc` to release boost-state lock. Reading "
            f"will be inflated by hysteresis.")

    ps = [s.power_w for s in use_samples]
    ts = [s.temp_c for s in use_samples if s.temp_c >= 0]
    trace = [(s.t, s.power_w, s.temp_c) for s in sampler.samples
             if s.power_w >= 0]
    if not ps:
        return {"power_w_mean": -1.0, "power_w_std": -1.0, "power_w_min": -1.0,
                "power_w_max": -1.0, "temp_c_mean": -1.0, "n": 0,
                "samples": [], "pstate_filter_note": pstate_filter_note}
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
        "n_total_samples": len(all_valid),
        "samples":      trace,
        "pstate_filter_note": pstate_filter_note,
    }


def wait_for_cooldown(handle, target_c: int = 45, timeout_s: float = 180.0,
                      poll_s: float = 1.0, min_s: float = 0.0,
                      verbose: bool = True) -> dict:
    """Block until GPU temp falls at/below `target_c` or `timeout_s` elapses.

    The `min_s` floor guarantees we always idle at least that long regardless
    of the die temperature — important because the die sensor cools faster
    than HBM / VRMs, and starting a new measurement while those are still
    warm inflates the static-power baseline. Set to 0 to disable the floor.

    Returns a dict with the final temperature, elapsed wait time, and a
    `reached` flag so the driver can log per-experiment thermal context.
    """
    t0 = time.perf_counter()
    last_print = 0.0
    cur: Optional[int] = None
    while True:
        cur = _nvml_or(pynvml.nvmlDeviceGetTemperature, handle,
                       pynvml.NVML_TEMPERATURE_GPU, default=-1)
        elapsed = time.perf_counter() - t0
        if cur < 0:
            # No temp sensor accessible → obey the min-floor, then bail out.
            if elapsed < min_s:
                time.sleep(poll_s); continue
            return {"final_temp_c": -1, "elapsed_s": elapsed, "reached": False}
        if cur <= target_c and elapsed >= min_s:
            if verbose:
                print(f"[cooldown] {cur}°C ≤ {target_c}°C reached in {elapsed:.1f}s "
                      f"(min {min_s:.1f}s)")
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


def wait_for_pstate_idle(handle, threshold_mhz: int = 500,
                         timeout_s: float = 30.0, poll_s: float = 0.5,
                         min_consecutive: int = 3,
                         verbose: bool = True) -> dict:
    """Block until SM clock drops below `threshold_mhz` for `min_consecutive`
    consecutive samples — proving the GPU has actually entered P8 idle.

    Why this exists : `wait_for_cooldown` blocks on GPU TEMPERATURE, but
    NVIDIA driver's P0→P8 transition is driven by ACTIVITY HYSTERESIS, not
    temperature. After CUDA context init / build_matmul warmup the chip
    can sit in P0 (boost clocks) for tens of seconds even with zero
    utilization and falling die temp. If `measure_static_power()` runs
    before P8 is reached, its P-state filter (sm_mhz < 500) finds 0%
    of samples below threshold and degrades to "use all samples", so
    the reported P_static is inflated by 30..50 W — exactly the warning
    the user reports as "0/1201 samples reached P8 idle".

    `min_consecutive` samples — typically 3 — guards against the chip
    momentarily dipping below threshold during a clock ramp. Real P8
    stays low for many polling intervals.

    `threshold_mhz` defaults to 500 (override via env
    `PSTATE_IDLE_CLOCK_THRESHOLD_MHZ` — same env var that
    `measure_static_power` honours, so they're consistent).

    Returns dict with:
        final_sm_mhz       — last sm_mhz observed
        elapsed_s          — wall time in this wait
        reached            — True if P8 condition met
        n_consecutive_idle — how many consecutive idle samples accumulated
        reason             — "ok" | "timeout" | "sm_mhz_unavailable" | "disabled"
    """
    if timeout_s <= 0:
        return {"final_sm_mhz": -1, "elapsed_s": 0.0, "reached": False,
                "n_consecutive_idle": 0, "reason": "disabled"}
    threshold_mhz = int(os.environ.get(
        "PSTATE_IDLE_CLOCK_THRESHOLD_MHZ", str(threshold_mhz)))
    t0 = time.perf_counter()
    last_print = 0.0
    consecutive = 0
    last_mhz = -1
    while True:
        sm = _nvml_or(pynvml.nvmlDeviceGetClockInfo, handle,
                      pynvml.NVML_CLOCK_SM, default=-1)
        elapsed = time.perf_counter() - t0
        if sm < 0:
            # Driver doesn't expose sm_mhz reliably — bail out so caller
            # falls through to existing P-state filter logic, which
            # itself handles the "sm_mhz unavailable" case.
            if verbose:
                print(f"[pstate-idle] sm_mhz unavailable from NVML "
                      f"after {elapsed:.1f}s — skipping P8 wait")
            return {"final_sm_mhz": -1, "elapsed_s": elapsed,
                    "reached": False, "n_consecutive_idle": 0,
                    "reason": "sm_mhz_unavailable"}
        last_mhz = sm
        if sm < threshold_mhz:
            consecutive += 1
            if consecutive >= min_consecutive:
                if verbose:
                    print(f"[pstate-idle] P8 reached: sm_mhz={sm}<{threshold_mhz} "
                          f"({consecutive} consecutive samples) in {elapsed:.1f}s")
                return {"final_sm_mhz": sm, "elapsed_s": elapsed,
                        "reached": True, "n_consecutive_idle": consecutive,
                        "reason": "ok"}
        else:
            consecutive = 0
        if elapsed >= timeout_s:
            if verbose:
                print(f"[pstate-idle] timeout after {elapsed:.1f}s — "
                      f"GPU still in P0 (last sm_mhz={last_mhz}, target<{threshold_mhz}). "
                      f"Static reading WILL be inflated by hysteresis. "
                      f"Mitigations: `nvidia-smi -rgc` before sweep, "
                      f"longer --pstate-idle-wait, or longer --static-seconds.")
            return {"final_sm_mhz": last_mhz, "elapsed_s": elapsed,
                    "reached": False, "n_consecutive_idle": consecutive,
                    "reason": "timeout"}
        if verbose and (elapsed - last_print) >= 5.0:
            print(f"[pstate-idle] waiting for P8… sm_mhz={sm} (target<{threshold_mhz}) "
                  f"elapsed={elapsed:.0f}s consecutive_idle={consecutive}")
            last_print = elapsed
        time.sleep(poll_s)


def force_p8_for_measurement(handle, base_clock_mhz: int = 210,
                              verbose: bool = True):
    """Aggressively try to drive the GPU into P8 (low SM clock) so the
    upcoming `measure_static_power()` actually captures *cold idle*,
    not P0-locked boost-clock idle.

    Returns a context-manager-like dict :
        {"success": bool, "method": str, "restore": callable_or_None}

    Tries the following, in order :

      1. `nvmlDeviceSetGpuLockedClocks(handle, 210, 210)` — pin SM clock
         to base via NVML field-values API. Strongest path : the GPU
         physically can't run boost clocks while locked. Usually requires
         root + `nvidia-smi -pm 1` (persistence) on most distros, but
         worth trying — it Just Works on some setups.
         `restore` = `nvmlDeviceResetGpuLockedClocks(handle)`.

      2. `nvidia-smi -rgc -i <idx>` via subprocess — releases the
         "boost clock retention" hold. Doesn't lock low, just tells the
         driver "stop holding the boost". Then natural P-state transition
         takes effect (usually within 1-2 s of zero util).
         `restore` = no-op (subprocess already handed control back).

      3. Give up — return success=False, caller will rely on the fallback
         P-state filter inside `measure_static_power` and log a clear
         warning that the reading may be P0-inflated.

    Caller is expected to invoke `restore()` AFTER the static measurement
    completes so the workload phase isn't artificially clock-locked.
    """
    # --- Attempt 1 : NVML lock ---------------------------------------------
    try:
        if hasattr(pynvml, "nvmlDeviceSetGpuLockedClocks"):
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, base_clock_mhz,
                                                  base_clock_mhz)
            if verbose:
                print(f"[force-p8] NVML clock lock @ {base_clock_mhz} MHz "
                      f"(success — measurement guaranteed P8)")
            def _restore():
                try:
                    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                except pynvml.NVMLError:
                    pass
            return {"success": True, "method": "nvml_locked",
                    "restore": _restore}
    except (pynvml.NVMLError, AttributeError) as e:
        if verbose:
            err = str(e).lower()
            hint = ""
            if "permission" in err or "no_permission" in err or "not_supported" in err:
                hint = " (typically needs root + persistence mode : `sudo nvidia-smi -pm 1`)"
            print(f"[force-p8] NVML clock lock unavailable ({type(e).__name__}: {e}){hint}")

    # --- Attempt 2 : nvidia-smi -rgc ---------------------------------------
    try:
        import subprocess
        try:
            idx = pynvml.nvmlDeviceGetIndex(handle)
        except pynvml.NVMLError:
            idx = 0
        result = subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(idx)],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            if verbose:
                print(f"[force-p8] `nvidia-smi -rgc -i {idx}` OK — "
                      f"boost-clock retention released")
            return {"success": True, "method": "nvidia_smi_rgc",
                    "restore": None}
        else:
            if verbose:
                err = (result.stderr or result.stdout or "").strip().splitlines()
                first = err[0] if err else "(no stderr)"
                print(f"[force-p8] nvidia-smi -rgc failed (rc={result.returncode}): "
                      f"{first[:120]}")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        if verbose:
            print(f"[force-p8] nvidia-smi unavailable ({type(e).__name__}: {e})")

    # --- Give up -----------------------------------------------------------
    if verbose:
        print(f"[force-p8] could not actively force P8. The static reading "
              f"will reflect whatever P-state the GPU is in. If the GPU is "
              f"stuck in P0 (boost-clock idle), the measurement will be "
              f"30..50 W INFLATED vs true P8 idle (~70W on H100). "
              f"Mitigation : `sudo nvidia-smi -pm 1 && sudo nvidia-smi -rgc` "
              f"before sweep, or run as root, or `--pstate-idle-wait 120` "
              f"to allow longer hysteresis ride-out.")
    return {"success": False, "method": "none", "restore": None}
