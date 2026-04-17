#!/usr/bin/env python3
"""DRAM read utilization experiment in pure Python via CuPy (no nvcc).

25 / 50 / 75 / 100% DRAM read utilization 을 10 초씩 강제 구동하고
pynvml 로 memory-controller utilization 을 폴링해서 CSV + PNG 로 남긴다.
NVTX range 도 기록하므로 nsys 가 있으면 그대로 타임라인 프로파일링도 가능.

실행:
    python3 dram_util_cupy.py
    python3 dram_util_cupy.py --buf-bytes $((8*1024**3)) --phase-seconds 10
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
import threading
import time
import warnings
from pathlib import Path


def _preload_nvrtc() -> str | None:
    """CuPy 의 libnvrtc.so 로더가 pip wheel 경로를 못 찾는 경우가 있어 미리 dlopen."""
    targets = ["libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"]
    roots = [Path(sp) for sp in sys.path if sp]
    # Also try user-site and the common pip wheel layout.
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
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                    return str(p)
                except OSError:
                    pass
    return None


_nvrtc_path = _preload_nvrtc()

import cupy as cp
import numpy as np
import nvtx

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import pynvml


# ---- streaming read kernel (compiled by NVRTC in CuPy — no nvcc needed) ----
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
            float4 v = __ldcg(in + i);
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (acc.x == 1.2345e-30f) sink[tid % 1024] = acc;
}
"""


class Poller(threading.Thread):
    """Background pynvml poller: (t_s, mem_util%, gpu_util%, phase_label)."""

    def __init__(self, handle, interval_s: float):
        super().__init__(daemon=True)
        self.h = handle
        self.interval = interval_s
        self._stopev = threading.Event()
        self.rows: list[tuple[float, int, int, str]] = []
        self.phase = ""

    def set_phase(self, name: str) -> None:
        self.phase = name

    def run(self) -> None:
        t0 = time.perf_counter()
        while not self._stopev.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(self.h)
            self.rows.append((time.perf_counter() - t0, u.memory, u.gpu, self.phase))
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stopev.set()


def prop(props: dict, key: str):
    v = props[key]
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


# Published theoretical DRAM peak (GB/s) for common GPUs, keyed by slug fragments.
# Used only to compute an "achieved / theoretical" efficiency hint — not as ground truth.
KNOWN_PEAK_GBPS = {
    "rtx_3090":   936.0,   # GDDR6X 19.5 Gbps × 384-bit
    "rtx_3090_ti": 1008.0,
    "rtx_4090":  1008.0,
    "rtx_4080":  716.8,
    "a100_80":   2039.0,   # HBM2e 3.2 Gbps × 5120-bit
    "a100":      1555.0,   # A100 40GB HBM2
    "h100_sxm":  3350.0,   # HBM3 5.23 Gbps × 5120-bit
    "h100_pcie": 2000.0,
    "h100":      2000.0,
    "v100":      900.0,
    "l40":       864.0,
    "l4":        300.0,
}


def lookup_known_peak(slug: str):
    # Longest-match key wins (e.g. "a100_80" before "a100").
    for k in sorted(KNOWN_PEAK_GBPS, key=len, reverse=True):
        if k in slug:
            return k, KNOWN_PEAK_GBPS[k]
    return None, None


def gpu_diagnostics(handle, props) -> dict:
    """Print everything relevant to DRAM BW ceiling. Returns a summary dict."""
    mem_clock_khz = props["memoryClockRate"]
    bus_width     = props["memoryBusWidth"]
    # GDDR-formula peak. For HBM this is NOT the marketing 2039 GB/s — HBM's
    # effective per-pin data rate differs from the reported `memoryClockRate`.
    theo_gddr = 2.0 * mem_clock_khz * 1e3 * bus_width / 8.0 / 1e9

    def _nvml(fn, *args, default=None):
        try:
            return fn(*args)
        except pynvml.NVMLError:
            return default

    ecc = _nvml(pynvml.nvmlDeviceGetEccMode, handle)
    ecc_state = (f"current={bool(ecc[0])} pending={bool(ecc[1])}"
                 if ecc else "n/a")
    cur_sm  = _nvml(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM,  default=-1)
    cur_mem = _nvml(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM, default=-1)
    max_sm  = _nvml(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_SM,  default=-1)
    max_mem = _nvml(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_MEM, default=-1)
    power_w = _nvml(pynvml.nvmlDeviceGetPowerUsage, handle, default=-1)
    power_lim = _nvml(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, default=-1)
    persist = _nvml(pynvml.nvmlDeviceGetPersistenceMode, handle, default=None)
    temp = _nvml(pynvml.nvmlDeviceGetTemperature, handle,
                 pynvml.NVML_TEMPERATURE_GPU, default=-1)
    reasons = _nvml(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons,
                    handle, default=0) or 0

    throttle_bits = []
    for bit_name in (
        "nvmlClocksThrottleReasonGpuIdle",
        "nvmlClocksThrottleReasonApplicationsClocksSetting",
        "nvmlClocksThrottleReasonSwPowerCap",
        "nvmlClocksThrottleReasonHwSlowdown",
        "nvmlClocksThrottleReasonHwThermalSlowdown",
        "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
        "nvmlClocksThrottleReasonSwThermalSlowdown",
        "nvmlClocksThrottleReasonSyncBoost",
    ):
        v = getattr(pynvml, bit_name, None)
        if v is not None and reasons & v:
            throttle_bits.append(bit_name.replace("nvmlClocksThrottleReason", ""))

    print(f"[diag] ECC:          {ecc_state}   "
          f"(ECC on → HBM 실효 BW ~ 이론치의 88–90%)")
    print(f"[diag] mem bus:      {bus_width}-bit")
    print(f"[diag] clocks now:   SM {cur_sm} / max {max_sm} MHz,  "
          f"MEM {cur_mem} / max {max_mem} MHz")
    print(f"[diag] power:        {power_w/1000:.0f} W / cap {power_lim/1000:.0f} W,  "
          f"temp {temp} °C,  persistence={persist}")
    print(f"[diag] GDDR-formula peak : {theo_gddr:.1f} GB/s  "
          f"(= 2 × clk × width ÷ 8; HBM 은 부정확)")
    if throttle_bits:
        print(f"[diag] !! throttling:   {', '.join(throttle_bits)}")
    return {
        "theo_gddr": theo_gddr,
        "ecc_on": bool(ecc[0]) if ecc else None,
        "throttling": throttle_bits,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf-bytes", type=int, default=None,
                    help="buffer bytes (default: max(1 GiB, 64 * L2))")
    ap.add_argument("--phase-seconds", type=float, default=10.0)
    ap.add_argument("--window-ms", type=float, default=20.0,
                    help="duty cycle window (smaller => smoother avg, larger => less launch overhead)")
    ap.add_argument("--targets", type=int, nargs="+",
                    default=[25, 50, 75, 100])
    ap.add_argument("--poll-hz", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="reports")
    ap.add_argument("--tag", type=str, default="",
                    help="extra suffix on output file names")
    ap.add_argument("--device", type=int, default=0,
                    help="CUDA device index (multi-GPU systems)")
    args = ap.parse_args()

    # ---- GPU info & buffer sizing ----
    dev = cp.cuda.Device(args.device)
    dev.use()
    p = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = prop(p, "name")
    sm_count = p["multiProcessorCount"]
    l2_bytes = p["l2CacheSize"]

    # GPU slug for per-GPU output files (A100 / RTX3090 / H100 자동 식별).
    def _slugify(name: str) -> str:
        import re
        s = re.sub(r"(?i)nvidia|geforce|pcie|sxm\d?|\bhbm\d*\b|\bon\b", "", name)
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
        return s or "gpu"
    gpu_slug = _slugify(gpu_name)

    if args.buf_bytes is None:
        args.buf_bytes = max(1 << 30, l2_bytes * 64)
    n_f4 = args.buf_bytes // 16  # float4 = 16 bytes

    print(f"[info] GPU={gpu_name} (slug={gpu_slug})  SMs={sm_count}  "
          f"L2={l2_bytes/(1<<20):.1f} MiB  "
          f"buf={args.buf_bytes/(1<<30):.2f} GiB  n_f4={n_f4}")
    if _nvrtc_path:
        print(f"[info] preloaded nvrtc: {_nvrtc_path}")

    buf  = cp.empty(n_f4 * 4, dtype=cp.float32)
    sink = cp.empty(1024 * 4, dtype=cp.float32)
    buf.fill(1.0)

    kernel = cp.RawKernel(KERNEL_CODE, "stream_read", options=("--std=c++14",))
    threads = 256
    blocks  = sm_count * 32

    stream = cp.cuda.Stream(non_blocking=True)
    # warm-up (triggers NVRTC compile)
    with stream:
        kernel((blocks,), (threads,),
               (buf, sink, np.uint64(n_f4), np.int32(1)))
    stream.synchronize()

    # ---- GPU diagnostics (ECC, clocks, power) ----
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    diag = gpu_diagnostics(handle, p)

    # ---- calibrate ms/pass (repeat 3× for a stable min reading) ----
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
    CAL = 8
    best_ms_per_pass = float("inf")
    for _ in range(3):
        with stream:
            start.record(stream=stream)
            kernel((blocks,), (threads,),
                   (buf, sink, np.uint64(n_f4), np.int32(CAL)))
            end.record(stream=stream)
        end.synchronize()
        ms = cp.cuda.get_elapsed_time(start, end) / CAL
        best_ms_per_pass = min(best_ms_per_pass, ms)
    ms_per_pass = best_ms_per_pass
    peak_gbps = args.buf_bytes / (ms_per_pass * 1e-3) / 1e9
    print(f"[calib] {ms_per_pass:.3f} ms/pass (best of 3)  "
          f"~{peak_gbps:.1f} GB/s achieved peak DRAM read")

    # Compare against the published theoretical peak, if we know this GPU.
    _k, known_peak = lookup_known_peak(gpu_slug)
    if known_peak is not None:
        eff = peak_gbps / known_peak * 100.0
        print(f"[peak]  published theoretical: {known_peak:.0f} GB/s  "
              f"→ achieved / theoretical = {eff:.1f}%")
        if eff < 80.0:
            print("[peak]  !! 실효 < 80%: ECC, clock throttling, 드라이버 TDR, "
                  "백그라운드 프로세스 확인")
        elif eff < 90.0:
            print("[peak]  참고: HBM2e + ECC on 에서 85–90% 가 정상 범위 (A100/H100)")
    else:
        print(f"[peak]  (slug '{gpu_slug}' 의 published peak unknown — "
              f"원하면 KNOWN_PEAK_GBPS 에 추가)")

    # ---- pynvml poller ----
    poller = Poller(handle, interval_s=1.0 / args.poll_hz)
    poller.start()

    window_s = args.window_ms / 1000.0

    try:
        for target in args.targets:
            tag = f"util_{target}"
            # same duty-cycle logic for all targets (incl. 100% => no sleep)
            # — this avoids long single kernels that WSL2 WDDM could TDR-kill.
            active_ms = args.window_ms * target / 100.0
            passes = max(1, int(round(active_ms / ms_per_pass)))
            with nvtx.annotate(tag, color="blue"):
                poller.set_phase(tag)
                print(f"[phase] {tag} start  (passes/window={passes})")
                t0 = time.perf_counter()
                deadline = t0 + args.phase_seconds
                while time.perf_counter() < deadline:
                    w0 = time.perf_counter()
                    with stream:
                        kernel((blocks,), (threads,),
                               (buf, sink, np.uint64(n_f4), np.int32(passes)))
                    stream.synchronize()
                    if target < 100:
                        rest = window_s - (time.perf_counter() - w0)
                        if rest > 2e-4:
                            time.sleep(rest)
            with nvtx.annotate("gap", color="gray"):
                poller.set_phase("gap")
                time.sleep(0.5)
    finally:
        poller.stop()
        poller.join(timeout=1.0)

    # ---- convert % to GB/s using measured peak ----
    # pynvml memory util% = memory-controller busy fraction; scale by the
    # calibrated peak (one pass fully saturates the DRAM) to recover GB/s.
    def to_gbps(pct: float) -> float:
        return pct / 100.0 * peak_gbps

    # ---- save CSV ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"util_cupy_{gpu_slug}_{stamp}{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "mem_util_pct", "gpu_util_pct",
                    "bandwidth_gbps", "phase"])
        for t_s, mem_pct, gpu_pct, ph in poller.rows:
            w.writerow([t_s, mem_pct, gpu_pct, f"{to_gbps(mem_pct):.2f}", ph])
    print(f"[save] {csv_path}  ({len(poller.rows)} samples)")

    # ---- summary (GB/s) ----
    print()
    if known_peak is not None:
        print(f"peak DRAM read BW: achieved {peak_gbps:.1f} GB/s "
              f"/ theoretical {known_peak:.0f} GB/s "
              f"= {peak_gbps/known_peak*100:.1f}%")
    else:
        print(f"peak DRAM read BW (calibrated) = {peak_gbps:.1f} GB/s")
    print()
    hdr = f"{'phase':<10} {'target':>7} {'expected':>10} {'measured':>10} {'std':>8} {'n':>6}"
    print(hdr)
    print(f"{'':<10} {'(%)':>7} {'(GB/s)':>10} {'(GB/s)':>10} {'(GB/s)':>8}")
    print("-" * len(hdr))
    for target in args.targets:
        tag = f"util_{target}"
        vals = [to_gbps(r[1]) for r in poller.rows if r[3] == tag]
        expected = peak_gbps * target / 100.0
        if vals:
            a = np.asarray(vals, dtype=float)
            print(f"{tag:<10} {target:>7} {expected:>10.1f} "
                  f"{a.mean():>10.1f} {a.std():>8.1f} {len(a):>6}")
        else:
            print(f"{tag:<10} {target:>7} {expected:>10.1f} "
                  f"{'--':>10} {'--':>8} {0:>6}")

    # ---- plot (GB/s) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t  = [r[0] for r in poller.rows]
        bw = [to_gbps(r[1]) for r in poller.rows]
        t_max = max(t) if t else 1.0

        # Y-axis ceiling = larger of achieved and (known) theoretical peak
        y_top = max(peak_gbps, known_peak or 0) * 1.15
        y_norm = known_peak if known_peak else peak_gbps

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t, bw, lw=0.8, color="#1f77b4",
                label=f"measured (pynvml mem util × {peak_gbps:.0f} GB/s achieved)")

        # Theoretical peak reference (if known) + achieved peak reference.
        if known_peak is not None:
            ax.axhline(known_peak, color="green", ls="-", lw=1.0, alpha=0.6,
                       label=f"theoretical peak {known_peak:.0f} GB/s")
        ax.axhline(peak_gbps, color="orange", ls="-", lw=1.0, alpha=0.6,
                   label=f"achieved peak {peak_gbps:.0f} GB/s")

        # target reference lines + phase shading
        phase_ranges: dict[str, tuple[float, float]] = {}
        for r in poller.rows:
            ph = r[3]
            if ph.startswith("util_"):
                lo, hi = phase_ranges.get(ph, (r[0], r[0]))
                phase_ranges[ph] = (min(lo, r[0]), max(hi, r[0]))
        for ph, (lo, hi) in phase_ranges.items():
            tgt_pct  = int(ph.split("_")[1])
            tgt_gbps = peak_gbps * tgt_pct / 100.0
            ax.hlines(tgt_gbps, lo, hi, color="red", ls="--", lw=1.2)
            ax.text((lo + hi) / 2, tgt_gbps + y_top * 0.02,
                    f"{ph} → {tgt_gbps:.0f} GB/s",
                    ha="center", fontsize=9, color="red")

        ax.set_xlabel("time (s)")
        ax.set_ylabel("DRAM read bandwidth (GB/s)")
        ax.set_ylim(-y_top * 0.05, y_top)
        title_extra = ""
        if known_peak:
            title_extra = (f"  (achieved {peak_gbps:.0f} / theoretical "
                           f"{known_peak:.0f} = {peak_gbps/known_peak*100:.0f}%)")
        else:
            title_extra = f"  (peak {peak_gbps:.0f} GB/s)"
        ax.set_title(f"DRAM read bandwidth — {gpu_name}{title_extra}")
        # Right-side secondary axis in % of *theoretical* (or achieved, fallback).
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim()[0] / y_norm * 100.0,
                     ax.get_ylim()[1] / y_norm * 100.0)
        ax2.set_ylabel(f"% of {('theoretical' if known_peak else 'achieved')} peak")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        png = out_dir / f"util_cupy_{gpu_slug}_{stamp}{suffix}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        print(f"[save] {png}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
