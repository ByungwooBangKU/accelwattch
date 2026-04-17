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
    args = ap.parse_args()

    # ---- GPU info & buffer sizing ----
    dev = cp.cuda.Device(0)
    dev.use()
    p = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = prop(p, "name")
    sm_count = p["multiProcessorCount"]
    l2_bytes = p["l2CacheSize"]

    if args.buf_bytes is None:
        args.buf_bytes = max(1 << 30, l2_bytes * 64)
    n_f4 = args.buf_bytes // 16  # float4 = 16 bytes

    print(f"[info] GPU={gpu_name}  SMs={sm_count}  "
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

    # ---- calibrate ms/pass ----
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
    CAL = 4
    with stream:
        start.record(stream=stream)
        kernel((blocks,), (threads,),
               (buf, sink, np.uint64(n_f4), np.int32(CAL)))
        end.record(stream=stream)
    end.synchronize()
    cal_ms = cp.cuda.get_elapsed_time(start, end)
    ms_per_pass = cal_ms / CAL
    peak_gbps = args.buf_bytes / (ms_per_pass * 1e-3) / 1e9
    print(f"[calib] {ms_per_pass:.3f} ms/pass  ~{peak_gbps:.1f} GB/s peak DRAM read")

    # ---- pynvml poller ----
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
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

    # ---- save CSV ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"util_cupy_{stamp}{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "mem_util_pct", "gpu_util_pct", "phase"])
        w.writerows(poller.rows)
    print(f"[save] {csv_path}  ({len(poller.rows)} samples)")

    # ---- summary ----
    print()
    print(f"{'phase':<10} {'target%':>8} {'mean':>8} {'std':>8} {'n':>6}")
    print("-" * 48)
    for target in args.targets:
        tag = f"util_{target}"
        vals = [r[1] for r in poller.rows if r[3] == tag]
        if vals:
            a = np.asarray(vals, dtype=float)
            print(f"{tag:<10} {target:>8} {a.mean():>8.2f} {a.std():>8.2f} {len(a):>6}")
        else:
            print(f"{tag:<10} {target:>8} {'--':>8} {'--':>8} {0:>6}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t  = [r[0] for r in poller.rows]
        m  = [r[1] for r in poller.rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, m, lw=0.8, color="#1f77b4", label="DRAM (mem ctrl) util")
        # shade phase regions
        phase_ranges: dict[str, tuple[float, float]] = {}
        for r in poller.rows:
            ph = r[3]
            if ph.startswith("util_"):
                lo, hi = phase_ranges.get(ph, (r[0], r[0]))
                phase_ranges[ph] = (min(lo, r[0]), max(hi, r[0]))
        for ph, (lo, hi) in phase_ranges.items():
            tgt = int(ph.split("_")[1])
            ax.axhspan(tgt - 3, tgt + 3, xmin=lo / max(t), xmax=hi / max(t),
                       color="red", alpha=0.08)
            ax.hlines(tgt, lo, hi, color="red", ls=":", lw=1.0)
            ax.text((lo + hi) / 2, tgt + 4, ph, ha="center", fontsize=9,
                    color="red")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("utilization (%)")
        ax.set_ylim(-5, 110)
        ax.set_title(f"DRAM read utilization — {gpu_name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        png = out_dir / f"util_cupy_{stamp}{suffix}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        print(f"[save] {png}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
