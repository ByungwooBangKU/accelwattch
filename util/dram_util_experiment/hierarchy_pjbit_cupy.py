#!/usr/bin/env python3
"""Memory-hierarchy lower-bound experiment for GPU data movement energy.

This experiment complements dram_pjbit_cupy.py.  Instead of reporting only
board-level dynamic pJ per user-visible DRAM bit, it separates matched control,
L2-resident, and DRAM-streaming phases:

    control_l2    address/pattern loop only, normalized to the L2 buffer size
    l2            same loop shape on a buffer intended to stay resident in L2
    control_dram  address/pattern loop only, normalized to the DRAM buffer size
    dram          same loop shape on a buffer much larger than L2

The result is still not a DRAM-rail-only number.  It is a lower-bound style
microbenchmark decomposition of NVML GPU/board dynamic power, designed to be
cross-checked with Nsight Compute physical DRAM/L2 counters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nvtx

import dram_pjbit_cupy as base

cp = base.cp
plt = base.plt
pynvml = base.pynvml


KERNEL_CODE = r"""
__device__ __forceinline__
unsigned int mix32(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__device__ __forceinline__
uint4 consume_write_value(uint4 acc, uint4 v, unsigned long long i) {
    unsigned int lo = (unsigned int)i;
    unsigned int hi = (unsigned int)(i >> 32);
    acc.x ^= v.x ^ lo;
    acc.y += v.y ^ hi;
    acc.z ^= v.z + (lo * 0x9e3779b9U);
    acc.w += v.w ^ (hi * 0x85ebca6bU);
    return acc;
}

__device__ __forceinline__
uint4 write_pattern_value(unsigned long long i, int p, int pattern) {
    unsigned int lo = (unsigned int)i;
    unsigned int hi = (unsigned int)(i >> 32);
    unsigned int seed = lo ^ (hi * 0x9e3779b9U) ^ ((unsigned int)p * 0x85ebca6bU);

    if (pattern == 0) {
        return make_uint4(0U, 0U, 0U, 0U);
    }
    if (pattern == 1) {
        float x = (float)((p & 255) + 1);
        return make_uint4(__float_as_uint(x), __float_as_uint(x + 1.f),
                          __float_as_uint(x + 2.f), __float_as_uint(x + 3.f));
    }
    if (pattern == 2) {
        unsigned int a = lo ^ (hi * 0x9e3779b9U) ^ ((unsigned int)p * 0x10001U);
        return make_uint4(a, a + 0x3c6ef372U, a + 0xdaa66d2bU, a + 0x78dde6e4U);
    }
    if (pattern == 3) {
        unsigned int a = mix32(seed);
        unsigned int b = mix32(seed ^ 0x9e3779b9U);
        unsigned int c = mix32(seed ^ 0x7f4a7c15U);
        unsigned int d = mix32(seed ^ 0x94d049bbU);
        return make_uint4(a, b, c, d);
    }

    unsigned int a = ((lo + (unsigned int)p) & 1U) ? 0xffffffffU : 0U;
    return make_uint4(a, ~a, a ^ 0xaaaaaaaaU, a ^ 0x55555555U);
}

extern "C" __global__
void hierarchy_control_read(uint4* __restrict__ sink,
                            unsigned long long n, int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);
    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            unsigned int lo = (unsigned int)i;
            unsigned int hi = (unsigned int)(i >> 32);
            acc.x ^= lo + (unsigned int)p;
            acc.y ^= hi ^ 0x9e3779b9U;
            acc.z += lo ^ 0x7f4a7c15U;
            acc.w += hi + 0x94d049bbU;
        }
    }
    sink[tid & 1023U] = acc;
}

extern "C" __global__
void hierarchy_read(const uint4* __restrict__ in,
                    uint4* __restrict__ sink,
                    unsigned long long n, int passes) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);
    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            uint4 v;
            const unsigned int* ptr = reinterpret_cast<const unsigned int*>(in + i);
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(ptr));
            acc.x ^= v.x;
            acc.y ^= v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
    }
    sink[tid & 1023U] = acc;
}

extern "C" __global__
void hierarchy_control_write(uint4* __restrict__ sink,
                             unsigned long long n, int passes, int pattern) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);
    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            uint4 v = write_pattern_value(i, p, pattern);
            acc = consume_write_value(acc, v, i);
        }
    }
    sink[tid & 1023U] = acc;
}

extern "C" __global__
void hierarchy_write(uint4* __restrict__ out,
                     uint4* __restrict__ sink,
                     unsigned long long n, int passes, int pattern) {
    unsigned long long tid    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0U, 0U, 0U, 0U);
    for (int p = 0; p < passes; ++p) {
        for (unsigned long long i = tid; i < n; i += stride) {
            uint4 v = write_pattern_value(i, p, pattern);
            unsigned int* ptr = reinterpret_cast<unsigned int*>(out + i);
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                         :
                         : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
            acc = consume_write_value(acc, v, i);
        }
    }
    sink[tid & 1023U] = acc;
}
"""


WRITE_PATTERN_CODES = {
    "zero": 0,
    "const": 1,
    "address": 2,
    "random": 3,
    "toggle": 4,
}


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    mode: str
    pattern: str
    stage: str
    buffer_kind: str
    does_global_access: bool
    kernel_name: str
    n_u4: int
    buf_bytes: int
    pattern_code: int


def pj_per_bit_from_power(power_w: float, bandwidth_gbs: float) -> float:
    if not math.isfinite(power_w) or not math.isfinite(bandwidth_gbs) or bandwidth_gbs <= 0:
        return float("nan")
    return power_w / (bandwidth_gbs * 1e9 * 8.0) * 1e12


def align16(v: int) -> int:
    return max(16, (int(v) // 16) * 16)


def launch_workload(spec: WorkloadSpec, kernels: dict[str, object], stream,
                    blocks: int, threads: int, buffers: dict[str, object],
                    sink, passes: int) -> None:
    kernel = kernels[spec.kernel_name]
    with stream:
        if spec.kernel_name == "hierarchy_control_read":
            kernel((blocks,), (threads,), (sink, np.uint64(spec.n_u4), np.int32(passes)))
        elif spec.kernel_name == "hierarchy_read":
            kernel((blocks,), (threads,),
                   (buffers[spec.buffer_kind], sink, np.uint64(spec.n_u4), np.int32(passes)))
        elif spec.kernel_name == "hierarchy_control_write":
            kernel((blocks,), (threads,),
                   (sink, np.uint64(spec.n_u4), np.int32(passes), np.int32(spec.pattern_code)))
        elif spec.kernel_name == "hierarchy_write":
            kernel((blocks,), (threads,),
                   (buffers[spec.buffer_kind], sink, np.uint64(spec.n_u4),
                    np.int32(passes), np.int32(spec.pattern_code)))
        else:
            raise ValueError(f"unknown kernel: {spec.kernel_name}")


def calibrate(spec: WorkloadSpec, kernels: dict[str, object], stream,
              blocks: int, threads: int, buffers: dict[str, object], sink,
              cal_passes: int, repeats: int) -> tuple[float, float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    best_ms = float("inf")
    for _ in range(repeats):
        start.record(stream=stream)
        launch_workload(spec, kernels, stream, blocks, threads, buffers, sink, cal_passes)
        end.record(stream=stream)
        end.synchronize()
        best_ms = min(best_ms, cp.cuda.get_elapsed_time(start, end) / cal_passes)
    return best_ms, 1.0 / (best_ms * 1e-3)


def run_phase(spec: WorkloadSpec, kernels: dict[str, object], stream,
              blocks: int, threads: int, buffers: dict[str, object], sink,
              phase_seconds: float, launch_ms: float, ms_per_pass: float,
              poller: base.PowerPoller, idle_power_w: float) -> dict:
    passes = max(1, int(round(launch_ms / max(ms_per_pass, 1e-9))))
    launches = 0

    with nvtx.annotate(spec.name, color="orange" if spec.mode == "write" else "green"):
        poller.set_phase(spec.name)
        t0_abs = time.perf_counter()
        t0 = t0_abs - poller.t0
        deadline = t0_abs + phase_seconds
        while time.perf_counter() < deadline:
            launch_workload(spec, kernels, stream, blocks, threads, buffers, sink, passes)
            stream.synchronize()
            launches += 1
        t1 = time.perf_counter() - poller.t0

    wall_s = max(t1 - t0, 1e-12)
    nominal_bytes = int(launches * passes * spec.buf_bytes)
    bandwidth_gbs = nominal_bytes / wall_s / 1e9
    total_energy_j = poller.energy_j(t0, t1)
    idle_energy_j = idle_power_w * wall_s
    dynamic_energy_j = max(0.0, total_energy_j - idle_energy_j)
    dynamic_power_w = dynamic_energy_j / wall_s
    samples = poller.slice(t0, t1)
    powers = [s.power_w for s in samples if s.power_w >= 0]
    sm_clocks = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz >= 0]
    mem_clocks = [s.mem_clock_mhz for s in samples if s.mem_clock_mhz >= 0]
    gpu_utils = [s.gpu_util_pct for s in samples if s.gpu_util_pct >= 0]
    mem_utils = [s.mem_util_pct for s in samples if s.mem_util_pct >= 0]

    return {
        "name": spec.name,
        "mode": spec.mode,
        "pattern": spec.pattern,
        "stage": spec.stage,
        "buffer_kind": spec.buffer_kind,
        "does_global_access": int(spec.does_global_access),
        "kernel_name": spec.kernel_name,
        "buffer_bytes": spec.buf_bytes,
        "n_uint4": spec.n_u4,
        "passes_per_launch": passes,
        "launches": launches,
        "t0_s": t0,
        "t1_s": t1,
        "wall_s": wall_s,
        "nominal_bytes": nominal_bytes,
        "nominal_bandwidth_gbs": bandwidth_gbs,
        "total_energy_j": total_energy_j,
        "idle_energy_j": idle_energy_j,
        "dynamic_energy_j": dynamic_energy_j,
        "avg_power_w": total_energy_j / wall_s if total_energy_j > 0 else float("nan"),
        "dynamic_power_w": dynamic_power_w,
        "pj_per_nominal_bit": (
            dynamic_energy_j / (nominal_bytes * 8.0) * 1e12
            if nominal_bytes > 0 else float("nan")
        ),
        "samples": len(samples),
        "power_std_w": statistics.pstdev(powers) if len(powers) > 1 else 0.0,
        "sm_clock_mhz_mean": base.mean_or_nan(sm_clocks),
        "mem_clock_mhz_mean": base.mean_or_nan(mem_clocks),
        "gpu_util_pct_mean": base.mean_or_nan(gpu_utils),
        "mem_util_pct_mean": base.mean_or_nan(mem_utils),
    }


def build_workloads(modes: list[str], patterns: list[str],
                    l2_bytes: int, dram_bytes: int) -> list[WorkloadSpec]:
    l2_n = l2_bytes // 16
    dram_n = dram_bytes // 16
    specs: list[WorkloadSpec] = []

    if "read" in modes:
        specs.extend([
            WorkloadSpec("control_l2_read", "read", "none", "control_l2", "l2", False,
                         "hierarchy_control_read", l2_n, l2_bytes, 0),
            WorkloadSpec("l2_read", "read", "none", "l2", "l2", True,
                         "hierarchy_read", l2_n, l2_bytes, 0),
            WorkloadSpec("control_dram_read", "read", "none", "control_dram", "dram",
                         False, "hierarchy_control_read", dram_n, dram_bytes, 0),
            WorkloadSpec("dram_read", "read", "none", "dram", "dram", True,
                         "hierarchy_read", dram_n, dram_bytes, 0),
        ])

    if "write" in modes:
        for pattern in patterns:
            code = WRITE_PATTERN_CODES[pattern]
            suffix = f"write_{pattern}"
            specs.extend([
                WorkloadSpec(f"control_l2_{suffix}", "write", pattern, "control_l2",
                             "l2", False, "hierarchy_control_write", l2_n, l2_bytes, code),
                WorkloadSpec(f"l2_{suffix}", "write", pattern, "l2", "l2", True,
                             "hierarchy_write", l2_n, l2_bytes, code),
                WorkloadSpec(f"control_dram_{suffix}", "write", pattern, "control_dram",
                             "dram", False, "hierarchy_control_write", dram_n,
                             dram_bytes, code),
                WorkloadSpec(f"dram_{suffix}", "write", pattern, "dram", "dram", True,
                             "hierarchy_write", dram_n, dram_bytes, code),
            ])
    return specs


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_trace(path: Path, samples: list[base.PowerSample]) -> None:
    fieldnames = [
        "t_s", "power_w", "power_instant_w", "power_average_w",
        "gpu_util_pct", "mem_util_pct", "sm_clock_mhz", "mem_clock_mhz",
        "temp_gpu_c", "pstate", "phase",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: getattr(s, k) for k in fieldnames})


def make_analysis_rows(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["pattern"]))
        by_key.setdefault(key, {})[str(row["stage"])] = row

    out: list[dict] = []
    for (mode, pattern), stages in by_key.items():
        ctrl_l2 = stages.get("control_l2")
        l2 = stages.get("l2")
        ctrl_dram = stages.get("control_dram")
        dram = stages.get("dram")
        if not (ctrl_l2 and l2 and ctrl_dram and dram):
            continue

        l2_power_delta = l2["dynamic_power_w"] - ctrl_l2["dynamic_power_w"]
        dram_power_delta = dram["dynamic_power_w"] - ctrl_dram["dynamic_power_w"]
        l2_pj_delta = pj_per_bit_from_power(l2_power_delta, l2["nominal_bandwidth_gbs"])
        dram_pj_delta = pj_per_bit_from_power(
            dram_power_delta, dram["nominal_bandwidth_gbs"])
        l2_pj_delta_clamped = max(0.0, l2_pj_delta) if math.isfinite(l2_pj_delta) else l2_pj_delta
        dram_over_l2 = dram_pj_delta - l2_pj_delta
        dram_over_l2_clamped = (
            dram_pj_delta - l2_pj_delta_clamped
            if math.isfinite(dram_pj_delta) and math.isfinite(l2_pj_delta_clamped)
            else float("nan")
        )
        out.append({
            "mode": mode,
            "pattern": pattern,
            "control_l2_pj_per_nominal_bit": ctrl_l2["pj_per_nominal_bit"],
            "l2_pj_per_nominal_bit": l2["pj_per_nominal_bit"],
            "control_dram_pj_per_nominal_bit": ctrl_dram["pj_per_nominal_bit"],
            "dram_pj_per_nominal_bit": dram["pj_per_nominal_bit"],
            "l2_minus_control_power_w": l2_power_delta,
            "dram_minus_control_power_w": dram_power_delta,
            "l2_minus_control_pj_per_nominal_bit": l2_pj_delta,
            "l2_minus_control_clamped_pj_per_nominal_bit": l2_pj_delta_clamped,
            "dram_minus_control_pj_per_nominal_bit": dram_pj_delta,
            "dram_over_l2_pj_per_nominal_bit": dram_over_l2,
            "dram_over_l2_clamped_pj_per_nominal_bit": dram_over_l2_clamped,
            "dram_nominal_bandwidth_gbs": dram["nominal_bandwidth_gbs"],
            "l2_nominal_bandwidth_gbs": l2["nominal_bandwidth_gbs"],
            "notes": (
                "Power deltas are board-level dynamic lower-bound estimates. "
                "Use clamped columns when L2-control is negative from over-subtraction/noise. "
                "Use NCU physical DRAM/L2 bytes for final denominator checks."
            ),
        })
    return out


def stage_color(stage: str) -> str:
    return {
        "control_l2": "#8c8c8c",
        "l2": "#4c78a8",
        "control_dram": "#bab0ac",
        "dram": "#e45756",
    }.get(stage, "#6f6f6f")


def short_row_label(row: dict) -> str:
    stage_label = {
        "control_l2": "ctrl L2",
        "l2": "L2",
        "control_dram": "ctrl DRAM",
        "dram": "DRAM",
    }.get(str(row["stage"]), str(row["stage"]))
    if row["mode"] == "read":
        return f"read\n{stage_label}"
    return f"write:{row['pattern']}\n{stage_label}"


def analysis_label(row: dict) -> str:
    return row["mode"] if row["mode"] == "read" else f"write:{row['pattern']}"


def save_plot(path: Path, gpu_name: str, rows: list[dict], analysis_rows: list[dict]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"Memory hierarchy lower-bound experiment - {gpu_name}", fontsize=13)

    labels = [r["name"] for r in rows]
    x = np.arange(len(labels))
    colors = [
        "#8c8c8c" if "control" in r["stage"] else
        "#4c78a8" if r["stage"] == "l2" else "#e45756"
        for r in rows
    ]

    axes[0, 0].bar(x, [r["dynamic_power_w"] for r in rows], color=colors)
    axes[0, 0].set_title("Dynamic power by phase")
    axes[0, 0].set_ylabel("W above idle")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, [r["pj_per_nominal_bit"] for r in rows], color=colors)
    axes[0, 1].set_title("Board dynamic pJ / nominal bit")
    axes[0, 1].set_ylabel("pJ/bit")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(x, [r["nominal_bandwidth_gbs"] for r in rows], color=colors)
    axes[1, 0].set_title("Nominal throughput")
    axes[1, 0].set_ylabel("GB/s")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    if analysis_rows:
        a_labels = [
            r["mode"] if r["mode"] == "read" else f"write:{r['pattern']}"
            for r in analysis_rows
        ]
        ax = axes[1, 1]
        ax.bar(np.arange(len(a_labels)) - 0.2,
               [r["l2_minus_control_pj_per_nominal_bit"] for r in analysis_rows],
               width=0.4, label="L2 - control", color="#4c78a8")
        ax.bar(np.arange(len(a_labels)) + 0.2,
               [r["dram_minus_control_pj_per_nominal_bit"] for r in analysis_rows],
               width=0.4, label="DRAM - control", color="#e45756")
        ax.set_title("Control-subtracted estimates")
        ax.set_ylabel("pJ/nominal bit")
        ax.set_xticks(np.arange(len(a_labels)))
        ax.set_xticklabels(a_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    else:
        axes[1, 1].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_power_trace_plot(path: Path, gpu_name: str, idle_power_w: float,
                          rows: list[dict], samples: list[base.PowerSample]) -> None:
    if not samples:
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    power_samples = [s for s in samples if s.power_w >= 0]
    t = [s.t_s for s in power_samples]
    p = [s.power_w for s in power_samples]
    ax0.plot(t, p, lw=0.8, color="#1f77b4", label="NVML total power")
    ax0.axhline(idle_power_w, color="black", ls="--", lw=1.0,
                label=f"idle {idle_power_w:.1f} W")
    ymax = max(p, default=idle_power_w)
    for row in rows:
        color = stage_color(str(row["stage"]))
        ax0.axvspan(row["t0_s"], row["t1_s"], color=color, alpha=0.12)
        ax0.text((row["t0_s"] + row["t1_s"]) / 2, ymax,
                 str(row["name"]), ha="center", va="bottom",
                 fontsize=7, rotation=75)
    ax0.set_title(f"Power trace by hierarchy phase - {gpu_name}")
    ax0.set_ylabel("power (W)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left")

    util_samples = [s for s in samples if s.gpu_util_pct >= 0 or s.mem_util_pct >= 0]
    ax1.plot([s.t_s for s in util_samples], [max(s.gpu_util_pct, 0) for s in util_samples],
             lw=0.9, color="#54a24b", label="GPU util")
    ax1.plot([s.t_s for s in util_samples], [max(s.mem_util_pct, 0) for s in util_samples],
             lw=0.9, color="#f58518", label="memory util")
    for row in rows:
        ax1.axvspan(row["t0_s"], row["t1_s"], color=stage_color(str(row["stage"])),
                    alpha=0.08)
    ax1.set_xlabel("time since poller start (s)")
    ax1.set_ylabel("utilization (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_decomposition_plot(path: Path, gpu_name: str,
                            rows: list[dict], analysis_rows: list[dict]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    labels = [short_row_label(r) for r in rows]
    x = np.arange(len(rows))
    colors = [stage_color(str(r["stage"])) for r in rows]

    axes[0, 0].bar(x, [r["dynamic_power_w"] for r in rows], color=colors)
    axes[0, 0].set_title("Dynamic power by matched phase")
    axes[0, 0].set_ylabel("W above idle")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, [r["pj_per_nominal_bit"] for r in rows], color=colors)
    axes[0, 1].set_title("Raw board dynamic pJ / nominal bit")
    axes[0, 1].set_ylabel("pJ/nominal bit")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.25)

    if analysis_rows:
        a_labels = [analysis_label(r) for r in analysis_rows]
        ax = axes[1, 0]
        ax.bar(np.arange(len(a_labels)) - 0.25,
               [r["l2_minus_control_pj_per_nominal_bit"] for r in analysis_rows],
               width=0.25, label="L2 - ctrl L2", color="#4c78a8")
        ax.bar(np.arange(len(a_labels)),
               [r["dram_minus_control_pj_per_nominal_bit"] for r in analysis_rows],
               width=0.25, label="DRAM - ctrl DRAM", color="#e45756")
        ax.bar(np.arange(len(a_labels)) + 0.25,
               [r["dram_over_l2_clamped_pj_per_nominal_bit"] for r in analysis_rows],
               width=0.25, label="DRAM over max(L2, 0)", color="#b279a2")
        ax.set_title("Control-subtracted lower-bound estimates")
        ax.set_ylabel("pJ/nominal bit")
        ax.set_xticks(np.arange(len(a_labels)))
        ax.set_xticklabels(a_labels, rotation=45, ha="right")
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, 1]
        ax.scatter(
            [r["l2_nominal_bandwidth_gbs"] for r in analysis_rows],
            [r["l2_minus_control_power_w"] for r in analysis_rows],
            s=70, color="#4c78a8", label="L2 - control")
        ax.scatter(
            [r["dram_nominal_bandwidth_gbs"] for r in analysis_rows],
            [r["dram_minus_control_power_w"] for r in analysis_rows],
            s=70, color="#e45756", label="DRAM - control")
        for r in analysis_rows:
            label = analysis_label(r)
            ax.annotate(label, (r["dram_nominal_bandwidth_gbs"],
                                r["dram_minus_control_power_w"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_title("Control-subtracted power vs nominal bandwidth")
        ax.set_xlabel("nominal bandwidth (GB/s)")
        ax.set_ylabel("W above matched control")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.suptitle(
        f"Hierarchy energy decomposition - {gpu_name}\n"
        "NVML GPU/board dynamic lower-bound, not DRAM-rail-only.",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bandwidth_plot(path: Path, gpu_name: str, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [short_row_label(r) for r in rows]
    x = np.arange(len(rows))
    colors = [stage_color(str(r["stage"])) for r in rows]
    fig, ax = plt.subplots(figsize=(max(13, min(22, 0.65 * len(rows) + 8)), 6))
    bars = ax.bar(x, [r["nominal_bandwidth_gbs"] for r in rows], color=colors)
    for bar, row in zip(bars, rows):
        bw = row["nominal_bandwidth_gbs"]
        ax.annotate(f"{bw:.0f}", (bar.get_x() + bar.get_width() / 2, bw),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Nominal bandwidth used for pJ/bit denominator - {gpu_name}")
    ax.set_ylabel("GB/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, 0.02,
             "control phases use loop-equivalent nominal bytes; NCU physical bytes "
             "should be used for final cache/DRAM denominator validation.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", choices=["read", "write"],
                    default=["read", "write"])
    ap.add_argument("--write-patterns", nargs="+", choices=sorted(WRITE_PATTERN_CODES),
                    default=["zero", "const", "address", "toggle"],
                    help="random is supported but treated as compute-mixed")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dram-buf-bytes", type=int, default=None,
                    help="default: max(1 GiB, 64 * L2)")
    ap.add_argument("--l2-buf-bytes", type=int, default=None,
                    help="default: 25%% of L2, at least 4 MiB")
    ap.add_argument("--l2-fraction", type=float, default=0.25)
    ap.add_argument("--phase-seconds", type=float, default=12.0)
    ap.add_argument("--idle-seconds", type=float, default=15.0)
    ap.add_argument("--poll-hz", type=int, default=100)
    ap.add_argument("--gap-seconds", type=float, default=1.0)
    ap.add_argument("--launch-ms", type=float, default=50.0,
                    help="target milliseconds per kernel launch")
    ap.add_argument("--cal-passes", type=int, default=4)
    ap.add_argument("--cal-repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=None,
                    help="override total blocks; default SMs * --blocks-per-sm")
    ap.add_argument("--blocks-per-sm", type=int, default=32)
    ap.add_argument("--only-workload", default="",
                    help="run one workload name, useful under Nsight Compute")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--flat-output", action="store_true")
    ap.add_argument("--tag", default="hierarchy")
    args = ap.parse_args()

    if args.poll_hz <= 0:
        raise SystemExit("--poll-hz must be positive")
    if args.l2_fraction <= 0 or args.l2_fraction >= 1.0:
        raise SystemExit("--l2-fraction must be in (0, 1)")

    args.write_patterns = list(dict.fromkeys(args.write_patterns))
    cp.cuda.Device(args.device).use()
    props = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = base.prop(props, "name")
    run_stamp_minute = time.strftime("%Y%m%d%H%M")
    sm_count = int(props["multiProcessorCount"])
    l2_size_bytes = int(props["l2CacheSize"])

    if args.dram_buf_bytes is None:
        args.dram_buf_bytes = max(1 << 30, l2_size_bytes * 64)
    args.dram_buf_bytes = align16(args.dram_buf_bytes)
    if args.l2_buf_bytes is None:
        args.l2_buf_bytes = max(4 << 20, int(l2_size_bytes * args.l2_fraction))
    args.l2_buf_bytes = min(align16(args.l2_buf_bytes), args.dram_buf_bytes)

    blocks = args.blocks if args.blocks is not None else sm_count * args.blocks_per_sm
    workloads = build_workloads(
        args.modes, args.write_patterns, args.l2_buf_bytes, args.dram_buf_bytes)
    if args.only_workload:
        workloads = [w for w in workloads if w.name == args.only_workload]
        if not workloads:
            valid = ", ".join(w.name for w in build_workloads(
                args.modes, args.write_patterns, args.l2_buf_bytes, args.dram_buf_bytes))
            raise SystemExit(f"unknown --only-workload {args.only_workload}; valid: {valid}")

    print(f"[info] GPU={gpu_name} SMs={sm_count} L2={l2_size_bytes/(1<<20):.1f} MiB")
    print(f"[info] buffers l2={args.l2_buf_bytes/(1<<20):.2f} MiB "
          f"dram={args.dram_buf_bytes/(1<<30):.2f} GiB")
    print(f"[info] blocks={blocks} threads={args.threads} workloads={len(workloads)}")
    if base._nvrtc_path:
        print(f"[info] preloaded nvrtc: {base._nvrtc_path}")

    dram_buf = cp.empty(args.dram_buf_bytes // 4, dtype=cp.uint32)
    l2_buf = cp.empty(args.l2_buf_bytes // 4, dtype=cp.uint32)
    sink = cp.empty(1024 * 4, dtype=cp.uint32)
    dram_buf.fill(np.uint32(0x3c6ef372))
    l2_buf.fill(np.uint32(0x9e3779b9))
    sink.fill(np.uint32(0))
    buffers = {"dram": dram_buf, "l2": l2_buf}

    module = cp.RawModule(code=KERNEL_CODE, options=("--std=c++14",))
    kernels = {
        name: module.get_function(name)
        for name in [
            "hierarchy_control_read",
            "hierarchy_read",
            "hierarchy_control_write",
            "hierarchy_write",
        ]
    }
    stream = cp.cuda.Stream(non_blocking=True)

    # Warmup and L2 population are outside measured phases.
    for spec in workloads:
        launch_workload(spec, kernels, stream, blocks, args.threads, buffers, sink, 1)
    stream.synchronize()

    calibration: dict[str, dict[str, float]] = {}
    for spec in workloads:
        ms_per_pass, passes_per_s = calibrate(
            spec, kernels, stream, blocks, args.threads, buffers, sink,
            args.cal_passes, args.cal_repeats)
        calibration[spec.name] = {
            "ms_per_pass": ms_per_pass,
            "passes_per_second": passes_per_s,
            "nominal_peak_bandwidth_gbs": spec.buf_bytes * passes_per_s / 1e9,
        }
        print(f"[calib] {spec.name:<28} {ms_per_pass:.3f} ms/pass  "
              f"~{calibration[spec.name]['nominal_peak_bandwidth_gbs']:.1f} GB/s")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    metadata_before = base.nvml_snapshot(handle)
    print(f"[idle] measuring baseline for {args.idle_seconds:.1f} s ...", flush=True)
    idle_power_w, idle_std_w, idle_n = base.measure_idle_power(
        handle, args.idle_seconds, args.poll_hz)
    print(f"[idle] {idle_power_w:.2f} W +/- {idle_std_w:.2f} W n={idle_n}", flush=True)

    poller = base.PowerPoller(handle, args.poll_hz)
    rows: list[dict] = []
    metadata_after: dict = {}
    poller.start()
    try:
        for spec in workloads:
            print(f"[phase] {spec.name} start")
            row = run_phase(
                spec, kernels, stream, blocks, args.threads, buffers, sink,
                args.phase_seconds, args.launch_ms,
                calibration[spec.name]["ms_per_pass"], poller, idle_power_w)
            rows.append(row)
            print(f"[phase] {spec.name:<28} BW={row['nominal_bandwidth_gbs']:.1f} GB/s "
                  f"Pdyn={row['dynamic_power_w']:.1f} W "
                  f"pJ/nominal-bit={base.fmt_or_na(row['pj_per_nominal_bit'])}")
            with nvtx.annotate("gap", color="gray"):
                poller.set_phase("gap")
                time.sleep(max(0.0, args.gap_seconds))
    finally:
        poller.stop()
        metadata_after = base.nvml_snapshot(handle)
        pynvml.nvmlShutdown()

    out_dir = base.resolve_output_dir(
        args.out_dir, gpu_name, args.flat_output, run_stamp_minute)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"hierarchy_pjbit_{base.safe_name(gpu_name)}_{stamp}{suffix}"
    summary_csv = out_dir / f"{stem}_summary.csv"
    trace_csv = out_dir / f"{stem}_trace.csv"
    analysis_csv = out_dir / f"{stem}_analysis.csv"
    metadata_json = out_dir / f"{stem}_metadata.json"
    png = out_dir / f"{stem}.png"
    power_trace_png = out_dir / f"{stem}_power_trace.png"
    decomposition_png = out_dir / f"{stem}_decomposition.png"
    bandwidth_png = out_dir / f"{stem}_bandwidth.png"

    analysis_rows = make_analysis_rows(rows)
    save_rows(summary_csv, rows)
    save_trace(trace_csv, poller.samples)
    save_rows(analysis_csv, analysis_rows)
    save_plot(png, gpu_name, rows, analysis_rows)
    save_power_trace_plot(power_trace_png, gpu_name, idle_power_w, rows, poller.samples)
    save_decomposition_plot(decomposition_png, gpu_name, rows, analysis_rows)
    save_bandwidth_plot(bandwidth_png, gpu_name, rows)

    metadata = {
        "args": vars(args),
        "device": {
            "name": gpu_name,
            "sm_count": sm_count,
            "l2_bytes": l2_size_bytes,
        },
        "buffers": {
            "l2_buf_bytes": args.l2_buf_bytes,
            "dram_buf_bytes": args.dram_buf_bytes,
            "l2_fraction": args.l2_fraction,
        },
        "blocks": blocks,
        "threads": args.threads,
        "idle": {
            "power_w_mean": idle_power_w,
            "power_w_std": idle_std_w,
            "samples": idle_n,
        },
        "calibration": calibration,
        "nvml_before": metadata_before,
        "nvml_after": metadata_after,
        "plots": {
            "overview": str(png),
            "power_trace": str(power_trace_png),
            "decomposition": str(decomposition_png),
            "bandwidth": str(bandwidth_png),
        },
        "notes": [
            "This is not DRAM rail-only energy.",
            "control phases normalize loop/address/pattern overhead to the same nominal bytes.",
            "l2 phases are intended to be L2-resident, but writeback behavior must be checked by NCU.",
            "dram phases use a buffer much larger than L2.",
            "Use NCU physical DRAM/L2 bytes to validate denominator and cache residency.",
            "write:random includes pattern-generation ALU and should be treated as compute-mixed.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"[save] {summary_csv}")
    print(f"[save] {analysis_csv}")
    print(f"[save] {trace_csv}")
    print(f"[save] {metadata_json}")
    print(f"[save] {png}")
    print(f"[save] {power_trace_png}")
    print(f"[save] {decomposition_png}")
    print(f"[save] {bandwidth_png}")
    print()
    print(f"{'name':<30} {'BW(GB/s)':>10} {'Pdyn(W)':>10} {'pJ/bit':>10}")
    print("-" * 66)
    for row in rows:
        print(f"{row['name']:<30} {row['nominal_bandwidth_gbs']:>10.1f} "
              f"{row['dynamic_power_w']:>10.1f} "
              f"{base.fmt_or_na(row['pj_per_nominal_bit']):>10}")
    print()
    print(f"{'mode/pattern':<18} {'L2-control':>12} {'DRAM-control':>14} "
          f"{'DRAM-over-L2':>14} {'clamped':>10}")
    print("-" * 73)
    for row in analysis_rows:
        label = row["mode"] if row["mode"] == "read" else f"write:{row['pattern']}"
        print(f"{label:<18} "
              f"{base.fmt_or_na(row['l2_minus_control_pj_per_nominal_bit']):>12} "
              f"{base.fmt_or_na(row['dram_minus_control_pj_per_nominal_bit']):>14} "
              f"{base.fmt_or_na(row['dram_over_l2_pj_per_nominal_bit']):>14} "
              f"{base.fmt_or_na(row['dram_over_l2_clamped_pj_per_nominal_bit']):>10}")


if __name__ == "__main__":
    main()
