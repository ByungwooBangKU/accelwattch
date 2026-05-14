#!/usr/bin/env python3
"""Summarize repeated dram_pjbit_cupy.py analysis CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def f_or_none(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def save_plot(rows: list[dict[str, str]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_rows = [
        r for r in rows
        if r["method"] in {"slope_avg_power_vs_bw", "100_minus_0_avg_power"}
        and r["pj_per_bit_mean"]
    ]
    if not plot_rows:
        return

    workloads: list[str] = []
    for row in plot_rows:
        if row["workload"] not in workloads:
            workloads.append(row["workload"])

    def row_for(workload: str, method: str) -> dict[str, str] | None:
        for row in plot_rows:
            if row["workload"] == workload and row["method"] == method:
                return row
        return None

    x = np.arange(len(workloads))
    width = 0.35
    methods = [
        ("slope_avg_power_vs_bw", "50/75/100 slope", "#f58518"),
        ("100_minus_0_avg_power", "100%-0%", "#4c78a8"),
    ]

    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(workloads) + 4), 6))
    for idx, (method, label, color) in enumerate(methods):
        vals = []
        errs = []
        for workload in workloads:
            row = row_for(workload, method)
            vals.append(float(row["pj_per_bit_mean"]) if row else float("nan"))
            errs.append(float(row["pj_per_bit_std"]) if row and row["pj_per_bit_std"] else 0.0)
        xpos = x + (idx - 0.5) * width
        bars = ax.bar(xpos, vals, width, yerr=errs, capsize=4, label=label,
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val != val:
                continue
            ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2.0, val),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", va="bottom", fontsize=8)

    ax.set_title("Repeat summary pJ/bit")
    ax.set_ylabel("pJ/bit mean across repeats")
    ax.set_xticks(x, workloads, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.text(0.01, 0.01,
             "Error bars are sample standard deviation across repeat runs. "
             "Slope uses measured bandwidth, not requested target labels.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+",
                    help="analysis CSV path or glob, e.g. reports/*_rep*_analysis.csv")
    ap.add_argument("--out", default="",
                    help="optional output CSV path")
    ap.add_argument("--plot-out", default="",
                    help="optional output PNG path for repeat summary")
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            paths.extend(Path(p) for p in hits)
        else:
            paths.append(Path(pattern))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no analysis CSV files matched")

    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_file"] = str(path)
                mode = row["mode"]
                pattern = row.get("pattern", "")
                workload = row.get("workload") or (
                    "read" if mode == "read" else f"write:{pattern}"
                )
                groups[(
                    workload, mode, pattern, row["method"], row.get("target_points", "")
                )].append(row)

    out_rows: list[dict[str, str]] = []
    for (workload, mode, pattern, method, target_points), rows in sorted(groups.items()):
        pj = [x for r in rows if (x := f_or_none(r.get("pj_per_bit", ""))) is not None]
        r2 = [x for r in rows if (x := f_or_none(r.get("r2", ""))) is not None]
        resid = [
            x for r in rows
            if (x := f_or_none(r.get("max_abs_residual_w", ""))) is not None
        ]
        out_rows.append({
            "workload": workload,
            "mode": mode,
            "pattern": pattern,
            "method": method,
            "target_points": target_points,
            "runs": str(len(rows)),
            "pj_per_bit_mean": f"{statistics.fmean(pj):.6f}" if pj else "",
            "pj_per_bit_std": (
                f"{statistics.stdev(pj):.6f}" if len(pj) > 1 else "0.000000"
            ) if pj else "",
            "r2_mean": f"{statistics.fmean(r2):.6f}" if r2 else "",
            "max_abs_residual_w_mean": f"{statistics.fmean(resid):.6f}" if resid else "",
            "files": ";".join(r["_file"] for r in rows),
        })

    fields = [
        "workload", "mode", "pattern", "method", "target_points", "runs",
        "pj_per_bit_mean", "pj_per_bit_std", "r2_mean",
        "max_abs_residual_w_mean", "files",
    ]

    stdout = csv.DictWriter(sys.stdout, fieldnames=fields)
    stdout.writeheader()
    stdout.writerows(out_rows)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f"[save] {out}")

    if args.plot_out:
        plot_out = Path(args.plot_out)
        save_plot(out_rows, plot_out)
        print(f"[save] {plot_out}")


if __name__ == "__main__":
    main()
