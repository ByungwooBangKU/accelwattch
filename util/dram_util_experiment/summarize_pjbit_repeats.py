#!/usr/bin/env python3
"""Summarize repeated dram_pjbit_cupy.py analysis CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def f_or_none(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+",
                    help="analysis CSV path or glob, e.g. reports/*_rep*_analysis.csv")
    ap.add_argument("--out", default="",
                    help="optional output CSV path")
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

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_file"] = str(path)
                groups[(row["mode"], row["method"], row.get("target_points", ""))].append(row)

    out_rows: list[dict[str, str]] = []
    for (mode, method, target_points), rows in sorted(groups.items()):
        pj = [x for r in rows if (x := f_or_none(r.get("pj_per_bit", ""))) is not None]
        r2 = [x for r in rows if (x := f_or_none(r.get("r2", ""))) is not None]
        resid = [
            x for r in rows
            if (x := f_or_none(r.get("max_abs_residual_w", ""))) is not None
        ]
        out_rows.append({
            "mode": mode,
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
        "mode", "method", "target_points", "runs", "pj_per_bit_mean", "pj_per_bit_std",
        "r2_mean", "max_abs_residual_w_mean", "files",
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


if __name__ == "__main__":
    main()
