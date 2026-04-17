#!/usr/bin/env python3
"""Verify per-phase DRAM read utilization from an nsys sqlite export.

nsys 의 GPU metrics 샘플 중 'DRAM Read' 관련 지표를 찾아서
NVTX range (util_25 / util_50 / util_75 / util_100) 동안의 평균/표준편차를
계산해 목표치와 비교한다.

사용:
    python3 analyze.py reports/dram_util_*.sqlite
"""

import sqlite3
import statistics
import sys
from pathlib import Path


def pick_dram_read_metric(cur):
    """Return list of (typeId, name) whose name matches DRAM read throughput."""
    cur.execute("SELECT typeId, name FROM TARGET_INFO_GPU_METRICS")
    rows = cur.fetchall()
    # 예: 'DRAM Read Throughput', 'DRAM Bandwidth [%]', 'dram__bytes_read.sum' ...
    candidates = [
        (tid, n) for tid, n in rows
        if "dram" in n.lower() and "read" in n.lower()
    ]
    if not candidates:
        # fallback: 전체 DRAM throughput 관련
        candidates = [(tid, n) for tid, n in rows if "dram" in n.lower()]
    return rows, candidates


def phases(cur):
    """Return [(name, start_ns, end_ns)] for our util_* NVTX ranges, sorted."""
    cur.execute("""
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE text LIKE 'util\\_%' ESCAPE '\\'
          AND end IS NOT NULL
        ORDER BY start
    """)
    return cur.fetchall()


def mean_metric_in_range(cur, type_id, t0, t1):
    cur.execute(
        "SELECT value FROM GPU_METRICS "
        "WHERE typeId = ? AND timestamp BETWEEN ? AND ?",
        (type_id, t0, t1),
    )
    vals = [r[0] for r in cur.fetchall()]
    if not vals:
        return None, None, 0
    return (statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def main():
    if len(sys.argv) != 2:
        print("usage: analyze.py <report.sqlite>", file=sys.stderr)
        sys.exit(2)
    db = Path(sys.argv[1])
    if not db.exists():
        print(f"not found: {db}", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(db)
    cur = con.cursor()

    # 테이블 존재 확인
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    for need in ("NVTX_EVENTS", "GPU_METRICS", "TARGET_INFO_GPU_METRICS"):
        if need not in tables:
            print(f"[warn] table {need} 없음 — nsys 옵션에 "
                  f"--gpu-metrics-device 추가했는지 확인")
    if "GPU_METRICS" not in tables:
        sys.exit(1)

    all_metrics, dram_metrics = pick_dram_read_metric(cur)
    print(f"[info] GPU metrics 개수: {len(all_metrics)}")
    if not dram_metrics:
        print("[warn] DRAM read 관련 지표를 못 찾음. 사용 가능한 지표:")
        for _, n in all_metrics:
            print(f"   - {n}")
        sys.exit(1)
    print("[info] 매칭된 DRAM read 지표:")
    for tid, n in dram_metrics:
        print(f"   typeId={tid}  name={n}")

    phs = phases(cur)
    if not phs:
        print("[warn] NVTX 'util_*' range 없음")
        sys.exit(1)

    print()
    print(f"{'phase':<10} {'metric':<35} {'target%':>8} "
          f"{'mean':>8} {'stdev':>8} {'samples':>8}")
    print("-" * 88)
    for name, t0, t1 in phs:
        target = int(name.split("_")[1])
        for tid, mname in dram_metrics:
            mean, sd, ns = mean_metric_in_range(cur, tid, t0, t1)
            if mean is None:
                print(f"{name:<10} {mname:<35} {target:>8} "
                      f"{'--':>8} {'--':>8} {0:>8}")
            else:
                print(f"{name:<10} {mname:<35} {target:>8} "
                      f"{mean:>8.2f} {sd:>8.2f} {ns:>8}")

    con.close()


if __name__ == "__main__":
    main()
