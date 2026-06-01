#!/usr/bin/env python3
"""Generate a human-readable strict FP16 result report.

The report intentionally separates publishable strict-audit rows from diagnostic
or rejected rows. It does not select new results; it summarizes the output from
audit_strict_results.py and compare_architectures.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def parse_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NAN"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value: Any, digits: int = 4, missing: str = "") -> str:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return f"{parsed:.{digits}g}"
    text = str(value).strip() if value is not None else ""
    return text if text else missing


def yes_no(value: Any) -> str:
    return "yes" if parse_bool(value) else "no"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(out)


def relative_link(target: Path, base: Path) -> str:
    return os.path.relpath(target.resolve(), base.resolve()).replace(os.sep, "/")


def load_inputs(audit_dir: Path | None, compare_dir: Path | None, suite_preflight_json: Path | None) -> Dict[str, Any]:
    audit_rows: List[Dict[str, Any]] = []
    audit_json: Dict[str, Any] = {}
    preflight_json: Dict[str, Any] = {}
    best_rows: List[Dict[str, Any]] = []
    thread_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    resource_rows: List[Dict[str, Any]] = []

    if audit_dir:
        audit_rows = read_csv(audit_dir / "strict_result_audit.csv")
        audit_json = read_json(audit_dir / "strict_result_audit.json")
    if suite_preflight_json:
        preflight_json = read_json(suite_preflight_json)
    if compare_dir:
        best_rows = read_csv(compare_dir / "architecture_best_fp16.csv")
        thread_rows = read_csv(compare_dir / "architecture_thread_sweep_summary.csv")
        quality_rows = read_csv(compare_dir / "architecture_quality_gates.csv")
        resource_rows = read_csv(compare_dir / "architecture_resource_occupancy.csv")

    return {
        "audit_rows": audit_rows,
        "audit_json": audit_json,
        "preflight_json": preflight_json,
        "best_rows": best_rows,
        "thread_rows": thread_rows,
        "quality_rows": quality_rows,
        "resource_rows": resource_rows,
    }


def artifact_links(audit_dir: Path | None, compare_dir: Path | None, outdir: Path) -> List[str]:
    candidates: List[Path] = []
    if audit_dir:
        candidates.extend(
            [
                audit_dir / "figures" / "strict_result_audit.png",
                audit_dir / "figures" / "strict_result_matmul_input_pj_per_bit.png",
                audit_dir / "figures" / "strict_result_tflops.png",
                audit_dir / "figures" / "strict_result_sm_utilization.png",
                audit_dir / "figures" / "strict_result_tensor_model_utilization.png",
                audit_dir / "figures" / "strict_result_ncu_tensor_activity.png",
                audit_dir / "figures" / "strict_result_incremental_energy_fraction.png",
                audit_dir / "figures" / "strict_result_counter_trace_ratio.png",
            ]
        )
    if compare_dir:
        candidates.extend(
            [
                compare_dir / "architecture_best_matmul_input_pj_per_bit.png",
                compare_dir / "architecture_best_tflops.png",
                compare_dir / "architecture_best_tensor_model_utilization.png",
                compare_dir / "architecture_best_incremental_energy_fraction.png",
            ]
        )
        candidates.extend(sorted(compare_dir.glob("architecture_thread_sweep_util_*.png")))
        candidates.extend(sorted(compare_dir.glob("architecture_thread_sweep_pjbit_*.png")))
        candidates.extend(sorted(compare_dir.glob("architecture_thread_sweep_model_util_*.png")))
        candidates.extend(sorted(compare_dir.glob("architecture_resource_occupancy_*.png")))

    links = []
    seen = set()
    for path in candidates:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        links.append(f"- [{path.name}]({relative_link(path, outdir)})")
    return links


def requirement_rows(
    audit_rows: List[Dict[str, Any]],
    audit_json: Dict[str, Any],
    preflight_json: Dict[str, Any],
    required_architectures: Sequence[str],
) -> List[Dict[str, Any]]:
    passed = {str(row.get("architecture_chip", "")) for row in audit_rows if parse_bool(row.get("audit_pass"))}
    missing = [chip for chip in required_architectures if chip not in passed]
    all_audit_pass = bool(audit_rows) and all(parse_bool(row.get("audit_pass")) for row in audit_rows)
    strict_sources = bool(audit_rows) and all(
        str(row.get("measurement_grade", "")) == "strict_nvml_counter" for row in audit_rows
    )
    structural = bool(audit_rows) and all(
        str(row.get("baseline_match_grade", "")) == "structural_baseline" for row in audit_rows
    )
    schema_current = bool(audit_rows) and all(parse_bool(row.get("benchmark_schema_current")) for row in audit_rows)
    pipeline_manifest = bool(audit_rows) and all(
        parse_bool(row.get("pipeline_manifest_present"))
        and str(row.get("pipeline_manifest_schema", "")) == "fp16-strict-pipeline-manifest-v1"
        and str(row.get("pipeline_status", "")) == "completed"
        and str(row.get("pipeline_git_head", "")).strip()
        and str(row.get("pipeline_binary_sha256", "")).strip()
        for row in audit_rows
    )
    ncu = bool(audit_rows) and all(parse_bool(row.get("ncu_validation_pass")) for row in audit_rows)
    ncu_context = bool(audit_rows) and all(
        parse_bool(row.get("ncu_validation_context_match")) for row in audit_rows
    )
    no_spill = bool(audit_rows) and all(
        not parse_bool(row.get("test_resource_has_spills")) and not parse_bool(row.get("baseline_resource_has_spills"))
        for row in audit_rows
    )
    positive_pjbit = bool(audit_rows) and all(
        parse_float(row.get("matmul_input_pj_per_bit_mean")) > 0.0 for row in audit_rows
    )
    denominator = bool(audit_rows) and all(
        parse_bool(row.get("matmul_denominator_valid"))
        and parse_bool(row.get("matmul_denominator_metadata_complete"))
        and str(row.get("matmul_denominator_source", "")) == "bench_json_metadata"
        and parse_float(row.get("matmul_input_bits_per_logical_mma")) == 8192.0
        and parse_float(row.get("matmul_flops_per_logical_mma")) == 8192.0
        for row in audit_rows
    )
    model_util = bool(audit_rows) and all(
        parse_float(row.get("tensor_model_utilization_pct_mean")) > 0.0 for row in audit_rows
    )
    overall_json_pass = bool(audit_json.get("overall_pass", False))
    preflight_schema = str(preflight_json.get("preflight_schema", "") or "")
    preflight_supplied = bool(preflight_json)
    preflight_ok = (
        preflight_supplied
        and preflight_schema == "fp16-strict-architecture-suite-preflight-v1"
        and parse_bool(preflight_json.get("overall_pass"))
        and not parse_bool(preflight_json.get("dry_run"))
    )
    preflight_evidence = "strict_architecture_suite_preflight.json"
    if not preflight_supplied:
        preflight_evidence = "suite preflight JSON not supplied to report"
    elif parse_bool(preflight_json.get("dry_run")):
        preflight_evidence = "suite preflight was dry_run=true"
    else:
        preflight_evidence = (
            f"schema={preflight_schema or 'missing'}, overall_pass={preflight_json.get('overall_pass')}"
        )

    def row(requirement: str, ok: bool, evidence: str) -> Dict[str, Any]:
        return {"requirement": requirement, "status": "pass" if ok else "missing_or_fail", "evidence": evidence}

    return [
        row(
            "required architectures present and passing",
            bool(required_architectures) and not missing,
            "passed=" + ",".join(sorted(passed)) + ("; missing=" + ",".join(missing) if missing else ""),
        ),
        row("suite preflight passed", preflight_ok, preflight_evidence),
        row("strict audit overall pass", overall_json_pass and all_audit_pass, "strict_result_audit.json overall_pass"),
        row("NVML total-energy counter source", strict_sources, "measurement_grade=strict_nvml_counter"),
        row("structural baseline separation", structural, "baseline_match_grade=structural_baseline"),
        row("current benchmark schema", schema_current, "benchmark_schema_current=true"),
        row(
            "strict pipeline provenance manifest",
            pipeline_manifest,
            "strict_pipeline_manifest.json schema/status/git/binary hash",
        ),
        row("Nsight Compute no-L2/HMMA validation", ncu, "ncu_validation_pass=true"),
        row("NCU validation context matches measurement", ncu_context, "ncu_validation_context_match=true"),
        row("ptxas resource audit has no spills", no_spill, "test/baseline_resource_has_spills=false"),
        row(
            "logical m16n16k16 denominator",
            denominator,
            "bench_json_metadata with input_bits=8192 and flops=8192",
        ),
        row("positive logical FP16 pJ/bit", positive_pjbit, "matmul_input_pj_per_bit_mean > 0"),
        row("positive Tensor Core model utilization", model_util, "tensor_model_utilization_pct_mean > 0"),
    ]


def summary_rows(audit_rows: List[Dict[str, Any]], best_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if audit_rows:
        return [
            {
                "source": "strict_audit",
                "architecture_chip": row.get("architecture_chip", ""),
                "gpu": row.get("gpu", ""),
                "publishable": parse_bool(row.get("audit_pass")),
                "selection_note": "audit_pass" if parse_bool(row.get("audit_pass")) else "audit_failed",
                "threads_per_sm": row.get("threads_per_sm", ""),
                "threads": row.get("threads", ""),
                "matmul_input_pj_per_bit": row.get("matmul_input_pj_per_bit_mean", ""),
                "matmul_input_bits_per_logical_mma": row.get("matmul_input_bits_per_logical_mma", ""),
                "matmul_denominator_source": row.get("matmul_denominator_source", ""),
                "tflops": row.get("tflops_mean", ""),
                "avg_sm_util_pct": row.get("avg_sm_util_pct_mean", ""),
                "tensor_model_util_pct": row.get("tensor_model_utilization_pct_mean", ""),
                "measurement_grade": row.get("measurement_grade", ""),
                "baseline_match_grade": row.get("baseline_match_grade", ""),
                "benchmark_schema_current": row.get("benchmark_schema_current", ""),
                "pipeline_status": row.get("pipeline_status", ""),
                "pipeline_git_head": row.get("pipeline_git_head", ""),
                "ncu_validation_pass": row.get("ncu_validation_pass", ""),
                "ncu_validation_context_match": row.get("ncu_validation_context_match", ""),
                "ncu_tensor_activity_pct": row.get("test_ncu_tensor_activity_pct", ""),
                "fail_reasons": row.get("fail_reasons", ""),
            }
            for row in audit_rows
        ]
    return [
        {
            "source": "architecture_compare",
            "architecture_chip": row.get("architecture_chip", ""),
            "gpu": row.get("gpu", ""),
            "publishable": parse_bool(row.get("target_pass")),
            "selection_note": row.get("selection_note", ""),
            "threads_per_sm": row.get("threads_per_sm", ""),
            "threads": row.get("threads", ""),
            "matmul_input_pj_per_bit": row.get("matmul_input_pj_per_bit_mean", ""),
            "tflops": row.get("tflops_mean", ""),
            "avg_sm_util_pct": row.get("avg_sm_util_pct_mean", ""),
            "tensor_model_util_pct": row.get("tensor_model_utilization_pct_mean", ""),
            "measurement_grade": row.get("measurement_grade", ""),
            "baseline_match_grade": row.get("baseline_match_grade", ""),
            "ncu_validation_pass": row.get("ncu_validation_pass", ""),
            "ncu_tensor_activity_pct": row.get("test_ncu_tensor_activity_pct", ""),
            "fail_reasons": row.get("fail_reasons", ""),
        }
        for row in best_rows
    ]


def plot_dashboard(rows: List[Dict[str, Any]], fig_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    xs = [parse_float(row.get("tflops")) for row in rows]
    ys = [parse_float(row.get("matmul_input_pj_per_bit")) for row in rows]
    if not any(math.isfinite(x) and math.isfinite(y) for x, y in zip(xs, ys)):
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for row, x, y in zip(rows, xs, ys):
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        passed = parse_bool(row.get("publishable"))
        color = "tab:green" if passed else "tab:red"
        marker = "o" if passed else "x"
        label = str(row.get("architecture_chip", "") or row.get("gpu", "") or "unknown")
        ax.scatter([x], [y], color=color, marker=marker, s=72)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel("Selected TFLOPS")
    ax.set_ylabel("pJ/logical FP16 input bit")
    ax.set_title("Strict FP16 selected result dashboard")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def write_markdown(
    path: Path,
    title: str,
    audit_dir: Path | None,
    compare_dir: Path | None,
    rows: List[Dict[str, Any]],
    req_rows: List[Dict[str, Any]],
    audit_json: Dict[str, Any],
    suite_preflight_json: Path | None,
    suite_preflight_csv: Path | None,
    links: List[str],
) -> None:
    requirements_pass = bool(req_rows) and all(str(r.get("status", "")) == "pass" for r in req_rows)
    overall = "pass" if audit_json.get("overall_pass", False) and requirements_pass else "not publishable"
    if not audit_json and rows:
        overall = "diagnostic only"

    report: List[str] = [
        f"# {title}",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Overall status: **{overall}**",
        "",
    ]
    if audit_dir:
        report.append(f"- Audit directory: `{audit_dir}`")
    if compare_dir:
        report.append(f"- Architecture compare directory: `{compare_dir}`")
    if suite_preflight_json:
        report.append(f"- Suite preflight JSON: `{suite_preflight_json}`")
    if suite_preflight_csv:
        report.append(f"- Suite preflight CSV: `{suite_preflight_csv}`")
    report.append("")

    report.extend(
        [
            "## Completion Evidence Matrix",
            "",
            markdown_table(
                ["Requirement", "Status", "Evidence"],
                [[r["requirement"], r["status"], r["evidence"]] for r in req_rows],
            ),
            "",
            "## Selected Results",
            "",
            markdown_table(
                [
                    "Arch",
                    "GPU",
                    "Publishable",
                    "Threads/SM",
                    "Threads/block",
                    "pJ/bit",
                    "TFLOPS",
                    "SM util %",
                    "Tensor model %",
                    "Energy source",
                    "Baseline",
                    "NCU",
                    "NCU tensor %",
                ],
                [
                    [
                        r.get("architecture_chip", ""),
                        r.get("gpu", ""),
                        yes_no(r.get("publishable")),
                        fmt_value(r.get("threads_per_sm")),
                        fmt_value(r.get("threads")),
                        fmt_value(r.get("matmul_input_pj_per_bit")),
                        fmt_value(r.get("tflops")),
                        fmt_value(r.get("avg_sm_util_pct")),
                        fmt_value(r.get("tensor_model_util_pct")),
                        r.get("measurement_grade", ""),
                        r.get("baseline_match_grade", ""),
                        yes_no(r.get("ncu_validation_pass")),
                        fmt_value(r.get("ncu_tensor_activity_pct")),
                    ]
                    for r in rows
                ],
            )
            if rows
            else "No selected result rows were found.",
            "",
        ]
    )

    failed = [r for r in rows if not parse_bool(r.get("publishable")) and str(r.get("fail_reasons", "")).strip()]
    if failed:
        report.extend(
            [
                "## Fail Reasons",
                "",
                markdown_table(
                    ["Arch", "Reason"],
                    [[r.get("architecture_chip", ""), r.get("fail_reasons", "")] for r in failed],
                ),
                "",
            ]
        )

    if links:
        report.extend(["## Figures", "", *links, ""])

    report.extend(
        [
            "## Interpretation",
            "",
            "- Use only rows with `Publishable=yes` for final pJ/bit claims.",
            "- `pJ/bit` is logical FP16 matmul input-bit energy, not DRAM bit energy.",
            "- `strict_nvml_counter` plus structural baseline, NCU validation, resource audit, and measurement-resolution gates are required for A100/H100/RTX3090 comparison.",
            "- Suite preflight must pass with `dry_run=false`; dry-run output is command validation only.",
            "- Missing A100 or H100 strict rows means the architecture comparison is incomplete, even if RTX 3090 diagnostics exist.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a strict FP16 Markdown report")
    parser.add_argument("--audit-dir", type=Path, default=None, help="Directory from audit_strict_results.py")
    parser.add_argument("--compare-dir", type=Path, default=None, help="Directory from compare_architectures.py")
    parser.add_argument("--suite-preflight-json", type=Path, default=None)
    parser.add_argument("--suite-preflight-csv", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--title", default="Strict FP16 Energy Result Report")
    parser.add_argument("--require-architectures", default="ga100,gh100,ga102")
    args = parser.parse_args()

    if not args.audit_dir and not args.compare_dir:
        raise SystemExit("Provide --audit-dir, --compare-dir, or both")

    args.outdir.mkdir(parents=True, exist_ok=True)
    loaded = load_inputs(args.audit_dir, args.compare_dir, args.suite_preflight_json)
    required = [item.strip() for item in args.require_architectures.split(",") if item.strip()]
    rows = summary_rows(loaded["audit_rows"], loaded["best_rows"])
    req_rows = requirement_rows(loaded["audit_rows"], loaded["audit_json"], loaded["preflight_json"], required)
    links = artifact_links(args.audit_dir, args.compare_dir, args.outdir)

    write_csv(args.outdir / "fp16_strict_report_summary.csv", rows)
    write_csv(args.outdir / "fp16_strict_report_requirements.csv", req_rows)
    plot_dashboard(rows, args.outdir / "fp16_strict_report_dashboard.png")
    if (args.outdir / "fp16_strict_report_dashboard.png").exists():
        links = [f"- [fp16_strict_report_dashboard.png](fp16_strict_report_dashboard.png)", *links]
    write_markdown(
        args.outdir / "fp16_strict_report.md",
        args.title,
        args.audit_dir,
        args.compare_dir,
        rows,
        req_rows,
        loaded["audit_json"],
        args.suite_preflight_json,
        args.suite_preflight_csv,
        links,
    )

    print(f"Wrote: {args.outdir / 'fp16_strict_report.md'}")
    print(f"Wrote: {args.outdir / 'fp16_strict_report_summary.csv'}")
    print(f"Wrote: {args.outdir / 'fp16_strict_report_requirements.csv'}")
    if (args.outdir / "fp16_strict_report_dashboard.png").exists():
        print(f"Wrote: {args.outdir / 'fp16_strict_report_dashboard.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
