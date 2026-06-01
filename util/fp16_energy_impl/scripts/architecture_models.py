"""Shared architecture constants for FP16 Tensor Core result normalization."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


DERIVED_DENSE_FORMULA = (
    "dense_tensor_fp16_flop_per_sm_cycle * reference_sm_count * "
    "reference_boost_clock_mhz * 1e6 / 1e12"
)
COMMON_TENSOR_INSTRUCTION_PATH = "warp-level HMMA mma.sync.aligned.m16n8k16 pair -> logical m16n16k16"
NORMALIZATION_SCOPE = "dense FP16 Tensor Core HMMA normalization only; no sparsity, no WGMMA"


ARCH_MODELS: Dict[str, Dict[str, Any]] = {
    "ga100": {
        "architecture_generation": "ampere",
        "architecture_chip": "ga100",
        "recommended_cuda_arch": "80",
        "registers_per_sm": 65536,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "max_warps_per_sm": 64,
        "dense_tensor_fp16_flop_per_sm_cycle": 2048.0,
        "reference_sm_count": 108,
        "reference_boost_clock_mhz": 1410.0,
        "reference_dense_tensor_fp16_tflops": 312.0,
        "reference_sparse_tensor_fp16_tflops": 624.0,
        "reference_source_url": "https://www.nvidia.com/en-us/data-center/a100/",
        "reference_note": "A100 SXM dense FP16 Tensor Core peak, no sparsity",
        "sparsity_mode": "dense_no_sparsity",
        "uses_wgmma_model": False,
    },
    "ga102": {
        "architecture_generation": "ampere",
        "architecture_chip": "ga102",
        "recommended_cuda_arch": "86",
        "registers_per_sm": 65536,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 16,
        "max_warps_per_sm": 48,
        "dense_tensor_fp16_flop_per_sm_cycle": 1024.0,
        "reference_sm_count": 82,
        "reference_boost_clock_mhz": 1695.0,
        "reference_dense_tensor_fp16_tflops": 142.0,
        "reference_sparse_tensor_fp16_tflops": 284.0,
        "reference_source_url": (
            "https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/"
            "NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf"
        ),
        "reference_note": "RTX 3090 FE dense FP16 Tensor Core peak, no sparsity",
        "sparsity_mode": "dense_no_sparsity",
        "uses_wgmma_model": False,
    },
    "gh100": {
        "architecture_generation": "hopper",
        "architecture_chip": "gh100",
        "recommended_cuda_arch": "90",
        "registers_per_sm": 65536,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "max_warps_per_sm": 64,
        "dense_tensor_fp16_flop_per_sm_cycle": 4096.0,
        "reference_sm_count": 132,
        "reference_boost_clock_mhz": 1830.0,
        "reference_dense_tensor_fp16_tflops": 989.5,
        "reference_sparse_tensor_fp16_tflops": 1979.0,
        "reference_source_url": "https://www.nvidia.com/en-us/data-center/h100/",
        "reference_note": (
            "H100 SXM dense FP16 Tensor Core peak model, no sparsity; NVIDIA's public H100 "
            "product table marks FP16 Tensor Core values with sparsity, so dense is half"
        ),
        "sparsity_mode": "dense_no_sparsity",
        "uses_wgmma_model": False,
    },
}

ARCH_BY_CUDA = {
    str(model["recommended_cuda_arch"]): model
    for model in ARCH_MODELS.values()
}


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


def architecture_model(chip: Any = "", cuda_arch: Any = "") -> Dict[str, Any]:
    chip_text = str(chip or "").strip().lower()
    if chip_text in ARCH_MODELS:
        return ARCH_MODELS[chip_text]
    arch_text = str(cuda_arch or "").strip().replace("sm_", "")
    return ARCH_BY_CUDA.get(arch_text, {})


def architecture_model_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return architecture_model(
        row.get("architecture_chip", ""),
        row.get("recommended_cuda_arch", row.get("sm", "")),
    )


def reference_dense_tflops(model: Dict[str, Any]) -> float:
    flops_per_sm_cycle = parse_float(model.get("dense_tensor_fp16_flop_per_sm_cycle"))
    sm_count = parse_float(model.get("reference_sm_count"))
    clock_mhz = parse_float(model.get("reference_boost_clock_mhz"))
    if (
        math.isfinite(flops_per_sm_cycle)
        and math.isfinite(sm_count)
        and sm_count > 0.0
        and math.isfinite(clock_mhz)
        and clock_mhz > 0.0
    ):
        return flops_per_sm_cycle * sm_count * clock_mhz * 1.0e6 / 1.0e12
    return math.nan


def model_summary_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in ARCH_MODELS.values():
        derived = reference_dense_tflops(model)
        reference = parse_float(model.get("reference_dense_tensor_fp16_tflops"))
        sparse_reference = parse_float(model.get("reference_sparse_tensor_fp16_tflops"))
        sparse_to_dense_ratio = (
            sparse_reference / reference
            if math.isfinite(sparse_reference) and math.isfinite(reference) and reference > 0.0
            else math.nan
        )
        error_pct = (
            (derived - reference) / reference * 100.0
            if math.isfinite(derived) and math.isfinite(reference) and reference > 0.0
            else math.nan
        )
        rows.append(
            {
                "architecture_generation": model.get("architecture_generation", ""),
                "architecture_chip": model.get("architecture_chip", ""),
                "recommended_cuda_arch": model.get("recommended_cuda_arch", ""),
                "registers_per_sm": model.get("registers_per_sm", ""),
                "max_threads_per_sm": model.get("max_threads_per_sm", ""),
                "max_blocks_per_sm": model.get("max_blocks_per_sm", ""),
                "max_warps_per_sm": model.get("max_warps_per_sm", ""),
                "dense_tensor_fp16_flop_per_sm_cycle": model.get(
                    "dense_tensor_fp16_flop_per_sm_cycle",
                    math.nan,
                ),
                "reference_sm_count": model.get("reference_sm_count", ""),
                "reference_boost_clock_mhz": model.get("reference_boost_clock_mhz", ""),
                "reference_dense_tensor_fp16_tflops": reference,
                "derived_dense_tensor_fp16_tflops": derived,
                "reference_error_pct": error_pct,
                "reference_sparse_tensor_fp16_tflops": model.get(
                    "reference_sparse_tensor_fp16_tflops",
                    math.nan,
                ),
                "sparse_to_dense_ratio": sparse_to_dense_ratio,
                "sparsity_mode": model.get("sparsity_mode", ""),
                "uses_wgmma_model": model.get("uses_wgmma_model", ""),
                "reference_source_url": model.get("reference_source_url", ""),
                "reference_note": model.get("reference_note", ""),
                "dense_reference_formula": DERIVED_DENSE_FORMULA,
                "common_tensor_instruction_path": COMMON_TENSOR_INSTRUCTION_PATH,
                "normalization_scope": NORMALIZATION_SCOPE,
                "normalization_note": (
                    "Dense Tensor Core peak is used only to normalize measured FP16 HMMA throughput; "
                    "it is not an energy source and does not imply H100 WGMMA was benchmarked."
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_model_summary(rows: List[Dict[str, Any]], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row.get("architecture_chip", "")) for row in rows]
    reference = [parse_float(row.get("reference_dense_tensor_fp16_tflops")) for row in rows]
    derived = [parse_float(row.get("derived_dense_tensor_fp16_tflops")) for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar([i - width / 2 for i in x], reference, width=width, label="reference dense TFLOPS")
    ax.bar([i + width / 2 for i in x], derived, width=width, label="derived from SM*clock*FLOP/cycle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dense FP16 Tensor Core TFLOPS")
    ax.set_title("A100/H100/RTX3090 FP16 Tensor Core peak model sanity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "architecture_model_dense_peak.png", dpi=160)
    plt.close(fig)


def plot_per_sm_capacity(rows: List[Dict[str, Any]], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row.get("architecture_chip", "")) for row in rows]
    capacity = [parse_float(row.get("dense_tensor_fp16_flop_per_sm_cycle")) for row in rows]
    if not any(math.isfinite(value) for value in capacity):
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, capacity, color=["tab:blue", "tab:orange", "tab:green"][: len(labels)])
    ax.set_ylabel("Dense FP16 Tensor Core FLOP / SM / cycle")
    ax.set_title("Per-SM FP16 Tensor Core capacity model")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, capacity):
        if math.isfinite(value):
            ax.annotate(
                f"{value:.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(outdir / "architecture_model_per_sm_capacity.png", dpi=160)
    plt.close(fig)


def plot_resource_limits(rows: List[Dict[str, Any]], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row.get("architecture_chip", "")) for row in rows]
    x = list(range(len(rows)))
    width = 0.34
    max_threads = [parse_float(row.get("max_threads_per_sm")) for row in rows]
    warp_thread_slots = [parse_float(row.get("max_warps_per_sm")) * 32.0 for row in rows]
    max_blocks = [parse_float(row.get("max_blocks_per_sm")) for row in rows]
    registers_k = [parse_float(row.get("registers_per_sm")) / 1024.0 for row in rows]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    ax_top.bar([i - width / 2 for i in x], max_threads, width=width, label="max threads/SM")
    ax_top.bar([i + width / 2 for i in x], warp_thread_slots, width=width, label="max warps/SM * 32")
    ax_top.set_ylabel("Thread slots")
    ax_top.set_title("Architecture occupancy resource limits")
    ax_top.grid(True, axis="y", alpha=0.25)
    ax_top.legend(loc="best")

    ax_bottom.bar([i - width / 2 for i in x], max_blocks, width=width, label="max blocks/SM")
    ax_bottom.bar([i + width / 2 for i in x], registers_k, width=width, label="registers/SM / 1024")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels)
    ax_bottom.set_ylabel("Blocks or Ki-registers")
    ax_bottom.grid(True, axis="y", alpha=0.25)
    ax_bottom.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "architecture_model_resource_limits.png", dpi=160)
    plt.close(fig)


def tensor_peak_metrics(row: Dict[str, Any], achieved_tflops: Any) -> Dict[str, Any]:
    """Return dense Tensor Core peak-normalized metrics for the common HMMA path."""
    model = architecture_model_from_row(row)
    flops_per_sm_cycle = parse_float(model.get("dense_tensor_fp16_flop_per_sm_cycle"))
    sm_count = parse_float(row.get("sm_count"))
    clock_mhz = parse_float(row.get("avg_sm_clock_mhz"))
    tflops = parse_float(achieved_tflops)

    peak_tflops = (
        flops_per_sm_cycle * sm_count * clock_mhz * 1.0e6 / 1.0e12
        if math.isfinite(flops_per_sm_cycle)
        and math.isfinite(sm_count)
        and sm_count > 0
        and math.isfinite(clock_mhz)
        and clock_mhz > 0
        else math.nan
    )
    achieved_flops_per_sm_cycle = (
        tflops * 1.0e12 / (sm_count * clock_mhz * 1.0e6)
        if math.isfinite(tflops)
        and math.isfinite(sm_count)
        and sm_count > 0
        and math.isfinite(clock_mhz)
        and clock_mhz > 0
        else math.nan
    )
    utilization_pct = (
        tflops / peak_tflops * 100.0
        if math.isfinite(tflops) and math.isfinite(peak_tflops) and peak_tflops > 0.0
        else math.nan
    )
    return {
        "tensor_model_architecture_chip": model.get("architecture_chip", ""),
        "tensor_model_reference_note": model.get("reference_note", ""),
        "tensor_model_reference_url": model.get("reference_source_url", ""),
        "tensor_model_reference_dense_tflops": model.get("reference_dense_tensor_fp16_tflops", math.nan),
        "tensor_model_reference_sparse_tflops": model.get("reference_sparse_tensor_fp16_tflops", math.nan),
        "tensor_model_flop_per_sm_cycle": flops_per_sm_cycle,
        "tensor_model_common_instruction_path": model.get(
            "common_tensor_instruction_path",
            COMMON_TENSOR_INSTRUCTION_PATH,
        ),
        "tensor_model_normalization_scope": model.get("normalization_scope", NORMALIZATION_SCOPE),
        "tensor_peak_tflops_model": peak_tflops,
        "achieved_flops_per_sm_cycle": achieved_flops_per_sm_cycle,
        "tensor_model_utilization_pct": utilization_pct,
    }


def model_failures(
    rows: List[Dict[str, Any]],
    *,
    required_architectures: List[str],
    max_error_pct: float,
    require_metadata: bool,
) -> List[str]:
    failures: List[str] = []
    by_chip = {str(row.get("architecture_chip", "")): row for row in rows}
    for chip in required_architectures:
        if chip not in by_chip:
            failures.append(f"missing required architecture model: {chip}")
    for row in rows:
        chip = str(row.get("architecture_chip", "unknown"))
        error_pct = abs(parse_float(row.get("reference_error_pct")))
        if not math.isfinite(error_pct) or error_pct > max_error_pct:
            failures.append(f"{chip} reference_error_pct {error_pct:g} exceeds {max_error_pct:g}")
        dense = parse_float(row.get("reference_dense_tensor_fp16_tflops"))
        sparse = parse_float(row.get("reference_sparse_tensor_fp16_tflops"))
        ratio = parse_float(row.get("sparse_to_dense_ratio"))
        if not (math.isfinite(dense) and dense > 0.0):
            failures.append(f"{chip} dense FP16 Tensor Core reference is missing")
        if math.isfinite(sparse) and math.isfinite(dense) and dense > 0.0:
            expected_ratio = sparse / dense
            if not math.isfinite(ratio) or abs(ratio - expected_ratio) > 1.0e-9:
                failures.append(f"{chip} sparse_to_dense_ratio does not match sparse/dense reference")
        if require_metadata:
            for key in (
                "reference_source_url",
                "reference_note",
                "dense_reference_formula",
                "common_tensor_instruction_path",
                "normalization_scope",
                "normalization_note",
                "sparsity_mode",
            ):
                if not str(row.get(key, "")).strip():
                    failures.append(f"{chip} missing architecture model metadata field: {key}")
            if parse_bool(row.get("uses_wgmma_model")):
                failures.append(f"{chip} model unexpectedly uses WGMMA for common-HMMA normalization")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Write FP16 architecture model summary and sanity figures")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--require-architectures", default="ga100,gh100,ga102")
    parser.add_argument("--fail-on-model-error-pct", type=float, default=None)
    parser.add_argument("--fail-on-missing-metadata", action="store_true")
    args = parser.parse_args()

    rows = model_summary_rows()
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "architecture_model_summary.csv"
    write_csv(csv_path, rows)
    if not args.no_figures:
        plot_model_summary(rows, args.outdir)
        plot_per_sm_capacity(rows, args.outdir)
        plot_resource_limits(rows, args.outdir)
    print(f"Wrote: {csv_path}")
    if not args.no_figures:
        print(f"Wrote: {args.outdir / 'architecture_model_dense_peak.png'}")
        print(f"Wrote: {args.outdir / 'architecture_model_per_sm_capacity.png'}")
        print(f"Wrote: {args.outdir / 'architecture_model_resource_limits.png'}")
    if args.fail_on_model_error_pct is not None or args.fail_on_missing_metadata:
        required = [item.strip() for item in args.require_architectures.split(",") if item.strip()]
        failures = model_failures(
            rows,
            required_architectures=required,
            max_error_pct=args.fail_on_model_error_pct if args.fail_on_model_error_pct is not None else math.inf,
            require_metadata=args.fail_on_missing_metadata,
        )
        if failures:
            print("Architecture model sanity failed:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
