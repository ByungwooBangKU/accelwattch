"""Shared architecture constants for FP16 Tensor Core result normalization."""

from __future__ import annotations

import math
from typing import Any, Dict


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
        "tensor_peak_tflops_model": peak_tflops,
        "achieved_flops_per_sm_cycle": achieved_flops_per_sm_cycle,
        "tensor_model_utilization_pct": utilization_pct,
    }
