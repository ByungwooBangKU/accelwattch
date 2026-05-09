"""GPU profile metadata used by benchmark and validation tooling.

The profiles are intentionally experiment-facing, not a full hardware
database.  They encode only the assumptions that affect benchmark defaults,
headline eligibility, and report classification for the component-energy
workflow.
"""

from __future__ import annotations

import math
import re
from typing import Any


DEFAULT_GPU_PROFILE = "h100_sxm"


GPU_PROFILES: dict[str, dict[str, Any]] = {
    "rtx3090": {
        "label": "RTX 3090",
        "arch": "Ampere GA102",
        "expected_cc": "8.6",
        "memory_type": "GDDR6X",
        "memory_capacity_gb": 24,
        "peak_bw_gbps": 936.0,
        "l2_mb": 6.0,
        "power_envelope_w": 350.0,
        "native_fp8": False,
        "native_bf16": False,
        "default_dtypes": ["fp16"],
        "matmul_variants": [("fp32", "simt"), ("tf32", "tc"), ("fp16", "tc")],
        "llm_dtypes": [("fp16", "tc")],
        "fused_dtypes": ["fp16"],
        "l2_windows_mb": [1, 2, 3, 4],
        "l2_delta_kb": [0, 64, 256, 1024],
        "role": "local smoke / GDDR6X reference",
    },
    "a100_sxm": {
        "label": "A100 SXM 80GB",
        "arch": "Ampere GA100",
        "expected_cc": "8.0",
        "memory_type": "HBM2E",
        "memory_capacity_gb": 80,
        "peak_bw_gbps": 2039.0,
        "l2_mb": 40.0,
        "power_envelope_w": 400.0,
        "native_fp8": False,
        "native_bf16": True,
        "default_dtypes": ["fp16"],
        "matmul_variants": [("fp32", "simt"), ("tf32", "tc"), ("fp16", "tc"), ("bf16", "tc")],
        "llm_dtypes": [("bf16", "tc")],
        "fused_dtypes": ["fp16", "bf16"],
        "l2_windows_mb": [8, 16, 24, 32],
        "l2_delta_kb": [0, 64, 256, 1024, 4096, 8192],
        "role": "HBM2E / Ampere Tensor Core headline",
    },
    "h100_sxm": {
        "label": "H100 SXM 80GB",
        "arch": "Hopper GH100",
        "expected_cc": "9.0",
        "memory_type": "HBM3",
        "memory_capacity_gb": 80,
        "peak_bw_gbps": 3350.0,
        "l2_mb": 50.0,
        "power_envelope_w": 700.0,
        "native_fp8": True,
        "native_bf16": True,
        "default_dtypes": ["fp16", "fp8"],
        "matmul_variants": [
            ("fp32", "simt"), ("tf32", "tc"), ("fp16", "tc"),
            ("bf16", "tc"), ("fp8", "te"),
        ],
        "llm_dtypes": [("bf16", "tc"), ("fp8", "te")],
        "fused_dtypes": ["fp16", "bf16", "fp8"],
        "l2_windows_mb": [16, 24, 32, 40],
        "l2_delta_kb": [0, 64, 256, 1024, 4096, 8192, 16384],
        "role": "HBM3 / native FP8 headline",
    },
}


GPU_PROFILE_CHOICES = tuple(GPU_PROFILES.keys()) + ("auto",)


def normalize_cc(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    sm = re.search(r"sm[_-]?(\d)(\d)", s.lower())
    if sm:
        return f"{sm.group(1)}.{sm.group(2)}"
    m = re.search(r"(\d+)(?:\.(\d+))?", s)
    if not m:
        return s
    return f"{m.group(1)}.{m.group(2) or '0'}"


def infer_gpu_profile(*texts: object) -> str:
    hay = " ".join("" if t is None else str(t).lower() for t in texts)
    if "h100" in hay or "hopper" in hay or "gh100" in hay:
        return "h100_sxm"
    if "a100" in hay or "ga100" in hay:
        return "a100_sxm"
    if "3090" in hay or "rtx_3090" in hay or "rtx3090" in hay or "ga102" in hay:
        return "rtx3090"
    cc = normalize_cc(hay)
    if cc == "9.0":
        return "h100_sxm"
    if cc == "8.0":
        return "a100_sxm"
    if cc == "8.6":
        return "rtx3090"
    return "unknown"


def resolve_gpu_profile(requested: str, gpu_name: str = "", compute_cap: object = "") -> tuple[str, dict[str, Any], str, str]:
    """Return (profile_key, profile, observed_profile, warning_reason)."""
    observed = infer_gpu_profile(gpu_name, compute_cap)
    if requested == "auto":
        key = observed if observed in GPU_PROFILES else DEFAULT_GPU_PROFILE
    else:
        key = requested
    profile = GPU_PROFILES[key]
    reason = ""
    if observed in GPU_PROFILES and observed != key:
        reason = f"requested profile {key} differs from observed GPU profile {observed}"
    elif observed == "unknown":
        reason = "could not infer observed GPU profile from name/compute capability"
    return key, profile, observed, reason


def profile_cc_status(profile_key: str, observed_cc: object) -> tuple[str, str]:
    profile = GPU_PROFILES.get(profile_key)
    cc = normalize_cc(observed_cc)
    if not profile or not cc:
        return "WARN", "missing_or_unknown_compute_capability"
    expected = str(profile["expected_cc"])
    if cc == expected:
        return "PASS", ""
    if profile_key == "h100_sxm" and cc.startswith("9."):
        return "LOW_CONF", f"expected {expected}, observed {cc}"
    return "WARN", f"expected {expected}, observed {cc}"


def profile_choices() -> tuple[str, ...]:
    return GPU_PROFILE_CHOICES


def profile_label(profile_key: str) -> str:
    return str(GPU_PROFILES.get(profile_key, {}).get("label", profile_key))


def profile_default_dtypes(profile_key: str) -> list[str]:
    return list(GPU_PROFILES[profile_key]["default_dtypes"])


def profile_l2_windows_mb(profile_key: str) -> list[int]:
    return list(GPU_PROFILES[profile_key]["l2_windows_mb"])


def profile_l2_delta_kb(profile_key: str) -> list[int]:
    return list(GPU_PROFILES[profile_key]["l2_delta_kb"])


def profile_matmul_variants(profile_key: str) -> list[tuple[str, str]]:
    return [tuple(v) for v in GPU_PROFILES[profile_key]["matmul_variants"]]


def profile_llm_dtypes(profile_key: str) -> list[tuple[str, str]]:
    return [tuple(v) for v in GPU_PROFILES[profile_key]["llm_dtypes"]]


def profile_fused_dtypes(profile_key: str) -> list[str]:
    return list(GPU_PROFILES[profile_key]["fused_dtypes"])


def filter_with_profile(
    requested: list[Any],
    allowed: list[Any],
    *,
    label: str,
    allow_non_headline: bool,
) -> tuple[list[Any], list[Any]]:
    """Filter requested values to the profile headline set.

    Returns (kept, dropped).  If allow_non_headline is true nothing is
    dropped, but the caller can still use the returned empty dropped list to
    avoid warning noise.
    """
    if allow_non_headline:
        return list(requested), []
    allowed_set = set(allowed)
    kept = [v for v in requested if v in allowed_set]
    dropped = [v for v in requested if v not in allowed_set]
    if not kept:
        # Keep the run usable instead of returning an empty experiment axis.
        # The caller prints the drop; the fallback is explicit profile policy.
        kept = list(allowed)
    return kept, dropped


def native_fp8_headline_allowed(profile_key: str, observed_cc: object = "") -> bool:
    profile = GPU_PROFILES.get(profile_key, {})
    if not bool(profile.get("native_fp8", False)):
        return False
    cc = normalize_cc(observed_cc)
    return not cc or cc.startswith("9.")


def headline_status(
    profile_key: str,
    *,
    category: str,
    op: str,
    dtype: str,
    mode: str,
    compute_unit: str = "",
    emulated: bool = False,
    observed_cc: object = "",
) -> tuple[int, str, str]:
    """Classify whether one measured row can be used as a GPU headline.

    Returns (eligible, status, reason).  Status values are:
      HEADLINE      usable for the selected GPU's component headline
      PROXY         useful measurement, but bundled/proxy by design
      NOT_HEADLINE  valid smoke/fallback row only
    """
    profile = GPU_PROFILES.get(profile_key, {})
    cc = normalize_cc(observed_cc)
    if emulated:
        return 0, "NOT_HEADLINE", "emulated_or_fallback_path"

    if dtype == "fp8":
        if mode == "te" and category in ("matmul", "matmul_llm"):
            if native_fp8_headline_allowed(profile_key, cc):
                return 1, "HEADLINE", ""
            return 0, "NOT_HEADLINE", "native_fp8_tensor_core_requires_h100_sm90"
        if category == "fused" and op == "attention_flash":
            if native_fp8_headline_allowed(profile_key, cc):
                return 1, "HEADLINE", "fp8_fused_headline_limited_to_attention_flash"
            return 0, "NOT_HEADLINE", "fp8_fused_headline_requires_h100_sm90"
        return 0, "NOT_HEADLINE", "fp8_non_tensorcore_or_cast_path"

    if dtype == "bf16" and not bool(profile.get("native_bf16", False)):
        return 0, "NOT_HEADLINE", "bf16_not_a_headline_path_for_profile"

    if category == "l2":
        if profile_key == "rtx3090":
            return 1, "PROXY", "small_l2_gddr6x_reference_not_h100_l2_headline"
        return 1, "HEADLINE", "l2_hit_traffic_path_not_sram_bitcell"

    if category == "elementwise" and op.startswith("stream_"):
        return 1, "HEADLINE", f"{profile.get('memory_type', 'memory')}_traffic_path"

    if category == "elementwise" and op in ("softmax", "gelu", "layernorm"):
        return 1, "PROXY", "standalone_nonlinear_includes_memory_and_sm_local_path"

    if category == "fused":
        if dtype in profile_fused_dtypes(profile_key):
            return 1, "HEADLINE", "fused_residual_path"
        return 0, "NOT_HEADLINE", "dtype_not_enabled_for_profile_fused_headline"

    if category in ("matmul", "matmul_llm"):
        if (dtype, mode) in profile_matmul_variants(profile_key):
            if "fallback" in compute_unit.lower():
                return 0, "NOT_HEADLINE", "fallback_compute_unit"
            return 1, "HEADLINE", ""
        return 0, "NOT_HEADLINE", "matmul_variant_not_headline_for_profile"

    return 1, "HEADLINE", ""
