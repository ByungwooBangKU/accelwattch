# Multi-GPU Component Energy Requirements

## Scope

`util/gpu_power_bench/` must support component-energy experiments for:

| Profile | GPU | Memory | L2 | Headline role |
|---|---|---|---:|---|
| `rtx3090` | RTX 3090 / sm_86 | 24GB GDDR6X | 6 MB | local smoke and GDDR6X reference only |
| `a100_sxm` | A100 SXM 80GB / sm_80 | 80GB HBM2E | 40 MB | FP32/TF32/FP16/BF16 TC, HBM2E, L2 path |
| `h100_sxm` | H100 SXM 80GB / sm_90 | 80GB HBM3 | 50 MB | primary headline target, including native FP8 TE |

The default experiment profile is `h100_sxm`.

## Functional Requirements

1. `gpu_power_bench.py` exposes `--gpu-profile {h100_sxm,a100_sxm,rtx3090,auto}` and defaults to `h100_sxm`.
2. Profile defaults drive the experiment axes:
   - `h100_sxm`: `--dtypes fp16 fp8`, L2 windows `16 24 32 40`, deltas `0 64 256 1024 4096 8192 16384`.
   - `a100_sxm`: `--dtypes fp16`, L2 windows `8 16 24 32`, deltas `0 64 256 1024 4096 8192`.
   - `rtx3090`: `--dtypes fp16`, L2 windows `1 2 3 4`, deltas `0 64 256 1024`.
3. Profile gating prevents unsupported rows from being mistaken for headline results:
   - RTX 3090: no BF16 headline, no FP8 headline.
   - A100 SXM: BF16 headline allowed, native FP8 headline not allowed.
   - H100 SXM: native FP8 TE headline allowed only on sm_90-class devices.
4. `--allow-non-headline` keeps fallback/proxy rows for debugging, but the CSV must still mark them as `NOT_HEADLINE` or `PROXY`.
5. Each measurement row records:
   - active `gpu_profile`, observed profile, profile status/reason
   - expected memory type/capacity/BW, expected and reported L2, power envelope/limit
   - `headline_eligible`, `headline_status`, and `headline_reason`
6. Each run writes `<main-stem>_gpu_spec_snapshot.csv` so reports can validate the GPU assumptions without relying on filename inference.
7. `--suite full` and `--suite all` include fused variants by default. `--no-fused` is available only as a dependency-debug opt-out.
8. `component_validation_report.py` writes tables to `--out-dir` and image report PNGs to a separate `--image-dir`.
9. Image report filenames are category-numbered:
   - `00_component_coverage_matrix.png`
   - `01_model_vs_measured_scatter_by_gpu.png`
   - `02_delta_by_gpu_and_workload.png`
   - `03_component_reconstruction_residual.png`
10. Report classification must distinguish:
   - `pass`: headline coefficient exists and quality gates pass.
   - `low_conf`: coefficient exists but regression/proxy confidence is limited.
   - `not_headline`: row exists only as fallback/emulation/smoke.
   - `not_applicable`: the GPU lacks that hardware headline path.
   - `missing`: no usable coefficient was found.
   - `fail`: coefficient was produced but quality gates failed.

## Measurement Boundary

All coefficients are board-level NVML measurements. The report must not claim isolated bit-cell or circuit energy when the benchmark only measures a traffic/execution path.

- HBM/GDDR pJ/bit means memory path energy at the board measurement boundary.
- L2 pJ/bit means L2-hit traffic path energy, not isolated SRAM bit-cell energy.
- Nonlinear coefficients include memory traffic and SM-local residual unless explicitly decomposed by the report.
- L1, shared memory, register file, scheduler/front-end, NoC/fabric, HBM controller/PHY subcomponents remain bundled or residual terms unless a future counter-calibrated benchmark is added.

## Recommended Runs

RTX 3090 local smoke:

```bash
./run_bench.sh \
  --gpu-profile rtx3090 \
  --cases elementwise matmul dram soc \
  --quick \
  --window-ms 1500 \
  --tag rtx3090_smoke
```

A100 SXM headline:

```bash
./run_bench.sh \
  --gpu-profile a100_sxm \
  --suite all \
  --tag a100_sxm_component
```

H100 SXM headline:

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --suite all \
  --tag h100_sxm_component
```

Focused L2-only H100 validation, optional when the full/all run needs to be
repeated only for L2:

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --cases l2 \
  --window-ms 8000 \
  --rebaseline-every 10 \
  --tag h100_l2
```

Cross-GPU report:

```bash
python3 component_validation_report.py \
  --reports-dir reports \
  --out-dir reports/component_validation \
  --image-dir reports/component_validation/image_report
```
