# RTX 3090 FP16 launch-shape diagnostic, 2026-06-02 12:51 KST

This run is an RTX 3090/GA102 local diagnostic sweep for logical FP16 Tensor Core
`m16n16k16` matmul input energy. It is not a strict final pJ/bit claim because
Nsight Compute performance-counter access was blocked separately by
`ERR_NVGPUCTRPERM`, so no hardware no-L2/HMMA counter validation is available.

Command:

```bash
source env/toolchain_rtx3090_sm86_cuda121.sh
MPLCONFIGDIR=/tmp/mpl_fp16_diag_rtx3090_20260602_125100 \
  "$PYTHON_BIN" scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_launch_shape_sweep.json \
  --gpu 0 \
  --nvidia-smi /usr/lib/wsl/lib/nvidia-smi \
  --nvidia-smi-id GPU-6176a2fd-d534-e78c-edd2-78b8db8109b0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100
```

Postprocessing:

```bash
python3 scripts/analyze_results.py --input results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100
python3 scripts/quality_gate.py --input results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100
python3 scripts/summarize_kernel_resources.py \
  --result-dir results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100 \
  --outdir results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100/resource_audit \
  --cuda-arch 86 \
  --ptxas-log results/strict_fp16_launch_shape_rtx3090_20260602_124900/build_ptxas.log
```

Sweep:

| Parameter | Values |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, GA102, CUDA arch 86 |
| Matrix | logical `m16n16k16`; denominator `8192` FP16 input bits/logical MMA |
| Kernel | `tensor_mma_f16acc` vs `tensor_baseline_mov` |
| Output store | suppressed |
| threads/block | `32,64,128,256` |
| blocks/SM | `1,2,4,8` |
| launched threads/SM | `32,64,128,256,512,1024,2048` |
| Repeat | 10 full matrix repeats |

Diagnostic selected point:

| threads/block | blocks/SM | threads/SM | SM util | Tensor model util | TFLOPS | pJ/bit | valid no-L2 metadata |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 256 | 99.4% | 98.37% | 159.82 | 0.1833 +/- 0.1084 | 5/10 |

The diagnostic selection is the first Tensor Core model-utilization saturation
point. `quality_gate.py` did not produce a strict selected target:
`selected_targets` is empty and `selected_diagnostics` contains this point with
`quality_pass=false`, `target_pass=false`, `measurement_grade=mixed_or_unavailable`.
The recorded fail reason is `energy source is unavailable or undersampled;
baseline elapsed_s is below 0.25s`.

Lowest observed diagnostic pJ/bit rows:

| threads/block | blocks/SM | threads/SM | SM util | Tensor model util | TFLOPS | pJ/bit | valid no-L2 metadata |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 512 | 100.0% | 98.14% | 159.45 | 0.1143 +/- 0.1455 | 2/10 |
| 64 | 8 | 512 | 100.0% | 98.35% | 160.42 | 0.1412 +/- 0.1130 | 2/10 |
| 256 | 2 | 512 | 100.0% | 98.13% | 159.44 | 0.1573 +/- 0.1354 | 4/10 |
| 64 | 2 | 128 | n/a | 94.64% | 153.77 | 0.1740 +/- 0.0415 | 4/10 |
| 256 | 8 | 2048 | 100.0% | 98.21% | 158.74 | 0.1773 +/- 0.0519 | 7/10 |

Resource audit from ptxas:

| Kernel | Registers/thread | Stack | Spills |
|---|---:|---:|---|
| `tensor_mma_f16acc` | 14 | 0 B | none |
| `tensor_baseline_mov` | 14 | 0 B | none |

Core artifacts:

| File | Purpose |
|---|---|
| `thread_sweep_summary.csv` | launch-shape grouped pJ/bit, TFLOPS, SM util, model util |
| `quality_gates.csv` | diagnostic/strict gate status per candidate |
| `quality_gate_summary.json` | selected diagnostic and empty strict target list |
| `resource_audit/kernel_resource_summary.csv` | ptxas register/stack/spill evidence |
| `figures/thread_sweep_tensor_mma_f16acc_vs_tensor_baseline_mov.png` | x=threads/SM, y=SM utilization/model fallback |
| `figures/thread_sweep_pjbit_tensor_mma_f16acc_vs_tensor_baseline_mov.png` | x=threads/SM, y=pJ/bit |
| `figures/quality_gate_thread_sweep_tensor_mma_f16acc_vs_tensor_baseline_mov.png` | quality-gate annotated sweep |
| `figures/tflops_vs_pj_per_flop.png` | throughput vs energy diagnostic |
