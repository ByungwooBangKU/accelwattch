# RTX 3090 strict FP16 pipeline attempt, 2026-06-02 12:49 KST

This RTX 3090 strict run stopped before calibration/sweep because Nsight Compute
performance-counter access was denied.

Command:

```bash
source env/toolchain_rtx3090_sm86_cuda121.sh
./scripts/run_strict_fp16_pipeline.sh \
  --gpu 0 \
  --cuda-arch 86 \
  --nvidia-smi-id GPU-6176a2fd-d534-e78c-edd2-78b8db8109b0 \
  --matrix configs/fp16_matmul_launch_shape_sweep.json \
  --threads 32,64,128,256 \
  --ncu-blocks-per-sm-csv 1,2,4,8 \
  --outdir results/strict_fp16_launch_shape_rtx3090_20260602_124900
```

Status:

| Check | Result |
|---|---|
| GPU/runtime preflight | pass |
| Architecture check | pass: RTX 3090/GA102, CUDA arch 86, common HMMA path |
| Build | pass |
| ptxas resources | `tensor_mma_f16acc` 14 regs/thread, no stack/spills; `tensor_baseline_mov` 14 regs/thread, no stack/spills |
| NCU permission probe | fail: `ERR_NVGPUCTRPERM` |
| Strict sweep | not started |

`ncu_permission_probe/ncu_permission_probe.json` records:

```text
status=permission_denied
permission_probe_pass=false
permission_denied=true
fail_reasons=ncu returned 1; Nsight Compute performance counters are blocked by ERR_NVGPUCTRPERM; Nsight Compute did not profile any kernels
```

This is an NVIDIA driver/admin policy issue, not a CUDA build issue. The strict
FP16 pJ/bit claim requires rerunning the same command in a session where NVIDIA
GPU performance counters are enabled.
