# FP16 Energy Experiment Results

This directory contains the curated RTX 3090 CUDA 13.2 experiment outputs used by the
current README and analysis scripts.

Included result sets:

| Path | Contents |
|---|---|
| `fp16_matmul_pjbit_gpu0_default_clock_cuda132_20260528_1331/` | FP16 Tensor Core matmul logical pJ/bit experiment |
| `env_gpu0_cuda132_conda_20260528_1335.txt` | GPU, driver, OS, and CUDA compiler environment snapshot |
| `ncu_gpu0_default_clock_cuda132/fp16_half2.ncu.txt` | Nsight Compute attempt log; profiling counters were blocked by `ERR_NVGPUCTRPERM` |

The RTX 3090 WSL2/WDDM driver rejected clock locking, so these results are default/boost-clock measurements rather than fixed-clock measurements.
