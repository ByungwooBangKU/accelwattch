# RTX 3090 strict FP16 launch-shape diagnostic, 2026-06-02

This run targeted an NVIDIA GeForce RTX 3090 (`sm_86`, GA102) with the strict FP16 Tensor Core matrix configuration.

Status: diagnostic, not strict-final. The build, calibration, runtime sweep, analysis, quality gate, and resource audit completed. Nsight Compute no-L2 validation failed because the host denied GPU performance counter access (`ERR_NVGPUCTRPERM`), so this result does not prove the no-L2 condition with hardware counters.

Sweep:

- test kernel: `tensor_mma_f16acc`
- baseline kernel: `tensor_baseline_mov`
- logical MMA denominator: `m16n16k16`, 8192 input bits per logical MMA
- output stores: suppressed in the timed loop
- threads/block: `32,64,128,256`
- blocks/SM: `1,2,4,8`
- repeats: 10 per launch shape

Quality-gated diagnostic target:

- threads/block: 256
- blocks/SM: 1
- threads/SM: 256
- target selection metric: `tensor_model_utilization_pct_mean`
- Tensor Core model utilization: 98.3395%
- TFLOPS: 159.148
- FP16 matmul input energy: 0.308459 pJ/bit
- valid no-L2 metadata count: 10/10
- NCU validation: missing/failed due counter permission

Lowest pJ/bit diagnostic candidate:

- threads/block: 256
- blocks/SM: 8
- threads/SM: 2048
- Tensor Core model utilization: 98.235%
- TFLOPS: 158.621
- FP16 matmul input energy: 0.074226 pJ/bit
- valid no-L2 metadata count: 5/10

Use `quality_gate_summary.json`, `thread_sweep_summary.csv`, `resource_audit/thread_resource_occupancy.csv`, and `ncu_no_l2_thread_sweep/ncu_run_failures.csv` as the primary evidence files. The core figures are under `figures/`.
