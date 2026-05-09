# Multi-GPU Component Energy Modeling Validation Plan

## 1. 목적과 환경 전제

이 문서는 `util/gpu_power_bench/`가 RTX 3090, A100 SXM 80GB, H100 SXM 80GB의 주요 컴포넌트별 에너지 계수를 측정하고, 그 계수로 workload energy를 재구성했을 때 measured energy와 얼마나 맞는지 검증하기 위한 실행 계획이다.

전제:

- 로컬 개발 GPU는 RTX 3090이다.
- 실제 headline 비교 대상은 RTX 3090, A100 SXM 80GB HBM2E, H100 SXM 80GB HBM3이다.
- RTX 3090에서는 native FP8 Tensor Core, H100 FP8 Transformer Engine, H100/A100 L2/HBM 특성을 검증할 수 없다. RTX 3090 결과는 local smoke와 consumer-Ampere/GDDR6X reference로만 사용한다.
- A100 SXM은 80GB HBM2E/HGX-class baseline이다. native FP8 Tensor Core headline은 만들지 않는다.
- H100 SXM은 80GB HBM3/native FP8 Transformer Engine headline 대상이다.
- 모든 결과는 board-level NVML 기반이므로, 일부 하위 컴포넌트는 "개별 회로 에너지"가 아니라 "traffic path / execution path energy"로 해석한다.

최종 목표:

1. 세 GPU 각각에 대해 `P_static`, leakage, SM/Tensor Core compute, HBM/GDDR read/write, L2 hit path, 주요 nonlinear/fused op의 계수를 얻는다.
2. 계수별 신뢰도와 한계를 명시한다.
3. H100은 native FP8 path까지 포함해 measured workload energy를 component model로 재구성하고, delta를 수치와 plot으로 보고한다.
4. RTX 3090/A100/H100 사이의 memory generation, L2 capacity, TDP/power envelope 차이가 결과 해석에 반영되었는지 검증한다.

구현 요구사양은 `docs/MULTI_GPU_COMPONENT_ENERGY_REQUIREMENTS.md`에 따로 둔다. 핵심 구현 상태:

- `gpu_power_bench.py`의 기본 profile은 `--gpu-profile h100_sxm`이다.
- GPU profile은 `gpu_profiles.py`가 단일 기준이며 RTX 3090/A100 SXM/H100 SXM의 default dtype, L2 window, headline 가능 여부를 정의한다.
- 명시적인 `--suite`/`--cases`/legacy scope flag 없이 실행하면 기본 suite는 `full`이다. 따라서 H100에서 `./run_bench.sh --device 0`은 GPU 0 전체 component validation을 실행한다.
- `--suite full`과 `--suite all`은 모두 `elementwise/matmul/llm-matmul/dram/l2/soc`와 fused residual을 포함하는 전체 suite다. L2만 다시 돌릴 때만 `--suite l2` 또는 `--cases l2`를 쓴다.
- 각 run은 `_gpu_spec_snapshot.csv`와 row-level `headline_status`/`headline_reason`을 남긴다.
- `component_validation_report.py`는 table output과 image report output을 분리하고, PNG는 `00_`부터 category 순번으로 저장한다.
- report는 `pass/low_conf/not_headline/not_applicable/missing/fail`로 수치를 분류한다.

## 2. 대상 GPU spec matrix

아래 표는 실험 planning에 직접 영향을 주는 스펙만 정리한다. 실행 전 실제 장비에서 `nvidia-smi`, PyTorch device properties, NVML power limit을 다시 기록해야 한다.

| GPU | Arch / CC | Memory | Peak memory BW | L2 cache | TDP / power envelope | FP8 headline | 실험상 의미 |
|---|---:|---|---:|---:|---:|---|---|
| RTX 3090 | Ampere GA102 / sm_86 | 24GB GDDR6X | 936 GB/s | 6 MB | 350 W class | No | local smoke, GDDR6X reference. H100/A100 coefficient 대체 금지 |
| A100 SXM 80GB | Ampere GA100 / sm_80 | 80GB HBM2E | 2,039 GB/s | 40 MB | 400 W | No | HBM2E baseline, BF16/TF32/FP16 TC headline, no native FP8 |
| H100 SXM 80GB | Hopper GH100 / sm_90 | 80GB HBM3 | 3.35 TB/s | 50 MB | up to 700 W configurable | Yes | HBM3/native FP8/L2 headline target |

Spec references:

- [NVIDIA RTX 3090 product page](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/): 24GB GDDR6X, 384-bit memory interface, 10496 CUDA cores.
- [TechPowerUp RTX 3090 database](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622): 936 GB/s bandwidth, 6 MB L2, 350 W board power, sm_86.
- [NVIDIA A100 product specs](https://www.nvidia.com/en-us/data-center/a100/): A100 80GB SXM has 80GB HBM2E, 2,039 GB/s bandwidth, 400 W TDP.
- [NVIDIA CUDA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html): A100 L2 is 40 MB and H100 L2 is 50 MB.
- [NVIDIA H100 product specs](https://www.nvidia.com/en-us/data-center/h100/): H100 SXM has 80GB memory, 3.35 TB/s bandwidth, up to 700 W configurable TDP, native FP8 Tensor Core throughput.

Planning consequences:

- 기본 `--gpu-profile`은 `h100_sxm`이며 `--l2-window-mb` 기본값도 H100 중심 `16 24 32 40`이다. A100/RTX 3090 run은 `--gpu-profile a100_sxm` 또는 `--gpu-profile rtx3090`를 지정하면 GPU별 L2 window/delta가 자동 적용된다.
- `--cache-sweep`는 runtime L2 detection을 사용하므로 세 GPU 모두에서 더 안전하다. 단, manual L2 probe는 GPU별 window를 직접 준다.
- `--dram-bw-test` 결과는 RTX 3090에서는 GDDR6X path, A100/H100에서는 HBM full-stack path이다. 같은 "pJ/bit"라도 memory technology가 다르므로 plot과 table에서 분리한다.
- A100의 `fp8:te` row는 native FP8가 아니다. 실행되더라도 fallback/emulated sanity row로만 둔다.
- H100 power cap이 configurable이므로 TDP를 상수로 가정하지 않는다. run metadata에 power limit과 clocks를 남긴다.
- A100/H100에서 MIG가 켜져 있으면 L2/HBM capacity와 power attribution이 달라질 수 있으므로 headline run은 full GPU 또는 MIG 상태를 명시한다.

실행 전 spec 확인 명령:

```bash
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm,clocks.max.memory,mig.mode.current --format=csv

python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print("name=", p.name)
print("cc=", f"{p.major}.{p.minor}")
print("total_memory_gb=", p.total_memory / 2**30)
print("l2_cache_mb=", getattr(p, "l2_cache_size", 0) / 2**20)
PY
```

## 3. 현재 커버리지와 누락 컴포넌트

### 이미 구현된 측정 축

| 컴포넌트/항 | 현재 구현 | 해석 |
|---|---|---|
| Idle/static | `measure_static_power`, `--rebaseline-every`, SoC static | board-level idle baseline |
| Thermal leakage | `--suite soc` / `--cases soc` | hot idle minus cold idle, board-level leakage delta |
| CUDA core compute | `matmul fp32:simt` | SIMT FP32 path coefficient |
| Tensor Core compute | `tf32:tc`, `fp16:tc`, `bf16:tc`, `fp8:te` | dtype별 GEMM execution path |
| Native FP8 TC | `fp8:te` via Transformer Engine | H100에서만 headline 가능 |
| Elementwise/nonlinear | `mul/add/softmax/gelu/layernorm` | standalone op energy, memory traffic 포함 |
| Fused nonlinear | `--suite full/all` 기본 포함, 또는 `--include-fused` | FlashAttention/linear+gelu/ln+linear residual decomposition |
| HBM traffic | `stream_read/write/copy/scale/triad`, DRAM marginal analysis | board-level HBM path pJ/bit |
| L2 hit path | `--cases l2` custom CUDA extension | L2-hit traffic path pJ/bit, isolated SRAM bit-cell 아님 |
| Drift/noise quality | baseline/rebaseline CSV, bootstrap CI, clip-bias, R2 | coefficient quality gate |

GPU별 headline 가능 여부:

| 항 | RTX 3090 | A100 SXM 80GB | H100 SXM 80GB |
|---|---|---|---|
| Static/leakage | 가능, local reference | headline | headline |
| FP32 SIMT / TF32 TC / FP16 TC | 가능 | headline | headline |
| BF16 TC | 사전 확인 필요 | headline | headline |
| Native FP8 TE | 불가 | 불가 | headline |
| HBM/GDDR pJ/bit | GDDR6X reference | HBM2E headline | HBM3 headline |
| L2 hit path | small-L2 smoke/reference | A100 40MB headline | H100 50MB headline |
| Fused nonlinear | FP16 중심 smoke | FP16/BF16 headline | FP16/BF16/FP8 attention headline |

### 주요 누락 또는 별도 해석이 필요한 컴포넌트

| 항 | 상태 | 계획 |
|---|---|---|
| L1 cache | NVML만으로 직접 분리 불가 | Nsight Compute counter-only validation으로 traffic sanity check. Power run과 분리 |
| Shared memory | 직접 계수 없음 | 필요 시 별도 CUDA microbench 설계. 1차 H100 model에서는 SM local path에 bundled |
| Register file | 직접 계수 없음 | `reg_spin`은 L2 baseline용 control cost이며 register energy 계수로 해석 금지 |
| Instruction front-end / scheduler / issue | 직접 분리 없음 | compute path intercept/control overhead로 bundled |
| On-chip NoC / L2 fabric | L2 probe에 포함 | "L2-hit traffic path"로 보고 |
| HBM controller vs PHY vs stack | HBM probe에 포함 | "HBM full-stack board-level pJ/bit"로 보고 |
| NVLink / PCIe | 현재 workload 범위 밖 | multi-GPU/data-transfer energy가 필요할 때 별도 phase 추가 |
| Fan/VRM/system power | NVML board boundary 일부 포함/일부 제외 | 모든 headline에 measurement boundary 명시 |

결론: GPU energy model v1에는 `static`, `thermal leakage`, `compute path`, `memory path`, `L2 path`, `standalone/fused nonlinear residual`까지 포함한다. L1/SMEM/register/front-end는 v1에서 개별 계수로 주장하지 않고, SM-local 또는 residual 항으로 둔다.

## 4. 코드/문서 Double Check 계획

### 4.1 로컬 RTX 3090에서 수행할 검증

목적은 H100 전용 경로를 실행하는 것이 아니라, 코드가 깨지지 않고 사용자가 안전하게 실행 가능한지 확인하는 것이다.

필수:

```bash
cd util/gpu_power_bench

python3 -m py_compile \
  benchmarks.py \
  gpu_power_bench.py \
  analyze.py \
  power_monitor.py \
  preflight.py \
  soc_power_bench.py

python3 gpu_power_bench.py --help
python3 analyze.py --help
python3 preflight.py --help
```

RTX 3090에서 가능한 smoke:

```bash
./run_bench.sh \
  --gpu-profile rtx3090 \
  --cases elementwise matmul dram soc \
  --dtypes fp16 \
  --matmul-variants fp32:simt tf32:tc fp16:tc \
  --quick \
  --window-ms 1500 \
  --no-cooldown \
  --tag rtx3090_nonfp8_smoke
```

금지/비권장:

- RTX 3090에서 `fp8:te` 결과를 H100 FP8 energy로 해석하지 않는다.
- RTX 3090에서 `--suite l2` 결과를 H100 L2 계수로 해석하지 않는다. L2 크기와 cache behavior가 다르다.
- RTX 3090에서 BF16/FP8 fallback 또는 emulated row를 native H100 coefficient로 사용하지 않는다.

RTX 3090에서 L2 smoke를 꼭 하고 싶다면 H100 default를 쓰지 말고 작은 window를 쓴다.

```bash
./run_bench.sh \
  --gpu-profile rtx3090 \
  --cases l2 \
  --l2-window-mb 1 2 3 4 \
  --l2-repeat-inner auto \
  --l2-target-energy-j 3 \
  --l2-delta-kb 0 64 256 1024 \
  --window-ms 3000 \
  --tag rtx3090_l2_smoke
```

### 4.2 코드 연결성 확인 항목

검토 대상:

- `gpu_power_bench.py`
  - `--cases`와 `--suite`가 `elementwise/matmul/llm-matmul/dram/l2/soc`를 일관되게 선택하는지 확인한다.
  - FP8/TE 실패 시 `emulated`, `path_semantics`, `notes`, `broken_variants`가 사용자를 속이지 않는지 확인한다.
  - L2 row의 `working_set_bytes`, `repeat_inner`, `estimated_l2_*_bits`, `l2_policy`가 CSV에 빠짐없이 기록되는지 확인한다.
- `benchmarks.py`
  - RTX 3090 fallback과 H100 native path가 명확히 구분되는지 확인한다.
  - L2 CUDA extension이 warmup/build overhead를 측정 구간 밖으로 빼는지 확인한다.
  - `BenchSpec.extra`가 non-L2 row에 부작용을 만들지 않는지 확인한다.
- `analyze.py`
  - FP8 emulated row filtering이 H100 native FP8 row까지 숨기지 않는지 확인한다.
  - HBM direct/marginal/read-write split이 simple memory-bound ops에만 headline을 주는지 확인한다.
  - L2 분석이 `reg_spin` baseline 누락, 음수 delta, insufficient R sweep을 `FAIL/LOW_CONF`로 표시하는지 확인한다.
  - plot 파일명이 README/TestCases와 일치하는지 확인한다.
- 문서
  - `README.md`, `TestCases.md`, `docs/REVIEW.md`가 L2 추가 후의 실제 구현과 맞는지 확인한다.
  - "SRAM bit-cell energy"처럼 과도한 표현이 남아 있지 않은지 확인한다.

## 5. GPU별 실행 계획

### 5.1 공통 사전 조건

각 GPU 환경에서 확인:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

필수 조건:

- compute capability가 spec matrix와 맞아야 한다.
- `transformer_engine[pytorch]`가 정상 import되고 GPU가 `sm_90` 계열이어야 native H100 FP8 결과로 인정한다.
- A100/H100 MIG 사용 여부를 기록한다. MIG이면 L2 capacity, persisting L2, power telemetry 해석이 달라질 수 있다.
- 가능하면 persistence mode, application clocks, thermal 상태를 고정하거나 로그에 남긴다.

### 5.2 RTX 3090 실행 순서

RTX 3090은 local integration/reference 용도다. `fp8:te`, H100 L2/HBM headline, A100/H100 delta 판단에는 사용하지 않는다.

```bash
cd util/gpu_power_bench
python3 preflight.py

./run_bench.sh \
  --gpu-profile rtx3090 \
  --cases elementwise matmul dram soc \
  --dtypes fp16 \
  --matmul-variants fp32:simt tf32:tc fp16:tc \
  --quick \
  --window-ms 1500 \
  --no-cooldown \
  --tag rtx3090_nonfp8_smoke
```

### 5.3 A100 SXM 80GB 실행 순서

A100은 HBM2E baseline과 Ampere Tensor Core baseline이다. native FP8 headline을 만들지 않는다.

1. Preflight/static sanity:

```bash
cd util/gpu_power_bench
python3 preflight.py
./run_bench.sh --gpu-profile a100_sxm --suite smoke --device 0 --dtypes fp16 --tag a100_sxm_smoke
```

2. Static/leakage/SoC envelope:

```bash
./run_bench.sh \
  --gpu-profile a100_sxm \
  --suite soc \
  --device 0 \
  --soc-static-seconds 20 \
  --tag a100_sxm_soc
```

3. Compute path coefficients:

```bash
./run_bench.sh \
  --gpu-profile a100_sxm \
  --cases matmul llm-matmul \
  --matmul-variants fp32:simt tf32:tc fp16:tc bf16:tc \
  --llm-dtypes bf16:tc \
  --llm-shapes \
  --window-ms 6000 \
  --rebaseline-every 10 \
  --tag a100_sxm_compute
```

4. HBM2E read/write and marginal memory:

```bash
./run_bench.sh \
  --gpu-profile a100_sxm \
  --cases dram elementwise \
  --dtypes fp16 \
  --cache-sweep \
  --dram-bw-test \
  --window-ms 6000 \
  --rebaseline-every 10 \
  --tag a100_sxm_hbm2e
```

5. A100 L2 hit traffic path:

```bash
./run_bench.sh \
  --gpu-profile a100_sxm \
  --cases l2 \
  --l2-window-mb 8 16 24 32 \
  --l2-repeat-inner auto \
  --l2-target-energy-j 10 \
  --l2-delta-kb 0 64 256 1024 4096 8192 \
  --window-ms 8000 \
  --rebaseline-every 10 \
  --tag a100_sxm_l2
```

6. Fused nonlinear residuals focused rerun:

`--suite full/all`에는 fused가 기본 포함된다. 아래 명령은 full run 후 fused 의존성 또는 dtype별 residual만 좁혀 다시 확인할 때 사용한다.

```bash
./run_bench.sh \
  --gpu-profile a100_sxm \
  --cases elementwise matmul \
  --include-fused \
  --dtypes fp16 \
  --fused-dtypes fp16 bf16 \
  --matmul-variants fp16:tc bf16:tc \
  --window-ms 6000 \
  --rebaseline-every 10 \
  --tag a100_sxm_fused
```

### 5.4 H100 SXM 80GB 실행 순서

1. Preflight/static sanity:

```bash
cd util/gpu_power_bench
python3 preflight.py
./run_bench.sh --gpu-profile h100_sxm --suite smoke --device 0 --dtypes fp16 --tag h100_sxm_smoke
```

2. Static/leakage/SoC envelope:

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --suite soc \
  --device 0 \
  --soc-static-seconds 20 \
  --tag h100_sxm_soc
```

3. Compute path coefficients:

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --cases matmul llm-matmul \
  --matmul-variants fp32:simt tf32:tc fp16:tc bf16:tc fp8:te \
  --llm-dtypes bf16:tc fp8:te \
  --llm-shapes \
  --window-ms 6000 \
  --rebaseline-every 10 \
  --tag h100_sxm_compute
```

4. HBM read/write and marginal memory:

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --cases dram elementwise \
  --dtypes fp16 \
  --cache-sweep \
  --dram-bw-test \
  --window-ms 6000 \
  --rebaseline-every 10 \
  --tag h100_sxm_hbm3
```

5. L2 hit traffic path:

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --cases l2 \
  --l2-window-mb 16 24 32 40 \
  --l2-repeat-inner auto \
  --l2-target-energy-j 10 \
  --l2-delta-kb 0 64 256 1024 4096 8192 16384 \
  --window-ms 8000 \
  --rebaseline-every 10 \
  --tag h100_sxm_l2
```

6. Fused nonlinear residuals focused rerun:

`--suite full/all`에는 fused가 기본 포함된다. 아래 명령은 full run 후 H100 fp8 attention fused 경로만 좁혀 다시 확인할 때 사용한다.

```bash
./run_bench.sh \
  --gpu-profile h100_sxm \
  --cases elementwise matmul \
  --include-fused \
  --dtypes fp16 fp8 \
  --fused-dtypes fp16 bf16 fp8 \
  --matmul-variants fp16:tc bf16:tc fp8:te \
  --window-ms 6000 \
  --rebaseline-every 10 \
  --tag h100_sxm_fused
```

## 6. 분석과 Visualization 검증

각 run 후:

```bash
python3 analyze.py --reports-dir reports --tag a100_sxm_compute
python3 analyze.py --reports-dir reports --tag a100_sxm_hbm2e
python3 analyze.py --reports-dir reports --tag a100_sxm_l2
python3 analyze.py --reports-dir reports --tag a100_sxm_fused
python3 analyze.py --reports-dir reports --tag a100_sxm_soc

python3 analyze.py --reports-dir reports --tag h100_sxm_compute
python3 analyze.py --reports-dir reports --tag h100_sxm_hbm3
python3 analyze.py --reports-dir reports --tag h100_sxm_l2
python3 analyze.py --reports-dir reports --tag h100_sxm_fused
python3 analyze.py --reports-dir reports --tag h100_sxm_soc
```

필수 plot/sidecar 확인:

| 목적 | 산출물 | 합격 기준 |
|---|---|---|
| Static/drift | baseline/rebaseline plot/CSV | idle std가 작고 drift가 rebaseline으로 추적됨 |
| Compute | J/FLOP bar, K/T scaling plot | H100 native FP8 row가 emulated로 표시되지 않고, A100/RTX FP8는 headline에서 제외 |
| HBM/GDDR | DRAM pJ/bit, read/write split, marginal plot | RTX GDDR6X, A100 HBM2E, H100 HBM3를 분리 표시 |
| L2 | `_02_l2_summary.csv`, L2 overview/fit/stability plots | GPU별 L2 window가 spec에 맞고 read/write PASS 또는 LOW_CONF |
| Fused | fused decomposition CSV/plot | standalone nonlinear와 fused residual이 분리되어 표시됨 |
| MECE | energy decomposition plot | component 합이 measured total과 일치하고 caveat가 plot에 보임 |

추가로 필요한 visualization:

- `gpu_spec_matrix.csv`: actual run-time GPU spec snapshot.
- `image_report/00_component_coverage_matrix.png`: row=component, col=GPU/experiment, cell=pass/low_conf/not_headline/not_applicable/missing/fail.
- `image_report/01_model_vs_measured_scatter_by_gpu.png`: x=measured dynamic energy, y=model-predicted dynamic energy.
- `image_report/02_delta_by_gpu_and_workload.png`: GPU/workload별 `(model - measured) / measured`.
- `image_report/03_component_reconstruction_residual.png`: static/compute/HBM-GDDR/L2/nonlinear로 분류한 뒤 남는 residual.
- `coefficient_confidence_table.csv`: coefficient, GPU, CI, R2, source run, status, caveat.

`component_validation_report.py`가 현재 독립 post-processing entrypoint다. Benchmark 실행 경로를 바꾸지 않고, 생성된 `gpu_power_bench_*.csv`와 sidecar를 모아 spec matrix, coefficient confidence table, coverage matrix, row-level model-vs-measured delta, component reconstruction residual을 만든다.

```bash
python3 component_validation_report.py \
  --reports-dir reports \
  --out-dir reports/component_validation \
  --image-dir reports/component_validation/image_report \
  --tags rtx3090_nonfp8_smoke a100_sxm h100_sxm \
  --capture-runtime-spec
```

## 7. 구성 검증과 Delta 산출

### 7.1 계수 테이블 생성

각 분석 결과에서 다음 형태의 coefficient table을 만든다.

| coefficient | unit | source | required status |
|---|---|---|---|
| `P_static` | W | baseline/rebaseline | PASS |
| `P_leak_hot_delta` | W | soc | PASS/LOW_CONF |
| `k_fp32_simt` | J/FLOP | matmul | PASS |
| `k_tf32_tc` | J/FLOP | matmul | PASS |
| `k_fp16_tc` | J/FLOP | matmul | PASS |
| `k_bf16_tc` | J/FLOP | matmul | PASS |
| `k_fp8_te` | J/FLOP | matmul/llm | PASS, native H100 only |
| `k_mem_read` | J/bit | stream_read | PASS, memory type recorded |
| `k_mem_write` | J/bit | stream_write | PASS, memory type recorded |
| `k_l2_read_hit_path` | J/bit | l2_read_hit | PASS/LOW_CONF |
| `k_l2_write_hit_path` | J/bit | l2_write_hit | PASS/LOW_CONF |
| `k_softmax_fused_residual` | J/op or J/elem | fused | PASS/LOW_CONF |
| `k_gelu_fused_residual` | J/op or J/elem | fused | PASS/LOW_CONF |
| `k_layernorm_fused_residual` | J/op or J/elem | fused | PASS/LOW_CONF |

### 7.2 재구성 수식

측정 workload row 또는 대표 workload에 대해:

```text
E_model =
  P_static * T
  + E_leakage_correction(T, temp)
  + sum(FLOP_path_i * k_compute_i)
  + memory_read_bits * k_mem_read
  + memory_write_bits * k_mem_write
  + L2_read_bits * k_l2_read_hit_path
  + L2_write_bits * k_l2_write_hit_path
  + nonlinear_residual_terms
```

Delta:

```text
delta_j = E_model - E_measured
delta_pct = 100 * delta_j / E_measured
```

보고 기준:

- `|delta_pct| <= 10%`: v1 model 적합.
- `10% < |delta_pct| <= 20%`: 사용 가능하나 missing component 또는 counter mismatch 조사.
- `|delta_pct| > 20%`: component accounting 또는 measurement 품질 문제로 FAIL.

### 7.3 Delta 원인 분해

큰 delta가 나오면 다음 순서로 조사한다.

1. `P_static` drift: rebaseline trace와 clip-bias 확인.
2. Native/emulated mismatch: `emulated`, `path_semantics`, GPU CC 확인.
3. HBM traffic mismatch: STREAM 계수와 workload logical bytes가 맞는지 확인.
4. L2 traffic mismatch: logical L2 bits와 Nsight Compute `lts__t_sectors` counter-only run 비교.
5. Thermal leakage: hot run에서 cold static만 뺀 영향 확인.
6. Missing local path: L1/SMEM/register/front-end residual로 분류.

GPU별 comparison rule:

- RTX 3090 vs A100/H100의 memory coefficient는 GDDR6X vs HBM2E/HBM3 차이로 분리해서 본다.
- A100 vs H100의 memory delta는 HBM2E 2,039 GB/s vs HBM3 3.35 TB/s 차이를 고려하되, pJ/bit가 bandwidth ratio와 정확히 같은 비율로 변한다고 가정하지 않는다.
- A100 vs H100의 compute delta는 Tensor Core generation과 FP8 availability를 분리해서 본다.
- H100 FP8 delta는 native row만 사용하고, A100/RTX에는 `not_applicable_native_fp8`로 표기한다.

## 8. Nsight Compute Counter Validation

Power run과 counter run은 분리한다. NCU는 replay/cache/clock behavior를 바꿀 수 있으므로 power headline에는 섞지 않는다.

권장 방식:

- GPU별 power run CSV에서 대표 cells를 고른다.
- 동일 shape/flag로 짧은 NCU run을 별도 수행한다.
- 다음 counter를 logical estimate와 비교한다.
  - L2: `lts__t_sectors.sum` 계열
  - DRAM: `dram__sectors_read.sum`, `dram__sectors_write.sum` 계열
  - Tensor Core: `smsp__inst_executed_pipe_tensor.sum` 계열
  - FP32/SIMT: `smsp__inst_executed_pipe_fma.sum` 또는 유사 계열
- counter mismatch가 크면 coefficient 자체가 아니라 traffic accounting을 먼저 수정한다.

## 9. 문서와 사용성 개선 계획

사용자가 쉽게 실행하려면 다음 문서 정리가 필요하다.

- README 첫 실행 섹션에 "RTX 3090 local smoke / A100 SXM HBM2E / H100 SXM HBM3"를 분리한다.
- TestCases의 A.6 L2 설명에 "GPU별 L2 window override"를 추가한다.
- REVIEW의 Axis 3 summary를 L2 추가 이후 상태로 갱신한다. 현재 큰 방향은 맞지만 "DRAM까지만 깨끗" 표현은 L2 probe 추가 후 조정이 필요하다.
- `run_bench.sh --suite component-smoke`, `--suite a100-component`, `--suite h100-component` 같은 preset 추가를 검토한다.
- GPU별 workflow용 helper script를 별도 추가할지 검토한다. 예: `run_a100_component_plan.sh`, `run_h100_component_plan.sh`.

## 10. Acceptance Criteria

계획 완료 조건:

- 로컬 RTX 3090에서 syntax/help/non-FP8 smoke가 통과한다.
- A100 SXM 80GB에서 `soc`, `compute`, `hbm2e`, `l2`, `fused` run이 완료되고 analyze sidecar가 생성된다.
- H100 SXM 80GB에서 `soc`, `compute`, `hbm3`, `l2`, `fused` run이 완료되고 analyze sidecar가 생성된다.
- 모든 headline coefficient가 source, unit, status, CI/R2/caveat와 함께 정리된다.
- Native FP8 headline은 H100 `sm_90` + Transformer Engine 성공 row만 사용한다.
- L2 headline은 GPU별 L2 capacity에 맞는 window로 산출하고 "L2-hit traffic path"로 표기하며 isolated SRAM energy로 부르지 않는다.
- GPU별 model-vs-measured delta table과 plot이 생성된다.
- delta가 큰 workload는 missing component/residual 원인으로 분류된다.

## 11. 우선순위

P0:

- RTX 3090/A100 SXM/H100 SXM 실행 경로 분리 문서화.
- A100/H100 smoke + compute/HBM/L2/SoC 실행.
- L2/FP8 native 여부와 emulated row labeling 검증.
- coefficient table과 model-vs-measured delta 산출.

P1:

- component coverage matrix와 delta waterfall plot 추가.
- REVIEW.md의 memory hierarchy 평가를 L2 추가 이후 기준으로 갱신.
- H100 workflow preset 또는 helper script 추가.

P2:

- NCU counter-only validation importer 추가.
- L1/SMEM/register residual을 별도 "unmodeled local path"로 추정하는 리포트 추가.
- leakage temperature correction을 per-cell model reconstruction에 통합.

P3:

- NVLink/PCIe transfer energy microbench 추가.
- multi-GPU scaling energy model과 cross-GPU comparison 자동화.
