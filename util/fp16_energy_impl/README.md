# GPU FP16 Incremental Compute Energy Benchmark

이 구현은 `GPU FP16 순수 연산 에너지 추출 실험 설계서`의 P0/P1 범위를 코드로 옮긴 것이다. 목표는 GPU 내부 FP16 unit만의 물리적 절대 에너지를 직접 측정하는 것이 아니라, register-resident 또는 fragment-resident FP16 compute workload에서 관찰되는 **baseline-subtracted incremental FP16 compute energy estimate**를 산출하는 것이다.

## 1. 구성

| 경로 | 역할 |
|---|---|
| `src/fp16_energy_bench.cu` | CUDA microbenchmark binary. FP16 half2, Tensor Core MMA, baseline, memory policy kernel 포함 |
| `configs/primary_matrix.json` | P0 primary FP16 실험 matrix |
| `configs/fp16_matmul_pjbit_matrix.json` | Tensor Core FP16 matmul logical pJ/bit 전용 matrix |
| `configs/fp16_matmul_thread_sweep.json` | L2/global output store를 억제한 Tensor Core thread-count sweep matrix |
| `configs/fp16_matmul_thread_sweep_fine.json` | 낮은 thread 후보까지 포함한 Tensor Core fine thread-count sweep matrix |
| `configs/fp16_matmul_thread_sweep_low_append.json` | 기존 fine sweep 결과 디렉터리에 32/64/96/128 후보를 append하는 matrix |
| `configs/p1_memory_policy_matrix.json` | P1 memory/cache policy 보정용 matrix |
| `scripts/run_experiment.py` | benchmark 실행 + `nvidia-smi` power/clock/temp 및 dmon SM utilization logging |
| `scripts/analyze_results.py` | NVML energy counter 우선 분석, power trace fallback, baseline subtraction, pJ/FLOP 계산, CSV/시각화 생성 |
| `scripts/quality_gate.py` | 결과 채택 전 energy source, valid no-L2 반복 수, clock 안정성, SM utilization 포화 여부를 gate |
| `scripts/compare_architectures.py` | A100/H100/RTX3090 등 여러 결과 디렉터리의 FP16 energy/throughput/thread-sweep 비교 시각화 |
| `scripts/ncu_validate.sh` | Nsight Compute validation run 예시 |
| `scripts/ncu_validate_no_l2_thread_sweep.sh` | thread sweep 후보의 no-L2/global-memory validation run 예시 |
| `scripts/lock_clocks.sh` | GPU clock lock helper |
| `scripts/reset_clocks.sh` | GPU clock reset helper |
| `scripts/query_env.sh` | 실험 환경 metadata 수집 |

## 2. 구현된 kernel

| 우선순위 | Kernel | 목적 | Memory policy |
|---|---|---|---|
| P0 | `fp16_half2` | CUDA core 기반 `half2` FMA 반복 | timed loop 내부 global/shared memory 접근 없음 |
| P0 | `baseline_nop` | loop/control baseline | 동일 launch/loop 구조, FP16 연산 없음 |
| P0 | `baseline_regmove` | register/integer movement baseline | FP16 연산 없음, integer/register overhead 참고 |
| P0 | `tensor_mma_f16acc` | logical `m16n16k16` FP16 input + FP16 accumulate | timed loop 내부 register operand 기반 |
| P0 | `tensor_mma_f32acc` | logical `m16n16k16` FP16 input + FP32 accumulate | timed loop 내부 register operand 기반 |
| P0 | `tensor_baseline_u32`, `tensor_baseline_f32` | Tensor kernel용 baseline | MMA 없음 |
| P1 | `memory_default`, `memory_cg`, `memory_cs` | residual L1/L2/DRAM 보정 및 cache policy sanity check | default / `.cg` / `.cs`만 제한 반영 |

`fp16_half2`의 연산량 계산은 다음과 같다.

```text
N_FP16_ops = blocks × threads × repeats × iters × unroll × 4 half2-FMA × 4 FLOP/half2-FMA
```

Tensor Core kernel은 Ampere/Hopper에서 지원되는 `mma.sync.aligned.m16n8k16` 두 번으로 logical `m16n16k16` tile 하나를 구성한다. 따라서 logical MMA 1회당 `2 × 16 × 16 × 16 = 8192` FP16 FLOP으로 계산한다.

```text
N_FP16_ops = warps × repeats × iters × unroll × 8192
```

## 3. Build

```bash
cd fp16_energy_impl
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
cmake --build build -j
```

대상 GPU별 architecture는 일반적으로 다음과 같이 둘 수 있다.

| GPU | CUDA architecture |
|---|---:|
| A100 SXM4 | 80 |
| RTX 3090 | 86 |
| H100 SXM5 | 90 |

한 장비에서 하나의 GPU만 측정할 경우 빌드 시간을 줄이기 위해 해당 architecture만 지정해도 된다.

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="90"
```

## 4. A100/H100 실행 범위와 자동화 범위

이 코드는 A100/H100에서도 실행 가능하다. GPU별 권장 CUDA architecture는 A100 `80`, RTX 3090 `86`, H100 `90`이다.

```bash
# A100
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j

# H100
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

실험 runner가 자동으로 처리하는 항목은 다음과 같다.

1. `--blocks 0`일 때 GPU SM 개수를 감지해 `blocks = SM_count * blocks_per_sm`로 설정한다.
2. matrix에 정의된 baseline/test 조건을 순서대로 실행한다.
3. `--repeat N`으로 전체 matrix를 N회 반복한다.
4. benchmark timed loop 직전/직후 `nvmlDeviceGetTotalEnergyConsumption()` 누적 에너지 카운터를 읽는다. 이 기능은 `libnvidia-ml.so.1`을 동적으로 load하므로 NVML header/link dependency 없이 빌드된다.
5. run별 `nvidia-smi` power/clock/temperature trace와 dmon SM utilization trace를 수집한다.
6. `analyze_results.py`가 `summary.csv`, `condition_summary.csv`, thread sweep summary, figure를 생성한다.

수동으로 맞춰야 하는 항목은 다음과 같다.

1. CMake CUDA architecture 선택.
2. GPU clock 고정 및 실제 clock 안정성 확인.
3. 다른 GPU process 제거와 실험 환경 격리.
4. Nsight Compute performance counter 접근 권한 설정.
5. GPU별 `iters`/`repeats` 조정으로 power sample 수 확보.

H100에서도 현재 Tensor Core kernel은 `mma.sync.m16n8k16` 두 개로 logical `m16n16k16`을 만드는 warp-level 경로를 사용한다. 따라서 A100/H100/RTX 3090 간 같은 HMMA 계열 FP16 matmul path 비교에는 사용할 수 있지만, H100 고유 WGMMA/TMA 경로의 최대 matmul energy를 측정하는 실험은 아니다. H100 WGMMA 경로까지 측정하려면 별도 kernel과 validation matrix를 추가해야 한다.

### Energy source policy

최종 energy 계산은 timed loop 내부의 NVML 누적 에너지 카운터 delta를 우선 사용한다. 즉 `bench.json`의 `nvml_energy_supported=true`이고 `nvml_energy_delta_j > 0`이면 `power_energy_j`와 `avg_power_w`는 `nvmlDeviceGetTotalEnergyConsumption()` 기반 값이다. 카운터가 지원되지 않는 GPU/driver 조합에서는 기존 방식대로 `nvidia-smi --query-gpu=power.draw` trace를 host timed interval에 적분한 값을 fallback으로 사용한다.

`nvidia-smi` power trace는 제거하지 않는다. H100처럼 `power.draw`가 averaging/smoothing된 값을 줄 수 있는 환경에서는 NVML total energy counter가 더 직접적인 최종 에너지 값이고, power trace는 clock/temperature/throttling 및 counter-vs-trace sanity check 용도다. 분석 결과에는 `energy_source`, `power_trace_energy_j`, `nvml_energy_delta_j`, `energy_counter_vs_trace_delta_j`, `energy_counter_vs_trace_ratio`가 함께 기록된다.

### Quality gate policy

`analyze_results.py`가 만든 수치는 바로 최종값으로 채택하지 않고, `quality_gate.py`로 다음 조건을 확인한다.

```bash
python3 scripts/quality_gate.py --input results/fp16_matmul_thread_sweep_fine_gpu0
```

Gate가 확인하는 핵심 조건은 다음과 같다.

| Gate | 의미 |
|---|---|
| positive increment | baseline subtraction 뒤 incremental power/energy가 양수 |
| no intended L2/global traffic | `suppress_output_store=true`이고 `memory_bytes_estimate=0`이라 timed kernel이 global/L2 traffic을 의도하지 않음 |
| enough valid repeats | thread point별 `valid_no_l2_count >= max(3, ceil(run_count/2))` |
| stable clock | 기본값으로 `clock_span_mhz <= 60` |
| reliable energy source | `nvml_total_energy_counter` 우선. 미지원 시 power trace fallback은 최소 sample 수를 만족할 때만 diagnostic grade로 통과 |
| common instruction path | A100/H100/RTX3090 비교에서는 WGMMA가 아니라 공통 HMMA `mma.sync.m16n8k16` pair path |
| utilization target | SM utilization 최대값에서 0.1 percentage point 이내로 포화된 가장 작은 `threads_per_sm` |

출력은 `quality_gates.csv`, `quality_gate_summary.json`, `figures/quality_gate_thread_sweep_*.png`이다. `target_pass=true`인 row가 최종 thread-count 추천점이다. `measurement_grade=power_trace_fallback`은 기존 RTX 3090 결과처럼 NVML energy counter가 없는 legacy run을 의미하므로, A100/H100 최종 비교에서는 같은 matrix를 다시 실행해 `strict_nvml_counter` 결과를 우선 사용한다.

### Architecture comparison policy

A100, H100, RTX 3090 비교는 같은 logical workload와 같은 instruction family를 비교하는 방식으로 해석한다. 현재 Tensor Core kernel은 세 GPU 모두에서 warp-level `mma.sync.m16n8k16` 두 개를 묶어 logical `m16n16k16`을 만들며, H100에서 지원되는 WGMMA 경로를 사용하지 않는다. 따라서 이 결과는 "H100의 최대 WGMMA matmul efficiency"가 아니라 "A100/H100/RTX3090에서 공통 HMMA FP16 path를 같은 baseline subtraction으로 측정한 값"이다.

benchmark JSON과 분석 CSV에는 `architecture_generation`, `architecture_chip`, `recommended_cuda_arch`, `fp16_tensor_instruction_path`, `wgmma_supported`, `benchmark_uses_wgmma`가 기록된다. 오래된 결과 JSON도 `analyze_results.py`가 `device_name`과 `compute_capability`로 fallback 분류한다. Summary에는 `baseline_energy_fraction`, `incremental_energy_fraction`, `baseline_power_fraction`, `valid_no_l2`, `pure_fp16_candidate`, `separation_quality`가 추가되어, baseline이 너무 크거나 L2/global traffic이 예상되는 run을 pJ/bit 최종 후보에서 분리할 수 있다.

## 5. 실험 전 환경 수집

```bash
./scripts/query_env.sh 0 results/env_gpu0.txt
```

세 번째 인자로 binary path를 넘기면 CUDA runtime probe와 resource-usage dump도 같이 시도한다.

```bash
./scripts/query_env.sh 0 results/env_gpu0.txt build/fp16_energy_bench
```

`cuobjdump`가 설치되어 있으면 `query_env.sh` 출력에 kernel별 resource usage가 포함된다. 설치되어 있지 않은 환경에서는 CMake가 이미 `-Xptxas=-v`로 build output에 register 수와 spill 정보를 출력하므로, 다음처럼 build log를 남겨 확인한다.

```bash
cmake --build build -j 2 2>&1 | tee results/build_ptxas.log
rg "Used .* registers|spill" results/build_ptxas.log
```

가능하면 GPU clock을 고정한다.

```bash
sudo ./scripts/lock_clocks.sh 0 <SM_CLOCK_MHZ> <MEM_CLOCK_MHZ>
```

실험 후 원복한다.

```bash
sudo ./scripts/reset_clocks.sh 0
```

`nvidia-smi -q -d CLOCK`으로 가능한 clock 값을 확인한 뒤 실제 장비에 맞는 값을 넣는다. `-lgc`/`-lmc`가 제한되는 datacenter 환경에서는 application clocks 또는 cluster policy에 맞춰 동일한 목적의 clock 고정 절차를 사용한다.

## 6. P0 primary 실험 실행

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/primary_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/p0_gpu0
```

분석과 시각화는 다음 명령으로 생성한다.

```bash
python3 scripts/analyze_results.py --input results/p0_gpu0
```

`run_experiment.py`는 기존 `runs.jsonl`이 있는 outdir에는 기본적으로 쓰지 않는다. 같은 outdir에 의도적으로 run을 누적할 때만 `--append`를 추가한다.

For FP16 Tensor Core matmul logical pJ/bit estimates:

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_pjbit_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/fp16_matmul_pjbit_gpu0

python3 scripts/analyze_results.py --input results/fp16_matmul_pjbit_gpu0
```

For FP16 Tensor Core thread-count sweep under no intended L2/global traffic:

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_thread_sweep.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/fp16_matmul_thread_sweep_gpu0

python3 scripts/analyze_results.py --input results/fp16_matmul_thread_sweep_gpu0
```

이 sweep은 `threads = 32, 64, 128, 256, 512, 1024`를 훑는다. Matrix default의 `suppress_output_store=true`가 test와 `baseline_nop` timed kernel의 final global store를 끄므로, 의도된 timed-loop L2/global memory traffic 없이 Tensor Core utilization만 비교한다. 분석기는 `thread_sweep_summary.csv`를 만들고, `valid_basic=True`이며 `expected_l2_touch=False`인 후보 중 dmon `avg_sm_util_pct_mean`이 포화되는 가장 작은 `threads_per_sm` point를 `selected_optimal=True`로 표시한다. SM utilization이 없으면 `avg_gpu_util_pct_mean`으로 fallback한다.

Coarse sweep에서 유효 후보가 좁혀지면 fine sweep을 추가로 실행한다. Fine matrix는 `threads = 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384`를 훑는다. 160 이상 후보는 test `repeats=2`, baseline `repeats=20`을 사용한다. 32/64/96/128 후보는 duration과 sample 수를 맞추기 위해 각각 더 큰 repeats를 사용한다.

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_thread_sweep_fine.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/fp16_matmul_thread_sweep_fine_gpu0

python3 scripts/analyze_results.py --input results/fp16_matmul_thread_sweep_fine_gpu0
python3 scripts/quality_gate.py --input results/fp16_matmul_thread_sweep_fine_gpu0
```

여러 반복이 있는 thread sweep에서는 `valid_no_l2_count >= max(3, ceil(run_count/2))`인 후보를 우선 선택한다. 이 조건을 만족하는 후보가 없을 때만 더 약한 후보군으로 fallback한다. 선택 기준은 max SM utilization에서 0.1 percentage point 이내로 포화된 후보 중 가장 작은 `threads_per_sm`이며, 같은 `threads_per_sm`에서는 TFLOPS와 clock stability로 tie-break한다.

2026-05-28 RTX 3090 local run 결과는 다음과 같다. 기준 matmul은 logical `m16n16k16`이며, 구현은 `mma.sync.m16n8k16` instruction 두 개로 N 방향 16 columns를 채운다. pJ/bit denominator는 A/B logical FP16 input bit인 `(16*16 + 16*16) * 16 = 8192 bit/logical MMA`다. Nsight Compute counter validation은 이 환경에서 `ncu`가 PATH에 없어 수행하지 못했다.

| Sweep | 후보 threads/block | Selected threads/SM | Selected threads/block | valid no-L2 | Avg SM util (%) | TFLOPS | `matmul_input_pj_per_bit` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine + dmon SM util | 32,64,96,128,160,192,224,256,288,320,384 | 512 | 64 | 9/10 | 100.000 +/- 0.000 | 159.1 | 0.2514 +/- 0.0142 pJ/bit |

Fine + dmon run에서는 `threads_per_sm=512`에서 평균 SM utilization이 100%에 도달했다. 따라서 SM utilization 첫 포화점 기준 target thread count는 512 threads/SM, 즉 64 threads/block로 지정한다. pJ/bit 자체는 224 threads/block 후보에서 `0.0819 +/- 0.0121 pJ/bit`로 더 낮게 관찰되었지만, valid no-L2 count가 5/10이고 incremental power가 3.23 W 수준이라 target saturation point로 채택하지 않았다. 256 threads/block 이상 후보는 baseline subtraction 후 incremental power가 음수로 나와 `all_runs_no_valid`로 집계되었다. 이 값들은 board-level `nvidia-smi` power trace와 baseline subtraction 기반 estimate이므로, 최종 수치 채택 전에는 Nsight Compute로 no-L2/global-memory 조건과 HMMA instruction path를 별도 확인해야 한다.

여러 GPU에서 같은 matrix를 실행한 뒤 architecture-level 비교 figure를 생성하려면 각 결과 디렉터리에 대해 `analyze_results.py`를 먼저 실행하고, 다음처럼 묶는다.

```bash
python3 scripts/compare_architectures.py \
  --input results/fp16_matmul_thread_sweep_fine_a100 \
          results/fp16_matmul_thread_sweep_fine_h100 \
          results/fp16_matmul_thread_sweep_fine_rtx3090 \
  --outdir results/architecture_compare_fp16
```

주요 산출물은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `architecture_best_fp16.csv` | 결과 디렉터리별 pure FP16 후보 중 utilization/valid count 기준 best row |
| `architecture_best_matmul_input_pj_per_bit.png` | GPU architecture별 best logical matmul input pJ/bit 비교 |
| `architecture_best_tflops.png` | GPU architecture별 best FP16 throughput 비교 |
| `architecture_thread_sweep_util_*.png` | x축 launched threads/SM, y축 SM utilization의 multi-GPU 비교 |
| `architecture_thread_sweep_pjbit_*.png` | x축 launched threads/SM, y축 logical pJ/bit의 multi-GPU 비교 |

For memory/cache-policy calibration and DRAM pJ/bit estimates:

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/p1_memory_policy_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/p1_gpu0

python3 scripts/analyze_results.py --input results/p1_gpu0
```

생성물은 다음과 같다.

| 파일/디렉터리 | 내용 |
|---|---|
| `results/p0_gpu0/runs.jsonl` | run별 benchmark metadata |
| `results/p0_gpu0/raw/*/power.csv` | run별 power/clock/temp trace |
| `results/p0_gpu0/raw/*/sm_util.csv` | run별 dmon SM utilization trace |
| `results/p0_gpu0/raw/*/bench.json` | timed loop의 CUDA event timing과 optional NVML total-energy counter delta |
| `results/p0_gpu0/summary.csv` | baseline/test pair별 baseline subtraction 결과 |
| `results/p0_gpu0/condition_summary.csv` | condition별 반복 측정 통계(mean/std/min/max/95% CI) |
| `results/p0_gpu0/thread_sweep_summary.csv` | thread-count sweep일 때 thread별 utilization/TFLOPS 집계와 `selected_optimal` 표시 |
| `results/p0_gpu0/quality_gates.csv` | pair/thread point별 quality gate 통과 여부와 실패 이유 |
| `results/p0_gpu0/quality_gate_summary.json` | 선택된 target point와 gate threshold 요약 |
| `results/p0_gpu0/run_level_summary.csv` | run 단위 selected energy, NVML counter delta, power trace integration 결과 |
| `results/p0_gpu0/figures/pj_per_flop_bar.png` | pJ/FLOP bar chart |
| `results/p0_gpu0/figures/tflops_vs_pj_per_flop.png` | TFLOPS vs pJ/FLOP scatter |
| `results/p0_gpu0/figures/fp16_energy_separation_stack.png` | test interval energy를 baseline-scaled energy와 FP16 incremental energy로 분리한 stack plot |
| `results/p0_gpu0/figures/thread_sweep_*.png` | launched threads/SM별 SM utilization/TFLOPS plot |
| `results/p0_gpu0/figures/thread_sweep_pjbit_*.png` | launched threads/SM별 matmul logical pJ/bit plot. label은 threads/block와 pJ/bit 값 |
| `results/p0_gpu0/figures/quality_gate_thread_sweep_*.png` | quality gate pass/fail과 pJ/bit label이 포함된 thread sweep plot |
| `results/p0_gpu0/figures/power_trace_*.png` | baseline/test power trace 비교 |
| `results/p0_gpu0/figures/clock_*.png`, `temperature_*.png` | clock/temperature timeline |

## 7. P1 memory/cache policy 보정 실험

P0 결과에서 L1/L2/DRAM residual traffic이 관찰될 때만 아래 실험을 사용한다.

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/p1_memory_policy_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/p1_memory_gpu0

python3 scripts/analyze_results.py --input results/p1_memory_gpu0
```

이 구현은 cache operator sweep을 과도하게 늘리지 않는다. 보정용으로만 `default`, `.cg`, `.cs`를 제공한다. `.ca`는 기본 정책과 중복되기 쉬워 primary matrix에 넣지 않았고, `.cv`는 일반 device DRAM access 강제 수단으로 오해될 수 있어 제외했다.

## 8. Nsight Compute 검증

Power 실험과 별도로 짧은 validation run을 수행한다.

```bash
./scripts/ncu_validate.sh ./build/fp16_energy_bench results/ncu_gpu0 0
```

Thread sweep에서 선택된 후보가 L2/global memory를 의도적으로 touch하지 않는지 확인하려면 no-L2 validation을 별도로 수행한다.

```bash
./scripts/ncu_validate_no_l2_thread_sweep.sh \
  ./build/fp16_energy_bench \
  results/ncu_no_l2_thread_sweep_gpu0 \
  0 \
  32,64,128,256,512,1024
```

이 validation은 `--suppress-output-store`로 `tensor_mma_f16acc`와 `baseline_nop`의 final global store를 제거한 상태에서 Nsight Compute `MemoryWorkloadAnalysis`를 남긴다. GeForce/WSL 환경에서는 NVIDIA performance counter 권한 때문에 `ERR_NVGPUCTRPERM`으로 막힐 수 있다.

P0 결과 채택 기준은 최소한 다음을 확인해야 한다.

| 기준 | 확인 방법 |
|---|---|
| 의도한 FP16/HMMA instruction path | Nsight Compute `ComputeWorkloadAnalysis`, SASS 확인 |
| local memory spill 없음 | Nsight Compute memory/local 관련 항목, `ptxas -v` |
| timed loop 내부 L1/L2/DRAM traffic 최소 | Nsight Compute `MemoryWorkloadAnalysis` |
| clock 안정 | `power.csv`, `clock_*.png`, `summary.csv`의 `clock_span_mhz` |
| thermal/power throttling 없음 | `nvidia-smi -q`, power/temperature trace |
| 다른 process 없음 | `nvidia-smi pmon`, 실험 환경 격리 |

## 9. 결과 해석

`summary.csv`의 핵심 컬럼은 다음이다.

| 컬럼 | 의미 |
|---|---|
| `tflops` | CUDA event elapsed time 기준 FP16 throughput |
| `pair_index`, `repeat_index` | baseline/test pair 번호와 runner 반복 번호 |
| `test_avg_power_w` | test run의 평균 power |
| `baseline_avg_power_w` | baseline run의 평균 power |
| `test_energy_j` | test power trace를 test elapsed interval로 적분/스케일한 energy |
| `baseline_energy_j` | baseline run 자체 elapsed interval의 energy. duration이 다를 수 있으므로 pJ/FLOP 계산에는 직접 빼지 않음 |
| `baseline_scaled_energy_j` | baseline 평균 power × test elapsed time |
| `incremental_power_w` | test 평균 power - baseline 평균 power |
| `incremental_energy_j` | test_energy_j - baseline_scaled_energy_j |
| `pj_per_flop` | incremental energy / FP16 ops × 1e12 |
| `matmul_input_pj_per_bit` | Tensor Core matmul의 A/B FP16 input bits 기준 incremental pJ/bit |
| `matmul_arithmetic_read_pj_per_bit` | A/B input bits + accumulator read bits 기준 incremental pJ/bit |
| `matmul_register_read_write_pj_per_bit` | A/B input bits + accumulator read bits + output bits 기준 incremental pJ/bit |
| `w_per_tflops` | incremental power / achieved TFLOPS |
| `avg_gpu_util_pct`, `max_gpu_util_pct` | test interval 안의 `nvidia-smi utilization.gpu` 평균/최대값 |
| `avg_sm_util_pct`, `max_sm_util_pct` | test interval 안의 `nvidia-smi dmon -s u` `sm` 컬럼 평균/최대값 |
| `suppress_output_store` | compute kernel의 final global output store를 제거했는지 여부 |
| `expected_l2_touch` | timed kernel이 의도적으로 global/L2 traffic을 만들 것으로 예상되는지 여부 |
| `valid_basic` | power sample, work estimate, positive incremental power/energy에 대한 최소 sanity flag. Nsight 검증을 대체하지 않음 |
| `valid_no_l2` | `valid_basic=True`이고 `expected_l2_touch=False`인 pair. 의도된 L2/global traffic이 없다는 metadata gate이며, 실제 L2 traffic 0을 증명하지는 않음 |
| `pure_fp16_candidate` | `valid_no_l2=True`이고 kernel이 FP16 half2 또는 Tensor Core FP16 compute 계열인 후보 |
| `separation_quality` | `pure_fp16_candidate_no_l2`, `valid_but_expected_l2_touch`, `invalid_or_nonpositive_increment` 등 baseline subtraction 품질 분류 |

`thread_sweep_summary.csv`의 핵심 컬럼은 다음이다.

| 컬럼 | 의미 |
|---|---|
| `threads` | threads per block |
| `threads_per_sm` | launched threads per SM. 기본 matrix에서는 `threads * blocks_per_sm`와 같음 |
| `valid_no_l2_count` | `valid_basic=True`이고 `expected_l2_touch=False`인 반복 수 |
| `avg_sm_util_pct_mean` | thread point별 평균 SM utilization |
| `avg_gpu_util_pct_mean` | dmon SM utilization이 없을 때 fallback으로 쓰는 평균 GPU utilization |
| `tflops_mean` | thread point별 평균 Tensor Core throughput |
| `matmul_input_pj_per_bit_mean` | thread point별 logical input bit 기준 pJ/bit |
| `selected_optimal` | 충분한 반복 수의 valid no-L2 후보 중 SM utilization 첫 포화점으로 선택한 추천 point |

`stats_scope=all_runs_no_valid`는 해당 thread point에서 `valid_basic=True`인 반복이 없었다는 뜻이다. 이 경우 mean/std는 plot과 원인 분석을 위한 전체 run 통계일 뿐, 최종 pJ/bit 후보로 쓰면 안 된다. `valid_no_l2` 역시 “코드가 의도적으로 L2/global memory를 touch하지 않는다”는 조건이지, hardware counter 기반 증명은 아니므로 최종 보고 전에는 Nsight Compute로 `MemoryWorkloadAnalysis`를 확인한다.

최종 보고서에는 `p0_cuda_core_half2_vs_nop`과 `p0_cuda_core_half2_vs_regmove`를 모두 제시하는 것이 좋다. 두 baseline 간 차이는 baseline sensitivity로 취급한다. Tensor Core는 `f16acc`와 `f32acc`를 분리 보고한다.

## 10. 단일 kernel 수동 실행 예시

```bash
./build/fp16_energy_bench \
  --device 0 \
  --kernel fp16_half2 \
  --blocks 0 \
  --blocks-per-sm 8 \
  --threads 256 \
  --iters 2000000 \
  --unroll 8 \
  --warmup 2
```

Tensor Core 예시:

```bash
./build/fp16_energy_bench --device 0 --kernel tensor_mma_f16acc --iters 1000000 --unroll 8
./build/fp16_energy_bench --device 0 --kernel tensor_mma_f32acc --iters 1000000 --unroll 8
```

Memory policy sanity 예시:

```bash
./build/fp16_energy_bench --device 0 --kernel memory_cg --iters 2048 --unroll 4 --mem-mib 1024
```

## 11. 주의사항

이 코드는 실험 자동화의 출발점이다. 실제 논문/보고서 수준의 결과로 사용하려면 다음을 반드시 수행해야 한다.

1. 각 GPU에서 `iters`를 조정해 run duration이 power sampling interval보다 충분히 길도록 만든다.
2. clock 고정 후 실제 clock trace가 안정적인지 확인한다.
3. Nsight Compute로 P0 kernel의 memory traffic과 spill을 검증한다.
4. `summary.csv`의 `valid_basic`만으로 결과를 채택하지 않는다.
5. H100/A100/RTX 3090은 power telemetry 범위와 정확도가 다르므로 GPU 간 비교 시 같은 clock 조건, default 조건, max stable 조건을 분리한다.

## Metric definitions

P0 FP16 compute energy:

```text
baseline_scaled_energy  = avg_power_baseline * elapsed_test_s
incremental_energy      = energy_test - baseline_scaled_energy
incremental_pJ_per_FLOP = incremental_energy / fp16_ops * 1e12
```

Tensor Core FP16 matmul logical pJ/bit:

```text
mma_count                    = fp16_ops / 8192
matmul_input_bits            = mma_count * (16*16 + 16*16) * 16
matmul_input_pJ_per_bit      = incremental_energy / matmul_input_bits * 1e12
matmul_arithmetic_read_bits  = matmul_input_bits + mma_count * 16*16 * accumulator_bits
matmul_register_rw_bits      = matmul_arithmetic_read_bits + mma_count * 16*16 * accumulator_bits
```

`matmul_input_pJ_per_bit`는 DRAM bit energy가 아니라 logical `m16n16k16`의 A/B FP16 operand bit 기준 compute energy estimate다. 구현은 `mma.sync.m16n8k16` instruction 두 개로 N 방향 16 columns를 채운다. `accumulator_bits`는 `tensor_mma_f16acc`에서 16, `tensor_mma_f32acc`에서 32다.

P1 memory/cache-policy energy:

```text
baseline_scaled_energy = avg_power_baseline * elapsed_test_s
incremental_pJ_per_bit = (energy_memory_test - baseline_scaled_energy) / (memory_bytes * 8) * 1e12
total_pJ_per_bit       = total_energy_memory_test / (memory_bytes * 8) * 1e12
```

Use `incremental_pJ_per_bit` for residual memory-traffic calibration. `total_pJ_per_bit` includes idle/leakage/static platform power and is usually much larger or more workload-duration dependent.
