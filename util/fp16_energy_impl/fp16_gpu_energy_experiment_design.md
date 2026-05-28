# GPU FP16 순수 연산 에너지 추출 실험 설계서

문서 버전: v0.1  
작성일: 2026-05-27  
대상 GPU: NVIDIA H100 SXM5, NVIDIA A100 SXM4, NVIDIA RTX 3090

---

## 0. Executive Summary

본 설계서는 GPU에서 **FP16 연산 workload의 증가분 에너지**를 최대한 분리하여 추정하기 위한 실험 절차를 정의한다. 실험의 목표는 일반 GEMM benchmark의 전체 board power가 아니라, FP16 연산량을 크게 증가시켰을 때 추가로 관찰되는 에너지를 산출하는 것이다.

실제 GPU에서는 FP16 arithmetic unit만의 물리적 에너지를 소프트웨어 실험으로 완전히 분리할 수 없다. 측정값에는 register file, instruction issue, warp scheduler, operand collector, clock tree, leakage, instruction fetch, shared/L1/L2/DRAM residual traffic이 일부 포함될 수 있다. 따라서 본 문서의 최종 결과는 **absolute physical FP16 unit energy**가 아니라 **baseline subtraction 기반 measured incremental FP16 compute energy estimate**로 정의한다.

가장 중요한 설계 원칙은 다음과 같다.

| 우선순위 | 원칙 | 설명 |
|---|---|---|
| P0 | timed loop 내부 memory traffic 제거 | register-resident 또는 fragment-resident 연산으로 L1/L2/DRAM 접근을 최소화한다. |
| P0 | baseline subtraction | 동일 loop/control 구조에서 FP16 연산만 제거한 baseline을 사용한다. |
| P0 | clock/power/thermal 통제 | DVFS, thermal throttling, power throttling을 제거하거나 별도 분류한다. |
| P0 | counter 기반 검증 | Nsight Compute로 instruction path, spill, L1/L2/DRAM traffic을 확인한다. |
| P1 | memory hierarchy 보정 | residual memory traffic이 관찰될 때만 L1/L2/DRAM 보정 실험을 수행한다. |
| P1 | cache policy 최소 반영 | cache operator는 primary sweep이 아니라 보정/검증용으로 제한한다. |
| P2 | power limit/L2 policy 확장 | primary 결과 확보 후 보조 실험으로 수행한다. |

---

## 1. 실험 목표와 Claim Boundary

### 1.1 실험 목표

본 실험의 목표는 다음 세 GPU에서 FP16 compute workload의 증가분 에너지를 추정하는 것이다.

1. NVIDIA H100 SXM5
2. NVIDIA A100 SXM4
3. NVIDIA RTX 3090

최종 비교 지표는 GPU별, FP16 경로별로 산출한다.

| 지표 | 의미 |
|---|---|
| pJ/FLOP | FP16 FLOP 1회당 증가분 에너지 |
| J/FLOP | 동일 지표의 SI 단위 표현 |
| W/TFLOPS | 단위 throughput당 증가분 전력 |
| achieved TFLOPS | 실험 kernel에서 실제 달성한 FP16 throughput |
| incremental power | test power와 baseline power의 차이 |

### 1.2 에너지 정의

기본 정의는 다음과 같다.

```text
E_FP16 = (E_test - avg(P_baseline) × elapsed_test) / N_FP16_ops
```

각 run에 대해서는 다음 값을 계산한다.

```text
E_test       = ∫ P_test(t) dt
P_baseline   = avg(P_baseline(t))
E_baseline_scaled = P_baseline × elapsed_test
E_increment = E_test - E_baseline_scaled
pJ/FLOP      = E_increment / N_FP16_ops × 10^12
TFLOPS       = N_FP16_ops / elapsed_time / 10^12
W/TFLOPS     = avg_incremental_power / TFLOPS
```

baseline과 test run의 elapsed time은 다를 수 있으므로 baseline run 자체의 적분 energy를
그대로 빼지 않고, baseline 평균 전력을 test elapsed time에 맞춰 스케일한다. power trace가
존재하는 경우 test run의 단순 평균 전력보다 시간 적분 기반 energy를 우선 사용한다.

### 1.3 최종 결론의 표현 방식

결론에서는 다음 표현을 사용한다.

```text
본 결과는 GPU 내부 FP16 unit만의 절대 물리 에너지가 아니라,
register-resident 또는 fragment-resident FP16 compute microbenchmark에서 관찰한
baseline-subtracted incremental FP16 compute energy estimate이다.
```

---

## 2. 실험 대상 GPU와 환경 확인

### 2.1 대상 GPU

| GPU | 구분 | 주요 관점 | 실험상 주의점 |
|---|---|---|---|
| H100 SXM5 | Datacenter | Hopper Tensor Core, 높은 FP16 throughput, SXM telemetry | MIG/MPS, chassis/BMC 전력 계측 범위 확인 |
| A100 SXM4 | Datacenter | Ampere Tensor Core, L2 persisting policy, SXM telemetry | MIG/MPS/ECC/application clock 상태 확인 |
| RTX 3090 | Consumer | GA102 Ampere, consumer boost behavior | display workload, fan/thermal, NVML telemetry 정확도 주의 |

공식 스펙은 참고값으로만 사용한다. 실제 실험에는 반드시 장비에서 수집한 값을 사용한다.

### 2.2 장비별 확인 항목

| 항목 | 확인 방법 | 사용 목적 |
|---|---|---|
| GPU 모델명/UUID | `nvidia-smi -L` | 결과 식별 |
| SM 수 | `deviceQuery`, CUDA Runtime API | 연산량 및 occupancy 계산 |
| compute capability | `deviceQuery` | instruction path 선택 |
| L2 cache size | `cudaGetDeviceProperties` | memory baseline working set 결정 |
| shared memory/L1 구성 | `deviceQuery`, Nsight Compute | kernel resource 설계 |
| register 사용량 | `ptxas -v`, Nsight Compute | spill 여부 확인 |
| clock range | `nvidia-smi -q -d CLOCK` | fixed clock 설정 |
| power limit | `nvidia-smi -q -d POWER` | power throttling 확인 |
| ECC/MIG/MPS 상태 | `nvidia-smi -q` | 실험 격리성 확인 |
| driver/CUDA/compiler version | `nvidia-smi`, `nvcc --version` | 재현성 확보 |

### 2.3 환경 격리

P0 실험은 단일 GPU를 단독으로 사용한다. 다른 process가 GPU를 사용하는 run은 invalid로 처리한다. H100/A100에서는 MIG와 MPS 상태를 기록하고, 가능하면 full GPU 단독 모드에서 측정한다. RTX 3090은 display output이 연결된 경우 background graphics workload가 전력 trace에 섞일 수 있으므로 headless 또는 최소 display workload 조건을 우선한다.

---

## 3. 실험 우선순위

### 3.1 P0: Primary 결과 산출에 필요한 필수 실험

| P0 실험 | 목적 | 성공 기준 |
|---|---|---|
| Register-resident FP16 CUDA core microbenchmark | memory traffic을 제거한 FP16 CUDA core 경로 측정 | spill 없음, L2/DRAM traffic 최소, FP16 instruction 확인 |
| Tensor Core FP16 MMA microbenchmark | HMMA/MMA 기반 FP16 경로 측정 | 의도한 MMA instruction 확인, Tensor utilization 확보 |
| Control-flow/register baseline | loop, issue, register overhead 제거 | test와 baseline 구조가 최대한 동일 |
| Clock/power/thermal 통제 | DVFS와 throttling 제거 | 실제 clock 안정, throttling 없음 |
| Nsight Compute 검증 | instruction mix, cache traffic, spill 확인 | acceptance criteria 통과 |
| Power trace 수집 | energy integration 수행 | sampling resolution 충분, timestamp 정렬 |

### 3.2 P1: 결과 신뢰도와 보정을 위한 실험

| P1 실험 | 목적 | 채택 방식 |
|---|---|---|
| Memory-only baseline | residual L1/L2/DRAM traffic 보정 | primary 결과에 residual traffic이 있을 때만 사용 |
| Cache policy 검증 | `cg/cs`가 traffic에 미치는 영향 확인 | 보정/검증용으로만 사용 |
| Occupancy sweep | fixed overhead와 spill trade-off 파악 | pJ/FLOP plateau 조건 선택 |
| Unroll factor sweep | loop overhead 최소화와 register pressure 균형 | spill 없는 안정 조건 선택 |
| Baseline sensitivity | baseline 선택에 따른 pJ/FLOP 변화 확인 | 최종 오차 범위에 반영 |

### 3.3 P2: 확장 분석

| P2 실험 | 목적 | 보고 방식 |
|---|---|---|
| Power limit sweep | power cap과 energy efficiency 관계 확인 | primary 결과와 분리 보고 |
| L2 persisting/streaming policy | Tensor operand reuse 또는 memory baseline 영향 확인 | memory hierarchy 분석으로 분리 |
| External power meter 비교 | NVML/DCGM telemetry 편차 확인 | 가능 시 보조 검증 |
| Extended cache hint sweep | 필요 시 세부 cache behavior 분석 | primary matrix에는 포함하지 않음 |

---

## 4. FP16 Instruction Path 분리

FP16 결과는 다음 경로별로 분리한다.

| 경로 | 설명 | 주요 검증 |
|---|---|---|
| CUDA core FP16 FMA | scalar `half` 또는 vectorized `half2` FMA | FP16/F16x2 instruction count, FLOP 수 계산 |
| Tensor Core FP16 MMA | `wmma`, `mma.sync`, CUTLASS, inline PTX 기반 | HMMA/MMA instruction count, Tensor Core utilization |
| FP16 input + FP16 accumulate | accumulator가 FP16인 경로 | instruction opcode와 accumulator type 확인 |
| FP16 input + FP32 accumulate | mixed precision accumulate 경로 | accumulator register type과 MMA variant 확인 |

동일한 “FP16”이라도 H100, A100, RTX 3090에서 실제 instruction path와 Tensor Core 세대가 다르다. 결과 표에는 반드시 `FP16 path`와 `accumulate type`을 명시한다.

---

## 5. P0 Register-resident FP16 Benchmark 설계

### 5.1 설계 목적

이 kernel은 global memory, L2, DRAM 영향을 최대한 제거하고 CUDA core 기반 FP16 연산 증가분을 측정한다.

### 5.2 Kernel 구조

1. 각 thread는 global memory에서 seed 값을 1회 load한다.
2. seed를 register 변수 여러 개로 확장한다.
3. timed loop 안에서는 `half2` FMA 또는 inline PTX 기반 FP16 FMA만 반복한다.
4. loop 안에서는 global/shared/local memory 접근이 발생하지 않아야 한다.
5. compiler dead-code elimination을 막기 위해 최종 accumulator를 global memory에 1회 store한다.

개념적 pseudo-code는 다음과 같다.

```cpp
__global__ void fp16_reg_kernel(const half2* in, half2* out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    half2 a = in[tid];          // 초기 load 1회
    half2 b = __float2half2_rn(1.0001f);
    half2 c = __float2half2_rn(0.9999f);

    half2 x0 = a;
    half2 x1 = __hadd2(a, b);
    half2 x2 = __hadd2(a, c);
    half2 x3 = __hmul2(a, b);

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        // unroll factor는 빌드 파라미터로 조절
        x0 = __hfma2(x0, b, c);
        x1 = __hfma2(x1, c, x0);
        x2 = __hfma2(x2, b, x1);
        x3 = __hfma2(x3, c, x2);
    }

    out[tid] = __hadd2(__hadd2(x0, x1), __hadd2(x2, x3)); // 최종 store 1회
}
```

### 5.3 연산량 계산

`half2` FMA 1개는 FP16 lane 2개에 대해 multiply-add를 수행한다. FMA를 FLOP 기준으로 계산하면 lane당 2 FLOP이므로, `half2` FMA 1개는 4 FP16 FLOP으로 계산한다.

```text
N_FP16_ops = threads × iters × unroll_fma_count_per_iter × 4
```

inline PTX 또는 SASS에서 실제 instruction 수가 달라지면, `N_FP16_ops`는 최종 instruction count 기준으로 재계산한다.

### 5.4 Compiler 최적화 방지

| 위험 | 대응 |
|---|---|
| loop 제거 | 최종 store, runtime `iters`, dependency chain 사용 |
| constant folding | 입력 seed를 global memory에서 읽고 runtime 값으로 사용 |
| instruction path 변경 | `cuobjdump`, `nvdisasm`, Nsight Compute로 SASS 확인 |
| register pressure 증가 | unroll factor와 accumulator 수를 제한 |
| local spill 발생 | `ptxas -v`, Nsight Compute local memory counter 확인 |

---

## 6. P0 Tensor Core FP16 Benchmark 설계

### 6.1 설계 목적

Tensor Core 기반 FP16 MMA 경로의 증가분 에너지를 측정한다. 일반 GEMM benchmark가 아니라, operand를 가능한 한 register fragment 또는 shared memory에 유지한 뒤 MMA instruction을 반복하는 microbenchmark로 설계한다.

### 6.2 Kernel 구조

1. 각 warp는 MMA tile fragment를 초기화한다.
2. timed loop 전에 operand를 준비한다.
3. timed loop 안에서는 `mma.sync` 또는 WMMA 연산을 반복한다.
4. loop 내부 global memory load/store는 금지한다.
5. accumulator fragment는 최종적으로 최소 횟수만 global memory에 store한다.

### 6.3 MMA FLOP 계산

MMA instruction의 shape를 `m × n × k`라고 하면, 한 번의 matrix multiply-accumulate는 다음 FLOP 수로 계산한다.

```text
FLOP_per_MMA = 2 × m × n × k
N_FP16_ops   = warps × iters × mma_per_iter × FLOP_per_MMA
```

실제 `m`, `n`, `k`, accumulator type은 사용한 `mma.sync` variant, WMMA API, CUTLASS kernel 설정에 따라 문서화한다.

### 6.4 Shared memory 사용 원칙

Tensor Core 실험에서 shared memory staging이 필요할 수 있다. 이 경우 shared memory traffic은 FP16 compute energy에 포함될 수 있으므로 결과 표에 별도 column으로 기록한다. P0 primary 결과는 가능한 한 loop 내부 global memory traffic이 없는 조건을 우선한다.

---

## 7. Baseline Subtraction 설계

### 7.1 Baseline 목록

| Baseline | 우선순위 | 목적 | 최종 계산 사용 여부 |
|---|---|---|---|
| Idle baseline | P0 | idle board/chip power 확인 | 참고값 |
| Empty kernel baseline | P0 | launch/persistent loop overhead 확인 | 보조 |
| Control-flow baseline | P0 | 동일 loop에서 FP16 arithmetic만 제거 | primary 후보 |
| Register-move baseline | P0 | register dependency, issue, move overhead 보정 | primary 후보 |
| Store-only baseline | P1 | 최종 output store 영향 보정 | 필요 시 |
| Memory-only baseline | P1 | residual L1/L2/DRAM traffic 보정 | 필요 시 |

### 7.2 Primary baseline 선택 기준

최종 pJ/FLOP 계산에는 test kernel과 가장 구조가 유사한 baseline을 사용한다. 즉, 같은 grid/block, 같은 loop count, 유사한 register pressure, 유사한 control flow를 유지하되 FP16 arithmetic instruction만 제거한 baseline을 우선한다.

### 7.3 Baseline sensitivity

baseline 선택이 결과에 미치는 영향을 반드시 보고한다.

```text
pJ/FLOP_control_baseline
pJ/FLOP_register_move_baseline
pJ/FLOP_store_corrected
```

baseline별 차이가 크면 “순수 FP16” 결과의 불확실성이 크다는 의미이므로 최종 결론에 오차 요인으로 반영한다.

---

## 8. L1/L2/DRAM 및 Cache Policy 반영 범위

### 8.1 기본 원칙

Primary FP16 energy 산출에서는 cache operator sweep을 주 실험으로 넣지 않는다. 이유는 timed loop 내부에서 memory access 자체를 제거하는 것이 목표이기 때문이다.

따라서 P0 결과는 다음 구조로 얻는다.

```text
memory traffic을 줄이는 cache operator 선택
→ primary 결과 산출
```

이 아니라,

```text
register-resident 또는 fragment-resident loop 구성
→ timed loop 내부 L1/L2/DRAM traffic 제거
→ counter로 검증
→ baseline subtraction
```

으로 산출한다.

### 8.2 Cache operator 최소 반영 조건

cache operator는 다음 상황에서만 P1 보조 실험으로 포함한다.

1. 초기 operand load 또는 최종 result store가 전체 energy에 영향을 준다고 판단되는 경우
2. Nsight Compute에서 residual L2/DRAM traffic이 관찰되는 경우
3. memory-only baseline을 구성해야 하는 경우
4. Tensor Core operand staging에서 shared/global memory 정책 차이를 확인해야 하는 경우

### 8.3 권장 최소 cache policy 세트

| 조건 | 사용 위치 | 목적 | primary 결과 반영 |
|---|---|---|---|
| default load/store | P0 primary | 기본 조건 | 예 |
| `ld.global.cg` | P1 residual check | L1 우회, L2 중심 traffic 확인 | 필요 시 보정용 |
| `ld.global.cs` | P1 memory baseline | streaming memory baseline 구성 | 보정용 |
| `st.global.cg` | P1 store baseline | 최종 store의 L1 영향 축소 확인 | 보정용 |
| L2 normal | P0 primary | 기본 L2 정책 | 예 |
| L2 streaming | P1 memory baseline | streaming access 영향 확인 | 보정용 |
| L2 persisting | P2 optional | operand reuse/L2 policy 분석 | primary와 분리 |

### 8.4 Primary matrix에서 제외할 조건

| 제외 항목 | 제외 이유 |
|---|---|
| `.ca` vs default load sweep | default load와 중복될 가능성이 높고 primary 목적과 거리가 있음 |
| `.cv` sweep | device DRAM access 강제 수단으로 단정하기 어려움 |
| eviction hint 전체 조합 | 실험 폭이 과도하게 커짐 |
| 모든 store policy 조합 | 최종 store를 최소화하고 baseline으로 보정하는 것이 우선 |
| L2 persisting hitRatio sweep | compute energy보다 memory behavior 분석에 가까움 |

### 8.5 Cache policy 검증 기준

cache operator와 L2 access policy는 성능 hint 성격을 가지므로 실제 동작을 반드시 counter로 확인한다. 사용 시 다음 항목을 확인한다.

| 검증 항목 | 목적 |
|---|---|
| PTX/SASS load/store modifier | 의도한 operator로 lowering되었는지 확인 |
| L1/TEX sector traffic | L1 영향 확인 |
| L2 sector read/write | residual L2 traffic 확인 |
| DRAM sector read/write | DRAM 접근 여부 확인 |
| local memory traffic | spill로 인한 L2/DRAM 오염 확인 |
| shared memory traffic | Tensor staging 영향 확인 |

---

## 9. Clock, Power, Thermal 통제

### 9.1 Clock 통제

가능한 경우 SM clock과 memory clock을 고정한다. 설정한 clock이 유지되었는지는 power logging과 별도로 clock trace를 저장하여 확인한다.

예시 command는 환경에 따라 조정한다.

```bash
# persistence mode
sudo nvidia-smi -pm 1

# GPU 상태 확인
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE

# 지원되는 clock 확인
nvidia-smi -q -d SUPPORTED_CLOCKS

# application clock 또는 lock clock 설정은 GPU/driver별 지원 여부 확인 후 적용
# 예: datacenter GPU에서는 -ac, 일부 환경에서는 -lgc/-lmc 사용 가능
```

### 9.2 Thermal 통제

실험 전 warm-up kernel을 실행하여 온도가 steady-state에 접근하도록 한다. 측정 중 thermal throttling이 발생한 run은 invalid로 분류한다. RTX 3090은 fan curve와 chassis airflow에 따른 boost behavior 변동이 크므로 fan speed와 ambient condition을 기록한다.

### 9.3 Power limit 통제

P0에서는 power limit에 걸리지 않는 조건을 우선한다. power throttling이 발생하면 FP16 energy estimate가 workload 특성이 아니라 power cap 정책을 반영할 수 있으므로 primary 결과에서 제외한다. power limit sweep은 P2 확장 실험으로 분리한다.

---

## 10. Power Measurement 방법

### 10.1 계측 방법 비교

| 방법 | 적용 GPU | 장점 | 한계 | 사용 우선순위 |
|---|---|---|---|---|
| NVML / `nvidia-smi` | 전체 | 구현이 쉬움 | sampling interval, board/chip 구분 한계 | P0 |
| DCGM | H100/A100 중심 | datacenter 운영 환경에 적합 | 환경 설정 필요 | P0/P1 |
| BMC/IPMI | 서버 GPU | chassis-level 관측 가능 | GPU 단독 분리 어려움 | P1 |
| External power meter | 특히 RTX 3090 | telemetry 검증 가능 | 설치 난이도 높음 | P1/P2 |
| Wall power | 전체 시스템 | 쉬운 보조 관측 | CPU/PSU/system noise 포함 | 참고용 |

### 10.2 Power trace 수집 원칙

1. kernel 실행 구간과 power sampling timestamp를 동기화한다.
2. long-running kernel을 사용해 launch overhead가 측정 구간에서 차지하는 비중을 줄인다.
3. warm-up 구간과 measurement 구간을 분리한다.
4. power, clock, temperature, utilization을 함께 기록한다.
5. 전력 sampling interval이 너무 크면 해당 run은 energy integration 신뢰도가 낮은 것으로 표시한다.

### 10.3 Logging 예시

```bash
# 간단한 nvidia-smi logging 예시
nvidia-smi \
  --query-gpu=timestamp,index,name,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate,clocks_throttle_reasons.active \
  --format=csv \
  -lms 100 \
  -f power_trace.csv
```

DCGM 사용 가능 환경에서는 DCGM exporter 또는 `dcgmi dmon`을 병행하여 비교한다.

---

## 11. Nsight Compute Counter 검증

### 11.1 필수 검증 범위

| 범위 | 확인 내용 |
|---|---|
| Instruction mix | FP16 FMA 또는 MMA instruction이 예상대로 발생했는지 |
| Tensor Core utilization | Tensor Core path에서 실제 Tensor pipeline 사용 여부 |
| L1/L2/DRAM traffic | timed loop 내부 memory traffic이 없는지 |
| Local memory | register spill 발생 여부 |
| Occupancy | active warp/block, achieved occupancy |
| Scheduler/warp stall | dependency 또는 issue bottleneck 확인 |
| Shared memory | Tensor staging 사용 시 traffic 규모 확인 |

### 11.2 Nsight Compute 실행 예시

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:fp16_.* \
  --export ncu_report \
  ./fp16_energy_bench --kernel fp16_reg --iters <N>
```

측정 overhead가 큰 경우에는 section을 줄인다.

```bash
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --kernel-name regex:fp16_.* \
  ./fp16_energy_bench --kernel tensor_mma --iters <N>
```

metric 이름은 Nsight Compute 버전에 따라 달라질 수 있으므로, 설계서에는 metric class를 고정하고 실제 수집 metric 이름은 raw report와 함께 저장한다.

---

## 12. Experiment Matrix

### 12.1 P0 Primary matrix

| 축 | 조건 | 비고 |
|---|---|---|
| GPU | H100 SXM5 / A100 SXM4 / RTX 3090 | 단일 GPU 단독 실행 |
| FP16 path | CUDA core FMA / Tensor Core MMA | 결과 분리 |
| Accumulate | FP16 / FP32 | 가능한 경로만 수행 |
| Clock | fixed clock / default clock | fixed clock 우선 |
| Baseline | control-flow / register-move | primary 계산 후보 |
| Kernel duration | long-running | power sampling resolution 대비 충분히 길게 |
| Cache policy | default | primary는 cache sweep 제외 |
| Repetition | 10회 이상 | 통계 처리 |

### 12.2 P1 보정 matrix

| 축 | 조건 | 수행 조건 |
|---|---|---|
| Occupancy | low / mid / high | P0 결과의 plateau 확인 필요 시 |
| Unroll factor | small / mid / large | loop overhead 또는 spill 의심 시 |
| Memory baseline | L1-size / L2-size / L2 초과 working set | residual memory traffic 보정 필요 시 |
| Cache policy | default / `cg` / `cs` selected | residual traffic 또는 memory baseline에서만 |
| Store baseline | default / `st.global.cg` | 최종 store 영향이 큰 경우 |

### 12.3 P2 확장 matrix

| 축 | 조건 | 보고 방식 |
|---|---|---|
| Power limit | default / reduced / max stable | primary와 분리 |
| L2 policy | normal / streaming / persisting | memory hierarchy 분석으로 분리 |
| External meter | NVML 대비 비교 | 가능 시 오차 분석 |
| Extended cache hint | 필요한 hint만 선별 | primary conclusion에 직접 사용하지 않음 |

---

## 13. Acceptance Criteria

run은 다음 기준을 만족할 때 valid로 분류한다.

| 항목 | 기준 | 실패 시 처리 |
|---|---|---|
| Clock stability | 설정 clock에서 큰 변동 없음 | invalid 또는 별도 분류 |
| Thermal throttling | 없음 | invalid |
| Power throttling | 없음 | P2 power-cap 결과로 분리 |
| Other GPU process | 없음 | invalid |
| Local memory spill | 없음 | invalid |
| DRAM traffic | 0 또는 초기/최종 접근 수준 | memory correction 또는 invalid |
| L2 traffic | 낮고 예측 가능 | residual 보정 또는 invalid |
| Instruction path | 의도한 FP16/MMA instruction 확인 | invalid |
| SM activity | 충분히 높음 | occupancy 조정 후 재실험 |
| Power sampling | measurement window 대비 충분 | invalid 또는 low-confidence |

---

## 14. 데이터 분석 절차

### 14.1 Run 단위 계산

각 run에서 다음 값을 산출한다.

| 값 | 설명 |
|---|---|
| elapsed_time | kernel measurement 구간 시간 |
| avg_power_test | test kernel 평균 전력 |
| avg_power_baseline | baseline 평균 전력 |
| energy_test | test power trace 적분값 |
| energy_baseline | baseline run 자체의 power trace 적분값 |
| energy_baseline_scaled | baseline 평균 power × test elapsed time |
| energy_increment | energy_test - energy_baseline_scaled |
| N_FP16_ops | instruction count 또는 설계식 기반 총 FLOP |
| achieved TFLOPS | 실제 throughput |
| pJ/FLOP | 최종 energy estimate |
| W/TFLOPS | energy efficiency 보조 지표 |

### 14.2 통계 처리

각 조건은 최소 10회 반복한다. 결과는 평균, 표준편차, min/max, 95% confidence interval을 함께 보고한다. invalid run은 primary 통계에서 제외하되, invalid reason summary에 기록한다.

### 14.3 Residual memory correction

P0 primary run에서 residual L2/DRAM traffic이 관찰되면 다음 순서로 처리한다.

1. 먼저 kernel을 수정하여 memory traffic을 제거한다.
2. 제거가 불가능한 경우 memory-only baseline으로 L2/DRAM traffic당 energy를 추정한다.
3. 보정값이 전체 incremental energy에서 차지하는 비중을 보고한다.
4. 보정 비중이 크면 해당 run은 primary conclusion에서 제외한다.

---

## 15. Visualization 계획

### 15.1 필수 시각화

| Visualization | X축 | Y축 | Grouping | 목적 |
|---|---|---|---|---|
| Power trace plot | time | power | GPU/kernel/baseline | steady-state와 spike 확인 |
| Clock/temperature timeline | time | clock/temp | GPU | clock 안정성, thermal 상태 확인 |
| pJ/FLOP bar chart | GPU | pJ/FLOP | FP16 path, accumulate | GPU별 energy estimate 비교 |
| TFLOPS vs pJ/FLOP scatter | TFLOPS | pJ/FLOP | GPU/path | 성능-에너지 trade-off 확인 |

### 15.2 보조 시각화

| Visualization | 목적 | 수행 조건 |
|---|---|---|
| Valid/invalid run heatmap | 실패 조건 분포 확인 | 모든 실험 후 |
| L2/DRAM traffic vs pJ/FLOP | residual memory 영향 확인 | residual traffic 관찰 시 |
| Occupancy/unroll sweep plot | pJ/FLOP plateau 확인 | P1 sweep 수행 시 |
| Baseline sensitivity plot | baseline 선택 영향 확인 | P0 결과 분석 시 |
| Power limit sweep plot | power cap 영향 확인 | P2 수행 시 |

### 15.3 권장 output 파일

| 파일 | 내용 |
|---|---|
| `fig_power_trace_<gpu>_<kernel>.png` | power trace |
| `fig_clock_temp_<gpu>.png` | clock/temperature timeline |
| `fig_pj_per_flop_summary.png` | GPU/path별 pJ/FLOP bar chart |
| `fig_tflops_vs_energy.png` | TFLOPS vs pJ/FLOP scatter |
| `fig_validity_heatmap.png` | invalid reason heatmap |
| `fig_baseline_sensitivity.png` | baseline별 결과 민감도 |

---

## 16. 실험 결과 정리 Template

### 16.1 Primary result summary

| GPU | FP16 path | Accumulate | Clock mode | Baseline | TFLOPS | Incremental power | pJ/FLOP | L2 traffic | DRAM traffic | Valid |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| H100 SXM5 | CUDA core FMA | FP16 | fixed | register-move | TBD | TBD | TBD | TBD | TBD | TBD |
| H100 SXM5 | Tensor Core MMA | FP16/FP32 | fixed | register-move | TBD | TBD | TBD | TBD | TBD | TBD |
| A100 SXM4 | CUDA core FMA | FP16 | fixed | register-move | TBD | TBD | TBD | TBD | TBD | TBD |
| A100 SXM4 | Tensor Core MMA | FP16/FP32 | fixed | register-move | TBD | TBD | TBD | TBD | TBD | TBD |
| RTX 3090 | CUDA core FMA | FP16 | fixed/default | register-move | TBD | TBD | TBD | TBD | TBD | TBD |
| RTX 3090 | Tensor Core MMA | FP16/FP32 | fixed/default | register-move | TBD | TBD | TBD | TBD | TBD | TBD |

### 16.2 Baseline sensitivity summary

| GPU | Kernel | Baseline | Incremental power | pJ/FLOP | Delta vs selected baseline | Decision |
|---|---|---|---:|---:|---:|---|
| TBD | fp16_reg | control-flow | TBD | TBD | TBD | 후보/제외 |
| TBD | fp16_reg | register-move | TBD | TBD | TBD | selected |
| TBD | tensor_mma | control-flow | TBD | TBD | TBD | 후보/제외 |
| TBD | tensor_mma | register-move | TBD | TBD | TBD | selected |

### 16.3 Validity summary

| GPU | Run count | Valid runs | Invalid runs | Main invalid reason | Notes |
|---|---:|---:|---:|---|---|
| H100 SXM5 | TBD | TBD | TBD | TBD | TBD |
| A100 SXM4 | TBD | TBD | TBD | TBD | TBD |
| RTX 3090 | TBD | TBD | TBD | TBD | TBD |

### 16.4 Cache/memory policy summary

| GPU | Kernel | Cache policy | L1 traffic | L2 traffic | DRAM traffic | pJ/FLOP impact | Decision |
|---|---|---|---:|---:|---:|---:|---|
| TBD | fp16_reg | default | TBD | TBD | TBD | TBD | primary 사용 |
| TBD | memory_baseline | `ld.global.cg` | TBD | TBD | TBD | TBD | 보정용 |
| TBD | memory_baseline | `ld.global.cs` | TBD | TBD | TBD | TBD | 보정용 |

---

## 17. 실행 순서

### 17.1 Phase 0: 환경 준비

1. GPU exclusive 사용 가능 여부 확인
2. driver/CUDA/Nsight Compute 버전 기록
3. GPU 상태 확인: clock, power, ECC, MIG, MPS
4. clock 고정 가능 여부 확인
5. power logging pipeline 검증

### 17.2 Phase 1: Kernel bring-up

1. register-resident FP16 kernel 구현
2. Tensor Core MMA kernel 구현
3. baseline kernel 구현
4. `ptxas -v`로 register count와 spill 확인
5. `cuobjdump` 또는 `nvdisasm`로 instruction path 확인

### 17.3 Phase 2: P0 primary measurement

1. warm-up 실행
2. idle/control/register baseline 측정
3. FP16 test kernel 측정
4. power/clock/temp trace 저장
5. Nsight Compute report 저장
6. acceptance criteria 적용

### 17.4 Phase 3: P1 보정 및 민감도 분석

1. occupancy/unroll sweep 수행
2. baseline sensitivity 분석
3. residual memory traffic이 있으면 memory-only baseline 수행
4. 필요한 경우에만 `cg/cs` cache policy 검증

### 17.5 Phase 4: Visualization 및 결론 작성

1. raw data 정리
2. pJ/FLOP, TFLOPS, W/TFLOPS 산출
3. 필수 시각화 생성
4. valid/invalid summary 작성
5. GPU별, FP16 path별 결론 작성

---

## 18. Raw Data Schema

### 18.1 Run metadata CSV

| column | 설명 |
|---|---|
| run_id | 고유 run ID |
| timestamp_start | 시작 시간 |
| gpu_name | GPU 모델 |
| gpu_uuid | GPU UUID |
| kernel_name | kernel 이름 |
| fp16_path | cuda_core_fma / tensor_core_mma |
| accumulate_type | fp16 / fp32 / mixed |
| clock_mode | fixed / default |
| sm_clock_target | 설정 clock |
| mem_clock_target | 설정 memory clock |
| power_limit | 설정 power limit |
| block_dim | block 크기 |
| grid_dim | grid 크기 |
| iters | loop iteration |
| unroll_factor | unroll factor |
| baseline_type | control/register/store/memory |
| cache_policy | default/cg/cs/etc |
| valid | true/false |
| invalid_reason | invalid 사유 |

### 18.2 Result CSV

| column | 설명 |
|---|---|
| run_id | metadata join key |
| elapsed_s | 측정 시간 |
| avg_power_w | 평균 power |
| avg_baseline_power_w | baseline 평균 power |
| energy_j | 적분 energy |
| baseline_energy_j | baseline run 자체 energy |
| baseline_scaled_energy_j | baseline 평균 power × test elapsed time |
| incremental_energy_j | 증가분 energy |
| fp16_ops | 총 FP16 FLOP |
| tflops | achieved TFLOPS |
| pj_per_flop | pJ/FLOP |
| w_per_tflops | W/TFLOPS |
| l1_traffic | L1 traffic counter |
| l2_traffic | L2 traffic counter |
| dram_traffic | DRAM traffic counter |
| spill_detected | spill 여부 |

---

## 19. GPU 간 비교 방법

GPU 간 비교는 다음 세 관점으로 분리한다.

| 비교 방식 | 목적 | 주의점 |
|---|---|---|
| Fixed clock 비교 | architecture/path 차이 관찰 | 동일 clock 설정 가능 여부 확인 |
| Default clock 비교 | 실제 운영 조건 비교 | DVFS/boost 영향 포함 |
| Max stable clock 비교 | 각 GPU 최적 조건 비교 | thermal/power cap 영향 분리 필요 |

H100, A100, RTX 3090은 telemetry 방식, Tensor Core 세대, power management, cooling, board design이 다르므로 단순 board power만으로 “FP16 unit energy”를 비교하지 않는다. 최종 결론은 동일 조건 비교와 운영 조건 비교를 분리하여 작성한다.

---

## 20. 한계와 오차 요인

| 오차 요인 | 영향 | 대응 |
|---|---|---|
| register file energy | FP16 연산과 함께 포함 | baseline subtraction, register-move baseline |
| scheduler/issue energy | loop 실행 overhead 포함 | control-flow baseline |
| instruction fetch/cache | 완전 제거 불가 | long-running loop, baseline |
| clock tree/leakage | workload와 무관한 전력 포함 | idle/control baseline, thermal 통제 |
| residual L2/DRAM traffic | pJ/FLOP 과대평가 | counter 검증, memory-only 보정 |
| telemetry sampling latency | energy integration 오차 | long-running kernel, timestamp 정렬 |
| RTX 3090 boost behavior | run 간 power/clock 변동 | clock trace 기록, fan/thermal 통제 |
| compiler optimization | instruction 제거 또는 변경 | SASS 검증, final store, dependency chain |

---

## 21. 재현성 체크리스트

| 항목 | 기록값 |
|---|---|
| GPU 모델명 | TBD |
| GPU UUID | TBD |
| Driver version | TBD |
| CUDA version | TBD |
| Nsight Compute version | TBD |
| OS/kernel version | TBD |
| Compiler version | TBD |
| Compile command | TBD |
| Run command | TBD |
| Nsight Compute command | TBD |
| Power logging command | TBD |
| Clock setting | TBD |
| Memory clock setting | TBD |
| Power limit | TBD |
| Temperature/fan 상태 | TBD |
| ECC 상태 | TBD |
| MIG 상태 | TBD |
| MPS 상태 | TBD |
| Benchmark source hash | TBD |
| Raw power trace path | TBD |
| Raw Nsight Compute report path | TBD |
| Analysis script hash | TBD |
| Figure output path | TBD |

---

## 22. 최종 보고서 작성 구조

최종 실험 보고서는 다음 순서로 작성한다.

1. 실험 목적과 claim boundary
2. 실험 대상 GPU와 환경
3. FP16 path별 kernel 설계
4. baseline 설계와 선택 근거
5. clock/power/thermal 통제 결과
6. Nsight Compute validation 결과
7. primary pJ/FLOP 결과
8. baseline sensitivity 결과
9. cache/memory residual 영향 분석
10. GPU별 비교 결과
11. visualization 요약
12. 한계와 오차 요인
13. 최종 결론

최종 결론 예시는 다음 형식을 따른다.

```text
H100 SXM5/A100 SXM4/RTX 3090에서 register-resident 및 Tensor Core FP16 microbenchmark를 수행한 결과,
valid run 기준 FP16 compute workload의 baseline-subtracted incremental energy는 GPU 및 instruction path별로 [TBD] pJ/FLOP 범위로 측정되었다.
해당 결과는 board-level 측정, baseline subtraction, Nsight Compute counter 검증에 기반한 estimate이며,
FP16 arithmetic unit만의 절대 물리 에너지로 해석하지 않는다.
```

---

## 23. 참고 기준

본 설계서는 다음 공식 문서를 기준으로 정책과 검증 범위를 정한다.

1. NVIDIA PTX ISA Documentation: cache operator는 load/store의 performance hint이며 memory consistency behavior를 바꾸지 않는다. 또한 `.ca`, `.cg`, `.cs`, `.cv`의 의미를 정의한다.  
   <https://docs.nvidia.com/cuda/parallel-thread-execution/>
2. NVIDIA CUDA C++ Programming Guide: L2 persisting access, access policy window, set-aside, reset 절차를 정의한다.  
   <https://docs.nvidia.com/cuda/cuda-programming-guide/>
3. NVIDIA Nsight Compute CLI Documentation: CLI 기반 kernel profiling과 report 저장 방법을 정의한다.  
   <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
4. NVIDIA H100 Product Specifications: H100 SXM의 공개 FP16 Tensor Core throughput, memory, bandwidth 참고값.  
   <https://www.nvidia.com/en-us/data-center/h100/>
5. NVIDIA A100 Datasheet: A100 SXM/PCIe의 공개 Tensor Core throughput, memory bandwidth, TDP, MIG 참고값.  
   <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>
6. NVIDIA GeForce RTX 3090 Product Page 및 GA102 Architecture Whitepaper: RTX 3090/GA102 Ampere, 3rd Gen Tensor Core, GDDR6X memory 관련 참고값.  
   <https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/>  
   <https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf>

---

## Appendix A. 최소 구현 산출물

실험 repo에는 최소한 다음 파일을 포함한다.

| 파일 | 목적 |
|---|---|
| `src/fp16_reg_kernel.cu` | CUDA core FP16 register-resident benchmark |
| `src/tensor_mma_kernel.cu` | Tensor Core FP16 MMA benchmark |
| `src/baseline_kernels.cu` | control/register/store/memory baseline |
| `scripts/run_p0.sh` | P0 primary run 실행 |
| `scripts/profile_ncu.sh` | Nsight Compute 수집 |
| `scripts/log_power.sh` | power/clock/temp trace 수집 |
| `analysis/compute_energy.py` | energy integration 및 pJ/FLOP 계산 |
| `analysis/plot_results.py` | visualization 생성 |
| `results/raw/` | raw CSV, NCU report 저장 |
| `results/figures/` | visualization output 저장 |
| `results/summary/` | 최종 summary table 저장 |

## Appendix B. P0 성공 판정 요약

P0 결과를 채택하려면 다음을 모두 만족해야 한다.

1. 의도한 FP16 CUDA core 또는 Tensor Core instruction이 확인된다.
2. local memory spill이 없다.
3. timed loop 내부 L2/DRAM traffic이 없거나 초기/최종 접근 수준으로 제한된다.
4. clock이 안정적으로 유지된다.
5. thermal/power throttling이 없다.
6. baseline과 test kernel의 구조 차이가 FP16 arithmetic 외에는 최소화되어 있다.
7. 반복 run에서 pJ/FLOP가 안정적인 plateau를 보인다.
