# AccelWattch: GPU Power Modeling Framework 백서

> **목적**: AccelWattch 논문(MICRO 2021)과 소스코드를 기반으로, GPU Power Modeling 방법론을 처음 접하는 사람도 이해할 수 있도록 상세히 정리한 백서.  
> **대상 독자**: GPU 아키텍처 연구자, Power Modeling 입문자  
> **작성일**: 2026-04-02  

---

## 목차

1. [개요 및 동기](#1-개요-및-동기)
2. [AccelWattch 전체 아키텍처](#2-accelwattch-전체-아키텍처)
3. [Power Modeling 3요소: Constant, Static, Dynamic](#3-power-modeling-3요소)
4. [DVFS-Aware Constant Power Modeling](#4-dvfs-aware-constant-power-modeling)
5. [Power-Gating-Aware Static Power Modeling](#5-power-gating-aware-static-power-modeling)
6. [Dynamic Power Modeling](#6-dynamic-power-modeling)
7. [Quadratic Programming 최적화](#7-quadratic-programming-최적화)
8. [GPU Configuration 구조 분석](#8-gpu-configuration-구조-분석)
9. [Microbenchmark 설계 및 역할](#9-microbenchmark-설계-및-역할)
10. [SASS Opcode → Power Component 매핑](#10-sass-opcode--power-component-매핑)
11. [소스코드 호출구조](#11-소스코드-호출구조)
12. [Validation 결과 요약](#12-validation-결과-요약)
13. [핵심 Equation 정리](#13-핵심-equation-정리)
14. [용어 사전](#14-용어-사전)

---

## 1. 개요 및 동기

### 1.1 왜 GPU Power Modeling이 필요한가?

GPU는 데이터 분석, 머신러닝, HPC에서 핵심 가속기로 자리잡았다. TOP500 HPC 리스트의 147개 시스템이 GPU를 사용하며, 상위 50개 중 70%가 GPU 가속 시스템이다. GPU의 성능이 높아질수록 전력 소모도 증가하며, **Performance per Watt**는 아키텍처 평가의 핵심 지표가 되었다.

### 1.2 기존 도구의 한계

| 도구 | 한계 |
|------|------|
| **GPUWattch** (2013) | Fermi GTX 480 기준, DVFS 미지원, PTX만 지원, 최신 GPU에서 MAPE 219-225% |
| **GPUSimPow** (2013) | Eq.(2) 기반 constant power 추정 → 현대 GPU에 부적합 |
| **IPP** (2010) | 소스 수준 PTX 분석 필요, closed-source 워크로드 불가 |
| **Guerreiro et al.** (2018) | 고정 power component, power gating 미반영, 8개 component만 모델링 |

### 1.3 AccelWattch의 핵심 기여

1. **DVFS-Aware Constant Power**: 3차 다항식 (missing quadratic term)으로 constant power 정확 추정
2. **Power Gating Modeling**: chip-wide, SM-wide, lane-specific power gating을 최초로 분석적으로 모델링
3. **Thread Divergence 반영**: Half-warp static power model로 실행 divergence에 따른 전력 변화 포착
4. **Cycle-level 정확도**: SASS ISA 기반 cycle-level power trace 제공
5. **Closed-source 지원**: hand-tuned SASS 워크로드 (cuDNN, cuBLAS 등)의 전력 추정 가능

---

## 2. AccelWattch 전체 아키텍처

### 2.1 Modeling Workflow (Figure 1 기반)

AccelWattch의 power modeling은 8단계로 구성된다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AccelWattch Power Modeling Flow                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ① Constant Power Modeling                                          │
│     └─ DVFS 실험 → 3차 다항식 fitting → P_const 추정               │
│                                                                     │
│  ② Static Power Modeling (μBenchmarks)                              │
│     └─ Divergence-aware static power model 구축                     │
│                                                                     │
│  ③ Idle SM Static Power Modeling                                    │
│     └─ Active SM 수 변화 실험 → Idle SM power 추정                  │
│                                                                     │
│  ④ Final Static Power Model                                         │
│     └─ ②+③ 결합 → Eq.(10)의 static 항 완성                        │
│                                                                     │
│  ⑤ μBenchmarks for Dynamic Power                                    │
│     └─ 102개 microbenchmarks + HW profiling                         │
│                                                                     │
│  ⑥ Performance Modeling                                              │
│     ├─ SASS SIM: Accel-Sim (SASS traces)                            │
│     ├─ PTX SIM: Accel-Sim (PTX)                                     │
│     ├─ HW: Hardware perf counters                                    │
│     └─ HYBRID: HW counters + Accel-Sim (L2, NoC)                   │
│                                                                     │
│  ⑦ Quadratic Programming                                            │
│     └─ 반복적 최적화 → scaling factors 도출                         │
│                                                                     │
│  ⑧ Validation                                                       │
│     └─ 26개 validation kernels + Technology scaling                 │
│                                                                     │
│  결과: AccelWattch Power Model (config XML files)                    │
│     └─ constant_power + static_power + dynamic_power                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 4가지 AccelWattch Variants

| Variant | Performance Source | Activity Source | 유연성 | 정확도 (MAPE) |
|---------|-------------------|-----------------|--------|--------------|
| **SASS SIM** | Accel-Sim (SASS) | 시뮬레이터 | 높음 (Design exploration) | 9.2% |
| **PTX SIM** | Accel-Sim (PTX) | 시뮬레이터 | 높음 | 13.7% |
| **HW** | 실제 하드웨어 | HW perf counters | 낮음 (실물 필요) | 7.5% |
| **HYBRID** | HW + Accel-Sim | HW + Sim (L2, NoC) | 중간 | 8.2% |

---

## 3. Power Modeling 3요소

GPU의 총 전력은 3가지 요소로 분해된다:

### Eq.(1): 총 전력 분해

```
P_total = P_proc,dyn + P_mem,dyn + P_proc,static + P_mem,static + P_const
```

| 요소 | 설명 | 의존 변수 |
|------|------|----------|
| **P_proc,dyn** | GPU 칩 dynamic power | 주파수, 전압, 기술 노드 (C, V, f) |
| **P_mem,dyn** | 메모리 dynamic power | 주파수, 전압, 기술 노드 |
| **P_proc,static** | GPU 칩 static (leakage) power | 전압, 기술 노드 |
| **P_mem,static** | 메모리 static power | 전압, 기술 노드 |
| **P_const** | 보드 팬, 주변 회로 등 | 고정 (주파수/전압 무관) |

### Eq.(2): 주파수/전압 의존성 표현

```
P_total = aCV²f + a'CV²f + bV + b'V + P_const
       = mCV²f + nV + P_const
```

여기서:
- `C`: gate capacitance
- `V`: 공급 전압
- `f`: 클럭 주파수
- `a, a', b, b'`: 기술/설계 상수
- `m, n`: 요약 상수

---

## 4. DVFS-Aware Constant Power Modeling

### 4.1 기존 방법의 문제점

GPUWattch는 Eq.(2)를 사용하여 주파수를 0으로 외삽(extrapolation)하면 `mCV²f` 항이 사라져 `nV + P_const`만 남는다고 가정했다. 그러나 **현대 GPU는 DVFS를 사용**하므로 주파수가 변하면 전압도 함께 변한다:

```
V ≈ kf  (전압은 주파수에 근사적으로 비례)
```

이를 Eq.(2)에 대입하면:

### Eq.(3): DVFS-Aware 총 전력

```
P_total = βCf³ + τf + P_const
```

> **핵심 발견**: DVFS 하에서 전력은 **quadratic term이 빠진 3차 다항식**으로 근사된다. 이는 GPUWattch의 선형 가정과 근본적으로 다르다.

### 4.2 P_const 추정 방법

1. 다양한 클럭 주파수에서 microbenchmark 실행
2. 하드웨어 전력 측정 수행 (NVML API, nvidia-smi)
3. Eq.(3) 형태의 3차 다항식 fitting (Pearson r = 0.998)
4. f = 0으로 외삽 → y절편 = P_const

**Volta GV100 결과**: P_const = 32.5 W

---

## 5. Power-Gating-Aware Static Power Modeling

### 5.1 Power Gating의 3계층

현대 GPU는 세밀한 power gating을 수행한다:

```
┌──────────────────────────────────────────────────┐
│ Chip-Wide Components (L2 cache 등)               │
│   └─ 1개 SM이라도 active → 전체 활성화           │
│      첫 SM 활성화 시 47× 더 많은 전력 소모       │
├──────────────────────────────────────────────────┤
│ SM-Wide Components (L1 cache, shared memory 등)  │
│   └─ 1개 lane이라도 active → SM 전체 활성화      │
│      첫 lane 활성화 시 31× 더 많은 전력 소모     │
├──────────────────────────────────────────────────┤
│ Lane-Specific Components (INT32, FP32 cores 등)  │
│   └─ 해당 lane의 기능 유닛만 활성화              │
└──────────────────────────────────────────────────┘
```

### 5.2 Linear Static Power Model — Eq.(4)

```
P_static,addLane = (P_static,32Lanes - P_static,firstLane) / 31
P_static,yLanes = P_static,firstLane + P_static,addLane · (y - 1)
```

- `P_static,firstLane`: 첫 lane 활성화 시 static power (SM-wide + 1 lane)
- `P_static,addLane`: 추가 lane당 static power (lane-specific만)

### 5.3 Half-Warp Static Power Model — Eq.(5)

Volta의 SM은 4개 processing block으로 나뉘며, 각각 16 CUDA cores를 가진다. Warp(32 threads)는 2개의 16-thread half-warp으로 실행된다:

```
                    ┌── y ≤ 16: 1개 processing block만 active
P_static,yLanes = ──┤
                    └── y > 16: "full half-warp" + "partial half-warp" 교대
```

```
P_static,yLanes = { P_static,firstLane + P_static,addLane · (y-1),           if y ≤ 16
                  { P_static,firstLane + ½·P_static,addLane · 15
                  {                     + ½·P_static,addLane · (y-17),        if y > 16
```

> **핵심 발견**: y = 16 → y = 17로 넘어갈 때 전력이 오히려 **감소**하는 톱니파(sawtooth) 패턴이 발생한다. 이는 16 lanes일 때 모든 processing block이 항상 active이지만, 17 lanes일 때는 "full" + "partial" half-warp이 교대로 실행되어 평균적으로 일부만 active이기 때문이다.

### 5.4 Instruction Mix 기반 9개 카테고리

AccelWattch는 instruction mix에 따라 적절한 static power model을 선택한다:

| 카테고리 | 사용 유닛 | Static Model |
|---------|----------|-------------|
| INT only | INT32 | Half-warp |
| INT ADD only | INT32 (ADD) | Half-warp |
| INT MUL only | INT32 (MUL) | Half-warp |
| INT + FP | INT32 + FP32 | Linear → Half-warp 혼합 |
| INT + FP + DP | INT32 + FP32 + FP64 | Linear |
| INT + FP + SFU | INT32 + FP32 + SFU | Linear |
| INT + FP + TEX | INT32 + FP32 + TEX | Linear |
| INT + FP + TENSOR | INT32 + FP32 + Tensor | Linear |
| LIGHT (nanosleep) | 최소 유닛 | Half-warp |

### 5.5 Idle SM Power — Eq.(6)~(8)

Active SM 수를 변화시키며 idle SM의 전력을 측정:

```
P_dyn+static,perActiveSM,i = (P_total,80SMs,i - P_const) / 80        ... (6)
P_idleSMs,i = P_total,i - P_const - P_dyn+static,perActiveSM,i · N_activeSMs  ... (7)
P_perIdleSM,i = P_idleSMs,i / N_idleSMs                               ... (8)
```

최종 idle SM power는 모든 microbenchmark에 대한 **기하평균**으로 결정.

### 5.6 전체 Static Power Model — Eq.(10)

```
P_total,yLanes,kSMs = P_dyn + P_static,yLanes,perActiveSM · k
                     + P_perIdleSM · (80 - k) + P_const
```

여기서 80은 GV100의 총 SM 수 (GPU마다 다름).

---

## 6. Dynamic Power Modeling

### 6.1 기본 개념 — Eq.(11)

N개의 마이크로아키텍처 컴포넌트가 있을 때, dynamic power는:

```
P_dyn = Σ(i=1→N) [a_i · E_i / T_elapsed]
```

- `a_i`: 컴포넌트 i의 activity factor (접근 횟수)
- `E_i`: 컴포넌트 i의 접근당 에너지 (초기 추정치, 부정확)
- `T_elapsed`: 실행 시간

### 6.2 Scaling Factor 도입 — Eq.(12)

초기 `E_i` 추정치가 부정확하므로 unknown 변수 `x_i`를 도입:

```
P_est = Σ(i=1→N) [a_i · Ê_i / T_elapsed · x_i]
      + P_static,yLanes,perActiveSM · k
      + P_perIdleSM · (80 - k) + P_const
```

### 6.3 22개 Dynamic Power Components (Table 1)

AccelWattch는 다음 22개 하드웨어 컴포넌트의 dynamic power를 추적한다:

| # | Component | Hardware Unit | 설명 |
|---|-----------|---------------|------|
| 1 | IBP | Instruction Buffer | 명령어 버퍼 |
| 2 | ICP | L0 Inst. Cache | L0 명령어 캐시 |
| 3 | DCP | L1d Cache | L1 데이터 캐시 |
| 4 | TCP | Texture Cache | 텍스처 캐시 |
| 5 | CCP | Constant Cache | 상수 캐시 |
| 6 | SHRDP | Shared Memory | 공유 메모리 |
| 7 | RFP | Register File | 레지스터 파일 |
| 8 | INTP | INT32 core | 정수 ALU |
| 9 | FPUP | FP32 core | 부동소수점 유닛 |
| 10 | DPUP | FP64 core | 배정밀도 유닛 |
| 11 | INT_MULP | int mul/mad | 정수 곱셈 |
| 12 | FP_MULP | fp mul/fma | FP 곱셈 |
| 13 | FP_DIVP | fp div | FP 나눗셈 (SFU) |
| 14 | FP_SQRTP | sqrt | 제곱근 (SFU) |
| 15 | FP_LGP | log | 로그 (SFU) |
| 16 | FP_SINP | sin/cos | 삼각함수 (SFU) |
| 17 | FP_EXP | exp | 지수 (SFU) |
| 18 | DP_MULP | dp mul/fma | DP 곱셈 |
| 19 | TENSORP | Tensor Core | 텐서 코어 |
| 20 | TEXP | Texture Unit | 텍스처 유닛 |
| 21 | L2CP+NOCP | L2 Cache + NoC | L2 캐시 + 인터커넥트 (결합) |
| 22 | DRAMP+MCP | DRAM + MC | DRAM + 메모리 컨트롤러 (결합) |
| + | SCHEDP | Scheduler | 스케줄러 (Others에 포함) |
| + | PIPEP | SM Pipeline | 파이프라인 (Others에 포함) |
| + | IDLE_COREP | Idle SM | 유휴 SM |
| + | CONSTP | Constant | 상수 전력 |
| + | STATICP | Static | 정적 전력 |

---

## 7. Quadratic Programming 최적화

### 7.1 문제 정의 — Eq.(13)

M개의 microbenchmark(워크로드)에서 수집한 activity factor와 power 측정값으로 연립방정식 구성:

```
P_est^{M×(N+3)} × X^{(N+3)×1} = P_meas^{M×1}
```

- 행렬 A (M × 31): 각 microbenchmark의 31개 power counter 값
- 벡터 X (31 × 1): 찾고자 하는 scaling factors
- 벡터 b (M × 1): 하드웨어에서 측정한 실제 전력

### 7.2 최적화 공식 — Eq.(14)

```
X* = arg min_X { X^T · P_est^T · P_est · X - (P_est^T · P_meas)^T · X }

제약 조건:
  ∀i: 0.001 ≤ X_i ≤ 1000                    (범위 제약)
  X_static = X_idleSM = X_const = 1          (이미 모델링 완료)
  X_ALU ≤ X_FPU ≤ X_DPU ≤ X_iMAD            (에너지 순서 제약)
  X_fpmul ∈ {X_imul, X_dpmul, X_log, X_sin, X_exp, X_tensor, X_tex}
                                               (SFU 에너지 제약)
```

### 7.3 MATLAB 구현 (quadprog_solver.m)

```matlab
% 입력: CSV 파일 (102 microbenchmarks × 31 power counters + 1 measured power)
input = csvread('accelwattch_volta_sass_sim.csv');
A = input(:,1:31);    % activity factor matrix
b = input(:,32);       % measured power vector

% 범위 제약
l = 0.1 * ones(1,31);   % lower bounds
u = 1000 * ones(1,31);  % upper bounds

% Static, Idle, Const는 이미 모델링됨 → scaling = 1 고정
l(29)=1; u(29)=1;  % IDLE_COREP
l(30)=1; u(30)=1;  % CONSTP
l(31)=1; u(31)=1;  % STATICP

% McPAT 기반 에너지 순서 제약 (16개)
C = zeros(16,31);  D = zeros(16,1);
C(1,8)=1; C(1,9)=-1.843;    % INT ≤ 1.843 × FPU
C(2,9)=1; C(2,10)=-0.999;   % FPU ≤ DPU
C(3,8)=1; C(3,11)=-1.107;   % INT ≤ 1.107 × INT_MUL24
...
C(15,15)=1; C(15,23)=-75.07; % FP_MUL ≤ 75.07 × TENSOR
C(16,15)=1; C(16,24)=-0.999; % FP_MUL ≤ TEXP

% Quadratic Programming 실행
result = quadprog(2*A'*A, -2*A'*b, C, D, [], [], l, u);
csvwrite('scaled_coefficients.csv', result);
```

### 7.4 반복 최적화 과정

```
1. 초기 scaling factors 설정 (Fermi GPUWattch 모델 or 모두 1.0)
2. AccelWattch 시뮬레이션 실행 → activity factors 수집
3. Quadratic programming → 새 scaling factors 도출
4. 새 scaling factors로 XML config 업데이트
5. 다시 시뮬레이션 → 오차 감소 확인
6. 수렴할 때까지 2-5 반복
```

> **참고**: Fermi 시작점에서 출발한 모델이 더 높은 정확도를 달성 (9.2% vs 14.8% MAPE on validation set).

---

## 8. GPU Configuration 구조 분석

### 8.1 설정 파일 체계

AccelWattch는 GPU마다 3종류의 설정 파일을 사용한다:

```
configs/tested-cfgs/SM7_QV100/
├── gpgpusim.config              # GPGPU-Sim 아키텍처 설정
├── trace.config                 # 명령어 latency/throughput 설정
└── accelwattch_sass_sim.xml     # Power model 설정 (AccelWattch 핵심)
```

### 8.2 Power XML 구조 (accelwattch_sass_sim.xml)

XML은 3개 섹션으로 구성된다:

#### 섹션 1: Dynamic Power Activity Factors (34개 파라미터)

| 파라미터 | 의미 | V100 값 (예시) |
|---------|------|---------------|
| TOT_INST | 명령어 버퍼 activity | 10.0 |
| FP_INT | 스케줄러 activity | 4.661 |
| IC_H / IC_M | I-cache hit/miss | 8.59 / 29.74 |
| DC_RH / DC_RM | L1D read hit/miss | 9.84 / 10.95 |
| DC_WH / DC_WM | L1D write hit/miss | 0.68 / 17.68 |
| INT_ACC | 정수 ALU 접근 | 14.99 |
| FP_ACC | FPU 접근 | 0.53 |
| DP_ACC | DPU 접근 | 0.78 |
| TENSOR_ACC | 텐서 코어 접근 | 0.82 |
| MEM_RD / MEM_WR | DRAM read/write | 0.026 / 0.031 |
| L2_RH / L2_RM | L2 read hit/miss | 1.26 / 2.39 |
| NOC_A | NoC 접근 | 32.09 |

> **이 값들은 quadratic programming에서 도출된 scaling factors이다.** 시뮬레이터가 수집한 raw activity count에 이 값을 곱하여 dynamic power를 계산한다.

#### 섹션 2: Static & Constant Power (18개 파라미터)

| 파라미터 | 의미 | V100 값 |
|---------|------|--------|
| constant_power | P_const | 32.33 W |
| idle_core_power | Idle SM당 전력 | 0.283 W |
| static_cat1_flane | INT First Lane | 15.29 W |
| static_cat1_addlane | INT Additional Lane | 0.586 W |
| static_cat6_flane | INT+FP+TENSOR First Lane | 48.95 W |
| static_cat6_addlane | INT+FP+TENSOR Additional Lane | 0.0 W |

#### 섹션 3: Legacy GPUWattch 파라미터

McPAT 기반의 아키텍처 설정 (캐시 구성, NoC 토폴로지, 메모리 컨트롤러 등). AccelWattch에서 초기 energy 추정치 (Ê_i)를 얻는데 사용.

### 8.3 현재 지원 GPU 및 설정 상태

| GPU | Arch | 공정 | SM수 | AccelWattch XML | 상태 |
|-----|------|------|------|----------------|------|
| GTX 480 | Fermi | 40nm | 15 | 없음 | 구형 |
| Kepler Titan | Kepler | 28nm | 14 | 없음 | 구형 |
| Titan X | Pascal | 16nm | 56 | **있음** | Case Study |
| Titan V | Volta | 12nm | 80 | **있음** | 지원 |
| Quadro V100 | Volta | 12nm | 80 | **있음** | **주력 (Validation)** |
| GV100 | Volta | 12nm | 80 | **있음** | 지원 |
| RTX 2060 | Turing | 12nm | 30 | 없음 | 부분 지원 |
| RTX 2060S | Turing | 12nm | 34 | **있음** | Case Study |
| **A100** | **Ampere** | **7nm** | **108** | **없음** | **미지원 (개선 필요)** |
| RTX 3070 | Ampere | 8nm | 46 | 없음 | 미지원 |
| **H100** | **Hopper** | **4nm** | **132** | **없음** | **미지원 (개선 필요)** |

---

## 9. Microbenchmark 설계 및 역할

### 9.1 목적

102개의 microbenchmark는 GPU의 개별 하드웨어 컴포넌트를 **격리하여 스트레스**하기 위해 설계되었다. 이를 통해:

1. 각 컴포넌트의 activity factor를 분리 측정
2. Static power model의 파라미터 (firstLane, addLane) 추출
3. Quadratic programming의 입력 데이터 생성

### 9.2 Microbenchmark 카테고리 (Table 2)

| 카테고리 | μBench 수 | 대상 컴포넌트 |
|---------|-----------|-------------|
| Active/Idle SMs | 12 | SM 활성화/비활성화 패턴 |
| INT32 core | 9 | 정수 연산 유닛 |
| FP32 core | 8 | 단정밀도 부동소수점 |
| FP64 core | 8 | 배정밀도 부동소수점 |
| SFU | 9 | Special Function Unit |
| Texture Unit | 7 | 텍스처 처리 유닛 |
| Tensor Core | 6 | 텐서 코어 |
| Register File | 1 | 레지스터 파일 |
| dCaches + Sh.Mem + NoC | 11 | 캐시/공유메모리/인터커넥트 |
| DRAM + MC | 2 | 메모리 시스템 |
| Mix | 29 | 다양한 조합 |
| Other (L0, L1i, Pipeline, Scheduler) | 102 (전체) | 파이프라인, 스케줄러 등 |

### 9.3 μBenchmark 설계 원칙

1. **컴파일러 최적화 우회**: inline assembly (PTX), pointer-chasing
2. **ROI (Region of Interest)**: unrolled loop 안에서 측정
3. **높은 반복 횟수**: NVML 샘플링 (50-100Hz)에 충분한 실행 시간 확보
4. **온도 제어**: 65°C에서 안정화 후 측정 (static power의 온도 의존성 제거)

### 9.4 Dynamic Power Heat-Map (Figure 6)

각 microbenchmark가 어떤 컴포넌트를 주로 사용하는지 heat-map으로 확인:
- 대각선이 "뜨거울수록" 좋은 격리
- INT32 벤치 → INT32 Core가 가장 높은 비율
- Tensor Core 벤치 → Tensor가 20%+ 차지

---

## 10. SASS Opcode → Power Component 매핑

### 10.1 매핑 테이블 (accelwattch_component_mapping.h)

SASS 명령어를 power component에 매핑하는 것이 AccelWattch의 핵심이다:

```
┌─────────────────────────────────────────────────┐
│         SASS Opcode → Power Component           │
├─────────────────┬───────────────────────────────┤
│ FADD, FSEL, ... │ FP__OP (FP Addition)          │
│ FFMA, FMUL, ... │ FP_MUL_OP (FP Multiply/FMA)  │
│ IADD3, MOV, ... │ INT__OP (Integer ALU)         │
│ IMAD, IMUL, ... │ INT_MUL_OP (Integer Multiply) │
│ DADD, DSETP     │ DP___OP (Double Precision)    │
│ DFMA, DMUL      │ DP_MUL_OP (DP Multiply)       │
│ MUFU             │ FP_SIN_OP (SFU - 세분화*)    │
│ HMMA, IMMA      │ TENSOR__OP (Tensor Core)      │
│ TEX, TLD, TXD   │ TEX__OP (Texture)             │
│ LD, ST, ATOM    │ OTHER_OP (Memory/Branch/Sync)  │
│ BRA, EXIT, NOP  │ OTHER_OP                      │
└─────────────────┴───────────────────────────────┘

* MUFU (Multi-Function Unit)는 trace_driven.cc에서 operand를 분석하여
  SIN/COS, EX2, RSQ, LG2로 세분화함
```

### 10.2 아키텍처별 고유 명령어

| 아키텍처 | 고유 명령어 (예시) |
|---------|-------------------|
| Pascal | RRO, XMAD, TEXS, CAL, SSY |
| Volta | BMSK, IADD3, BSSY, WARPSYNC |
| Turing | BMMA, LDSM, R2UR, UCLEA |
| Ampere | DMMA, HMNMX2, LDGSTS, REDUX |

---

## 11. 소스코드 호출구조

### 11.1 End-to-End 파이프라인

```
[1. Trace Generation]
    util/tracer_nvbit/run_hw_trace.py
    └─ NVBit으로 GPU 실행 trace 수집 (SASS instructions)

[2. Simulation]
    gpu-simulator/main.cc → accel-sim.cc
    └─ trace-driven/trace_driven.cc
       └─ trace-parser/trace_parser.cc (trace 파싱)
       └─ .vendor/gpgpu-sim_distribution/ (GPGPU-Sim 코어)
          └─ src/accelwattch/ (power 계산 엔진)

[3. Power Calculation]
    .vendor/gpgpu-sim_distribution/src/accelwattch/
    ├─ accelwattch_component_mapping.h (opcode → component)
    ├─ 매 500 cycles마다 activity stats 수집
    ├─ XML config의 scaling factors 적용
    └─ accelwattch_power_report.log 출력

[4. Post-Processing]
    util/accelwattch/gen_sim_power_csv.py
    └─ power report → CSV 변환
    └─ component별 power breakdown

[5. Coefficient Tuning]
    util/accelwattch/quadprog_solver.m
    └─ CSV 입력 → quadratic programming
    └─ scaled_coefficients.csv 출력

[6. Validation]
    util/plotting/plot-correlation.py
    └─ 시뮬레이션 vs 하드웨어 전력 비교
```

### 11.2 핵심 데이터 흐름

```
SASS Trace → [Trace Parser] → inst_trace_t
                                    │
                                    ▼
                          [trace_driven.cc]
                          trace_warp_inst_t → OpcodePowerMap 조회
                                    │
                                    ▼
                          [GPGPU-Sim Core]
                          매 cycle 성능 시뮬레이션
                          activity counters 누적
                                    │
                                    ▼ (매 500 cycles)
                          [AccelWattch Engine]
                          activity × scaling_factor = component power
                          Σ components = P_dyn
                          P_total = P_dyn + P_static(y,k) + P_const
                                    │
                                    ▼
                          accelwattch_power_report.log
```

---

## 12. Validation 결과 요약

### 12.1 Volta GV100 (주력 검증)

| Variant | MAPE | 95% CI | Max Error | Pearson r |
|---------|------|--------|-----------|-----------|
| SASS SIM | **9.2%** | ±3.12% | 30% | 0.83-0.91 |
| PTX SIM | 13.7% | - | - | - |
| HW | **7.5%** | - | - | - |
| HYBRID | 8.2% | - | - | - |

26개 validation kernels 중:
- 17/26 (65%)가 < 10% absolute error
- 4/26 (15%)만 > 20% error

### 12.2 Case Study: Technology Scaling

| 대상 GPU | 방법 | SASS MAPE | PTX MAPE |
|---------|------|-----------|----------|
| Pascal Titan X (16nm) | Volta 모델 + tech scaling | **11%** | 10.8% |
| Turing RTX 2060S (12nm) | Volta 모델 직접 적용 | **13%** | 14% |

> **핵심**: Volta에서 학습한 모델을 재학습 없이 Pascal/Turing에 적용해도 합리적 정확도 달성.

### 12.3 DeepBench (Deep Learning)

- CONV, RNN-LSTM, GEMM의 train + inference
- 전체 MAPE: **12.79%**
- 제한사항: 커널 동시 실행 스케줄링 차이로 인한 오차

### 12.4 GPUWattch와 비교

| 메트릭 | AccelWattch | GPUWattch |
|--------|------------|-----------|
| Volta MAPE | 9.2% | **219%** |
| Maximum Error | 30% | **447%** |
| 평균 Power 추정 | ~현실적 | 530W (실제 max ~250W) |
| Error Factor | 1× | **22-24×** |

---

## 13. 핵심 Equation 정리

### 총 전력 모델

```
Eq.(1):  P_total = P_proc,dyn + P_mem,dyn + P_proc,static + P_mem,static + P_const

Eq.(2):  P_total = mCV²f + nV + P_const

Eq.(3):  P_total = βCf³ + τf + P_const     (DVFS-Aware, V≈kf)
```

### Static Power 모델

```
Eq.(4):  P_static,yLanes = P_static,firstLane + P_static,addLane · (y - 1)
                                                          [Linear Model]

Eq.(5):  P_static,yLanes = { firstLane + addLane·(y-1),            y ≤ 16
                            { firstLane + ½·addLane·15 + ½·addLane·(y-17), y > 16
                                                          [Half-warp Model]
```

### Idle SM 모델

```
Eq.(6):  P_dyn+static,perActiveSM = (P_total,80SMs - P_const) / 80

Eq.(7):  P_idleSMs = P_total - P_const - P_dyn+static,perActiveSM · N_activeSMs

Eq.(8):  P_perIdleSM = ⁿ√(∏ P_perIdleSM,i)     [기하평균]
```

### 전체 모델

```
Eq.(10): P_total = P_dyn + P_static,yLanes,perActiveSM · k
                 + P_perIdleSM · (80 - k) + P_const
```

### Dynamic Power 모델

```
Eq.(11): P_dyn = Σ(i=1→N) [a_i · E_i / T_elapsed]

Eq.(12): P_est = Σ [a_i · Ê_i · x_i / T_elapsed]
               + P_static · k + P_idle · (80-k) + P_const

Eq.(13): P_est^{M×(N+3)} × X^{(N+3)×1} = P_meas^{M×1}

Eq.(14): X* = argmin_X { ||P_est · X - P_meas||² }
         s.t.  0.001 ≤ X_i ≤ 1000
               X_ALU ≤ X_FPU ≤ X_DPU
               X_fpmul ≤ k · X_tensor    (k = 75.07)
```

---

## 14. 용어 사전

| 용어 | 설명 |
|------|------|
| **DVFS** | Dynamic Voltage and Frequency Scaling. 동적으로 전압과 주파수를 조절하는 기법 |
| **MAPE** | Mean Absolute Percentage Error. 평균 절대 백분율 오차 |
| **SM** | Streaming Multiprocessor. GPU의 기본 연산 단위 |
| **Lane** | SM 내 하나의 CUDA core에 해당하는 실행 경로 |
| **Warp** | 32개 thread로 구성된 GPU 실행 단위 |
| **Half-warp** | 16개 thread 단위의 실행 (processing block 1개) |
| **Power Gating** | 비활성 회로의 전원을 차단하여 leakage 절감 |
| **Activity Factor** | 단위 시간당 컴포넌트 접근 횟수 |
| **Scaling Factor** | Quadratic programming으로 도출된 보정 계수 |
| **McPAT** | Multi-core Power, Area, and Timing 모델링 프레임워크 |
| **NVBit** | NVIDIA Binary Instrumentation Tool. SASS trace 수집 도구 |
| **NVML** | NVIDIA Management Library. GPU 전력/온도 모니터링 API |
| **SASS** | Shader ASSembly. NVIDIA GPU의 native machine ISA |
| **PTX** | Parallel Thread eXecution. NVIDIA의 virtual ISA |
| **SFU** | Special Function Unit. sin, cos, log, exp 등 연산 |
| **Accel-Sim** | SASS trace 기반 GPU 성능 시뮬레이터 |
| **GPGPU-Sim** | GPU 아키텍처 시뮬레이터 (Accel-Sim의 기반) |
| **Technology Scaling** | 공정 노드 차이에 따른 전력 보정 (예: 12nm→7nm) |
| **ILP** | Instruction-Level Parallelism. 명령어 수준 병렬성 |
| **Processing Block** | SM 내 4개의 실행 블록 (각 16 CUDA cores) |

---

> **다음 문서**: [02_Improvement_Points.md](02_Improvement_Points.md) — 개선 포인트 분석
