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

GPU의 총 소비 전력(P_total)은 5개 항으로 분해된다. GPU 칩이 실제로 연산을 수행할 때 소모하는 dynamic power(P_proc,dyn)와 메모리가 읽기/쓰기를 수행할 때 소모하는 dynamic power(P_mem,dyn), GPU 칩 내 트랜지스터의 누설 전류로 인해 연산 여부와 무관하게 흐르는 static power(P_proc,static, P_mem,static), 그리고 보드의 팬, 전압 조정기 등 GPU 칩 외부에서 소모되는 일정한 constant power(P_const)로 나뉜다.

| 기호 | 의미 | 설명 | 의존 변수 |
|------|------|------|----------|
| **P_total** | 총 전력 | GPU 시스템 전체의 소비 전력 | 전체 |
| **P_proc,dyn** | GPU 칩 dynamic power | 연산 유닛, 캐시 등이 활성 동작 시 소모하는 전력 | 주파수(f), 전압(V), 커패시턴스(C) |
| **P_mem,dyn** | 메모리 dynamic power | DRAM(HBM) 읽기/쓰기 시 소모하는 전력 | 메모리 주파수, 전압 |
| **P_proc,static** | GPU 칩 static power | 트랜지스터 누설 전류에 의한 전력 (전원이 켜져 있으면 항상 소모) | 전압, 공정 노드 |
| **P_mem,static** | 메모리 static power | 메모리 셀의 누설 전력 | 전압, 공정 노드 |
| **P_const** | 상수 전력 | 보드 팬, 전압 조정기, PCIe 인터페이스 등 주변 회로 전력 | 고정 (주파수/전압 무관) |

### Eq.(2): 주파수/전압 의존성 표현

```
P_total = aCV²f + a'CV²f + bV + b'V + P_const
       = mCV²f + nV + P_const
```

이 수식은 Eq.(1)의 각 항을 물리적 파라미터로 전개한 것이다. Dynamic power는 CMOS 회로에서 `CV²f`에 비례하는데(스위칭할 때 커패시터를 충방전하는 에너지), GPU 칩(계수 a)과 메모리(계수 a')를 합쳐 `mCV²f`로 단순화했다. Static power는 전압(V)에 비례하는 누설 전류이므로 `nV`로 표현된다.

| 기호 | 의미 | 단위 |
|------|------|------|
| **C** | 게이트 커패시턴스(gate capacitance). 트랜지스터 게이트에 축적되는 전하량의 척도 | F (패럿) |
| **V** | 공급 전압(supply voltage). GPU 코어에 공급되는 전원 전압 | V (볼트) |
| **f** | 클럭 주파수(clock frequency). GPU 코어의 동작 주파수 | Hz |
| **a, a'** | GPU 칩과 메모리 각각의 dynamic power 기술 계수. 공정/설계에 의해 결정되는 상수 | 무차원 |
| **b, b'** | GPU 칩과 메모리 각각의 static power 계수. 누설 전류 특성을 반영 | W/V |
| **m** | a + a'를 합친 통합 dynamic power 계수 | 무차원 |
| **n** | b + b'를 합친 통합 static power 계수 | W/V |

---

## 4. DVFS-Aware Constant Power Modeling

### 4.1 기존 방법의 문제점

GPUWattch는 Eq.(2)를 사용하여 주파수를 0으로 외삽(extrapolation)하면 `mCV²f` 항이 사라져 `nV + P_const`만 남는다고 가정했다. 이 방법은 주파수와 전압이 독립적으로 변한다는 전제에 기반한다.

그러나 **현대 GPU는 DVFS(Dynamic Voltage and Frequency Scaling)를 사용**한다. DVFS란 GPU가 동작 주파수를 높일 때 안정적 동작을 위해 전압도 함께 올리는 기술이다. 따라서 주파수가 변하면 전압도 함께 변하며, 두 변수는 독립이 아니다:

```
V ≈ kf
```

여기서 k는 전압-주파수 비례 상수로, GPU의 V-F curve(전압-주파수 특성 곡선)의 기울기에 해당한다. 이 근사는 V100의 실측 데이터에서 높은 선형 상관(r ≈ 0.99)을 보였다.

이 관계를 Eq.(2)의 `V`에 대입하면 `mC(kf)²f = mk²Cf³`이 되고, `n(kf) = nkf`가 된다. 이를 정리하면:

### Eq.(3): DVFS-Aware 총 전력

```
P_total = βCf³ + τf + P_const
```

| 기호 | 의미 | 유도 과정 | 단위 |
|------|------|----------|------|
| **β (베타)** | dynamic power의 전압-주파수 결합 계수. Eq.(2)의 `mk²`에 해당. 공정 기술과 설계에 의해 결정됨 | β = mk² (m은 dynamic 계수, k는 V-F 비례 상수) | W·s³ |
| **C** | 게이트 커패시턴스. Eq.(2)와 동일 | - | F |
| **f** | SM 클럭 주파수 | - | Hz |
| **τ (타우)** | static power의 주파수 비례 계수. DVFS에서 전압이 주파수에 비례하므로, 전압에 비례하는 누설 전력이 주파수의 함수가 됨 | τ = nk (n은 static 계수, k는 V-F 비례 상수) | W·s |
| **P_const** | 상수 전력. 주파수와 전압에 무관한 보드/팬/주변회로 전력 | Eq.(1)과 동일 | W |

이 수식의 물리적 의미는 다음과 같다. 첫째 항 `βCf³`은 GPU 코어가 실제로 스위칭할 때 소모하는 dynamic power인데, DVFS로 인해 전압도 주파수와 함께 올라가므로 주파수의 **3승**에 비례한다(일반적인 `CV²f`는 2승이지만, `V=kf`를 대입하면 3승이 됨). 둘째 항 `τf`는 트랜지스터 누설 전류에 의한 static power인데, 전압이 올라가면 누설도 증가하므로 주파수에 **1승** 비례한다. 셋째 항 `P_const`는 주파수/전압과 무관한 일정 전력이다.

> **핵심 발견**: DVFS 하에서 전력은 **2차(quadratic) 항이 빠진 3차 다항식**으로 근사된다. 즉, f⁰(상수), f¹(1차), f³(3차) 항만 있고 f²(2차) 항이 없다. 이는 GPUWattch의 선형(1차) 가정과 근본적으로 다르며, V-F 비례 관계에서 자연스럽게 유도되는 결과이다.

### 4.2 P_const 추정 방법

이 3차 다항식 모델을 실측 데이터에 fitting하면 P_const를 추정할 수 있다:

1. GPU의 SM 클럭 주파수를 여러 단계로 변경한다 (예: 200~1400 MHz).
2. 각 주파수에서 동일한 microbenchmark를 실행하며 NVML API 또는 nvidia-smi로 전력을 측정한다.
3. 측정된 (f, P_total) 데이터쌍을 Eq.(3) 형태의 3차 다항식 `P = βCf³ + τf + P_const`으로 curve fitting한다. V100에서 Pearson r = 0.998로 매우 높은 상관을 보였다.
4. fitting된 다항식에 f = 0을 대입하면 `βC·0³ + τ·0 + P_const = P_const`만 남으므로, y절편이 곧 P_const이다.

**Volta GV100 결과**: P_const = 32.5 W (TDP 250W의 약 13%)

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
P_static,addLane = (P_static,32Lanes − P_static,firstLane) / 31
P_static,yLanes = P_static,firstLane + P_static,addLane · (y − 1)
```

GPU에서 warp(32 threads)가 실행될 때, 모든 thread가 항상 활성인 것은 아니다. 조건 분기(thread divergence) 등으로 일부 lane만 활성일 수 있다. 이 수식은 활성 lane 수(y)에 따른 static power를 선형으로 모델링한다.

첫 lane이 활성화되면 해당 SM의 **SM-wide 공유 자원**(L1 캐시, 공유 메모리 등)이 모두 켜지므로 P_static,firstLane은 상대적으로 크다. 이후 추가되는 lane은 자신의 **연산 유닛(INT32, FP32 등)**만 켜므로 P_static,addLane은 작다.

| 기호 | 의미 | 측정 방법 |
|------|------|----------|
| **y** | warp 내 활성(active) thread 수. 1~32 범위 | 시뮬레이터 또는 HW counter에서 `avg_threads_per_warp`로 수집 |
| **P_static,firstLane** | 첫 번째 lane 활성화 시 static power. SM-wide 컴포넌트 + 1개 lane의 leakage 포함 | microbenchmark에서 1 thread만 실행하여 측정 |
| **P_static,addLane** | 추가 lane 1개당 static power 증분. lane 전용 연산 유닛의 leakage만 포함 | (32 thread 전력 − 1 thread 전력) / 31로 계산 |
| **P_static,32Lanes** | 32개 lane 모두 활성 시의 static power | microbenchmark에서 32 thread 실행하여 측정 |
| **P_static,yLanes** | y개 lane 활성 시의 static power (이 수식의 출력) | 위 파라미터로 계산 |

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

GPU의 모든 SM이 활성인 것은 아니다. 커널이 적은 수의 thread block을 사용하면 일부 SM은 유휴(idle) 상태가 된다. Idle SM도 전원이 완전히 꺼지는 것이 아니라 leakage 전류로 소량의 전력을 소모한다. 이를 모델링하기 위해 active SM 수를 변화시키며 실험한다.

```
P_dyn+static,perActiveSM,i = (P_total,80SMs,i − P_const) / 80        ... (6)
P_idleSMs,i = P_total,i − P_const − P_dyn+static,perActiveSM,i · N_activeSMs  ... (7)
P_perIdleSM,i = P_idleSMs,i / N_idleSMs                               ... (8)
```

| 기호 | 의미 |
|------|------|
| **P_dyn+static,perActiveSM,i** | microbenchmark i를 실행할 때, **active SM 1개당** dynamic+static 전력. 80개 SM 모두 활성인 실험에서 P_const를 빼고 80으로 나누어 구함 |
| **P_total,80SMs,i** | microbenchmark i를 80개 SM 모두 사용하여 실행했을 때 측정한 총 전력 |
| **P_total,i** | microbenchmark i를 **N_activeSMs개의 SM**만 사용하여 실행했을 때 측정한 총 전력 |
| **N_activeSMs** | 활성 SM 수 (1~80 중 실험에서 설정한 값) |
| **N_idleSMs** | 비활성(idle) SM 수 = 80 − N_activeSMs |
| **P_idleSMs,i** | microbenchmark i에서 모든 idle SM의 **총** 전력 |
| **P_perIdleSM,i** | microbenchmark i에서 idle SM **1개당** 전력 |
| **i** | microbenchmark 인덱스 (1~n, n은 실험에 사용한 microbenchmark 수) |

Eq.(6)은 80개 SM 모두 활성인 상태에서 active SM 1개당 평균 전력을 구한다. Eq.(7)은 SM 수를 줄여 실험했을 때 총 전력에서 active SM의 전력과 P_const를 빼면 idle SM들의 전력만 남는다는 원리이다. Eq.(8)은 이를 idle SM 개수로 나누어 1개당 전력을 구한다.

최종 idle SM power는 여러 microbenchmark(n개)에서 구한 P_perIdleSM,i의 **기하평균**으로 결정한다. 기하평균을 사용하는 이유는 벤치마크 간 편차가 클 때 산술평균보다 안정적이기 때문이다.

### 5.6 전체 Static Power Model — Eq.(10)

```
P_total = P_dyn + P_static,yLanes,perActiveSM · k + P_perIdleSM · (80 − k) + P_const
```

이 수식은 AccelWattch의 **최종 전력 모델**이다. GPU의 총 전력을 4개 항으로 분리한다.

| 기호 | 의미 |
|------|------|
| **P_dyn** | 모든 active SM의 dynamic power 합계. 다음 섹션(Section 6)에서 상세히 다룸 |
| **P_static,yLanes,perActiveSM** | active SM 1개당 static power. y개 lane이 활성이고 특정 instruction mix 카테고리(cat1~cat6 등)에 따라 결정됨 |
| **k** | 현재 active(사용 중인) SM 수 |
| **80 − k** | idle(유휴) SM 수. 80은 V100(GV100)의 총 SM 수이며 GPU마다 다름 (A100은 108, H100은 132) |
| **P_perIdleSM** | idle SM 1개당 leakage 전력. Eq.(8)에서 도출 |
| **P_const** | 상수 전력. Eq.(3)에서 도출 |

즉, "dynamic power + active SM들의 static power + idle SM들의 leakage + 보드 상수 전력"으로 총 전력을 구성한다.

---

## 6. Dynamic Power Modeling

### 6.1 기본 개념 — Eq.(11)

GPU 내부에는 캐시, 연산 유닛, 레지스터 파일 등 N개의 마이크로아키텍처 컴포넌트가 있다. 각 컴포넌트는 접근될 때마다 에너지를 소모한다. 단위 시간당 소모 에너지(= 전력)는 다음과 같다:

```
         N
P_dyn = Σ  (aᵢ · Eᵢ / T_elapsed)
        i=1
```

이 수식의 의미는, 컴포넌트 i가 T_elapsed 시간 동안 aᵢ번 접근되었고, 한 번 접근당 Eᵢ 줄(Joule)의 에너지를 소모한다면, 해당 컴포넌트의 평균 전력은 aᵢ·Eᵢ/T_elapsed 와트(Watt)라는 것이다. 모든 컴포넌트의 전력을 합산하면 총 dynamic power가 된다.

| 기호 | 의미 | 단위 |
|------|------|------|
| **N** | 마이크로아키텍처 컴포넌트 총 수 (AccelWattch에서 22개) | 개 |
| **i** | 컴포넌트 인덱스 (1=IBP, 2=ICP, ..., 22=DRAMP+MCP) | - |
| **aᵢ** | 컴포넌트 i의 activity factor. 샘플링 구간 동안의 접근 횟수 | 회 |
| **Eᵢ** | 컴포넌트 i의 접근당 에너지. McPAT 등 분석 도구로 추정한 초기값이며 부정확할 수 있음 | J (줄) |
| **T_elapsed** | 샘플링 구간의 실행 시간 (= 사이클 수 / 클럭 주파수) | s (초) |
| **P_dyn** | 전체 dynamic power (모든 컴포넌트의 합) | W (와트) |

### 6.2 Scaling Factor 도입 — Eq.(12)

McPAT이 제공하는 초기 에너지 추정치 Eᵢ는 실제 하드웨어와 차이가 있다. 이 오차를 보정하기 위해 각 컴포넌트에 미지수 xᵢ (scaling factor, 보정 계수)를 도입한다. xᵢ = 1이면 McPAT 추정치가 정확한 것이고, xᵢ > 1이면 실제 에너지가 추정치보다 큰 것이다.

```
         N
P_est = Σ  (aᵢ · Êᵢ · xᵢ / T_elapsed)  +  P_static,yLanes,perActiveSM · k
        i=1
                                           +  P_perIdleSM · (80 − k)  +  P_const
```

| 기호 | 의미 |
|------|------|
| **P_est** | 모델이 추정한 전력 (예측값). 이것이 하드웨어 측정값 P_meas에 가까워지도록 xᵢ를 최적화함 |
| **Êᵢ** | 컴포넌트 i의 접근당 에너지 **초기 추정치** (McPAT 기반). Eᵢ와 구분하기 위해 hat(^) 표기 |
| **xᵢ** | 컴포넌트 i의 scaling factor (보정 계수). 0.1~1000 범위. Quadratic Programming으로 최적화 |
| **k** | 현재 active SM 수 |
| **80 − k** | idle SM 수 (V100 기준, GPU마다 다름) |

이 수식은 Eq.(10)과 같은 구조이되, P_dyn을 "보정된 컴포넌트별 전력의 합"으로 전개한 것이다.

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

M개의 microbenchmark(워크로드)에서 수집한 activity factor와 하드웨어 전력 측정값으로 연립방정식을 구성한다. 각 microbenchmark는 하나의 행(row)을 생성하고, 각 power component의 activity가 열(column)이 된다.

```
              M×(N+3)           (N+3)×1         M×1
          ┌            ┐   ┌          ┐   ┌          ┐
          │ a₁₁ ... a₁ₙ 1 1 1 │   │ x₁     │   │ P_meas,1 │
P_est  ·  │ a₂₁ ... a₂ₙ 1 1 1 │ × │ x₂     │ = │ P_meas,2 │
          │ ...         ... │   │ ...    │   │ ...      │
          │ aₘ₁ ... aₘₙ 1 1 1 │   │ xₙ     │   │ P_meas,M │
          └            ┘   │ x_idle │   └          ┘
                            │ x_const│
                            │ x_static│
                            └          ┘
```

| 기호 | 의미 |
|------|------|
| **M** | microbenchmark 수 (AccelWattch에서 102개) |
| **N** | dynamic power component 수 (22개) |
| **N+3** | 전체 파라미터 수 = 22 dynamic + IDLE_COREP + CONSTP + STATICP. 실제 코드에서는 **31열** (중간 합산/제거 후) |
| **P_est** | 추정 전력 행렬. 각 행은 하나의 microbenchmark, 각 열은 해당 component의 전력 기여 |
| **X** | scaling factor 벡터. 이것을 찾는 것이 최적화의 목표 |
| **P_meas** | 하드웨어에서 측정한 실제 전력 벡터 |
| **aᵢⱼ** | microbenchmark i에서 component j의 activity 기반 전력 추정값 |

즉, "각 microbenchmark의 component별 전력 추정 × scaling factor = 실측 전력"이 되도록 X를 찾는 것이다.

### 7.2 최적화 공식 — Eq.(14)

이 연립방정식은 일반적으로 overdetermined(행>열)이므로, 정확한 해 대신 **오차를 최소화하는 해**를 구한다. 이를 constrained quadratic programming(제약 조건부 2차 계획법)으로 풀며, 목적함수는 추정 전력과 실측 전력 간의 **잔차 제곱합(sum of squared residuals)**을 최소화하는 것이다.

```
X* = argmin ‖P_est · X − P_meas‖²
      X

이를 전개하면:  argmin { Xᵀ · Pᵀ_est · P_est · X  −  2 · (Pᵀ_est · P_meas)ᵀ · X }
```

| 기호 | 의미 |
|------|------|
| **X*** | 최적의 scaling factor 벡터. 잔차 제곱합을 최소화하는 해 |
| **argmin** | 뒤따르는 식을 최소화하는 X를 구하라는 연산자 |
| **‖·‖²** | L2 노름의 제곱 = 벡터 원소들의 제곱합. 즉 Σ(P_est,i − P_meas,i)² |
| **Pᵀ_est** | P_est의 전치 행렬 (행과 열을 바꾼 것) |

**제약 조건:**
```
범위 제약:     0.1 ≤ xᵢ ≤ 1000  (모든 i에 대해)
고정 파라미터:  x_IDLE_COREP = x_CONSTP = x_STATICP = 1
               (이 3개는 이미 별도 모델링되었으므로 보정하지 않음)

에너지 순서 제약 (McPAT의 per-instruction 에너지 비율에 기반):
  x_INT   ≤ 1.843 × x_FPU      (INT ALU ≤ 1.843 × FP ALU)
  x_FPU   ≤ x_DPU              (FP32 ≤ FP64)
  x_INT   ≤ 1.107 × x_INT_MUL  (INT add ≤ 1.107 × INT mul)
  x_FP_MUL ≤ 75.07 × x_TENSOR  (FP mul ≤ 75.07 × Tensor)
  ... (총 13개 active 부등식 제약)
```

이 제약들은 물리적 사실에 기반한다. 예를 들어, 정수 덧셈(INT)은 부동소수점 덧셈(FPU)보다 에너지가 작아야 하고, 텐서 코어 연산(TENSOR)은 단일 FP 곱셈(FP_MUL)보다 에너지가 훨씬 크다. 이러한 제약 없이 QP를 풀면 비물리적인 해가 나올 수 있다.

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
