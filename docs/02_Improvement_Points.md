# AccelWattch 개선 포인트 분석

> **목적**: AccelWattch를 A100/H100 등 최신 GPU에 적용하고 MAPE를 낮추기 위한 개선 포인트를 우선순위별로 분석  
> **기준 모델**: AccelWattch SASS SIM (Volta GV100, MAPE 9.2%)  
> **목표**: A100/H100에서 < 10% MAPE 달성  
> **작성일**: 2026-04-02  

---

## 목차

1. [아키텍처 변화 요약: V100 → A100 → H100](#1-아키텍처-변화-요약)
2. [개선 포인트 우선순위](#2-개선-포인트-우선순위)
3. [P1: Power Component 확장](#3-p1-power-component-확장)
4. [P2: DVFS/Constant Power 모델 재보정](#4-p2-dvfsconstant-power-모델-재보정)
5. [P3: Static Power Model 구조 변경](#5-p3-static-power-model-구조-변경)
6. [P4: Microbenchmark Suite 확장](#6-p4-microbenchmark-suite-확장)
7. [P5: Quadratic Programming 모델 개선](#7-p5-quadratic-programming-모델-개선)
8. [P6: GPU Config 파일 생성](#8-p6-gpu-config-파일-생성)
9. [P7: Technology Scaling 정교화](#9-p7-technology-scaling-정교화)
10. [P8: SASS Opcode 매핑 확장](#10-p8-sass-opcode-매핑-확장)
11. [P9: ML-Hybrid 모델링 도입](#11-p9-ml-hybrid-모델링-도입)
12. [추론 전용 가속기의 Component 최적화](#12-추론-전용-가속기의-component-최적화)
13. [예상 Equation 변화](#13-예상-equation-변화)
14. [최신 관련 연구 동향](#14-최신-관련-연구-동향)
15. [실행 로드맵](#15-실행-로드맵)

---

## 1. 아키텍처 변화 요약

### V100 → A100 → H100 핵심 변화

| 특성 | V100 (Volta) | A100 (Ampere) | H100 (Hopper) |
|------|-------------|---------------|---------------|
| **공정** | 12nm FinFET | 7nm FinFET | 4nm FinFET |
| **SM 수** | 80 | 108 | 132 |
| **Processing Blocks/SM** | 4 | 4 | 4 |
| **INT32 cores/SM** | 16 (전용) | 64 (FP32 공유) | 64 (일부 전용) |
| **FP32 cores/SM** | 16 (전용) | 64 (INT32 공유) | 128 |
| **FP64 cores/SM** | 8 (전용) | 32 | 64 |
| **Tensor Cores/SM** | 2 (1st gen) | 4 (3rd gen) | 4 (4th gen) |
| **Tensor 지원 타입** | FP16 | FP16, BF16, **TF32**, INT8, INT4 | FP16, BF16, TF32, **FP8**, INT8 |
| **메모리** | HBM2, 900GB/s | HBM2e, 2TB/s | HBM3, 3.35TB/s |
| **TDP** | 250W | 400W | 700W |
| **Sparsity** | 없음 | **2:4 Structured** | 2:4 Structured |
| **특수 유닛** | - | - | **Transformer Engine**, DPX |
| **MIG** | 없음 | **있음 (1st gen)** | 있음 (2nd gen) |
| **Thread Block Clusters** | 없음 | 없음 | **있음** |
| **Async Engine** | 기본 | **비동기 Copy** | **TMA** (Tensor Memory Accelerator) |

### Power Modeling에 미치는 영향

```
┌──────────────────────────────────────────────────────────────────┐
│                  AccelWattch 모델 영향 분석                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ [CRITICAL] INT32/FP32 공유 실행경로 (A100)                       │
│   → Half-warp static power model (Eq.5) 근본 변경 필요          │
│   → Volta: INT32와 FP32가 별도 → 동시 실행 = 2x lane            │
│   → Ampere: 동일 core에서 INT/FP 교대 또는 동시 → power 겹침    │
│                                                                  │
│ [CRITICAL] 새로운 연산 유닛                                       │
│   → TF32 (A100): 기존 FP_MUL_OP? 새 component?                 │
│   → FP8 (H100): 완전 새로운 component 필요                      │
│   → Transformer Engine (H100): dynamic precision switching      │
│   → DPX (H100): Dynamic Programming 가속                        │
│                                                                  │
│ [HIGH] 공정 노드 간 비선형 scaling                               │
│   → 12nm→7nm→4nm: IRDS 선형 scaling 부정확                     │
│   → Leakage 특성 크게 변화 (FinFET vs GAA)                      │
│                                                                  │
│ [HIGH] SM 수 대폭 증가                                           │
│   → 80→108→132: Idle SM power 모델 재보정 필요                  │
│   → P_const 비중 변화 (250W→700W)                               │
│                                                                  │
│ [MEDIUM] 메모리 시스템 변화                                       │
│   → HBM2→HBM2e→HBM3: 대역폭 4x 증가                           │
│   → DRAM power 모델 파라미터 변경                                │
│                                                                  │
│ [MEDIUM] 새로운 실행 계층                                         │
│   → Thread Block Clusters (H100): SM occupancy 모델 변경        │
│   → TMA: compute-data 분리로 activity factor 변화               │
│                                                                  │
│ [LOW] MIG 파티셔닝                                               │
│   → partial-GPU 설정 지원 필요                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 개선 포인트 우선순위

| 순위 | 개선 포인트 | 영향도 | 난이도 | MAPE 개선 예상 |
|------|-----------|--------|--------|---------------|
| **P1** | Power Component 확장 (TF32, FP8, TE 등) | **Critical** | 높음 | 5-10% |
| **P2** | DVFS/Constant Power 재보정 | **Critical** | 중간 | 3-5% |
| **P3** | Static Power Model 구조 변경 | **Critical** | 높음 | 3-7% |
| **P4** | Microbenchmark Suite 확장 | **High** | 높음 | 2-5% |
| **P5** | QP 모델 개선 (비선형/ML) | **High** | 중간 | 2-4% |
| **P6** | GPU Config 파일 생성 | **High** | 중간 | 전제 조건 |
| **P7** | Technology Scaling 정교화 | **Medium** | 낮음 | 1-3% |
| **P8** | SASS Opcode 매핑 확장 | **Medium** | 낮음 | 1-2% |
| **P9** | ML-Hybrid 모델링 도입 | **Medium** | 중간 | 2-5% |

---

## 3. P1: Power Component 확장

### 3.1 현재 상태

AccelWattch는 22개 dynamic power component를 추적한다 (Table 1). 이는 Volta 아키텍처에 최적화되어 있다.

### 3.2 필요한 신규 Component

#### A100 (Ampere) 추가 Component

| 신규 Component | 설명 | 근거 |
|---------------|------|------|
| **TF32_ACC** | TF32 Tensor Core 연산 | TF32는 FP32 범위 + FP16 정밀도, 기존 TENSOR_ACC와 에너지 특성 다름 |
| **SPARSE_ACC** | 2:4 Structured Sparsity | Sparsity 활성화 시 Tensor Core 처리량 2x, 전력 특성 변화 |
| **BF16_ACC** | BFloat16 연산 | FP16과 다른 exponent bit → 다른 에너지 소모 |
| **ASYNC_COPY** | 비동기 메모리 복사 | `cp.async` 명령어, 기존 LD/ST와 다른 경로 |
| **L2_PARTITION** | L2 Residency Control | A100의 L2 파티셔닝 → 캐시 power 변화 |

#### H100 (Hopper) 추가 Component

| 신규 Component | 설명 | 근거 |
|---------------|------|------|
| **FP8_ACC** | FP8 Tensor Core 연산 | E4M3/E5M2 포맷, 최저 에너지 연산 |
| **TE_ACC** | Transformer Engine | 동적 정밀도 전환 하드웨어 |
| **DPX_ACC** | Dynamic Programming 가속 | Smith-Waterman 등 DP 알고리즘 가속 |
| **TMA_ACC** | Tensor Memory Accelerator | 비동기 bulk 데이터 이동 |
| **TBC_MGMT** | Thread Block Cluster 관리 | 새로운 스케줄링 계층의 오버헤드 |

### 3.3 수정해야 할 파일

```
수정 대상:
1. gpu-simulator/ISA_Def/accelwattch_component_mapping.h
   → 새 component enum 추가 (TF32__OP, FP8__OP, DPX__OP 등)
   → 새 opcode→component 매핑 추가

2. gpu-simulator/ISA_Def/ampere_opcode.h
   → 새 opcode 추가 (HFMA2 with TF32, FP8 variants)

3. util/accelwattch/gen_sim_power_csv.py
   → power_counters 리스트에 새 component 추가

4. util/accelwattch/quadprog_solver.m
   → 행렬 차원 확장 (31 → 35+)
   → 새 제약조건 추가

5. .vendor/gpgpu-sim_distribution/configs/tested-cfgs/SM80_A100/
   → accelwattch_sass_sim.xml 신규 생성
```

### 3.4 예상 Equation 변화

기존 (31개 parameter):
```
P_est = Σ(i=1→22) [a_i · Ê_i · x_i / T] + P_static·k + P_idle·(N-k) + P_const
```

확장 (A100, ~36개 parameter):
```
P_est = Σ(i=1→27) [a_i · Ê_i · x_i / T] + P_static·k + P_idle·(108-k) + P_const
         ↑ TF32, SPARSE, BF16, ASYNC_COPY, L2_PART 추가
```

확장 (H100, ~40개 parameter):
```
P_est = Σ(i=1→32) [a_i · Ê_i · x_i / T] + P_static·k + P_idle·(132-k) + P_const
         ↑ FP8, TE, DPX, TMA, TBC 추가
```

---

## 4. P2: DVFS/Constant Power 모델 재보정

### 4.1 현재 모델의 문제점

AccelWattch의 Eq.(3) `P_total = βCf³ + τf + P_const`는 Volta에서 검증되었다.

| 항목 | V100 | A100 | H100 |
|------|------|------|------|
| P_const | 32.5W | **~50-60W** (추정) | **~80-120W** (추정) |
| TDP | 250W | 400W | 700W |
| P_const / TDP | 13% | ~15% | ~14-17% |
| Base Clock | 1245 MHz | 765 MHz | 1095 MHz |
| Boost Clock | 1380 MHz | 1410 MHz | 1755 MHz |
| V-F 관계 | 비교적 선형 | **더 복잡** | **가장 복잡** |

### 4.2 개선 방향

#### (a) A100의 DVFS 특성

A100 SXM4 80GB에서 조절 가능한 클럭은 **SM clock 하나뿐**이며, HBM2e memory clock은 1593 MHz로 **고정**이다 (`nvidia-smi -q -d SUPPORTED_CLOCKS`로 확인 가능). 따라서 Eq.(3)의 단일 변수 모델이 그대로 적용 가능하다:

```
A100:  SM clock  → DVFS 가능 (210~1410 MHz, 81단계)
       Mem clock → 고정 (1593 MHz), DVFS 불가
```

memory clock이 고정이므로, HBM2e의 기본 전력은 P_const에 흡수된다:

```
기존 Eq.(3):   P_total = βCf³ + τf + P_const
A100도 동일:   P_total = β'C'f_sm³ + τ'f_sm + P_const'
```

여기서 P_const'는 V100의 P_const(32.3W)보다 큰 값이다. HBM2e @ 1593MHz의 고정 전력이 포함되기 때문이다.

| 기호 | 의미 |
|------|------|
| **β' (베타 프라임)** | A100의 dynamic power 전압-주파수 결합 계수. V100의 β와는 다른 값 (7nm 공정) |
| **C'** | A100의 게이트 커패시턴스. 7nm에서 12nm보다 작음 |
| **f_sm** | SM 클럭 주파수. 210~1410 MHz 범위에서 DVFS 조절 가능 |
| **τ' (타우 프라임)** | A100의 static power 주파수 비례 계수 |
| **P_const'** | A100의 상수 전력 = 보드/팬 전력 + HBM2e 고정 전력. 약 50~65W 추정 |

#### (b) P_const' 재측정 방법

A100의 P_const'를 측정하려면 SM clock만 변경하는 단일 변수 실험으로 충분하다 (memory clock이 고정이므로):

1. `nvidia-smi --lock-gpu-clocks=210,210` ~ `1410,1410`으로 SM clock 변경 (8~10 단계)
2. 각 주파수에서 동일한 microbenchmark 실행, NVML로 전력 측정
3. (f_sm, P_total) 데이터를 `P = β'C'f_sm³ + τ'f_sm + P_const'`로 fitting
4. f_sm = 0 외삽 → y절편 = P_const'

#### (c) 비선형 V-F 관계 고려

7nm 공정에서 `V ≈ kf` (전압이 주파수에 선형 비례한다는 가정)이 12nm보다 덜 정확할 수 있다. 이 경우 Eq.(3)의 f³ 지수가 정확히 3이 아닐 수 있으며, fitting 시 지수를 자유 파라미터로 둘 수 있다:

```
일반화:  P_total = α · f_sm^n + τ · f_sm + P_const'
         여기서 n ∈ [2.5, 3.5] (fitting으로 결정)
```

| 기호 | 의미 |
|------|------|
| **α (알파)** | dynamic power 계수. β'C'를 하나로 합친 것 |
| **n** | 주파수 지수. V≈kf가 정확하면 n=3, 비선형이면 2.5~3.5 |

다만, 우선 n=3으로 fitting을 시도한 뒤 잔차(residual)를 확인하여 지수 조정 여부를 판단하는 것이 합리적이다.

---

## 5. P3: Static Power Model 구조 변경

### 5.1 핵심 변경 사유: INT32/FP32 공유 경로

**Volta (현재 모델):**
```
SM 내 Processing Block:
├── 16 INT32 cores (전용 datapath)  ← 별도 power gating
├── 16 FP32 cores (전용 datapath)  ← 별도 power gating
├── 8 FP64 cores
├── 2 Tensor Cores
└── 1 SFU

→ INT와 FP가 동시 실행 가능
→ Half-warp model: 각 유닛 독립적 on/off
```

**Ampere (변경 필요):**
```
SM 내 Processing Block:
├── 16 FP32/INT32 cores (공유!)     ← 하나의 power gating 단위
├── 16 FP32 cores (전용)            ← FP 전용
├── 8 FP64 cores
├── 1 Tensor Core (3rd gen)
└── 1 SFU

→ INT 실행 시 FP32 core의 절반이 INT로 전환
→ 동시 실행 = 다른 power gating 패턴
```

### 5.2 Half-warp Model 수정

현재 Eq.(5)에서 `y`는 "active lanes" 수만 고려한다. Ampere에서는 **어떤 유닛이 active인지**도 함께 고려해야 한다:

```
# 기존 (Volta)
P_static,yLanes = f(y)    # y = active thread count만

# 개선 (Ampere)  
P_static,yLanes = f(y, mode)
  where mode ∈ {FP32_only, INT32_only, FP32+INT32_concurrent}
```

### 5.3 Instruction Mix 카테고리 확장

현재 9개 → 확장 필요:

| 기존 | 추가 필요 (A100) | 추가 필요 (H100) |
|------|-----------------|-----------------|
| INT only | INT+FP concurrent | FP8 only |
| INT+FP | TF32 | FP8+FP16 mixed |
| INT+FP+DP | BF16 | Transformer Engine |
| INT+FP+SFU | Sparse Tensor | DPX |
| INT+FP+TEX | | TMA active |
| INT+FP+TENSOR | | |
| LIGHT | | |

### 5.4 SM 수 증가에 따른 Idle SM 모델

```
# 기존
P_total = P_dyn + P_static·k + P_idle·(80 - k) + P_const

# A100
P_total = P_dyn + P_static·k + P_idle·(108 - k) + P_const

# H100
P_total = P_dyn + P_static·k + P_idle·(132 - k) + P_const
```

→ Idle SM이 더 많아질 수 있으므로 `P_perIdleSM` 재측정 필수.

---

## 6. P4: Microbenchmark Suite 확장

### 6.1 신규 Microbenchmark 필요

| 대상 유닛 | Benchmark 설명 | 수 |
|----------|---------------|-----|
| TF32 Tensor Core | TF32 행렬 곱셈 (다양한 크기) | 4-6 |
| FP8 Tensor Core | FP8 E4M3/E5M2 행렬 곱셈 | 4-6 |
| BF16 연산 | BF16 벡터/행렬 연산 | 3-4 |
| 2:4 Sparsity | Sparse vs Dense Tensor Core | 4 |
| Transformer Engine | 동적 정밀도 전환 벤치 | 2-3 |
| DPX 명령어 | Smith-Waterman, Viterbi 등 | 2-3 |
| TMA | 비동기 bulk copy 벤치 | 3-4 |
| L2 Residency | L2 파티셔닝 벤치 | 2-3 |
| Thread Block Cluster | 다양한 cluster 크기 | 3-4 |
| Concurrent INT/FP | INT+FP 동시 실행 비율 변경 | 4-6 |
| HBM3 | 대역폭 스트레스 테스트 | 2-3 |

**총 추가**: ~35-50개 → 전체 ~140-150개

### 6.2 기존 Microbenchmark 수정

1. **Active/Idle SM 벤치**: SM 수 변경 (80→108/132)
2. **INT32 벤치**: 공유 경로 반영
3. **온도 제어**: 700W GPU에서 65°C 유지 전략 변경 필요

---

## 7. P5: Quadratic Programming 모델 개선

### 7.1 현재 방식의 한계

1. **선형 가정**: `P_dyn = Σ a_i · E_i · x_i` → 컴포넌트 간 **상호작용 무시**
2. **수렴 불안정**: 반복 최적화 수렴 보장 없음
3. **전역 최적해 미보장**: QP는 convex이지만 제약 조건이 복잡할 때 local minima 가능

### 7.2 개선 방안

#### (a) 교차항(Interaction Terms) 추가

```
# 기존
P_dyn = Σ a_i · x_i

# 개선: 2차 교차항 추가
P_dyn = Σ a_i · x_i + Σ_j>i a_i · a_j · x_ij
```

특히 중요한 교차항:
- INT_ACC × FP_ACC (concurrent execution on Ampere)
- TENSOR_ACC × MEM_RD (memory-bound vs compute-bound Tensor ops)
- L2_RH × NOC_A (cache hit이 NoC 트래픽 감소)

#### (b) Piece-wise Linear 또는 Non-linear Scaling

```
# 기존: 단일 scaling factor
E_i = Ê_i · x_i

# 개선: piece-wise linear
E_i = { Ê_i · x_i1,    if a_i < threshold
      { Ê_i · x_i2,    if a_i ≥ threshold
```

→ 저활동 vs 고활동 시 에너지 효율이 다름 (예: 캐시 유휴 leakage vs 풀스로틀)

#### (c) Regularization 추가

```
# 기존 Eq.(14)
X* = argmin ||P_est · X - P_meas||²

# 개선: L2 정규화
X* = argmin ||P_est · X - P_meas||² + λ||X||²

# 또는 L1 정규화 (sparse solution)
X* = argmin ||P_est · X - P_meas||² + λ||X||₁
```

→ Overfitting 방지, 불필요한 component 자동 제거

#### (d) MATLAB → Python 전환 (cvxpy)

```python
import cvxpy as cp
import numpy as np

# 데이터 로드
A = np.loadtxt('activity_factors.csv', delimiter=',')
b = np.loadtxt('measured_power.csv', delimiter=',')

# 변수
x = cp.Variable(n_components, pos=True)

# 목적함수
objective = cp.Minimize(cp.sum_squares(A @ x - b) + lambda_reg * cp.norm(x, 2))

# 제약조건
constraints = [
    x >= 0.001,
    x <= 1000,
    x[idx_static] == 1,
    x[idx_idle] == 1,
    x[idx_const] == 1,
    x[idx_int] <= 1.843 * x[idx_fpu],
    x[idx_fpmul] <= 75.07 * x[idx_tensor],
    # ... 추가 제약조건
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP)
```

### 7.3 MATLAB 대체 이점

1. **재현성**: Python 환경이 더 널리 사용
2. **확장성**: scipy, cvxpy, sklearn 등 풍부한 라이브러리
3. **ML 통합**: PyTorch/TensorFlow로 neural network hybrid 가능
4. **자동화**: end-to-end 파이프라인 Python으로 통일

---

## 8. P6: GPU Config 파일 생성

### 8.1 현재 상태

```
SM80_A100/
└── gpgpusim.config     ✓ 존재
└── trace.config        ✓ 존재 (상위 폴더)
└── accelwattch_*.xml   ✗ 없음  ← 생성 필요
```

### 8.2 A100 XML 생성 시 필요한 파라미터

```xml
<!-- A100 AccelWattch XML 주요 변경 사항 -->

<!-- 1. Dynamic Power Activity Factors -->
<param name="TOT_INST" value="??"/>    <!-- 재측정 필요 -->
<param name="TF32_ACC" value="??"/>    <!-- 신규 -->
<param name="SPARSE_ACC" value="??"/>  <!-- 신규 -->
<param name="BF16_ACC" value="??"/>    <!-- 신규 -->
<param name="ASYNC_COPY_ACC" value="??"/> <!-- 신규 -->

<!-- 2. Static & Constant Power -->
<param name="constant_power" value="??"/>      <!-- 재측정: ~50-60W 예상 -->
<param name="idle_core_power" value="??"/>     <!-- 재측정: SM 108개 기준 -->

<!-- 3. 새 instruction mix 카테고리 -->
<param name="static_cat_concurrent_intfp_flane" value="??"/>  <!-- 신규 -->
<param name="static_cat_tf32_flane" value="??"/>               <!-- 신규 -->
<param name="static_cat_bf16_flane" value="??"/>               <!-- 신규 -->

<!-- 4. Architecture 파라미터 업데이트 -->
<param name="core_tech_node" value="7"/>       <!-- 12→7 nm -->
<param name="ALU_per_core" value="64"/>        <!-- 32→64 -->
<param name="FPU_per_core" value="64"/>        <!-- 32→64 -->
```

### 8.3 자동화 스크립트

`scripts/build_gpu_config_workbook.py`를 활용하여:
1. 기존 V100 config을 템플릿으로 사용
2. A100 아키텍처 파라미터로 치환
3. Microbenchmark 결과로 activity factor 채우기
4. QP solver로 scaling factor 최적화

---

## 9. P7: Technology Scaling 정교화

### 9.1 현재 방식

AccelWattch는 IRDS(International Roadmap for Devices and Systems) 데이터 기반으로 단순 비례 scaling:

```
P_target = P_volta × scaling_factor(tech_source, tech_target)
```

논문에서 12nm→16nm (Volta→Pascal) 적용 시 MAPE 1.22% 감소.

### 9.2 문제점

| 전환 | 난이도 | 이유 |
|------|--------|------|
| 12nm → 16nm | 쉬움 | 같은 FinFET, 비슷한 특성 |
| 12nm → 7nm | **중간** | 다른 FinFET 세대, EUV 일부 도입 |
| 12nm → 4nm | **어려움** | 완전 다른 트랜지스터 특성, leakage 패턴 변화 |

### 9.3 개선 방안

#### (a) 컴포넌트별 차별적 Scaling

```
# 기존: 모든 component에 동일 scaling
P_component_new = P_component_old × k_tech

# 개선: component 유형별 다른 scaling
P_logic_new = P_logic_old × k_logic(7nm/12nm)
P_sram_new = P_sram_old × k_sram(7nm/12nm)    # SRAM은 scaling 속도 다름
P_io_new = P_io_old × k_io(7nm/12nm)           # I/O는 거의 scaling 안됨
P_analog_new = P_analog_old × k_analog(7nm/12nm)
```

#### (b) 실측 기반 Scaling Factor

IRDS 이론치 대신 실측 데이터 활용:
- 같은 워크로드를 V100, A100, H100에서 실행
- 전력 비를 직접 측정하여 scaling factor 도출

---

## 10. P8: SASS Opcode 매핑 확장

### 10.1 현재 ampere_opcode.h 분석

현재 지원되는 Ampere 고유 명령어:

```cpp
// ampere_opcode.h에 정의된 것
OP_HMNMX2, OP_DMMA, OP_I2FP, OP_F2IP,
OP_LDGDEPBAR, OP_LDGSTS, OP_REDUX,
OP_UF2FP, OP_SUQUERY
```

### 10.2 추가 필요한 Opcode

```cpp
// A100 (SM80) 추가 필요
OP_HMMA_TF32,      // TF32 Tensor Core operation
OP_HMMA_BF16,      // BF16 Tensor Core operation
OP_HMMA_INT8,      // INT8 Tensor Core operation
OP_HMMA_INT4,      // INT4 Tensor Core operation
OP_LDSM_16,        // Async shared memory load (16B)
OP_CP_ASYNC,       // Asynchronous copy
OP_CP_ASYNC_BULK,  // Bulk async copy

// H100 (SM90) 추가 필요
OP_HMMA_FP8_E4M3,  // FP8 E4M3 Tensor Core
OP_HMMA_FP8_E5M2,  // FP8 E5M2 Tensor Core
OP_WGMMA,          // Warp Group MMA (Hopper)
OP_SETMAXNREG,     // Dynamic register allocation
OP_FENCE,          // Memory fence variants
OP_TMA_LOAD,       // Tensor Memory Accelerator load
OP_TMA_STORE,      // Tensor Memory Accelerator store
OP_DPX_*,          // DPX instructions
```

### 10.3 매핑 변경

```cpp
// accelwattch_component_mapping.h 에 추가
{OP_HMMA_TF32,     TF32__OP},    // 새 component
{OP_HMMA_BF16,     BF16__OP},    // 새 component
{OP_HMMA_FP8_E4M3, FP8__OP},    // 새 component
{OP_CP_ASYNC,      ASYNC_OP},    // 새 component
{OP_WGMMA,         TENSOR__OP},  // 또는 새 WGMMA__OP
{OP_TMA_LOAD,      TMA__OP},     // 새 component
```

---

## 11. P9: ML-Hybrid 모델링 도입

### 11.1 개념

AccelWattch의 물리 기반 분석 모델에 ML을 결합:

```
┌──────────────────────────────────────────────────┐
│             Hybrid Power Model                    │
├──────────────────────────────────────────────────┤
│                                                   │
│  P_total = P_analytical + P_residual_ml           │
│                                                   │
│  P_analytical = AccelWattch 기존 모델             │
│    (constant + static + dynamic)                  │
│                                                   │
│  P_residual_ml = f_nn(features)                   │
│    → 분석 모델이 포착하지 못한 비선형 효과 보정   │
│    features: activity factors, SM occupancy,      │
│              instruction mix ratio, cache miss     │
│              rate, memory bandwidth utilization    │
│                                                   │
│  장점:                                            │
│  - 물리적 해석 가능성 유지 (P_analytical)         │
│  - 비선형 효과 포착 (P_residual_ml)              │
│  - 적은 학습 데이터로도 효과적 (잔차만 학습)     │
└──────────────────────────────────────────────────┘
```

### 11.2 구현 방향

```python
# 1단계: AccelWattch 분석 모델로 기본 전력 추정
P_analytical = accelwattch_model(activity_factors, config)

# 2단계: 잔차 (오차) 학습
residuals = P_measured - P_analytical

# 3단계: 잔차 예측 모델 훈련
from sklearn.ensemble import GradientBoostingRegressor
# 또는 PyTorch neural network

features = [activity_factors, occupancy, instruction_mix, ...]
model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
model.fit(features_train, residuals_train)

# 4단계: Hybrid 예측
P_hybrid = P_analytical + model.predict(features_test)
```

### 11.3 기대 효과

- AccelWattch SASS SIM: 9.2% MAPE (현재)
- AccelWattch + ML Residual: **5-7% MAPE** (예상)
- 특히 **DeepBench 류의 복잡한 워크로드**에서 큰 개선 기대 (현재 12.79%)

---

## 12. 추론 전용 가속기의 Component 최적화

학습(Training)이 아닌 추론(Inference) 전용 가속기를 설계할 경우, 필요하지 않은 하드웨어 유닛을 제거하여 전력과 면적을 크게 절감할 수 있다. AccelWattch의 22개 dynamic power component 중 상당수가 불필요해지므로, power model도 이에 맞게 단순화된다.

### 12.1 추론 전용 가속기에서 제거 가능한 Component

추론 워크로드의 특성: FP32/FP16/INT8 위주 연산, 배치 크기 고정, gradient 계산 없음, 가중치(weight)는 읽기 전용.

| Component | Training 시 역할 | 추론에서 필요? | 제거 시 영향 | 근거 |
|-----------|----------------|--------------|------------|------|
| **DPUP** (FP64) | Gradient 누적, 수치 안정성 | **제거 가능** | Dynamic power 2~5% 절감 | 추론은 FP32 이하 정밀도면 충분. FP64는 과학 계산/학습 전용 |
| **DP_MULP** (FP64 곱셈) | FP64 FMA 연산 | **제거 가능** | 위와 동일 | FP64 유닛 전체 불필요 |
| **FP_DIVP** (FP 나눗셈) | Softmax, LayerNorm의 나눗셈 | **축소 가능** | 소폭 절감 | 추론에서 나눗셈은 역수 곱셈(reciprocal multiply)으로 대체 가능. SFU 1개면 충분 |
| **FP_SQRTP** (제곱근) | BatchNorm, LayerNorm | **축소 가능** | 소폭 절감 | 추론에서 sqrt는 pre-computed 가능 |
| **FP_LGP** (로그) | Softmax log, loss 계산 | **제거 가능** | SFU power 절감 | loss 계산은 학습 전용. 추론은 argmax만 필요 |
| **FP_EXP** (지수) | Softmax exp | **축소 가능** | 소폭 절감 | Softmax는 lookup table 또는 근사로 대체 가능 |
| **FP_SINP** (삼각함수) | Position encoding 등 | **제거 가능** | 소폭 절감 | Pre-computed table로 대체 |
| **FPUP** (FP32 add/cmp) | 범용 FP32 연산 | **축소 가능** | Dynamic power 절감 | INT8/INT4 추론에서는 FP32 코어 수를 줄일 수 있음 |
| **SHRDP** (Shared Memory) | Tiling, 중간 결과 공유 | **축소 가능** | Cache power 절감 | 추론은 고정 패턴이므로 더 작은 shared memory로 충분 |
| **RFP** (Register File) | 대용량 레지스터 (학습 시 중간값 다수) | **축소 가능** | 상당한 절감 (V100에서 전체의 ~11%) | 추론은 학습보다 레지스터 사용이 적음. 65536→32768개로 절반 가능 |

### 12.2 추론 전용에서 유지/강화해야 하는 Component

| Component | 추론에서의 역할 | 방향 | 근거 |
|-----------|--------------|------|------|
| **TENSORP** (Tensor Core) | INT8/INT4 행렬 곱셈 | **강화** | 추론의 핵심 연산. INT8 Tensor Core 수 증가 또는 전용 INT8 MAC 유닛 추가 |
| **INTP** (INT32 ALU) | 주소 계산, 인덱싱 | **유지** | 메모리 주소 계산에 항상 필요 |
| **INT_MULP** (INT 곱셈) | Quantization scaling | **유지** | INT8 양자화 시 scale factor 곱셈 |
| **DCP** (L1D Cache) | Weight/activation 읽기 | **유지/강화** | 추론은 weight가 읽기 전용 → 캐시 적중률 높음. 크기 유지 또는 확대 |
| **L2CP** (L2 Cache) | 모델 파라미터 캐싱 | **강화** | 추론 모델 전체가 L2에 들어가면 DRAM 접근 대폭 감소 → 전력 절감 |
| **DRAMP** (DRAM) | 입력 데이터, 대형 모델 | **유지** | 대역폭은 유지하되 DRAM 용량 축소 가능 (학습 시 gradient 저장 불필요) |
| **IBP, ICP** (명령어 버퍼/캐시) | 명령어 fetch | **유지** | 아키텍처 기본 구성 |
| **SCHEDP, PIPEP** (스케줄러, 파이프라인) | 실행 제어 | **유지** | 아키텍처 기본 구성 |

### 12.3 Component 수 비교

```
┌──────────────────────────────────────────────────────────────┐
│          Power Component 수 비교                             │
├────────────────┬─────────┬─────────┬─────────────────────────┤
│                │ V100    │ A100    │ 추론 전용 가속기         │
│                │ (범용)  │ (범용)  │ (Accel-B)               │
├────────────────┼─────────┼─────────┼─────────────────────────┤
│ FP64 관련      │ 3개     │ 3개     │ 0개 (전부 제거)         │
│ SFU 관련       │ 4개     │ 4개     │ 1개 (EXP만 축소 유지)   │
│ FP32 관련      │ 2개     │ 2개     │ 1개 (축소)              │
│ INT 관련       │ 4개     │ 4개     │ 2개 (INT + INT_MUL)     │
│ Tensor/INT8    │ 1개     │ 3개     │ 4개 (INT8/INT4/FP16/BF16)│
│ 캐시/메모리    │ 7개     │ 7개     │ 6개 (Shared 축소)       │
│ 인프라         │ 3개     │ 3개     │ 3개 (유지)              │
├────────────────┼─────────┼─────────┼─────────────────────────┤
│ Dynamic 합계   │ 22개    │ 27개    │ 15~17개                 │
│ Non-dynamic    │ 3개     │ 3개     │ 3개 (유지)              │
└────────────────┴─────────┴─────────┴─────────────────────────┘
```

### 12.4 추론 전용 가속기의 전력 구성 변화 (예상)

```
V100 sgemm (범용, 148W):              추론 전용 Accel-B (INT8 GEMM, ~80W):
┌──────────────────────┐              ┌──────────────────────┐
│ ■■■■ RFP     15.8W   │              │ ■■ RFP        6W     │ ← 절반
│ ■■■■ FP_MUL  12.4W   │              │                      │ ← FP_MUL 거의 없음
│ ■■■ DCP       8.5W   │              │ ■■ DCP        7W     │
│ ■■ PIPEP      6.2W   │              │ ■■ PIPEP      5W     │
│ ■■ L2CP       5.2W   │              │ ■■■ L2CP      8W     │ ← 강화
│ ■■ INTP       5.1W   │              │ ■ INTP        3W     │ ← 축소
│ ■■ SCHEDP     4.5W   │              │ ■ SCHEDP      3W     │
│ ■ SHRDP       4.2W   │              │ ■ SHRDP       2W     │ ← 축소
│ ■ NOCP        3.8W   │              │ ■ NOCP        3W     │
│ ■ IBP         3.2W   │              │ ■ IBP         2W     │
│ ■ ICP         2.8W   │              │ ■ ICP         2W     │
│ ■ DRAMP       2.4W   │              │ ■ DRAMP       3W     │
│ ■ FPUP        1.8W   │              │ ■■■■ INT8_TENSOR 12W │ ← 핵심!
│   DPUP        0.0W   │              │   DPUP        0W     │ ← 제거
│   TENSOR      0.0W   │              │   FP64        0W     │ ← 제거
│   (기타)      1.6W   │              │   (기타)      1W     │
├──────────────────────┤              ├──────────────────────┤
│ ★ P_static   38.6W   │              │ ★ P_static   10W     │ ← 유닛 감소로 대폭↓
│ ★ P_const    32.3W   │              │ ★ P_const    15W     │ ← 보드 단순화
├──────────────────────┤              ├──────────────────────┤
│ Total       148.3W   │              │ Total        ~82W    │ ← 45% 절감!
└──────────────────────┘              └──────────────────────┘
```

### 12.5 추론 전용 가속기의 Power Model 단순화

**QP solver 차원 축소:**

```
범용 GPU:   A(102 × 31) × X(31 × 1) = b(102 × 1)   → 31차원 최적화
추론 전용:  A(80 × 20)  × X(20 × 1) = b(80 × 1)    → 20차원 최적화
```

차원이 줄어들면:
- QP solver의 수렴이 빨라짐 (변수 감소)
- 더 적은 microbenchmark로도 충분 (제거된 유닛의 벤치 불필요)
- Overfitting 위험 감소 (파라미터/데이터 비율 개선)
- 제약조건도 단순화 (FP64 관련 제약 전부 제거)

**Static Power 카테고리 축소:**

```
범용 GPU: 9개 카테고리 (INT, INT+FP, INT+FP+DP, INT+FP+SFU, ...)
추론 전용: 4~5개 카테고리로 충분
  - INT only (주소 계산)
  - INT + INT8_TENSOR (핵심 추론 연산)
  - INT + FP16 (Mixed precision 추론)
  - INT + FP32 (후처리)
  - LIGHT (idle)
```

### 12.6 추론 전용 가속기 설계 시 주의사항

1. **Quantization 정확도 검증**: INT8/INT4로 정밀도를 낮추면 전력은 줄지만 추론 정확도가 떨어질 수 있다. Power model과 별도로 accuracy-power trade-off 분석이 필요하다.

2. **Batch size 고정 가정**: 추론은 보통 batch size가 작고 고정적이므로, SM occupancy가 학습보다 낮을 수 있다. Idle SM power(P_perIdleSM)의 비중이 상대적으로 커진다.

3. **Memory bandwidth vs Compute**: 추론은 학습보다 **memory-bound**인 경우가 많다 (weight 읽기 중심). 따라서 DRAMP/L2CP의 전력 비중이 상대적으로 높아지고, 연산 유닛 전력 비중은 낮아진다.

4. **Latency 제약**: 추론은 throughput뿐 아니라 latency도 중요하다. SM 수를 무한정 줄이면 전력은 줄지만 latency가 늘어나는 trade-off가 있다.

---

## 13. 예상 Equation 변화

### 12.1 현재 AccelWattch 전체 모델

```
P_total = Σ(i=1→22) [a_i · Ê_i · x_i / T]           ... Dynamic (22 components)
        + P_static,yLanes,perActiveSM · k              ... Static (9 categories)
        + P_perIdleSM · (80 - k)                       ... Idle SM
        + P_const                                       ... Constant (32.5W)

where:
  P_total = βCf³ + τf + P_const                        ... DVFS model
  P_static uses half-warp or linear model per category
```

### 12.2 제안: 확장 모델 (A100)

```
P_total = Σ(i=1→27) [a_i · Ê_i · x_i / T]           ... Dynamic (27 components)
        + Σ(j>i) [a_i·a_j · x_ij / T]                 ... 교차항 (핵심 쌍만)
        + P_static,yLanes,mode,perActiveSM · k         ... Static (mode 추가)
        + P_perIdleSM · (108 - k)                      ... Idle SM (108 SMs)
        + P_const                                       ... Constant (~55W)
        + f_ml(features)                                ... ML 잔차 보정

where:
  P_total = β_sm·C_sm·f_sm³ + β_mem·C_mem·f_mem³      ... Multi-domain DVFS
          + τ_sm·f_sm + τ_mem·f_mem + P_const
  
  P_static에서 mode ∈ {FP_only, INT_only, concurrent_FP_INT, ...}
```

### 12.3 제안: 확장 모델 (H100)

```
P_total = Σ(i=1→32) [a_i · Ê_i · x_i / T]           ... Dynamic (32 components)
        + Σ(j>i) [a_i·a_j · x_ij / T]                 ... 교차항
        + P_static,yLanes,mode,perActiveSM · k         ... Static
        + P_static_cluster(cluster_config)              ... Thread Block Cluster
        + P_perIdleSM · (132 - k)                      ... Idle SM (132 SMs)
        + P_const                                       ... Constant (~100W)
        + P_TE(precision_mode)                          ... Transformer Engine
        + f_ml(features)                                ... ML 잔차 보정

where:
  P_total = Σ_d [α_d · f_d^n_d] + P_const             ... Multi-domain, 가변 지수
  P_TE depends on dynamic FP8↔FP16 switching overhead
```

---

## 14. 최신 관련 연구 동향

### 13.1 AccelWattch 이후 주요 연구 방향

| 연구 방향 | 대표 접근 | AccelWattch 대비 장단점 |
|----------|----------|----------------------|
| **ML 기반 전력 예측** | HW counter → Neural Network | (+) 비선형 포착 (-) 해석불가, 아키텍처 전이 불가 |
| **분석적 Multi-domain** | V-F 도메인별 분리 모델 | (+) DVFS 정확도 (-) 컴포넌트 수준 분해 부족 |
| **Ampere Microbenchmarking** | Jia et al. 후속 연구 | (+) A100 아키텍처 파라미터 제공 (-) 전력 모델 아님 |
| **GNN 기반 커널 전력** | 그래프로 커널 구조 인코딩 | (+) 커널 특성 반영 (-) 학습 데이터 많이 필요 |
| **Transformer 기반 예측** | 시계열 전력 패턴 학습 | (+) 시간적 패턴 (-) cycle-level 비용 높음 |

### 13.2 활용 가능한 외부 리소스

1. **Accel-Sim SM80/SM90 지원**: 성능 시뮬레이션 기반 확보 중
2. **NVIDIA Nsight Compute**: A100/H100 하드웨어 카운터 수집 가능
3. **NVML API 업데이트**: 최신 GPU 전력/온도 모니터링 지원
4. **MLPerf Power**: ML 워크로드 전력 측정 표준화

---

## 15. 실행 로드맵

### Phase 1: 기반 구축 (1-2개월)

```
□ A100 하드웨어 접근 확보 (또는 클라우드 인스턴스)
□ A100 SASS trace 수집 환경 구축 (NVBit + CUDA 12.x)
□ 기존 validation suite를 A100에서 실행하여 baseline 전력 측정
□ A100 DVFS 실험 → P_const 측정
□ A100 gpgpusim.config 검증 및 보완
```

### Phase 2: 모델 확장 (2-3개월)

```
□ 새 microbenchmark 설계 및 구현 (TF32, Sparsity, BF16, Concurrent INT/FP)
□ Opcode 매핑 업데이트 (SM80 SASS)
□ Power component 확장 (22→27개)
□ Static power model 수정 (concurrent execution mode)
□ quadprog_solver → Python(cvxpy) 전환
□ A100 accelwattch_sass_sim.xml 생성
```

### Phase 3: 최적화 및 검증 (1-2개월)

```
□ QP 반복 최적화 실행 (convergence 확인)
□ Validation suite 실행 및 MAPE 측정
□ 교차항 추가 실험
□ ML residual 모델 학습 및 평가
□ Technology scaling factor 재보정
```

### Phase 4: H100 확장 (2-3개월)

```
□ H100 접근 확보
□ SM90 opcode/component 추가
□ Transformer Engine, FP8, TMA 벤치마크
□ Thread Block Cluster 모델링
□ Multi-domain DVFS 모델 구현
□ 전체 검증 및 논문 작성
```

---

> **다음 문서**: [03_A100_Config_Analysis.md](03_A100_Config_Analysis.md) — A100 Configuration 상세 분석  
> **이전 문서**: [01_AccelWattch_Whitepaper.md](01_AccelWattch_Whitepaper.md) — AccelWattch 전체 백서
