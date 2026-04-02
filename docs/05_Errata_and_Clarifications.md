# 문서 정오표 및 보충 설명

> **목적**: 01~04 문서에서 발견된 오류, 부정확한 표현, 애매한 서술을 수정하고 보충  
> **검증 방법**: 논문 원문, 소스코드(`gpgpu_sim_wrapper.cc`, `quadprog_solver.m`, `gen_sim_power_csv.py`), `gpgpusim.config`, XML config 파일과 교차 대조  
> **작성일**: 2026-04-02  

---

## 문서별 정오표

---

### 01_AccelWattch_Whitepaper.md

#### E1. Dynamic Power Component 수가 "22개"라는 표현은 부정확

**문제**: 본문에서 "22개 하드웨어 컴포넌트의 dynamic power를 추적한다"고 했으나, 실제 코드의 `pwr_cmp_label[]`에는 **33개** 항목이 있다.

```cpp
// gpgpu_sim_wrapper.cc:36-43 실제 코드
static const char* pwr_cmp_label[] = {
    "IBP,", "ICP,", "DCP,", "TCP,", "CCP,",        // 5
    "SHRDP,", "RFP,", "INTP,", "FPUP,", "DPUP,",   // 5
    "INT_MUL24P,", "INT_MUL32P,", "INT_MULP,",      // 3
    "INT_DIVP,", "FP_MULP,",                         // 2
    "FP_DIVP,", "FP_SQRTP,", "FP_LGP,",             // 3
    "FP_SINP,", "FP_EXP,",                           // 2
    "DP_MULP,", "DP_DIVP,",                          // 2
    "TENSORP,", "TEXP,", "SCHEDP,",                  // 3
    "L2CP,", "MCP,", "NOCP,", "DRAMP,", "PIPEP,",   // 5
    "IDLE_COREP,", "CONSTP", "STATICP"               // 3
};                                                    // 총 33개
```

**정확한 정리**: 
- 코드에는 **33개** power component label이 있다
- 이 중 CONSTP, STATICP, IDLE_COREP = 3개는 dynamic이 아님
- 나머지 30개가 dynamic power component이다
- 논문의 Table 1에는 22개 하드웨어 유닛이 나열되어 있으나, 이는 SASS SIM variant 기준에서 일부를 결합/제거한 것이다
- `gen_sim_power_csv.py`에서 SASS SIM 모드는 `MCP, TCP, INT_MUL24P, INT_MUL32P, INT_DIVP, FP_DIVP, DP_DIVP, NOCP` **8개를 제거**하여, 실제 QP solver 입력은 `30 - 8 = 22`개의 dynamic component + IDLE_COREP + CONSTP + STATICP = **25개 열**이 된다
- 그러나 quadprog_solver.m은 `A = input(:,1:31)`로 **31열**을 읽으므로, CSV에 31개 열 + 1개 measured power = 32열 구조임

**수정 권장**: "22개 dynamic power component (Table 1)"이라는 표현을 유지하되, 다음 주석 추가: "코드 내부적으로는 33개 label이 정의되어 있으며, SASS SIM 모드에서는 8개를 결합/제거하여 22개로 축소한다. QP solver의 입력은 IDLE_COREP, CONSTP, STATICP를 포함한 총 25개 파라미터이다."

#### E2. QP solver 하한(lower bound)이 0.001이 아니라 0.1

**문제**: 01 문서 Section 7.2에서 제약조건을 `0.001 ≤ X_i ≤ 1000`이라고 썼으나, 실제 코드는:

```matlab
l = 0.1*ones(1,31);  % lower bounds (0.1, not 0.001)
u = 1000*ones(1,31); % upper bounds
```

**수정**: `0.001` → `0.1`

#### E3. 제약조건 설명에서 "X_ALU ≤ X_FPU ≤ X_DPU ≤ X_iMAD" 표현이 부정확

**문제**: 제약조건의 방향과 의미가 뒤바뀌어 서술됨. 실제 코드를 보면:

```matlab
C(1,8)=1; C(1,9)=-1.843;  % X[8] - 1.843*X[9] ≤ 0, 즉 X_INT ≤ 1.843*X_FPU
C(2,9)=1; C(2,10)=-0.999; % X[9] - 0.999*X[10] ≤ 0, 즉 X_FPU ≤ X_DPU
```

이것은 "INT의 scaling factor가 FPU의 1.843배보다 작아야 한다"는 의미이다. 그러나 이것이 "INT가 FPU보다 에너지가 작다"는 뜻은 아니다. Scaling factor와 실제 에너지는 다른 개념이다. Scaling factor는 McPAT의 초기 추정치를 **보정**하는 계수이므로, 제약조건은 "McPAT 기반 에너지 비율로 보정 계수를 제한한다"는 의미이다.

**수정 권장**: 제약조건 설명을 다음과 같이 수정:
```
제약 조건:
  ∀i: 0.1 ≤ X_i ≤ 1000                       (범위 제약)
  X_IDLE_COREP = X_CONSTP = X_STATICP = 1     (이미 모델링 완료)
  
  McPAT 에너지 비율 기반 보정 계수 제약:
  X_INT ≤ 1.843 × X_FPU
  X_FPU ≤ X_DPU
  X_INT ≤ 1.107 × X_INT_MUL24
  ...
  X_FP_MUL ≤ 75.07 × X_TENSOR
```

#### E4. 제약 C(6), C(7), C(14)가 코드에서 누락되어 있지만 문서에서 언급하지 않음

**문제**: `quadprog_solver.m`에서 C 행렬의 행 6, 7, 14가 정의되지 않았다 (zeros로 남아있음). 즉 실제로는 **16개 제약 중 13개만 active**하다. 문서에서 "16개 부등식 제약"이라고 했으나 정확히는 13개이다.

```matlab
% 정의된 행: 1,2,3,4,5, 8,9,10,11,12,13, 15,16 = 13개
% 미정의 행: 6,7,14 = 3개 (zeros = 항상 충족되는 trivial 제약)
```

**수정 권장**: "13개 active 부등식 제약 (코드의 C 행렬 16행 중 3행은 미사용)"

#### E5. Sampling period "500 cycles"는 config 가능한 값

**문제**: 여러 문서에서 "매 500 cycles마다"라고 고정적으로 서술했으나, 이는 `gpgpusim.config`의 `-gpgpu_runtime_stat 500` 값에 의해 결정되며 변경 가능하다.

**수정 권장**: "기본값 500 cycles마다 (gpgpu_runtime_stat 설정으로 변경 가능)"

#### E6. Eq.(8)에서 기하평균 표현이 부정확

**문제**: `P_perIdleSM = ⁸⁰√(∏ P_perIdleSM,i)`라고 썼으나, 80은 SM 수이고 n은 microbenchmark 수이다. 

실제 논문 Eq.(8): `P_perIdleSM = ⁿ√(∏(i=1→n) P_perIdleSM,i)` — n은 microbenchmark 수.

01 문서의 Section 13 Equation 정리에는 `ⁿ√(∏ P_perIdleSM,i)` [기하평균]으로 올바르게 표기했으나, Section 5.5에서는 설명이 부족.

**수정 권장**: "n개 microbenchmark 각각에서 도출된 P_perIdleSM,i의 기하평균 (n = microbenchmark 수)"

#### E7. V100 SM당 Tensor Core 수

**문제**: V100 SM에 "2× Tensor Cores (1st gen)"이라고 여러 곳에서 적었으나, NVIDIA Volta Whitepaper에 따르면 V100의 SM당 Tensor Core는 **8개**이다 (processing block당 2개 × 4 blocks = 8). 다만 `gpgpusim.config`에서는 `gpgpu_num_tensor_core_units 4`로 설정되어 있는데, 이는 sub-core model에서 block당 unit 수를 의미한다.

**수정 권장**: "SM당 8 Tensor Cores (processing block당 2개 × 4 blocks). config의 `gpgpu_num_tensor_core_units 4`는 sub-core model에서 4개 block을 나타낸다."

---

### 02_Improvement_Points.md

#### E8. A100 SM당 구성 서술 수정 필요

**문제**: "64 INT32 + 64 FP32" 라고 쓴 부분이 있으나, 정확하게는 A100 SM당:
- 64 FP32 cores (전용 16 + 공유 16, × 4 blocks)
- 공유 16 cores가 INT32로도 사용 가능
- 따라서 "64 FP32/INT32(공유) + 64 FP32(전용)"이 아니라 "64 FP32(전용) + 64 FP32/INT32(공유)"

**수정 권장**: NVIDIA Ampere Whitepaper 기준 정확한 표현:
```
A100 SM: 4 Processing Blocks, 각각:
  - 16 FP32 cores (전용, FP32 only)
  - 16 FP32/INT32 cores (공유, FP32 또는 INT32 실행)
  - 8 FP64 cores
  - 1 Tensor Core (3rd gen)
  → SM당 총: 64 FP32(전용) + 64 FP32/INT32(공유) + 32 FP64 + 4 Tensor Cores
  → FP32 최대 처리량: 128 cores (전용+공유 모두 FP32)
  → INT32 최대 처리량: 64 cores (공유 코어만)
```

#### E9. "INT32 cores/SM = 16 (전용)" 표현이 V100에서 부정확

**문제**: Section 1 비교 표에서 V100을 "INT32 cores/SM = 16 (전용)"이라고 썼으나, V100은 processing block당 16 INT32 cores × 4 blocks = SM당 **64 INT32 cores**.

**수정 권장**: "V100: INT32 64/SM (전용), FP32 64/SM (전용)" 으로 수정

---

### 03_Equation_Examples.md

#### E10. Dynamic Power 계산 공식의 정확한 flow가 애매함

**문제**: "Component_Power = (McPAT_base_energy × Activity_Count × Scaling_Factor) / Execution_Time" 이라고 했으나, 실제 코드 흐름은 더 복잡하다.

실제 흐름:
```
1. activity count × scaling_factor → effpower_coeff[i]에 저장
   (update_coefficients() 함수)

2. effpower_coeff가 McPAT의 내부 변수(stats)로 설정됨
   (set_*_power() 함수들)

3. McPAT의 proc->compute()가 실행되어 rt_power.readOp.dynamic 계산
   (이때 내부적으로 technology-dependent energy × activity 계산)

4. 결과를 executionTime으로 나누어 power(W) 변환
   sample_cmp_pwr[X] = rt_power.readOp.dynamic / executionTime
```

즉, scaling factor는 activity count에 곱해진 후 McPAT에 전달되어 McPAT 내부에서 energy를 계산한다. "McPAT_base_energy × Activity × Scaling / Time"이라는 단순화는 대략적으로 맞지만, 실제로는 McPAT이 technology-dependent 모델을 사용하여 energy를 계산하므로 단순 곱셈이 아니다.

**수정 권장**: 다음 주석 추가: "위 공식은 개념적 설명이다. 실제로는 scaling_factor가 적용된 activity count가 McPAT에 전달되고, McPAT이 cache/ALU/interconnect의 technology-dependent energy 모델을 사용하여 에너지를 계산한 뒤 execution time으로 나눈다."

#### E11. Static Power에서 코드는 Linear Model만 사용 (Half-warp 아님)

**문제**: 01 문서에서 Half-warp model (Eq.5)을 상세히 설명했으나, **실제 코드 `calculate_static_power()`는 Linear Model (Eq.4)만 구현**하고 있다.

```cpp
// gpgpu_sim_wrapper.cc:914-917
total_static_power =
    base_static_power + (((double)avg_threads_per_warp - 1.0) *
                         lane_static_power);  // Linear Model  ← 주석도 "Linear"
return (total_static_power * per_active_core);
```

논문에서는 Half-warp model이 더 정확하다고 분석했지만 (Figure 4a의 sawtooth 패턴), **코드에 실제로 구현된 것은 Linear model뿐**이다. 논문에서도 "we create the appropriate half-warp or linear models for each instruction mix category and integrate them in AccelWattch"라고 했으나, 현재 오픈소스 코드에서는 Linear model만 사용한다.

이것은 매우 중요한 차이점이다. Half-warp model은 논문의 분석적 발견이지, 현재 공개된 코드의 구현은 아니다.

**수정 권장**: 01, 03, 04 문서 모든 곳에서 다음 명시: "논문에서는 Half-warp model (Eq.5)을 제안했으나, 현재 오픈소스 코드(`calculate_static_power()`)에는 **Linear model (Eq.4)만 구현**되어 있다. 이는 향후 개선 포인트이기도 하다."

---

### 04_A100_Equation_Changes.md

#### E12. A100 Processing Block당 FP64 core 수

**문제**: "8× FP64 cores (Processing Block당)"이라고 썼으나, NVIDIA Whitepaper에 따르면 A100은 SM당 총 32 FP64 cores = processing block당 **8 FP64 cores**가 맞다. 다만 V100에서도 block당 8 FP64 = SM당 32 FP64인데, 01 문서 Table에서는 "FP64 cores/SM = 8"이라고 써서 혼란을 줄 수 있다.

**수정**: V100 FP64: block당 4개 × 4 blocks = SM당 16개가 올바른 수치 (V100 whitepaper 기준). 02 문서의 "8× FP64 cores"는 block당 기준이므로 SM당 32인 A100 기준에서는 맞지만, V100 비교 시 주의 필요.

실제 스펙:
- V100: SM당 32 FP64 cores (block당 8) — 그러나 NVIDIA Volta whitepaper에 따르면 처리량은 "each SM has 32 FP64 cores"
- A100: SM당 32 FP64 cores (block당 8) — FP64 처리량은 V100의 2배 (clock 기준이 아닌 throughput 기준)

이 부분은 config 파일에서 `gpgpu_num_dp_units 4`로 되어있어 (sub-core model에서 block 수) 혼란을 줄 수 있다.

#### E13. gpgpusim.config의 `gpgpu_n_mem 40`의 의미

**문제**: "메모리 파티션 40개"라고 썼는데, 이 값의 정확한 의미를 명확히 할 필요가 있다.

GA100 풀 다이는 12개의 512-bit Memory Controller를 가지며, A100 제품은 harvesting으로 **10개**가 활성화된다. `gpgpu_n_mem 40`과 `gpgpu_n_sub_partition_per_mchannel 4`는 10 MC × 4 sub-partitions = 40을 의미한다. 즉 `gpgpu_n_mem`은 memory controller 수가 아니라 **memory partition(sub-partition) 수**이다.

**수정 권장**: "`gpgpu_n_mem 40` = 10개 Memory Controller × 4 sub-partitions/MC. V100의 `gpgpu_n_mem 32` = 32개 Memory Controller × 1 (V100은 sub-partition이 아닌 pseudo-channel 구조)"

실제로 V100은 `gpgpu_n_sub_partition_per_mchannel 2`이므로: 32 × 2 = 64 sub-partitions. 이는 L2 캐시 슬라이스 수와 관련된다.

---

## 문서 전반에 걸친 보충 사항

#### S1. Eq.(12)의 벡터 관점 해석 보충

논문 Eq.(12)를 dot product로 이해하면 직관적이다:

```
P_est = [P̂₁·x₁, P̂₂·x₂, ..., P̂ₙ·xₙ, P_static, P_idle, P_const] · [a₁, ..., aₙ, 1, 1, 1]ᵀ
```

여기서 마지막 3개 항(static, idle, const)의 activity factor는 항상 1이고 scaling factor도 1로 고정된다. 이것이 quadprog_solver.m에서 `l(29)=u(29)=1` 등으로 고정하는 이유이다.

#### S2. gen_sim_power_csv.py의 component 결합/제거 로직

이 스크립트는 AccelWattch variant에 따라 다른 component를 결합하거나 제거한다:

```python
# SASS SIM 모드: 8개 제거
for each in ["MCP,", "TCP,", "INT_MUL24P,", "INT_MUL32P,",
             "INT_DIVP,", "FP_DIVP,", "DP_DIVP,", "NOCP,"]:
    del power_dict[benchmark_idx][each]

# 추가로 DRAMP = DRAMP + MCP, L2CP = L2CP + NOCP로 결합됨
```

이 결합 로직은 논문 Table 1에서 "L2 Cache and NoC" / "DRAM and Memory Controller"가 하나의 component로 결합된 이유이다. MCP와 NOCP는 개별 하드웨어 카운터로 분리할 수 없기 때문이다.

#### S3. Static Power 계산에서 `per_active_core`의 의미

```cpp
double per_active_core = (num_cores - num_idle_cores) / num_cores;
```

이 값은 0~1 사이의 비율이며, active SM 비율을 나타낸다. Static power 전체가 이 비율로 곱해지는데, 이는 **active SM들의 static power 총합**을 의미한다. Idle SM의 전력은 별도로 `IDLE_COREP` component에서 처리된다.

#### S4. DVFS 적용 시 코드의 전압 비율 처리

```cpp
if (g_dvfs_enabled) {
    double voltage_ratio = modeled_chip_voltage / modeled_chip_voltage_ref;
    
    // Static power (leakage): V에 비례
    IDLE_COREP *= voltage_ratio;
    STATICP *= voltage_ratio;
    
    // Dynamic power: V²에 비례
    for (all other components)
        component *= voltage_ratio * voltage_ratio;
}
```

문서에서 "dynamic power는 V²에 비례"라고 했는데, 이는 `P_dyn ∝ CV²f`에서 f가 이미 activity factor에 반영되어 있기 때문이다. Voltage만 별도로 보정하는 것이다. 그러나 **CONSTP(constant power)도 V²로 스케일링**된다는 점이 약간 의아하다. 물리적으로 constant power(팬, 보드)는 전압에 무관해야 하지만, 코드에서는 clock distribution 등의 dynamic 요소가 CONSTP에 포함되어 있기 때문이다.

#### S5. 코드와 논문의 Eq.(14) 형식 차이

논문의 Eq.(14):
```
X* = argmin_X { Xᵀ · P_est^T · P_est · X - (P_est^T · P_meas)^T · X }
```

MATLAB 코드:
```matlab
result = quadprog(2*A'*A, -2*A'*b, C, D, [], [], l, u);
```

MATLAB `quadprog`의 표준 형식은 `min ½xᵀHx + fᵀx`이므로:
- `H = 2·A'·A` (factor 2는 ½과 상쇄)
- `f = -2·A'·b`

이를 전개하면: `min (Ax-b)'(Ax-b)` = `min ||Ax - b||²` — 즉 least squares 문제이다. 논문의 표현은 같은 문제를 다르게 쓴 것이다.

---

## 수정 우선순위 요약

| 우선순위 | ID | 영향 문서 | 내용 | 유형 |
|---------|-----|----------|------|------|
| **높음** | E11 | 01, 03, 04 | **코드는 Linear Model만 구현, Half-warp 미구현** | 사실 오류 |
| **높음** | E2 | 01 | QP lower bound: 0.001 → **0.1** | 수치 오류 |
| **높음** | E1 | 01 | Component 수: 22개 vs 33개 label, 정확한 관계 설명 필요 | 애매함 |
| **중간** | E8/E9 | 02 | A100/V100 SM당 core 수 표현 정리 (block당 vs SM당) | 혼란 유발 |
| **중간** | E3 | 01 | 제약조건 방향/의미 정확히 서술 | 부정확 |
| **중간** | E4 | 01 | 13개 active 제약 (16개 중 3개 미사용) | 누락 |
| **중간** | E10 | 03 | McPAT 내부 energy 계산이 단순 곱셈이 아님 | 과단순화 |
| **낮음** | E5 | 01, 03 | Sampling period 500 cycles는 설정 가능 | 맥락 부족 |
| **낮음** | E6 | 01 | 기하평균의 n = microbenchmark 수 | 표기 애매 |
| **낮음** | E7 | 01, 04 | V100 Tensor Core: SM당 8개 (block당 2개) | 표현 불일치 |
| **낮음** | E12 | 04 | V100/A100 FP64 core 수 혼란 | 맥락별 차이 |
| **낮음** | E13 | 04 | gpgpu_n_mem의 정확한 의미 | 설명 부족 |

---

> **가장 중요한 발견**: **E11** — 논문에서 제안한 Half-warp model (Eq.5)이 현재 오픈소스 코드에 구현되어 있지 않고, Linear model (Eq.4)만 구현되어 있다. 이는 AccelWattch를 개선할 때 Half-warp model을 실제로 구현하면 정확도를 높일 수 있는 기회이기도 하다.
