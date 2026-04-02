# AccelWattch를 A100에 적용할 때의 Equation 변화 분석

> **목적**: AccelWattch의 모든 수식(Eq.1~14)이 V100→A100으로 바뀔 때 구체적으로 무엇이 변하는지 분석  
> **방법**: V100/A100 실제 `gpgpusim.config` 값 기반 비교 + 아키텍처 차이 반영  
> **작성일**: 2026-04-02  

---

## 목차

1. [V100 vs A100 아키텍처 파라미터 직접 비교](#1-v100-vs-a100-아키텍처-파라미터-직접-비교)
2. [Eq.(1)~(3): Constant Power 변화](#2-eq1eq3-constant-power-변화)
3. [Eq.(4)~(5): Static Power Model 변화](#3-eq4eq5-static-power-model-변화)
4. [Eq.(6)~(8): Idle SM Power 변화](#4-eq6eq8-idle-sm-power-변화)
5. [Eq.(10): 전체 모델 변화](#5-eq10-전체-모델-변화)
6. [Eq.(11)~(12): Dynamic Power 변화](#6-eq11eq12-dynamic-power-변화)
7. [Eq.(13)~(14): Quadratic Programming 변화](#7-eq13eq14-quadratic-programming-변화)
8. [A100 수치 예시: sgemm 커널](#8-a100-수치-예시-sgemm-커널)
9. [고려해야 할 핵심 사항 체크리스트](#9-고려해야-할-핵심-사항-체크리스트)
10. [자가점검: 정확성 검증](#10-자가점검-정확성-검증)

---

## 1. V100 vs A100 아키텍처 파라미터 직접 비교

### 1.1 gpgpusim.config에서 확인한 실제 값

| 파라미터 | V100 (SM7_QV100) | A100 (SM80_A100) | 변화 | Power Model 영향 |
|---------|-----------------|------------------|------|-----------------|
| `gpgpu_n_clusters` (=SM 수) | **80** | **108** | +35% | Idle SM 모델, Static Power |
| `gpgpu_n_mem` (메모리 파티션) | 32 | **40** | +25% | DRAM/MC power |
| `gpgpu_n_sub_partition_per_mchannel` | 2 | **4** | 2x | L2 bank 구조 |
| Core Clock (MHz) | 1132 | **1410** | +25% | DVFS, Dynamic Power |
| DRAM Clock (MHz) | 850 | **1512** | +78% | Memory power |
| `gpgpu_num_sp_units` (FP32) | 4 | 4 | 동일 | - |
| `gpgpu_num_int_units` (INT32) | 4 | 4 | 동일* | ★ 주의 |
| `gpgpu_num_dp_units` (FP64) | 4 | 4 | 동일* | ★ 주의 |
| `gpgpu_num_tensor_core_units` | 4 | 4 | 동일* | ★ 주의 |
| `gpgpu_num_sfu_units` | 4 | 4 | 동일 | - |
| `gpgpu_shader_registers` | 65536 | 65536 | 동일 | - |
| `gpgpu_num_reg_banks` | 16 | **32** | 2x | Register File power |
| `gpgpu_clock_gated_lanes` | **1** (활성화) | 없음 | ★ | Power gating 모델 |
| Tensor latency/initiation | 64/64 | **12/8** | ★ 대폭 감소 | Tensor throughput |
| L1D (unified) size | 128KB | **192KB** | +50% | Cache power |
| Shared memory max | 96KB | **164KB** | +71% | Shared mem power |
| L2 sub-partitions total | 32×2=64 | 40×4=**160** | 2.5x | L2 power |
| Tech node | 12nm | **7nm** | -42% | 전체 power scaling |

> **★ 주의**: `gpgpu_num_sp/int/dp_units = 4`는 sub-core model에서 **processing block당 유닛 수**이다.  
> V100: 각 block에 전용 INT32 16개 + 전용 FP32 16개 = 실제로 4×16=**64 INT + 64 FP** per SM  
> A100: 각 block에 공유 FP32/INT32 16개 + 전용 FP32 16개 = 실제로 4×(16+16)=**64 FP32+64 FP32/INT32** per SM  
> **config 파일의 숫자가 같더라도 내부 동작이 근본적으로 다르다.**

### 1.2 아키텍처 수준 핵심 차이

```
V100 SM 내부 구조 (Processing Block 1개):
┌──────────────────────────────────────┐
│ 16× INT32 cores (전용 datapath)     │ ← 별도 power gating
│ 16× FP32 cores (전용 datapath)      │ ← 별도 power gating
│ 8× FP64 cores                       │
│ 2× Tensor Cores (1st gen, FP16)     │
│ 1× SFU                              │
│ 1× LD/ST unit                       │
└──────────────────────────────────────┘
→ INT32와 FP32 동시 실행 가능 (별도 파이프라인)

A100 SM 내부 구조 (Processing Block 1개):
┌──────────────────────────────────────┐
│ 16× FP32 cores (전용 FP32 only)     │ ← FP 전용
│ 16× FP32/INT32 cores (공유!)        │ ← FP 또는 INT, 택1 ★
│ 8× FP64 cores                       │
│ 1× Tensor Core (3rd gen, TF32/BF16) │
│ 1× SFU                              │
│ 1× LD/ST unit                       │
└──────────────────────────────────────┘
→ INT32 전용 datapath 없음
→ FP32와 INT32는 공유 코어에서 교대 실행
→ FP32만 쓸 때: 32 FP32 cores 전부 활용 (V100의 2배 처리량)
→ INT32만 쓸 때: 16 cores만 사용 가능 (V100과 동일)
→ FP32+INT32 동시: 16 FP전용 + 16 INT공유 (V100과 처리량 동일하지만 power profile 다름)
```

---

## 2. Eq.(1)~(3): Constant Power 변화

### 2.1 Eq.(1): 변하지 않음 (구조 동일)

```
V100: P_total = P_proc,dyn + P_mem,dyn + P_proc,static + P_mem,static + P_const
A100: P_total = P_proc,dyn + P_mem,dyn + P_proc,static + P_mem,static + P_const
                                                                         (동일 구조)
```

### 2.2 Eq.(2): 파라미터 값 변화

```
V100: P_total = mCV²f + nV + P_const

A100: P_total = m'C'V'²f' + n'V' + P_const'
```

| 파라미터 | V100 | A100 | 변화 이유 |
|---------|------|------|----------|
| C (capacitance) | C_12nm | C_7nm ≈ 0.6×C_12nm | 7nm 공정으로 capacitance 감소 |
| f (clock) | 1132-1380 MHz | 765-1410 MHz | 더 넓은 범위 |
| V (voltage) | ~0.8V | ~0.75V (추정) | 7nm 저전압 |
| P_const | **32.3W** | **~50-65W** (추정) | 보드 복잡도↑, 400W TDP |

### 2.3 Eq.(3): DVFS 모델 — 핵심 변화

```
V100: P_total = βCf³ + τf + P_const       (단일 clock domain)

A100: P_total = β_sm·C_sm·f_sm³ + β_mem·C_mem·f_mem³
             + τ_sm·f_sm + τ_mem·f_mem + P_const'
```

**변화 이유:**

V100의 clock domains (`gpgpusim.config`):
```
-gpgpu_clock_domains 1132.0:1132.0:1132.0:850.0
                     Core   :Icnt   :L2    :DRAM
→ Core/Icnt/L2가 동일 → 사실상 2개 domain (Core, DRAM)
```

A100의 clock domains:
```
-gpgpu_clock_domains 1410:1410:1410:1512
                     Core:Icnt:L2  :DRAM
→ Core와 DRAM clock이 더 독립적으로 제어됨
→ 실제 HW에서는 SM clock과 Memory clock이 독립 DVFS
```

**실험 방법 변화:**
```
V100: nvidia-smi로 SM clock만 변경 → 단일 변수 실험 가능
A100: SM clock과 Memory clock을 독립 변경해야 함
      → 2차원 실험 grid 필요
      → 각 (f_sm, f_mem) 조합에서 P_total 측정
      → 2변수 다항식 fitting
```

**예상 실험 설계:**
```
f_sm  ∈ {210, 420, 630, 840, 1050, 1200, 1410} MHz  (7개)
f_mem ∈ {600, 900, 1215, 1512} MHz                   (4개)
→ 28개 (f_sm, f_mem) 조합 × 5개 microbenchmark = 140회 측정

각 조합에서:
  P_total(f_sm, f_mem) = β_sm·f_sm³ + β_mem·f_mem³ + τ_sm·f_sm + τ_mem·f_mem + P_const
  
모든 f = 0 → y절편 = P_const (추정)
```

### 2.4 고려사항: V≈kf 가정의 유효성

```
V100 (12nm): V-F 관계가 비교적 선형 → V≈kf 합리적
A100 (7nm):  V-F 관계가 비선형화 → V≈k₁f + k₂f² 또는 더 복잡
             → Eq.(3)의 f³ 지수가 정확하지 않을 수 있음
             → 실측 fitting에서 지수를 자유 파라미터로 둘 수 있음:
               P ∝ f^α (α를 2.5~3.5 범위에서 fitting)
```

---

## 3. Eq.(4)~(5): Static Power Model 변화

### 3.1 Eq.(4): Linear Model — 구조적 변경 필요

**V100 (현재):**
```
P_static,addLane = (P_static,32Lanes - P_static,firstLane) / 31
P_static,yLanes = P_static,firstLane + P_static,addLane · (y - 1)
```
- `y` = active threads per warp (1~32)
- 각 lane이 자신의 INT32 또는 FP32 코어를 독립적으로 활성화

**A100 (변경 필요):**
```
P_static,addLane = (P_static,32Lanes - P_static,firstLane) / 31  (구조 동일)
P_static,yLanes = P_static,firstLane + P_static,addLane · (y - 1)

단, firstLane과 addLane의 "의미"가 달라짐:
```

| 항목 | V100 firstLane에 포함 | A100 firstLane에 포함 |
|------|---------------------|---------------------|
| Chip-wide (L2) | O | O |
| SM-wide (L1, ShMem) | O | O |
| INT32 core 1개 | O (전용 core) | △ (공유 core의 INT 모드) |
| FP32 core 1개 | O (전용 core) | O (전용 core) + △ (공유 core의 FP 모드) |
| Tensor Core | X | X |

**핵심 차이: INT32/FP32 공유 코어의 전력**

V100에서는 INT32 lane과 FP32 lane의 power gating이 완전히 독립적이었다.
A100에서는 공유 코어가 INT 또는 FP 모드로 전환되므로:

```
A100에서 firstLane이 INT를 실행할 때:
  → 16 전용 FP32 cores 중 1개 + 16 공유 cores 중 1개(INT 모드) 활성화? 
  → 아니면 공유 core 1개만 INT로 활성화?

A100에서 firstLane이 FP를 실행할 때:
  → 전용 FP32 1개 + 공유 FP32 1개 = 2개 활성화?
  → 아니면 전용 FP32 1개만?
```

**제안하는 수정:**
```
# A100용 확장 Linear Model
P_static,yLanes,mode = P_static,firstLane(mode) + P_static,addLane(mode) · (y - 1)

where mode ∈ {
  FP_ONLY:       전용FP + 공유FP 모두 FP 실행
  INT_ONLY:      공유core만 INT 실행, 전용FP idle
  FP_INT_CONC:   전용FP는 FP, 공유core는 INT (동시)
}

각 mode별로 firstLane, addLane 값이 다름 → microbenchmark로 각각 측정
```

### 3.2 Eq.(5): Half-Warp Model — 근본적 재설계 필요

**V100 (현재):**
```
         ┌ P_static,firstLane + P_static,addLane·(y-1),       y ≤ 16
P_static = ┤
         └ P_static,firstLane + ½·addLane·15 + ½·addLane·(y-17), y > 16
```

이 모델은 **4개 processing block이 각각 16 cores를 가지며, warp이 2개의 half-warp로 실행**된다는 V100의 구조를 전제한다.

**A100에서의 변화:**

A100도 4 processing blocks를 가지지만, 내부 구조가 다르다:
```
V100 Processing Block: 16 INT32(전용) + 16 FP32(전용) = 32 cores
A100 Processing Block: 16 FP32(전용) + 16 FP32/INT32(공유) = 32 cores
```

**Half-warp 실행 패턴 변화:**

```
V100에서 warp 32 threads 실행 (FP32 연산):
  Half-warp 1: FP32 core 0-15 (Block A) → 활성
  Half-warp 2: FP32 core 0-15 (Block B) → 활성
  동시에: INT32 core도 INT 연산 가능

A100에서 warp 32 threads 실행 (FP32 연산):
  모든 32 FP32 cores 활용 가능 (전용16 + 공유16)
  → Half-warp 경계가 V100과 다름
  → y=16 에서 y=17로 넘어갈 때의 sawtooth 패턴이 다를 수 있음
```

**제안하는 수정:**
```
# A100용 Half-warp Model
# 전용 FP32: 16개, 공유 FP32/INT32: 16개 = 총 32개

# FP_ONLY 모드에서:
P_static,yLanes = {
  P_firstLane + P_addLane_fp·(y-1),                    y ≤ 16  (전용 FP만 사용)
  P_firstLane + P_addLane_fp·15 
    + P_addLane_shared·(y-17),                          16 < y ≤ 32 (공유core도 FP 사용)
}

# INT_ONLY 모드에서:
P_static,yLanes = {
  P_firstLane_int + P_addLane_int·(y-1),               y ≤ 16  (공유 core만 사용)
  (y > 16은 불가: INT는 공유 16개만 사용 가능)
}

# FP_INT_CONCURRENT 모드에서:
P_static,yLanes = {
  P_firstLane_conc + P_addLane_fp·(y_fp-1)
    + P_addLane_int·(y_int-1),                          y_fp, y_int 각각 추적 필요
}
```

### 3.3 Static Power 카테고리 확장

**V100 (현재 9개):**
```
cat1: INT(ADD+MUL)    cat2: INT+FP        cat3: INT+FP+DP
cat4: INT+FP+SFU      cat5: INT+FP+TEX    cat6: INT+FP+TENSOR
intadd: INT ADD only   intmul: INT MUL only  light: LIGHT_SM
```

**A100 (필요한 추가 카테고리):**
```
기존 유지:
  cat1~cat6, intadd, intmul, light

추가 필요:
  cat7:  FP_ONLY (INT 없이 FP32만 사용 → A100에서 32 cores 전부 FP)
  cat8:  INT+FP+TF32 (TF32 Tensor Core 사용)
  cat9:  INT+FP+BF16 (BF16 연산)
  cat10: INT+FP+SPARSE (2:4 Sparsity Tensor)
  cat_concurrent: FP+INT 동시 실행 (전용FP=FP, 공유core=INT)
```

**XML config 변화:**
```xml
<!-- V100: 9개 카테고리 × 2 (flane, addlane) = 18 파라미터 -->
<param name="static_cat1_flane" value="15.29"/>
<param name="static_cat1_addlane" value="0.586"/>
...
<param name="static_cat6_flane" value="48.95"/>
<param name="static_cat6_addlane" value="0.0"/>

<!-- A100: 확장 → 13+ 카테고리 × 2 = 26+ 파라미터 -->
<param name="static_cat7_flane" value="??"/>      <!-- FP_ONLY -->
<param name="static_cat7_addlane" value="??"/>
<param name="static_cat8_flane" value="??"/>      <!-- INT+FP+TF32 -->
<param name="static_cat8_addlane" value="??"/>
<param name="static_cat_concurrent_flane" value="??"/>  <!-- FP+INT concurrent -->
<param name="static_cat_concurrent_addlane" value="??"/>
```

---

## 4. Eq.(6)~(8): Idle SM Power 변화

### 4.1 Eq.(6): SM 수 변경

```
V100: P_dyn+static,perActiveSM = (P_total,80SMs - P_const) / 80

A100: P_dyn+static,perActiveSM = (P_total,108SMs - P_const') / 108
                                               ↑             ↑
                                          새 P_const     SM 수 변경
```

### 4.2 Eq.(7): Idle SM 전력 분리

```
V100: P_idleSMs = P_total - P_const - P_dyn+static,perActiveSM · N_activeSMs
      (N_activeSMs 범위: 1~80)

A100: P_idleSMs = P_total - P_const' - P_dyn+static,perActiveSM · N_activeSMs
      (N_activeSMs 범위: 1~108)
      → Idle SM이 최대 107개 → Idle SM power의 기여도가 상대적으로 더 커질 수 있음
```

### 4.3 Eq.(8): 기하평균 — 동일 구조, 다른 값

```
V100: P_perIdleSM = ⁸⁰√(∏ P_perIdleSM,i)    (n개 microbenchmark)
A100: P_perIdleSM = ¹⁰⁸√(∏ P_perIdleSM,i)   (n개 microbenchmark → 더 많이 필요)
```

**고려사항:**
- A100은 7nm 공정이므로 idle SM의 leakage가 V100(12nm)과 다름
- 7nm에서 leakage current는 게이트 밀도 대비 상대적으로 증가할 수 있음
- idle_core_power 값이 V100의 0.283W와 상당히 다를 것으로 예상
- microbenchmark에서 Active SM 수를 1, 8, 16, 32, 54, 80, 108 등으로 변화시켜야 함

**예상 측정 시나리오:**
```
Active SMs: [1, 4, 8, 16, 27, 36, 54, 72, 90, 108]
각 SM 수에서 여러 microbenchmark 실행 → P_total 측정
→ Eq.(6),(7)로 P_perIdleSM 도출
→ Eq.(8)로 기하평균
```

---

## 5. Eq.(10): 전체 모델 변화

### V100 (현재):
```
P_total = P_dyn + P_static,yLanes,perActiveSM · k + P_perIdleSM · (80 - k) + P_const
```

### A100 (변경):
```
P_total = P_dyn + P_static,yLanes,mode,perActiveSM · k + P_perIdleSM · (108 - k) + P_const'
                              ↑                          ↑                             ↑
                         mode 추가                    108 SMs                      재측정
```

**수치 비교 예시 (가상):**

```
시나리오: FP32 GEMM, 모든 SM active, 32 threads/warp

V100:
  P_dyn = 77 W
  P_static = static_cat2(32 threads) × 80/80 = (18.6 + 31×0.645) × 1.0 = 38.6 W
  P_idle = 0.283 × 0 = 0 W
  P_const = 32.3 W
  P_total = 77 + 38.6 + 0 + 32.3 = 147.9 W

A100 (추정):
  P_dyn = 110 W   (더 많은 cores, 높은 clock → dynamic power ↑)
  P_static = static_cat7(32 threads) × 108/108 = (22? + 31×0.5?) × 1.0 ≈ 37.5 W
  P_idle = 0.20? × 0 = 0 W    (7nm에서 idle power 감소?)
  P_const = 55 W               (400W TDP에 비례)
  P_total = 110 + 37.5 + 0 + 55 = 202.5 W

  * A100 TDP = 400W이므로 heavy workload에서는 더 높아짐
```

---

## 6. Eq.(11)~(12): Dynamic Power 변화

### 6.1 Eq.(11): Component 확장

```
V100: P_dyn = Σ(i=1→22) [a_i · E_i / T]

A100: P_dyn = Σ(i=1→27+) [a_i · E_i / T]
                    ↑
              5개 이상 component 추가
```

**새 component 상세:**

| # | 신규 Component | Activity Counter | 하드웨어 유닛 |
|---|---------------|-----------------|-------------|
| 23 | **TF32P** | TF32_ACC | 3rd gen Tensor Core (TF32 모드) |
| 24 | **BF16P** | BF16_ACC | 3rd gen Tensor Core (BF16 모드) |
| 25 | **SPARSEP** | SPARSE_ACC | 2:4 Structured Sparsity Engine |
| 26 | **ASYNCP** | ASYNC_COPY_ACC | Async Copy Engine (cp.async) |
| 27 | **L2_PARTP** | L2_PART_ACC | L2 Residency Control |

### 6.2 Eq.(12): Scaling Factor 변화

```
V100:
P_est = Σ(i=1→22) [a_i · Ê_i · x_i / T]
      + P_static,yLanes,perActiveSM · k
      + P_perIdleSM · (80 - k) + P_const

A100:
P_est = Σ(i=1→27) [a_i · Ê_i · x_i / T]
      + P_static,yLanes,mode,perActiveSM · k
      + P_perIdleSM · (108 - k) + P_const'
```

**Ê_i (초기 에너지 추정치) 변화:**

McPAT이 계산하는 base energy가 7nm 공정에서 달라짐:
```
V100 (12nm): Ê_i 기반 = McPAT with core_tech_node=23 (XML 설정)
A100 (7nm):  Ê_i 기반 = McPAT with core_tech_node=7 (변경 필요)

→ XML의 core_tech_node를 23 → 7로 변경
→ McPAT의 CACTI/Interconnect 모델이 다른 에너지 값 출력
→ 모든 cache, ALU, memory의 base energy가 변경됨
```

### 6.3 Component별 Power 예상 변화

```
┌────────────────────────────────────────────────────────────────────┐
│           V100 vs A100 Dynamic Power 예상 비교 (sgemm)            │
├────────────┬────────────┬────────────┬─────────────────────────────┤
│ Component  │ V100 (W)   │ A100 (W)   │ 변화 이유                   │
├────────────┼────────────┼────────────┼─────────────────────────────┤
│ IBP        │ 3.2        │ ~3.5       │ 명령어 수 증가               │
│ ICP        │ 2.8        │ ~3.0       │ I-cache 크기 ↑ (128KB)     │
│ DCP        │ 8.5        │ ~10.0      │ L1D 192KB, 더 많은 접근     │
│ SHRDP      │ 4.2        │ ~5.5       │ ShMem 164KB ↑               │
│ RFP        │ 15.8       │ ~14.0      │ 32 banks (V100:16), 더 효율 │
│ INTP       │ 5.1        │ ~4.0       │ FP 위주 커널, INT ↓        │
│ FPUP       │ 1.8        │ ~2.5       │ FP add/cmp                  │
│ FP_MULP    │ 12.4       │ ~18.0      │ 더 많은 FP32 cores (32개)  │
│ TENSORP    │ 0.0        │ 0.0        │ sgemm은 Tensor 미사용       │
│ TF32P(신규)│ -          │ 0.0        │ sgemm은 TF32 미사용         │
│ L2CP       │ 5.2        │ ~8.0       │ L2 160 sub-partitions ↑     │
│ DRAMP      │ 2.4        │ ~4.0       │ HBM2e 대역폭 2x            │
│ NOCP       │ 3.8        │ ~5.0       │ 108 SM interconnect ↑       │
│ PIPEP      │ 6.2        │ ~7.0       │ 더 많은 SM, 클럭 ↑         │
├────────────┼────────────┼────────────┼─────────────────────────────┤
│ Σ Dynamic  │ 77 W       │ ~95-110 W  │ +23~43%                     │
└────────────┴────────────┴────────────┴─────────────────────────────┘
```

---

## 7. Eq.(13)~(14): Quadratic Programming 변화

### 7.1 Eq.(13): 행렬 차원 변화

```
V100: P_est^{M×(22+3)} × X^{(25)×1} = P_meas^{M×1}
      M = 102 (microbenchmarks)
      N+3 = 22+3 = 25 power parameters (실제 코드에서는 31열)

A100: P_est^{M'×(27+3)} × X^{(30)×1} = P_meas^{M'×1}
      M' ≈ 140+ (추가 microbenchmarks 포함)
      N+3 = 27+3 = 30+ power parameters
```

### 7.2 Eq.(14): 제약조건 변화

```
V100 제약조건 (quadprog_solver.m에서):
  C = zeros(16,31)  → 16개 부등식 제약

  INT ≤ 1.843 × FPU
  FPU ≤ DPU
  INT ≤ 1.107 × INT_MUL24
  FP_MUL ≤ 14.17 × FP_DIV
  FP_MUL ≤ 1.064 × DP_MUL
  FP_MUL ≤ 5.587 × FP_SQRT
  FP_MUL ≤ 2.083 × FP_LG
  FP_MUL ≤ 1.768 × FP_SIN
  FP_MUL ≤ 1.439 × FP_EXP
  FP_MUL ≤ 75.07 × TENSOR
  FP_MUL ≤ 1.000 × TEXP
```

**A100에서 추가/변경할 제약:**

```
A100 추가 제약조건:

  # 기존 McPAT 비율 계수는 7nm에서 재계산 필요
  # 12nm→7nm에서 FPU vs INT의 에너지 비율이 변할 수 있음
  INT ≤ k₁' × FPU          (k₁'은 7nm 기준으로 재계산)
  
  # 새 component 제약
  TENSOR ≤ k_tf32 × TF32    (TF32가 FP16 Tensor보다 에너지 높음)
  BF16 ≤ k_bf16 × TF32      (BF16이 TF32보다 에너지 낮음)
  SPARSE ≤ k_sp × TENSOR    (Sparsity는 Dense보다 에너지 낮음, 단 2x throughput)
  ASYNC_COPY ≤ k_async × DCP (Async copy가 일반 load보다 에너지 다름)
  
  # 고정 파라미터
  X_IDLE_COREP = 1   (이미 모델링됨)
  X_CONSTP = 1       (이미 모델링됨)
  X_STATICP = 1      (이미 모델링됨)
```

### 7.3 Python 구현 (제안)

```python
import cvxpy as cp
import numpy as np

# A100용 확장된 데이터
data = np.loadtxt('accelwattch_ampere_sass_sim.csv', delimiter=',')
A = data[:, :35]    # 35 activity columns (31→35 확장)
b = data[:, 35]     # measured power

n = 35
x = cp.Variable(n, pos=True)

# 목적함수: ||Ax - b||² + 정규화
lambda_reg = 0.01
objective = cp.Minimize(
    cp.sum_squares(A @ x - b) + lambda_reg * cp.norm(x, 2)
)

# 제약조건
constraints = [
    x >= 0.001,
    x <= 1000,
    # 고정 파라미터
    x[idx_idle] == 1,
    x[idx_const] == 1,
    x[idx_static] == 1,
    # 에너지 순서 (7nm 기준 재계산 필요)
    x[idx_int] <= k1_7nm * x[idx_fpu],
    x[idx_fpu] <= x[idx_dpu],
    # 새 component 제약
    x[idx_tensor] <= k_tf32 * x[idx_tf32],
    x[idx_bf16] <= k_bf16 * x[idx_tf32],
    x[idx_sparse] <= x[idx_tensor],
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP)

print("Scaling factors:", x.value)
```

---

## 8. A100 수치 예시: sgemm 커널

### 전체 계산 흐름 (A100 추정)

```
┌─────────────────────────────────────────────────────────┐
│            A100 sgemm Power Estimation (추정)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [1] Dynamic Power (27 components)                      │
│      FP_MULP = 18.0W  (32 FP32 cores/block, 주력)      │
│      RFP     = 14.0W  (32 reg banks)                   │
│      DCP     = 10.0W  (192KB L1D)                      │
│      L2CP    = 8.0W   (40MB L2, 160 sub-partitions)   │
│      PIPEP   = 7.0W   (108 SMs)                        │
│      SHRDP   = 5.5W   (164KB ShMem)                   │
│      NOCP    = 5.0W   (108-node interconnect)          │
│      SCHEDP  = 5.0W   (4 schedulers/SM)                │
│      DRAMP   = 4.0W   (HBM2e)                          │
│      INTP    = 4.0W   (INT 보조)                       │
│      ICP     = 3.0W   (128KB I-cache)                  │
│      IBP     = 3.5W                                     │
│      FPUP    = 2.5W   (FP add/cmp)                     │
│      기타    = 5.5W                                     │
│      TF32P   = 0W     (미사용)                          │
│      TENSORP = 0W     (미사용)                          │
│      ──────────────────────                             │
│      Σ P_dynamic ≈ 95 W                                │
│                                                         │
│  [2] Static Power                                       │
│      mode: FP_ONLY (INT 없이 FP32 위주)                │
│      → cat7 (새 카테고리) 적용                          │
│      base (firstLane)  ≈ 20 W                           │
│      lane (addLane×31) ≈ 15 W                           │
│      per_active_core   = 108/108 = 1.0                  │
│      P_static ≈ 35 W                                    │
│                                                         │
│  [3] Idle SM Power                                      │
│      Active SMs = 108, Idle = 0                         │
│      P_idle = 0 W                                       │
│                                                         │
│  [4] Constant Power                                     │
│      P_const ≈ 55 W                                     │
│                                                         │
│  ═══════════════════════                                │
│  P_total ≈ 95 + 35 + 0 + 55 = 185 W                   │
│                                                         │
│  비교: V100 sgemm ≈ 148W                               │
│  증가율: +25% (TDP 비율: 185/400=46% vs 148/250=59%)   │
│  → A100이 TDP 대비 더 효율적 (7nm 공정 이점)           │
└─────────────────────────────────────────────────────────┘
```

### Tensor Core 사용 시 (TF32 GEMM)

```
  P_dynamic:
    TF32P   ≈ 35W  (3rd gen Tensor Core, TF32 모드)
    FP_MULP ≈ 5W   (보조 FP 연산)
    기타    ≈ 55W
    Σ ≈ 130W

  P_static (cat8: INT+FP+TF32):
    firstLane ≈ 45W  (Tensor Core 활성화 시 높음)
    addLane   ≈ 0W   (V100 cat6와 유사하게 0)
    P_static ≈ 45W

  P_total ≈ 130 + 45 + 0 + 55 = 230W  (TDP 400W의 58%)
```

---

## 9. 고려해야 할 핵심 사항 체크리스트

### 9.1 반드시 변경해야 하는 것 (MUST)

```
□ [Eq.3]  P_const 재측정 (A100 DVFS 실험)
□ [Eq.4]  firstLane/addLane 값 재측정 (INT/FP 공유 경로 반영)
□ [Eq.5]  Half-warp model에 execution mode 도입
□ [Eq.6]  SM 수: 80 → 108
□ [Eq.8]  idle_core_power 재측정 (7nm leakage)
□ [Eq.10] 상수 80 → 108 변경
□ [Eq.11] TF32, BF16, Sparsity component 추가
□ [Eq.14] 행렬 차원 확장, 제약조건 7nm 기준 재계산
□ [XML]   core_tech_node: 23 → 7
□ [XML]   constant_power: 32.3 → 재측정값
□ [XML]   idle_core_power: 0.283 → 재측정값
□ [XML]   static_cat*: 전부 재측정
□ [XML]   모든 dynamic scaling factors: QP로 재도출
□ [Code]  accelwattch_component_mapping.h: TF32, BF16, Sparse opcode 매핑
□ [Code]  calculate_static_power(): 새 카테고리 추가
□ [Code]  gen_sim_power_csv.py: 새 component 추가
□ [Code]  quadprog_solver.m: 차원 확장 + 새 제약조건
```

### 9.2 주의 깊게 확인해야 하는 것 (SHOULD)

```
□ V≈kf 가정이 7nm에서도 유효한지 실측 확인
□ Multi-domain DVFS (SM/Mem 독립 클럭) 실험 설계
□ INT32/FP32 공유 코어의 power gating 패턴 microbenchmark 확인
□ Tensor Core 3rd gen의 에너지 특성 (TF32 vs FP16 vs BF16 vs INT8)
□ 2:4 Sparsity 활성화 시 power 변화 (2x throughput이지만 power는?)
□ L2 cache partitioning이 power에 미치는 영향
□ Async copy (cp.async)의 power 특성
□ 온도 65°C 유지 가능 여부 (400W TDP에서)
□ NVML 샘플링 주파수 확인 (A100에서 더 높을 수 있음)
□ Register file bank 수 증가(16→32)가 RFP power에 미치는 영향
```

### 9.3 향후 확장 가능 (NICE-TO-HAVE)

```
□ MIG 파티셔닝 시 power 모델 (1/2, 1/3, 1/7 GPU)
□ ML residual correction 모델 적용
□ 교차항 (INT×FP concurrent) 추가
□ Python(cvxpy)으로 QP solver 전환
□ 실시간 power 예측 (온라인 모델)
```

---

## 10. 자가점검: 정확성 검증

### 10.1 Config 값 검증

| 항목 | 본 문서 사용값 | 실제 config 확인 | 일치? |
|------|--------------|-----------------|-------|
| V100 SM 수 | 80 | `gpgpu_n_clusters 80` (SM7_QV100) | ✓ |
| A100 SM 수 | 108 | `gpgpu_n_clusters 108` (SM80_A100) | ✓ |
| V100 Core Clock | 1132 MHz | `gpgpu_clock_domains 1132.0:...` | ✓ |
| A100 Core Clock | 1410 MHz | `gpgpu_clock_domains 1410:...` | ✓ |
| V100 DRAM Clock | 850 MHz | `gpgpu_clock_domains ...:850.0` | ✓ |
| A100 DRAM Clock | 1512 MHz | `gpgpu_clock_domains ...:1512` | ✓ |
| V100 Mem Partitions | 32 | `gpgpu_n_mem 32` | ✓ |
| A100 Mem Partitions | 40 | `gpgpu_n_mem 40` | ✓ |
| V100 Sub-partitions/ch | 2 | `gpgpu_n_sub_partition_per_mchannel 2` | ✓ |
| A100 Sub-partitions/ch | 4 | `gpgpu_n_sub_partition_per_mchannel 4` | ✓ |
| V100 Reg Banks | 16 | `gpgpu_num_reg_banks 16` | ✓ |
| A100 Reg Banks | 32 | `gpgpu_num_reg_banks 32` | ✓ |
| V100 Tensor latency | 64/64 | `tensor 64/64` | ✓ |
| A100 Tensor latency | 12/8 | `tensor 12,8` (trace.config) | ✓ |
| V100 P_const | 32.3W | XML `constant_power=32.325` | ✓ |
| V100 idle_core_power | 0.283W | XML `idle_core_power=0.283` | ✓ |
| V100 static_cat2_flane | 18.618W | XML `static_cat2_flane=18.618` | ✓ |
| V100 static_cat6_flane | 48.949W | XML `static_cat6_flane=48.949` | ✓ |
| A100 power_simulation | 미지원 | `power_simulation_enabled 0` | ✓ (확인) |

### 10.2 수식 일관성 검증

| 검증 항목 | 결과 |
|----------|------|
| Eq.(1) 구조가 A100에서도 성립하는가? | ✓ 물리적 분해이므로 아키텍처 무관 |
| Eq.(3) V≈kf 가정: V100에서 검증됨, A100에서 재검증 필요 | ⚠ 명시함 |
| Eq.(5) Half-warp: V100의 16+16 구조 전제 → A100 변경 필요 | ✓ 상세 분석 |
| Eq.(10) 80→108 변경 명시 | ✓ |
| Eq.(14) 행렬 차원 확장: 31→35+ | ✓ |
| INT/FP 공유 경로 문제 식별 | ✓ Section 3 상세 분석 |
| TF32/BF16/Sparsity 신규 component | ✓ Section 6 |
| A100 config에 AccelWattch XML 미존재 확인 | ✓ `power_simulation_enabled 0` |

### 10.3 추정값 합리성 검증

| 항목 | 추정값 | 검증 |
|------|--------|------|
| A100 P_const | 50-65W | TDP 400W의 13-16% → V100(13%)과 비슷 → ✓ 합리적 |
| A100 sgemm P_total | ~185W | TDP 400W의 46% → V100(59%)보다 낮음 → ✓ 7nm 이점 |
| A100 TF32 GEMM P_total | ~230W | TDP의 58% → Tensor 사용 시 증가 → ✓ 합리적 |
| A100 idle_core_power | ~0.2W/SM | 7nm에서 V100(0.283W) 대비 감소 → ✓ 공정 스케일링 |
| Dynamic power 23-43% 증가 | 95-110W vs 77W | SM +35%, clock +25% → ✓ 합리적 |

### 10.4 발견된 불확실성

| # | 불확실한 사항 | 해결 방법 |
|---|-------------|----------|
| 1 | A100 FP32/INT32 공유 코어의 정확한 power gating 단위 | A100 microbenchmark 실측 필요 |
| 2 | 7nm에서 V-F 관계의 비선형 정도 | DVFS sweep 실험 필요 |
| 3 | TF32 vs FP16 Tensor의 에너지 비율 | McPAT 또는 실측 기반 추정 필요 |
| 4 | 2:4 Sparsity의 power 절감 비율 | Dense vs Sparse microbenchmark 비교 필요 |
| 5 | L2 160 sub-partition의 power gating 단위 | 아키텍처 문서 확인 필요 |
| 6 | A100의 P_const 정확한 값 | DVFS 실험으로만 확인 가능 |
| 7 | McPAT core_tech_node=7 지원 여부 | McPAT 소스코드 확인 필요 |

---

> **결론**: AccelWattch를 A100에 적용할 때 수식의 **구조(형태)**는 대부분 유지되지만, **파라미터 값과 component 수**가 크게 변한다. 가장 큰 도전은 (1) INT32/FP32 공유 실행경로에 따른 Static Power Model 재설계, (2) TF32/BF16/Sparsity 신규 component 추가, (3) 7nm 공정에서의 DVFS/Constant Power 재보정이다.  
> 이전 문서: [03_Equation_Examples.md](03_Equation_Examples.md) — V100 수치 예시  
> 개선 포인트 전체: [02_Improvement_Points.md](02_Improvement_Points.md)
