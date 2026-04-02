# AccelWattch 최종 Equation: 실제 수치 예시

> **목적**: AccelWattch가 실제로 전력을 계산하는 과정을 V100(QV100)의 실제 XML 값과 함께 단계별로 보여줌  
> **소스코드 기반**: `gpgpu_sim_wrapper.cc`의 `update_components_power()`, `calculate_static_power()`, `power_metrics_calculations()`  
> **작성일**: 2026-04-02  

---

## 1. 최종 전력 공식 (코드 기준)

소스코드 `gpgpu_sim_wrapper.cc:524`에서 매 sampling period(500 cycles)마다 계산되는 전력:

```
P_total = P_dynamic + P_const + P_static

여기서:
  P_dynamic = Σ(모든 component의 dynamic power)
            = IBP + ICP + DCP + TCP + CCP + SHRDP + RFP
            + INTP + FPUP + DPUP
            + INT_MUL24P + INT_MUL32P + INT_MULP + INT_DIVP
            + FP_MULP + FP_DIVP + FP_SQRTP + FP_LGP + FP_SINP + FP_EXP
            + DP_MULP + DP_DIVP
            + TENSORP + TEXP + SCHEDP
            + L2CP + MCP + NOCP + DRAMP
            + PIPEP + IDLE_COREP

  P_const  = constant_power (XML에서 직접 읽음)

  P_static = calculate_static_power() (instruction mix 기반)
```

---

## 2. Dynamic Power 계산: 각 Component별 공식

### 2.1 일반 공식

각 component의 dynamic power는 다음과 같이 계산된다:

```
Component_Power = (McPAT_base_energy × Activity_Count × Scaling_Factor) / Execution_Time
```

코드로 보면:

```cpp
// 1단계: activity count에 scaling factor 곱하기
effpower_coeff[i] = initpower_coeff[i] * scaling_coefficients[i]

// 2단계: execution time으로 나누어 power(W) 변환
effpower_coeff[i] /= executionTime

// 3단계: McPAT이 base energy를 계산하고, activity가 적용된 결과가 component power
sample_cmp_pwr[INTP] = (exeu->rt_power.readOp.dynamic / executionTime)
                     × (rf_fu_clockRate / clockRate)
```

### 2.2 V100 XML Scaling Factors (실제 값)

XML 파일 `accelwattch_sass_sim.xml`에서 읽는 scaling factors:

```
┌─────────────────┬─────────────┬───────────────────────────────────┐
│ Parameter       │ Value       │ 의미                              │
├─────────────────┼─────────────┼───────────────────────────────────┤
│ TOT_INST        │ 10.000      │ Instruction Buffer scaling       │
│ FP_INT          │ 4.661       │ Scheduler scaling                │
│ IC_H            │ 8.593       │ I-cache hit energy scaling       │
│ IC_M            │ 29.735      │ I-cache miss energy scaling      │
│ DC_RH           │ 9.835       │ L1D read hit scaling             │
│ DC_RM           │ 10.954      │ L1D read miss scaling            │
│ DC_WH           │ 0.680       │ L1D write hit scaling            │
│ DC_WM           │ 17.676      │ L1D write miss scaling           │
│ CC_H            │ 0.111       │ Constant cache hit scaling       │
│ CC_M            │ 0.123       │ Constant cache miss scaling      │
│ SHRD_ACC        │ 0.780       │ Shared memory scaling            │
│ REG_RD          │ 0.101       │ Register read scaling            │
│ REG_WR          │ 0.141       │ Register write scaling           │
│ INT_ACC         │ 14.988      │ Integer ALU scaling              │
│ FP_ACC          │ 0.530       │ FP32 unit scaling                │
│ DP_ACC          │ 0.777       │ FP64 unit scaling                │
│ INT_MUL_ACC     │ 0.115       │ Integer multiply scaling         │
│ FP_MUL_ACC      │ 0.090       │ FP multiply scaling              │
│ FP_SQRT_ACC     │ 0.195       │ SFU sqrt scaling                 │
│ FP_LG_ACC       │ 0.126       │ SFU log scaling                  │
│ FP_SIN_ACC      │ 0.133       │ SFU sin/cos scaling              │
│ FP_EXP_ACC      │ 0.362       │ SFU exp scaling                  │
│ DP_MUL_ACC      │ 0.132       │ DP multiply scaling              │
│ TENSOR_ACC      │ 0.815       │ Tensor core scaling              │
│ TEX_ACC         │ 0.115       │ Texture unit scaling             │
│ MEM_RD          │ 0.026       │ DRAM read scaling                │
│ MEM_WR          │ 0.031       │ DRAM write scaling               │
│ MEM_PRE         │ 0.009       │ DRAM precharge scaling           │
│ L2_RH           │ 1.261       │ L2 read hit scaling              │
│ L2_RM           │ 2.395       │ L2 read miss scaling             │
│ L2_WH           │ 4.125       │ L2 write hit scaling             │
│ L2_WM           │ 1.223       │ L2 write miss scaling            │
│ NOC_A           │ 32.090      │ NoC access scaling               │
│ PIPE_A          │ 0.514       │ Pipeline scaling                 │
│ constant_power  │ 32.325      │ Constant power (W, 직접 사용)    │
│ idle_core_power │ 0.283       │ Idle SM당 power (W)              │
└─────────────────┴─────────────┴───────────────────────────────────┘
```

---

## 3. 구체적 수치 예시: `sgemm` 커널 (FP32 행렬곱)

### 3.1 시나리오 설정

가상의 sgemm 커널이 V100 GV100에서 실행된다고 가정:

```
GPU: Quadro GV100 (80 SMs, 12nm, 1417 MHz)
커널: sgemm (FP32 행렬곱)
Active SMs: 80 (전부 사용)
Active threads per warp: 32 (full warp, divergence 없음)
Sampling period: 500 cycles
Execution time (T): 500 cycles × (1/1417MHz) = 0.353 μs
```

### 3.2 Step 1: Activity Counts (시뮬레이터가 수집)

500 cycles 동안 시뮬레이터가 수집한 raw activity counts (가상 예시):

```
┌─────────────────┬───────────────┬──────────────────────────────┐
│ Counter         │ Raw Count     │ 설명                          │
├─────────────────┼───────────────┼──────────────────────────────┤
│ TOT_INST        │ 12,800        │ 총 warp instructions          │
│ FP_INT          │ 10,240        │ 비메모리 instructions         │
│ IC_H            │ 11,520        │ I-cache hits                  │
│ IC_M            │ 128           │ I-cache misses                │
│ DC_RH           │ 3,200         │ L1D read hits                 │
│ DC_RM           │ 640           │ L1D read misses               │
│ DC_WH           │ 800           │ L1D write hits                │
│ DC_WM           │ 160           │ L1D write misses              │
│ SHRD_ACC        │ 5,120         │ Shared memory accesses        │
│ REG_RD          │ 25,600        │ Register reads                │
│ REG_WR          │ 12,800        │ Register writes               │
│ INT_ACC         │ 2,560         │ Integer ALU ops               │
│ FP_ACC          │ 1,280         │ FP32 add/cmp ops              │
│ FP_MUL_ACC      │ 7,680         │ FP32 FMA/MUL ops (주력!)     │
│ DP_ACC          │ 0             │ FP64 ops (사용 안함)          │
│ TENSOR_ACC      │ 0             │ Tensor ops (사용 안함)        │
│ L2_RH           │ 4,800         │ L2 read hits                  │
│ L2_RM           │ 320           │ L2 read misses                │
│ MEM_RD          │ 256           │ DRAM reads                    │
│ MEM_WR          │ 128           │ DRAM writes                   │
│ NOC_A           │ 1,920         │ NoC accesses                  │
│ PIPE_A          │ 12,800        │ Pipeline cycles               │
└─────────────────┴───────────────┴──────────────────────────────┘
```

### 3.3 Step 2: Scaling Factor 적용 (effpower_coeff)

```cpp
effpower_coeff[i] = raw_count × scaling_factor
```

```
┌─────────────────┬───────────┬──────────────┬────────────────────┐
│ Counter         │ Raw Count │ × Scaling    │ = Effective Count  │
├─────────────────┼───────────┼──────────────┼────────────────────┤
│ TOT_INST        │ 12,800    │ × 10.000     │ = 128,000          │
│ FP_INT          │ 10,240    │ × 4.661      │ = 47,729           │
│ IC_H            │ 11,520    │ × 8.593      │ = 98,992           │
│ IC_M            │ 128       │ × 29.735     │ = 3,806            │
│ DC_RH           │ 3,200     │ × 9.835      │ = 31,472           │
│ DC_RM           │ 640       │ × 10.954     │ = 7,011            │
│ SHRD_ACC        │ 5,120     │ × 0.780      │ = 3,994            │
│ REG_RD          │ 25,600    │ × 0.101      │ = 2,574            │
│ REG_WR          │ 12,800    │ × 0.141      │ = 1,800            │
│ INT_ACC         │ 2,560     │ × 14.988     │ = 38,369           │
│ FP_ACC          │ 1,280     │ × 0.530      │ = 678              │
│ FP_MUL_ACC      │ 7,680     │ × 0.090      │ = 689              │
│ L2_RH           │ 4,800     │ × 1.261      │ = 6,053            │
│ L2_RM           │ 320       │ × 2.395      │ = 766              │
│ MEM_RD          │ 256       │ × 0.026      │ = 6.6              │
│ MEM_WR          │ 128       │ × 0.031      │ = 4.0              │
│ NOC_A           │ 1,920     │ × 32.090     │ = 61,613           │
│ PIPE_A          │ 12,800    │ × 0.514      │ = 6,579            │
└─────────────────┴───────────┴──────────────┴────────────────────┘
```

### 3.4 Step 3: McPAT Base Energy → Component Power

McPAT이 effective count를 사용하여 각 하드웨어 유닛의 에너지를 계산하고, execution time으로 나누어 power(W)로 변환한다.

```
Component_Power(W) = McPAT_Energy(effective_count) / T_execution
```

**가상 결과 (각 component power):**

```
┌──────────────────────────────────────────────────────┐
│         Dynamic Power Breakdown (예시)                │
├────────────┬──────────┬──────────────────────────────┤
│ Component  │ Power(W) │ 설명                          │
├────────────┼──────────┼──────────────────────────────┤
│ IBP        │ 3.2      │ Instruction Buffer            │
│ ICP        │ 2.8      │ I-cache                       │
│ DCP        │ 8.5      │ L1D cache                     │
│ TCP        │ 0.0      │ Texture cache (미사용)        │
│ CCP        │ 0.1      │ Constant cache                │
│ SHRDP      │ 4.2      │ Shared memory                 │
│ RFP        │ 15.8     │ Register file (★ 높음)       │
│ INTP       │ 5.1      │ Integer ALU                   │
│ FPUP       │ 1.8      │ FP32 add/cmp                  │
│ DPUP       │ 0.0      │ FP64 (미사용)                │
│ INT_MULP   │ 0.3      │ Integer multiply              │
│ FP_MULP    │ 12.4     │ FP32 FMA/MUL (★ 주력)       │
│ FP_DIVP    │ 0.0      │ FP div (미사용)              │
│ FP_SQRTP   │ 0.0      │ SFU sqrt (미사용)            │
│ TENSORP    │ 0.0      │ Tensor core (미사용)          │
│ TEXP       │ 0.0      │ Texture unit (미사용)         │
│ SCHEDP     │ 4.5      │ Scheduler                     │
│ L2CP       │ 5.2      │ L2 cache                      │
│ MCP        │ 1.1      │ Memory controller             │
│ NOCP       │ 3.8      │ NoC/Interconnect              │
│ DRAMP      │ 2.4      │ DRAM                          │
│ PIPEP      │ 6.2      │ Pipeline                      │
│ IDLE_COREP │ 0.0      │ Idle SM (0개, 전부 active)    │
├────────────┼──────────┼──────────────────────────────┤
│ Σ Dynamic  │ 77.4 W   │                               │
└────────────┴──────────┴──────────────────────────────┘
```

### 3.5 Step 4: Static Power 계산

코드 `calculate_static_power()` (line 790-918):

```
1) Instruction mix 분류:
   - int_accesses = 2,560 (INT_ACC) + 0 + 0 + 0 + 0 = 2,560  → ≠ 0
   - fp_accesses  = 1,280 (FP_ACC) + 7,680 (FP_MUL) + 0       → ≠ 0
   - dp_accesses  = 0                                           → = 0
   - sfu_accesses = 0                                           → = 0
   - tensor_accesses = 0                                        → = 0
   - tex_accesses = 0                                           → = 0

2) 카테고리 결정:
   INT ≠ 0, FP ≠ 0, DP = 0, SFU = 0, TENSOR = 0, TEX = 0
   → INT_FP 카테고리 (cat2)

3) XML 값 적용:
   base_static_power = static_cat2_flane   = 18.618 W
   lane_static_power = static_cat2_addlane = 0.645 W

4) Thread divergence 반영 (Linear Model):
   avg_threads_per_warp = 32 (full warp)

   total_static_power = base + (threads - 1) × lane
                      = 18.618 + (32 - 1) × 0.645
                      = 18.618 + 20.00
                      = 38.618 W

5) Active core 비율:
   per_active_core = (80 - 0) / 80 = 1.0  (전부 active)

6) 최종 Static Power:
   P_static = 38.618 × 1.0 = 38.6 W
```

### 3.6 Step 5: Constant Power

```
P_const = constant_power = 32.325 W  (XML에서 직접)
```

### 3.7 Step 6: 총 전력 계산

```
┌─────────────────────────────────────────────────────────┐
│                  P_total 최종 계산                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  P_dynamic  = 77.4 W  (22개 component 합)              │
│  P_static   = 38.6 W  (INT_FP cat2, 32 threads)       │
│  P_const    = 32.3 W  (보드, 팬, 주변회로)             │
│  ─────────────────────────                              │
│  P_total    = 148.3 W                                   │
│                                                         │
│  구성비:                                                │
│  ├─ Dynamic: 52.2%                                      │
│  ├─ Static:  26.0%                                      │
│  └─ Const:   21.8%                                      │
│                                                         │
│  주요 Dynamic 기여자:                                    │
│  ├─ Register File (RFP):  15.8W (10.7%)                │
│  ├─ FP_MUL (FP_MULP):    12.4W (8.4%)                  │
│  ├─ L1D Cache (DCP):      8.5W (5.7%)                  │
│  ├─ Pipeline (PIPEP):     6.2W (4.2%)                  │
│  └─ L2 Cache (L2CP):      5.2W (3.5%)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 다른 시나리오 비교

### 4.1 Tensor Core 사용 시 (CUTLASS GEMM)

```
카테고리: INT_FP_TENSOR (cat6)
  base_static_power = static_cat6_flane   = 48.949 W  (★ 매우 높음)
  lane_static_power = static_cat6_addlane = 0.0 W     (추가 lane power 없음)

P_static = 48.949 × 1.0 = 48.9 W

추가 Dynamic:
  TENSORP = ~25-40W (Tensor Core 전력 큼)

→ P_total ≈ 180-220W (V100 TDP 250W에 근접)
```

### 4.2 Idle 상태 (일부 SM만 사용)

```
Active SMs: 8 (80개 중)
Idle SMs: 72

P_idle = idle_core_power × num_idle = 0.283 × 72 = 20.4 W
P_static = 38.6 × (8/80) = 3.9 W  (active SM 비율만큼)

→ P_total = P_dyn(작음) + 3.9 + 20.4 + 32.3 ≈ 60-70W
```

### 4.3 Thread Divergence 있을 때 (y=20 threads)

```
Half-warp Model 적용 (코드에서는 Linear 사용):
  avg_threads_per_warp = 20

Linear Model:
  P_static = 18.618 + (20-1) × 0.645 = 18.618 + 12.255 = 30.9 W

→ Full warp(32) 대비: 30.9W vs 38.6W → 약 20% 감소
→ Divergence가 심할수록 static power 감소 (activity 줄어듦)
```

---

## 5. DVFS 적용 시 전력 변환

코드 `gpgpu_sim_wrapper.cc:1060-1073`:

```cpp
if (g_dvfs_enabled) {
    double voltage_ratio = modeled_chip_voltage / modeled_chip_voltage_ref;
    
    // Static power: V에 비례
    IDLE_COREP *= voltage_ratio;
    STATICP    *= voltage_ratio;
    
    // Dynamic power: V²에 비례
    for (all other components) {
        component *= voltage_ratio × voltage_ratio;  // V²
    }
}
```

### 예시: 클럭 1417MHz → 900MHz 변경

```
V100 기준 V-F 관계: V ≈ kf
  voltage_ratio ≈ 900/1417 = 0.635

Dynamic power 변화:
  P_dyn_new = P_dyn × 0.635² = 77.4 × 0.403 = 31.2 W

Static power 변화:
  P_static_new = P_static × 0.635 = 38.6 × 0.635 = 24.5 W

Constant power (변화 없음):
  P_const = 32.3 W

→ P_total = 31.2 + 24.5 + 32.3 = 88.0 W  (148.3W에서 40% 감소)
```

---

## 6. Quadratic Programming 입출력 예시

### 6.1 입력: accelwattch_volta_sass_sim.csv

102개 microbenchmark × (31 activity columns + 1 measured power) 형태:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ benchmark    │ IBP  │ ICP  │ DCP │...│ INTP │ FP_MULP │...│ CONSTP│ P_meas │
├──────────────┼──────┼──────┼─────┼───┼──────┼─────────┼───┼───────┼────────┤
│ int_add_only │ 3.1  │ 2.5  │ 0.2 │...│ 45.2 │ 0.0     │...│ 32.3  │ 142.5  │
│ fp_mul_only  │ 3.0  │ 2.4  │ 0.1 │...│ 2.1  │ 38.7    │...│ 32.3  │ 155.8  │
│ tensor_only  │ 2.8  │ 2.2  │ 0.3 │...│ 1.5  │ 0.0     │...│ 32.3  │ 198.3  │
│ mem_stress   │ 3.2  │ 2.6  │ 12.4│...│ 5.3  │ 0.0     │...│ 32.3  │ 168.7  │
│ nanosleep    │ 0.5  │ 0.3  │ 0.0 │...│ 0.0  │ 0.0     │...│ 32.3  │ 42.1   │
│ ...          │ ...  │ ...  │ ... │...│ ...  │ ...     │...│ ...   │ ...    │
│ (102 rows)   │      │      │     │   │      │         │   │       │        │
└──────────────┴──────┴──────┴─────┴───┴──────┴─────────┴───┴───────┴────────┘

A = 행렬 (102 × 31)  ← activity factor columns
b = 벡터 (102 × 1)   ← measured power column
```

### 6.2 QP Solver 실행

```matlab
% Eq.(14): X* = argmin ||AX - b||²
%          s.t. constraints

result = quadprog(2*A'*A, -2*A'*b, C, D, [], [], l, u);

% 결과: 31개 scaling factors
```

### 6.3 출력: scaled_coefficients.csv

```
┌────────────────┬─────────────┬──────────────────────────────────┐
│ Component      │ Scaling X_i │ 의미                              │
├────────────────┼─────────────┼──────────────────────────────────┤
│ X_IBP          │ 10.000      │ → XML의 TOT_INST가 됨           │
│ X_ICP          │ 8.593       │ → XML의 IC_H가 됨               │
│ X_DCP          │ 9.835       │ → XML의 DC_RH가 됨              │
│ ...            │ ...         │                                   │
│ X_INTP         │ 14.988      │ → XML의 INT_ACC가 됨            │
│ X_FP_MULP      │ 0.090       │ → XML의 FP_MUL_ACC가 됨        │
│ X_TENSORP      │ 0.815       │ → XML의 TENSOR_ACC가 됨        │
│ ...            │ ...         │                                   │
│ X_IDLE_COREP   │ 1.000       │ (고정, 이미 모델링됨)            │
│ X_CONSTP       │ 1.000       │ (고정, 이미 모델링됨)            │
│ X_STATICP      │ 1.000       │ (고정, 이미 모델링됨)            │
└────────────────┴─────────────┴──────────────────────────────────┘

이 값들이 XML config의 Activity Factor 값으로 기록된다.
```

### 6.4 제약조건이 보장하는 것

```
QP constraints 예시:

X_INT ≤ 1.843 × X_FPU     → INT add가 FPU add보다 에너지 작음 (합리적)
  14.988 ≤ 1.843 × 0.530?  → 14.988 ≤ 0.977? ❌ 
  
  (주의: 이 제약은 "INT component의 power가 FPU의 1.843배를 넘지 않는다"
   가 아니라, McPAT per-instruction 에너지 비율로 scaling factor를 제약하는 것)
  
  실제로는: C(1,8)=1, C(1,9)=-1.843 → X[8] - 1.843×X[9] ≤ 0
  즉 X_INT ≤ 1.843 × X_FPU (scaling factor 비율 제약)

X_FP_MUL ≤ 75.07 × X_TENSOR → Tensor가 FP_MUL보다 에너지 높음
  0.090 ≤ 75.07 × 0.815 = 61.2 ✓
```

---

## 7. Power Report 출력 형식

`accelwattch_power_report.log`에 출력되는 실제 형식:

```
kernel_name=_Z9mysgemmNTPKfiS0_iPfiiff
kernel_launch_uid=1
kernel_total_runtime=35000
kernel_total_sampling_periods=70

Kernel Average Power Data:
kernel_avg_power = 148.3
gpu_avg_IBP, = 3.2
gpu_avg_ICP, = 2.8
gpu_avg_DCP, = 8.5
gpu_avg_TCP, = 0.0
gpu_avg_CCP, = 0.1
gpu_avg_SHRDP, = 4.2
gpu_avg_RFP, = 15.8
gpu_avg_INTP, = 5.1
gpu_avg_FPUP, = 1.8
gpu_avg_DPUP, = 0.0
gpu_avg_INT_MUL24P, = 0.0
gpu_avg_INT_MUL32P, = 0.0
gpu_avg_INT_MULP, = 0.3
gpu_avg_INT_DIVP, = 0.0
gpu_avg_FP_MULP, = 12.4
gpu_avg_FP_DIVP, = 0.0
gpu_avg_FP_SQRTP, = 0.0
gpu_avg_FP_LGP, = 0.0
gpu_avg_FP_SINP, = 0.0
gpu_avg_FP_EXP, = 0.0
gpu_avg_DP_MULP, = 0.0
gpu_avg_DP_DIVP, = 0.0
gpu_avg_TENSORP, = 0.0
gpu_avg_TEXP, = 0.0
gpu_avg_SCHEDP, = 4.5
gpu_avg_L2CP, = 5.2
gpu_avg_MCP, = 1.1
gpu_avg_NOCP, = 3.8
gpu_avg_DRAMP, = 2.4
gpu_avg_PIPEP, = 6.2
gpu_avg_IDLE_COREP, = 0.0
gpu_avg_CONSTP = 32.3
gpu_avg_STATICP = 38.6

Kernel Maximum Power Data:
kernel_max_power = 165.2
...

Kernel Minimum Power Data:
kernel_min_power = 95.4
...
```

---

## 8. 한눈에 보는 전체 흐름도 (수치 포함)

```
[시뮬레이터: 500 cycles 실행]
        │
        ▼
[Activity Counts 수집]
  INT_ACC = 2,560
  FP_MUL_ACC = 7,680
  ...
        │
        ▼
[Scaling Factor 적용]  ← XML config에서 읽음
  INT_ACC × 14.988 = 38,369
  FP_MUL_ACC × 0.090 = 689
  ...
        │
        ▼
[McPAT Energy 계산]   ← 하드웨어 모델 (캐시, ALU 등)
  ÷ executionTime → Power(W)
        │
        ├──→ P_dyn = Σ components = 77.4 W
        │
        ├──→ P_static = calculate_static_power()
        │      INT_FP (cat2): 18.618 + 31×0.645 = 38.6 W
        │      × per_active_core(1.0)
        │
        ├──→ P_const = 32.3 W (XML 직접)
        │
        ▼
[P_total = 77.4 + 38.6 + 32.3 = 148.3 W]
        │
        ▼
[DVFS 보정] (활성화 시)
  Dynamic × V² ratio
  Static × V ratio
        │
        ▼
[Power Report 출력]
  kernel_avg_power = 148.3
  component breakdown...
```

---

> **참고**: 위 수치는 실제 AccelWattch 실행 결과가 아닌, XML config 값과 코드 로직을 기반으로 한 **설명용 예시**입니다. 실제 결과는 benchmark/커널에 따라 다릅니다.
