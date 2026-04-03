# CUDA PTX 기반 가상 가속기 Power Modeling 적용 가이드

> **목적**: CUDA PTX를 활용하여 "존재하지 않는 가상 가속기"의 전력을 AccelWattch로 예측하는 구체적 방법을 문서화  
> **전제**: 가상 가속기는 NVIDIA GPU와 유사한 SIMT 구조이나 상세 사양이 다름 (SM 수, 코어 구성, 캐시 크기, 공정 노드 등)  
> **핵심 가정**: 가상 가속기도 PTX(또는 PTX 호환 IR)를 실행할 수 있다고 가정  
> **작성일**: 2026-04-03  

---

## 1. Accel-Sim의 두 가지 시뮬레이션 모드

Accel-Sim은 GPU 워크로드를 시뮬레이션하는 두 가지 경로를 제공한다.

```
경로 1: SASS Trace-Driven (실제 GPU에서 추출)
────────────────────────────────────────────
CUDA 프로그램
  ↓ nvcc (compute capability 지정)
SASS 바이너리 (실제 기계어)
  ↓ NVBit (GPU 위에서 실행하며 trace 추출)
SASS Trace 파일 (*.traceg)
  ↓ Accel-Sim (trace-driven mode)
성능 시뮬레이션 → AccelWattch 전력 계산

※ 실제 NVIDIA GPU가 있어야 trace를 추출할 수 있음
※ MAPE ~9.2% (가장 정확)

경로 2: PTX Functional-Driven (GPU 없이 가능)
────────────────────────────────────────────
CUDA 프로그램
  ↓ nvcc -ptx (또는 nvcc --keep)
PTX 파일 (.ptx)
  ↓ GPGPU-Sim 내장 PTX 에뮬레이터
PTX 명령어를 해석하여 functional 시뮬레이션
  ↓ Accel-Sim (emulation mode)
성능 시뮬레이션 → AccelWattch 전력 계산

※ 실제 GPU 없이도 가능
※ MAPE ~13.7% (SASS보다 4~5%p 낮은 정확도)
※ 가상 가속기에 적합한 경로
```

**가상 가속기 설계에서는 경로 2(PTX)를 사용한다.** 이유: 가상 가속기의 SASS 기계어는 존재하지 않으므로 NVBit trace를 추출할 수 없다. PTX는 NVIDIA의 가상 ISA로서, 특정 GPU 아키텍처에 종속되지 않는 중간 표현이다.

---

## 2. Trace 파일 형식 이해

### 2.1 SASS Trace 파일 구조

NVBit이 생성하는 SASS trace 파일은 다음 형식이다 (`trace_parser.cc:135-224` 기반):

```
# Kernel Header (kernelslist.g에서 참조)
-kernel name = _Z9mysgemmNTPKfiS0_iPfiiff
-kernel id = 1
-grid dim = (128,1,1)
-block dim = (256,1,1)
-shmem = 0
-nregs = 32
-binary version = 80          ← SM 버전 (A100 = 80, V100 = 70)
-cuda stream id = 0
-accelsim tracer version = 4

# 명령어 Trace (warp 단위)
# PC         mask     dst_regs  opcode     src_regs  mem_width  ...
0x0010       ffffffff 1 R4      IMAD.MOV   2 R1 RZ   0
0x0020       ffffffff 1 R5      S2R        0         0
0x0030       ffffffff 1 R2      IADD3      3 R4 R5 RZ 0
0x0040       ffffffff 1 R6      LDG.E.SYS 1 R2       4 1 0x7f8a00000000 16
```

각 필드의 의미:

| 필드 | 의미 | 예시 |
|------|------|------|
| PC | 프로그램 카운터 (명령어 주소) | 0x0010 |
| mask | 활성 thread mask (32-bit hex) | ffffffff = 32 threads 모두 active |
| dst_regs | 목적지 레지스터 수 + 레지스터 이름 | 1 R4 = 1개, R4에 저장 |
| opcode | SASS 명령어 이름 (modifier 포함) | IMAD.MOV, LDG.E.SYS |
| src_regs | 소스 레지스터 수 + 레지스터 이름 | 2 R1 RZ = 2개, R1과 RZ |
| mem_width | 메모리 접근 폭 (0이면 비메모리 명령) | 4 = 4 bytes |
| address_mode | 주소 인코딩 방식 (0=list, 1=base_stride, 2=base_delta) | 1 |
| addresses | 메모리 주소 (mode에 따라 형식 다름) | 0x7f8a00000000 16 |

### 2.2 binary version의 역할

`binary version`은 Accel-Sim이 **어떤 opcode 매핑 테이블을 사용할지** 결정하는 핵심 값이다:

```cpp
// trace_driven.cc:104-122
if (binary_verion == 80 || binary_verion == 86)
    OpcodeMap = &Ampere_OpcodeMap;     // ampere_opcode.h
else if (binary_verion == 70)
    OpcodeMap = &Volta_OpcodeMap;      // volta_opcode.h
else if (binary_verion == 75)
    OpcodeMap = &Turing_OpcodeMap;     // turing_opcode.h
else
    exit(0);  // 미지원 → 종료
```

### 2.3 Opcode → 시뮬레이터 명령어 변환

trace의 SASS opcode는 시뮬레이터 내부에서 다음과 같이 변환된다:

```
SASS Opcode (trace 파일)
    ↓ OpcodeMap 조회 (예: Ampere_OpcodeMap)
op_type (명령어 유형: SP_OP, DP_OP, SFU_OP, ...)
    ↓ accelwattch_component_mapping.h의 OpcodePowerMap 조회
special_ops (전력 컴포넌트: FP__OP, INT__OP, TENSOR__OP, ...)
    ↓ 시뮬레이션 실행
activity counter 증가 (FP_ACC++, INT_ACC++, TENSOR_ACC++ 등)
    ↓ AccelWattch
component별 전력 계산
```

---

## 3. 가상 가속기에 PTX를 적용하는 구체적 방법

### 3.1 전체 워크플로우

```
Step 1: CUDA 프로그램 작성 (워크로드)
    ↓
Step 2: nvcc -ptx로 PTX 생성
    ↓
Step 3: 가상 가속기의 gpgpusim.config 작성
    ↓
Step 4: 가상 가속기의 accelwattch XML 작성
    ↓
Step 5: Accel-Sim PTX 모드로 시뮬레이션
    ↓
Step 6: AccelWattch 전력 계산
    ↓
Step 7: Power report 분석
```

### 3.2 Step 1-2: PTX 생성

CUDA 프로그램에서 PTX를 생성하는 방법:

```bash
# 방법 1: PTX 파일만 생성
nvcc -ptx -o matmul.ptx matmul.cu

# 방법 2: 바이너리와 PTX 모두 생성 (PTX 보존)
nvcc --keep -o matmul matmul.cu
# → matmul.ptx 파일이 중간 결과로 남음

# 방법 3: 특정 compute capability 지정
nvcc -ptx -arch=compute_80 -o matmul.ptx matmul.cu
```

생성된 PTX 파일 예시:

```
.version 7.5
.target sm_80
.address_size 64

.visible .entry _Z6matmulPfS_S_i(
    .param .u64 _Z6matmulPfS_S_i_param_0,
    .param .u64 _Z6matmulPfS_S_i_param_1,
    .param .u64 _Z6matmulPfS_S_i_param_2,
    .param .u32 _Z6matmulPfS_S_i_param_3
)
{
    .reg .f32   %f<5>;
    .reg .b32   %r<10>;
    .reg .b64   %rd<8>;

    ld.param.u64    %rd1, [_Z6matmulPfS_S_i_param_0];
    ld.param.u64    %rd2, [_Z6matmulPfS_S_i_param_1];
    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mad.lo.s32      %r3, %r2, 256, %r1;
    ...
    fma.rn.f32      %f3, %f1, %f2, %f3;   // ← 이것이 FP_MUL power component로 매핑
    ...
}
```

### 3.3 Step 3: gpgpusim.config 작성 (가상 가속기 정의)

가상 가속기의 하드웨어 구조를 정의하는 핵심 파일이다. A100 config을 템플릿으로 수정한다.

```bash
# 추론 전용 가속기 (Accel-B) 예시
# 원본: SM80_A100/gpgpusim.config

# ─── 아키텍처 규모 ───
-gpgpu_n_clusters 216           # SM 수: 108 → 216 (소형 SM 2배)
-gpgpu_n_cores_per_cluster 1
-gpgpu_n_mem 20                 # 메모리 파티션: 40 → 20 (용량 축소)
-gpgpu_n_sub_partition_per_mchannel 4

# ─── 클럭 ───
-gpgpu_clock_domains 1200:1200:1200:1593
#                    Core:Icnt:L2  :DRAM

# ─── SM 내부 구성 ───
-gpgpu_shader_registers 32768   # 레지스터: 65536 → 32768 (절반)
-gpgpu_shader_core_pipeline 1024:32   # 워프 스케줄러 용량 축소
-gpgpu_pipeline_widths 2,0,2,1,2,2,0,2,1,2,4,2,2
#                      SP,DP,INT,SFU,MEM,...,TENSOR
# DP = 0: FP64 유닛 완전 제거!
# SFU = 1: SFU 축소 (4→1)
-gpgpu_num_sp_units 2           # FP32: 4 → 2 (축소)
-gpgpu_num_sfu_units 1          # SFU: 4 → 1 (축소)
-gpgpu_num_dp_units 0           # FP64: 4 → 0 (완전 제거!)
-gpgpu_num_int_units 2          # INT32: 4 → 2 (축소)
-gpgpu_num_tensor_core_units 4  # Tensor Core: 유지 (INT8 강화)

# ─── 캐시 ───
-gpgpu_unified_l1d_size 96      # L1D: 192KB → 96KB (축소)
-gpgpu_shmem_size 49152         # ShMem: 164KB → 48KB (축소)
-gpgpu_cache:dl2 S:128:128:16,L:B:m:L:X,A:192:4,32:0,32  # L2 유지

# ─── Tensor Core latency (INT8 최적화) ───
-ptx_opcode_latency_tesnor 8    # Tensor latency: 25 → 8 (INT8 고속화)
-ptx_opcode_initiation_tensor 4 # Tensor initiation: 16 → 4

# ─── 전력 시뮬레이션 활성화 ───
-power_simulation_enabled 1
-power_simulation_mode 0        # 0 = 시뮬레이션 기반 (PTX SIM)
```

**가정**: PTX 모드에서 `gpgpu_num_dp_units 0`으로 설정하면, PTX의 `add.f64` 같은 FP64 명령어는 시뮬레이터가 FP32 유닛에서 에뮬레이션하거나, 실행 불가로 처리한다. 추론 전용 가속기에서는 FP64 워크로드를 실행하지 않는다고 가정하므로 문제없다.

### 3.4 Step 4: AccelWattch XML 작성 (전력 모델 파라미터)

```xml
<?xml version="1.0" ?>
<component id="root" name="root">
<component id="system" name="system">

<!-- ═══ Dynamic Power Activity Factors (추론 전용: 축소된 component) ═══ -->

<!-- 유지: 캐시/메모리 -->
<param name="TOT_INST" value="10.0"/>
<param name="IC_H" value="8.5"/>
<param name="DC_RH" value="9.8"/>
<param name="SHRD_ACC" value="0.5"/>      <!-- 축소된 ShMem -->
<param name="REG_RD" value="0.08"/>       <!-- 축소된 레지스터 -->
<param name="REG_WR" value="0.10"/>

<!-- 유지: 연산 유닛 (축소) -->
<param name="INT_ACC" value="10.0"/>      <!-- INT32 축소 반영 -->
<param name="FP_ACC" value="0.3"/>        <!-- FP32 축소 반영 -->

<!-- 제거: FP64 관련 → 0 으로 설정 -->
<param name="DP_ACC" value="0.0"/>        <!-- FP64 제거! -->
<param name="DP_MUL_ACC" value="0.0"/>    <!-- FP64 제거! -->

<!-- 유지/강화: Tensor Core (INT8) -->
<param name="TENSOR_ACC" value="1.5"/>    <!-- INT8 Tensor 강화 -->

<!-- 제거: SFU 대부분 -->
<param name="FP_SQRT_ACC" value="0.0"/>   <!-- SFU 제거 -->
<param name="FP_LG_ACC" value="0.0"/>     <!-- SFU 제거 -->
<param name="FP_SIN_ACC" value="0.0"/>    <!-- SFU 제거 -->
<param name="FP_EXP_ACC" value="0.1"/>    <!-- EXP만 최소 유지 (Softmax) -->

<!-- ═══ Static & Constant Power (추론 전용: 대폭 감소) ═══ -->

<param name="constant_power" value="15.0"/>       <!-- 200W TDP, 단순 보드 -->
<param name="idle_core_power" value="0.10"/>       <!-- 소형 SM, 낮은 leakage -->

<!-- Static 카테고리 (축소: 4개만) -->
<param name="static_cat1_flane" value="6.0"/>       <!-- INT only -->
<param name="static_cat1_addlane" value="0.15"/>
<param name="static_cat2_flane" value="8.0"/>       <!-- INT+FP16 -->
<param name="static_cat2_addlane" value="0.20"/>
<param name="static_cat6_flane" value="12.0"/>      <!-- INT+INT8_TENSOR -->
<param name="static_cat6_addlane" value="0.0"/>
<param name="static_light_flane" value="1.0"/>      <!-- LIGHT_SM -->
<param name="static_light_addlane" value="0.001"/>

<!-- ═══ Architecture Parameters ═══ -->
<param name="core_tech_node" value="7"/>            <!-- 또는 2nm이면 수정 필요 -->
<param name="ALU_per_core" value="16"/>             <!-- 32 → 16 (축소) -->
<param name="FPU_per_core" value="16"/>             <!-- 32 → 16 (축소) -->
<!-- ... 나머지 McPAT 파라미터 ... -->

</component>
</component>
```

### 3.5 Step 5-7: 시뮬레이션 실행

```bash
# 1. PTX 모드로 Accel-Sim 실행
cd accelwattch/gpu-simulator
./bin/release/accel-sim.out \
    -config ./configs/virtual-accel-B/gpgpusim.config \
    -trace <path_to_ptx_workloads>/kernelslist.g

# 2. Power report 수집
cd ../util/accelwattch
./collect_power_reports.sh

# 3. CSV 변환 (gen_sim_power_csv.py 수정 필요)
python gen_sim_power_csv.py accel_b_ptx_sim

# 4. MAPE 계산 (HW 측정값 대신 A100 실측을 참조값으로 사용)
cd ../plotting
python plot-correlation.py
```

---

## 4. 핵심 가정과 그 타당성

가상 가속기에 PTX 기반 시뮬레이션을 적용할 때, 다음 가정이 필요하다.

### 가정 1: 가상 가속기가 PTX를 실행할 수 있다

| 항목 | 설명 |
|------|------|
| **가정 내용** | 가상 가속기가 PTX(또는 호환 IR)를 기계어로 변환하여 실행할 수 있다 |
| **타당성** | PTX는 virtual ISA로 설계되었으며, 특정 하드웨어에 종속되지 않음. GPGPU-Sim이 PTX를 직접 해석/에뮬레이션하므로 실제 하드웨어 없이도 동작함 |
| **한계** | 가상 가속기의 고유 명령어(NVIDIA에 없는 연산)는 PTX로 표현 불가. 예를 들어 가상 가속기에 "4-bit 정수 MAC" 전용 명령어가 있다면 PTX에 대응하는 opcode가 없음 |
| **대응** | 기존 PTX 명령어 조합으로 근사하거나, 커스텀 PTX 확장(pseudo-opcode)을 Accel-Sim에 추가 |

### 가정 2: gpgpusim.config의 파라미터 변경이 실제 하드웨어 변경과 동등하다

| 항목 | 설명 |
|------|------|
| **가정 내용** | config에서 `gpgpu_num_dp_units 0`으로 설정하면 FP64 유닛이 없는 칩과 동등하다 |
| **타당성** | Accel-Sim은 config의 유닛 수에 따라 파이프라인을 구성하므로, 유닛이 0이면 해당 명령어는 실행 큐에서 대기(stall)하거나 다른 유닛으로 우회됨. 이는 실제 칩에서 해당 유닛이 없을 때의 동작을 근사함 |
| **한계** | 실제 칩 설계에서는 유닛 제거 시 면적, 배선, 전력 분배가 변하는데, Accel-Sim은 이를 반영하지 않음. 따라서 "유닛을 제거했을 때의 면적 절감 효과"는 별도로 계산해야 함 |
| **대응** | AccelWattch XML에서 해당 component의 scaling factor를 0으로 설정하여 전력 기여를 제거. Static power의 firstLane/addLane도 줄인 값으로 설정 |

### 가정 3: PTX 기반 activity factor가 SASS 기반과 비례한다

| 항목 | 설명 |
|------|------|
| **가정 내용** | PTX에서 수집한 activity factor(명령어 수, 캐시 접근 수 등)가 실제 기계어 실행과 비례 관계에 있다 |
| **타당성** | AccelWattch 논문에서 PTX SIM이 MAPE 13.7%를 달성했으므로, 비례 관계가 대략적으로 성립함. 그러나 PTX → 기계어 변환 시 명령어 합성/분리, 레지스터 할당 변화가 있어 정확한 1:1 대응은 아님 |
| **한계** | PTX의 `mad.lo.s32`가 기계어에서 `IMAD`로 1:1 매핑될 수도 있고, 여러 명령어로 분리될 수도 있음. 이로 인해 activity count가 ±20% 오차를 가질 수 있음 |
| **대응** | QP solver의 scaling factor가 이 차이를 흡수함. 추가로, ML residual correction을 적용하면 오차를 더 줄일 수 있음 |

### 가정 4: 공정 노드 차이를 technology scaling으로 보정할 수 있다

| 항목 | 설명 |
|------|------|
| **가정 내용** | CACTI의 23nm 기반 base energy를 7nm/2nm으로 scaling할 수 있다 |
| **타당성** | 23nm → 7nm (3.3배 차이): QP solver로 보정 가능 (기존 AccelWattch 방식) |
| **한계** | 23nm → 2nm (11.5배 차이): 트랜지스터 구조 자체가 FinFET→GAA로 변경되므로 단순 비례 불가. `cacti/technology.cc`에 2nm 파라미터를 직접 추가해야 함 |
| **대응** | Phase 2에서 IRDS 2024 데이터 기반으로 7nm/5nm/3nm/2nm 블록을 `technology.cc`에 추가 |

### 가정 5: 가상 가속기의 캐시/메모리 구조가 NVIDIA와 유사하다

| 항목 | 설명 |
|------|------|
| **가정 내용** | L1/L2/Shared Memory/Register File의 구조가 NVIDIA GPU와 유사한 계층 구조이다 |
| **타당성** | 대부분의 SIMT 가속기(AMD, Intel, 커스텀)가 유사한 메모리 계층을 가짐. Accel-Sim의 CACTI 기반 캐시 에너지 모델이 범용적임 |
| **한계** | scratchpad-only 아키텍처(캐시 없음), HBM 대신 GDDR 사용 등 근본적 차이가 있으면 메모리 power model 재설계 필요 |
| **대응** | gpgpusim.config에서 캐시 크기/associativity/bank 수를 변경하면 Accel-Sim이 자동 반영. 메모리 타입 변경은 DRAM timing 파라미터 수정으로 대응 |

---

## 5. PTX와 SASS의 Power Modeling 차이

### 5.1 명령어 매핑 차이

```
PTX (가상 ISA)                    SASS (기계어)
──────────────                    ──────────────
mad.lo.s32 %r3, %r1, %r2, %r3   IMAD R3, R1, R2, R3
→ 1:1 매핑 (단순한 경우)

fma.rn.f32 %f3, %f1, %f2, %f3   FFMA R3, R1, R2, R3
→ 1:1 매핑 (단순한 경우)

ld.global.f32 %f1, [%rd1]        LDG.E.SYS R1, [R2]
→ modifier가 다름 (PTX의 .global → SASS의 .E.SYS)

setp.lt.f32 %p1, %f1, %f2        FSETP.LT.AND P0, PT, R1, R2, PT
→ PTX는 단순, SASS는 predicate 합성이 추가됨

PTX에 없는 SASS 명령어:
  WARPSYNC, BSSY, BSYNC  → warp 동기화 (컴파일러가 삽입)
  DEPBAR, YIELD           → 파이프라인 제어
  S2R, CS2R               → 특수 레지스터 읽기
```

### 5.2 gen_sim_power_csv.py의 PTX 모드 처리

PTX 모드에서는 일부 power counter가 제거/합산된다:

```python
# gen_sim_power_csv.py:281-288
if config == "volta_ptx_sim" or ...:
    for each in ["MCP,", "NOCP,"]:
        del power_dict[benchmark_idx][each]
        # PTX 모드에서는 MC와 NoC를 별도로 추적할 수 없음
        # → DRAMP에 MCP가, L2CP에 NOCP가 합산됨
```

PTX SIM에서는 SASS SIM과 달리 **TCP(Texture Cache)도 유지**된다. 이는 PTX에 `tex` 명령어가 있어 텍스처 접근을 명시적으로 추적할 수 있기 때문이다.

### 5.3 Power Component 매핑 비교

| PTX 명령어 | SASS 대응 | Power Component | 비고 |
|-----------|----------|----------------|------|
| `add.f32` | FADD | FP__OP (FPUP) | 1:1 |
| `mul.f32` | FMUL | FP_MUL_OP (FP_MULP) | 1:1 |
| `fma.f32` | FFMA | FP_MUL_OP (FP_MULP) | 1:1 |
| `add.s32` | IADD3 | INT__OP (INTP) | PTX는 IADD, SASS는 IADD3 |
| `mad.lo.s32` | IMAD | INT_MUL_OP (INT_MULP) | 1:1 |
| `add.f64` | DADD | DP___OP (DPUP) | 추론 전용에서는 실행 안 됨 |
| `ld.global` | LDG | OTHER_OP | 메모리 |
| `st.global` | STG | OTHER_OP | 메모리 |
| `sin.approx.f32` | MUFU.SIN | FP_SIN_OP | SFU |
| `wmma.mma` | HMMA/IMMA | TENSOR__OP | Tensor Core |

---

## 6. 가상 가속기에 고유한 연산이 있을 때

가상 가속기에 NVIDIA PTX에 없는 고유 연산이 있다면, 두 가지 접근이 가능하다.

### 방법 A: 기존 PTX 명령어 조합으로 근사

```
가상 가속기 고유 연산: INT4_DOT_PRODUCT (4-bit 정수 내적)
PTX로 근사:
  // INT4는 PTX에 없으므로 INT8로 근사
  wmma.mma.sync.aligned.row.col.m8n8k32.s32.s4.s4
  // 또는 일반 정수 연산으로 분해
  mul.lo.s32 → add.s32 → add.s32 → ...
```

이 경우 activity count가 실제 전용 명령어보다 높게 나오므로(여러 명령어로 분해), power가 과대 추정될 수 있다. 이를 보정하기 위해 해당 component의 **scaling factor를 낮게 설정**한다.

### 방법 B: 커스텀 opcode를 Accel-Sim에 추가

```cpp
// 1. trace_opcode.h에 커스텀 opcode 추가
enum TraceInstrOpcode {
    ...
    OP_CUSTOM_INT4_DOT,    // 가상 가속기 전용
    SASS_NUM_OPCODES
};

// 2. accelwattch_component_mapping.h에 매핑 추가
{OP_CUSTOM_INT4_DOT, TENSOR__OP},  // INT4 내적 → Tensor component

// 3. 가상 trace 파일에서 이 opcode 사용
0x0050  ffffffff  1 R10  CUSTOM_INT4_DOT  2 R1 R2  0
```

이 방법은 더 정확하지만, Accel-Sim 코드를 직접 수정해야 한다.

---

## 7. 정확도에 대한 현실적 기대

| 시나리오 | 기대 MAPE | 근거 |
|---------|----------|------|
| A100 SASS SIM (실제 GPU, 실제 trace) | 9~12% | AccelWattch 논문 수준 |
| A100 PTX SIM (실제 GPU, PTX emulation) | 13~16% | 논문의 PTX SIM 수준 |
| **가상 가속기 PTX SIM (config만 변경)** | **15~25%** | PTX 오차 + config 미보정 + 공정 차이 |
| 가상 가속기 PTX SIM + QP 보정 | **12~18%** | A100 실측 기반 QP 보정 적용 |
| 가상 가속기 PTX SIM + QP + ML residual | **10~15%** | ML이 비선형 효과 보정 |

**15~25% MAPE는 Design Space Exploration에 충분하다.** 그 이유:

1. 칩 설계 초기 단계에서는 "이 설계가 200W인가 300W인가"를 파악하는 것이 중요하지, "정확히 237W인가 241W인가"는 중요하지 않다.
2. **상대 비교**는 절대 정확도보다 신뢰할 수 있다. "Accel-A가 Accel-B보다 30% 전력이 높다"는 결론은 MAPE 20%에서도 유효하다.
3. AccelWattch 논문도 Volta→Pascal/Turing 전이에서 11~13% MAPE를 보였으며, 이를 "reliable design space exploration"이라고 평가했다.

---

> **이전 문서**: [07_Research_Roadmap_8months.md](07_Research_Roadmap_8months.md) — 8개월 로드맵  
> **관련 문서**: [02_Improvement_Points.md Section 12](02_Improvement_Points.md) — 추론 전용 Component 최적화  
> **관련 문서**: [06_GPU_Feature_Support_Analysis.md](06_GPU_Feature_Support_Analysis.md) — 기호/약어 정의
