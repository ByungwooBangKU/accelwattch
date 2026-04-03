# AccelWattch 3-Layer 구조 상세 해설

> **목적**: Layer 1(ISA Opcode) → Layer 2(Performance Config) → Layer 3(Power Model XML)이 어떻게 연결되어 하나의 명령어가 최종 전력으로 변환되는지, 실제 코드와 값을 추적하며 설명  
> **방법**: SASS 명령어 `FFMA`(FP32 Fused Multiply-Add) 하나를 처음부터 끝까지 추적  
> **작성일**: 2026-04-03  

---

## 전체 구조 요약

GPU 전력 모델링에는 3개의 설정 레이어가 필요하다. 각 레이어는 서로 다른 질문에 답한다.

```
Layer 1: ISA Opcode 정의
  "이 명령어가 무엇이고, 어떤 하드웨어 유닛에서 실행되는가?"
  파일: trace_opcode.h, ampere_opcode.h, accelwattch_component_mapping.h

         ↓ 명령어 유형과 전력 컴포넌트 결정

Layer 2: Performance Config
  "이 명령어가 몇 사이클에 실행되고, 몇 개의 유닛이 있는가?"
  파일: gpgpusim.config, trace.config

         ↓ 실행 시간과 activity count 결정

Layer 3: Power Model XML
  "이 컴포넌트의 접근 1회당 에너지는 얼마이고, 보정 계수는 얼마인가?"
  파일: accelwattch_sass_sim.xml

         ↓ 에너지 × activity / 시간 = 전력(W)
```

---

## 실전 예제: FFMA 명령어 1개를 처음부터 끝까지 추적

### 시나리오

CUDA 프로그램에서 다음 코드가 실행된다:

```cuda
// CUDA 소스
float c = a * b + c;   // FMA (Fused Multiply-Add)
```

nvcc가 이를 컴파일하면 SASS 기계어 `FFMA`가 생성된다:

```
// SASS trace 파일의 한 줄
0x0050  ffffffff  1 R6  FFMA  3 R2 R4 R6  0
```

이 명령어가 전력으로 변환되는 과정을 Layer별로 추적한다.

---

## Layer 1: ISA Opcode 정의 — "이 명령어는 무엇인가?"

### Step 1-1: Opcode 번호 할당

`trace_opcode.h`에서 모든 SASS 명령어에 고유 번호를 부여한다.

```cpp
// trace_opcode.h:14-15
enum TraceInstrOpcode {
    OP_FADD = 1,      // FP32 덧셈
    ...
    OP_FFMA = 5,      // FP32 Fused Multiply-Add  ← 이것!
    ...
};
```

`FFMA`는 `OP_FFMA = 5`라는 정수 ID를 받는다.

### Step 1-2: 시뮬레이터 명령어 유형 매핑

`ampere_opcode.h`에서 SASS opcode → 시뮬레이터의 op_type으로 매핑한다.

```cpp
// ampere_opcode.h:24
{"FFMA", OpcodeChar(OP_FFMA, SP_OP)},
//                   ↑ opcode ID    ↑ 명령어 유형
```

| 필드 | 값 | 의미 |
|------|-----|------|
| 문자열 | `"FFMA"` | trace 파일에서 읽은 opcode 문자열 |
| TraceInstrOpcode | `OP_FFMA` | opcode 고유 번호 (=5) |
| op_type | `SP_OP` | **FP32 유닛(SP unit)**에서 실행됨 |

op_type은 시뮬레이터가 이 명령어를 **어떤 파이프라인 유닛에 스케줄링**할지 결정한다.

**op_type 전체 목록과 대응 하드웨어:**

| op_type | 하드웨어 유닛 | config 파라미터 | 예시 opcode |
|---------|-------------|----------------|------------|
| `SP_OP` | FP32 코어 | `gpgpu_num_sp_units` | FADD, FFMA, FMUL |
| `DP_OP` | FP64 코어 | `gpgpu_num_dp_units` | DADD, DFMA, DMUL |
| `INTP_OP` | INT32 코어 | `gpgpu_num_int_units` | IADD3, IMAD, IMUL |
| `SFU_OP` | Special Function Unit | `gpgpu_num_sfu_units` | MUFU (sin, cos, exp, log) |
| `LOAD_OP` | LD/ST 유닛 | 파이프라인 MEM 폭 | LDG, LDS, LD |
| `STORE_OP` | LD/ST 유닛 | 파이프라인 MEM 폭 | STG, STS, ST |
| `ALU_OP` | INT32 코어 (보조) | `gpgpu_num_int_units` | MOV, SEL, F2I |
| `SPECIALIZED_UNIT_1_OP` | 분기 유닛 | `specialized_unit_1` | BRA, EXIT |
| `SPECIALIZED_UNIT_2_OP` | 텍스처 유닛 | `specialized_unit_2` | TEX, TLD |
| `SPECIALIZED_UNIT_3_OP` | **Tensor Core** | `specialized_unit_3` | HMMA, IMMA, DMMA |

### Step 1-3: Power Component 매핑

`accelwattch_component_mapping.h`에서 opcode → 전력 컴포넌트로 매핑한다.

```cpp
// accelwattch_component_mapping.h:44-45
{OP_FFMA, FP_MUL_OP},
//  ↑ opcode    ↑ 전력 컴포넌트
```

| 필드 | 값 | 의미 |
|------|-----|------|
| TraceInstrOpcode | `OP_FFMA` | FFMA 명령어 |
| special_ops | `FP_MUL_OP` | 전력 계산에서 **FP_MULP** 컴포넌트로 집계됨 |

**주의**: `SP_OP`(시뮬레이터 유닛)와 `FP_MUL_OP`(전력 컴포넌트)는 다른 개념이다.
- `SP_OP`는 "어떤 파이프라인에서 실행할지" (성능 시뮬레이션용)
- `FP_MUL_OP`는 "전력 계산에서 어떤 component에 집계할지" (전력 모델용)

같은 `SP_OP`라도:
- FADD → `FP__OP` (FP add power component)
- FFMA → `FP_MUL_OP` (FP multiply power component) — 에너지가 더 높음

**special_ops 전체 목록과 대응 power component:**

| special_ops | Power Component | 의미 |
|-------------|----------------|------|
| `FP__OP` | FPUP | FP32 덧셈/비교 (에너지 낮음) |
| `FP_MUL_OP` | FP_MULP | FP32 곱셈/FMA (에너지 높음) |
| `FP_DIV_OP` | FP_DIVP | FP32 나눗셈 |
| `FP_SQRT_OP` | FP_SQRTP | SFU 제곱근 |
| `FP_LG_OP` | FP_LGP | SFU 로그 |
| `FP_SIN_OP` | FP_SINP | SFU sin/cos |
| `FP_EXP_OP` | FP_EXP | SFU 지수/역수 |
| `INT__OP` | INTP | INT32 덧셈/비교 |
| `INT_MUL_OP` | INT_MULP | INT32 곱셈/MAD |
| `DP___OP` | DPUP | FP64 덧셈/비교 |
| `DP_MUL_OP` | DP_MULP | FP64 곱셈/FMA |
| `TENSOR__OP` | TENSORP | Tensor Core (HMMA/IMMA) |
| `TEX__OP` | TEXP | 텍스처 유닛 |
| `OTHER_OP` | 해당 없음 | 메모리/분기/동기화 (별도 처리) |

### Layer 1 요약: FFMA의 경우

```
FFMA (trace 파일의 문자열)
  → OP_FFMA (opcode ID = 5)
  → SP_OP (FP32 유닛에서 실행)        ← 성능: Layer 2로 전달
  → FP_MUL_OP (FP_MULP component)    ← 전력: Layer 3로 전달
```

---

## Layer 2: Performance Config — "몇 사이클에, 몇 개 유닛에서 실행되는가?"

### Step 2-1: 유닛 수 결정 (gpgpusim.config)

FFMA는 `SP_OP`이므로 FP32(SP) 유닛에서 실행된다:

```bash
# gpgpusim.config (V100 예시)
-gpgpu_num_sp_units 4        # SM당 FP32 유닛 4개 (sub-core model: block당 1개 × 4 blocks)
```

이 값은 시뮬레이터가 **FP32 명령어를 동시에 몇 개나 발행(issue)할 수 있는지** 결정한다. 유닛이 4개이면 매 사이클 최대 4개의 warp에서 FP32 명령어를 동시 실행 가능.

**가상 가속기에서 이 값을 바꾸면?**
- `gpgpu_num_sp_units 2`: FP32 처리량 절반 → 같은 워크로드에 2배 시간 → activity 동일하지만 T_elapsed 증가 → 전력 감소
- `gpgpu_num_sp_units 8`: FP32 처리량 2배 → 시간 절반 → 전력 증가

### Step 2-2: 실행 지연시간 결정 (gpgpusim.config 또는 trace.config)

FFMA의 latency와 initiation interval:

```bash
# gpgpusim.config (PTX 모드)
-ptx_opcode_latency_fp 4,13,4,5,39
#                      ADD,MAX,MUL,MAD,DIV
# → FFMA는 MAD(FMA)에 해당: latency = 5 cycles

-ptx_opcode_initiation_fp 2,2,2,2,4
# → initiation interval = 2 cycles (다음 FMA를 2 cycle 후에 발행 가능)

# trace.config (SASS 모드)
-trace_opcode_latency_initiation_sp 2,2
# → latency = 2 cycles, initiation = 2 cycles (SASS 모드는 더 단순)
```

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| **latency** | 2~5 cycles | 결과가 준비되기까지의 시간. 다음 명령어가 이 결과를 사용하려면 기다려야 함 |
| **initiation interval** | 2 cycles | 같은 유닛에서 다음 명령어를 발행할 수 있는 최소 간격 |

### Step 2-3: 파이프라인 폭 결정 (gpgpusim.config)

```bash
# gpgpusim.config
-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
# 순서: ID_OC_SP, ID_OC_DP, ID_OC_INT, ID_OC_SFU, ID_OC_MEM,
#       OC_EX_SP, OC_EX_DP, OC_EX_INT, OC_EX_SFU, OC_EX_MEM,
#       EX_WB,    ID_OC_TENSOR, OC_EX_TENSOR
```

FFMA는 SP_OP이므로:
- `ID_OC_SP = 4`: Issue → Operand Collect 파이프라인 폭 = 4
- `OC_EX_SP = 4`: Operand Collect → Execute 파이프라인 폭 = 4

즉, 매 사이클 최대 4개 warp의 FFMA가 파이프라인을 통과할 수 있다.

### Step 2-4: 시뮬레이션에서 Activity Count 생성

시뮬레이터가 500 cycles 동안 FFMA를 실행하면:

```
시뮬레이션 결과:
  FP_MUL_ACC (FP multiply accesses) = 7,680회   ← Layer 3에 전달
  총 실행 시간 T = 500 cycles / 1417MHz = 0.353µs ← Layer 3에 전달
```

이 activity count가 Layer 3의 입력이 된다.

### Layer 2 요약: FFMA의 경우

```
gpgpusim.config:
  gpgpu_num_sp_units = 4     → FFMA는 FP32 유닛 4개에서 실행 가능
  pipeline_width SP = 4      → 매 사이클 최대 4 warp 발행
  
trace.config:
  latency = 2, initiation = 2 → 2 cycle마다 1개 FFMA 발행

시뮬레이션 결과:
  FP_MUL_ACC = 7,680회 (500 cycles 동안)
  T_elapsed = 0.353µs
```

---

## Layer 3: Power Model XML — "접근 1회당 에너지는 얼마인가?"

### Step 3-1: Scaling Factor 읽기 (XML)

`accelwattch_sass_sim.xml`에서 FP_MUL 관련 scaling factor를 읽는다:

```xml
<!-- accelwattch_sass_sim.xml -->
<param name="FP_MUL_ACC" value="0.090"/>  ← 이것!
```

이 값 `0.090`은 **scaling factor (xᵢ)**이다. McPAT이 계산한 base energy Êᵢ에 이 계수를 곱하여 보정한다.

### Step 3-2: Scaling Factor 적용 (update_coefficients)

```cpp
// gpgpu_sim_wrapper.cc:update_coefficients()

// 1. Raw activity count 가져오기
initpower_coeff[FP_MUL_ACC] = 7680;  // 시뮬레이터에서 수집

// 2. Scaling factor 곱하기
effpower_coeff[FP_MUL_ACC] = 7680 * 0.090 = 691.2;
//                                   ↑ XML에서 읽은 값

// 3. Execution time으로 나누기
effpower_coeff[FP_MUL_ACC] /= T_elapsed;  // 691.2 / 0.353µs = ...
```

이 `effpower_coeff`가 McPAT에 전달되어 FP MUL 유닛의 실행 활동량으로 사용된다.

### Step 3-3: McPAT이 Base Energy 계산

McPAT은 `technology.cc`의 공정 파라미터(23nm)와 유닛 구성(ALU_per_core 등)을 기반으로 **FP32 곱셈 유닛 1회 접근당 에너지 Êᵢ**를 계산한다:

```
McPAT 내부:
  FP MUL unit area = f(tech_node, ALU_per_core, FPU_per_core)
  Energy per access Ê = f(area, Vdd, capacitance, wire_length)
  
  Total energy = Ê × effpower_coeff[FP_MUL_ACC]
  Power = Total energy / T_elapsed
```

### Step 3-4: Component Power 추출

```cpp
// gpgpu_sim_wrapper.cc:update_components_power()

// FP 유닛의 전체 dynamic energy를 McPAT에서 추출
double sample_fp_pwr = fp_u->rt_power.readOp.dynamic / executionTime;

// FP_MUL의 비율만큼 분배 (FP_ACC + FP_MUL_ACC + FP_DIV_ACC 중)
if (tot_fpu_accesses != 0) {
    sample_cmp_pwr[FP_MULP] = sample_fp_pwr
                             * sample_perf_counters[FP_MUL_ACC]
                             / tot_fpu_accesses;
    // = 전체 FP power × (FP_MUL 접근 수 / 전체 FP 접근 수)
}
```

예시:
```
sample_fp_pwr = 14.2W (FP 유닛 전체)
FP_MUL_ACC = 7680 (FP 곱셈)
FP_ACC = 1280 (FP 덧셈)
tot_fpu_accesses = 7680 + 1280 = 8960

FP_MULP = 14.2 × (7680 / 8960) = 14.2 × 0.857 = 12.2W
```

### Step 3-5: 기타 XML 파라미터의 역할

FFMA 실행 시 FP_MULP 외에도 다른 component가 동시에 활성화된다:

```xml
<!-- 함께 사용되는 파라미터들 -->
<param name="TOT_INST" value="10.0"/>    <!-- 명령어 fetch → IBP 전력 -->
<param name="IC_H" value="8.593"/>       <!-- I-cache hit → ICP 전력 -->
<param name="REG_RD" value="0.101"/>     <!-- FFMA의 소스 레지스터 읽기 → RFP 전력 -->
<param name="REG_WR" value="0.141"/>     <!-- FFMA의 결과 레지스터 쓰기 → RFP 전력 -->
<param name="PIPE_A" value="0.514"/>     <!-- 파이프라인 활성 → PIPEP 전력 -->
```

즉, FFMA 1개 실행에도 IBP + ICP + RFP + FP_MULP + PIPEP + SCHEDP가 모두 관여한다.

### Step 3-6: Static Power와 Constant Power 추가

```xml
<!-- Static Power: instruction mix에 따라 카테고리 결정 -->
<!-- FFMA가 실행 중이면 FP 접근 ≠ 0, INT 접근 ≠ 0 (주소 계산) -->
<!-- → INT_FP 카테고리 (cat2) 선택 -->
<param name="static_cat2_flane" value="18.618"/>
<param name="static_cat2_addlane" value="0.645"/>

<!-- Constant Power -->
<param name="constant_power" value="32.325"/>

<!-- Idle SM Power -->
<param name="idle_core_power" value="0.283"/>
```

### Layer 3 요약: FFMA의 경우

```
XML scaling factor: FP_MUL_ACC = 0.090
  → effpower = 7680 × 0.090 = 691.2 (보정된 activity)
  → McPAT energy model → FP_MULP = 12.2W

동시에 활성화되는 다른 component:
  IBP = 3.2W, ICP = 2.8W, RFP = 15.8W, PIPEP = 6.2W, ...

Static Power (cat2: INT+FP): 38.6W
Constant Power: 32.3W

P_total = Σ(모든 component) + Static + Constant = 148.3W
```

---

## Layer 1 → 2 → 3 전체 데이터 흐름 (FFMA 기준)

```
Layer 1: ISA Opcode 정의
┌─────────────────────────────────────────────────────────┐
│ trace 파일: "FFMA"                                      │
│   → trace_opcode.h: OP_FFMA (= 5)                     │
│   → ampere_opcode.h: {OP_FFMA, SP_OP}                 │
│      ↓                           ↓                      │
│   성능 유닛: SP (FP32)     accelwattch_mapping.h:       │
│      ↓                    {OP_FFMA, FP_MUL_OP}          │
│      ↓                           ↓                      │
└──────┼───────────────────────────┼──────────────────────┘
       ↓                           ↓
Layer 2: Performance Config
┌──────┼───────────────────────────┼──────────────────────┐
│      ↓                           ↓                      │
│ gpgpusim.config:           전력 컴포넌트 결정:          │
│  num_sp_units = 4          FP_MUL_OP → FP_MULP          │
│  pipeline_width = 4              ↓                      │
│  latency = 2 cycles              ↓                      │
│      ↓                           ↓                      │
│ 시뮬레이션 실행                   ↓                      │
│  → FP_MUL_ACC = 7,680회        ↓                      │
│  → T_elapsed = 0.353µs          ↓                      │
└──────┼───────────────────────────┼──────────────────────┘
       ↓                           ↓
Layer 3: Power Model XML
┌──────┼───────────────────────────┼──────────────────────┐
│      ↓                           ↓                      │
│ accelwattch_sass_sim.xml:                               │
│  FP_MUL_ACC scaling = 0.090                             │
│      ↓                                                  │
│ effpower = 7680 × 0.090 = 691.2                        │
│      ↓                                                  │
│ McPAT (23nm model)                                      │
│  → FP MUL energy × 691.2 / T                           │
│  → FP_MULP = 12.2W                                     │
│                                                         │
│ + 다른 components (IBP, ICP, RFP, ...)                  │
│ + STATICP = 38.6W (cat2: INT+FP)                       │
│ + CONSTP = 32.3W                                        │
│ = P_total = 148.3W                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 가상 가속기에서 각 Layer를 변경할 때의 영향

| 변경 대상 | Layer | 파일 | 예시 | 전력 영향 |
|----------|-------|------|------|----------|
| FP64 유닛 제거 | Layer 2 | gpgpusim.config | `gpgpu_num_dp_units 0` | DPUP, DP_MULP = 0W |
| SM 수 변경 | Layer 2 | gpgpusim.config | `gpgpu_n_clusters 216` | idle SM power 변화 |
| FP32 유닛 축소 | Layer 2 | gpgpusim.config | `gpgpu_num_sp_units 2` | 실행 시간 증가 → 전력/성능 변화 |
| Tensor latency 변경 | Layer 2 | trace.config | `latency 8, init 4` | Tensor throughput 변화 |
| Scaling factor 변경 | Layer 3 | XML | `FP_MUL_ACC = 0.050` | FP_MULP power 직접 변화 |
| Static power 변경 | Layer 3 | XML | `static_cat2_flane = 10.0` | STATICP 변화 |
| 공정 노드 변경 | Layer 3 | XML + technology.cc | `core_tech_node = 7` | 모든 base energy 변화 |
| 새 opcode 추가 | Layer 1 | trace_opcode.h + mapping.h | `OP_CUSTOM_INT4` | 새 component 집계 |

---

## 핵심 정리: 3개 Layer의 역할과 파일

| Layer | 질문 | 핵심 파일 | 출력 |
|-------|------|----------|------|
| **Layer 1** | "이 명령어는 무엇인가?" | `trace_opcode.h` (opcode enum), `*_opcode.h` (opcode → op_type), `accelwattch_component_mapping.h` (opcode → power component) | 명령어 유형 + 전력 컴포넌트 |
| **Layer 2** | "몇 사이클, 몇 유닛?" | `gpgpusim.config` (유닛 수, 파이프라인 폭, SM 수, 캐시), `trace.config` (latency, initiation interval) | Activity count + 실행 시간 |
| **Layer 3** | "에너지는 얼마?" | `accelwattch_sass_sim.xml` (scaling factors, static params, constant power), `cacti/technology.cc` (공정 파라미터) | 전력(W) |

**Layer 1은 거의 변경하지 않는다** — 가상 가속기도 PTX/SASS 호환 명령어를 사용한다고 가정.
**Layer 2가 가장 자주 변경된다** — SM 수, 유닛 수, 캐시 크기 등 아키텍처 파라미터.
**Layer 3은 실측 후 보정한다** — QP solver가 scaling factor를 최적화. 공정 변경 시 technology.cc 수정.
