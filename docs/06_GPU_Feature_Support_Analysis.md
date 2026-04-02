# AccelWattch GPU 기능 지원 분석 및 Call Graph

> **목적**: AccelWattch가 A100/H100 등 최신 GPU의 주요 기능을 지원할 수 있는지 코드 수준에서 검증하고, XML→시뮬레이션→MAPE까지의 전체 Call Graph를 문서화  
> **검증 방법**: 소스코드(`ISA_Def/`, `trace_driven.cc`, `gpgpu_sim_wrapper.cc`, `XML_Parse.h`) 직접 확인  
> **참고**: Accel-Sim v1.3.0은 H100(SM90)의 **성능 시뮬레이션**을 지원하기 시작했으나, AccelWattch의 **전력 모델링**은 별도 확장이 필요  
> **작성일**: 2026-04-02  

---

## 기호 및 약어 정의

본 문서와 이전 문서(01~05)에서 사용하는 모든 기호를 정리한다.

### 수식 기호

| 기호 | 의미 | 단위 | 사용 수식 |
|------|------|------|----------|
| P_total | GPU 총 소비 전력 | W (와트) | Eq.(1),(2),(3),(10) |
| P_proc,dyn | GPU 칩 dynamic(활성) 전력 | W | Eq.(1) |
| P_mem,dyn | 메모리 dynamic 전력 | W | Eq.(1) |
| P_proc,static | GPU 칩 static(누설) 전력 | W | Eq.(1) |
| P_mem,static | 메모리 static 전력 | W | Eq.(1) |
| P_const | 상수 전력 (보드, 팬, 전압조정기) | W | Eq.(1),(2),(3) |
| P_dyn | 전체 dynamic 전력 합 | W | Eq.(10),(11) |
| C | 게이트 커패시턴스 (gate capacitance) | F (패럿) | Eq.(2),(3) |
| V | 공급 전압 (supply voltage) | V (볼트) | Eq.(2) |
| f | 클럭 주파수 | Hz | Eq.(2),(3) |
| f_sm | SM(Streaming Multiprocessor) 클럭 주파수 | MHz | Eq.(3) A100 확장 |
| β (베타) | dynamic power의 기술 계수 | W·s³ | Eq.(3) |
| τ (타우) | static power의 주파수 비례 계수 | W·s | Eq.(3) |
| k | 활성(active) SM 수 | 개 | Eq.(10),(12) |
| y | warp 내 활성 thread 수 (1~32) | 개 | Eq.(4),(5) |
| N | 마이크로아키텍처 컴포넌트 수 | 개 | Eq.(11) |
| M | microbenchmark(워크로드) 수 | 개 | Eq.(13) |
| aᵢ | 컴포넌트 i의 activity factor (접근 횟수) | 회 | Eq.(11),(12) |
| Eᵢ | 컴포넌트 i의 접근당 에너지 | J (줄) | Eq.(11) |
| Êᵢ | 컴포넌트 i의 접근당 에너지 **초기 추정치** (McPAT) | J | Eq.(12) |
| xᵢ | 컴포넌트 i의 scaling factor (보정 계수) | 무차원 | Eq.(12),(14) |
| X | scaling factor 벡터 (xᵢ들의 모음) | 벡터 | Eq.(13),(14) |
| X* | 최적 scaling factor 벡터 | 벡터 | Eq.(14) |
| T_elapsed | 실행 시간 | s (초) | Eq.(11),(12) |
| P_est | 추정 전력 (모델 예측값) | W | Eq.(12),(13) |
| P_meas | 실측 전력 (하드웨어 측정값) | W | Eq.(13) |
| P_static,firstLane | 첫 lane 활성화 시 static power | W | Eq.(4),(5) |
| P_static,addLane | 추가 lane당 static power 증분 | W | Eq.(4),(5) |
| P_static,yLanes | y개 lane 활성 시 static power | W | Eq.(4),(5),(10) |
| P_perIdleSM | idle SM 1개당 전력 소비 | W | Eq.(8),(10) |
| P_dyn+static,perActiveSM | active SM 1개당 dynamic+static 전력 | W | Eq.(6) |
| MAPE | 평균 절대 백분율 오차 (Mean Absolute Percentage Error) | % | 검증 |
| r | 피어슨 상관계수 (Pearson correlation coefficient) | 무차원 | 검증 |

### Power Component 약어

| 약어 | 정식 명칭 | 하드웨어 유닛 |
|------|----------|-------------|
| IBP | Instruction Buffer Power | 명령어 버퍼, 디코더 |
| ICP | Instruction Cache Power | L0 명령어 캐시 |
| DCP | Data Cache Power | L1 데이터 캐시 |
| TCP | Texture Cache Power | 텍스처 캐시 |
| CCP | Constant Cache Power | 상수 캐시 |
| SHRDP | Shared Memory Power | 공유 메모리 |
| RFP | Register File Power | 레지스터 파일 |
| INTP | Integer ALU Power | INT32 정수 연산 유닛 |
| FPUP | Floating Point Unit Power | FP32 부동소수점 유닛 (add/cmp) |
| DPUP | Double Precision Unit Power | FP64 배정밀도 유닛 (add/cmp) |
| INT_MUL24P | Integer 24-bit Multiply Power | 24비트 정수 곱셈 |
| INT_MUL32P | Integer 32-bit Multiply Power | 32비트 정수 곱셈 |
| INT_MULP | Integer Multiply Power | 정수 곱셈 (일반) |
| INT_DIVP | Integer Divide Power | 정수 나눗셈 |
| FP_MULP | FP Multiply Power | FP32 곱셈/FMA |
| FP_DIVP | FP Divide Power | FP32 나눗셈 |
| FP_SQRTP | FP Square Root Power | SFU 제곱근 |
| FP_LGP | FP Logarithm Power | SFU 로그 |
| FP_SINP | FP Sine/Cosine Power | SFU 삼각함수 |
| FP_EXP | FP Exponent Power | SFU 지수 |
| DP_MULP | DP Multiply Power | FP64 곱셈/FMA |
| DP_DIVP | DP Divide Power | FP64 나눗셈 |
| TENSORP | Tensor Core Power | 텐서 코어 |
| TEXP | Texture Unit Power | 텍스처 유닛 |
| SCHEDP | Scheduler Power | 워프 스케줄러 |
| L2CP | L2 Cache Power | L2 캐시 (NOCP와 결합됨) |
| MCP | Memory Controller Power | 메모리 컨트롤러 (DRAMP와 결합됨) |
| NOCP | Network-on-Chip Power | 인터커넥트/NoC |
| DRAMP | DRAM Power | DRAM 메모리 |
| PIPEP | Pipeline Power | SM 파이프라인 |
| IDLE_COREP | Idle Core Power | 비활성 SM의 leakage |
| CONSTP | Constant Power | 상수 전력 (팬, 보드) |
| STATICP | Static Power | 정적 전력 (활성 SM의 leakage) |

### 아키텍처 약어

| 약어 | 의미 |
|------|------|
| SM | Streaming Multiprocessor (GPU 연산 단위) |
| SM70 | Volta 아키텍처 (Compute Capability 7.0) |
| SM75 | Turing 아키텍처 (Compute Capability 7.5) |
| SM80 | Ampere 아키텍처 - A100 (Compute Capability 8.0) |
| SM86 | Ampere 아키텍처 - RTX 3070 등 (Compute Capability 8.6) |
| SM89 | Ada Lovelace 아키텍처 (Compute Capability 8.9) |
| SM90 | Hopper 아키텍처 - H100 (Compute Capability 9.0) |
| SFU | Special Function Unit (sin, cos, log, exp 등) |
| FMA | Fused Multiply-Add (곱셈-덧셈 융합 연산) |
| MMA | Matrix Multiply-Accumulate (행렬 곱셈-누적) |
| TF32 | TensorFloat-32 (19bit: FP32 범위 + FP16 정밀도) |
| BF16 | Brain Floating Point 16 (16bit: FP32 범위 + 8bit 가수) |
| FP8 | 8-bit Floating Point (E4M3 또는 E5M2 포맷) |
| HBM | High Bandwidth Memory |
| TMA | Tensor Memory Accelerator (H100 전용) |
| DPX | Dynamic Programming eXtension (H100 전용) |
| MIG | Multi-Instance GPU (GPU 분할 기술) |
| NVBit | NVIDIA Binary Instrumentation Tool |
| NVML | NVIDIA Management Library |

---

## 1. GPU 아키텍처별 기능 지원 현황

### 1.1 AccelWattch 코드의 아키텍처 지원 범위

AccelWattch가 GPU를 지원하려면 3가지 레이어가 모두 갖추어져야 한다:

```
Layer 1: ISA Opcode 정의     (trace_opcode.h, *_opcode.h)
Layer 2: Performance Config   (gpgpusim.config, trace.config)
Layer 3: Power Model XML      (accelwattch_sass_sim.xml + static/constant 파라미터)
```

| GPU | 아키텍처 | SM | Layer 1 (Opcode) | Layer 2 (Config) | Layer 3 (Power XML) | 전력 모델링 가능? |
|-----|---------|-----|-----------------|-----------------|--------------------|--------------------|
| Titan K20 | Kepler | SM35 | ✅ kepler_opcode.h | ✅ SM3_KEPLER_TITAN | ❌ | **불가** |
| Titan X | Pascal | SM61 | ✅ pascal_opcode.h | ✅ SM6_TITANX | ✅ (Case Study) | **가능** |
| Titan V | Volta | SM70 | ✅ volta_opcode.h | ✅ SM7_TITANV | ✅ | **가능** |
| Quadro V100 | Volta | SM70 | ✅ volta_opcode.h | ✅ SM7_QV100 | ✅ (주력 검증) | **가능** |
| GV100 | Volta | SM70 | ✅ volta_opcode.h | ✅ SM7_GV100 | ✅ | **가능** |
| RTX 2060 | Turing | SM75 | ✅ turing_opcode.h | ✅ SM75_RTX2060 | ❌ | **불가** |
| RTX 2060S | Turing | SM75 | ✅ turing_opcode.h | ✅ SM75_RTX2060_S | ✅ (Case Study) | **가능** |
| **A100** | **Ampere** | **SM80** | ✅ ampere_opcode.h | ✅ SM80_A100 | **❌ 없음** | **불가** |
| RTX 3070 | Ampere | SM86 | ✅ ampere_opcode.h | ✅ SM86_RTX3070 | ❌ | **불가** |
| **H100** | **Hopper** | **SM90** | **❌ 없음** | **❌ 없음** | **❌ 없음** | **불가** |
| RTX 4090 | Ada | SM89 | **❌ 없음** | **❌ 없음** | **❌ 없음** | **불가** |

### 1.2 최신 GPU 주요 기능별 지원 상태

각 기능에 대해 코드에서 grep하여 확인한 결과이다:

| 기능 | 도입 GPU | 코드에 존재? | 지원 상태 | 미지원 원인 |
|------|---------|------------|----------|------------|
| **FP32/INT32 동시 실행** | A100 | ❌ | **미지원** | 코드에 공유 경로 개념 없음. `calculate_static_power()`가 INT/FP를 별도 datapath로 가정 |
| **TF32 Tensor Core** | A100 | ❌ | **미지원** | `TF32`, `tf32` 문자열 코드 전체에 없음. HMMA opcode가 모두 단일 `TENSOR__OP`으로 매핑 |
| **BF16 Tensor Core** | A100 | ❌ | **미지원** | `BF16`, `bf16` 문자열 코드 전체에 없음. BMMA가 `TENSOR__OP`으로 통합 매핑 |
| **2:4 Structured Sparsity** | A100 | ❌ | **미지원** | `sparsity`, `sparse` 관련 코드 없음. Tensor Core throughput 2배 변화 미반영 |
| **Async Copy (cp.async)** | A100 | ⚠️ 부분 | **제한적** | `OP_LDGSTS` opcode만 정의됨. `OTHER_OP`으로 매핑되어 별도 전력 component 없음 |
| **L2 Residency Control** | A100 | ❌ | **미지원** | L2 파티셔닝/캐시 persistence 관련 코드 없음 |
| **MIG (Multi-Instance GPU)** | A100 | ❌ | **미지원** | `MIG`, `multi_instance` 문자열 없음. 부분 GPU 설정 불가 |
| **3rd Gen Tensor Core** | A100 | ⚠️ 부분 | **제한적** | HMMA/DMMA opcode는 있으나 모두 동일 `TENSOR__OP` → 데이터 타입별 에너지 구분 불가 |
| **FP8 (E4M3/E5M2)** | H100 | ❌ | **미지원** | `FP8`, `fp8` 문자열 없음. SM90 opcode 파일 자체가 없음 |
| **Transformer Engine** | H100 | ❌ | **미지원** | `transformer_engine` 문자열 없음. 동적 정밀도 전환 개념 없음 |
| **DPX Instructions** | H100 | ❌ | **미지원** | `DPX`, `dpx` 문자열 없음. SM90 opcode 미정의 |
| **TMA (Tensor Memory Accelerator)** | H100 | ❌ | **미지원** | `TMA`, `tma` 문자열 없음 |
| **Thread Block Clusters** | H100 | ❌ | **미지원** | `thread_block_cluster` 없음. SM occupancy 모델이 이 계층 미반영 |
| **WGMMA (Warp Group MMA)** | H100 | ❌ | **미지원** | `WGMMA` opcode 미정의. SM90 ISA 전체 부재 |
| **NVLink 4.0** | H100 | ❌ | **미지원** | NoC 모델이 NVLink 세대 구분 없음 |
| **HBM3** | H100 | ❌ | **미지원** | DRAM 모델이 HBM2/HBM2e까지만. HBM3 타이밍 파라미터 없음 |
| **Shader Execution Reordering** | RTX 4090 | ❌ | **미지원** | SM89 opcode 전체 부재 |
| **3rd Gen RT Core** | RTX 4090 | ❌ | **미지원** | RT Core 전력 모델 없음 |

### 1.3 미지원 원인 상세 분석

#### 원인 A: Power Component Enum에 정의 없음

현재 `abstract_hardware_model.h`의 `special_ops` enum:

```cpp
enum special_ops {
    OTHER_OP, INT__OP, INT_MUL24_OP, INT_MUL32_OP, INT_MUL_OP,
    INT_DIV_OP, FP_MUL_OP, FP_DIV_OP, FP__OP, FP_SQRT_OP,
    FP_LG_OP, FP_SIN_OP, FP_EXP_OP, DP_MUL_OP, DP_DIV_OP,
    DP___OP, TENSOR__OP, TEX__OP
    // ← TF32__OP, BF16__OP, FP8__OP, DPX__OP 등이 없음
};
```

HMMA 명령어가 TF32, BF16, FP16, INT8 중 어떤 모드로 실행되든 모두 **동일한 `TENSOR__OP`**으로 매핑된다. 데이터 타입별 에너지 소모 차이를 구분할 수 없다.

**영향받는 기능**: TF32, BF16, FP8, 2:4 Sparsity, WGMMA

#### 원인 B: SM90(Hopper) ISA 정의 파일 부재

`gpu-simulator/ISA_Def/` 디렉토리에 `hopper_opcode.h` 또는 `sm90_opcode.h`가 존재하지 않는다. trace_driven.cc의 binary version detection에도 SM90 분기가 없다:

```cpp
// trace_driven.cc:105-123 — SM90 분기 없음
if (binary_version == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
// ... Volta, Pascal, Kepler, Turing만 존재
// SM90 = ? → 매칭 실패 → 시뮬레이션 불가
```

**영향받는 기능**: H100의 모든 기능 (FP8, Transformer Engine, DPX, TMA, WGMMA, Thread Block Clusters)

#### 원인 C: Static Power Model의 아키텍처 고정 가정

`calculate_static_power()`의 instruction mix 분류는 V100의 전용 INT32/FP32 datapath를 전제한다:

```cpp
// gpgpu_sim_wrapper.cc:832-912
// INT와 FP 접근을 별도로 카운트하여 카테고리 결정
if (int_accesses != 0 && fp_accesses != 0 && ...)
    → INT_FP 카테고리 (cat2)
```

A100에서는 FP32/INT32 공유 코어가 있으므로, `int_accesses`와 `fp_accesses`가 동시에 0이 아닐 때의 전력 특성이 V100과 근본적으로 다르다.

**영향받는 기능**: A100의 INT32/FP32 동시 실행, 모든 Static Power 추정

#### 원인 D: XML 파라미터 구조의 한계

`XML_Parse.h`의 `scaling_coefficients[64]` 배열은 고정 크기이며, TF32_ACC, BF16_ACC 등의 인덱스가 정의되어 있지 않다. 새 component를 추가하려면 이 구조체를 확장해야 한다.

**영향받는 기능**: 모든 신규 power component

---

## 2. 전체 Call Graph: XML → 시뮬레이션 → MAPE

### 2.1 Phase 1: XML 로딩 및 초기화

```
GPU 커널 최초 실행 시:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gpu-sim.cc:1205
  └─ init_mcpat()                          ← 전력 시뮬레이션 초기화 진입점
      │
      ▼
power_interface.cc
  └─ init_mcpat(gpgpu_sim *gpu, ...)
      │
      ▼
gpgpu_sim_wrapper 생성자 (gpgpu_sim_wrapper.cc:82-136)
  │
  ├─ ParseXML *p = new ParseXML()          ← XML 파서 생성
  │   └─ p->parse(xml_filename)            ← accelwattch_sass_sim.xml 파싱
  │       │
  │       ▼
  │   XML_Parse.cc:71  ParseXML::parse()
  │     ├─ XMLNode::openFileHelper()       ← XML 파일 열기
  │     ├─ scaling_coefficients[] 채우기   ← Dynamic power scaling factors
  │     │   ├─ [TOT_INST] = 10.0
  │     │   ├─ [INT_ACC]  = 14.988
  │     │   ├─ [FP_ACC]   = 0.530
  │     │   ├─ [TENSOR_ACC] = 0.815
  │     │   └─ ... (34개 파라미터)
  │     ├─ constant_power = 32.325         ← P_const
  │     ├─ idle_core_power = 0.283         ← P_perIdleSM
  │     └─ static_cat*_flane/addlane       ← Static power 카테고리 파라미터
  │         ├─ cat1 (INT): flane=15.29, addlane=0.586
  │         ├─ cat2 (INT+FP): flane=18.62, addlane=0.645
  │         └─ ... (9개 카테고리)
  │
  └─ Processor *proc = new Processor(p)    ← McPAT 프로세서 모델 생성
      └─ 코어, 캐시, NoC, 메모리 컨트롤러 모델 초기화
```

### 2.2 Phase 2: 시뮬레이션 루프 (매 500 cycles)

```
시뮬레이션 매 sampling interval (기본 500 cycles):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gpu-sim.cc:2104
  └─ mcpat_cycle(gpu, ...)                 ← 주기적 전력 계산 호출
      │
      ▼
power_interface.cc:49  mcpat_cycle()
  │
  │ ┌──────────────────────────────────────────────────────┐
  │ │  Step 1: Activity Counter 수집 (power_stat_t에서)    │
  │ ├──────────────────────────────────────────────────────┤
  │ │                                                      │
  │ │  total_inst    = power_stats->get_total_inst()       │
  │ │  fp_inst       = power_stats->get_total_fp_inst()    │
  │ │  int_inst      = power_stats->get_total_int_inst()   │
  │ │  l1_read_hit   = power_stats->get_l1d_read_hit()    │
  │ │  l1_read_miss  = power_stats->get_l1d_read_miss()   │
  │ │  ialu_access   = power_stats->get_ialu_accessess()  │
  │ │  fpu_access    = power_stats->get_fp_accessess()    │
  │ │  tensor_access = power_stats->get_tensor_accessess()│
  │ │  noc_access    = power_stats->get_noc_accessess()   │
  │ │  ... (50+ 카운터)                                    │
  │ └──────────────────────────────────────────────────────┘
  │
  │ ┌──────────────────────────────────────────────────────┐
  │ │  Step 2: Activity를 Wrapper에 설정                   │
  │ ├──────────────────────────────────────────────────────┤
  │ │                                                      │
  │ │  wrapper->set_inst_power(total, fp, int)             │
  │ │    └─ p->sys.core[0].total_instructions              │
  │ │       = total × scaling_coefficients[TOT_INST]       │
  │ │                                                      │
  │ │  wrapper->set_l1cache_power(read_h, read_m, ...)     │
  │ │    └─ p->sys.core[0].dcache.read_accesses            │
  │ │       = read_h × scaling_coefficients[DC_RH]         │
  │ │                                                      │
  │ │  wrapper->set_exec_unit_power(ialu, fpu, sfu, ...)   │
  │ │  wrapper->set_l2cache_power(l2_rh, l2_rm, ...)       │
  │ │  wrapper->set_mem_power(mem_rd, mem_wr, mem_pre)      │
  │ │  wrapper->set_NoC_power(noc_access)                   │
  │ │  wrapper->set_idle_core_power(num_idle)                │
  │ │  wrapper->set_avg_active_threads(avg_threads)          │
  │ └──────────────────────────────────────────────────────┘
  │
  │ ┌──────────────────────────────────────────────────────┐
  │ │  Step 3: McPAT 에너지 계산                           │
  │ ├──────────────────────────────────────────────────────┤
  │ │                                                      │
  │ │  wrapper->compute()                                  │
  │ │    └─ proc->compute()         (processor.cc:482)     │
  │ │        ├─ cores[0]->compute() (core.cc)              │
  │ │        │   └─ 각 유닛(IFU,LSU,EXU)의 에너지 계산    │
  │ │        │      activity × technology energy model     │
  │ │        │      → rt_power.readOp.dynamic에 저장      │
  │ │        ├─ l2array[0]->computeEnergy()                │
  │ │        └─ nocs[0]->computeEnergy()                   │
  │ └──────────────────────────────────────────────────────┘
  │
  │ ┌──────────────────────────────────────────────────────┐
  │ │  Step 4: Component Power 추출                        │
  │ ├──────────────────────────────────────────────────────┤
  │ │                                                      │
  │ │  wrapper->update_components_power()                  │
  │ │    │                                                 │
  │ │    ├─ update_coefficients()                          │
  │ │    │   └─ effpower[i] = raw[i] × scaling[i] / T     │
  │ │    │                                                 │
  │ │    ├─ 각 component power 추출:                       │
  │ │    │   IBP  = ifu->IB->rt_power / T                 │
  │ │    │   ICP  = ifu->icache->rt_power / T             │
  │ │    │   DCP  = lsu->dcache->rt_power / T             │
  │ │    │   INTP = exu->exeu->rt_power / T × clock_ratio │
  │ │    │   FPUP = fp_u->rt_power × (FP_ACC/total_fpu)   │
  │ │    │   DPUP = fp_u->rt_power × (DP_ACC/total_fpu)   │
  │ │    │   ... (33개 component)                          │
  │ │    │                                                 │
  │ │    ├─ CONSTP = constant_power (XML에서 직접)         │
  │ │    │                                                 │
  │ │    ├─ STATICP = calculate_static_power()             │
  │ │    │   ├─ instruction mix 분류 (INT/FP/DP/SFU/...)  │
  │ │    │   ├─ 카테고리 선택 (cat1~cat6, intadd, ...)    │
  │ │    │   ├─ Linear Model 적용:                         │
  │ │    │   │   total = base + (threads-1) × lane         │
  │ │    │   └─ × per_active_core 비율                     │
  │ │    │                                                 │
  │ │    └─ DVFS 보정 (활성화 시):                         │
  │ │        ├─ Static: × V_ratio                          │
  │ │        └─ Dynamic: × V_ratio²                        │
  │ └──────────────────────────────────────────────────────┘
  │
  │ ┌──────────────────────────────────────────────────────┐
  │ │  Step 5: 전력 집계 및 기록                           │
  │ ├──────────────────────────────────────────────────────┤
  │ │                                                      │
  │ │  wrapper->power_metrics_calculations()               │
  │ │    P_total = proc->rt_power + CONSTP + STATICP       │
  │ │    kernel_avg, kernel_max, kernel_min 갱신           │
  │ │                                                      │
  │ │  wrapper->print_trace_files()  (trace 모드 시)       │
  │ └──────────────────────────────────────────────────────┘
```

### 2.3 Phase 3: 커널 종료 시 Power Report 출력

```
커널 실행 완료 시:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gpu-sim.cc:1550
  └─ print_power_kernel_stats()
      │
      ▼
gpgpu_sim_wrapper.cc:1092  print_power_kernel_stats()
  │
  └─ accelwattch_power_report.log에 출력:
      ├─ kernel_avg_power = 148.3
      ├─ gpu_avg_IBP = 3.2
      ├─ gpu_avg_ICP = 2.8
      ├─ ...
      ├─ gpu_avg_CONSTP = 32.3
      ├─ gpu_avg_STATICP = 38.6
      ├─ kernel_max_power = 165.2
      └─ kernel_min_power = 95.4
```

### 2.4 Phase 4: Microbenchmark → QP Solver

```
시뮬레이션 결과 후처리:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[A] 시뮬레이션 power 수집
    collect_power_reports.sh
      └─ 각 benchmark의 accelwattch_power_report.log 수집
          └─ accelwattch_power_reports/{config}/{benchmark}.log

[B] CSV 변환
    gen_sim_power_csv.py volta_sass_sim
      ├─ 각 benchmark.log 파싱
      ├─ kernel_name으로 해당 커널 식별
      ├─ 33개 power counter 추출
      ├─ SASS SIM 모드: 8개 제거 + 2개 결합
      │   ├─ DRAMP = DRAMP + MCP  (DRAM과 MC 결합)
      │   └─ L2CP = L2CP + NOCP   (L2와 NoC 결합)
      └─ 출력: accelwattch_volta_sass_sim.csv
              (benchmark × 25열 = 22 dynamic + IDLE + CONST + STATIC + P_meas)

[C] 하드웨어 전력 측정 (병렬 수행)
    profile_validation_power.sh volta 0
      └─ measureGpuPower (C++ 프로그램)
          ├─ NVML API로 GPU 전력 샘플링
          ├─ 65°C 안정화 대기
          ├─ 5회 반복 측정
          └─ 평균/표준편차 계산
              └─ hw_power_validation_volta.csv

[D] QP 최적화
    quadprog_solver.m
      ├─ 입력: accelwattch_volta_sass_sim.csv
      │   A = (102 benchmarks × 31 activity columns)
      │   b = (102 × 1 measured power)
      ├─ 최적화: min ‖Ax − b‖²
      │   제약: 0.1 ≤ xᵢ ≤ 1000
      │         x_IDLE = x_CONST = x_STATIC = 1
      │         13개 에너지 순서 부등식
      └─ 출력: scaled_coefficients.csv (31개 최적 scaling factor)

[E] XML 업데이트 (수동)
    scaled_coefficients.csv의 값을
    accelwattch_sass_sim.xml의 각 param value에 반영
      └─ 재시뮬레이션 → 수렴까지 반복
```

### 2.5 Phase 5: Validation → MAPE 산출

```
검증 및 MAPE 계산:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[F] Validation 커널 시뮬레이션
    run_simulations.py -C QV100-Accelwattch_SASS_SIM -B validation_suite
      └─ 26개 validation 커널 (학습에 사용하지 않은 독립 데이터)
          └─ accelwattch_power_report.log 생성

[G] MAPE 계산
    plot-correlation.py
      │
      ├─ 시뮬레이터 전력 (P_sim) 로드
      ├─ 하드웨어 전력 (P_hw) 로드
      │
      ├─ 각 커널 i에 대해:
      │
      │             │P_sim,i − P_hw,i│
      │   MAPE_i = ─────────────────── × 100%
      │                  P_hw,i
      │
      │         1   ᴺ
      │  MAPE = ─── Σ  MAPE_i
      │         N  i=1
      │
      ├─ 추가 지표:
      │   ├─ Pearson r = corrcoef(P_sim, P_hw)
      │   │
      │   │       Σ│P_sim,i − P_hw,i│
      │   ├─ Aggregate Error = ──────────────────── × 100%
      │   │                      Σ P_hw,i
      │   │
      │   │         ┌ 1   ᴺ                    ┐
      │   │         │─── Σ (P_sim,i − P_hw,i)² │
      │   │         └ N  i=1                    ┘
      │   └─ RMSE = ─────────────────────────────
      │                      P̄_hw
      │
      └─ 출력:
          ├─ Correlation plot (scatter plot)
          ├─ Per-kernel power breakdown (stacked bar)
          └─ MAPE = 9.2% (Volta SASS SIM 기준)
```

---

## 3. Accel-Sim v1.3.0과의 관계

Accel-Sim은 **성능 시뮬레이터**이고, AccelWattch는 그 위에 구축된 **전력 모델**이다.

```
                    ┌─────────────────────────┐
                    │   Accel-Sim v1.3.0      │
                    │   (성능 시뮬레이션)      │
                    │                         │
                    │  SM90(H100) 지원 시작    │
                    │  - hopper_opcode.h       │
                    │  - SM90 config           │
                    │  - H100 trace 지원       │
                    │                         │
                    │  ↓ activity counters     │
                    ├─────────────────────────┤
                    │   AccelWattch            │
                    │   (전력 모델)            │
                    │                         │
                    │  SM90 지원 ❌            │
                    │  - Power XML 없음        │
                    │  - TF32/FP8 component ❌ │
                    │  - Static model 미갱신   │
                    │  - QP solver 미확장      │
                    └─────────────────────────┘
```

Accel-Sim v1.3.0이 H100을 지원하더라도, AccelWattch의 전력 모델은 **별도로 확장해야** 한다. 성능 시뮬레이션이 가능하다는 것과 전력 모델링이 가능하다는 것은 다른 문제이다.

---

> **다음 문서**: 이전 문서의 기호 미비 사항은 본 문서 상단의 "기호 및 약어 정의" 섹션에 통합 정리하였다. 01~05 문서에서 참조할 수 있다.  
> **이전 문서**: [05_Errata_and_Clarifications.md](05_Errata_and_Clarifications.md) — 정오표
