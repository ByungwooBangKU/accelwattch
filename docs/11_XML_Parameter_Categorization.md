# XML 파라미터 카테고리 분류

> **출처:** `artifacts/AccelSim_GPU_Config_Matrix.xlsx` → 시트 `07_전력XML_param_컴포넌트`
> **대상:** `accelwattch_ptx_sim.xml`, `accelwattch_sass_hw.xml`, `accelwattch_sass_hybrid.xml`, `accelwattch_sass_sim.xml`, `gpuwattch_gtx480.xml`
> **파라미터 수:** 202개 (xml_파일 × param_name 기준 916행, 고유명 202개)
> **분류 기준:** ① 물리적 위치(HW 블록) ② 기능적 역할(전력 모델 내 쓰임)

이 문서는 AccelWattch XML이 **어떤 파라미터를 어디에 / 왜** 갖고 있는지를 한눈에 볼 수 있게 정리한 레퍼런스입니다. 02, 04, 06, 10 문서가 "어떻게 쓰이는가"를 설명한다면, 본 문서는 "무엇이 있는가"를 카테고리화한 인덱스 역할을 합니다.

---

## 1. 개요: 두 가지 분류 축

| 축 | 질문 | 활용 |
|---|---|---|
| **물리적 (Physical)** | "이 파라미터는 HW의 어느 블록과 연관되는가?" | 가상 가속기 설계 시 제거/수정할 하드웨어 블록 식별 |
| **기능적 (Functional)** | "이 파라미터는 전력 모델에서 어떤 역할을 하는가?" | Dynamic / Static / Constant / Config / Legacy 분리 |

### 1.1 기능 축 분포 (전력컴포넌트_힌트 기준)

| 기능 분류 | 개수 | 설명 |
|---|---|---|
| **Dynamic Activity (ISA buckets)** | 32 | INT/FP/DP/SFU/TEX/TENSOR + OTHER_OP 활동 카운터 |
| **STATIC/누설** | 25 | Static per-lane activation + constant + idle |
| **McPAT 기타 (Config / Legacy)** | 160 | 구조 파라미터, DRAM 계수, CPU 유산 |
| **(중복 제외 고유)** | **202** | |

> `OTHER_OP`(22) 버킷은 McPAT_기타와 22개 항목이 중복 카운트됩니다.

### 1.2 물리 축 요약 (본 문서 §2)

| Physical Block | 고유 파라미터 | 핵심 파일/코드 |
|---|---|---|
| Chip / System | 12 | `gpgpu_sim_wrapper::set_inst_power()` |
| SM Core — Frontend | 35 | `XML_Parse.cc`, `core.cc` (McPAT) |
| SM Core — Execution Units | 22 | `ampere_opcode.h`, `component_mapping.h` |
| L1 / Shared / Texture / Constant Cache | 16 | `cacheunit.cc`, L1D 활동 카운터 |
| L2 / L3 / Directory | 12 | `logic.cc`, `memoryctrl.cc` |
| Memory Controller / DRAM | 24 | `gpgpu_sim::update_stats()` |
| NoC / Interconnect | 17 | `noc.cc` |
| STATIC/누설 Lane | 25 | `mcpat.cc` static path, `quadprog_solver.m` |
| Constant Power | 1 | DVFS 피팅 y-절편 |
| 기타 (Tech / Homo flags) | 38 | CACTI `technology.cc` |

---

## 2. 물리적 분류 (Physical Categorization)

### 2.1 Chip / System Level

`최상위 Processor 노드` — 칩 전체를 기술하는 글로벌 파라미터.

| param_name | 설명 |
|---|---|
| `clock_rate` | Processor 기본 클럭 (MHz) |
| `clockrate` | 서브 블록 클럭 (대소문자 변형) |
| `target_core_clockrate` | SM 타겟 주파수 (DVFS 스윕 대상) |
| `core_clock_ratio` | core vs. 시스템 클럭 비 |
| `core_tech_node` | 공정 노드 (nm). **CACTI `technology.cc` 키** |
| `mem_tech_node` | 메모리 공정 노드 (nm) |
| `device_type` | 0: HP / 1: LSTP / 2: LOP |
| `temperature` | 칩 동작 온도 (K) |
| `longer_channel_device` | 0 no / 1 use (leakage 감소) |
| `physical_address_width` | 물리 주소 bit |
| `virtual_address_width` | 가상 주소 bit |
| `virtual_memory_page_size` | 페이지 크기 (B) |
| `machine_bits` | 기본 word size |
| `number_of_cores` | SM 개수 (V100=80, A100=108, Accel-B=216) |
| `number_hardware_threads` | 코어당 HW thread (GPU에선 warp 기준) |
| `architecture` / `GPU_Architecture` | 0: G80 / 1: Fermi / other: unsupported |
| `homogeneous_cores` | 1 = homo |

### 2.2 SM Core — Frontend (Fetch / Decode / Issue / Commit)

대부분 **McPAT CPU 유산** 파라미터. GPGPU-Sim은 in-order wavefront scheduler라 OoO 구조는 사용하지 않지만, McPAT 파서가 XML을 요구하기 때문에 값은 채워 둡니다.

| 세부 | 파라미터 |
|---|---|
| **Branch Pred** | `BTB_config`, `RAS_size`, `prediction_width`, `chooser_predictor_bits`, `chooser_predictor_entries`, `global_predictor_bits`, `global_predictor_entries`, `local_predictor_entries`, `local_predictor_size` |
| **Fetch/Decode** | `fetch_width`, `decode_width`, `opcode_width`, `instruction_length`, `decoded_stream_buffer_size`, `number_instruction_fetch_ports`, `instruction_buffer_size` |
| **Issue/Window** | `issue_width`, `commit_width`, `instruction_window_size`, `instruction_window_scheme`, `fp_instruction_window_size`, `fp_issue_width` |
| **Pipeline** | `pipeline_depth`, `pipelines_per_core`, `machine_type` (0 OoO / 1 in-order) |
| **Register File** | `archi_Regs_IRF_size`, `archi_Regs_FRF_size`, `phy_Regs_IRF_size`, `phy_Regs_FRF_size`, `rename_scheme`, `register_windows_size`, `rf_banks`, `collector_units`, `ports` |
| **LSU** | `load_buffer_size`, `store_buffer_size`, `LSU_order`, `ROB_size`, `memory_ports` |
| **Warp/SIMD** | `warp_size` (32 fixed), `simd_width` |
| **Activity Counters** | `TOT_INST`, `FP_INT`, `PIPE_A`, `REG_RD`, `REG_WR`, `NON_REG_OPs`, `IDLE_CORE_N`, `CONST_DYNAMICN` |

> **주의:** 상당수(ROB, rename, chooser 등)는 V100/A100 XML에서 placeholder 값만 가지며, 실제 전력 기여는 **QP scaling factor가 0에 수렴**하거나 `STATIC/누설`에 흡수됩니다.

### 2.3 SM Core — Execution Units

**ISA 버킷 단위 Accesses 카운터.** 이 카운터들이 `ampere_opcode.h` → `component_mapping.h`를 거쳐 XML에 주입됩니다 (10 문서 Layer 1↔3).

| 기능 유닛 | param_name | 힌트 |
|---|---|---|
| INT ALU | `INT_ACC` | `INT__OP` |
| INT MUL | `INT_MUL_ACC` | `INT_MUL_OP` |
| INT MUL24 / MUL32 / DIV | `INT_MUL24_ACC`, `INT_MUL32_ACC`, `INT_DIV_ACC` | (unmapped) |
| FP32 ALU | `FP_ACC` | `FP__OP` |
| FP32 MUL | `FP_MUL_ACC` | `FP_MUL_OP` |
| FP32 DIV | `FP_DIV_ACC` | (unmapped) |
| DP64 ALU | `DP_ACC` | `DP___OP` |
| DP64 MUL | `DP_MUL_ACC` | `DP_MUL_OP` |
| DP64 DIV | `DP_DIV_ACC` | (unmapped) |
| SFU sin/cos | `FP_SIN_ACC` | `FP_SIN_OP` |
| SFU sqrt | `FP_SQRT_ACC` | `FP_SIN_OP` |
| SFU log | `FP_LG_ACC` | `FP_SIN_OP` |
| SFU exp | `FP_EXP_ACC` | `FP_SIN_OP` |
| Tensor Core | `TENSOR_ACC` | `TENSOR__OP` |
| Texture Unit | `TEX_ACC` | `TEX__OP` |
| *McPAT alias* | `SP_ACC`, `SFU_ACC`, `FPU_ACC` | — |
| **Provisioning** | `ALU_per_core`, `FPU_per_core`, `MUL_per_core` | 개수 |

> **추론형 가속기 (Accel-B)에서 제거 대상:** `DP_ACC`, `DP_MUL_ACC`, `DP_DIV_ACC`, `FP_EXP_ACC`, `FP_LG_ACC`, `FP_SIN_ACC`, `FP_SQRT_ACC`, `TEX_ACC` → 22 components → **15 components** (02 문서 §12 참고).

### 2.4 Cache Hierarchy

| 레벨 | Config | 활동 카운터 |
|---|---|---|
| **L1 I-cache** | `icache_config` (capacity, block_width, assoc, bank, throughput, latency) | `IC_H`, `IC_M` |
| **L1 D-cache** | `dcache_config` | `DC_RH`, `DC_RM`, `DC_WH`, `DC_WM` |
| **Constant cache** | `ccache_config` | `CC_H`, `CC_M` |
| **Texture cache** | `tcache_config` | `TC_H`, `TC_M` |
| **Shared memory** | `sharedmemory_config` | `SHRD_ACC` |
| **L2** | `L2_config`, `number_of_L2s`, `homogeneous_L2s` | `L2_RH`, `L2_RM`, `L2_WH`, `L2_WM` |
| **L3** | `L3_config`, `number_of_L3s`, `homogeneous_L3s` | — (GPU 미사용) |
| **Directory** | `Dir_config`, `Directory_type`, `number_of_L1Directories`, `number_of_L2Directories`, `homogeneous_L1Directorys`, `homogeneous_L2Directorys`, `homogeneous_ccs` | — (McPAT 유산) |
| **Buffers** | `buffer_sizes` (MSHR, fill, prefetch, wb) | — |
| **메타** | `number_cache_levels`, `number_entries`, `block_size` | — |

### 2.5 Memory Controller / DRAM

| 그룹 | 파라미터 |
|---|---|
| **Controller** | `number_mcs`, `memory_channels_per_mc`, `memory_ports`, `num_channels`, `mc_clock`, `device_clock`, `number_flashcs` |
| **DRAM chip** | `Block_width_of_DRAM_chip`, `burstlength_of_DRAM_chip`, `internal_prefetch_of_DRAM_chip`, `num_banks_of_DRAM_chip`, `output_width_of_DRAM_chip`, `page_size_of_DRAM_chip`, `number_ranks`, `PRT_entries`, `withPHY` |
| **Channel I/O** | `IO_buffer_size_per_channel`, `req_window_size_per_channel`, `peak_transfer_rate`, `capacity_per_channel`, `databus_width`, `addressbus_width` |
| **Activity (OTHER_OP)** | `MEM_RD`, `MEM_WR`, `MEM_PRE` |
| **DRAM coefficients** (**AccelWattch 고유**) | `dram_act_coeff`, `dram_activity_coeff`, `dram_cmd_coeff`, `dram_const_coeff`, `dram_nop_coeff`, `dram_pre_coeff`, `dram_rd_coeff`, `dram_req_coeff`, `dram_wr_coeff` |

> **DRAM coefficient 그룹**은 AccelWattch가 McPAT 기본 HBM 모델로는 현대 GPU DRAM 전력을 재현하지 못해 추가한 **계수형 회귀 파라미터**입니다. 04 문서 §4.3에서 설명한 대로 A100 HBM2e는 이 계수들만으로 커버되고, 멀티 도메인 DVFS 수식이 필요 없습니다.

### 2.6 NoC / Interconnect

| 파라미터 | 설명 |
|---|---|
| `type` | 1: NoC / 0: bus |
| `horizontal_nodes`, `vertical_nodes` | 토폴로지 |
| `has_global_link` | 1: 있음 |
| `input_ports`, `output_ports` | 라우터 포트 수 |
| `flit_bits` | flit 폭 |
| `link_throughput`, `link_latency` | 링크 성능 |
| `virtual_channel_per_port`, `input_buffer_entries_per_vc` | VC 구성 |
| `chip_coverage` | NoC 칩 면적 점유 (≤1) |
| `number_of_NoCs`, `homogeneous_NoCs` | NoC 개수 |
| `interconnect_projection_type` | 0: aggressive / 1: conservative wire |
| `number_units` | 단위 수 |
| `NOC_A` | **Activity counter (OTHER_OP)** |

### 2.7 Static / Leakage (Lane Activation)

AccelWattch가 추가한 **Half-warp → Linear Static Model** 파라미터. 각 카테고리마다 `firstLane` + `addLane` 한 쌍.

| 카테고리 | firstLane | addLane | 의미 |
|---|---|---|---|
| cat1 | `static_cat1_flane` | `static_cat1_addlane` | INT (ADD+MUL) |
| cat2 | `static_cat2_flane` | `static_cat2_addlane` | INT+FP |
| cat3 | `static_cat3_flane` | `static_cat3_addlane` | INT+FP+**DP** |
| cat4 | `static_cat4_flane` | `static_cat4_addlane` | INT+FP+**SFU** |
| cat5 | `static_cat5_flane` | `static_cat5_addlane` | INT+FP+**TEX** |
| cat6 | `static_cat6_flane` | `static_cat6_addlane` | INT+FP+**TENSOR** |
| geomean | `static_geomean_flane` | `static_geomean_addlane` | 6카테고리 기하평균 |
| intadd | `static_intadd_flane` | `static_intadd_addlane` | INT ADD only |
| intmul | `static_intmul_flane` | `static_intmul_addlane` | INT MUL only |
| light | `static_light_flane` | `static_light_addlane` | LIGHT_SM |
| L1D | `static_l1_flane` | — | L1 Data |
| L2 (INT) | `static_l2_flane` | — | L2 ADD+MUL |
| Shared | `static_shared_flane` | — | Shared Mem |
| **Idle SM** | `idle_core_power` | — | Power-gated SM base |
| **Constant** | `constant_power` | — | Chip-level constant |

> **Accel-B(추론 전용) 축약:** cat3 (DP), cat4 (SFU), cat5 (TEX) 제거 → **4 카테고리** (cat1, cat2, cat6, geomean) + intadd/intmul + L1/L2/shared/light = **14 static params** (22개에서 감소).

---

## 3. 기능적 분류 (Functional Categorization)

### 3.1 Class D: Dynamic Activity Counters → ISA Bucket

전력 방정식의 **동적 성분** `P_dyn = Σᵢ αᵢ · ACCᵢ · βᵢ` 에서 `ACCᵢ`에 해당.

| ISA Bucket (component_mapping) | 구성 Access 카운터 |
|---|---|
| `INT__OP` | INT_ACC |
| `INT_MUL_OP` | INT_MUL_ACC |
| `FP__OP` | FP_ACC |
| `FP_MUL_OP` | FP_MUL_ACC |
| `DP___OP` | DP_ACC |
| `DP_MUL_OP` | DP_MUL_ACC |
| `FP_SIN_OP` (SFU) | FP_SIN_ACC, FP_SQRT_ACC, FP_LG_ACC, FP_EXP_ACC |
| `TENSOR__OP` | TENSOR_ACC |
| `TEX__OP` | TEX_ACC |
| `OTHER_OP` (22개) | TOT_INST, FP_INT, PIPE_A, REG_RD, REG_WR, IC_H, IC_M, CC_H, CC_M, DC_RH, DC_RM, DC_WH, DC_WM, L2_RH, L2_RM, L2_WH, L2_WM, NOC_A, MEM_RD, MEM_WR, MEM_PRE, SHRD_ACC |
| **Unmapped** (XML 필드는 있으나 component_mapping에 없음) | INT_MUL24_ACC, INT_MUL32_ACC, INT_DIV_ACC, FP_DIV_ACC, DP_DIV_ACC, TC_H, TC_M |

### 3.2 Class S: Static / Leakage Power

`P_static = firstLane + addLane · (y-1)` (Linear Half-warp) — 2.7절 25개 파라미터 전체. 기능적으로는 다음 세 층:

| 계층 | 파라미터 |
|---|---|
| **Lane 활성 (y)** | static_cat1~6_* , static_intadd/intmul_*, static_light_* , static_geomean_* |
| **Cache/Shared 베이스** | static_l1_flane, static_l2_flane, static_shared_flane |
| **Idle SM** | idle_core_power (power-gated SM당 잔류 전력) |

### 3.3 Class C: Constant Power

| 파라미터 | 역할 |
|---|---|
| `constant_power` | DVFS 피팅 `P = β·C·f³ + τ·f + P_const` 의 y절편. HBM2e 고정 전력 + 클럭 트리 + PHY 등이 흡수됨. |

### 3.4 Class F: Architecture / Config (수식 외 구조 기술)

전력 계산에는 직접 계수로 들어가지 않지만, **CACTI/McPAT이 블록 면적·리키지를 추정**할 때 사용.

| 그룹 | 예 |
|---|---|
| **Tech scaling** | core_tech_node, mem_tech_node, device_type, temperature, longer_channel_device, interconnect_projection_type |
| **Core provisioning** | ALU_per_core, FPU_per_core, MUL_per_core, pipelines_per_core, simd_width, warp_size |
| **Cache sizing** | *cache_config, L2_config, sharedmemory_config, buffer_sizes |
| **Memory sizing** | num_channels, num_banks_of_DRAM_chip, page_size_of_DRAM_chip, ... |
| **NoC sizing** | horizontal_nodes, vertical_nodes, flit_bits, link_* |

### 3.5 Class L: McPAT Legacy (GPU 미사용)

McPAT 파서가 요구해서 존재만 하는 파라미터. QP가 대체로 0으로 수렴하거나 static에 흡수됨.

- **OoO 파이프라인**: ROB_size, instruction_window_size, rename_scheme, archi/phy_Regs_*_size, register_windows_size
- **분기 예측**: BTB_config, RAS_size, chooser_*, global_predictor_*, local_predictor_*
- **Homogeneity flags**: homogeneous_cores, homogeneous_L1/L2/L3Directorys, homogeneous_L2s, homogeneous_L3s, homogeneous_NoCs, homogeneous_ccs

### 3.6 Class M: DRAM Regression Coefficients (AccelWattch 확장)

`dram_*_coeff` 9종 — McPAT의 기본 HBM 모델이 부족해 AccelWattch가 도입한 **계수형 회귀 파라미터**. QP의 dim 31 중 DRAM 영역에 직접 들어감.

| param | 의미 |
|---|---|
| dram_const_coeff | DRAM 상수항 |
| dram_act_coeff / dram_activity_coeff | ACT 활성화 전력 |
| dram_cmd_coeff | 커맨드 버스 |
| dram_rd_coeff | Read |
| dram_wr_coeff | Write |
| dram_pre_coeff | Precharge |
| dram_nop_coeff | NOP/idle bus |
| dram_req_coeff | Request tracker |

---

## 4. 교차 분류 매트릭스 (요약)

| 물리 \ 기능 | D (Dyn) | S (Static) | C (Const) | F (Config) | L (Legacy) | M (DRAM coef) |
|---|---|---|---|---|---|---|
| Chip/System | — | — | ✓ | ✓ 12 | — | — |
| SM Frontend | TOT_INST, FP_INT, PIPE_A, REG_R/W | — | — | ✓ | ✓ 20+ | — |
| SM ExecUnits | 14 ACC | — | — | ✓ 3 | — | — |
| L1/Shared/TC/CC | IC_*, DC_*, CC_*, TC_*, SHRD_ACC | static_l1_flane, static_shared_flane | — | ✓ 5 | — | — |
| L2/L3/Dir | L2_* | static_l2_flane | — | ✓ 9 | ✓ (Dir) | — |
| MC/DRAM | MEM_RD/WR/PRE | — | — | ✓ 15 | — | ✓ 9 |
| NoC | NOC_A | — | — | ✓ 13 | — | — |
| Static Lane | — | ✓ 22 | — | — | — | — |
| Idle/Constant | — | idle_core_power | constant_power | — | — | — |

---

## 5. 가상 가속기 설계 시 "체크해야 할" 파라미터

본 분류를 **Accel-B(추론 전용, 216 SM, 2nm GAA)** 관점에서 우선순위로 정리:

### 5.1 🔴 필수 변경 (P0)

| 파라미터 | Accel-B 값 | 이유 |
|---|---|---|
| `number_of_cores` | 216 | SM 2배 |
| `core_tech_node` | 2 (nm) | CACTI `technology.cc`에 **신규 삽입 필요** |
| `clock_rate`, `target_core_clockrate` | DVFS 스윕 대상 | DVFS 재피팅 필요 |
| `ALU_per_core`, `FPU_per_core` | (기존) | Tensor unit 개수 반영 |
| **22 component → 15** | `DP_ACC`, `DP_MUL_ACC`, `DP_DIV_ACC`, `FP_SIN/SQRT/LG/EXP_ACC`, `TEX_ACC` 제거 |
| **static_cat3/4/5 제거** | DP/SFU/TEX 카테고리 삭제 |

### 5.2 🟠 재피팅 (P1)

- `constant_power` — DVFS 재스윕 후 y절편 재산출 (HBM2e → HBM3 / LPDDR6)
- `idle_core_power` — power-gated SM 개수 216 기준 geomean 재측정
- **9 DRAM coefficients** — 신규 메모리 기술 반영
- **static lane 파라미터 14개** (추론형 잔존분) — QP 재학습

### 5.3 🟡 검토 (P2)

- `warp_size` — 32 유지 (PTX 호환성)
- `dcache_config`, `L2_config`, `sharedmemory_config` — 용량 튜닝
- `horizontal_nodes`, `vertical_nodes` — NoC 토폴로지 재설계

### 5.4 ⚪ 유지 (P3) — McPAT Legacy

- BTB, RAS, ROB, chooser/global/local_predictor_* → **placeholder 유지** (파서 요구로 존재하나 전력 기여 없음)

---

## 6. 참조

- 시트 원본: `artifacts/AccelSim_GPU_Config_Matrix.xlsx` → `07_전력XML_param_컴포넌트`
- 관련 문서:
  - [02 개선 포인트](02_Improvement_Points.md) §12 — 추론형 가속기 component 축소
  - [04 A100 Equation 변화](04_A100_Equation_Changes.md) §4.3 — HBM2e 단일 도메인 DVFS
  - [06 Feature Support](06_GPU_Feature_Support_Analysis.md) — XML 로딩 콜그래프
  - [08 PTX Integration](08_PTX_Trace_Integration_Guide.md) — PTX 모드 가상 가속기 절차
  - [10 3-Layer Deep Dive](10_Three_Layer_Deep_Dive.md) — ISA → Perf → XML 완전 추적
