# GPU Power Modeling 연구 로드맵 (8개월)

> **연구 목표**: NVIDIA GPU와 유사하지만 상세 구조가 다른 새로운 유형의 가속기를 설계할 때, 해당 가속기의 전력 소모를 정확히 예측하는 power model을 구축한다.  
> **핵심 가치**: "칩을 제조하기 전에, 설계 파라미터를 바꿨을 때 전력이 얼마나 변하는지 예측할 수 있는 도구"를 만드는 것이다.  
> **기간**: 8개월 (Month 1 ~ Month 8)  
> **기반 도구**: AccelWattch (MICRO 2021) + Accel-Sim  
> **작성일**: 2026-04-02  

---

## 왜 AccelWattch를 기반으로 하는가?

새로운 가속기의 전력을 예측하려면, 기존에 검증된 GPU power model을 출발점으로 삼아 "구조를 바꿨을 때 전력이 어떻게 변하는지"를 시뮬레이션할 수 있어야 한다. AccelWattch가 적합한 이유는 다음과 같다:

1. **Design Space Exploration 능력**: AccelWattch는 논문 Section 7.1에서 Volta 모델을 재학습 없이 Pascal/Turing에 적용하여 11~13% MAPE를 달성했다. 이는 아키텍처 파라미터(SM 수, 캐시 크기, 코어 수 등)를 변경해도 합리적 예측이 가능하다는 것을 의미한다.

2. **Component-level 분해**: 22개 하드웨어 컴포넌트별로 전력을 분리하므로, 특정 유닛(예: Tensor Core)을 다른 구조로 교체했을 때 해당 컴포넌트의 전력만 재모델링하면 된다.

3. **오픈소스 + Configurable**: XML config 파일을 수정하여 가상의 GPU 아키텍처를 정의하고 전력을 시뮬레이션할 수 있다.

4. **Cycle-level 정확도**: 평균 전력뿐 아니라 시간에 따른 전력 변화(power trace)를 제공하여, 열 관리(thermal management) 설계에도 활용 가능하다.

---

## 전체 로드맵 요약

```
Month 1-2:  기반 구축 — A100 power model 재현 및 검증
Month 3-4:  모델 확장 — 신규 component 추가, QP solver 고도화
Month 5-6:  Design Space Exploration — 가상 가속기 설계 및 전력 예측
Month 7-8:  논문화 — 결과 분석, 모델 정확도 검증, 논문 작성
```

```
                        M1    M2    M3    M4    M5    M6    M7    M8
Phase 1: 기반 구축      ████  ████
Phase 2: 모델 확장                ████  ████
Phase 3: DSE                            ████  ████  ████
Phase 4: 논문화                                     ████  ████  ████
─────────────────────────────────────────────────────────────────────
마일스톤           MS1▲      MS2▲      MS3▲            MS4▲      MS5▲
```

| 마일스톤 | 시점 | 산출물 |
|---------|------|--------|
| **MS1** | M2 끝 | A100 baseline MAPE < 15% 달성 |
| **MS2** | M4 끝 | 신규 component 포함 MAPE < 12% 달성 |
| **MS3** | M5 끝 | 가상 가속기 3개 설계 및 전력 예측 완료 |
| **MS4** | M7 끝 | 논문 초고 완성 |
| **MS5** | M8 끝 | 논문 투고 |

---

## Phase 1: 기반 구축 (Month 1-2)

### 목표

AccelWattch를 A100 SXM4 80GB에서 재현하여, 기존 모델이 최신 GPU에서 어느 정도 정확한지 baseline을 확보한다. 이 단계 없이 모델을 확장하면 "개선이 진짜 개선인지" 판단할 기준이 없다.

### Month 1: 환경 구축 및 A100 데이터 수집

**주 1-2: 실험 환경 구축**

| 작업 | 근거 |
|------|------|
| A100 SXM4 80GB 접근 확보 (클라우드 또는 물리 서버) | 모든 실험의 전제 조건. SXM4 모델이어야 memory clock 1593MHz 기준 실험 가능 |
| CUDA 12.x + NVBit 설치, SASS trace 수집 환경 검증 | AccelWattch는 SASS trace 기반. NVBit이 A100에서 정상 동작하는지 확인 필요 |
| Accel-Sim 빌드, SM80_A100 config로 시뮬레이션 동작 확인 | 성능 시뮬레이션이 먼저 정상이어야 전력 모델을 올릴 수 있음 |
| NVML 기반 전력 측정 도구(measureGpuPower.cpp) A100 호환 확인 | A100의 NVML 샘플링 주파수(~20Hz)와 정확도 확인 |

**주 3-4: P_const 및 Static Power 측정**

| 작업 | 근거 |
|------|------|
| DVFS sweep 실험: SM clock 210~1410MHz (8~10 단계) × 5개 microbenchmark | Eq.(3)의 P_const' 도출. A100 memory clock이 1593MHz 고정이므로 단일 변수 실험으로 충분 |
| 3차 다항식 fitting → P_const' 추출 | V100에서 Pearson r=0.998을 달성한 방법론을 A100에 그대로 적용하여 검증 |
| Active SM 수 변화 실험 (1, 8, 27, 54, 108 SM) × 여러 microbenchmark | Eq.(6)~(8)의 idle_core_power 도출. A100은 108 SM이므로 V100(80 SM)보다 넓은 범위 필요 |
| Instruction mix별 static power 측정 (INT, FP, INT+FP, Tensor 등) | Eq.(4)의 firstLane/addLane 파라미터 도출. 9개 카테고리 각각에 대해 thread 수 변화 실험 |
| 온도 65°C 안정화 프로토콜 확립 | Static power는 온도에 지수적으로 의존. 400W TDP에서 65°C 유지가 V100보다 어려울 수 있으므로 cooling 전략 필요 |

### Month 2: A100 Baseline Power Model 구축

**주 5-6: AccelWattch XML 생성 및 초기 시뮬레이션**

| 작업 | 근거 |
|------|------|
| V100 XML을 템플릿으로 A100 accelwattch_sass_sim.xml 생성 | 기존 코드의 구조를 최대한 유지하면서 A100 파라미터 반영 |
| gpgpusim.config 수정: DRAM clock 1512→1593, power_simulation_enabled 0→1 | 04 문서에서 확인한 config 오류 수정 |
| core_tech_node: 23→7, SM 수: 80→108, 캐시 크기 업데이트 | A100 아키텍처 반영 |
| Month 1에서 측정한 P_const', idle_core_power, static_cat*를 XML에 반영 | 실측 기반 파라미터 |
| Validation suite (26개 커널) A100용 SASS trace 수집 및 시뮬레이션 | V100 validation suite와 동일 커널로 비교 가능성 확보 |

**주 7-8: QP 최적화 및 Baseline MAPE 측정**

| 작업 | 근거 |
|------|------|
| 102개 microbenchmark A100 실행 → activity factor + HW power 수집 | QP solver의 입력 데이터 생성 |
| gen_sim_power_csv.py 수정: A100 config 추가 | gen_sim_power_csv.py가 현재 Volta/Pascal/Turing만 지원 |
| quadprog_solver.m 실행 → A100 scaling factors 도출 | 기존 22 component 구조로 먼저 baseline 확보 |
| Validation → MAPE 측정 | **MS1 목표: MAPE < 15%**. V100의 9.2%보다 높겠지만, technology scaling만으로도 합리적 수준 |

**MS1 판단 기준**: MAPE 15% 이하면 AccelWattch의 기본 프레임워크가 A100에서도 동작함을 확인. 15% 이상이면 Phase 2의 component 확장이 더 절실한 것으로 판단.

---

## Phase 2: 모델 확장 (Month 3-4)

### 목표

A100의 고유 기능(INT/FP 공유 경로, TF32, Sparsity 등)을 반영하여 모델 정확도를 높이고, QP solver를 Python으로 전환하여 확장성을 확보한다.

### Month 3: 신규 Component 및 Static Model 확장

**주 9-10: INT32/FP32 공유 경로 반영 (가장 중요)**

| 작업 | 근거 |
|------|------|
| A100 INT/FP concurrent 실행 microbenchmark 설계 및 측정 | 05 문서 E11에서 지적한 바와 같이, 현재 코드는 Linear model만 구현. A100에서는 공유 경로로 인해 INT/FP 동시 실행 시 전력 특성이 V100과 근본적으로 다름 |
| calculate_static_power()에 concurrent 모드 카테고리 추가 | V100의 9개 카테고리 → A100은 12~13개로 확장. FP_ONLY, INT_ONLY, FP_INT_CONCURRENT 카테고리 필요 |
| static_cat*_flane/addlane 파라미터 실측 (새 카테고리) | 각 모드별 firstLane/addLane이 다르므로 개별 측정 |

**주 11-12: TF32/BF16 Tensor Core 분리**

| 작업 | 근거 |
|------|------|
| accelwattch_component_mapping.h에 TF32__OP, BF16__OP enum 추가 | 06 문서에서 확인: 현재 HMMA가 TF32/BF16/FP16 구분 없이 모두 TENSOR__OP으로 매핑됨. 데이터 타입에 따라 에너지가 다름 (TF32는 19bit 연산, FP16은 16bit) |
| trace_driven.cc에서 HMMA operand type 분석 로직 추가 | MUFU가 SIN/COS/EX2/RSQ/LG2로 세분화되는 것과 동일한 패턴으로, HMMA를 TF32/BF16/FP16/INT8로 세분화 |
| TF32/BF16 전용 microbenchmark 설계 및 실행 (4~6개씩) | QP solver에 새 component의 에너지를 학습시키기 위한 데이터 필요 |
| 2:4 Sparsity microbenchmark (Dense vs Sparse 비교) | Sparsity 활성화 시 throughput 2배이지만 에너지는 다를 수 있음. 이 차이를 정량화 |

### Month 4: QP Solver 고도화 및 MAPE 개선

**주 13-14: Python(cvxpy) QP solver 구현**

| 작업 | 근거 |
|------|------|
| quadprog_solver.m → Python(cvxpy) 전환 | MATLAB 라이선스 불필요, 팀 접근성 향상, ML 라이브러리와 통합 용이 |
| L2 정규화 추가: λ‖X‖² | Overfitting 방지. Component 수가 22→27+로 증가하면 microbenchmark 대비 파라미터 수가 늘어나 과적합 위험 |
| 제약조건 7nm 기준 재계산 | McPAT의 per-instruction 에너지 비율이 12nm과 7nm에서 다름. C 행렬의 계수 업데이트 필요 |
| 교차항 실험: INT_ACC × FP_ACC (concurrent 전력) | A100에서 INT/FP 동시 실행 시 단순 합 이상의 전력이 소모될 수 있음. 교차항이 MAPE를 개선하는지 실험 |

**주 15-16: 반복 최적화 및 MS2 달성**

| 작업 | 근거 |
|------|------|
| QP 반복 최적화 (3~5회 iteration) → 수렴 확인 | AccelWattch 논문에서도 Fermi 시작점 모델이 all-ones보다 정확 (9.2% vs 14.8%). 반복이 중요 |
| Validation suite로 MAPE 측정 | **MS2 목표: MAPE < 12%**. 신규 component 추가로 baseline 대비 3%p 이상 개선 기대 |
| Component별 MAPE 기여도 분석 | 어떤 component가 오차에 가장 크게 기여하는지 파악하여 추가 개선 방향 결정 |
| ML residual correction 실험 (선택적) | P_hybrid = P_analytical + f_ml(features). 분석 모델이 포착하지 못한 비선형 효과를 ML로 보정. MAPE 2~3%p 추가 개선 가능 |

---

## Phase 3: Design Space Exploration (Month 5-6)

### 목표

검증된 A100 power model을 기반으로, 가상의 가속기 아키텍처를 설계하고 전력을 예측한다. 이것이 이 연구의 핵심 기여이다. "칩을 만들기 전에 전력을 예측"하는 능력을 입증한다.

### 왜 이 단계가 핵심인가?

AccelWattch 논문이 이미 입증한 것은 "Volta 모델을 Pascal/Turing에 적용해도 합리적 정확도"라는 점이다(Section 7.1). 이는 **아키텍처 파라미터를 변경해도 모델이 전력 변화를 추적할 수 있다**는 의미이다. 우리는 이를 한 단계 더 확장하여, "존재하지 않는 가속기"의 전력을 예측한다.

### Month 5: 가상 가속기 설계

**주 17-18: 가속기 설계 파라미터 정의**

다음 3가지 가상 가속기를 설계한다. 각각은 특정 워크로드에 최적화된 설계이다:

| 가속기 | 설계 철학 | A100 대비 변경점 | 목표 워크로드 |
|--------|----------|-----------------|-------------|
| **Accel-A**: AI Training 특화 | Tensor Core 2배, FP32 절반, L2 2배 | SM당 Tensor 8개, FP32 32개, L2 80MB | LLM 학습 |
| **Accel-B**: AI Inference 특화 | INT8/INT4 강화, SM 수 증가, TDP 절반 | SM 216개 (축소), INT8 전용 유닛, 200W | 추론 서버 |
| **Accel-C**: HPC 특화 | FP64 2배, HBM 대역폭 1.5배 | SM당 FP64 64개, 메모리 3TB/s | 과학 시뮬레이션 |

이 설계들이 의미있는 이유:
- **Accel-A**: 현재 H100이 가는 방향과 유사하지만, FP32를 과감히 줄인 극단적 설계. "Tensor Core에 all-in하면 전력 효율이 얼마나 좋아지는가?"에 답함
- **Accel-B**: Inference 전용 칩(Groq, AWS Inferentia 등)의 방향. "SM을 작게 만들고 많이 넣으면?"에 답함
- **Accel-C**: 과학 계산용. "FP64를 극단적으로 강화하면 전력 비용이 얼마인가?"에 답함

**주 19-20: gpgpusim.config 및 XML 생성**

| 작업 | 근거 |
|------|------|
| 각 가속기의 gpgpusim.config 작성 | SM 수, 코어 수, 캐시 크기, 메모리 대역폭 등 아키텍처 파라미터 반영 |
| 각 가속기의 accelwattch XML 작성 | A100 XML을 템플릿으로, component별 scaling factor를 설계 의도에 맞게 조정 |
| Technology scaling factor 적용 | 동일 7nm 공정 가정 시 scaling 불필요. 5nm/4nm 가정 시 IRDS 기반 + 컴포넌트별 차별 scaling 적용 |

### Month 6: 전력 예측 및 분석

**주 21-22: 시뮬레이션 및 전력 예측**

| 작업 | 근거 |
|------|------|
| 대표 워크로드에서 각 가속기 시뮬레이션 | GEMM, Conv, LSTM, FFT, SpMV 등 AI + HPC 워크로드 |
| Component별 power breakdown 비교 | 3개 가속기 × A100 = 4개 설계의 전력 구성 비교. "Tensor Core 2배면 전체 전력의 몇 %를 차지하는가?" |
| Performance per Watt 분석 | 절대 전력뿐 아니라 성능/전력 효율 비교. 이것이 가속기 설계의 핵심 지표 |

**주 23-24: Design Trade-off 분석**

| 작업 | 근거 |
|------|------|
| 파라미터 sensitivity 분석 | SM 수, Tensor Core 수, L2 크기 등을 ±20% 변화시켰을 때 전력/성능 변화율 측정. "어떤 파라미터가 전력에 가장 민감한가?" |
| Pareto frontier 도출 | 성능 vs 전력의 Pareto 최적 설계점들을 찾아 "이 이상은 전력을 늘려도 성능이 안 오른다"는 한계 식별 |
| **MS3**: 가상 가속기 3개의 전력 예측 보고서 완성 | Component별 breakdown, 워크로드별 비교, design trade-off 분석 포함 |

---

## Phase 4: 논문화 (Month 7-8)

### 목표

연구 결과를 학술 논문으로 정리하여 MICRO, ISCA, HPCA 등 top-tier 컨퍼런스에 투고한다.

### 논문 구성 (제안)

| 섹션 | 내용 | Phase 연관 |
|------|------|-----------|
| 1. Introduction | GPU power modeling의 필요성, AccelWattch 한계, 연구 목표 | - |
| 2. Background | AccelWattch 모델, A100 아키텍처, design space exploration 필요성 | Phase 1 |
| 3. A100 Power Model Extension | INT/FP 공유 경로, TF32/BF16 component, Python QP solver | Phase 2 |
| 4. Validation | A100 MAPE 결과, V100 대비 개선, component별 분석 | Phase 1-2 |
| 5. Design Space Exploration | 3개 가상 가속기 설계, 전력 예측, trade-off 분석 | Phase 3 |
| 6. Insights & Discussion | design rule 도출, 모델 한계, 미래 방향 | Phase 3 |
| 7. Related Work | GPUWattch, IPP, ML 기반 모델, Guerreiro et al. | - |
| 8. Conclusion | 기여 요약, 향후 연구 | - |

### Month 7: 논문 작성

| 주 | 작업 |
|----|------|
| 25-26 | Figure/Table 확정: correlation plot, power breakdown bar chart, Pareto frontier, design comparison table |
| 27-28 | 본문 작성: Section 1-5 초고 완성. **MS4: 논문 초고** |

### Month 8: 리뷰 및 투고

| 주 | 작업 |
|----|------|
| 29-30 | 내부 리뷰, 실험 보완, 누락 데이터 수집 |
| 31-32 | 최종 교정, 카메라 레디 작성. **MS5: 논문 투고** |

### 투고 대상 학회

| 학회 | Tier | 분야 | 보통 마감 |
|------|------|------|----------|
| **MICRO** | Top | Computer Architecture | 4월 또는 6월 |
| **ISCA** | Top | Computer Architecture | 11월 |
| **HPCA** | Top | High Performance Architecture | 7월 |
| **ISPASS** | Mid | Performance Analysis | 10월 |
| **DAC** | Top | Design Automation | 11월 |

---

## 리스크 및 대응 방안

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| A100 접근 불가 또는 지연 | 중 | 높음 | 클라우드(AWS p4d, GCP A2) 대안 확보. 1시간 단위 과금으로 비용 절감 가능 |
| NVBit이 최신 CUDA에서 동작 안함 | 낮 | 중 | Accel-Sim에 이미 포함된 A100 trace 활용. 또는 Nsight Compute profiling 대체 |
| A100 baseline MAPE > 20% | 중 | 중 | Technology scaling만으로 부족하다는 의미. Phase 2를 앞당겨 component 확장에 집중 |
| 가상 가속기 전력 예측이 비현실적 | 중 | 높음 | 실제 존재하는 H100 또는 MI300X를 "가상 가속기" 대신 사용. 실측과 비교 가능 |
| 8개월 내 완료 불가 | 낮 | 높음 | Phase 3를 1개 가속기로 축소하여 깊이 있게 분석. Phase 4를 workshop paper로 전환 |

---

## 필요 자원

| 자원 | 용도 | 예상 비용/시간 |
|------|------|-------------|
| A100 SXM4 80GB GPU | 실험 (DVFS sweep, microbenchmark, validation) | 클라우드: ~$3/hr × 200hr = $600 |
| CUDA 12.x + NVBit | SASS trace 수집 | 무료 |
| Accel-Sim + AccelWattch | 시뮬레이션 | 오픈소스, 무료 |
| Python + cvxpy + scipy | QP solver, 데이터 분석 | 무료 |
| 시뮬레이션 서버 (56+ cores) | Accel-Sim 시뮬레이션 병렬 실행 | 기존 서버 활용 |
| MATLAB (선택) | 기존 quadprog_solver.m 대조용 | 대학 라이선스 |

---

## 최종 산출물 목록

| # | 산출물 | 시점 |
|---|--------|------|
| 1 | A100 SXM4 80GB AccelWattch power model (XML config) | M2 |
| 2 | Python QP solver (cvxpy 기반, 정규화 + 교차항 지원) | M4 |
| 3 | 확장 AccelWattch 코드 (TF32/BF16/Sparsity component, concurrent static model) | M4 |
| 4 | 가상 가속기 3개의 config + 전력 예측 보고서 | M6 |
| 5 | Design trade-off 분석 (Pareto frontier, sensitivity analysis) | M6 |
| 6 | 학술 논문 (MICRO/ISCA/HPCA급) | M8 |
| 7 | 오픈소스 코드 + 재현 가능한 실험 스크립트 | M8 |

---

> **핵심 메시지**: 이 로드맵은 "기존 GPU의 전력을 정확히 예측하는 것(Phase 1-2)"을 넘어, "아직 존재하지 않는 가속기의 전력을 예측하는 것(Phase 3)"까지 나아간다. 이것이 가능한 이유는 AccelWattch의 component-level 분해와 configurable 설계 덕분이다. 칩을 실제로 제조하기 전에 전력 envelope을 파악할 수 있다면, 이는 수십억 원의 설계 비용과 수개월의 시간을 절약하는 실질적 가치를 제공한다.
