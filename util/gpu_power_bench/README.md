# GPU Power-per-Operation Benchmark Suite — Technical Reference

> **한 줄 요약**: 같은 연산을 서로 다른 precision / 서로 다른 compute unit 으로 돌렸을 때 몇 Joule 이 드는지를 A100 (sm_80) 과 H100 (sm_90) 에서 동일 코드로 재서, AccelWattch-style analytical GPU power model 의 per-op coefficient (`k_op`) 를 실측으로 뽑기 위한 microbenchmark 스위트.

## 초록 (Abstract)

본 문서는 15개의 GPU power microbenchmark — (FP16/FP8) × (MUL/ADD/Softmax/GeLU/LayerNorm) 10개 elementwise 와, matmul 5 variant (`fp32_simt`, `tf32_tc`, `fp16_tc`, `bf16_tc`, `fp8_te`) — 를 통해 **(a)** GPU 의 정적 전력 `P_static` 과 **(b)** 연산당 동적 에너지 계수 `k_op` 를 분리 추출하는 워크플로우를 기술한다. NVML power telemetry 를 100 Hz 로 샘플링하고 trapezoidal rule 로 적분하여 구간 에너지를 얻은 뒤, load sweep 에 대한 선형회귀로 `slope_dyn = k_op` 를 추출한다. R² ≥ 0.99 가 선형성(1차 모델 가정의 유효성) 조건이다. 산출물은 per-cell CSV + 전체 power/temperature trace CSV + 7종의 분석 plot + cross-GPU 비교용 3종 plot 으로 구성된다. 본 README 는 설계 근거(background), 각 벤치마크의 물리적 의미, 측정 방법론, 분석 이론, 그리고 모든 산출 차트의 해석 가이드를 포함한다.

## 목차

- [0. 용어와 기호](#0-용어와-기호)
- [1. 배경 (Motivation & Background)](#1-배경-motivation--background)
- [2. 전력 모델 (Power Model)](#2-전력-모델-power-model)
- [3. 벤치마크 설계](#3-벤치마크-설계)
- [4. 측정 방법론](#4-측정-방법론)
- [5. 분석 방법](#5-분석-방법)
- [6. 차트 해석 가이드](#6-차트-해석-가이드)
- [7. 예상 결과 (A100 vs H100)](#7-예상-결과-a100-vs-h100)
- [8. 설치 & 사전 점검](#8-설치--사전-점검)
- [9. 실행](#9-실행)
- [10. 산출 파일 레퍼런스](#10-산출-파일-레퍼런스)
- [11. 권장 워크플로우](#11-권장-워크플로우)
- [12. 유효성 체크리스트](#12-유효성-체크리스트)
- [13. 알려진 한계](#13-알려진-한계)
- [14. 확장 아이디어](#14-확장-아이디어)
- [15. 파일 구성](#15-파일-구성)
- [부록 A. 수치해석 주의사항](#부록-a-수치해석-주의사항)
- [부록 B. NVML power telemetry semantics](#부록-b-nvml-power-telemetry-semantics)

## 0. 용어와 기호

| 기호 | 의미 |
|---|---|
| `P_static` (W) | GPU 가 idle 상태에서 소모하는 전력. 누설 전류 + 컨트롤러 + PLL + 메모리 리프레시 등. |
| `P_dyn(t)` (W) | 워크로드가 추가로 유발하는 순간 전력. `P_dyn = P_total − P_static`. |
| `E_total` (J) | 구간 `[t0, t1]` 에서 NVML power 를 적분한 총 에너지. |
| `E_static` (J) | `P_static · (t1 − t0)`. 해당 구간에 idle 이었어도 나갔을 에너지. |
| `E_dyn` (J) | `E_total − E_static`. "그 연산이 추가로 만든 에너지". |
| `N_op` | 연산 실행 횟수 (elementwise 는 element 수, matmul 은 FLOP 수). |
| `k_op` (J/op) | 한 번의 op 이 평균적으로 소비하는 동적 에너지. regression slope 로 추정. |
| `T_workload` (s) | 전체 워크로드 실행 시간. |
| `R²` | 선형회귀 결정계수. `k_op` 를 상수로 본 가정의 유효성을 가늠. |
| TC | Tensor Core. matrix-multiply-accumulate 전용 유닛. |
| SIMT | Streaming Multi-Threaded. 일반 CUDA core 경로. |
| TE | Transformer Engine. NVIDIA FP8 GEMM 래퍼 라이브러리 (`fp8_autocast`). |

## 1. 배경 (Motivation & Background)

### 1.1 왜 per-operation 에너지 측정인가

GPU 시스템의 **총 에너지** 는 두 축으로 분해된다:

1. **누가 무엇을 얼마나 하느냐** — 모델/워크로드 특성.
2. **각 연산이 평균적으로 얼마의 전력을 요구하느냐** — 하드웨어 고유 특성.

(1) 은 profiler (PyTorch `torch.profiler`, NSight Systems, nvprof) 로 얻을 수 있지만, (2) 는 동일 하드웨어에서 **동일 연산을 load 축으로 sweep** 해서 실측하지 않으면 알 수 없다. 이 coefficient 가 없으면:

- **총량만 측정** (nvidia-smi power × time) → 어느 연산이 비효율적인지 모름.
- **FLOPs 기반 에너지 추정** → Tensor Core vs CUDA core, precision 차이를 반영 못함.
- **DVFS / scheduler 최적화** → 각 op 의 real cost 몰라서 heuristic.

본 스위트는 (2) 를 A100/H100 양쪽에서 **통일된 코드로** 추출해 `k_op` 테이블을 만드는 것이 목적이다.

### 1.2 Analytical GPU power model 의 per-op 계수

[AccelWattch (ISCA'21, Kandiah et al.)](https://ieeexplore.ieee.org/document/9499915) 는 GPGPU-Sim 기반의 GPU power simulator 로, 다음 형태의 에너지 모델을 가진다:

```
E(kernel) = E_idle(T) + Σ_i (N_i · e_i)
```

- `E_idle` : static/leakage 부분.
- `e_i` : instruction class `i` (예: FP32_FMA, MEM_LD, REG_READ…) 당 평균 에너지.
- `N_i` : instruction counter (PTX trace / HW counter 에서 추출).

`e_i` 의 기본값은 Volta/Turing HW specs 에서 fit 되어 있으나, 새로운 SKU (Ampere A100, Hopper H100) 나 precision (FP8, BF16, TF32) 에 대해서는 **다시 fit 해야 정확하다**. 본 스위트의 `slope_dyn` 컬럼이 그 입력이다.

### 1.3 관련 선행 연구

- **nvprof / NSight Compute** : per-kernel energy counter 를 제공하지만, board-level NVML 기반이고 제한된 GPU SKU 에서만 지원.
- **Hong & Kim (ISCA '10)** : analytical power model 의 효시. Pre-Kepler.
- **GPUWattch (ISCA '13)** : GPGPU-Sim 통합 이전 세대.
- **AccelWattch (ISCA '21)** : GPGPU-Sim 4.0 통합. **본 repo 가 속한 프로젝트.**
- **LLM carbon accounting (e.g., Luccioni et al. 2023)** : 훈련 에너지 정량화. per-op cost 가 있어야 carbon attribution 이 미래 하드웨어까지 확장 가능.

### 1.4 이 스위트가 답하는 질문

1. A100 에서 `fp16 softmax` 한 element 는 몇 mJ 인가? H100 에서는?
2. Tensor Core 을 쓰지 않고 CUDA core 로만 GEMM 을 돌리면 에너지 overhead 는 몇 배인가?
3. A100 에서 FP16 TC → H100 에서 FP8 TC 로 가면 energy/FLOP 이 얼마나 줄어드는가? (≈ 2-3× 기대)
4. memory-bound elementwise 와 compute-bound matmul 의 thermal footprint (`temp_rise_c`) 차이는?
5. cross-GPU ratio 를 bandwidth ratio 로 모델링하면 elementwise 는 잘 맞는가?

## 2. 전력 모델 (Power Model)

### 2.1 1차(선형) 분해

임의의 워크로드 `W` 가 구간 `[t0, t1]` 에 실행되는 동안 NVML 이 측정한 전력 trace 를 `P(t)` 라 하자. 우리가 채택하는 **first-order** 전력 모델은:

```
P(t) = P_static + P_dyn(t)
```

양변을 `[t0, t1]` 위에서 적분:

```
E_total = ∫ P(t) dt = P_static · (t1 − t0) + ∫ P_dyn(t) dt
        ≡ E_static + E_dyn
```

동일 kernel 을 `N` 번 실행한다고 하자. 모든 op 이 통계적으로 동일한 에너지 `k_op` 를 소비하고, 시간 간섭(간섭 간 queuing overhead 등) 이 `N` 에 선형이라 가정하면:

```
E_dyn(N) = k_op · N + c
```

`c` 는 launch overhead + warmup artifact 등의 상수항이다. `N` 을 sweep 하여 linear fit 을 돌리면:

- `slope` = `k_op` (J/op) — 본 스위트의 핵심 coefficient.
- `intercept` = `c` — 유효한 측정이면 `c ≪ k_op · N_max`.

### 2.2 왜 1차인가 — 유효성의 criterion 으로서의 R²

1차 모델은 정확한 HW 동작이 아니다. 실제 GPU 는:

- DVFS 로 주파수/전압을 바꾸고,
- 일정 occupancy 를 넘으면 scheduler 가 포화되며,
- 온도에 따라 leakage 가 바뀌고,
- instruction mix 에 따라 Tensor Core 활용률이 달라진다.

그러나 **좁은 load range 안에서** — 예컨대 한 SM 의 issue bandwidth 안에 머무는 한 — 위 효과는 2차 이하로 유지된다. 그래서 우리는 **R² ≥ 0.99** 를 "이 구간에서 1차 모델이 충분히 유효" 의 판정 기준으로 삼는다:

- `R² ≥ 0.99` → `k_op` 값이 의미 있음. AccelWattch coefficient 로 업로드 가능.
- `R² ∈ [0.95, 0.99)` → 경계. sweep 범위를 좁히거나 n_points 를 늘려 재측정 권장.
- `R² < 0.95` → 비선형. launch overhead 지배, BW 포화, thermal drift 중 하나를 의심.

### 2.3 예상되는 비선형 원인

| 증상 | 진단 | 대응 |
|---|---|---|
| 저부하 구간 에너지가 음수 | `E_static` 가 `E_total` 을 넘어섬 — static 측정값이 높거나 드리프트 | static 재측정 또는 `--static-seconds` 를 늘림 |
| 곡선이 concave (상향 꺾임) | BW 포화 / cache thrash — BW 부족이면 에너지는 동일 N 에 더 듦 | load 상한 낮추거나 N_op 기반 normalization 검토 |
| 곡선이 convex (하향 꺾임) | launch overhead 지배 — 작은 N 에서 `c` 가 지배적 | n_points 를 더 높여 low-N 샘플 뺌, 또는 iterations 수 늘림 |
| 점이 산포됨 | thermal drift / DVFS oscillation | `--cooldown-min-s` 늘림, `--cooldown-c` 낮춤 |

### 2.4 P_static 측정의 중요성

`k_op` 는 `P_static` 에 의존한다. `P_static` 을 5W 잘못 측정하면 (즉 실제 60W 인데 65W 라고 보면) 모든 구간의 `E_dyn` 이 `5W · T_workload` 만큼 과소평가된다. 30 초짜리 sweep 이라면 150 J 오차 — `k_op · N_max` 대비 수 % 에 이를 수 있다.

방어책:

1. **Pre-measurement baseline** : 본 스위트는 첫 1회 `--static-seconds` (default 12 s) idle 측정으로 `P_static` 을 확정.
2. **Periodic re-check** (향후 개선 방향) : 각 cell 직전에 2 s idle 을 재측정하여 drift 감지.
3. **Baseline trace 저장** : `*_baseline.csv` 에 idle trace 를 기록, `plot_static_power()` 가 mean ± std 를 보여줌.

## 3. 벤치마크 설계

본 스위트는 15개의 microbenchmark 로 이루어진다. 10개는 **elementwise** (FP16/FP8 × 5 연산), 5개는 **matmul variant** 이다.

### 3.1 Elementwise benchmarks (10개)

elementwise 는 vector in → vector out 의 **memory-bandwidth dominated** 커널이다. 이름표기 `{dtype}_{op}` :

| ID | 연산 | 수식 | compute unit | 메모리 I/O | 연산 intensity | 비고 |
|---|---|---|---|---|---|---|
| `fp16_mul` | element-wise multiply | `y = a · b` | **CUDA core** | 2R + 1W | 1 FLOP/3 words | — |
| `fp16_add` | element-wise add | `y = a + b` | **CUDA core** | 2R + 1W | 1 FLOP/3 words | — |
| `fp16_softmax` | row-wise softmax | `y_i = exp(x_i)/Σ exp(x_j)` | **CUDA core** | ~2R + 1W (2pass) | ~5 FLOP/elem | — |
| `fp16_gelu` | GeLU activation | `y = 0.5·x·(1+tanh(…))` | **CUDA core** | 1R + 1W | ~8 FLOP/elem | — |
| `fp16_layernorm` | layer normalization | `y = (x−μ)/σ · γ + β` | **CUDA core** | 2R + 1W (2pass) | ~5 FLOP/elem | — |
| `fp8_mul` | FP8 multiply (cast in/out) | 위와 동일, dtype 만 E4M3 | **CUDA core** | 2R + 1W | 1 FLOP/3 words | **emulated** ※ |
| `fp8_add` | FP8 add | 동일 | **CUDA core** | 2R + 1W | 1 FLOP/3 words | **emulated** ※ |
| `fp8_softmax` | FP8 softmax | 동일 | **CUDA core** | 2R + 1W | ~5 FLOP/elem | **emulated** ※ |
| `fp8_gelu` | FP8 GeLU | 동일 | **CUDA core** | 1R + 1W | ~8 FLOP/elem | **emulated** ※ |
| `fp8_layernorm` | FP8 LayerNorm | 동일 | **CUDA core** | 2R + 1W | ~5 FLOP/elem | **emulated** ※ |

> ※ **emulated** : PyTorch 는 native FP8 elementwise kernel 이 없어서 `fp8 → fp16 → op → fp8` 의 cast-compute-cast 패턴을 사용한다. 이는 **어느 GPU 에서든**(A100 / H100 모두) 발생하는 오버헤드로, 측정된 에너지에는 FP16 compute 비용 + 중간 FP16 텐서 materialization 비용이 섞여 있다. 따라서 `fp8_*` elementwise 의 `k_op` 는 순수 FP8 HW 비용이 아니며, 플롯에서도 hatch (`///`) 및 `*EMU` 표시로 구분된다.

**FP8 구현 노트** : PyTorch 의 `torch.float8_e4m3fn` 은 native 연산이 제한적이라, 본 스위트는 `a.to(float8) → to(fp16) → op → to(fp8)` 의 cast-compute-cast 패턴을 사용한다. 실제 FP8 HW 경로(E4M3 native multiply)는 matmul(`fp8_te`)에서만 트리거된다. 따라서 `fp8_*` elementwise 의 `k_op` 는 "FP8 로 포맷된 tensor 에 대한 cast+compute cost" 로 해석해야 한다.

**왜 이 5 연산인가** : transformer/LLM 훈련에서 attention block 의 대부분 시간을 이들이 잡아먹는다. softmax 는 reduction 이라 단순 elementwise 보다 느리고, LayerNorm 도 2-pass reduction 이다. GeLU 는 point-wise transcendental. MUL/ADD 는 baseline.

### 3.2 Matmul variants (5개)

matmul 은 **compute dominated** 커널이며, 같은 문제를 다른 compute unit 으로 돌렸을 때의 에너지 차이를 드러낸다.

| ID | dtype | compute unit | A100 (sm_80) | H100 (sm_90) |
|---|---|---|---|---|
| `matmul_fp32_simt` | FP32 | **CUDA core** (SIMT — TF32 명시 차단) | ✓ 참조 baseline | ✓ 참조 baseline |
| `matmul_tf32_tc` | TF32 | **Tensor Core** (mma.sync m16n8k8) | ✓ native | ✓ native |
| `matmul_fp16_tc` | FP16 | **Tensor Core** (mma.sync m16n8k16) | ✓ native | ✓ native |
| `matmul_bf16_tc` | BF16 | **Tensor Core** (동일 TC 경로, mantissa만 다름) | ✓ native | ✓ native |
| `matmul_fp8_te` | FP8 (E4M3) | **Tensor Core** (Transformer Engine wrapper) | ✗ **emulated — FP16 TC fallback** ※ | ✓ native FP8 TC |

> ※ **A100 의 FP8 은 "가짜"** : Ampere (sm_80) 는 native FP8 Tensor Core 가 없으므로, Transformer Engine 이 내부적으로 **FP16 Tensor Core 경로로 자동 폴백**한다. A100 CSV 의 `matmul_fp8_te` 행은 **FP16-TC 성능 수치**이며 FP8 HW 비용을 나타내지 않는다. CSV 의 `emulated=1` 컬럼과 플롯의 hatched (`///`) bar, 그리고 `[TC·FP16-fallback]` / `*EMU` 태그로 이를 명시한다. Hopper (sm_90) 에서만 이 cell 이 의미를 가진다.

**핵심 포인트** : `matmul_fp32_simt` 는 **의도적으로 Tensor Core 를 끈다** (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.set_float32_matmul_precision("highest")`). 그래야 "TC 를 썼을 때 에너지가 얼마나 절약되는가" 의 baseline 이 생긴다.

**Transformer Engine 경로** : `fp8_te` 는 `te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(...))` context 안에서 `te.Linear` 를 호출한다. 이는 내부적으로 `cublasLtMatmul` 의 FP8 경로를 호출하며, H100 Hopper 의 FP8 Tensor Core 를 직접 사용한다. A100 에서는 미지원이므로 preflight 에서 스킵 처리.

**Matmul size sweep** : `N ∈ {512, 1024, 1536, 2048, 3072, 4096, 6144, 8192}` 를 sweep. FLOP = `2 · N³`. 메모리 footprint 는 `3 · N² · sizeof(dtype)`, N=8192 FP32 는 768 MB — 80 GB HBM 안에 충분히 들어감.

### 3.3 Load sweep 설계 원칙

- **최소 load** (`1<<17` = 128 K elem) : launch overhead 가 커질 정도로 작지 않으면서 모니터 window 안에 여러 iteration 이 들어갈 만큼 작음.
- **최대 load** (`(1<<28)+(1<<27)` = 384 M elem) : 80 GB HBM 의 1.5% 수준 — OOM 피하면서 충분히 큼.
- **9 points, 2배씩 2배 증가 + 끝점 1.5x** : log-linear 커버리지. R² 평가를 왜곡하는 dense cluster 피함.
- **`iters` 자동계산** : `target_ms / per-iter-us`. 최소 window (`--window-ms 3000`) 를 채우도록 반복 횟수 결정.

### 3.4 Cache locality regime — L2 hit rate 와 에너지

elementwise 벤치마크는 memory-bound 이므로 **working-set 이 L2 에 얹히는지 여부가 J/element 를 1 order 이상 갈라놓습니다**. 각 cell 은 working-set 크기와 탐지된 L2 용량을 비교해 세 regime 으로 자동 분류됩니다:

| `cache_regime` | Working-set 조건 | 해석 | 예상 L2 hit rate |
|---|---|---|---|
| `l2_resident` | `ws ≤ L2/2` | 여유있게 L2 안에 들어감. 두 번째 iter 부터 resident. | ≈ **100%** |
| `l2_partial` | `L2/2 < ws ≤ 2·L2` | 경계 — thrashing / 일부 hit. | ≈ **50%** (rough) |
| `dram_stream` | `ws > 2·L2` | 매 iter 마다 DRAM 에서 streaming. | ≈ **0%** |

Working-set 정의:
- `mul` / `add` : `3·N·bytes_per_elem` (a read + b read + out write)
- `gelu` / `softmax` / `layernorm` : `2·N·bytes_per_elem`
- `matmul` : `(M·K + K·N + M·N)·bytes_per_elem`. 단, matmul 은 reuse (`K` times per element) 가 있어서 대용량에서도 tile-level L2 hit 가 살아있습니다 — 라벨은 working-set 기준이므로 matmul_dram_stream 도 compute-bound 일 수 있음.

**기본 sweep 은 이미 세 regime 을 다 포함** (A100 40MB L2 기준: `N≤8M` L2-resident, `N=32M` transition, `N≥64M` DRAM-stream). 분석 시 `cache_regime` 컬럼으로 grouping 하면 cache 효과를 분리해서 볼 수 있습니다.

**3 regime 에 clean 하게 각각 1 포인트만 찍고 싶다면** `--cache-sweep` 플래그를 쓰면 각 (op, dtype) 마다 정확히 3 개 load size (L2/8, L2, 8·L2 working-set) 만 실행합니다:

```bash
python3 gpu_power_bench.py --device 0 --cache-sweep --tag h100_cache --out-dir reports/
```

출력은 평소처럼 CSV 에 쌓이고, `analyze.py` 는 `_cache_regime.png` 플롯을 새로 그립니다 (§6.5.1 참조).

## 4. 측정 방법론

### 4.1 NVML power telemetry

NVML (NVIDIA Management Library) 은 `nvmlDeviceGetPowerUsage()` 로 board-level 전력(단위 mW)을 반환한다. 중요한 특성:

- **Update rate** : 내부적으로 약 **20 Hz** (50 ms 간격) 로 HW sensor 를 샘플링한다. 본 스위트는 100 Hz 로 폴링하지만, 실제 고유 샘플은 50 ms 마다 업데이트된 값이다 (중복 리턴). 이는 Aliasing 을 최소화하기 위한 oversampling.
- **Board-level** : GPU core 뿐 아니라 HBM, NVLink, VRM 효율 손실까지 포함한다. 즉 `k_op` 는 "HW 전체가 연산 하나 당 소비한 에너지" 를 의미한다.
- **Latency** : `nvmlDeviceGetPowerUsage()` 호출 자체는 ~50 µs. 100 Hz 에서 CPU 점유는 무시 가능.

### 4.2 Window 구조 — 한 cell 의 타임라인

```
[ idle 0.5 s ] [ WORKLOAD (--window-ms 3000) ] [ trailing 0.5 s ]
      ↓                     ↓                         ↓
  pre-baseline        적분 구간                 post-baseline
      (drift check)       (E_total)              (drift check)
```

- 측정 시작 전 0.5 s idle — monitor thread 안정화 목적.
- `--window-ms` 동안 kernel 을 연속 실행 — loop iter 단위로 launch/wait.
- 종료 후 trailing idle — NVML 의 sensor delay 흡수.

적분은 kernel-active 구간만 떼어서 trapezoidal rule 로 계산:

```python
E_total = np.trapz(power, t)      # W·s = J
E_static = P_static · (t[-1] − t[0])
E_dyn    = E_total − E_static
```

### 4.3 Cooldown — thermal drift 방지

GPU 가 뜨거워지면 leakage current 가 증가하여 `P_static` 이 올라간다. 이는 다음 cell 의 `E_dyn` 을 과소평가하게 만든다. 본 스위트는 **각 cell 사이에** 아래 조건을 만족할 때까지 대기한다:

1. **min_s** (`--cooldown-min-s`, default **5 s**) : 최소 이 시간은 무조건 대기. HBM/VRM 잔열이 센서에 반영되려면 수 초 필요.
2. **target_c** (`--cooldown-c`, default **45 °C**) : GPU core 온도가 이 값 이하로 떨어질 때까지 대기.
3. **timeout** (`--cooldown-timeout`, default **180 s**) : 이 시간 안에 target 에 못 미치면 포기하고 진행 (warning 로그). 공용 서버처럼 ambient 가 높을 때의 안전장치.

Cooldown 로그는 각 cell 행 (`cooldown_elapsed_s`, `cooldown_reached`) 에 기록되어 사후 검토가 가능하다.

### 4.4 Preflight check

`preflight.py` 는 benchmark 전에 다음을 확인한다:

- CUDA available, SM capability ≥ 7.0.
- Persistence mode on (privilege 가 있으면 `nvidia-smi -pm 1`).
- `pynvml`, `nvtx`, `pandas`, `matplotlib` import 가능.
- FP8 (E4M3/E5M2) dtype 지원 (PyTorch ≥ 2.1).
- Transformer Engine 의 **실제 설치 여부** (meta-package 만 있으면 스킵 + 경고).
- 충분한 VRAM (최대 matmul size 기준 필요량 × 3).

실패하는 경우 해당 cell 만 스킵되고 이유가 CSV `error` 컬럼에 기록된다.

### 4.5 측정 불확실성 (Measurement uncertainty)

`k_op` 의 주된 불확실성 원천:

| 원천 | 크기 | 감쇄 방법 |
|---|---|---|
| NVML quantization (1 mW) | ±0.001 W | 무시 가능 |
| Sensor update jitter (50 ms) | ±2% at short window | `--window-ms` 늘림 |
| Thermal drift (idle → hot) | 2-5 W over 30 s | Cooldown 강제 |
| Cast overhead in FP8 path | 수 % | 해석 단계에서 주의 |
| OS-level power cap (DVFS) | 가변 | `nvidia-smi -pl` 로 고정 권장 |

regression 의 standard error (`slope_err_W`) 를 CSV 에 포함 — 이를 통해 ±2σ 구간을 그릴 수 있다.

## 5. 분석 방법

`analyze.py` 는 각 cell CSV 를 받아 다음을 수행한다.

### 5.1 Per-cell linear fit

각 CSV 파일에는 `N_op`, `E_total_J`, `E_dyn_J`, `T_workload_s` 등이 load 별로 기록되어 있다. `scipy.stats.linregress` (또는 numpy polyfit) 로:

```
E_dyn = k_op · N_op + c
```

를 fit 하고 `(k_op, intercept, r2, slope_err)` 를 뽑는다.

- `k_op` 단위: elementwise 는 J/element, matmul 은 J/FLOP.
- J/FLOP 을 얻으려면 matmul 의 N_op 는 `2·M·N·K` 로 계산 (multiply-accumulate = 2 FLOP 관례).

### 5.2 Total-energy regression vs dynamic-energy regression

두 가지 fit 모두 수행한다:

- **dynamic** (`slope_dyn`) : `E_dyn` vs `N_op`. `P_static` 을 빼고 fit. → **이 값이 `k_op` 로 간주된다.**
- **total** (`slope_tot`) : `E_total` vs `N_op`. `P_static` 포함. 시간 의존성까지 포함된 "체감 에너지". intercept 가 `P_static · T_avg` 에 가까우면 모델 일관.

두 slope 가 `slope_dyn ≈ slope_tot − P_static · dT/dN` 관계에 있다면 static 분리는 성공. 크게 다르면 tracking 실패 — static 재측정 필요.

### 5.3 Cross-GPU normalization

`compare_gpus.py` 는 A100 과 H100 의 `k_op` 를 같은 cell_id 단위로 비교한다:

- **ratio** = `k_op^H100 / k_op^A100`
- **log-scale scatter** — 순서 보존.
- **bandwidth-normalized ratio** = `ratio · (BW_H100/BW_A100)` — elementwise 에서 1 근처면 BW 로 설명 가능.

### 5.4 Sanity checks

- R² ≥ 0.99 인 cell 만 coefficient table 의 primary row 로 올림.
- intercept / (k_op · N_max) > 5% 면 launch overhead 가 무시 못 할 수준 — warning.
- slope < 0 은 불가능한 값 — sign flip 시 자동 에러.

## 6. 차트 해석 가이드

`analyze.py` 가 생성하는 plot 은 크게 7 종류 (single-GPU) + 3 종류 (cross-GPU 비교) 다.

### 6.1 `energy_vs_load.png` — 1차 모델 유효성 검증

**무엇을 보여주나** : x축 `N_op`, y축 `E_dyn_J`. 각 cell 당 9개 점과 fit line.

**어떻게 읽나** :
- 점들이 선 위에 잘 얹혀 있으면 (R² ≥ 0.99) 1차 모델 유효.
- 저부하 쪽에서 꺾이면 launch overhead 지배 — low-N 샘플 제외 재fit 고려.
- 고부하 쪽에서 기울기가 올라가면 BW 포화 — 해당 영역 제외.

### 6.2 `joule_per_op.png` — bar chart of `k_op`

**무엇을 보여주나** : 각 cell_id 의 `slope_dyn` (J/op) bar.

**어떻게 읽나** :
- FP16 MUL vs ADD 비교 → raw compute cost 차이 거의 없어야 함 (둘 다 동일 HW path).
- softmax/LayerNorm 이 MUL/ADD 보다 4-6 배 높으면 reduction 의 cost 가 잘 잡힘.
- matmul 은 J/FLOP 단위 → 범위가 다르다 (훨씬 작음), 별도 subplot 또는 log scale 주의.

### 6.3 `dyn_power.png` — 평균 동적 전력 per cell

**무엇을 보여주나** : 각 cell 의 `P_dyn_avg_W` (= `E_dyn / T_workload`) bar.

**어떻게 읽나** :
- matmul TC 경로가 SIMT 보다 총 전력은 높지만 시간은 훨씬 짧다 (J/FLOP 은 낮음).
- elementwise 는 BW 한계 때문에 cell 간 차이가 크지 않음. 차이가 크면 cast overhead 의심.

### 6.4 `static_power.png` — P_static 측정 시각화

**무엇을 보여주나** : 세로 3 panel (위에서 아래로).
- Top (A): baseline idle trace (시간 vs power). mean ± σ band 와 함께. flat 일수록 좋음.
- Middle (B): 각 cell 의 static (회색) + dyn stacked bar. **각 sweep 그룹 위에 대괄호로 어떤 벤치 (`fp16·mul`, `matmul_fp16_tc`, …) 를 돌렸는지 라벨을 표시**하고, 그룹 사이에는 점선 구분자를 그린다.
- Bottom (C): `E_static / E_total` 비율 — load 가 크거나 kernel 이 효율적일수록 비율 낮음. x축 label 은 45° 기울여서 겹침을 방지. sweep cell 수가 많으면 plot 폭이 자동으로 늘어난다 (최대 32 inch).

**어떻게 읽나** :
- idle trace stdev/mean > 5% 면 측정 환경 불안정 — background process 의심.
- static share 가 50% 를 넘으면 "kernel 이 idle 보다 약간 바쁜 수준" — load 상한 늘림.

### 6.4.1 `cache_regime.png` — L2 hit rate 별 J/element

**무엇을 보여주나** : 좌 panel = 개별 cell 을 `(l2_resident / l2_partial / dram_stream)` x축에 strip plot 으로, 우 panel = (op × regime) 의 J/element median bar.

**어떻게 읽나** :
- **좌 panel 의 수직 gap 이 곧 cache miss 의 에너지 비용** — 같은 op 가 DRAM-stream 에서 L2-resident 대비 몇 배 비싼지 한 눈에. mul/add 에서 5-10x 정도가 일반적.
- 점이 세 regime 사이에 "계단" 모양이 아니라 "비스듬한 선" 으로 퍼지면 working-set 경계가 L2 에 가깝다는 뜻 — transition 지점 근처에서 sub-regime 변동이 있다는 신호.
- reduction op (softmax / layernorm) 는 elementwise 에 비해 세 regime 간격이 좁게 나옵니다. reduction 은 compute overhead 가 BW 만큼 차지하기 때문.
- **fp8** 는 cast-compute-cast 때문에 regime 에 관계없이 fp16 대비 높게 나옵니다 — `--include-emulated` 로 비교.

### 6.5 `temperature.png` — 열 특성

**무엇을 보여주나** : 3 panel.
- 좌: 각 cell 의 start/avg/peak temperature bar.
- 중: `cooldown_elapsed_s` — 실제로 target 에 도달하는 데 걸린 시간.
- 우: `J/op` vs `peak_temp_c` scatter — 상관 있으면 thermal-dependent regime.

**어떻게 읽나** :
- peak temp 가 일관되게 85 °C 이상이면 thermal throttle 가능성 → DVFS 왜곡 우려.
- cooldown 시간이 timeout 에 거의 닿으면 ambient 가 너무 높음 — 측정 환경 재고.

### 6.6 `r2_heatmap.png` — fit 품질 한눈에

**무엇을 보여주나** : cell_id × {dyn, total} grid. 색은 R².

**어떻게 읽나** : 빨강(R² ≥ 0.99) 은 믿을 수 있는 coefficient. 노랑/파랑은 재측정 대상.

### 6.7 `power_trace.png` — 대표 trace

**무엇을 보여주나** : 대표 cell (예: max load) 의 실제 P(t) 과 target window 를 얹어 그린다.

**어떻게 읽나** : 완만한 plateau 가 있으면 samping 이 sufficient. spike 가 대부분이면 window 를 늘려야 함.

### 6.8 Cross-GPU : `cmp_joule.png`, `cmp_ratio.png`, `cmp_bw_normalized.png`

- `cmp_joule.png` : A100 vs H100 의 `k_op` 를 같은 x축에 나란히 bar.
- `cmp_ratio.png` : `k_op^H100 / k_op^A100` — 1 미만이면 H100 이 효율적.
- `cmp_bw_normalized.png` : elementwise 만 해당. BW ratio (2039 GB/s vs 2039 GB/s 등) 로 나눈 뒤 1 근처면 "BW 만으로 설명 가능".

## 7. 예상 결과 (A100 vs H100)

**참고용 order-of-magnitude** 값 (실측은 환경/드라이버에 따라 ±30%):

| cell | A100 k_op | H100 k_op | ratio | 비고 |
|---|---|---|---|---|
| `fp16_mul` | ~6 pJ/elem | ~4 pJ/elem | 0.67 | BW 개선 주도 |
| `fp16_softmax` | ~40 pJ/elem | ~28 pJ/elem | 0.70 | reduction cost 비슷 |
| `fp16_gelu` | ~25 pJ/elem | ~18 pJ/elem | 0.72 | transcendental 약간 빠름 |
| `fp16_layernorm` | ~35 pJ/elem | ~25 pJ/elem | 0.71 | 2-pass |
| `fp8_*` | ~12-45 pJ/elem | ~8-30 pJ/elem | 0.6-0.7 | cast cost 포함 |
| `matmul_fp32_simt` | ~3 pJ/FLOP | ~2.5 pJ/FLOP | 0.8 | TC 안씀 |
| `matmul_tf32_tc` | ~0.5 pJ/FLOP | ~0.3 pJ/FLOP | 0.6 | |
| `matmul_fp16_tc` | ~0.25 pJ/FLOP | ~0.14 pJ/FLOP | 0.56 | |
| `matmul_bf16_tc` | ~0.25 pJ/FLOP | ~0.14 pJ/FLOP | 0.56 | fp16_tc 와 거의 동일 |
| `matmul_fp8_te` | *(FP16 TC 폴백, emulated)* | ~0.07 pJ/FLOP | — | A100 수치는 실제로 FP16 TC |

핵심 관찰:

- **FP16 TC → FP8 TE** : ~2× 에너지 절감 — Hopper 의 핵심 세일즈 포인트.
- **SIMT vs TC** : 10-20× 차이 — Tensor Core 쓰지 않으면 에너지 낭비 심각.
- **Elementwise BW normalize ratio** ≈ 1 이면 모델 검증 성공.
- **⚠️ FP8 elementwise ≥ FP16 elementwise 는 A100 / H100 모두에서 정상** : PyTorch 에 native FP8 elementwise kernel 이 없고 H100 의 FP8 실리콘은 Tensor Core (matmul) 에만 있다. 따라서 fp8_{mul,add,softmax,gelu,layernorm} 는 양 GPU 모두 `fp8 → fp16 → op → fp8` cast-compute-cast 로 실행되며, 4 개 커널 + FP16 중간 텐서 materialization 때문에 **FP16 보다 더 비싸게** 나온다. 이건 실험 버그가 아니라 SW 한계를 정직하게 반영한 값이다.
- **⚠️ FP8 matmul 은 H100 에서만 의미가 있다** : `matmul_fp8_te` 는 **A100 에서 Transformer Engine 이 FP16 TC 경로로 자동 폴백**한다. 따라서 A100 CSV 의 `matmul_fp8_te` 수치는 `matmul_fp16_tc` 와 같은 HW 경로 (= 거의 같은 값) 여야 정상이며, `emulated=1` / `compute_unit="Tensor Core (FP16 fallback)"` 로 플래그된다. H100 에서만 native FP8 TC 경로가 활성화되어 FP16 TC 대비 ~2× 절감을 보인다. `analyze.py` 는 기본적으로 emulated 행을 플롯에서 숨기므로, A100 플롯에는 `matmul_fp8_te` bar 가 나타나지 않는 것이 정상 (볼 필요가 있으면 `--include-emulated`).

## 8. 설치 & 사전 점검

### 8.1 요구 조건

- Linux, NVIDIA GPU (sm_70 이상 — Volta 이상).
- Python 3.9+.
- CUDA driver ≥ 525 (H100 FP8 은 ≥ 535 권장).
- NVML (드라이버에 기본 포함).

### 8.2 의존성 설치

```bash
cd util/gpu_power_bench
python3 -m pip install -r requirements.txt
```

`requirements.txt` 는 torch, nvidia-ml-py (pynvml), nvtx, matplotlib, pandas, scipy 를 포함한다.

### 8.3 Transformer Engine (H100 FP8 용, 선택)

H100 에서 `matmul_fp8_te` 를 실행하려면 Transformer Engine 이 필요하다. 일반 pip 설치는 meta-package 만 들어가고 실제 모듈은 빠져 제대로 동작하지 않을 수 있다. 동봉된 helper 를 사용:

```bash
./install_transformer_engine.sh
```

이 스크립트는 다음을 수행한다:

- CUDA toolkit / nvcc 존재 확인.
- PyTorch 가 CUDA build 인지 확인.
- `--no-build-isolation` + `[pytorch]` extra 로 소스 빌드 (meta-package 함정 회피).
- 빌드 후 실제로 `te.Linear` + `fp8_autocast` forward 를 태워보고 torch backend `.so` 로드까지 검증 (단순 `import` 는 lazy 로딩 때문에 통과하므로 부족).

#### 8.3.1 Troubleshooting: `could not find shared object file for transformer engine torch lib`

`matmul_fp8_te` 셀 (H100 에서 8 K 값 × 1 variant = 8 cells) 이 전부 `build failed` 로 스킵되고 위 에러가 나온다면 **TE Python 모듈은 import 되지만 torch backend 공유 라이브러리 (`libtransformer_engine_torch.so`) 가 로딩되지 않는 상태**다. 원인 대부분은 다음 중 하나:

1. **meta-package 설치** : `pip install transformer-engine` 만 하고 `[pytorch]` extra 를 안 붙임. `transformer_engine.pytorch` import 는 성공할 수도 있지만 `te.Linear()` 호출 시점에 `.so` 로딩이 실패.
2. **torch 버전 불일치** : TE wheel 이 빌드된 torch 와 현재 설치된 torch ABI 가 달라서 prebuilt `.so` 가 load 되지 않음. 예: `torch==2.3` 으로 빌드된 TE wheel + `torch==2.1` 환경.
3. **CUDA lib path 문제** : `LD_LIBRARY_PATH` 에 CUDA runtime libs 가 없거나 cudnn-dev 미설치.

해결:

```bash
# 가장 확실한 방법 — 현재 torch 에 맞춰 TE 를 강제 소스 재빌드
./install_transformer_engine.sh                         # 이 스크립트 자체가 재빌드 + runtime probe 포함

# 동등한 수동 명령:
pip install --force-reinstall --no-build-isolation 'transformer-engine[pytorch]'
```

재설치 후 `preflight.py` 를 다시 돌려 "transformer_engine" 항목이 **버전 문자열** 만 표시하는지 확인한다 (`torch-backend BROKEN (...)` 이 뜨면 아직 문제 있는 것). 문제가 해결될 때까지 H100 FP8 native TC 수치는 수집되지 않고, CSV 에는 `matmul_fp16_tc` 와 기타 variants 만 남는다.

### 8.4 환경 권장 사항

- `sudo nvidia-smi -pm 1` : persistence mode.
- 가능하면 `sudo nvidia-smi -lgc <freq>` : 주파수 고정 — DVFS 변동 제거.
- 백그라운드 프로세스 최소화 (Xorg, 다른 CUDA 작업 금지).
- 공기 흐름이 좋은 서버 권장 (cooldown 시간 단축).

## 9. 실행

### 9.1 기본 실행

```bash
./run_bench.sh
```

GPU 0 에서 full sweep (15 cells × 9 loads = ~30-40 분) 을 돈다.

### 9.2 다중 GPU / 태깅

```bash
./run_bench.sh --device 0 --tag a100
./run_bench.sh --device 1 --tag h100
```

`--tag` 는 출력 파일 이름과 메타데이터에 붙는다.

### 9.3 Quick 모드

```bash
./run_bench.sh --quick
```

`--quick` 은 n_points=5, window=1500ms 로 짧게. 완전 검증용이 아니라 pipeline smoke test 용.

### 9.4 주요 옵션

```
--device N              대상 GPU index
--tag STR               출력 파일 태그 (예: a100, h100)
--window-ms MS          측정 window (default 3000)
--static-seconds S      idle baseline 측정 시간 (default 12)
--cooldown-c TEMP       cooldown 목표 온도 (default 45)
--cooldown-min-s S      cooldown 최소 대기 시간 (default 5)
--cooldown-timeout S    cooldown 최대 대기 (default 180)
--loads N1 N2 ...       elementwise load list (manual override)
--matmul-sizes N1 ...   matmul M=N=K list
--matmul-variants ...   e.g. `fp8:te bf16:tc` (default: 5 variants 전부)
--no-elementwise        elementwise sweep 생략 (재실행 / matmul-only 용)
--no-matmul             matmul sweep 생략
--skip-preflight        preflight check 우회 (이미 통과했을 때)
--out-dir DIR           output 디렉토리 (default reports/)
```

#### 9.4.1 실패한 cell 다시 돌리기 (예: TE 설치 수정 후 `matmul_fp8_te` 재측정)

기본 sweep 은 **130 cells** 로 구성됩니다:
- elementwise 90 = 2 dtypes (fp16, fp8) × 5 ops × 9 loads
- matmul 40 = 5 variants × 8 K sizes

순서대로 번호를 매기면 **cell 123–130 이 정확히 `matmul_fp8_te`** 의 K ∈ {512, 1024, 1536, 2048, 3072, 4096, 6144, 8192} 8 개입니다. TE 설치 문제로 이 8 개만 스킵됐다면 전체 130 cells 를 재측정할 필요 없이 이 부분만 다시 돌리면 됩니다 — 약 30 분 → 3 분으로 단축.

**Step 1 — TE 재설치 + preflight 통과 확인**

```bash
cd util/gpu_power_bench
./install_transformer_engine.sh        # 끝에 runtime probe 까지 성공해야 함
python3 preflight.py                   # "transformer_engine: <version>" 가 찍혀야 정상
```

**Step 2 — matmul fp8_te 8 cells 만 재실행**

```bash
python3 gpu_power_bench.py \
    --device 0 \
    --no-elementwise \
    --matmul-variants fp8:te \
    --tag h100_fp8te_redo \
    --out-dir reports/
```

출력: `reports/gpu_power_bench_<slug>_<stamp>_h100_fp8te_redo.csv` (8 rows) + baseline / samples sidecar. 런타임은 **Thermal cooldown 포함 약 3–5 분**.

**Step 3 — 기존 CSV 와 병합**

새 CSV 를 원본과 합쳐서 `analyze.py` 에 먹이려면 간단히 pandas 로 concat 하면 됩니다. 원본의 `matmul_fp8_te` 행은 비어 있으므로 중복도 없습니다.

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

orig = Path("reports/gpu_power_bench_<slug>_<stamp>_h100.csv")       # 수정: 원본 파일명
redo = Path("reports/gpu_power_bench_<slug>_<stamp>_h100_fp8te_redo.csv")  # 수정

a = pd.read_csv(orig)
b = pd.read_csv(redo)
# redo 쪽 matmul_fp8_te 행만 뽑아 원본에 append (원본에 이미 있으면 교체)
a = a[a["variant"] != "matmul_fp8_te"]
merged = pd.concat([a, b[b["variant"] == "matmul_fp8_te"]], ignore_index=True)
out = orig.with_name(orig.stem + "_merged.csv")
merged.to_csv(out, index=False)
print(f"wrote {out}  ({len(merged)} rows)")
PY
```

**Step 4 — 병합된 CSV 로 분석**

```bash
python3 analyze.py reports/gpu_power_bench_<slug>_<stamp>_h100_merged.csv
```

> 참고: `static_power_w` 는 각 run 시작 시점에 측정되므로 원본 / 재실행 row 사이에 수 W 차이가 날 수 있습니다. `dyn_energy_j` 는 각자의 baseline 으로 이미 정규화되어 있어 분석에 영향 없습니다. 크게 차이나면 (`>10%`) 장비 상태가 변했다는 신호이므로 병합 대신 재실행 분을 독립적으로 보는 게 안전합니다.

### 9.5 분석 단계

전체 파이프라인은 **"sweep → per-GPU analyze → cross-GPU compare"** 의 3 단계로 진행됩니다. A100 과 H100 은 별개의 GPU 이므로 각각의 GPU 에서 sweep 을 돌려야 하며, 동일한 `reports/` 디렉토리에 누적하는 것을 권장합니다 (이름 prefix 로 자동 구분됨).

#### 9.5.1 Step 1 — GPU 별 sweep 실행

A100 머신에서:

```bash
cd util/gpu_power_bench
./run_bench.sh --device 0 --tag a100
# → reports/gpu_power_bench_a100_80gb_<stamp>_a100.csv
# → reports/gpu_power_bench_a100_80gb_<stamp>_a100_baseline.csv
# → reports/gpu_power_bench_a100_80gb_<stamp>_a100_baseline_stats.csv
# → reports/gpu_power_bench_a100_80gb_<stamp>_a100_samples.csv
```

H100 머신에서 (같은 `--tag` 규약만 유지하면 ok):

```bash
./run_bench.sh --device 0 --tag h100
# → reports/gpu_power_bench_h100_sxm_<stamp>_h100.csv
# → 및 동일한 3 종 sidecar
```

파일 이름의 `_<tag>.csv` 접미사가 뒤에 오는 `analyze.py --tag` 검색 키로 쓰입니다. 여러 번 돌리면 같은 태그로 여러 timestamp 가 쌓이며, `analyze.py` 는 **가장 최근 수정된 파일**을 자동 선택 (여러 개 매칭 시 상위 5개를 콘솔에 로그).

#### 9.5.2 Step 2 — per-GPU 분석 (plot 생성)

두 가지 호출 방식 중 편한 쪽을 고르면 됩니다.

**방식 A** — `--reports-dir` + `--tag` (권장):

```bash
python3 analyze.py --reports-dir reports/ --tag a100
# → reports/a100/  아래에 모든 PNG + summary CSV 생성
python3 analyze.py --reports-dir reports/ --tag h100
# → reports/h100/  아래에 생성
```

`--tag` 를 함께 주면 출력이 자동으로 `<reports-dir>/<tag>/` 로 분리됩니다 (A100 과 H100 plot 이 섞이지 않도록).

**방식 B** — CSV 를 직접 지정:

```bash
python3 analyze.py reports/gpu_power_bench_h100_sxm_20260421_123456_h100.csv
# → reports/  (CSV 와 같은 디렉토리) 아래에 PNG 들 생성
```

이 방식에선 `--out-dir` 로 직접 출력 위치를 바꿀 수 있습니다.

**각 per-GPU 분석에서 나오는 파일** (`<stem>` = CSV 파일명에서 `.csv` 뺀 부분):

| 파일 | 내용 |
|---|---|
| `<stem>_summary.csv` | cell 당 1 행, `slope_dyn` / `R2_dyn` / `compute_unit` / `emulated` 등 집계 |
| `<stem>_linearity_elementwise.png` | elementwise 10 종 log-log 선형성 + wall time + J/elem |
| `<stem>_linearity_matmul.png` | matmul 5 variant log-log — `[CUDA]` · `[TC]` 태그 + 각 point 의 swept K 와 J/FLOP 값 annotate |
| `<stem>_joule_per_op_bar.png` | bar chart (좌: elementwise, 우: matmul) |
| `<stem>_cache_regime.png` | L2-resident / L2-partial / DRAM-stream regime 별 J/element (strip + median bar) |
| `<stem>_static_power.png` | 3 패널 P_static 진단 (idle trace + 구성비 + 점유율) |
| `<stem>_temperature.png` | 3 패널 thermal 진단 (start/avg/peak + cooldown + J/op vs T) |
| `<stem>_timeline.png` | 전체 run 의 power/temp/clock 타임라인 (samples CSV 존재 시) |

콘솔에는 summary 표가 다음 컬럼으로 출력됩니다:

```
category  variant              compute_unit                   emulated  n_points  fit_axis   slope_dyn  R2_dyn
elementwise fp16_mul           CUDA core                      0         9         J/element  6.21e-12   0.997
elementwise fp8_mul            CUDA core                      1         9         J/element  1.43e-11   0.993   (cast-compute-cast emulated)
matmul     matmul_fp16_tc      Tensor Core                    0         8         J/FLOP     2.41e-13   0.998
matmul     matmul_fp8_te       Tensor Core                    0         8         J/FLOP     1.12e-13   0.996   (H100 native)
matmul     matmul_fp8_te       Tensor Core (FP16 fallback)    1         8         J/FLOP     2.38e-13   0.997   (A100 — emulated)
```

`emulated = 1` 행의 기본 처리는 **카테고리별로 다릅니다**:

| 카테고리 | 기본 플롯 노출 | 이유 |
|---|---|---|
| elementwise (fp8_{mul,add,softmax,gelu,layernorm}) | **숨김** | PyTorch 의 native FP8 elementwise 커널 부재로 인한 cast-compute-cast 오버헤드. FP16 bar 와 나란히 그리면 착시 유발. |
| matmul (`matmul_fp8_te` A100 폴백) | **노출** (hatched + `*EMU` + `[TC·FP16-fallback]` 태그) | fp16_tc 와 같은 값에 수렴해야 정상 — 이 수렴 여부 자체가 TE 폴백이 제대로 동작했다는 sanity check 가 된다. |

즉 A100 에서도 `_linearity_matmul.png`, `_joule_per_op_bar.png` 에 `matmul_fp8_te` bar 가 그려지고, H100 의 native FP8 수치와 시각적으로 직접 비교할 수 있습니다. cross-GPU 플롯 (`compare_gpus.py`) 에서도 `matmul_fp8_te` 가 두 GPU 모두 bar 로 나타나며, A100 쪽은 hatch + `*EMU` 주석으로 폴백임을 명시합니다.

**Summary CSV 에는 두 카테고리 모두 그대로 남깁니다** (`_summary.csv`). 숨겨진 fp8 elementwise 까지 플롯에 포함하려면:

```bash
# fp8 elementwise cast-compute-cast bar 까지 플롯에 포함
python3 analyze.py --reports-dir reports/ --tag a100 --include-emulated
```

배제 발생 시 콘솔에 다음과 같은 로그가 찍힙니다:

```
[filter] hiding 5 emulated elementwise variants (45 rows) from plots — emulated matmul stays visible. Pass --include-emulated to show elementwise fp8 too. Full data: ..._summary.csv.
```

#### 9.5.3 Step 3 — 두 GPU 교차 비교

두 GPU 의 per-cell CSV 를 함께 `compare_gpus.py` 에 넘기면 됩니다. 인자는 **positional CSV path** 방식입니다 (이전 문서의 `--a100-dir` / `--h100-dir` 는 잘못된 표기였음):

```bash
python3 compare_gpus.py \
    reports/gpu_power_bench_a100_80gb_20260421_*_a100.csv \
    reports/gpu_power_bench_h100_sxm_20260421_*_h100.csv \
    --baseline a100_80gb \
    --out-dir reports/compare \
    --tag v1
```

- `--baseline` : ratio 계산의 분모가 될 GPU 이름 (CSV 의 `gpu` 컬럼 값 중 하나). 미지정 시 첫 번째 CSV 의 GPU.
- `--out-dir` : 결과 PNG/CSV 저장 위치 (default `reports/`).
- `--tag` : 출력 파일명에 붙는 식별자 (여러 실험 비교 시).

**생성되는 파일**:

| 파일 | 내용 |
|---|---|
| `gpu_compare_<stamp>_<tag>_summary.csv` | (variant × GPU) 슬로프 + ratio 표 |
| `gpu_compare_<stamp>_<tag>_bar.png` | variant 별 J/op grouped bar (GPU 별 색) |
| `gpu_compare_<stamp>_<tag>_heatmap.png` | ratio heatmap (녹색 < 1 = 효율 ↑) |
| `gpu_compare_<stamp>_<tag>_static.png` | GPU 별 P_static 비교 |

#### 9.5.4 한 GPU 만 있을 때

A100 만 있거나 H100 만 있을 때는 Step 3 을 건너뛰고 Step 1–2 로 종료합니다. 결과 해석에는 `_summary.csv` 의 `slope_dyn` 컬럼과 `_joule_per_op_bar.png` 가 핵심입니다.

#### 9.5.5 전체 디렉토리 구조 (참조)

권장 워크플로우를 따르면 `reports/` 는 다음 형태가 됩니다:

```
reports/
├── gpu_power_bench_a100_80gb_20260421_120000_a100.csv           # A100 per-cell
├── gpu_power_bench_a100_80gb_20260421_120000_a100_baseline.csv
├── gpu_power_bench_a100_80gb_20260421_120000_a100_baseline_stats.csv
├── gpu_power_bench_a100_80gb_20260421_120000_a100_samples.csv
├── gpu_power_bench_h100_sxm_20260421_140000_h100.csv            # H100 per-cell
├── gpu_power_bench_h100_sxm_20260421_140000_h100_baseline.csv
├── gpu_power_bench_h100_sxm_20260421_140000_h100_baseline_stats.csv
├── gpu_power_bench_h100_sxm_20260421_140000_h100_samples.csv
├── a100/                                                        # Step 2 output (A100)
│   ├── <stem>_summary.csv
│   ├── <stem>_linearity_elementwise.png
│   └── … (다른 6 종 PNG)
├── h100/                                                        # Step 2 output (H100)
│   └── … (동일 7 종)
└── compare/                                                     # Step 3 output
    ├── gpu_compare_20260421_150000_v1_summary.csv
    ├── gpu_compare_20260421_150000_v1_bar.png
    ├── gpu_compare_20260421_150000_v1_heatmap.png
    └── gpu_compare_20260421_150000_v1_static.png
```

## 10. 산출 파일 레퍼런스

### 10.1 Per-cell CSV : `<tag>_<cell_id>.csv`

| column | unit | 설명 |
|---|---|---|
| `cell_id` | — | e.g. `fp16_softmax` |
| `compute_unit` | — | `"CUDA core"` \| `"Tensor Core"` \| `"Tensor Core (FP16 fallback)"` — 실제 FLOP 을 수행한 HW path |
| `emulated` | 0/1 | 1 이면 해당 cell 은 native 경로가 아님 (fp8 elementwise 전부, A100 의 `matmul_fp8_te`) |
| `cache_regime` | — | `"l2_resident"` \| `"l2_partial"` \| `"dram_stream"` \| `"unknown"` — working-set 과 L2 용량 비교로 자동 분류 (§3.4 참조) |
| `load` | elem or N | elementwise: element count; matmul: M=N=K |
| `iters` | — | 자동 계산된 반복 횟수 |
| `N_op` | count | elementwise: load · iters; matmul: 2·N³·iters |
| `T_workload_s` | s | window 지속 시간 |
| `E_total_J` | J | `∫ P(t) dt` |
| `E_static_J` | J | `P_static · T_workload_s` |
| `E_dyn_J` | J | `E_total − E_static` |
| `P_avg_W` | W | `E_total / T_workload_s` |
| `P_dyn_avg_W` | W | `E_dyn / T_workload_s` |
| `start_temp_c` | °C | kernel 직전 GPU 코어 온도 |
| `avg_temp_c` | °C | window 평균 |
| `peak_temp_c` | °C | window peak |
| `temp_rise_c` | °C | `peak − start` |
| `cooldown_elapsed_s` | s | 이 cell 직전 cooldown 대기 시간 |
| `cooldown_reached` | bool | target_c 도달 여부 |
| `error` | str | 예외 발생 시 사유 |

### 10.2 집계 CSV : `<tag>_summary.csv`

cell 단위로 `slope_dyn`, `slope_tot`, `intercept_dyn`, `intercept_tot`, `r2_dyn`, `r2_tot`, `slope_err_W` 를 집계.

### 10.3 Baseline 측정 산출 : `<tag>_baseline.csv`, `<tag>_baseline_stats.csv`

- `baseline.csv` : idle trace. 컬럼 `t_s, power_w, temp_c`. 12 초 × 100 Hz = ~1200 row.
- `baseline_stats.csv` : `{mean_w, std_w, min_w, max_w, duration_s}`.

### 10.4 Plot 파일

- `energy_vs_load.png`, `joule_per_op.png`, `dyn_power.png`
- `static_power.png`, `temperature.png`
- `r2_heatmap.png`, `power_trace.png`
- `cmp_joule.png`, `cmp_ratio.png`, `cmp_bw_normalized.png`

### 10.5 Log

`<tag>_run.log` — 실행 시각, 각 cell 의 cooldown 시작/종료, 예외 등이 순서대로 기록.

## 11. 권장 워크플로우

1. `preflight.py` 를 먼저 실행하여 의존성·GPU 상태 확인.
2. `--quick` 으로 smoke test — 전체 pipeline 이 정상 동작하는지 5분 내 확인.
3. 본 측정: A100 과 H100 에서 각각 full sweep.
4. `analyze.py` 로 per-GPU plot 생성. R² 낮은 cell 재측정.
5. `compare_gpus.py` 로 cross-GPU 비교.
6. `summary.csv` 를 AccelWattch coefficient table 업데이트에 사용.

## 12. 유효성 체크리스트

측정 결과를 신뢰하려면 **모두** 충족해야 한다:

- [ ] 모든 cell 의 `r2_dyn ≥ 0.99`.
- [ ] `slope_dyn > 0` (당연하지만).
- [ ] intercept / (slope · N_max) < 5%.
- [ ] idle baseline stdev/mean < 5%.
- [ ] `peak_temp_c` 모든 cell 에서 < 85 °C (throttle 회피).
- [ ] `cooldown_reached = True` 비율 ≥ 90%.
- [ ] `fp16_mul` 과 `fp16_add` 의 `slope_dyn` 이 ±15% 이내 (HW path 동일).

## 13. 알려진 한계

1. **Board-level NVML** : GPU core 와 HBM 이 구별되지 않음. 원한다면 별도로 NVIDIA Nsight 의 per-unit counter 필요.
2. **20 Hz sensor rate** : 초단기 power spike 는 smooth 처리됨. 평균값은 보존되므로 `k_op` 에는 영향 없음.
3. **FP8 elementwise cast overhead** : 순수 FP8 HW path 측정은 matmul 만 가능. elementwise 는 "cast + compute" 의 composite.
4. **Single-GPU focus** : multi-GPU NCCL 등은 대상이 아님.
5. **훈련 중 workload 와 차이** : microbenchmark 는 순수 kernel 을 무한 반복하므로, 실제 훈련의 scheduling gap / memory fragment 는 재현 못함.

## 14. 확장 아이디어

- **per-kernel Nsight Compute energy** : NVML 대신 HW counter 사용. sm_90 일부 지원.
- **DVFS sweep** : `--frequencies` 를 추가하여 `nvidia-smi -lgc` 로 frequency sweep.
- **공조 조건 변화** : ambient temperature 와 `k_op` 의 관계 (leakage temp coefficient).
- **Network-bound benchmark** : NCCL allreduce per-byte energy (`k_nccl`).
- **Auto retry on low R²** : 재측정 루프 (현재 수동).
- **JSON 포맷 출력** : AccelWattch config YAML 로 바로 꽂을 수 있도록.

## 15. 파일 구성

```
util/gpu_power_bench/
├── README.md                    (이 문서)
├── requirements.txt
├── run_bench.sh                 실행 런처
├── gpu_power_bench.py           메인 드라이버
├── benchmarks.py                15 cell 커널 정의
├── power_monitor.py             NVML 폴링 + 적분
├── preflight.py                 의존성/GPU 체크
├── analyze.py                   per-GPU plot
├── compare_gpus.py              cross-GPU plot
├── install_transformer_engine.sh  TE 설치 헬퍼
└── reports/                     출력 디렉토리
```

## 부록 A. 수치해석 주의사항

### A.1 적분 오차

Trapezoidal rule 의 오차는 샘플 간격의 제곱에 비례하며 함수의 2계 도함수에 비례한다. 100 Hz 샘플링에서 간격은 10 ms. power 가 급변(spike)하지 않는 한 오차는 sub-%.

### A.2 FLOP 계산

matmul `C = A·B` with `A∈ℝ^{M×K}, B∈ℝ^{K×N}` 의 FLOP 는 관례적으로 `2·M·N·K` (muladd = 2 FLOP). batched matmul 은 batch size 만큼 곱함. softmax 의 FLOP 는 exp(1) + add(1) + div(1) 로 3 FLOP/elem 으로 잡는 게 보통이지만, 본 스위트는 element 단위 비교가 목적이라 FLOP 기반이 아니라 element 기반 (`N_op = load · iters`) 으로 본다.

### A.3 Float cast energy

FP16 ↔ FP8 cast 는 PTX 의 `cvt.rn` 명령으로 구현되며 register-local 이라 추가 메모리 접근이 없다. 그러나 compute-dominant 하지 않은 elementwise 에서는 이 cast instruction 자체가 iteration 당 2 회 (입력 downcast + 출력 upcast) 실행되어 `k_op` 에 직접 반영된다.

## 부록 B. NVML power telemetry semantics

### B.1 `nvmlDeviceGetPowerUsage`

- 반환값 unit : mW (milliwatt).
- Scope : full board (GPU die + HBM + NVLink controller + VRM 포함). PCIe 포트 자체의 호스트 쪽 손실은 제외.
- 업데이트 주기 : 일반적으로 50 ms (20 Hz), driver 버전에 따라 변동.
- 정확도 : vendor spec 상 ±5% 수준. 하지만 우리는 절대값이 아니라 **차분** (total − static) 을 쓰므로 systematic offset 은 상쇄.

### B.2 `nvmlDeviceGetPowerManagementLimit`

TDP / power cap. DVFS 에 의해 실제 소비가 이 값에 수렴하면 throttle 가능. `peak_temp` 와 함께 검토.

### B.3 `nvmlDeviceGetTemperature(GPU_TEMP)`

GPU core hot-spot temperature. HBM/GDDR 온도는 별도 API. thermal monitoring 은 본 스위트가 GPU_TEMP 만 사용.

### B.4 Alternatives

- `nvidia-smi dmon -s p` : 1 Hz CLI 스트림. 간단한 더블체크용.
- `sysfs /sys/class/hwmon/...` : 일부 카드만.
- `NVIDIA DCGM` : datacenter 용, per-GPU power profile API 가 더 풍부.

---

*Maintained under `util/gpu_power_bench/` — PR contributions welcome.*
