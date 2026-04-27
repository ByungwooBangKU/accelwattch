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

**Matmul size sweep** : `N ∈ {512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288}` 를 sweep. FLOP = `2 · N³`. 메모리 footprint 는 `3 · N² · sizeof(dtype)`, N=12288 FP32 는 ~1.7 GB — 80 GB HBM 안에 충분히 들어감.

### 3.2.1 LLM-shape matmul sweep (`--llm-shapes`, opt-in)

§3.2 의 square GEMM 은 **R² / 선형성 검증용 베이스라인** 입니다. 실제 LLM 은 대부분 **skinny 또는 fat GEMM** 이라서 square 수치 만으로는 "이 모델 한 스텝에 몇 J 쓰나?" 에 답할 수 없어요. gpt-oss-120B 를 대표 모델로 삼아 다음 8 개 layer shape 를 `bm.LLM_SHAPES` 에 하드코딩하고, token 수 T (= M dim) 을 따로 sweep 합니다:

| preset | K | N | 역할 | 특징 |
|---|---|---|---|---|
| `qkv`     | 2880 | 5120   | Merged QKV projection     | K<N, 약 1.8× fat |
| `q_only`  | 2880 | 4096   | Q-only projection         | K<N, 1.4× fat |
| `kv`      | 2880 |  512   | K/V projection (GQA)      | **K≫N, 5.6× skinny** |
| `attn_o`  | 4096 | 2880   | Attention output          | square-ish |
| `router`  | 2880 |  128   | MoE gate                  | **K≫N, 22× skinny** |
| `mlp1`    | 2880 | 5760   | MoE expert up-projection  | K<N, 2× fat |
| `mlp2`    | 2880 | 2880   | MoE expert down-projection| square |
| `lm_head` | 2880 | 201088 | LM unembedding            | **K≪N, 70× fat** |

**T sweep** (= M dim, batch × seq): `[1, 256, 2048, 8192, 32768]` — decode step / small prefill / standard prefill / long context / very long.

**Dtype**: 기본 `bf16:tc` (현대 LLM 의 de-facto). `--llm-dtypes fp8:te bf16:tc` 같은 식으로 복수 지정 가능.

**셀 수**: 8 presets × 5 T = **40 cells / dtype**. `bf16` 만 돌리면 ~25 분.

**메모리 가드**: `llm_matmul_footprint_bytes(preset, T, dtype)` 로 A+B+C 크기를 계산하고, 25% HBM budget 을 초과하는 cell (예: 80 GB HBM 미만 GPU 에서 `lm_head @ T=32768`) 은 sweep 시작 전에 자동 drop 되고 로그됩니다:
```
[memcheck] dropped 2 LLM-shape cells that would exceed the 25% HBM budget:
[memcheck]    lm_head @ T=32768  bf16  ≈ 14.20 GB
[memcheck]    lm_head @ T=8192   fp32  ≈ 7.00 GB
```

**사용**:
```bash
# bf16 기본, 8 preset × 5 T
python3 gpu_power_bench.py --device 0 --llm-shapes --tag h100_llm

# 다른 모델 매핑하려면 LLM_SHAPES 를 수정하거나 subset 선택
python3 gpu_power_bench.py --llm-shapes \
    --llm-presets qkv kv mlp1 lm_head \
    --llm-ts 1 8192 32768 \
    --llm-dtypes bf16:tc fp8:te
```

**분석**: `_01_powermodel_llm_jperflop.png` (J/FLOP vs T) 와 `_01_powermodel_llm_per_call.png` (per-call mJ vs T) 두 개로 나눠 저장됩니다:
- **Panel A** — preset 별 **J/FLOP vs T** (log-log). 수평선이면 BW-bound (skinny 의 전형), 기울어지면 compute-bound 이 지배. K≫N 인 `kv` / `router` 는 T 가 작을 때 BW 병목이 커서 J/FLOP 이 평탄하게 높게 나오는 게 정상.
- **Panel B** — preset 별 **layer 한 번 call 당 에너지 (mJ)** vs T. 포인트마다 `T=..., {mJ} mJ` 라벨. 이 숫자가 **LLM inference cost model 의 per-layer 입력값** 이에요.

Square sweep 과 LLM sweep 은 CSV 안에서 `category == "matmul"` vs `"matmul_llm"` 으로 구분됩니다. `summary_by_regime` 도 LLM preset 별로 따로 행이 나오게 grouping 됩니다 (`variant = llm_{preset}_{dtype}_{mode}`).

### 3.3 Load sweep 설계 원칙

**최소 load** (`1<<17` = 128 K elem)
  : launch overhead 가 커질 정도로 작지 않으면서 모니터 window 안에 여러 iteration 이 들어갈 만큼 작음. 모든 op 가 확실히 `l2_resident` regime 에 들어가는 anchor point.

**최대 load** (`1<<30` = 1 G elem)
  : **80 GB A100 (HBM2E) / 80 GB H100** 기준 `mul` fp16 working set 6 GB (~8 %), fp8 (cast-compute-cast intermediate 포함) 기준 ~9 GB (~11 %). 기준은 **"단일 cell 이 HBM 의 25 % 를 초과하지 않을 것"** (아래 참조). 이 상한 덕에 LLM inference 의 큰 activation tensor (예: batch × seq × hidden = 32 × 2048 × 4096 = 256 M, 또는 8k seq 에서 4× 큰 것) 까지 sweep 이 직접 커버.

**11 points, 2× 로그 스케일 (중간에 `1<<24` = 16 M 추가)**
  : cache regime 을 5 bucket 으로 쪼갰을 때 (§3.4 참조) 모든 bucket 에 최소 1 point 가 들어가도록 배치. 이전 9-point 설계에서 `l2_partial` 이 1 point 뿐이었던 맹점 해소.

**메모리 안전장치 (`_MEM_SAFETY_FRACTION = 0.25`)**
  : sweep 시작 전에 `torch.cuda.get_device_properties(device).total_memory` 로 HBM 용량을 확인하고, 각 (op, dtype, N) 조합의 **worst-case 메모리 footprint** (fp8 는 fp16 intermediate 포함 3× 계수) 가 HBM × 25 % 를 넘으면 그 N 을 자동 drop 합니다. 예시 로그:
```
[info] HBM total: 80.0 GB (per-cell budget: 25% = 20.0 GB)         ← A100/H100 80GB 는 모두 통과
[info] HBM total: 40.0 GB (per-cell budget: 25% = 10.0 GB)
[memcheck] dropped N=1,073,741,824 (worst case softmax/fp8 ≈ 9.00 GB)   ← 구형 40GB 카드에서만 drop
```
80 GB GPU 에서는 11 point 모두 그대로 진행. 40 GB 이하 카드는 최상위 point 1–2 개만 자동 drop 되고 나머지는 정상 수행.

**`iters` 자동계산**
  : `target_ms / per-iter-us`. 최소 window (`--window-ms 3000`) 를 채우도록 반복 횟수 결정. load 가 커져 per-iter 시간이 길어지면 iter 수가 자동으로 줄어들어 **wall time 은 cell 당 일정**.

### 3.4 Cache locality regime — L2 hit rate 와 에너지

elementwise 벤치마크는 memory-bound 이므로 **working-set 이 L2 에 얹히는지 여부가 J/element 를 1 order 이상 갈라놓습니다**. 각 cell 은 working-set 크기와 탐지된 L2 용량을 비교해 **5 단계 L2 hit rate bucket** 으로 자동 분류됩니다 (기존 3-bucket 체계에서 세분화):

| `cache_regime` | Working-set 조건 | 예상 L2 hit rate | 해석 |
|---|---|---|---|
| `l2_hit_100` | `ws ≤ L2/4` | ≈ **100 %** | 여유있게 L2 안에 들어감. 두 번째 iter 부터 완전 resident. |
| `l2_hit_75`  | `L2/4 < ws ≤ L2/2` | ≈ **75 %** | L2 에 거의 다 들어가지만 약간의 eviction. |
| `l2_hit_50`  | `L2/2 < ws ≤ 2·L2` | ≈ **50 %** | L2 경계 — thrashing / 절반 수준 hit. |
| `l2_hit_25`  | `2·L2 < ws ≤ 4·L2` | ≈ **25 %** | L2 를 크게 초과. 대부분 miss, 일부 spatial reuse 만 살아남음. |
| `l2_hit_0`   | `ws > 4·L2` | ≈ **0 %** | 매 iter 마다 DRAM 에서 streaming. |

경계는 L2 크기를 중심으로 `L2/4, L2/2, 2·L2, 4·L2` 의 **log-symmetric** 배치 — hit rate 가 100 % → 0 % 로 매끄럽게 내려가는 trend 를 볼 수 있도록 설계했습니다. 기존 3-bucket CSV (`l2_resident` / `l2_partial` / `dram_stream`) 도 `analyze.py` 에서 자동으로 5-bucket 으로 매핑되어 backward-compatible 하게 분석됩니다.

Working-set 정의:
- `mul` / `add` : `3·N·bytes_per_elem` (a read + b read + out write)
- `gelu` / `softmax` / `layernorm` : `2·N·bytes_per_elem`
- `matmul` : `(M·K + K·N + M·N)·bytes_per_elem`. 단, matmul 은 reuse (`K` times per element) 가 있어서 대용량에서도 tile-level L2 hit 가 살아있습니다 — 라벨은 working-set 기준이므로 matmul 의 `l2_hit_0` 도 compute-bound 일 수 있음.

**기본 sweep 은 이미 5 regime 을 다 포함** (A100 40 MB L2 기준: `N≤1M` l2_hit_100, `N=2M-8M` l2_hit_75, `N=16M` l2_hit_50, `N=32M` l2_hit_25, `N≥64M` l2_hit_0). 분석 시 `cache_regime` 컬럼으로 grouping 하면 cache 효과를 분리해서 볼 수 있습니다.

**5 regime 에 각각 1 포인트만 찍고 싶다면** `--cache-sweep` 플래그를 쓰면 각 (op, dtype) 마다 정확히 **5 개** load size 만 실행 (working-set targets: `L2/8`, `L2/3`, `L2`, `3·L2`, `8·L2` — 각 bucket 의 log 중심):

```bash
python3 gpu_power_bench.py --device 0 --cache-sweep --tag h100_cache --out-dir reports/
```

출력은 평소처럼 CSV 에 쌓이고, `analyze.py` 는 `_02_cache_regime_*.png` 6 개 (elementwise 3 + matmul 3) plot 을 새로 그립니다 (§6.4.1 참조).

### 3.5 DRAM bandwidth energy — `pJ/bit` 측정

**목표**: HBM ↔ GPU 간 한 비트를 옮기는 데 드는 동적 에너지를 실측해서 선행 연구의 전형값과 비교. 즉 측정값이 "GPU 가 DRAM 트래픽으로 쓰는 pJ/bit" 인지 검증.

**측정 원리**: cache regime `l2_hit_0` (working set ≥ 4·L2) 의 elementwise cell 은 매 iter 마다 데이터를 DRAM 에서 streaming 으로 가져와요. 이 영역에서:

```
bytes_traffic   = (read+write 횟수) × N × bytes_per_elem × iters
pJ_per_bit      = dyn_energy_J × 1e12 / (bytes_traffic × 8)
achieved_BW     = bytes_traffic / wall_s
```

`analyze.py` 가 모든 elementwise row 에 대해 자동으로 `bytes_traffic`, `pj_per_bit_traffic`, `achieved_bw_gbps` 컬럼을 derive 합니다 (per-cell CSV 에 추가). l2_hit_0 cell 의 median 이 곧 DRAM 전송 에너지.

**선행 연구 참조 표** (full stack: DRAM cells + PHY + controller — 우리 측정 boundary 와 가장 가까움):

| 메모리 | 보고 pJ/bit | 출처 / 비고 |
|---|---|---|
| **HBM2** (V100) | **~7.0** | NVIDIA / industry consensus |
| **HBM2E** (A100) | **~5.0** | A100 white-paper 추정 |
| **HBM3** (H100) | **~3.9** | Hopper white-paper, "up to 50% energy reduction vs HBM2" 로부터 derive |
| DDR4 | ~7.0 | O'Connor et al., MICRO 2017 — "Fine-Grained DRAM" |
| DRAM core only | ~2.5 | Horowitz, ISSCC 2014, "Computing's Energy Problem" — 256-bit access ≈ 640 pJ, controller 제외 |

**측정 boundary 와의 차이**: 우리는 board-level NVML 을 적분하므로 **HBM PHY + L2 → HBM 라우팅 + idle controller overhead 까지 포함**. 따라서 실측치는 위 reference 보다 **1.5–3× 높게** 나오는 게 정상 (HBM3 H100 에서 6–10 pJ/bit 가 일반적). reference 와 정확히 일치하지 않더라도, **L2-resident 와 DRAM-stream 간의 비율** (보통 5–10×) 이 cache 의 에너지적 가치를 보여줍니다.

**옵션 1 — 기존 sweep 만으로**: `mul`, `add` 의 l2_hit_0 cell 이 사실상 STREAM-style 프로브와 동일. 별도 플래그 없이 `analyze.py` 만 돌리면 됩니다.

**옵션 2 — Dedicated STREAM 프로브 (`--dram-bw-test`)**: 더 깔끔한 측정을 원하면 5 종 STREAM-style 커널을 큰 working set 4 점에서 추가:

| 커널 | 의미 | 트래픽 (per call) | pJ/bit 해석 |
|---|---|---|---|
| `stream_read`  | `y = x.sum()` | 1·N·bpe | **DRAM read** 단독 |
| `stream_write` | `y.fill_(c)` | 1·N·bpe | **DRAM write** 단독 |
| `stream_copy`  | `out.copy_(x)` | 2·N·bpe | mixed 50/50 — read/write 평균 cross-check |
| `stream_scale` | `y = α·x` | 2·N·bpe | mixed 50/50 |
| `stream_triad` | `y = α·x + z` | 3·N·bpe | mixed 67/33 |

```bash
python3 gpu_power_bench.py --device 0 --dram-bw-test --tag h100_dram
python3 analyze.py --reports-dir reports/ --tag h100_dram
```

해당 커널들은 **연산이 거의 없어서** 동적 에너지가 거의 전부 메모리 트래픽. 따라서 derive 된 pJ/bit 가 mul/add 의 그것보다 noise 가 더 작음.

`stream_read` / `stream_write` 는 단방향 트래픽이라 **read 와 write 의 pJ/bit 을 따로 분리** 할 수 있고, mixed 3종은 같은 dtype 의 (R + W) / 2 와 자기들이 측정한 평균이 일치하는지 cross-check 합니다 (오차 < 5% 면 측정 quality 양호).

### 3.5.1 출력 파일 4 종

| 파일 | 내용 |
|---|---|
| `_02_dram_energy_pjbit.png` | 모든 cell 의 pJ/bit strip — l2_hit_100 → l2_hit_0 progression + HBM2/HBM3/DDR4 reference 라인 |
| `_02_dram_energy_bw.png` | l2_hit_0 sustained BW (GB/s) per kernel + HBM peak 비교 |
| `_02_dram_energy_rw_split.png` | **read vs write 분리 bar** (stream_read/stream_write 만 활성). 회색 hatched bar 는 mixed kernel 의 측정 vs 이론치 (impl) 비교 |
| `_02_dram_energy_marginal.png` | **direct (l2_hit_0)** vs **marginal (l2_hit_0 − l2_hit_100)** 두 해석. marginal 이 SM compute + L2 transit baseline 을 cancel 해서 literature DRAM-stack 정의에 더 가까움. marginal 이 음수면 P_static 문제 |

### 3.5.2 콘솔 자동 출력 예시

```
== DRAM read vs write energy (l2_hit_0, pJ/bit) ==
 dtype           op   role  r_per_call  w_per_call  n_cells  pj_per_bit_med  pj_per_bit_implied  implied_error_pct
  fp16  stream_read   READ           1           0        4           2.844                 NaN                NaN
  fp16 stream_write  WRITE           0           1        4           4.640                 NaN                NaN
  fp16  stream_copy  MIXED           1           1        4           3.621               3.742            -3.227
  fp16 stream_scale  MIXED           1           1        4           3.727               3.742            -0.402
  fp16 stream_triad  MIXED           2           1        4           3.558               3.443             3.349

== DRAM marginal cost (direct vs marginal pJ/bit) ==
          op  dtype   direct_dram_pJ_per_bit   marginal_pJ_per_bit
 stream_copy   fp16                    3.621                 3.214
 stream_read   fp16                    2.844                 2.439
stream_write   fp16                    4.640                 4.240
```

read 와 write 가 분리돼서 나오고, mixed 의 implied (= (r·R+w·W)/(r+w)) 와 측정값의 오차도 표시. marginal 컬럼은 direct 에서 ~0.4 pJ/bit 빠진 값 (l2_hit_100 baseline 이 SM/L2 비용으로 빠짐) — HBM3 literature 3.9 와 더 가깝게 정렬.

**Static vs dynamic 분리**: `dyn_energy_j = total_energy_j − static_power_w × wall_s`. static 은 §3.4 의 baseline 측정 (12 초 idle) 에서 mean ± std 가 5% 이내일 때만 신뢰 가능. baseline plot (`_03_baseline_static_power.png`) 에서 idle trace 가 평탄한지 먼저 확인 후 pJ/bit 해석. **Marginal plot 의 음수 bar 는 P_static 이 너무 높게 잡혀서 l2_hit_100 의 dyn_energy 가 인플레됐다는 신호** — 이 경우 `--rebaseline-every 20` 으로 재측정 권장.

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

### 4.4.1 Drift correction — periodic re-baseline

기본 동작 (`--rebaseline-every 0`) 은 sweep 시작 시 한 번만 P_static 을 측정하고 그 값을 모든 cell 에 적용. 30 분짜리 sweep 에서는 rack ambient + HBM controller idle 패턴 변화 때문에 **idle 자체가 1–3 W 정도 drift** 합니다. P_static 이 outdated 되면:

- 저부하 cell 의 `dyn_energy = total − p_static·wall` 가 **음수로 빠짐** → `max(0, ...)` clip 발동 → 회귀 slope `k_op` 가 위로 평탄화 (저부하 측 underestimate).
- 고부하 cell 은 dyn 이 워낙 커서 영향 미미.

**`--rebaseline-every N`** : N cell 마다 짧은 (default 4 초) idle 재측정으로 P_static 갱신. 130-cell sweep 에 wall time 약 1–2 분 추가, drift 추적 효과 큼. 권장 N=20.

```bash
python3 gpu_power_bench.py --rebaseline-every 20 --tag h100
```

매 갱신마다 다음과 같이 로그가 찍힘:
```
[rebaseline @ cell 20/130] P_static 67.42 W → 68.91 W (Δ +1.49 W, σ 0.21 W, 53.4°C)
```

CSV 의 각 row 에 그 cell 이 사용한 `static_power_w` + `baseline_age_s` (해당 P_static 측정 후 경과 시간) 가 함께 기록되어 분석 시 drift 영향을 추적 가능. sweep 끝나면 `_rebaseline.csv` sidecar 에 모든 P_static 값과 wall timestamp 가 저장됨 (drift 시계열 분석용).

### 4.4.2 Clip-bias 가시성 — `dyn_*_raw` 컬럼

CSV 의 `dyn_power_w` / `dyn_energy_j` 는 항상 `max(0, raw)` clipped 값. 그 옆에 **clip 전 raw 값** (`dyn_power_w_raw`, `dyn_energy_j_raw`) 도 함께 저장되어:

- 저부하 cell 에서 raw 가 자주 음수면 → P_static 이 너무 높게 잡혔음을 확인 (drift 또는 baseline measurement noise).
- sweep 끝에 콘솔 자동 요약:
  ```
  [clip] dyn_power_w  clipped to 0 on 7/130 cells (5.4%)
  [clip] dyn_energy_j clipped to 0 on 5/130 cells (3.8%)
  [clip] inspect dyn_power_w_raw / dyn_energy_j_raw columns to see the unclipped
         residual; large clip rate (≥ 20%) at small N means P_static drifted above
         the true idle — consider --rebaseline-every 20.
  ```
- clip rate ≥ 20% 면 무조건 `--rebaseline-every` 를 켜고 재측정 권장.

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

### 5.0 모듈 구조 (analyze.py)

긴 한 파일이지만 7 개 섹션으로 나뉘어 있어서 무엇이 어디에 있는지 빠르게 찾을 수 있도록 구성했어요. 재활용 / 중복 제거를 위한 helper 들도 한 곳에 모음.

| 섹션 | 내용 | 주요 심볼 |
|---|---|---|
| 1. Constants | regime / palette / DRAM reference / dtype bytes 모음 | `REGIME_ORDER`, `REGIME_HIT_PCT`, `LEGACY_REGIME_MAP`, `PALETTE_*`, `DRAM_REFERENCES_PJBIT`, `DTYPE_BYTES`, `RW_PER_CALL` |
| 2. Regression helpers | OLS / WLS / bootstrap CI | `linear_fit`, `linear_fit_wls`, `bootstrap_slope_ci` |
| 3. DataFrame normalisation | NaN guards + back-compat regime mapping | `add_traffic_metrics`, `_normalize_for_summary`, `_variant_name`, `_fit_one_group` |
| 4. Summary builders | per-cell / per-regime 회귀 → CSV | `summarize`, `summarize_by_regime` |
| 5. Plot helpers | 공유 save / 라벨 함수 | `_get_mpl`, `_save_fig`, `_annot_bar_pj` |
| 6. Plot functions | 그룹별 단독-패널 PNG 생성 | `plot_linearity_*`, `plot_joule_per_op_bar`, `plot_cache_regime`, `plot_dram_energy`, `plot_static_power`, `plot_temperature`, `plot_llm_matmul`, `plot_timeline` |
| 7. CLI / main | argparse + 디스커버리 + 디스패치 | `_resolve_csv`, `main` |

### 5.0.1 핵심 helper — `_fit_one_group()`

이전엔 `summarize()` 와 `summarize_by_regime()` 양쪽에 OLS + WLS + bootstrap CI + clip-bias 계산이 **각자 약 30 줄씩 복붙** 되어 있어서 한쪽만 고치면 둘이 어긋날 위험이 있었어요. 이제 한 함수로 통합:

```python
fit = _fit_one_group(g, x_col="total_flops", want_ci=True)
# fit = {n_points, slope_dyn, slope_dyn_wls, slope_dyn_ci_lo, slope_dyn_ci_hi,
#        slope_dyn_unclipped, clip_bias_pct, slope_total,
#        R2_dyn, R2_dyn_wls, R2_total}
```

새로운 회귀를 추가하려면 이 한 함수만 손보면 됩니다.

### 5.0.2 핵심 helper — `_normalize_for_summary()`

이전엔 column-default boilerplate (compute_unit / emulated 백필 + cache_regime fillna + legacy 3-bucket → 5-bucket 매핑) 가 plot 함수들과 두 summary 함수에 흩어져 있었어요. 이제:

```python
df = _normalize_for_summary(df, include_cache_regime=True)
# 보장:
#   - category / op / dtype / mode / llm_preset 컬럼 존재 + NaN 없음 (groupby 탈락 방지)
#   - cache_regime (옵션) 컬럼 존재 + 5-bucket 으로 매핑
#   - compute_unit / emulated 컬럼 존재
```

### 5.1 Per-cell linear fit

각 CSV 파일에는 `N_op`, `E_total_J`, `E_dyn_J`, `T_workload_s` 등이 load 별로 기록되어 있다. 두 가지 회귀를 동시에 수행:

```
E_dyn = k_op · N_op + c
```

(a) **OLS** — `numpy.polyfit(x, y, 1)`. `slope_dyn` / `R2_dyn` 컬럼. 모든 점에 동일한 weight → high-N 점이 절대 분산이 커서 평균을 끌어당김.

(b) **WLS** — weights = `1/y²`. `slope_dyn_wls` / `R2_dyn_wls` 컬럼. 각 점의 *상대* 오차 (log-space 에서 보이는 것) 가 동일한 영향을 갖도록 함. **이 슬로프를 headline `k_op` 로 권장**.

(c) **Bootstrap 95% percentile CI** — (x, y) 쌍을 1000회 resampling 후 매번 WLS slope 재계산, 2.5% / 97.5% percentile 을 `slope_dyn_ci_lo` / `slope_dyn_ci_hi` 로 보고. CI 폭이 슬로프의 ±5% 이내면 상당히 안정적.

(d) **Clip-bias 추정** — `dyn_energy_j_raw` (PR A 에서 추가) 가 있으면 unclipped raw 로도 WLS 다시 돌려 `slope_dyn_unclipped` 와 `clip_bias_pct = (slope_clipped − slope_raw) / slope_raw × 100` 을 보고. ±2% 이내가 정상; 그 이상이면 P_static drift 의심 (§4.4.1 → `--rebaseline-every 20`).

- `k_op` 단위: elementwise 는 J/element, matmul 은 J/FLOP.
- J/FLOP 을 얻으려면 matmul 의 N_op 는 `2·M·N·K` 로 계산 (multiply-accumulate = 2 FLOP 관례).

`<stem>_01_powermodel_coef_bar_elementwise.png` 와 `<stem>_01_powermodel_coef_bar_matmul.png` 의 bar 위에는 WLS slope + R² 가 라벨링되고, **bootstrap CI 가 error bar 로** 직접 표시됩니다 — bar 위에 작은 whisker 가 보이면 그것이 95% CI.

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

### 6.4.1 `cache_regime_*.png` — L2 hit rate 별 J/element 와 regime-specific `k_op`

> **참고**: 이 plot 군은 **default sweep 의 5 개 elementwise op (mul, add, softmax, gelu, layernorm) 모두** 를 한꺼번에 표시합니다. `_resolve_keys()` 가 데이터에 존재하는 모든 op 를 자동으로 가져와 그리므로 (하드코딩된 "3개 제한" 등은 없음), CSV 가 5 op 를 담고 있다면 plot 도 5 op × 5 regime = 25 bar 를 모두 보여줍니다.

**무엇을 보여주나** : (split 후) **6 개 단독 PNG** — elementwise 3 + matmul 3.

- **Panel A (Raw spread)** : 개별 cell 의 `j_per_element_dyn` 을 regime x축에 strip plot. 한 regime 안의 세로 분포가 그 regime 의 내부 variance.
- **Panel B (Power-model coefficient)** : `summarize_by_regime()` 이 각 `(op, dtype, cache_regime)` 에 대해 독립적으로 회귀한 `slope_dyn` (= `k_op`) 을 regime 별 grouped bar 로. **각 bar 에 pJ/elem 숫자와 R² 값이 직접 라벨링**되어 있어서 "L2 에서 mul 은 0.31 pJ/elem, DRAM 에서 5.53 pJ/elem" 식으로 바로 읽힘.
- **Panel C (Steady-state dyn power)** : regime 별 평균 `dyn_power_w` 를 W 단위 bar 로. regime 이 뜨거워질수록 — 보통 L2 60W → partial 120W → DRAM 200W 근처 — **energy 뿐 아니라 순간 전력도 증가**.

**어떻게 읽나** :
- **Panel B 가 power model 이 소비할 숫자** — 같은 op 의 `k_op` 가 regime 에 따라 1 order 차이나면 모델이 cache locality 를 인자로 받아야 한다는 뜻.
- Panel A 의 수직 gap 이 곧 cache miss 의 에너지 비용. mul/add 에서 5–10× 가 일반적.
- 점이 "계단" 이 아니라 "비스듬한 선" 이면 sweep point 가 L2 경계에 가까움 — transition 지점 근처에서 sub-regime 변동 신호.
- reduction op (softmax/layernorm) 는 compute overhead 가 BW 만큼 있어서 regime 간격이 elementwise mul/add 보다 좁게 나옴.
- **fp8** 는 cast-compute-cast 때문에 regime 에 관계없이 fp16 대비 높게 나옵니다 — `--include-emulated` 로 비교.

**숫자로 확인하고 싶다면** `<stem>_summary_by_regime.csv` 를 여세요. 이 CSV 가 `(op, dtype, mode, cache_regime)` 당 한 행, `slope_dyn` 컬럼이 그 regime 의 `k_op` 입니다. 콘솔에도 analyze.py 실행 시 자동 출력됩니다:

```
k_op per cache regime (J/element for elementwise, J/FLOP for matmul):
 variant     compute_unit  cache_regime  n_points   slope_dyn     R2_dyn   median_j_per_unit   mean_dyn_power_w
 fp16_mul    CUDA core     l2_resident   4          3.02e-13      0.999    3.05e-13            60.0
 fp16_mul    CUDA core     l2_partial    2          1.12e-12      1.000    9.63e-13            120.0
 fp16_mul    CUDA core     dram_stream   3          4.08e-12      0.985    5.03e-12            200.0
 ...
```

### 6.4.2 `dram_energy.png` — DRAM 트래픽의 pJ/bit + sustained BW

**무엇을 보여주나** : 가로 2 panel (§3.5 자세한 배경).

- **Panel A** : 각 cell 의 `pj_per_bit_traffic` 을 strip plot 으로 cache_regime x축에 표시. l2_hit_0 cluster 가 곧 DRAM 비용. HBM2/HBM2E/HBM3/DDR4/Horowitz '14 의 reference 값이 panel 우측에 dashed 가로선 + 라벨로 그려져 즉시 비교 가능.
- **Panel B** : l2_hit_0 cell 의 op×dtype 별 median sustained BW (GB/s). HBM peak 가 알려져 있으면 빨간 dashed line 으로 표시 — 50% 이상이면 BW-bound 이 정상, 30% 이하면 launch overhead / 비효율 의심.

**어떻게 읽나** :
- l2_hit_0 strip 의 median 이 HBM3 reference (3.9 pJ/bit) 의 1.5–3 배 안이면 측정 정상 (board-level boundary 보정 후). 5 배 이상이면 baseline static power 가 잘못 잡혔거나 thermal throttle 가능성.
- l2_hit_100 cluster 가 l2_hit_0 보다 **1 order 정도 낮으면** (예: 0.5 vs 5 pJ/bit), L2 가 잘 동작하고 있다는 강한 증거.
- Panel B 에서 fp8 / fp16 / fp32 가 같은 op 에 대해 비슷한 sustained BW 면 BW-bound; fp16 보다 fp32 가 절반이면 BW 가 fp16 에서 saturate 안 함 → launch overhead 가 아직 존재.

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

#### 8.3.2 Troubleshooting: `Transformer Engine requires CUDA 12.0 or newer` (에서 torch 는 12.8 인데?)

설치 중 pip 가 **소스 tarball** (`transformer_engine_torch-2.x.tar.gz`) 을 받아서 `setup.py` 를 돌릴 때 다음 에러가 나오는 경우:

```
File "build_tools/pytorch.py", line 68, in setup_pytorch_extension
    raise RuntimeError("Transformer Engine requires CUDA 12.0 or newer")
```

`torch.version.cuda = 12.8` 로 보여도 이 에러가 뜨는 이유:

- **`torch.version.cuda`** 는 torch wheel 이 **번들한** CUDA runtime 버전. pip 로 설치된 torch 안에만 존재.
- TE 의 `setup.py` 는 torch 의 bundle 이 아니라 **시스템의 `nvcc` 가 가리키는 toolkit 을 보고** 컴파일한다. 따라서 시스템 `/usr/local/cuda` 가 구형 (예: CUDA 11.x) 을 가리키면 TE 는 그 버전으로 판단해 에러를 낸다.

진단:

```bash
which nvcc
nvcc --version | grep release          # 12.0 이상이어야 함
ls -la /usr/local/cuda                 # 실제 어디로 링크?
echo $CUDA_HOME
```

해결 — 두 가지 경로:

**(a) [권장, 간단] NVIDIA 의 prebuilt wheel 사용** — nvcc 필요 없음:

```bash
pip install --no-build-isolation \
    --extra-index-url https://pypi.nvidia.com \
    'transformer-engine[pytorch]'
```

`install_transformer_engine.sh` 는 이제 이 경로를 **먼저 자동으로 시도**합니다. torch / Python / CUDA 조합이 prebuilt 에 맞으면 nvcc 없이 즉시 설치 완료.

**(b) [복잡] system CUDA toolkit 업그레이드** — prebuilt 가 안 맞을 때:

```bash
# conda 환경이면
conda install -c nvidia cuda-toolkit=12.1

# 또는 시스템 패키지 (sudo 권한 필요)
sudo apt install cuda-toolkit-12-1

# 설치 후 환경변수 확인
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

그 뒤 `./install_transformer_engine.sh` 재실행.

**preflight 에서 사전 감지**: `preflight.py` 는 이제 `nvcc` 버전을 파싱해서 `torch.version.cuda` 와 비교합니다. 공용 서버에서 잘 발생하는 "pip torch 는 신 CUDA, 시스템 nvcc 는 구 CUDA" 함정이 사전에 경고로 표시됩니다:

```
[warnings]
  - system CUDA toolkit is 11.6 — Transformer Engine requires ≥ 12.0 to build from source
    (your torch reports 12.8 because the wheel bundles its own runtime — TE ignores that and uses nvcc).
    Either install a newer toolkit (conda install -c nvidia cuda-toolkit=12.1) or use the prebuilt wheel path —
    ./install_transformer_engine.sh now tries pypi.nvidia.com first
```

#### 8.3.3 Troubleshooting: `CUDA error: an illegal memory access` in `torch.cat(amax_buffer)` (Blackwell + fp8_te)

`matmul_fp8_te` 또는 `--llm-shapes` 가 **Blackwell (sm_120)** GPU 에서 다음과 같이 죽는 경우:

```
File ".../transformer_engine/pytorch/fp8.py", line 340, in reduce_and_update_fp8_tensors
    contiguous_amax = torch.cat(amax_buffer)
RuntimeError: CUDA error: an illegal memory access was encountered
```

**원인**: TE 의 amax history buffer 관리 코드가 **Blackwell 의 새로운 FP8 tile 경로 + 작은 M dim** (예: T=1 decode-size matmul) 조합에서 amax 텐서를 잘못 다룸. TE 가 Hopper sm_90 에 맞춰 튜닝되어 있어서 sm_120 의 새 패스가 amax 와 race 를 일으킴. NVIDIA TE 1.x 에서 알려진 이슈.

**증상의 무서운 점**: 한 번 illegal memory access 가 나면 그 프로세스의 **CUDA context 가 완전히 망가져서** 이후 모든 CUDA 호출이 같은 에러로 죽음. 옛 코드에선 sweep 전체가 crash 하면서 이미 측정한 cell 들도 CSV 로 못 남아 소실.

**현재 코드 (PR #31 이후)**:

1. **build() 단계의 5× warmup** : `_make_matmul_fp8_te` / `_make_llm_matmul_fp8_te` 가 단일 forward 가 아니라 5 회 + sync 로 검증. amax buffer state 가 안정될 때까지. 이 경계에서 fail 하면 cell 만 skip 되고 sweep 은 살아남음.

2. **fatal-CUDA-error 자동 감지 + 부분 CSV 저장** : 측정 중 `illegal memory access` / `CUDA error` / `device-side assert` 등이 보이면 `gpu_power_bench.py` 가 더 이상 CUDA 호출 시도하지 않고 그 시점까지 모은 row 를 즉시 CSV 로 dump 한 뒤 cleanly 종료:
   ```
   !! FATAL CUDA error at matmul_fp8_te K_size=1024: CUDA error: an illegal memory access...
   !! CUDA context is now unrecoverable — flushing 130 completed cells to CSV and exiting early.
   [save] reports/gpu_power_bench_*_blackwell.csv  (130 rows — partial; sweep aborted by '...')
   [recover] all completed cells are saved. To finish the sweep, re-run dropping the variant
             that crashed (e.g. --matmul-variants without fp8:te, or skip --llm-shapes).
   ```

3. **Variant-level skip** : 한 cell 이 (FATAL 외) 에러로 죽으면 같은 (op, dtype, mode, llm_preset) 의 나머지 cell 도 retry 안 하고 skip — CUDA context 가 살아있어도 동일 fault 가 deterministic 하게 반복되니 시간 낭비 방지.

**해결 / 우회**:

- **fp8_te 빼고 sweep 다시** :
  ```bash
  ./run_bench.sh --device 3 --suite full --tag blackwell \
      --matmul-variants fp32:simt tf32:tc fp16:tc bf16:tc   # fp8:te 제거
  ```
  또는 LLM sweep 안 켜기:
  ```bash
  ./run_bench.sh --device 3 --suite powermodel --tag blackwell   # llm 안 들어감
  ```

- **Square matmul 만은 보통 OK** : Hopper 처럼 T 가 충분히 크면 (M ≥ 128) Blackwell 에서도 `matmul_fp8_te` 가 통과합니다. `--llm-shapes` 의 `T=1` (decode), `T=256` (small prefill) 정도가 자주 fail. fp8_te + LLM 만 빼고 fp16/bf16 LLM 으로:
  ```bash
  ./run_bench.sh --suite llm --llm-dtypes bf16:tc fp16:tc   # fp8:te 제외
  ```

- **TE 업그레이드** : 최신 TE (≥ 2.x) 에서는 sm_120 amax 처리가 일부 개선됨. 테스트 환경 허용되면 `pip install --upgrade transformer-engine[pytorch]`.

- **Recovery 후 분석**: 부분 CSV 도 `analyze.py` 가 정상 처리합니다. fp8_te / llm 행이 없을 뿐 elementwise + 다른 matmul variants 의 k_op / cache_regime / DRAM-energy 분석은 모두 동작.

### 8.4 환경 권장 사항

- `sudo nvidia-smi -pm 1` : persistence mode.
- 가능하면 `sudo nvidia-smi -lgc <freq>` : 주파수 고정 — DVFS 변동 제거.
- 백그라운드 프로세스 최소화 (Xorg, 다른 CUDA 작업 금지).
- 공기 흐름이 좋은 서버 권장 (cooldown 시간 단축).

## 9. 실행

### 9.0 Test suites — `--suite NAME` (권장 진입점)

여러 플래그를 매번 외울 필요 없이 **named test-suite 프리셋** 을 사용합니다. 각 suite 는 적절한 플래그 조합으로 자동 펼쳐지고, 사용자가 추가로 명시한 플래그는 suite 의 default 를 override 합니다.

| Suite | 무엇을 도는가 | 펼쳐진 플래그 | 시간 |
|---|---|---|---|
| `smoke` | 5분 산뜻한 sanity check | `--quick --no-matmul` | ~5 분 |
| `powermodel` | 기존 baseline benchmark — elementwise + matmul, 추가 probe 없음 | (default) | ~30 분 |
| `cache` | 5-bucket cache regime 에 정확히 1 cell 씩 + matmul | `--cache-sweep` | ~15 분 |
| `dram` | STREAM-style 만 (pJ/bit 측정 전용) | `--dram-bw-test --no-matmul --no-elementwise` | ~10 분 |
| `llm` | gpt-oss-120B layer shape 만 | `--llm-shapes --no-elementwise --no-matmul` | ~25 분 |
| `full` | 전체 + drift correction (publication-quality) | `--llm-shapes --dram-bw-test --rebaseline-every 20` | ~75 분 |

```bash
# 권장: 기본 5분 smoke 후 본 측정 (full)
./run_bench.sh --suite smoke --tag h100_smoke
./run_bench.sh --suite full  --tag h100

# Suite + 사용자 override (--no-matmul 으로 matmul 빼기)
./run_bench.sh --suite full --tag h100 --no-matmul

# 특정 카테고리만 빠르게
./run_bench.sh --suite cache --tag h100_cache
./run_bench.sh --suite dram  --tag h100_dram
```

### 9.1 기본 실행 (auto-pipeline)

`run_bench.sh` 는 sweep 끝나면 **자동으로 `analyze.py` 를 호출**해서 plot 까지 생성합니다 (single-GPU). multi-GPU 모드는 자동으로 `multi_gpu_analysis.py` 를 호출.

```bash
./run_bench.sh                      # GPU 0 sweep + analyze 자동
./run_bench.sh --no-auto-analyze    # CSV 만 만들고 종료
```

### 9.2 다중 GPU / 태깅

```bash
./run_bench.sh --device 0 --tag a100        # 단일 GPU + auto-analyze
./run_bench.sh --num-gpus 8  --tag h100     # 8 장 병렬 + multi_gpu_analysis 자동
./run_bench.sh --devices "0,2,4" --tag h100 --suite full
```

### 9.2.1 `--device N` vs `CUDA_VISIBLE_DEVICES=N`

두 방식은 **이상적으로** 같은 GPU 를 측정하지만 NVML 의 동작 차이 때문에 **결과가 달라질 수 있는 sneaky 한 함정**이 있습니다. 권장은 항상 `--device`.

| 방식 | torch | NVML (power 측정) | 결과 |
|---|---|---|---|
| `./run_bench.sh --device 3` | 물리 GPU 3 | 물리 GPU 3 | ✅ 일관 |
| `CUDA_VISIBLE_DEVICES=3 ./run_bench.sh` | 물리 GPU 3 (logical 0) | (옛 코드) **물리 GPU 0** ✗ |
| `CUDA_VISIBLE_DEVICES=3 ./run_bench.sh` | 물리 GPU 3 (logical 0) | (현재 코드) 물리 GPU 3 (PCI bus id resolved) ✅ |

**이유**: `CUDA_VISIBLE_DEVICES` 는 CUDA-side 필터라 그 프로세스 안에서 보이는 GPU 만 logical index 0..N-1 로 다시 번호 매김. 그러나 **NVML 은 이 환경변수를 무시**하고 항상 물리 인덱스로 동작 → `nvmlDeviceGetHandleByIndex(0)` 가 보이는 logical 0 (= 물리 3) 이 아니라 **진짜 물리 GPU 0** 을 가리킴 → 워크로드는 GPU 3 에서 도는데 power 는 GPU 0 의 idle 값이 measured → dyn_energy 가 silently 잘못된 값으로 계산되는 무서운 시나리오.

**현재 코드 (PR #29 이후)**: `torch.cuda.get_device_properties(args.device).pci_bus_id` 로 GPU 의 PCI bus 주소를 받아 `pynvml.nvmlDeviceGetHandleByPciBusId()` 로 NVML handle 을 잡습니다. PCI bus id 는 양쪽 다 같은 식별자라서 `CUDA_VISIBLE_DEVICES` 가 어떻게 매핑하든 항상 같은 카드를 가리킴. 시작 시 다음과 같은 로그가 찍혀 어느 카드를 측정 중인지 확인 가능:

```
[info] GPU=NVIDIA H100 80GB HBM3  cc=9.0  slug=h100_sxm
[info] NVML handle resolved by PCI bus id 0000:81:00.0  (CUDA_VISIBLE_DEVICES=3)
```

**그래도 권장**: `--device 3 --tag blackwell_gpu0` 가 의도가 더 명확하고, multi-GPU launcher 와 결합하기도 쉬워요. `CUDA_VISIBLE_DEVICES` 는 cluster scheduler 가 자동으로 끼워주는 환경에서 자연스럽게 작동하도록 보호 차원으로 깔아둔 것.

### 9.3 Quick 모드 (별칭: `--suite smoke`)

```bash
./run_bench.sh --suite smoke
# 또는 동등하게:
./run_bench.sh --quick --no-matmul
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
# → reports/gpu_power_bench_a100_80gb_<stamp>_a100_rebaseline.csv  # § 4.4.1
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
| `<stem>_summary.csv` | cell 당 1 행 — `slope_dyn` (OLS) + **`slope_dyn_wls`** (권장 headline) + **`slope_dyn_ci_lo` / `slope_dyn_ci_hi`** (95% bootstrap CI) + `R2_dyn_wls` + `slope_dyn_unclipped` / `clip_bias_pct` (raw-vs-clipped 영향) |
| `<stem>_summary_by_regime.csv` | `(op, dtype, mode, cache_regime)` 당 1 행 — regime 별 `slope_dyn` (= `k_op`), `R2_dyn`, `median_j_per_unit`, `mean_dyn_power_w` |
| `<stem>_01_powermodel_linearity_elementwise.png` | elementwise 10 종 log-log 선형성 + wall time + J/elem |
| `<stem>_01_powermodel_linearity_matmul.png`      | matmul 5 variant log-log — `[CUDA]` · `[TC]` 태그 + 각 point 의 swept K 와 J/FLOP 값 annotate |
| `<stem>_01_powermodel_coef_bar_elementwise.png` | elementwise k_op bar (pJ/elem + R² + bootstrap CI whisker) — full-width 단독 패널 |
| `<stem>_01_powermodel_coef_bar_matmul.png`      | matmul k_op bar (pJ/FLOP + R² + bootstrap CI whisker) — full-width 단독 패널 |
| `<stem>_01_powermodel_llm_jperflop.png`          | LLM-shape: J/FLOP vs token count T (preset 별 line + T 라벨) |
| `<stem>_01_powermodel_llm_per_call.png`          | LLM-shape: per-call mJ vs T |
| `<stem>_02_cache_regime_elementwise_strip.png`   | elementwise: per-cell J/elem strip per regime (5 bucket) |
| `<stem>_02_cache_regime_elementwise_kop.png`     | elementwise: regime 별 k_op bar (pJ/elem + R²) |
| `<stem>_02_cache_regime_elementwise_dynpower.png` | elementwise: regime 별 평균 dyn power (W) |
| `<stem>_02_cache_regime_matmul_strip.png`        | matmul: 동일 strip plot |
| `<stem>_02_cache_regime_matmul_kop.png`          | matmul: regime 별 k_op (pJ/FLOP) |
| `<stem>_02_cache_regime_matmul_dynpower.png`     | matmul: regime 별 dyn power |
| `<stem>_02_dram_energy_pjbit.png`                | pJ/bit strip — HBM2/HBM3 reference 라인 포함 |
| `<stem>_02_dram_energy_bw.png`                   | l2_hit_0 sustained BW per kernel (HBM peak 비교) |
| `<stem>_02_dram_energy_rw_split.png`             | (--dram-bw-test 시) read vs write 분리 bar + mixed cross-check |
| `<stem>_02_dram_energy_marginal.png`             | direct vs marginal pJ/bit (l2_hit_0 − l2_hit_100) — DRAM-stack 만 |
| `<stem>_dram_rw_split.csv`                       | (--dram-bw-test 시) per (dtype, op) read/write/mixed pJ/bit 표 |
| `<stem>_dram_marginal.csv`                       | per (op, dtype) direct vs marginal pJ/bit 표 |
| `<stem>_03_baseline_static_power.png`            | 3 패널 P_static 진단 (idle trace + 구성비 + 점유율) |
| `<stem>_04_thermal_diagnostics.png`              | 3 패널 thermal 진단 (start/avg/peak + cooldown + J/op vs T) |
| `<stem>_05_trace_timeline.png`                   | 전체 run 의 power/temp/clock 타임라인 (samples CSV 존재 시) |

파일명은 **그룹 번호 prefix (`01_powermodel`, `02_cache`, `03_baseline`, `04_thermal`, `05_trace`)** 를 붙여서 `ls` / file manager 정렬 시 **읽는 순서대로** 나열됩니다: 모델 검증 → cache 분석 → baseline 진단 → thermal → raw trace.

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

즉 A100 에서도 `_01_powermodel_linearity_matmul.png`, `_01_powermodel_coef_bar_matmul.png` 에 `matmul_fp8_te` bar 가 그려지고, H100 의 native FP8 수치와 시각적으로 직접 비교할 수 있습니다. cross-GPU 플롯 (`compare_gpus.py`) 에서도 `matmul_fp8_te` 가 두 GPU 모두 bar 로 나타나며, A100 쪽은 hatch + `*EMU` 주석으로 폴백임을 명시합니다.

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

A100 만 있거나 H100 만 있을 때는 Step 3 을 건너뛰고 Step 1–2 로 종료합니다. 결과 해석에는 `_summary.csv` 의 `slope_dyn` 컬럼과 `_01_powermodel_coef_bar_*.png` 두 장이 핵심입니다.

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
│   ├── <stem>_summary_by_regime.csv
│   ├── <stem>_01_powermodel_linearity_elementwise.png
│   ├── <stem>_01_powermodel_linearity_matmul.png
│   ├── <stem>_01_powermodel_coef_bar_elementwise.png      # split per panel
│   ├── <stem>_01_powermodel_coef_bar_matmul.png
│   ├── <stem>_01_powermodel_llm_jperflop.png              # if --suite full / llm
│   ├── <stem>_01_powermodel_llm_per_call.png
│   ├── <stem>_02_cache_regime_elementwise_strip.png       # 6 cache panels
│   ├── <stem>_02_cache_regime_elementwise_kop.png
│   ├── <stem>_02_cache_regime_elementwise_dynpower.png
│   ├── <stem>_02_cache_regime_matmul_strip.png
│   ├── <stem>_02_cache_regime_matmul_kop.png
│   ├── <stem>_02_cache_regime_matmul_dynpower.png
│   ├── <stem>_02_dram_energy_pjbit.png                    # split DRAM
│   ├── <stem>_02_dram_energy_bw.png
│   ├── <stem>_02_dram_energy_rw_split.png                 # if --dram-bw-test
│   ├── <stem>_02_dram_energy_marginal.png                 # if both regimes present
│   ├── <stem>_03_baseline_static_power.png
│   ├── <stem>_04_thermal_diagnostics.png
│   └── <stem>_05_trace_timeline.png
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
| `static_power_w` | W | 이 cell 에 사용된 P_static 값 (`--rebaseline-every` 사용 시 cell 마다 갱신될 수 있음) |
| `baseline_age_s` | s | 사용된 P_static 측정 시점부터 경과 시간 — 0 = 방금 갱신, 큰 값 = drift 가능성 ↑ |
| `dyn_power_w_raw` | W | clip 전 `avg_power_w − p_static`. 음수면 noise / drift 가 P_static 위로 올라옴 — clip 이 발동 |
| `dyn_energy_j_raw` | J | clip 전 `total_energy_j − p_static·wall_s` |
| `bytes_traffic` | B | (analyze.py 가 derive) elementwise `(reads+writes) × N × bytes/elem × iters` — l2_hit_0 에서 DRAM 트래픽, l2_hit_100 에서 L2 트래픽 (§3.5) |
| `pj_per_bit_traffic` | pJ/bit | (analyze.py derive) `dyn_energy_J × 1e12 / (bytes_traffic × 8)` — l2_hit_0 cell 에서 DRAM 비용 (HBM3 ≈ 3.9 pJ/bit reference) |
| `achieved_bw_gbps` | GB/s | (analyze.py derive) `bytes_traffic / wall_s` — sustained BW; HBM peak 대비 50%+ 면 BW-bound |
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

### 11.1 Multi-GPU variance analysis (같은 모델 카드 여러 장)

같은 노드에 동일 GPU 가 여러 장 있을 때 (예: 8× H100 SXM5), 카드 간 variance — **cooling asymmetry**, **silicon binning**, **stuck clock**, **bad TIM** 등 — 을 측정하려면 병렬 sweep + 전용 분석기를 씁니다.

**Step 1 — 병렬 sweep** (`run_bench.sh`):

```bash
# 8 장 GPU 에 동시에 full sweep
./run_bench.sh --num-gpus 8 --tag h100

# 일부 GPU 만 (0, 2, 4, 6)
./run_bench.sh --devices "0,2,4,6" --tag h100

# 열 부담 최소화 — 한 번에 한 장씩 (느림, 8× 시간)
./run_bench.sh --num-gpus 8 --sequential --tag h100

# 기존 옵션 그대로 forward
./run_bench.sh --num-gpus 4 --llm-shapes --tag h100_llm
```

각 GPU 는 고유 tag 접미사 `_gpu<N>` 을 받고 (`h100_gpu0`, `h100_gpu1`, …), 로그는 `reports/logs/<tag>.log` 로 분리됩니다. 병렬 모드는 node 의 쿨링 예산을 공유하니 **cross-GPU variance 에는 cooling asymmetry 가 섞여 들어갑니다** — 순수 silicon 차이만 보고 싶다면 `--sequential`.

아무 플래그도 안 주면 기존대로 `--device 0` 하나만 실행.

**Step 2 — 분석** (`multi_gpu_analysis.py`):

```bash
python3 multi_gpu_analysis.py reports/ 8 --tag h100
```

출력 (`reports/multi_gpu_h100/`):

| 파일 | 내용 |
|---|---|
| `multi_gpu_<tag>_<stamp>_per_gpu_summary.csv` | per-GPU × variant 당 한 행 (analyze.summarize 결과 concat + `gpu_index` 컬럼) |
| `multi_gpu_<tag>_<stamp>_variance.csv` | variant 당 한 행 — `mean`, `std`, `cv_percent`, `min`, `max`, `outlier_gpus_2sigma` |
| `multi_gpu_<tag>_<stamp>_per_gpu_scalars.csv` | GPU 당 한 행 — `static_power_w`, `mean_dyn_power_w`, `mean_temp_c`, `peak_temp_c` |
| `multi_gpu_<tag>_<stamp>_01_coefficient_variance.png` | k_op bar (mean ± σ, CV% 라벨, CV ≥ 10 % 면 ⚠) |
| `multi_gpu_<tag>_<stamp>_02_deviation_heatmap.png` | (variant × GPU) 행렬, cell 값 = 해당 GPU 가 cross-GPU 평균 대비 몇 % 벗어났는지 (빨강=비쌈 / 파랑=쌈) |
| `multi_gpu_<tag>_<stamp>_03_per_gpu_health.png` | GPU 당 3 개 bar (idle power, mean dyn power, mean/peak temp) — per-card health card |

**어떻게 읽나**:
- `coefficient_variance.png` 에서 어떤 variant 든 **CV ≥ 10 %** 가 찍히면 (⚠ 표시), 그 variant 의 regression 이 한두 GPU 때문에 오염된 것. `variance.csv` 의 `outlier_gpus_2sigma` 에서 정확히 어떤 card 가 2σ 밖인지 확인.
- `deviation_heatmap.png` 에서 **특정 row 전체가 빨강/파랑** 이면 그 variant 의 kernel 이 환경 민감한 것 (드물음); **특정 column 전체가 빨강** 이면 그 card 가 전 workload 에서 더 비싼 — **cooling / silicon 문제 GPU** 로 retirement 후보.
- `per_gpu_health.png` 에서 한 card 의 idle 이 mean 보다 3-5 W 이상 높으면 stuck clock / bad binning 의심; peak temp 가 다른 card 대비 10 °C 이상 높으면 TIM / 공기 흐름 이슈.

**실 예시 (합성 데이터, gpu2 를 outlier 로 설정한 시나리오)**:

```
== cross-GPU variance on k_op (slope_dyn) ==
 variant    category  n_gpus      mean       std  cv_percent       min       max  outlier_gpus_2sigma
fp16_add elementwise       4 9.778e-13 1.555e-13    15.90       9.001e-13 1.211e-12
fp16_mul elementwise       4 9.778e-13 1.555e-13    15.90       9.001e-13 1.211e-12

⚠  2 variant(s) with CV ≥ 10% — likely a per-GPU outlier on those variants

== per-GPU health card ==
 gpu_index       gpu_name  static_power_w  mean_dyn_power_w  mean_temp_c  peak_temp_c
         0 H100 SXM5 80GB           65.10            102.20        51.00        56.00
         1 H100 SXM5 80GB           65.70            102.40        51.00        56.00
         2 H100 SXM5 80GB           73.00            115.00        68.00        73.00   ← outlier
         3 H100 SXM5 80GB           65.30            102.20        50.00        55.00
```

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
4. **Intra-GPU single-process focus** : 같은 노드 여러 GPU 의 **variance / outlier 분석**은 §11.1 `multi_gpu_analysis.py` 로 가능하지만, **GPU 간 NCCL / P2P 통신 에너지**는 여전히 대상이 아님 (각 카드는 독립 sweep 으로 측정됨).
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
