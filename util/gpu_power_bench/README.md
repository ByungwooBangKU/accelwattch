# GPU Power-per-Operation Benchmark (A100 / H100)

> "같은 연산을 **어느 precision · 어느 compute unit** 에서 돌릴 때 몇 Joule 을
> 쓰는가" 를 A100 과 H100 에서 동일 코드로 재서, AccelWattch 류 GPU power
> model 의 per-op 계수 (`k_op`) 를 실측으로 뽑기 위한 microbenchmark.

## 측정 대상 — 총 **15 가지 benchmark**

### A. Elementwise / reduction (10개, CUDA core 경로)

FP16 / FP8 × {MUL, ADD, Softmax, GeLU, LayerNorm}. **모두 SIMT CUDA core**
에서 실행. Tensor Core 는 건드리지 않음. Load 축은 텐서 element 수 N.

### B. Matmul (5개, Tensor Core vs CUDA core + TE FP8)

M = N = K 의 정사각 GEMM. Load 축은 행렬 변 길이 K (FLOPs = 2·K³).

| variant | compute unit | A100 (sm_80) | H100 (sm_90) | 의미 |
|---|---|---|---|---|
| `matmul_fp32_simt` | **CUDA cores** (TF32 off) | ✓ | ✓ | Tensor-Core-off baseline |
| `matmul_tf32_tc`   | Tensor Core TF32 | ✓ | ✓ | A100 에서 도입된 TC precision |
| `matmul_fp16_tc`   | Tensor Core FP16 | ✓ | ✓ | 양쪽 공통 reference |
| `matmul_bf16_tc`   | Tensor Core BF16 | ✓ | ✓ | FP16 와 동속, 다른 dynamic range |
| `matmul_fp8_te`    | Tensor Core FP8 (TE) | ✗ → FP16 fallback | ✓ **native** | H100 전용 이득 측정 |

즉 (A) 는 precision 비교축, (B) 는 **"Tensor Core vs CUDA core" + "H100 native FP8 vs A100 FP16"** 비교축. 두 축이 직교해서 power model 의 서로 다른 항을 분리 측정해줍니다.

## Power model context (왜 이 data 가 필요한가)

GPU 1-차 에너지 모델:

```
E(workload) = P_static · T(workload)  +  Σ_op  k_op · N_op
```

- **`P_static`** : idle GPU power. program 시작 시 `measure_static_power()` 가 평균을 산출 → `static_power_w` 컬럼.
- **`k_op`** : "해당 op 한 단위에 드는 dynamic Joule" 계수. 각 op 를 **여러 load 에 sweep** 해서 `E_dyn ~ N_op` 을 선형회귀 → **slope 가 곧 `k_op`**. `analyze.py` 가 계산.
- **`N_op`** : 모델링 대상 워크로드가 수행하는 op 수 (PTX trace / nvprof counters 등에서 추출).
- **`T(workload)`** : 워크로드 총 실행 시간 → static 에너지는 그 동안 소모.

linearity (E_dyn ∝ N) 가 성립해야 (R² ≥ 0.99) 이 모델 폼이 유효. 깨지면 launch overhead dominant (load 너무 작음) 또는 BW saturation (너무 큼). sweep 범위를 조정하거나 linear regime 만 추출.

## 무엇이 저장되나

### per-cell CSV (`gpu_power_bench_<gpu>_<stamp>.csv`)

한 행 = 하나의 (variant × load). 주요 컬럼:

| 컬럼 | 의미 |
|---|---|
| `category` | `elementwise` / `matmul` |
| `op` | `mul/add/softmax/gelu/layernorm/matmul` |
| `dtype` | `fp16/fp8/fp32/tf32/bf16` |
| `mode` | `elementwise/simt/tc/te` |
| `variant` | 통합 이름 (e.g. `matmul_fp8_te`) |
| `load_name` / `load_value` | `n_elements` 또는 `K_size` |
| `iters` | 해당 cell 의 반복 횟수 (측정창 길이 ≈ `--window-ms`) |
| `total_elements` / `total_flops` | sweep 축 (linearity 회귀 입력) |
| `static_power_w` | 이번 런 전체에서 고정된 P_static |
| `avg_power_w` / `dyn_power_w` | 구간 평균 전력 / (그 - P_static) |
| `total_energy_j` | NVML power 적분 (J) |
| `static_energy_j` | `P_static × wall_s` |
| `dyn_energy_j` | `total - static` — **dynamic 에너지** |
| `j_per_element_total` / `_dyn` | J/elem (total 또는 dyn) |
| `j_per_flop_total` / `_dyn` | J/FLOP (FLOP 추정치 기반) |
| `avg_temp_c` / `peak_temp_c` | 구간 온도 — 안정성 확인용 |
| `sm_clk_mhz` / `mem_clk_mhz` | 구간 말미 클럭 — throttle 여부 |
| `notes` | 주의 메모 (e.g. "fp8 emulated", "TE fallback") |

### raw samples CSV (`_samples.csv`)

100 Hz NVML 폴링 결과 전체. `t_s, power_w, temp_c, sm_mhz, mem_mhz, gpu_util, mem_util, phase`. timeline plot / 재분석용.

## 설치 & 사전 점검

```bash
pip install -r requirements.txt
# torch 는 CUDA 런타임 맞춰서
pip install torch --index-url https://download.pytorch.org/whl/cu121
# matmul_fp8_te variant 를 돌리려면 (A100/H100 모두):
./install_transformer_engine.sh
python3 preflight.py
```

### Transformer Engine 설치 주의사항

`pip install transformer_engine` 만 치면 다음 에러가 뜹니다:

```
RuntimeError: Found empty `transformer-engine` meta package installed.
Install `transformer-engine` with framework extensions via
'pip3 install --no-build-isolation transformer-engine[pytorch,jax]==VERSION'
```

이유: PyPI 의 `transformer-engine` 단독은 **meta-package** (shim) 이고 실제 기능은 `[pytorch]` / `[jax]` extra 에 들어있습니다. 올바른 설치:

```bash
# 제공된 스크립트 사용 (권장) — nvcc / torch 사전점검 + 설치 + 검증
./install_transformer_engine.sh
# 버전 고정:  TE_VERSION=1.11.0 ./install_transformer_engine.sh
# 수동 설치도 가능 :
pip install --no-build-isolation 'transformer-engine[pytorch]'
```

**반드시 `--no-build-isolation`** : TE 는 설치 시 CUDA 커널을 JIT 컴파일하므로 현재 환경의 `torch` 헤더와 `nvcc` 를 봐야 합니다. 기본 build-isolation 은 격리된 temp venv 에서 빌드해 torch/nvcc 를 못 찾고 실패합니다.

빌드 prerequisites:
- CUDA toolkit (`nvcc` on PATH) — `torch` 런타임만 있으면 부족. `apt install cuda-toolkit-12-1` 또는 `conda install -c nvidia cuda-toolkit=12.1`
- `cudnn-dev` (ubuntu) / `cudnn`
- 빌드에 5–10분, 수 GB RAM 소모

**A100 에서도 TE 를 설치하는 것이 의미 있나?** 네. A100 (sm_80) 에는 native FP8 tensor core 가 없어 TE 의 `fp8_autocast` 는 내부적으로 **FP16 Tensor Core 로 fallback** 합니다 (`notes` 컬럼에 "TE fallback" 이 찍힘). A100 의 `matmul_fp8_te` slope 가 `matmul_fp16_tc` 와 거의 동일하게 측정되는 것 자체가 **"FP8 이득은 Hopper 실리콘 의존"** 임을 증명하는 데이터 — cross-GPU 비교에서 빠져선 안 됩니다.

TE 빌드가 정말 번거로운 경우에는 A100 에서 `--matmul-variants fp32:simt tf32:tc fp16:tc bf16:tc` 로 fp8_te 를 제외해도 됩니다. `compare_gpus.py` 는 한 쪽에만 있는 variant 를 자동으로 "—" 로 표시합니다.

`preflight.py` 확인 항목:
- `nvidia-smi` / `pynvml` 동작
- `torch.cuda.is_available()` + compute capability
- `torch.float8_e4m3fn` 존재 여부 (torch ≥ 2.1)
- Tensor Core support matrix (`fp16_tc/bf16_tc/tf32_tc/int8_tc/fp8_tc`)
- `transformer_engine` 설치 (H100 에서 필수, A100 에선 선택)
- NVML `power.draw` 읽기 가능 (Joule 적분 필수)
- Persistence mode (권장: `sudo nvidia-smi -pm 1`)

FAIL 항목은 해결 후 재실행. A100 에서 `fp8 emulated` 는 정상 경고.

## 실행

```bash
cd util/gpu_power_bench

# A100 full sweep (약 15–30분 with cool-down)
./run_bench.sh --tag a100

# H100 full sweep
./run_bench.sh --tag h100

# 빠른 smoke test (3 loads, 3 matmul sizes)
./run_bench.sh --quick --tag smoke

# 서브셋만
./run_bench.sh --ops mul add --dtypes fp16     # FP16 eltwise mul/add
./run_bench.sh --matmul-variants fp16:tc fp8:te --no-cooldown
./run_bench.sh --no-matmul                     # elementwise 만
```

주요 옵션:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--device N` | 0 | CUDA 디바이스 인덱스 |
| `--ops ...` | 전체 5 | elementwise 연산 선택 |
| `--dtypes ...` | `fp16 fp8` | elementwise dtype 선택 |
| `--loads ...` | 256K..256M 6단 | elementwise load (tensor element 수) |
| `--no-matmul` | off | matmul 벤치 skip |
| `--matmul-sizes` | `512 1024 2048 4096 8192` | K 값 sweep |
| `--matmul-variants` | 전체 5 | `dtype:mode` 형식 (e.g. `fp16:tc`) |
| `--window-ms` | 1500 | cell 측정 길이 — 길수록 NVML 노이즈 ↓ |
| `--static-seconds` | 8 | idle 측정 시간 |
| `--cooldown-c` | 50 | 실험 간 목표 온도 (°C) — `-1` 이면 disable |
| `--cooldown-timeout` | 120 | cool-down 최대 대기 (초) |
| `--no-cooldown` | off | cool-down 생략 (빠름 / 덜 안정적) |
| `--tag` | — | 출력 파일명 suffix |
| `--poll-hz` | 100 | NVML 폴링 주파수 |

## 분석

### 1. 단일 GPU 분석 — `analyze.py`

```bash
python3 analyze.py reports/gpu_power_bench_a100_80gb_20260420_142301.csv
```

생성물 (5개):

| 파일 | 내용 |
|---|---|
| `<stem>_summary.csv` | **(category, op, dtype, mode) 별 `slope_dyn` = power-model 계수 `k_op`**, `R2_dyn` (선형성), 평균 power / temp |
| `<stem>_linearity_elementwise.png` | op × dtype: `E_dyn vs N`, `wall vs N`, `J/elem` 3-row grid. log-log 에서 기울기 1 직선이 이상적 |
| `<stem>_linearity_matmul.png` | matmul variant: `E_dyn vs FLOPs`, `wall vs FLOPs`, `J/FLOP` (x축이 FLOPs 인 이유는 K³ 스케일) |
| `<stem>_joule_per_op_bar.png` | 좌: elementwise op 별 J/elem, 우: matmul variant 별 J/FLOP. 막대 위에 R² 표시 |
| `<stem>_timeline.png` | 전체 런의 power / temp / SM·MEM clock 타임라인. 각 cell 구간이 살짝 shading |

summary CSV 를 읽는 법:
- `slope_dyn` (elementwise) = "이 op 의 J/element" → 그대로 `k_op` 로 사용
- `slope_dyn` (matmul) = "이 variant 의 J/FLOP" → matmul `k_op`
- `R2_dyn ≥ 0.99` 면 선형 모델 OK. 낮으면 해당 변형만 load 범위 좁혀서 재측정.

### 2. A100 vs H100 비교 — `compare_gpus.py`

```bash
python3 compare_gpus.py \
    reports/gpu_power_bench_a100_80gb_20260420_120000.csv \
    reports/gpu_power_bench_h100_sxm_20260420_140000.csv \
    --baseline a100_80gb --tag a100_vs_h100
```

생성물:

| 파일 | 내용 |
|---|---|
| `gpu_compare_<stamp>_summary.csv` | 모든 (GPU, variant) 의 `slope_dyn` + `ratio_vs_baseline` (= slope / baseline_GPU 의 slope) |
| `gpu_compare_<stamp>_bar.png` | variant 별 J/op 를 GPU 를 hue 로 나눈 막대 그래프 (좌: elementwise, 우: matmul) |
| `gpu_compare_<stamp>_heatmap.png` | (variant × GPU) 비율 heatmap — green < 1 (H100 이 싸다) / red > 1 (비싸다) |
| `gpu_compare_<stamp>_static.png` | GPU 별 `P_static` 막대 — `E = P_static·T + Σ k_op·N_op` 의 정적 항 |

**읽는 법 예시** (기대되는 결과):
- `matmul_fp32_simt` 이 두 GPU 에서 단연 가장 큰 J/FLOP (TC 미사용)
- `matmul_fp16_tc` 가 H100 에서 A100 대비 약 0.4–0.6× J/FLOP (TC 세대 개선)
- `matmul_fp8_te` 는 **H100 에서만** FP16_tc 대비 2–3× 더 저렴; A100 에선 `notes` 에 "TE fallback to FP16 TC" 가 찍히고 slope 가 FP16_tc 와 비슷 → FP8 이득이 Hopper 실리콘 의존임을 실측으로 증명
- Elementwise 는 BW bound 이므로 H100 이득은 HBM3/HBM2e 비율(≈ 1.5–2×) 을 반영
- `P_static`: A100 SXM 약 50–60 W / H100 SXM 약 70–90 W (카드 스펙에 따름)

## A100 ↔ H100 support 매트릭스 (한눈에)

| 기능 | A100 (sm_80) | H100 (sm_90) | 본 bench 가 드러내는 것 |
|---|---|---|---|
| FP32 CUDA core MAD | ✓ | ✓ | `matmul_fp32_simt` baseline |
| TF32 Tensor Core | ✓ (A100 도입) | ✓ | `matmul_tf32_tc` |
| FP16 Tensor Core | ✓ | ✓ (throughput ↑) | `matmul_fp16_tc` 두 GPU 간 비율 |
| BF16 Tensor Core | ✓ | ✓ | `matmul_bf16_tc` — FP16 과 같은 속도 / 다른 numerics |
| INT8 Tensor Core | ✓ | ✓ | (미구현 — 확장 후보) |
| **FP8 Tensor Core** | ✗ | ✓ (E4M3 / E5M2) | `matmul_fp8_te` — H100 의 핵심 에너지 이득 |
| Transformer Engine | fallback | native | 같은 `fp8_te` 코드가 A100 에선 FP16 TC 로 떨어짐 |
| FP8 elementwise | 에뮬 (cast→fp16→cast) | 에뮬 (Tensor Core 가 아니라 SIMT elem) | `fp8_mul/add/softmax/...` 10개 |
| DRAM | HBM2e 2039 GB/s raw | HBM3 ~3350 GB/s raw | Elementwise 에서 H100 이 대체로 더 싼 이유 |

## 안정성 체크리스트

실험 결과가 **신뢰 가능한 signal** 이려면:

1. **`R2_dyn ≥ 0.99`** (summary CSV 에서 per-variant) → linear 모델 OK. 낮으면 sweep 범위가 launch-overhead 또는 BW-saturation 을 포함.
2. **`peak_temp_c ≤ cooldown_c + 15°C`** — thermal throttle 없음. 초과 시 `--cooldown-c` 내리고 `--cooldown-timeout` 올리기.
3. **`sm_clk_mhz` ≈ max SM clock** — `nvidia-smi -q -d CLOCK` 의 Max 와 비교. 하락 → power cap 또는 thermal throttle. `sudo nvidia-smi -pl <W>` 로 상향.
4. **`P_static` 분산 < 1 W** — baseline std 가 크면 background 프로세스. `nvidia-smi` 로 다른 proc 확인.
5. **Persistence mode = ON** — `sudo nvidia-smi -pm 1`. 드라이버 재초기화 레이턴시 제거.
6. **같은 GPU 의 서로 다른 run 간 `slope_dyn` 편차 < 5%** — 시간대 / 냉각 / 이웃 카드 thermal cross-talk 체크.

## Power modeling 워크플로우 (권장)

1. A100 한 대에서 `./run_bench.sh --tag a100_run1` → CSV 생성.
2. 같은 조건으로 최소 한 번 더 실행 (`_run2`) → 재현성 (slope 편차) 확인.
3. H100 에서 동일. Transformer Engine 설치 전·후 두 번 돌리면 `fp8_te` 의 fallback vs native 도 같은 GPU 안에서 비교 가능.
4. `analyze.py` 로 각 CSV → summary + 선형성 plot.
5. `compare_gpus.py` 로 A100 run1 / A100 run2 / H100 run1 / H100 run2 통합 → cross-GPU ratio heatmap.
6. `summary.csv` 의 `slope_dyn` 컬럼을 per-op `k_op` 테이블로 export → AccelWattch (또는 자체 모델) 의 연산별 에너지 항에 대입.
7. 모델 예측 `E_model = P_static·T + Σ k_op·N_op` vs NVML 실측 `E_total` 을 실제 워크로드 (transformer layer, conv, 등) 로 검증.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `preflight.py` | deps / driver / TE / FP8 capability / NVML power 읽기 점검 |
| `power_monitor.py` | NVML 폴러 + 에너지 적분 + `measure_static_power` + `wait_for_cooldown` |
| `benchmarks.py` | 10 elementwise + 5 matmul = 15 benchmark factory |
| `gpu_power_bench.py` | 드라이버 (plan 생성 → cool-down → warmup → 측정 → CSV) |
| `analyze.py` | 단일 GPU summary CSV + 선형성 / bar / timeline plot |
| `compare_gpus.py` | 복수 GPU 비교 summary + bar / heatmap / static plot |
| `run_bench.sh` | persistence mode + deps 체크 + 런처 |
| `requirements.txt` | torch≥2.1, pynvml, numpy, pandas, nvtx, matplotlib |

## 알려진 한계

- **NVML `power.draw` 는 내부적으로 ~20 Hz 갱신**. 100 Hz 폴링은 중복 샘플을 포함 — 평균·적분은 정확하지만 "피크 power" 는 과소평가될 수 있음. 더 높은 정밀도가 필요하면 외부 HMC(HMP) 측정기.
- **FP16 matmul 은 TC 를 강제로 끄기 어려움** — PyTorch 는 Ampere+ 에서 fp16 matmul 을 항상 TC 로 보냄. 따라서 "CUDA core 만" baseline 은 `matmul_fp32_simt` (TF32 off) 하나이며, 같은 precision 축의 CUDA core 대비 TC 비교는 제공되지 않음.
- **Elementwise 는 memory-bound**: `j_per_element_dyn` 은 대부분 HBM read/write 에너지. compute-bound 모델 항을 직접 분리하고 싶다면 FMA 인텐시브 custom 커널 (dram_util_experiment 와 유사한 틀) 를 추가.
- **FP8 elementwise 는 Tensor Core 와 무관** — cast→fp16→compute→cast 경로. H100 도 이 부분은 SIMT 에서 돔. TC FP8 이득은 `matmul_fp8_te` 에서만 나타남.
- **FP8 scale factor 는 기본값**: 실제 학습/추론 파이프라인의 `DelayedScaling` amax history 동작과는 다름 — GEMM 자체 kernel 에너지 측정은 대표성 있으나, scale-update 오버헤드는 포함되지 않음.
- **TF32/FP32 flag 는 global**: 벤치마크 커널 내부에서 매번 재설정하지만, 같은 프로세스 안의 다른 코드와 공유되므로 외부 코드와 병행 실행 금지.

## 확장 아이디어

- **INT8/INT4 Tensor Core 변형**: `matmul_int8_tc` 추가 (양자화 추론 모델용)
- **Attention microbench**: scaled-dot-product + softmax 결합, FlashAttention 경로 비교
- **Conv**: cuDNN TF32 / FP16 / BF16 경로 — transformer 외 CNN 모델링
- **Per-SM power counters**: `nvidia-smi dmon -s pucvmet` 를 병행 로깅해 NVML board-level 과 SM-level 을 분리
