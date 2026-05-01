# TestCases.md — 실험 카탈로그

`util/gpu_power_bench/` 가 측정하는 모든 실험을 **대분류 → 중분류 → 개별 cell** 의 3 단계로 정리한 카탈로그. 각 항목엔 *목적 / 동작 / 입력 파라미터 / 산출물* 을 표기.

전체 흐름:

```
A. Workload 실험   (gpu_power_bench.py)   ─── per-cell J / W / k_op coefficient
   ├── A.1 Elementwise sweep              5 op × 2 dtype × 11 load
   ├── A.2 DRAM bandwidth probes (STREAM) 5 op × 2 dtype × N load
   ├── A.3 Square matmul                  5 variant × 9 K-size
   └── A.4 LLM-shape matmul               8 preset × 5 token × M variant

B. SoC envelope     (soc_power_bench.py)  ─── static / max / leakage 3 점
   ├── B.1 Static (idle baseline)
   ├── B.2 Max (large GEMM saturation)
   └── B.3 Leakage (5 cycle stress→decay)

C. Drift correction (gpu_power_bench.py)  ─── 주기적 P_static 재측정
   └── C.1 Periodic re-baseline (--rebaseline-every)

D. Test suites      (--suite NAME)         ─── 위 실험들의 사전정의 묶음
   └── smoke / powermodel / cache / dram / llm / full
```

본 문서는 **"무엇을 / 왜 / 어떻게" 측정하는가**를 정리한다. 알고리즘적 배경과 단위 환산 공식은 `README.md` 의 §2~§7 을 참조.

---

## 0. 분류 체계

| 축 | 값 | 의미 |
|---|---|---|
| **대분류** | A / B / C / D | A=workload, B=SoC envelope, C=drift, D=suite preset |
| **중분류** | A.1~A.4, B.1~B.3, … | 실험 *카테고리*. 분석 / k_op 회귀 / 파일이 카테고리 단위로 나뉨 |
| **cell** | (op, dtype, mode, load, llm_preset) 튜플 | 한 번의 power-sampled 측정 단위. CSV 의 한 row |

cell 키는 5-튜플 `(op, dtype, mode, llm_preset, load_value)` 이지만 **broken_variants 가 추적하는 *variant* 키는 앞 4 개** — load 만 다른 같은 (op, dtype, mode, llm_preset) 의 나머지 cell 들도 자동 skip 됨 (`README §8.3.3`). 즉 fp8_te 가 한 K 에서 죽으면 그 variant 의 모든 K 값이 skip 됨.

---

## A. Workload 실험 — `gpu_power_bench.py`

### A.1 Elementwise sweep (`category = elementwise`)

#### 목적
"compute-light, memory-heavy" 영역의 J/element 와 cache-regime 별 누설을 정량. AccelWattch 의 elementwise k_op 계수 회귀에 사용.

#### 동작
N 개 원소짜리 입력 텐서에 한 op 를 매 iteration 적용. iteration 수는 `--window-ms` 가 채워지도록 자동 결정 (기본 3000 ms ≈ NVML 60샘플).

#### Cell 정의

| op | 의미 | logical I/O (per call) | RW_PER_CALL | FLOP/elem | DRAM pass (실제) | 비고 |
|---|---|---|---|---|---|---|
| `mul` | `c = a · b` | 2R + 1W | 3 | 1 | 1× | simple — pJ/bit literature 비교 가능 |
| `add` | `c = a + b` | 2R + 1W | 3 | 1 | 1× | simple — pJ/bit literature 비교 가능 |
| `softmax` | row-wise softmax | 1R + 1W | 2 | ~5 | **2~3×** (max → exp+sum → norm) | reduction; pJ/bit 가 inflate |
| `gelu` | tanh-approx GeLU | 1R + 1W | 2 | ~8 | 1× | heavy compute → l2_hit_0/100 SM 패턴 비대칭 |
| `layernorm` | per-row LayerNorm | 1R + 1W | 2 | ~8 | **2~3×** (mean → var → norm) | reduction; pJ/bit 가 inflate |

`bytes/elem` (fp16) = `RW_PER_CALL × 2`. `RW_PER_CALL` 은 *논리적* read/write 횟수이고 (`analyze.py:145`), 실제 DRAM pass 는 reduction op 의 경우 더 많음 — **자세한 caveat 은 README §3.5.3 참조**.

`FLOP/elem` 는 `benchmarks.py:32` 의 `FLOP_PER_ELEMENT` 와 일치. 여기서 `softmax = 5` (max, sub, exp, sum, div), `gelu = 8` (tanh 근사 분해), `layernorm = 8` (mean, var, sub, div, mul, add 등). FLOP 카운트가 큰 op 는 J/byte 의 **direct** 측정값에 SM compute 가 비례적으로 섞여 들어감 — `dram_energy_pjbit.png` 에서 add/mul 보다 softmax/gelu/layernorm 이 2~3 배 높게 나오는 이유. 이건 측정 버그가 아니라 *모델의 적용 정확도가 op 마다 다른* 결과.

dtype × ops × load 카르테시안 곱 = **기본 5 op × 2 dtype × 11 load = 110 cell**. (`--cache-sweep` 사용 시엔 11 load 가 3 regime 점 (L2-resident / partial / dram-stream) 으로 줄어 5 × 2 × 3 = 30 cell. `--quick` 은 3 load 점 → 30 cell.)

#### 입력 파라미터
| flag | 기본 | 비고 |
|---|---|---|
| `--ops` | `mul add softmax gelu layernorm` | 부분집합 가능 |
| `--dtypes` | `fp16 fp8` | fp8 은 cast-compute-cast 경로 (emulated) |
| `--loads` | 11점 (`128K…1G`) | `_filter_loads` 가 25%-HBM 한도 초과 셀 자동 drop |
| `--quick` | off | 3점만 (`1M / 4M / 16M`) |
| `--cache-sweep` | off | 11점 대신 L2-resident / partial / dram-stream 3 regime 만 |
| `--no-elementwise` | off | 카테고리 전체 skip |
| `--window-ms` | `3000` | per-cell 측정 윈도 |

#### 산출물
- `gpu_power_bench_<gpu>_<ts>[_<tag>].csv` 의 row 들 — column: `j_per_element_dyn`, `j_per_element_total`, `j_per_flop_dyn`, `cache_regime`, `peak_temp_c`, `clip_bias_*`, …
- `analyze.py` 가 그리는 elementwise k_op bar / log-log slope / cache-regime split plot.

---

### A.2 DRAM bandwidth probes — STREAM-style (`category = stream`)

#### 목적
순수 DRAM 트래픽의 pJ/bit 추정. compute-light 라 dyn power ≈ HBM ↔ DRAM 이동 에너지 → 온보드 측정 경계 안에서의 메모리 비용.

#### Cell 정의
| op | 동작 | 데이터 패턴 | 비고 |
|---|---|---|---|
| `stream_copy` | `out = in` | R + W | STREAM `Copy` |
| `stream_scale` | `out = α·in` | R + W (compute 1 mul) | STREAM `Scale` |
| `stream_triad` | `out = a + α·z` | 2R + W | STREAM `Triad` |
| `stream_read` | `s = sum(in)` | R only | DRAM read 분리 측정 (R/W split) |
| `stream_write` | `out.fill_(c)` | W only | DRAM write 분리 측정 |

읽기/쓰기 분리 cell 이 있어 **marginal DRAM cost = `J/byte(l2_hit_0) − J/byte(l2_hit_100)`** 로 SM/L2 baseline 을 상쇄한 read 단가 / write 단가를 분리 추출 가능 (`README §3.5`, PR #30).

#### 입력 파라미터
| flag | 기본 | 비고 |
|---|---|---|
| `--dram-bw-test` | off | 카테고리 enable |
| `--dram-bw-loads` | 4 점 (deep `l2_hit_0`) | 사용자 override 가능 |
| `--dtypes` | A.1 와 공유 | fp16/fp8 |

#### 산출물
- 동일 sweep CSV 안의 `op = stream_*` row 들.
- `analyze.py` 의 DRAM pJ/bit 막대그래프 + read/write/copy/scale/triad overlay (HBM2 ≈ 7 pJ/bit, HBM3 ≈ 4 pJ/bit literature 가이드).
- **STREAM probe 가 elementwise 보다 literature 비교에 더 정확** : compute 가 거의 없어 (`stream_copy=0 FLOP`, `stream_scale=1`, `stream_triad=2`) SM baseline 이 깨끗이 cancel 됨. softmax/gelu/layernorm 의 marginal 이 inflate 되는 multi-pass / heavy-compute 문제 (README §3.5.3) 가 STREAM 에선 발생 안 함.

---

### A.3 Square Matmul (`category = matmul`, 5 variant)

#### 목적
Tensor Core 대 CUDA Core 의 **에너지 효율 격차**, dtype/precision 별 J/FLOP, FP8 의 (Hopper TC 또는 fp16 fallback) 비용 정량.

#### Cell 정의 (`(dtype, mode)` 5종)
| variant | 컴퓨트 path | 요구 | 비고 |
|---|---|---|---|
| `fp32:simt` | CUDA core (TF32 OFF) | any | TC 끔 baseline. J/FLOP 가 가장 큼. |
| `tf32:tc` | TF32 Tensor Core | sm_80+ | mantissa 10-bit |
| `fp16:tc` | FP16 Tensor Core | any | wmma path |
| `bf16:tc` | BF16 Tensor Core | sm_80+ | FP16 와 peak 동일, 더 큰 dynamic range |
| `fp8:te` | FP8 (E4M3) via Transformer Engine | sm_89+ native, 그 이하 fp16 fallback (`emulated=true`) | Hopper sm_90 권장. **Blackwell sm_120 의 amax buffer race** 에 대해 `README §8.3.3` 참조 |

각 variant 가 **K-sweep** 9 점 (`512..12288`) 을 돔.  M = N = K (square).

#### 입력 파라미터
| flag | 기본 | 비고 |
|---|---|---|
| `--no-matmul` | off | 카테고리 전체 skip |
| `--matmul-sizes` | `1024 2048 2880 4096 5760 8192 12288` | **GPT-OSS 120B aware** — README §3.2 |
| `--matmul-variants` | 5종 전체 | `dtype:mode` list. 예: `--matmul-variants fp16:tc fp8:te` |

**Default K 7 점 — GPT-OSS 120B layer dim 매핑** :

| K | 매핑 |
|---|---|
| 1024, 2048 | TC sweet spot baseline (fp32_simt / tf32_tc 에서 의미있는 dyn power) |
| **2880** | GPT-OSS hidden dim → `qkv` / `q_only` / `kv` / `mlp1` / `mlp2` / `lm_head` 의 contraction |
| 4096 | GPT-OSS `attn_o` input (head_dim × heads) |
| **5760** | GPT-OSS MLP intermediate (`mlp1` out / `mlp2` in) |
| 8192, 12288 | BW saturation 영역 |

이 set 의 의도 : square sweep 결과가 LLM-shape (A.4) 의 같은 K 점들과 J/FLOP 영역에서 **cross-check** 가능하게.

K=12288 fp32 ≈ 1.7 GB → A100/H100 80GB 모두 OK. K=16384 는 fp32 가 3.2 GB → default 빠짐, fp8_te 단독 sweep 시 `--matmul-sizes ... 16384 --matmul-variants fp8:te` 로 명시 추가 권장. 옛 default 의 K=512..1536 은 H100 fp8_te 의 noise floor 아래라 (§8.3.4) drop.

#### 산출물
- CSV row: `op=matmul, dtype, mode, load_value=K`, FLOPs = 2·K³.
- `analyze.py` 의 (a) variant 별 J/FLOP bar (b) K vs J(dyn) log-log (c) Tensor-Core 효율 gap heatmap.

---

### A.4 LLM-shape Matmul (`category = llm_matmul`)

#### 목적
정사각 GEMM 이 아닌 **실제 Transformer 추론 layer 의 비대칭 GEMM** 에서 J/FLOP 변화. Decode (`T=1`, latency-bound) → Long-context prefill (`T=32k`) 까지의 token-count 의존성 측정.

#### Preset 정의 (gpt-oss-120B 클래스 reference)
| preset | (K, N) | 의미 |
|---|---|---|
| `qkv` | (2880, 5120) | QKV 합쳐진 projection |
| `q_only` | (2880, 4096) | Q head |
| `kv` | (2880, 512) | GQA K/V (skinny output) |
| `attn_o` | (4096, 2880) | attention output |
| `router` | (2880, 128) | MoE gate (extreme skinny) |
| `mlp1` | (2880, 5760) | MoE expert up |
| `mlp2` | (2880, 2880) | MoE expert down |
| `lm_head` | (2880, 201088) | unembedding (extreme fat) |

총 cell = `presets × tokens × (dtype, mode)` = 기본 8 × 5 × 1 = **40 cell** (dtype 1 일 때).  M dim = 토큰 수 T.

#### 입력 파라미터
| flag | 기본 | 비고 |
|---|---|---|
| `--llm-shapes` | off | 카테고리 enable |
| `--llm-presets` | 8개 전부 | subset 가능 (예: `qkv mlp1 lm_head`) |
| `--llm-ts` | `1 256 2048 8192 32768` | M dim sweep |
| `--llm-dtypes` | `bf16:tc` | `dtype:mode` list. `fp8:te` 는 Blackwell sm_120 + 작은 M 에서 위험 (`README §8.3.3`) |

#### 산출물
- CSV row: `op=matmul, llm_preset=<name>, load_name=T_size, load_value=T`. 전체 cell 키는 `(matmul, dtype, mode, T, llm_preset)` 5-튜플.
- `analyze.py` 의 LLM-shape J/FLOP scatter — preset 별 색상, T-dim 별 점 크기.

---

## B. SoC Envelope — `soc_power_bench.py` (별도 스크립트)

`gpu_power_bench.py` 의 sweep 과는 독립. **GPU 자체의 power 봉투** (static / max / leakage) 3 점만 짧고 굵게.  실행 시간 ~5 분 (기본 옵션).  AccelWattch 의 `P_static` / `P_max` / leakage 의 온도 의존성 파라미터 cross-check 용.

> **Power source** : H100 (sm_90+) 이상에서는 `nvmlDeviceGetFieldValues(NVML_FI_DEV_POWER_INSTANT)` 로 ~1 ms 갱신 (transient 보존). 그 외 / 미지원 driver 면 legacy `nvmlDeviceGetPowerUsage` (~50 ms 평균). 자동 probe + fallback. 실제 사용 path 는 시작 시 `[info] power source: …` 로 노출되고 summary CSV `power_source` 컬럼에도 stamp.

### B.1 Static (idle baseline)

#### 목적 / 동작
`--static-seconds` (기본 20 s) 동안 어떤 CUDA kernel 도 launch 하지 않은 채 NVML 100 Hz 로 power/temp 샘플. clock gate 된 base 전력. `gpu_power_bench.py` 의 `static_power_w` 와 일치해야 함.

#### 산출물
- summary CSV row: `static_power_w_mean / _peak`, `static_temp_c_mean / _peak`.
- timeline PNG 의 첫 phase 영역.

---

### B.2 Max (TGP saturation)

#### 목적 / 동작
큰 정사각 GEMM (기본 `K=16384`, `fp16/tc`) 을 `--max-seconds` (기본 30 s) 동안 batch 32 단위로 연속 launch → SM 포화 → 데이터시트 TGP 근처로 수렴. 1 s 의 warmup 은 stat 에서 제외.

#### 입력 파라미터
| flag | 기본 |
|---|---|
| `--max-seconds` | 30 |
| `--matmul-K` | 16384 |
| `--dtype` | `fp16` |
| `--mode` | `tc` |
| `--no-max` | (skip) |

#### 산출물
- summary CSV: `max_power_w_mean / _peak`, `max_temp_c_mean / _peak`, `max_seconds`.
- `_phases.png` 의 max 영역에서 P(t) ramp + T(t) thermal-soak 곡선 가시.
- `_summary.png` 막대그래프의 max-mean / max-peak 두 막대.

---

### B.3 Leakage (hot-cold delta)

#### 목적
온도 의존 leakage current 를 분리. silicon 이 식어있을 때 (`P_static_cold`) 와 충분히 가열된 직후 (`P_hot_idle`) 의 차이가 thermal leakage component.

#### 동작
`--leakage-cycles` (기본 5) 회 반복:
1. `--leakage-stress-s` (기본 10 s) 동안 GEMM 스트레스 → 온도 ramp.
2. 즉시 멈추고 `--leakage-decay-s` (기본 15 s) 동안 idle. 첫 `--leak-window-s` (기본 1 s) 의 평균 power 가 **hot leakage** 값.
3. 15 s 후 다음 cycle 시작 (silicon 이 일부만 식음 — 의도적, cycle 간 thermal 누적도 평균).

#### 입력 파라미터
| flag | 기본 |
|---|---|
| `--no-leakage` | (skip) |
| `--leakage-cycles` | 5 |
| `--leakage-stress-s` | 10 |
| `--leakage-decay-s` | 15 |
| `--leak-window-s` | 1.0 |

#### 산출물
- summary CSV 본문: 5 cycle 평균 `leakage_power_w_mean`, `leakage_minus_static_w` (≡ thermal leakage Δ), `leakage_temp_c_mean`.
- summary CSV 하단 표: 사이클별 `stress_temp_c_peak`, `stress_power_w_mean`, `hot_temp_c_peak`, `hot_power_w_mean`, `hot_minus_static_w`.
- `_leakage.png`: 5 decay 곡선을 t=0 (스트레스 종료) 기준으로 overlay → 누설 감쇠 가시.
- `_leakage_enlarged.png`: 위와 동일 데이터의 **첫 3 s × 0–150 W 줌-인**. hot-window 안의 사이클 간 산포와 첫 1 s 의 급강하를 한눈에.

---

## C. Drift Correction

### C.1 Periodic re-baseline (`--rebaseline-every N`)

긴 sweep 도중 fan / coolant / 주변 온도 drift 로 `P_static` 이 1~3 W 흔들림. 이걸 보정하기 위해 **N cell 마다** 짧게 (`--rebaseline-seconds`, 기본 4 s) 재측정 → 직후 cell 들의 dyn power 계산에 새 baseline 사용.

#### 입력 파라미터
| flag | 기본 |
|---|---|
| `--rebaseline-every` | `0` (한 번만) |
| `--rebaseline-seconds` | `4.0` |

#### 산출물
- sidecar CSV: `<base>_rebaseline.csv` — 각 재측정 시점의 평균/표준편차/온도.
- `analyze.py` 의 P_static drift plot (시간축 vs `static_power_w`).

---

## D. Test cases & Test suites

실험 선택은 **두 축이 분리** :

- **Test case (`--cases`)** — 단일 카테고리. 5 개 중 자유 조합 가능 :
  - `elementwise` (A.1) / `matmul` (A.3) / `llm-matmul` (A.4) / `dram` (A.2) / `soc` (B)
- **Test suite (`--suite`)** — case 조합 + 튜닝의 사전정의. 사용자 explicit flag 가 항상 우선.

### D.1 Suite 표

| suite | cases | 추가 옵션 | 시간 |
|---|---|---|---|
| `smoke` | `elementwise` | `--quick` | ~5 분 |
| `powermodel` | `elementwise`, `matmul` | (default) | ~25 분 |
| `cache` | `elementwise`, `matmul` | `--cache-sweep` | ~20 분 |
| `dram` | `dram` | — | ~10 분 |
| `llm` | `llm-matmul` | — | ~15 분 |
| `soc` | `soc` | — | **~5 분** (옛 `run_soc_bench.sh` 와 동등) |
| `full` | `elementwise`, `matmul`, `llm-matmul`, `dram` | `--rebaseline-every 20` | ~75 분 |
| `all` | `full` + `soc` | `--rebaseline-every 20` | ~80 분 |

표 의 시간은 H100 SXM 80GB 기준. 작은 GPU 는 `_filter_loads` 가 큰 셀 drop → 더 짧음.

### D.2 사용 예

```bash
# Suite 한 줄
./run_bench.sh --suite all  --tag h100              # 모든 cases (sweep + SoC)
./run_bench.sh --suite full --tag h100              # SoC 제외 모든 sweep
./run_bench.sh --suite soc  --device 0 --tag h100   # SoC 만 (~5분)

# Cases 직접 조합
./run_bench.sh --cases dram soc --device 0 --tag h100_mem
./run_bench.sh --cases soc      --num-gpus 8 --tag h100   # 8 GPU SoC 동시

# Suite + 추가 override
./run_bench.sh --suite full --tag h100 --cases matmul   # full 에서 matmul 만
```

Suite 의 어떤 필드든 개별 flag 로 override 가능 (`_apply_suite_to_parser` 가 default 만 바꾸고 사용자 argv 가 우선).

### D.3 Legacy flag 호환성

옛 `--no-elementwise` / `--no-matmul` / `--llm-shapes` / `--dram-bw-test` 는 **`--cases` 가 명시 안 된 경우에만** 작동 — 그땐 자동으로 cases 셋으로 변환됨. `--cases` 명시되면 legacy flag 는 무시됨 (사용자가 명시한 cases 가 ground truth).

`run_soc_bench.sh` 도 deprecated alias 로 유지 — 옛 인자 (`--no-leakage`, `--leakage-stress-s`, `--matmul-K` 등) 는 새 `--soc-*` 이름으로 자동 번역되어 `run_bench.sh --suite soc` 로 forward.

### D.4 Suite 추가 / 변경

`gpu_power_bench.py` 의 `SUITES` dict 에 한 줄 추가 :

```python
"thermal": {
    "_doc":             "long-soak thermal study (3× window)",
    "cases":            ("elementwise", "matmul", "llm-matmul", "dram"),
    "rebaseline_every": 10,
    "window_ms":        9000,
},
```

`_doc` 는 `--help` epilog 에 자동 노출.

---

## E. 산출물 레퍼런스 요약

| 파일 | 출처 | 내용 |
|---|---|---|
| `reports/gpu_power_bench_<gpu>_<ts>[_<tag>].csv` | A.1 ~ A.4 | per-cell 측정값 (1 row = 1 cell) |
| `reports/...<tag>_baseline.csv` | A.* | sweep 시작/끝 idle 트레이스 (raw) |
| `reports/...<tag>_baseline_stats.csv` | A.* | mean/std/min/max 요약 |
| `reports/...<tag>_samples.csv` | A.* | 100 Hz raw sample 트레이스 |
| `reports/...<tag>_rebaseline.csv` | C.1 | drift correction 시점들 |
| `reports/soc_power_<gpu>_<ts>[_<tag>]_summary.csv` | B.1~B.3 | 한 줄 요약 + 5-cycle leakage detail |
| `reports/soc_power_..._timeseries.csv` | B.* | 100 Hz P/T/clock 트레이스 |
| `reports/soc_power_..._phases.png` | B.* | 전체 P(t) + T(t), phase 음영 |
| `reports/soc_power_..._leakage.png` | B.3 | 5 decay 곡선 overlay (전체 decay window) |
| `reports/soc_power_..._leakage_enlarged.png` | B.3 | 위 plot 의 첫 3 s × 0–150 W 줌-인 (hot-window 디테일) |
| `reports/soc_power_..._summary.png` | B.* | static / max / hot-leak 막대 |
| `reports/<tag>_*.png` (다수) | `analyze.py` | k_op bar / log-log / regime split / DRAM pJ/bit |
| `reports/gpu_compare_*_*.png` | `compare_gpus.py` | cross-GPU bar / heatmap |

자세한 column 명세는 `README §10` 참조.

---

## F. Cell 키와 Variant-skip 의 관계

5-튜플 `(op, dtype, mode, llm_preset, load_value)` 에서 앞 4 개가 *variant 키* — 한 cell 이 fatal CUDA 에러로 죽으면 같은 variant 의 나머지 load 점들이 자동 skip 됨 (`broken_variants` set). build / run / post-cell sync 세 계층에서 모두 fatal-marker 검사 (`README §8.3.3`).

→ cell 추가 / 새 카테고리 도입 시: plan dict 에 `op / dtype / mode / llm_preset / load_name / load_value` 6 키를 채워주면 자동으로 skip / drift-correction / rebaseline 흐름에 합류.

---

*이 문서는 `util/gpu_power_bench/` 의 README 에서 링크되며, 새 실험 카테고리가 추가될 때마다 §A~§D 표에 한 줄 추가하면 충분하다.*
