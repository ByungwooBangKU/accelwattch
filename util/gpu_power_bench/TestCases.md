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

cell 키 5-튜플은 `gpu_power_bench.py` 의 `broken_variants` 추적과 동일한 식별자 — 한 cell 이 죽으면 같은 (op, dtype, mode, llm_preset) 의 나머지 load 점들도 자동 skip 됨 (`README §8.3.3`).

---

## A. Workload 실험 — `gpu_power_bench.py`

### A.1 Elementwise sweep (`category = elementwise`)

#### 목적
"compute-light, memory-heavy" 영역의 J/element 와 cache-regime 별 누설을 정량. AccelWattch 의 elementwise k_op 계수 회귀에 사용.

#### 동작
N 개 원소짜리 입력 텐서에 한 op 를 매 iteration 적용. iteration 수는 `--window-ms` 가 채워지도록 자동 결정 (기본 3000 ms ≈ NVML 60샘플).

#### Cell 정의
| op | 의미 | bytes/elem (fp16) | flops/elem |
|---|---|---|---|
| `mul` | `c = a * b` | 6 | 1 |
| `add` | `c = a + b` | 6 | 1 |
| `softmax` | last-dim softmax | 4 | ~5 |
| `gelu` | torch.nn.functional.gelu | 4 | ~10 |
| `layernorm` | per-row LayerNorm | 4 | ~7 |

dtype × ops × load 카르테시안 곱 = **기본 5 op × 2 dtype × 11 load = 110 cell**.

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
| `--matmul-sizes` | `512 1024 1536 2048 3072 4096 6144 8192 12288` | K 리스트 |
| `--matmul-variants` | 5종 전체 | `dtype:mode` list. 예: `--matmul-variants fp16:tc fp8:te` |

K=12288 fp32 ≈ 1.7 GB → A100/H100 80GB 모두 OK. fp32 대형 K 는 OOM 위험 → `_filter_loads` 와 동일 보호.

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

`gpu_power_bench.py` 의 sweep 과는 독립. **GPU 자체의 power 봉투** (static / max / leakage) 3 점만 짧고 굵게.  실행 시간 ~10 분 (기본 옵션).  AccelWattch 의 `P_static` / `P_max` / leakage 의 온도 의존성 파라미터 cross-check 용.

### B.1 Static (idle baseline)

#### 목적 / 동작
`--static-seconds` (기본 60 s) 동안 어떤 CUDA kernel 도 launch 하지 않은 채 NVML 100 Hz 로 power/temp 샘플. clock gate 된 base 전력. `gpu_power_bench.py` 의 `static_power_w` 와 일치해야 함.

#### 산출물
- summary CSV row: `static_power_w_mean / _peak`, `static_temp_c_mean / _peak`.
- timeline PNG 의 첫 phase 영역.

---

### B.2 Max (TGP saturation)

#### 목적 / 동작
큰 정사각 GEMM (기본 `K=16384`, `fp16/tc`) 을 `--max-seconds` (기본 60 s) 동안 batch 32 단위로 연속 launch → SM 포화 → 데이터시트 TGP 근처로 수렴. 1 s 의 warmup 은 stat 에서 제외.

#### 입력 파라미터
| flag | 기본 |
|---|---|
| `--max-seconds` | 60 |
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
1. `--leakage-stress-s` (기본 20 s) 동안 GEMM 스트레스 → 온도 ramp.
2. 즉시 멈추고 `--leakage-decay-s` (기본 30 s) 동안 idle. 첫 `--leak-window-s` (기본 1 s) 의 평균 power 가 **hot leakage** 값.
3. 30 s 후 다음 cycle 시작 (silicon 이 일부만 식음 — 의도적, cycle 간 thermal 누적도 평균).

#### 입력 파라미터
| flag | 기본 |
|---|---|
| `--no-leakage` | (skip) |
| `--leakage-cycles` | 5 |
| `--leakage-stress-s` | 20 |
| `--leakage-decay-s` | 30 |
| `--leak-window-s` | 1.0 |

#### 산출물
- summary CSV 본문: 5 cycle 평균 `leakage_power_w_mean`, `leakage_minus_static_w` (≡ thermal leakage Δ), `leakage_temp_c_mean`.
- summary CSV 하단 표: 사이클별 `stress_temp_c_peak`, `stress_power_w_mean`, `hot_temp_c_peak`, `hot_power_w_mean`, `hot_minus_static_w`.
- `_leakage.png`: 5 decay 곡선을 t=0 (스트레스 종료) 기준으로 overlay → 누설 감쇠 가시.

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

## D. Test Suites — `--suite NAME`

자주 쓰이는 옵션 묶음의 사전 정의. 사용자 explicit flag 가 항상 우선.

| suite | 포함 | 시간 | 용도 |
|---|---|---|---|
| `smoke` | A.1 (quick), C off | ~5 분 | 파이프라인 sanity check |
| `powermodel` | A.1 + A.3 | ~25 분 | 표준 baseline (default) |
| `cache` | A.1 (cache-sweep, 3 regime 각 1점) + A.3 | ~20 분 | locality 분석 전용 |
| `dram` | A.2 only | ~10 분 | pJ/bit 단독 추출 |
| `llm` | A.4 only | ~15 분 | LLM-shape 만 |
| `full` | A.1 + A.2 + A.3 + A.4 + C.1 (`--rebaseline-every 20`) | ~75 분 | publication-quality |

표 의 시간은 H100 SXM 80GB 기준. 작은 GPU 는 `_filter_loads` 가 큰 셀 drop → 더 짧음.

#### 사용 예
```bash
./run_bench.sh --suite full --tag h100        # publication-grade
./run_bench.sh --suite cache --tag a100       # cache regime 만 빠르게
./run_bench.sh --suite dram --tag h100        # pJ/bit 단독
./run_bench.sh --suite llm  --tag h100 --llm-dtypes bf16:tc fp16:tc    # LLM, fp8 제외
./run_bench.sh --suite full --no-matmul       # full 에서 matmul 만 빼기
```

suite 의 *어떤 필드든* 개별 flag 로 override 가능 — `__init__` 시 `_apply_suite_to_parser` 가 default 만 바꾸고 사용자 argv 가 항상 우선이라 그렇다 (`gpu_power_bench.py:168`).

#### Suite 추가 / 변경
새 suite 는 `gpu_power_bench.py` 의 `SUITES` dict 에 한 줄 추가. 예시:

```python
"thermal": {
    "_doc":             "long-soak thermal study (3× window, full)",
    "llm_shapes":       True,
    "dram_bw_test":     True,
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
| `reports/soc_power_..._leakage.png` | B.3 | 5 decay 곡선 overlay |
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
