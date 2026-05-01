# Energy Decomposition Review

## 한 페이지 요약

본 suite (`util/gpu_power_bench/`) 가 **AccelWattch-style power model 의 각 항을 분리해 측정** 할 수 있는지 평가한 design/implementation review. 6 axis 별로 ✓ / △ / ✗ 평가 후 gap 과 priority 매김.

| Axis | 평가 | 핵심 |
|---|---|---|
| 1. Static vs Dynamic 분리 | ✓ | P-state filter + drift correction + clip-bias audit. **잘 됨** |
| 2. Compute path 분리 | ✓ | 5 matmul variant + emulated flag + LLM shape. **HW 한계까지 갖춤** |
| 3. Memory hierarchy 분리 | △ | DRAM read/write/marginal 깨끗. **L2 미만 (L1/SMEM/register) 은 bundled** |
| 4. k_op 추출 방법론 | ✓ | WLS + bootstrap CI + clip-bias + noise floor 자동 제외. **robust** |
| 5. Thermal & leakage | ✓ | SoC envelope (static / max / leakage 5-cycle). **board-level 적정** |
| 6. MECE Decomposition | ✓ | 3-component 항등식 (compute/cast/DRAM). **수학적 MECE** |

**한 줄 평** — DRAM 까지의 component 분리는 rigorous. **L2 안쪽 (L1 / register / pure compute) 은 NVML 측정의 *근본적 한계* 라 bundled** 되어 있고 그게 정직하게 명시됨. AccelWattch 수준의 power model 에 필요한 거의 모든 항을 채울 수 있음.

자세한 각 axis 의 ✓/△/✗ 근거와 한계는 아래 §1~§6, gap analysis 와 우선순위 권장은 §7~§8.

---

## Axis 1 — Static vs Dynamic Energy 분리   **✓ 잘 됨**

### 핵심 질문
*"`P_static` 이 정확하게 측정되어 `dyn_energy = E_total − P_static·t` 로 워크로드 비용만 깨끗이 분리되는가?"*

### 구현된 메커니즘

| 메커니즘 | 위치 | 역할 |
|---|---|---|
| Cold-idle 기본 측정 | `power_monitor.py:244` `measure_static_power()` | 시작 시 N 초 idle, mean/std/min/max 보고 |
| **P-state sample 필터** | `power_monitor.py:267-312` (PR #54) | `sm_mhz < 500 MHz` (= P8) sample 만 평균 → boost-clock idle hysteresis 자동 회피 |
| **Periodic re-baseline** | `gpu_power_bench.py` `--rebaseline-every N` | sweep 중 N cell 마다 재측정 → 1~3 W 의 thermal/coolant drift 보정 |
| **Pre-clip raw 값 보존** | `dyn_energy_j_raw` column | clip(max(0, raw)) 적용 전의 원본도 CSV 에 보관 |
| **Clip-bias audit** | `_fit_one_group()` `clip_bias_pct` | clipped vs unclipped slope 비교로 bias 정량 |
| **Noise-floor 자동 제외** | `_fit_one_group()` `pos_mask` (PR #53) | `dyn_energy_j ≤ 0` row 는 회귀에서 자동 drop |

### 강점

- **P-state hysteresis 자동 처리** — H100 의 116 W (P0 idle) vs 70 W (P8 cold idle) 차이를 sample 단계에서 자동 분리. nvidia-smi 와 ±2 W 안에서 일치. ([PR #54](https://github.com/ByungwooBangKU/accelwattch/pull/54))
- **Drift 추적 + 보정** — `_rebaseline.csv` sidecar 가 sweep 동안의 P_static drift trace 를 남김. analyze 가 plot 으로 시각화.
- **정직한 raw vs clip 보고** — clipped 값이 default 지만 raw 도 함께 저장 → clip-bias 가 ±2% 면 신뢰, 그 이상이면 user 에게 경고 (`P_static drift 의심, --rebaseline-every 권장`).
- **Sweep 시작 전 P_static 측정** — workload 가 GPU clock 을 띄우기 전 cold idle 에서 baseline 잡음. SoC envelope 도 build_matmul 을 phase_static 후로 옮겨 같은 보장 (PR #54).

### 한계

| 한계 | 의미 |
|---|---|
| `P_static` 은 **board-level whole-GPU idle** | 칩 leakage + HBM idle + PLL + VRM 합. 개별 component 분해는 하드웨어 telemetry 한계 (NVML 이 보드 전체 power 만 보고). AccelWattch 가 칩-단독 idle 만 원하면 별도 calibration 필요. |
| Drift correction 은 N-cell 단위 | 한 cell 안의 sub-second drift 는 못 잡음. `--rebaseline-every 20` 로 충분 정밀하지만 분 단위. |
| P-state filter 가 driver 의존 | 옛 NVML / 비-CUDA context 에선 `sm_mhz` 미보고 → filter 자동 disable + warn. fallback 동작 정확. |

### 평가 — ✓

P_static 측정에 필요한 모든 가드 (P-state, drift, clip-bias, noise floor) 가 갖춰져 있고 한계가 정직하게 noted. AccelWattch 의 `P_static` 항에 그대로 사용 가능한 품질.

---

## Axis 2 — Compute Path 분리   **✓ 잘 됨**

### 핵심 질문
*"CUDA core vs Tensor Core, 다른 dtype, native vs emulated path 가 명확히 식별 / 측정 / 라벨링되는가?"*

### 구현된 메커니즘

| 메커니즘 | 위치 | 역할 |
|---|---|---|
| 5 matmul variant | `benchmarks.py:541-547` `MATMUL_VARIANTS` | `(fp32, simt) / (tf32, tc) / (fp16, tc) / (bf16, tc) / (fp8, te)` — CUDA core 와 TC 의 4 dtype + TE FP8 |
| **TF32 강제 비활성** | `_make_matmul_fp32_simt()` | `torch.backends.cuda.matmul.allow_tf32 = False` per call → 확실히 SIMT 경로 |
| `compute_unit` column | `BenchSpec` dataclass | `"CUDA core"` / `"Tensor Core"` / `"Tensor Core (FP16 fallback)"` |
| **`emulated=1` flag** | `benchmarks.py:74` | fp8 elementwise (cast-compute-cast) + fp8_te on pre-Hopper (FP16 fallback) — 자동 식별 |
| LLM-shape 비대칭 GEMM | `benchmarks.py:750-760` `LLM_SHAPES` | gpt-oss-120B 의 8 layer presets × 5 token counts → square 가 아닌 실제 inference 모양 |
| Plot hatching | `analyze.py` `_coef_bar_*` | `///` hatched bar + `*EMU` 주석 → emulated path 시각 명시 |
| FP8 dedicated plot | `_coef_bar_fp8` (PR #55) | `--include-emulated` 와 무관하게 fp8 4 op 항상 표시 |

### 강점

- **CUDA core 와 Tensor Core 가 한 sweep 에서 동시 측정** — fp32_simt vs fp16_tc 의 J/FLOP 차이가 곧 "TC 효율 gap". 사용자 H100 측정 9.09 → 0.635 pJ/FLOP = ~14× — 정상.
- **Emulated path 가 silently 잡혀가지 않음** — fp8 elementwise 는 항상 emulated, plot 에 hatched 로 표시. fp8_te on A100 도 자동 fallback 감지 + 라벨.
- **LLM-shape 으로 실제 inference 모양 측정** — square sweep 만으로는 못 보는 skinny / fat GEMM (GQA `kv` K=2880 N=512 / `lm_head` K=2880 N=201088) 의 J/FLOP 변화 캡처.
- **GPT-OSS 120B aware default K** ([PR #50](https://github.com/ByungwooBangKU/accelwattch/pull/50)) — `2880 / 4096 / 5760` 이 default 에 들어가 square ↔ LLM-shape cross-check 가능.

### 한계

| 한계 | 의미 |
|---|---|
| **Pure compute 분리 불가** | Tensor Core 의 mma 명령어 자체 에너지를 데이터 이동 (SMEM → register) 와 분리해 측정 못 함. NVML 한계. AccelWattch 도 보통 mma 비용을 단일 항으로 다루므로 영향 적음. |
| **fp8 elementwise 는 native HW 없음** | 어떤 GPU 든 cast-compute-cast 라 "native FP8 elementwise" 는 측정 불가능 — HW 자체 부재. |
| **Native fp8_te small-K noise floor** | H100 fp8 가 너무 효율적이라 K < 3072 은 NVML noise 아래로 떨어짐. 자동 noise-floor exclusion (PR #53) 으로 회귀에서 drop. README §8.3.4 documented. |

### 평가 — ✓

GPU 의 compute path 종류는 사실상 모두 (5 matmul variant + LLM-shape + emulated 식별) 커버. 한계는 실제 HW/SW 레벨의 fundamental limit 이지 implementation gap 이 아님.

---

## Axis 3 — Memory Hierarchy 분리   **△ DRAM 까지만 깨끗, 그 안쪽은 bundled**

### 핵심 질문
*"GPU 의 register file → L1/SMEM → L2 → DRAM 계층의 에너지가 분리 측정되는가?"*

### 구현된 메커니즘

| 메커니즘 | 위치 | 어떤 layer 분리 |
|---|---|---|
| 5-bucket cache regime classifier | `benchmarks.py:380-407` `classify_cache_regime` | l2_hit_100 / 75 / 50 / 25 / 0 — working set vs L2 비율로 라벨 |
| Working-set formula | `_elementwise_working_set()` | op 별 R/W byte 수 카운트 (mul/add 3·N·bpe, gelu/softmax/ln 2·N·bpe) |
| **STREAM probes** | `stream_copy / scale / triad / read / write` | compute-light → 측정 에너지 거의 전부가 메모리 트래픽 |
| **DRAM read/write split** ([PR #30](#)) | `compute_dram_rw_split()` | `stream_read` / `stream_write` 로 read pJ/bit, write pJ/bit 분리 |
| **DRAM marginal 분석** | `compute_dram_marginal()` | `J(l2_hit_0) − J(l2_hit_100)` → SM compute + L2 baseline cancel, DRAM 단가 추출 |
| `bytes_traffic` / `pj_per_bit_traffic` | `add_traffic_metrics()` | 각 cell 의 logical byte traffic 정량 |
| Cross-check: implied vs measured | `plot_dram_rw_split()` | `(r·R + w·W)/(r+w)` implied 와 측정값 비교 (≤ 5% 오차면 quality 양호) |

### 강점

- **DRAM 단가 (pJ/bit) literature 비교 가능** — `dram_energy_marginal.png` 가 HBM2E (5.0) / HBM3 (3.9) / Horowitz (2.5) reference line 과 측정값 직접 비교.
- **Read vs Write 분리** — STREAM probe 4 종으로 R/W 단가 따로 추출 + mixed kernel 의 implied 와 cross-check.
- **Marginal subtraction 으로 SM/L2 cancel** — l2_hit_0 의 direct 값 (compute + L2 + DRAM 합) 에서 l2_hit_100 의 baseline 빼면 *순수 DRAM* 만 남음. PR #30 의 핵심 기법.

### 한계

| 한계 | 의미 |
|---|---|
| **L1 / SMEM / register file 별도 측정 없음** | NVML 이 sub-L2 traffic 보고 안 함. component A (L2-resident workload) 안에 bundled. PyTorch 가 register-resident microbench 허용 안 해서 fundamental. |
| **Cache hit rate 는 *heuristic label*, 측정값 아님** | `classify_cache_regime` 가 working_set / L2 ratio 만 봐서 분류. 실제 `l2_tex_hit_rate` 는 Nsight Compute 가 측정 가능하지만 instrumentation 이 power 측정 왜곡 → 의도적 미사용. |
| **Matmul 의 working-set classifier 는 부정확** | tile reuse 로 실제 DRAM 트래픽 ≠ logical working set. README §3.5.3 / §5.1 에 caveat 명시, marginal-DRAM plot 에서 matmul 의도적 제외. |
| **HBM PHY + 컨트롤러 + L2-DRAM bus 가 모두 "DRAM" 으로 합쳐짐** | board-level NVML 의 한계. literature 도 보통 이 묶음을 "full stack pJ/bit" 로 보고하니 호환. |

### 평가 — △

DRAM 까지의 분리는 rigorous (marginal subtraction + R/W split + literature 비교) — 그 영역은 ✓ 수준. **L2 안쪽 (L1 / SMEM / register / compute) 은 분리 못 함** 이 axis 의 한계 — 이건 NVML 측정의 fundamental limit 이고 본 suite 가 정직하게 인정 (component A bundled). AccelWattch 가 보통 L2 / DRAM 단가를 분리하면 충분하므로 실용적으로는 OK, 하지만 "L1 hit 의 에너지 효과" 같은 더 깊은 분석이 필요하면 NVML 만으로는 한계.

---

## Axis 4 — k_op 추출 방법론   **✓ 잘 됨**

### 핵심 질문
*"`E_dyn = k_op · N_op + ε` 의 k_op 가 robust 하게 회귀로 추출되는가? 노이즈 / clipping / drift 에 대해 가드가 있는가?"*

### 구현된 메커니즘

| 메커니즘 | 위치 | 역할 |
|---|---|---|
| OLS + WLS 동시 fit | `_fit_one_group()` `linear_fit()` / `linear_fit_wls()` | OLS (legacy 호환) + WLS (1/y² 가중, headline) 둘 다 보고 |
| **Bootstrap 95% CI** | `bootstrap_slope_ci()` 1000 resample | slope 의 sampling 불확실성 정량 → bar plot 의 whisker 로 표시 |
| **Per-regime fit** | `summarize_by_regime()` | (op, dtype, regime) 별 slope → cache locality 의 k_op 영향 분리 |
| **Clip-bias detection** | `clip_bias_pct` column | `(slope_dyn_wls − slope_unclipped) / slope_unclipped × 100` |
| **Noise-floor 자동 제외** | `pos_mask` (PR #53) | `dyn_energy_j ≤ 0` row 는 dyn 회귀에서 drop, `n_points_dyn_fit` / `n_dropped_clipped` 로 transparency |
| Single-point fallback | `n=1` 처리 | degenerate group 에 대해 `y/x` per-point coefficient 라도 보고 |
| Total-energy 회귀 | `slope_total` | `dyn` 과 별도로 `E_total` 회귀도 fit → P_static drift 영향 cross-check |

### 강점

- **WLS 가중 (1/y²) 이 power 측정의 noise 모델과 일치** — σ(y) ∝ y (constant relative error) 가정. log-space 에서 등가중. `_fit_one_group` (analyze.py:412)
- **Bootstrap CI 가 single-number slope 에 *불확실성* 명시** — 옛 OLS-only 구현의 false confidence 제거. 작은 K 의 fp8_te 같이 noisy 한 cell 은 CI 가 wide 해서 bar 위 whisker 로 즉시 가시.
- **Clip-bias 자동 alarm** — `dyn_energy_j_raw` 컬럼 보존 + clipped vs unclipped slope 비교 → ±2% 이내면 신뢰, 초과면 P_static drift 의심으로 user 에게 안내.
- **Noise floor 의 자동 처리** (PR #53) — H100 fp8_te K=1024..2048 같은 "전부 clipped" 변종은 slope_dyn_wls = NaN 으로 떨어져 bar plot 에서 invisible — 가짜 0 표시 안 함.

### 한계

| 한계 | 의미 |
|---|---|
| **Linear model 은 intercept = 0 가정** | 실제로는 launch overhead 가 작은 N 에선 일정 offset. WLS slope 는 N 변화량의 비율이라 큰 N 가중치 ↑ → overhead 영향 작음. 그래도 약간의 systematic bias 잔존. |
| **Matmul 의 단일 slope 이 K 변화 평균** | TC 효율이 K 에 따라 변화 (Hopper FP8 sweet spot K ≥ 8192) 인데 fit 은 모든 K 평균. README §3.5.3 acknowledged. per-K k_op 가 필요하면 additional 분석 필요. |
| **fp16 / fp8 mixed 그룹 제외** | 회귀가 `(category, op, dtype, mode, llm_preset)` 단위라 dtype 이 같은 op 의 slope 는 분리. fp8 vs fp16 직접 비교는 plot 단계에서 함. |

### 평가 — ✓

power-model 계수 추출의 statistical rigor 는 이 분야에서 흔히 보는 것보다 강력 (bootstrap CI, clip-bias, noise floor). methodology 자체는 fundamentally sound, 한계도 정직하게 노출.

---

## Axis 5 — Thermal & Leakage   **✓ 잘 됨**

### 핵심 질문
*"온도 의존 leakage current 가 측정 + 정량되어 AccelWattch 의 thermal 항에 들어갈 데이터가 나오는가?"*

### 구현된 메커니즘

| 메커니즘 | 위치 | 역할 |
|---|---|---|
| SoC envelope 3-phase | `soc_power_bench.py` (또는 `gpu_power_bench --cases soc`) | static / max / leakage 5-cycle |
| **Hot-leakage** | `phase_leakage()` + 1 s `--leak-window-s` post-stress | 5 사이클 평균 → noise 평균화, `leakage_minus_static_w` = 온도 의존 leakage Δ |
| Per-cell thermal 컨텍스트 | `peak_temp_c` / `temp_rise_c` / `start_temp_c` columns | sweep cell 마다 전후 온도 기록 |
| Cooldown between cells | `wait_for_cooldown()` | thermal carry-over 감소, target_c=45 default |
| Decay zoom plot ([PR #58 자매](#)) | `_soc_leakage_enlarged.png` | 첫 3 s × 50–150 W 줌인 + cycle 별 온도 dashed overlay |

### 강점

- **5-cycle averaging 으로 noise 감소** — 한 cycle 의 hot power 가 NVML noise 에 흔들려도 5 평균으로 ±1 W 안정.
- **Decay 곡선 + 온도 overlay** — `_soc_leakage_enlarged.png` 의 우측 y-axis dashed temp 와 좌측 power 동시 plot → "이 X W leakage 가 Y °C 에서 측정됨" 시각적 직접 매핑.
- **Stress→stop transition 이 정확히 caputred** — `_run_gemm_for()` 가 phase 끝에 `torch.cuda.synchronize()` → 첫 sample 부터 hot-idle 시작.
- **Build-defer pattern** ([PR #54](https://github.com/ByungwooBangKU/accelwattch/pull/54)) — `build_matmul()` 의 5× warmup 이 `phase_static` 후로 옮겨져 cold idle 측정 보존.

### 한계

| 한계 | 의미 |
|---|---|
| **Board-level leakage** | 칩 leakage + HBM idle leakage + VRM 합. 칩 단독 분리 못 함 (board-level NVML 한계). AccelWattch 도 보통 board total 사용하므로 호환. |
| **Leakage(T) curve 전체가 아닌 단일 hot point** | T_hot vs T_cold 두 점만 측정 → linear approximation. exponential 한 leakage(T) 모델 fitting 하려면 cooldown 중간 point 도 잡아야 함. 현재 `_leakage_enlarged.png` 의 decay trace 가 이 데이터를 *시각화* 하지만 fit 은 안 함. |
| **Thermal coupling between phases 미통제** | 매 phase 마다 cooldown 하지만 완벽하지 않음. 5 cycle 안에서 silicon 이 약간 누적 가열 → 첫 cycle vs 5번째 cycle hot 온도 다를 수 있음. 현재 plot 에서 cycle 간 온도 spread 보임. |
| **Leakage 가 sweep 의 dyn_energy 에 자동 보정 안 됨** | SoC envelope 결과가 별도 sidecar CSV. sweep 의 per-cell `dyn_energy_j` 계산은 cold P_static 만 사용. thermal-corrected dyn 은 사용자가 후처리. |

### 평가 — ✓

leakage 측정 자체는 깨끗 — "hot idle minus cold idle" 수식 그대로, 5-cycle 평균으로 noise 처리. AccelWattch 의 thermal 항에 그대로 입력 가능. 다만 sweep 데이터와 자동 통합은 안 됨 (P1 권장사항으로 §8 에서 다룰 예정).

---

## Axis 6 — MECE Decomposition   **✓ 잘 됨 (한계 명시)**

### 핵심 질문
*"한 측정값을 component (compute / memory / cast 등) 로 *수학적으로 깨끗하게* 분해할 수 있는가? overlap / missing 없는가?"*

### 구현된 메커니즘

| 메커니즘 | 위치 | 역할 |
|---|---|---|
| **3-component 항등식** | `plot_energy_decomposition()` ([PR #58](https://github.com/ByungwooBangKU/accelwattch/pull/58)) | A + B + C ≡ J(op, dtype, l2_hit_0) — algebraic |
| Component A "L2-resident workload" | `J(op, fp16, l2_hit_100)` | compute + L2 + register + launch overhead (bundled) |
| Component B "FP8 cast overhead" | `J(op, fp8, l2_hit_100) − J(op, fp16, l2_hit_100)` | cast-compute-cast 추가 비용 |
| Component C "DRAM round-trip" | `J(op, dtype, l2_hit_0) − J(op, dtype, l2_hit_100)` | HBM streaming marginal |
| Stacked bar visualization | `_03_energy_decomposition_mece.png` | A/B/C 비율 + 총 pJ/elem 표기 |
| **Caveat box** in plot | `fig.text()` at bottom | "A 는 NVML 측정으로 더 분리 못 해 bundled 유지. 추정으로 더 쪼개면 MECE 깨짐" 명시 |

### 강점

- **순수 algebraic identity** — A, B, C 정의가 측정값 *차이* 라 substitution 으로 합 = 측정 total 검증 가능. overlap 없고 missing 없음.
- **Stacked bar 가 분해 결과를 한 눈에** — softmax_fp8 의 1940 pJ/elem 이 "거의 100% A (resident workload)" 인지, "절반 B (cast)" 인지 즉시 판독.
- **명시적 한계 표시** — plot 밑 caveat box 에 "A 는 더 분리 못 함" 명문화 → 사용자가 추정값을 측정값처럼 오해 방지.
- **Window-ms 권장값 동봉** — README §10 + CLI help 에 `--window-ms 6000` 권장 (분해는 작은 cell 끼리 빼는 연산이라 noise floor 에 가장 민감).

### 한계

| 한계 | 의미 |
|---|---|
| **Component A 안의 compute vs L2 vs launch 분리 불가** | NVML 만으로는 fundamental limit (Axis 3 한계와 같음). 추정 (FLOP × J/FLOP_reference) 가능하나 *MECE 깨지므로* 의도적 미실시. |
| **Elementwise only — matmul 미지원** | 현재 decomposition 은 `category == "elementwise"` 만. matmul 의 analogous 분해 (A: register-tile resident, B: fp8 scaling overhead, C: DRAM) 는 미구현. |
| **fp8 baseline 없으면 B 항 0** | `mul_fp8` 만 측정하고 `mul_fp16` 측정 없으면 cast overhead 분리 못 함 → 그 (op, dtype) bar 자체 skip. cell coverage 의존. |
| **Single-regime decomposition** | 현재 `l2_hit_0` 에서만 분해. 다른 regime (l2_hit_100 / l2_hit_50) 에서도 같은 framework 적용 가능하나 plot 에 1 점만. |

### 평가 — ✓

분해 자체는 algebraic 으로 MECE — 수학적 보장. 한계 (A bundling, elementwise-only) 가 plot/문서에서 정직하게 명시. AccelWattch 의 component-별 비용 입력에 가장 직접적 활용 가능.

---

(continued — §7 Gap analysis, §8 Recommendations)
