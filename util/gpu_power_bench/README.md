# GPU Power-per-Operation Benchmark (A100 / H100)

FP8 / FP16 × {MUL, ADD, Softmax, GeLU, LayerNorm} = **10 benchmarks**. 각 벤치마크를
여러 load 크기로 sweep 하면서 NVML power 를 적분해 **Joule per operation** 을
산출한다. 정적(idle) / 동적(workload) 전력을 분리하고, 실험 간 온도 cool-down
및 안정성 로그 (온도/클럭/스로틀) 를 남긴다.

AccelWattch A100/H100 모델 검증을 위한 microbenchmark — GPU 가 같은 연산을
precision 별로 수행할 때 watt-hour 가 어떻게 변하는지 지상진(ground truth)
데이터를 얻는 용도.

## 무엇이 측정되나

| 항목 | 설명 |
|---|---|
| `total_energy_j` | 측정 구간 NVML `power.draw` 를 사다리꼴 적분한 Joule |
| `static_energy_j` | `P_static × wall_s` — 같은 구간에 idle 전력이 소모했을 에너지 |
| `dyn_energy_j` | `total − static` — **workload 가 추가로 쓴 dynamic 에너지** |
| `j_per_element_dyn` | `dyn_energy_j / (iters × N)` — **연산 하나당 dynamic Joule** |
| `j_per_flop_dyn` | 추정 FLOP 기준 (MUL=1, Softmax≈5, GeLU≈8, LayerNorm≈8 per elem) |
| `avg_power_w` / `dyn_power_w` | 구간 평균 전력 / (그-P_static) |
| `avg_temp_c` / `peak_temp_c` | NVML 온도 폴링 평균/피크 — 안정성 검증 |
| `sm_clk_mhz` / `mem_clk_mhz` | 구간 말미 클럭 — throttle 여부 판단 |

load 를 여러 단계로 변화시키면 `dyn_energy_j` ∝ `total_elements` 가
**선형**이어야 하고, `j_per_element_dyn` 은 **flat** 해야 한다.
`analyze.py` 가 각 (op, dtype) 에 대해 **R²** 를 계산해서 선형성을 수치화.

## A100 vs H100 instruction set 차이 처리

| dtype | A100 (sm_80) | H100 (sm_90) |
|---|---|---|
| FP16 | native Tensor Core | native Tensor Core |
| FP8  | **emulated** (cast→fp16→cast 경로로 돌아감) | native Tensor Core (Transformer Engine) |

- 벤치마크는 **동일 코드** 로 양쪽 GPU 에서 돌고, A100 에서는 FP8 결과 row 에
  `notes="fp8 emulated (no native FP8 tensor cores on this GPU)"` 가 찍힌다.
- 결과 해석: A100 의 FP8 측정은 "native FP8 대비 상한" 이 아니라
  "A100 이 FP8 데이터를 처리할 때 실제로 쓰는 에너지" 이므로 여전히 유효.
  H100 FP8 수치와 비교하면 Tensor Core FP8 의 에너지 효율 이득이 드러난다.

## 설치

```bash
pip install -r requirements.txt
# torch 는 CUDA 버전에 맞게 별도 설치 (선택)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

사전 점검:

```bash
python3 preflight.py
```

확인 항목:
- `nvidia-smi`, pynvml, torch 임포트
- CUDA device 가 보이는지 / compute capability
- `torch.float8_e4m3fn` / `float8_e5m2` 존재 (torch ≥ 2.1)
- NVML `power.draw` 읽기 가능 (Joule 적분 필수)
- Persistence mode (권장: `sudo nvidia-smi -pm 1`)

FAIL 이 뜨면 그 항목을 먼저 해결. A100 에서 "fp8 emulated" 경고는 정상.

## 실행

```bash
cd util/gpu_power_bench
./run_bench.sh                          # 전체 sweep (기본 load 6단계)
./run_bench.sh --quick                  # 빠른 smoke 테스트 (load 3단계)
./run_bench.sh --ops mul add --dtypes fp16   # 일부만
./run_bench.sh --tag a100_run1          # 출력 파일에 suffix

# 실험별 분리가 필요하면 cool-down 임계치 상향
./run_bench.sh --cooldown-c 40 --cooldown-timeout 180
./run_bench.sh --no-cooldown            # cool-down 생략 (빠름, 덜 안정적)
```

주요 옵션:

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--device N` | 0 | CUDA 디바이스 인덱스 |
| `--ops ...` | 전체 5 | `mul add softmax gelu layernorm` 중 선택 |
| `--dtypes ...` | 전체 2 | `fp16 fp8` 중 선택 |
| `--loads ...` | 256K..256M 6단계 | tensor element 수 직접 지정 |
| `--quick` | — | `loads = [1M, 4M, 16M]` 로 축소 |
| `--window-ms` | 1500 | 각 cell 측정 길이 (↑ NVML 노이즈 ↓) |
| `--static-seconds` | 8 | 기준 idle 전력 측정 시간 |
| `--cooldown-c` | 50 | 실험 간 도달해야 할 °C (`-1` 이면 비활성) |
| `--cooldown-timeout` | 120 | cool-down 최대 대기 (초) |
| `--tag` | — | 출력 파일명에 suffix |
| `--poll-hz` | 100 | NVML 폴링 주파수 |

## 출력

```
reports/
  gpu_power_bench_a100_80gb_20260420_142301.csv           # per-cell 요약
  gpu_power_bench_a100_80gb_20260420_142301_samples.csv   # 전체 NVML 타임라인
  gpu_power_bench_a100_80gb_20260420_142301_summary.csv   # (analyze 실행 후)
  gpu_power_bench_a100_80gb_20260420_142301_linearity.png # (analyze 실행 후)
  gpu_power_bench_a100_80gb_20260420_142301_timeline.png  # (analyze 실행 후)
```

주요 CSV 컬럼은 상단 표 참조. `samples.csv` 는 raw NVML — time, power, temp,
SM/MEM clock, util, phase label — 이라서 Nsight 없이도 timeline 재구성 가능.

## 분석 / 플롯

```bash
python3 analyze.py reports/gpu_power_bench_a100_80gb_20260420_142301.csv
```

생성물:

1. **`_summary.csv`** — (op, dtype) 별:
   - `slope_J_per_elem_dyn` : linear fit 기울기 = **실측 J/elem**
   - `R2_dyn_vs_N` : 선형성 R² (≥ 0.99 이면 측정이 잘 된 것)
   - `mean_dyn_power_w`, `mean_temp_c`, `peak_temp_c`
2. **`_linearity.png`** — 3 × 5 grid: op 별로
   - row 1: E_dyn vs load (log-log, 기울기 1 직선이 이상적)
   - row 2: wall time vs load
   - row 3: J/element (dyn) vs load (flat 해야 함)
3. **`_timeline.png`** — 전체 런의 power / temp 타임라인.
   각 cell 구간이 살짝 shading 되어 실험 경계 식별 가능.

## 안정성 검증 체크포인트

실험 결과가 signal 이려면 다음이 유지되어야 한다:

1. **`R2_dyn_vs_N` ≥ 0.99** — energy 가 load 에 선형이면 `iters` 가 충분.
   낮으면 `--window-ms` 를 늘리거나 `--poll-hz` 확인.
2. **`peak_temp_c` 가 cool-down 임계(+15°C) 이내** — thermal throttle 없음.
   초과 시 `--cooldown-c` 내리거나 `--cooldown-timeout` 증가.
3. **`sm_clk_mhz` ≈ max SM clock** — `nvidia-smi -q -d CLOCK` 의 max 와 비교.
   하락했으면 power cap 또는 thermal throttle. `nvidia-smi -pl <W>` 로 상향.
4. **`j_per_element_dyn` 의 변동계수 < 5%** — load 별로 거의 같아야 정상.
   값이 log-log 로 기울어지면 메모리 BW / launch overhead 에 dominated.
5. **Persistence mode = ON** — `sudo nvidia-smi -pm 1`. 드라이버 재초기화
   레이턴시가 짧은 kernel 에서 에너지 적분을 왜곡한다.

## 정적 / 동적 전력 분리

```
P_total(t) = P_static + P_dynamic(t)
E_total = ∫ P_total dt = P_static · T + ∫ P_dynamic dt
         = E_static + E_dynamic
```

`P_static` 은 실험 시작 시 `measure_static_power()` 로 idle 구간 평균을 뽑아
고정 상수로 사용. 각 cell 의 `dyn_energy_j = total - P_static × wall_s`.

**중요**: CPU-only idle 과 GPU idle 은 다르다. GPU 가 clock-gated 상태
(SM 저주파수) 일 수도 있고 wake-up 에 overhead 가 있을 수도 있다. 따라서:

- `measure_static_power` 는 CUDA 컨텍스트가 살아있는 상태에서 호출 → wake-up
  오버헤드를 baseline 안으로 흡수.
- cool-down 후 다시 baseline 을 찍고 싶으면 `--static-seconds` 는 유지한 채
  cell 마다 재측정하도록 코드를 수정 (현재는 시작 1회).

## 파일

| 파일 | 역할 |
|---|---|
| `preflight.py` | 의존성 / 드라이버 / FP8 support / NVML power 읽기 가능성 점검 |
| `power_monitor.py` | NVML 폴러 + 에너지 적분 + static baseline + cool-down |
| `benchmarks.py` | 10 benchmark factory (op × dtype), FP8 emulation 포함 |
| `gpu_power_bench.py` | 메인 드라이버 (sweep → CSV) |
| `analyze.py` | 선형성 plot + summary CSV |
| `run_bench.sh` | 편의 launcher (persistence mode, deps 체크) |
| `requirements.txt` | Python 의존성 |

## 알려진 한계

- **NVML `power.draw` 는 ~20 Hz 내부 갱신** 이라 100 Hz 폴링은 중복 샘플 존재.
  평균/적분 관점에서는 정확하지만 peak power 가 아닌 **average** 를 볼 것.
  더 높은 정밀도가 필요하면 NVML 대신 `nvidia-smi dmon` 혹은 HMC 외부 측정기.
- **FP8 elementwise 는 에뮬레이션**: torch 의 float8 dtype 은 FP8 tensor core
  GEMM 을 돌릴 때 의미가 있고, MUL/ADD 같은 eltwise 는 내부적으로 fp16 으로
  promote 되어 실행된다. 본 벤치는 이 비용도 같이 재는 것을 의도함.
- **H100 Transformer Engine 연동은 미구현**: 본격 FP8 GEMM 을 재고 싶으면
  `transformer_engine.pytorch` 의 `fp8_autocast` 로 래핑한 matmul 벤치를 추가.
  (현재 스펙은 eltwise + 정규화 연산 기준.)
- **Softmax/LayerNorm FLOP 추정은 순수 element 기준**: 실제 커널은 reduction
  패턴 때문에 FLOP 이 약간 많거나 적을 수 있음. 주 지표인 `j_per_element`
  는 FLOP 가정과 무관.
