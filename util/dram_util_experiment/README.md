# DRAM Bandwidth and pJ/bit Experiment Guide

이 디렉터리는 GPU DRAM read/write traffic을 제어해서 bandwidth, NVML power trace,
board-level marginal pJ/bit을 측정하는 실험 도구 모음이다. 현재 문서는
`dram_pjbit_cupy.py` 기반 pJ/bit 실험을 중심으로 정리한다. 기존 read-utilization 및
Nsight profiling 도구는 12장 부록에 요약한다.

## 0. 요약

1. 이 실험은 **DRAM rail-only pJ/bit**를 직접 측정하지 않는다. NVML power에서 idle 또는 phase-local baseline을 빼서 **GPU/board marginal dynamic pJ/bit**를 추정한다.
2. 최종 보고값은 `100%-0%` 한 점보다 **50/75/100% multi-point slope**를 우선한다.
3. RTX 3090 검증 결과, confidence 높은 값은 대략 read `30 pJ/bit`, write `31-32 pJ/bit`이다.
4. `100%-0%`는 read/write 모두 약 `44-45 pJ/bit`로 더 크다. 이 값은 clock/P-state, controller, L2/NoC/SM path 활성화 비용이 포함된 sanity check로 본다.
5. CuPy 경로는 RTX 3090, A100, H100에서 별도 arch 설정 없이 동작한다. Native CUDA 경로만 A100 `SM=80`, H100 `SM=90`처럼 직접 맞춘다.
6. 기본 buffer는 `max(1 GiB, 64 x L2)`다. H100/A100에서는 8 GiB sensitivity run도 권장한다.
7. 큰 buffer에서는 `--window-ms`가 중요하다. 8 GiB/H100 계열 보고용은 `--window-ms 100` 또는 `200`을 권장한다.
8. H100에서는 `NVML_FI_DEV_POWER_INSTANT`/`AVERAGE` field API를 같이 기록해서 cross-validation할 수 있다.

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `dram_pjbit_cupy.py` | read/write pJ/bit 주 실험 스크립트 |
| `run_pjbit_cupy.sh` | pJ/bit 실험 launcher. 필요한 Python 환경 자동 탐색 |
| `summarize_pjbit_repeats.py` | 여러 `*_analysis.csv` 반복 run 평균/표준편차 요약 |
| `dram_util_cupy.py` | legacy read-utilization 계단 실험 |
| `run_cupy.sh` | legacy CuPy read-utilization launcher |
| `run_nsys_cupy.sh` | CuPy 스크립트 Nsight Systems profiling |
| `dram_util.cu` | Native CUDA read-utilization 커널 |
| `Makefile` | Native CUDA 빌드. 기본 `SM=86` |
| `run_nsys.sh` | Native CUDA + `nsys` 실행/분석 |
| `run_nsys_a100.sh` | A100 80GB native preset. `SM=80`, 8 GiB buffer |
| `analyze.py` | `nsys` sqlite에서 phase별 DRAM read 지표 분석 |
| `reports/` | 실험 산출물. git ignore 대상 |

## 2. 측정 대상과 한계

### 2.1 측정 대상

`dram_pjbit_cupy.py`는 다음 값을 측정한다.

1. read/write streaming bandwidth.
2. 같은 시간 구간의 NVML GPU/board power.
3. idle 또는 phase-local baseline을 뺀 dynamic power.
4. transferred bit당 marginal energy.
5. `100%-0%`, all-pair delta, multi-point slope pJ/bit.

### 2.2 측정 한계

일반 NVIDIA GPU에서 NVML은 DRAM rail 전용 power sensor를 제공하지 않는다. 따라서 이 실험값은 다음을 모두 포함할 수 있다.

1. DRAM device/interface power.
2. memory controller power.
3. L2/NoC/SM load/store path power.
4. clock, P-state, boost, thermal, power-management transition.
5. board-level associated circuitry.

즉, 결과는 AccelWattch/DRAM power model 보정에 쓸 수 있는 **board-level calibration anchor**로 해석해야 한다. 공개 GDDR/HBM device-level pJ/bit와 직접 같은 물리량이 아니다.

## 3. 실험 원리

### 3.1 전체 흐름

```mermaid
flowchart TD
    A[GPU 상태 확인: nvidia-smi, process, power limit, clocks] --> B[CUDA device property 조회]
    B --> C[Working set 결정: max(1 GiB, 64 x L2)]
    C --> D[read/write kernel JIT]
    D --> E[1 pass calibration: effective peak GB/s]
    E --> F[idle baseline power 측정]
    F --> G[target phases: 0/25/50/75/100]
    G --> H[phase별 bytes/BW/power/clocks/temp/P-state 저장]
    H --> I[100-0, pairwise, 50/75/100 slope 계산]
    I --> J[반복 run 평균/표준편차/R2/residual 확인]
```

### 3.2 계산식

```text
E_dyn      = max(0, E_total - P_idle x wall_time)
BW_GBps    = transferred_bytes / wall_time / 1e9
pJ_per_bit = E_dyn / (transferred_bytes x 8) x 1e12
           = dynamic_power_W x 1000 / (8 x BW_GBps)
```

`target=0`은 kernel launch 없이 같은 phase 길이 동안 power만 기록한다. CSV에는 `bandwidth_gbps=0`, `bytes_transferred=0`, `pj_per_bit=nan`으로 남는다.

### 3.3 read/write 커널

1. `read`: 큰 `float4` buffer를 `__ldcg`로 streaming load한다. L1은 우회하고 L2/DRAM 경로에 pressure를 준다.
2. `write`: 같은 크기의 `float4` buffer 전체를 streaming store한다. byte 수는 user-data write byte 기준이다.
3. working set은 L2보다 훨씬 크게 잡기 때문에 DRAM-dominant access가 되도록 설계한다.

## 4. GPU 지원과 기본값

### 4.1 자동 지원 범위

CuPy 버전은 `cudaGetDeviceProperties`로 GPU를 식별하므로 RTX 3090, A100, H100 모두 별도 설정 없이 동작한다.

| GPU | L2 예시 | 기본 buffer |
|---|---:|---:|
| RTX 3090 | 6 MiB | 1 GiB |
| A100 | 40 MiB | 2.5 GiB |
| H100 | 50 MiB | 3.125 GiB |

Native CUDA 버전은 직접 arch를 맞춰야 한다.

| GPU | Native CUDA arch |
|---|---|
| RTX 3090 | `SM=86` |
| A100 | `SM=80` |
| H100 | `SM=90` |

### 4.2 왜 `64 x L2`인가?

`64 x L2`는 물리 상수가 아니라 보수적 휴리스틱이다. 목적은 한 pass 안의 reuse distance를 L2보다 훨씬 크게 만들어 다음 pass에서 같은 cache line을 다시 읽기 전에 대부분 eviction되도록 하는 것이다.

1. RTX 3090은 L2 6 MiB라 `64 x L2 = 384 MiB`지만, 기본 하한 1 GiB 때문에 약 `170 x L2`를 쓴다.
2. A100은 L2 40 MiB라 기본 buffer가 약 2.5 GiB다.
3. H100은 L2 50 MiB라 기본 buffer가 약 3.125 GiB다.
4. H100/A100은 page/channel/bank interleaving과 controller 정책 확인을 위해 8 GiB sensitivity run을 추가하는 것이 좋다.

### 4.3 L1/L2 hit-rate에 대한 기대

설계상 L1/L2 hit-rate는 거의 0에 가까워야 한다.

1. read는 `__ldcg`로 L1을 우회한다.
2. working set이 L2보다 훨씬 크다.
3. RTX 3090 sweep에서 1/2/4/8 GiB로 buffer를 키워도 slope 결과가 크게 바뀌지 않았다.

단, 이 환경에서는 `ncu`/`nsys`가 없어 hit-rate counter를 직접 확인하지 못했다. H100/A100 최종 실험에서는 Nsight Compute로 `l1tex`/`lts` hit-rate를 cross-check하는 것이 좋다.

## 5. 환경 준비

### 5.1 Python 환경

CUDA toolkit이나 `nvcc`는 필요 없다. 드라이버와 CUDA runtime/NVRTC wheel이 있으면 된다.

```bash
cd util/dram_util_experiment

python3 -m venv .venv-pjbit
source .venv-pjbit/bin/activate

pip install -U pip
pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 pynvml nvtx matplotlib numpy

python -c "import cupy, nvtx, pynvml, matplotlib; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
nvidia-smi
```

`run_pjbit_cupy.sh`는 `/home/bang001/miniforge3/envs/ssc21env/bin/python`과 `python3` 중 필요한 패키지를 import할 수 있는 Python을 자동으로 고른다.

### 5.2 실험 전 체크리스트

1. `nvidia-smi`로 다른 compute process가 없는지 확인한다.
2. datacenter GPU는 보통 ECC on 상태로 둔다.
3. 가능하면 persistence mode를 켠다: `sudo nvidia-smi -pm 1`.
4. power limit과 clocks가 의도한 상태인지 확인한다.
5. thermal throttling이 없도록 충분히 식힌 뒤 실행한다.
6. phase는 보고용 기준 최소 20초를 권장한다.

## 6. 실행 방법

### 6.1 빠른 smoke test

```bash
cd util/dram_util_experiment

./run_pjbit_cupy.sh \
  --modes read write \
  --targets 100 \
  --phase-seconds 5 \
  --idle-seconds 5 \
  --tag smoke
```

### 6.2 기본 보고용 run

```bash
./run_pjbit_cupy.sh \
  --modes read write \
  --targets 0 25 50 75 100 \
  --phase-seconds 20 \
  --idle-seconds 15 \
  --poll-hz 100 \
  --window-ms 200 \
  --tag baseline_w200_rep1
```

### 6.3 3회 반복 run

```bash
./run_pjbit_cupy.sh --modes read write --targets 0 25 50 75 100 \
  --phase-seconds 20 --idle-seconds 15 --poll-hz 100 --window-ms 200 \
  --tag baseline_w200_rep1

./run_pjbit_cupy.sh --modes read write --targets 0 25 50 75 100 \
  --phase-seconds 20 --idle-seconds 15 --poll-hz 100 --window-ms 200 \
  --tag baseline_w200_rep2

./run_pjbit_cupy.sh --modes read write --targets 0 25 50 75 100 \
  --phase-seconds 20 --idle-seconds 15 --poll-hz 100 --window-ms 200 \
  --tag baseline_w200_rep3

python3 summarize_pjbit_repeats.py \
  "reports/*baseline_w200_rep*_analysis.csv" \
  --out reports/baseline_w200_repeat_summary.csv
```

### 6.4 A100 run

A100도 CuPy 경로에서는 별도 설정이 필요 없다. 기본 buffer는 대략 2.5 GiB다.

```bash
./run_pjbit_cupy.sh \
  --modes read write \
  --targets 0 25 50 75 100 \
  --phase-seconds 20 \
  --idle-seconds 15 \
  --poll-hz 100 \
  --window-ms 100 \
  --tag a100_w100_rep1
```

더 방어적인 보고용 조건:

```bash
./run_pjbit_cupy.sh \
  --modes read write \
  --targets 0 25 50 75 100 \
  --buf-bytes 8589934592 \
  --phase-seconds 20 \
  --idle-seconds 15 \
  --poll-hz 100 \
  --window-ms 200 \
  --tag a100_8gib_w200_rep1
```

A100 80GB에서 raw HBM peak 2039 GB/s와 calibration peak 1700-1800 GB/s 수준을 직접 비교하면 안 된다. 스크립트는 raw bus peak가 아니라 effective user-data bandwidth를 기준으로 target을 만든다.

### 6.5 H100 run

H100은 기본 buffer가 약 3.125 GiB다. 다중 GPU 시스템에서는 `--device N`을 명시한다.

```bash
./run_pjbit_cupy.sh \
  --device 0 \
  --modes read write \
  --targets 0 25 50 75 100 \
  --phase-seconds 20 \
  --idle-seconds 15 \
  --poll-hz 100 \
  --window-ms 200 \
  --tag h100_w200_rep1
```

8 GiB sensitivity run:

```bash
./run_pjbit_cupy.sh \
  --device 0 \
  --modes read write \
  --targets 0 25 50 75 100 \
  --buf-bytes 8589934592 \
  --phase-seconds 20 \
  --idle-seconds 15 \
  --poll-hz 100 \
  --window-ms 200 \
  --tag h100_8gib_w200_rep1
```

## 7. 출력 파일

| 파일 | 내용 |
|---|---|
| `*_summary.csv` | phase별 BW, power, energy, pJ/bit, clocks, temp, P-state |
| `*_trace.csv` | time-series power/util/clocks/temp/phase |
| `*_analysis.csv` | `100%-0%`, all-pair delta, slope pJ/bit, R2, residual |
| `*_metadata.json` | GPU, driver, CUDA, L2, buffer, calibration, power limit |
| `*.png` | power timeline 및 phase별 pJ/bit |
| `*_analysis.png` | power-vs-bandwidth fit, residual, estimator 비교 |

`trace.csv`는 NVML field API가 지원되면 아래 컬럼도 포함한다.

| 컬럼 | 의미 |
|---|---|
| `power_w` | 기존 `nvmlDeviceGetPowerUsage()` 값. Ampere 이상에서는 1초 평균 성격 |
| `power_instant_w` | `NVML_FI_DEV_POWER_INSTANT` 값 |
| `power_average_w` | `NVML_FI_DEV_POWER_AVERAGE` 값 |
| `power_instant_status` | instant field `nvmlReturn`; `0`이면 성공 |
| `power_average_status` | average field `nvmlReturn`; `0`이면 성공 |

## 8. 결과 해석

### 8.1 pre-idle baseline pJ/bit

`summary.csv`의 `dynamic_power_w`, `dynamic_energy_j`, `pj_per_bit`는 pre-idle baseline 기준이다.

장점:

1. 모든 phase가 같은 baseline을 쓴다.
2. 단일 phase의 진단값으로 보기 쉽다.

단점:

1. pre-idle power가 이후 0% phase power와 다르면 저부하 구간이 왜곡될 수 있다.
2. 최종 보고값으로는 `analysis.csv`의 slope 값을 우선한다.

### 8.2 `100%-0%` phase-local delta

```text
delta_power_W = avg_power_w(target=100) - avg_power_w(target=0)
pJ_per_bit_100_minus_0 = delta_power_W x 1000 / (8 x bandwidth_gbps(target=100))
```

이 방식은 phase-local 0% baseline을 쓰므로 pre-idle drift에는 덜 민감하다. 하지만 100%에서 추가로 켜지는 clock, memory controller state, L2/SM path power가 포함되므로 DRAM rail-only 값이 아니다.

### 8.3 all-pair delta

```text
delta_power_W = avg_power_w(target=high) - avg_power_w(target=low)
delta_bw_GBps = bandwidth_gbps(target=high) - bandwidth_gbps(target=low)
pJ_per_bit_pair = delta_power_W x 1000 / (8 x delta_bw_GBps)
```

해석:

1. `75-50`, `100-75`, `100-50`이 slope와 비슷하면 고부하 marginal cost가 안정적이다.
2. `50-25`, `50-0`, `75-0`, `100-0`이 더 크면 저부하 clock/P-state/controller wake-up 비용이 섞인 것이다.
3. 모든 pair가 같은 값이어야 하는 것은 아니다. pair 차이가 크면 단일 DRAM rail-only pJ/bit로 해석하면 안 된다.

### 8.4 권장 보고값: multi-point slope

```text
avg_power_w = intercept + slope x bandwidth_gbps
pJ_per_bit_slope = slope_W_per_GBps x 1000 / 8
```

권장 기준:

1. 기본은 50/75/100% slope를 사용한다.
2. `r2`가 1에 가깝고 `max_abs_residual_w`가 작아야 한다.
3. 25% point는 launch overhead와 power-state transition 영향이 크면 제외한다.
4. 최종 보고는 최소 3회 반복 평균/표준편차로 한다.

## 9. `--window-ms` warning 해석

### 9.1 warning의 의미

큰 buffer에서는 1 pass 시간이 duty-cycle window와 비슷해질 수 있다. 그러면 25/50/75% target을 정수 pass 개수로 반올림해야 하므로 실제 bandwidth가 target label과 달라진다.

이 warning은 "계산 전체가 망가졌다"는 뜻이 아니다. **requested target이 정확하지 않을 수 있다**는 뜻이다.

| 목적 | warning 영향 | 권장 |
|---|---|---|
| `--targets 100` 단독 peak-load 측정 | 작음 | 실제 `bandwidth_gbps`가 calibration peak에 가까운지 확인 |
| 25/50/75/100 계단 모양 확인 | 있음 | A100/H100은 최소 `--window-ms 50`, 보고용은 `100` 또는 `200` |
| target별 pJ/bit 비교 | 있음 | `target_pct` label 대신 실제 `bandwidth_gbps` 기준 분석 |
| slope 기반 marginal pJ/bit | 중간 | 50/75/100 실제 BW 분리와 R2/residual 확인 |
| DRAM rail-only pJ/bit | 별도 문제 | NVML로는 불가 |

`target=100`은 sleep 없이 kernel을 연속 실행하므로 25/50/75% duty target과 다르게 취급한다. 최신 스크립트는 100%를 duty quantization warning 대상에서 제외한다.

### 9.2 권장 window

1. RTX 3090 1 GiB: `20 ms`도 동작하지만 보고용은 `100-200 ms`가 더 안전하다.
2. RTX 3090 8 GiB: `20 ms`는 invalid, `200 ms` 이상 권장.
3. A100/H100 기본 buffer: 최소 `50 ms`, 보고용 `100-200 ms` 권장.
4. A100/H100 8 GiB: `200 ms` 권장.

## 10. H100 instant power cross-validation

### 10.1 nvidia-smi 확인

```bash
nvidia-smi --query-gpu=index,name,power.draw,power.draw.instant,power.draw.average \
  --format=csv

nvidia-smi --query-gpu=index,name,module.power.draw.instant,module.power.draw.average \
  --format=csv
```

NVIDIA 문서 기준:

1. `power.draw`: Ampere 이상에서 1초 평균 성격.
2. `power.draw.instant`: 마지막 instant board power reading.
3. Hopper datacenter 제품은 module power reading도 지원할 수 있다.

### 10.2 Python/NVML 확인

```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
values = pynvml.nvmlDeviceGetFieldValues(handle, [
    pynvml.NVML_FI_DEV_POWER_INSTANT,
    pynvml.NVML_FI_DEV_POWER_AVERAGE,
])

for value in values:
    if value.nvmlReturn == pynvml.NVML_SUCCESS:
        print(value.fieldId, value.value.uiVal / 1000.0, "W")
```

### 10.3 이 실험에서의 사용법

`dram_pjbit_cupy.py`는 field API를 best-effort로 같이 polling한다. 기본 pJ/bit 계산은 기존 `power_w` 기반으로 유지하고, H100에서는 `avg_power_instant_w`와 `avg_power_average_w`가 같은 phase ordering과 유사한 delta를 보이는지 cross-validation한다.

참고:

1. NVIDIA `nvidia-smi` 문서: <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
2. NVML field query 문서: <https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueQueries.html>

## 11. RTX 3090 검증 결과

### 11.1 1 GiB 3회 반복

조건:

1. GPU/driver: RTX 3090, driver 591.86.
2. L2/buffer: 6 MiB L2, 1 GiB working set.
3. power limit: 370 W.
4. command: `--modes read write --targets 0 25 50 75 100 --phase-seconds 20 --idle-seconds 15`.
5. calibration peak: read 891-893 GB/s, write 847-856 GB/s effective user-data BW.

| mode | method | points | runs | pJ/bit mean | pJ/bit std | R2 mean | mean max residual |
|---|---|---:|---:|---:|---:|---:|---:|
| read | 100%-0% avg power | 0,100 | 3 | 44.368 | 0.807 | - | - |
| read | 50/75/100 slope | 50,75,100 | 3 | 30.487 | 0.340 | 0.999431 | 1.386 W |
| write | 100%-0% avg power | 0,100 | 3 | 43.203 | 1.068 | - | - |
| write | 50/75/100 slope | 50,75,100 | 3 | 30.690 | 0.855 | 0.999906 | 0.439 W |

### 11.2 Pairwise delta

| pair | read mean pJ/bit | write mean pJ/bit | 해석 |
|---|---:|---:|---|
| 50-25 | 51.830 | 55.261 | 저부하 전이/clock state 영향 큼 |
| 50-0 | 58.098 | 54.258 | 0% baseline과 50% state 차이 큼 |
| 75-25 | 41.338 | 42.720 | 25% 포함으로 slope보다 큼 |
| 75-50 | 30.986 | 30.425 | 고부하 marginal cost |
| 75-0 | 48.832 | 46.094 | 0% baseline 포함으로 큼 |
| 100-25 | 37.772 | 39.699 | 25% 포함으로 slope보다 큼 |
| 100-50 | 30.491 | 30.712 | 고부하 marginal cost |
| 100-75 | 30.023 | 31.139 | 고부하 marginal cost |
| 100-0 | 44.368 | 43.203 | phase-local 0% 대비, slope보다 큼 |

모든 pair가 30 pJ/bit로 나오지는 않는다. RTX 3090에서는 50% 이상 고부하 구간끼리의 incremental cost만 약 30 pJ/bit이고, 0% 또는 25%를 포함하는 pair는 38-58 pJ/bit로 커진다.

### 11.3 Buffer sweep

2026-05-14에 `1/2/4/8 GiB`, `--window-ms 200`, 각 3회 반복으로 추가 확인했다.

| buffer | read slope pJ/bit | write slope pJ/bit | read 100%-0% | write 100%-0% | 해석 |
|---:|---:|---:|---:|---:|---|
| 1 GiB | 28.891 ± 0.271 | 30.907 ± 0.550 | 43.659 | 43.713 | RTX 3090 L2 대비 약 170배 |
| 2 GiB | 30.127 ± 0.773 | 31.627 ± 0.164 | 44.498 | 44.527 | 1 GiB와 같은 범위 |
| 4 GiB | 30.501 ± 1.195 | 32.380 ± 0.389 | 44.782 | 45.132 | write가 약간 높지만 결론 유지 |
| 8 GiB | 30.063 ± 0.758 | 31.297 ± 0.405 | 44.631 | 44.113 | 큰 buffer에서도 결론 유지 |

결론: buffer를 1 GiB에서 8 GiB로 키워도 slope 기반 marginal estimate는 read 약 `29-30.5 pJ/bit`, write 약 `31-32.4 pJ/bit` 범위에 머문다. L2/cache hit가 결과를 지배했다면 더 큰 buffer-size 의존성이 보여야 한다.

### 11.4 Window sweep

8 GiB buffer에서 `--window-ms 20/50/100/200/400` 단일 run을 비교했다.

| window-ms | read slope pJ/bit | write slope pJ/bit | 25% pass/window | 판단 |
|---:|---:|---:|---:|---|
| 20 | 28.945 | 28.522 | read 0.52, write 0.49 | invalid: target quantization |
| 50 | 32.094 | 31.324 | read 1.30, write 1.21 | marginal |
| 100 | 31.151 | 32.231 | read 2.60, write 2.43 | marginal |
| 200 | 29.637 | 31.966 | read 5.19, write 4.85 | valid |
| 400 | 28.596 | 30.215 | read 10.38, write 9.71 | valid |

8 GiB에서 `--window-ms 20`은 실제로 깨졌다. read 25/50%가 약 `408/416 GB/s`, write 25/50/75%가 약 `410/417/417 GB/s`로 뭉쳐 target separation이 사라졌다.

### 11.5 RTX 3090 최종 해석

1. confidence 높은 값: read 약 `30 pJ/bit`, write 약 `31-32 pJ/bit`.
2. `100%-0%`: 약 `44-45 pJ/bit`, 상한성 sanity check.
3. 현재 값은 DRAM rail-only가 아니라 NVML board-level marginal dynamic pJ/bit.

## 12. Legacy read-utilization 및 Nsight 부록

### 12.1 CuPy read-utilization

기존 `dram_util_cupy.py`는 DRAM read utilization을 25/50/75/100% 계단으로 만드는 도구다. pJ/bit 계산은 하지 않는다.

```bash
cd util/dram_util_experiment
./run_cupy.sh

python3 dram_util_cupy.py --targets 25 50 75 100 --phase-seconds 10
```

출력:

1. `reports/util_cupy_<gpu_slug>_<timestamp>.csv`
2. `reports/util_cupy_<gpu_slug>_<timestamp>.png`

### 12.2 CuPy + Nsight Systems

```bash
./run_nsys_cupy.sh
./run_nsys_cupy.sh --phase-seconds 5
./run_nsys_cupy.sh --no-metrics

nsys-ui reports/nsys_cupy_*.nsys-rep
```

확인할 것:

1. NVTX range: `util_25 / util_50 / util_75 / util_100`.
2. GPU metrics: DRAM read throughput 계단.
3. CUDA kernels: `stream_read` duty-cycle 패턴.

### 12.3 Native CUDA + Nsight Systems

요구사항:

1. CUDA Toolkit 12.x 이상.
2. Nsight Systems 2024.x 이상.
3. GPU metrics 권한. 필요하면 `/etc/modprobe.d/nvidia.conf`에 `options nvidia NVreg_RestrictProfilingToAdminUsers=0`.

실행:

```bash
make clean
make SM=86
./run_nsys.sh

./run_nsys_a100.sh
```

수동 예시:

```bash
DRAM_BUF_BYTES=$((8*1024*1024*1024)) make SM=80
DRAM_BUF_BYTES=$((8*1024*1024*1024)) ./run_nsys.sh
```

## 13. Bandwidth 용어와 공개 pJ/bit 비교

### 13.1 Raw-bus peak vs effective peak

두 수치를 혼동하면 A100 80GB의 1779 GB/s calibration peak를 2039 GB/s raw peak보다 낮다고 잘못 해석하게 된다.

| 용어 | 의미 | A100 80GB 예 |
|---|---|---:|
| Raw-bus peak | HBM 물리 bus 전체 대역. 사용자 데이터 + ECC syndrome + controller overhead 포함 | 2039 GB/s |
| Effective user-data peak | load/store 프로그램이 실제 얻는 사용자 데이터 bandwidth | ~1700-1800 GB/s |

GPU별 기대 범위:

| GPU | Raw-bus peak | Effective streaming peak | 효율 |
|---|---:|---:|---:|
| RTX 3090 GDDR6X | 936 GB/s | 880-900 GB/s | 93-96% |
| A100 40GB HBM2 ECC on | 1555 GB/s | 1350-1450 GB/s | 87-93% |
| A100 80GB HBM2e ECC on | 2039 GB/s | 1700-1800 GB/s | 83-88% |
| H100 SXM HBM3 ECC on | 3350 GB/s | 2700-3000 GB/s | 80-90% |

### 13.2 공개 memory pJ/bit와 비교

| Memory | 공개/문헌상 대략 범위 | 주의 |
|---|---:|---|
| GDDR7 | 약 4.5 pJ/bit | Micron average device power 기준 |
| GDDR6X | 약 6-7.25 pJ/bit | 세대/속도 bin/조건 의존 |
| HBM3 | 약 3-5 pJ/bit | 제조사 exact value는 보통 비공개 |
| HBM3E | 약 2.5-4 pJ/bit | 공개 자료는 상대 power 개선 위주 |

이 표는 device/PHY 관점이다. 본 실험의 RTX 3090 `30 pJ/bit` 수준 값은 board-level marginal 값이라 직접 비교하면 안 된다.

## 14. 관련 연구와 방법론 체크

1. GPUWattch는 microbenchmark와 실제 hardware power 측정을 함께 써서 GPU power model을 검증했다. 이 실험도 제어된 DRAM traffic과 board-level power marginal slope를 사용한다.
2. AccelWattch는 modern GPU의 cycle-level power와 constant/static power까지 모델링한다. 본 측정값은 DRAM rail-only 상수가 아니라 memory-subsystem calibration anchor로 쓰는 것이 안전하다.
3. MEMPower는 GPU memory access energy가 data pattern, core, channel에 따라 달라질 수 있음을 보인다. data-dependent energy가 필요하면 all-zero, all-one, random, high-toggle pattern을 별도 run으로 추가한다.
4. Nsight Systems GPU Metrics가 있으면 DRAM read/write throughput counter로 NVML 기반 effective BW와 계단 모양을 cross-check한다.

## 15. 최종 보고 체크리스트

1. `nvidia-smi`로 다른 compute process가 없음을 확인했는가?
2. GPU name, driver, CUDA runtime, L2, buffer size가 metadata에 남았는가?
3. calibration peak GB/s가 GPU 기대 범위에 들어오는가?
4. `--window-ms` warning이 없거나, warning이 있는 target을 해석에서 제외했는가?
5. 50/75/100 실제 `bandwidth_gbps`가 잘 분리되는가?
6. `slope_avg_power_vs_bw`의 R2가 충분히 높고 residual이 작은가?
7. 3회 이상 반복 평균/표준편차를 냈는가?
8. H100이면 `power_instant_w`/`power_average_w` field를 cross-validation했는가?
9. `100%-0%`와 slope 값을 구분해서 보고했는가?
10. DRAM rail-only가 아니라 GPU/board marginal dynamic pJ/bit임을 명시했는가?
