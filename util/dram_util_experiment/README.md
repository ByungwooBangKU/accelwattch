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
7. 큰 buffer에서는 `--window-ms`가 중요하다. 8 GiB/A100/H100 계열 보고용은 `--window-ms 100` 또는 `200`을 권장한다.
8. H100에서는 `NVML_FI_DEV_POWER_INSTANT`/`AVERAGE` field API를 같이 기록해서 cross-validation할 수 있다.
9. H100 write 해석은 `--write-patterns zero const address random toggle` sweep으로 data pattern/compression 영향을 분리한다.

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `dram_pjbit_cupy.py` | read/write pJ/bit 주 실험 스크립트 |
| `run_pjbit_cupy.sh` | pJ/bit 실험 launcher. 필요한 Python 환경 자동 탐색 |
| `run_pjbit_repeats.sh` | 3회 반복 실험, per-run 이미지, 반복 요약 CSV/PNG 자동 생성 |
| `summarize_pjbit_repeats.py` | 여러 `*_analysis.csv` 반복 run 평균/표준편차 요약 및 요약 PNG 생성 |
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

```text
[1] GPU 상태 확인
    - nvidia-smi, 다른 process, power limit, clocks, temperature 확인
        |
        v
[2] CUDA device property 조회
    - GPU name, SM count, L2 size, driver/runtime 정보 저장
        |
        v
[3] Working set 결정
    - 기본값: max(1 GiB, 64 x L2)
    - H100/A100 보고용 sensitivity: 8 GiB도 추가 권장
        |
        v
[4] read/write kernel JIT 및 1-pass calibration
    - mode별 effective user-data peak GB/s 측정
        |
        v
[5] idle baseline power 측정
    - pre-idle baseline은 summary pJ/bit 계산에 사용
        |
        v
[6] target phase 실행
    - 예: 0/25/50/75/100%
    - target은 raw bus peak가 아니라 calibration peak 대비 duty target
    - write는 필요 시 zero/const/address/random/toggle pattern별로 분리
        |
        v
[7] phase별 raw data 저장
    - bytes, bandwidth, power, clocks, temperature, P-state, NVML field power
        |
        v
[8] pJ/bit 분석
    - 100%-0%, all-pair delta, 50/75/100 multi-point slope 계산
        |
        v
[9] 반복 run 요약
    - 평균, 표준편차, R2, residual로 confidence 판단
```

### 3.2 계산식

```text
E_dyn      = max(0, E_total - P_idle x wall_time)
BW_GBps    = transferred_bytes / wall_time / 1e9
pJ_per_bit = E_dyn / (transferred_bytes x 8) x 1e12
           = dynamic_power_W x 1000 / (8 x BW_GBps)
```

`target=0`은 kernel launch 없이 같은 phase 길이 동안 power만 기록한다. CSV에는 `bandwidth_gbps=0`, `bytes_transferred=0`, `pj_per_bit=nan`으로 남는다. 전송 bit가 0이므로 0% phase의 pJ/bit는 정의되지 않는다. 최신 `*.png`의 dynamic pJ/bit plot은 0% phase를 막대에서 제외하고 active phase만 표시한다.

### 3.3 read/write 커널

1. `read`: 큰 `float4` buffer를 `__ldcg`로 streaming load한다. L1은 우회하고 L2/DRAM 경로에 pressure를 준다.
2. `write`: 같은 크기의 `float4` buffer 전체를 streaming store한다. byte 수는 user-data write byte 기준이다. 현재 구현은 일반 global store를 사용하며, L2를 완전히 bypass한다고 가정하지 않는다.
3. read와 write를 같이 측정할 때는 buffer를 분리한다. write pattern calibration이 read buffer 내용을 덮어 read energy를 오염시키지 않게 하기 위해서다.
4. working set은 L2보다 훨씬 크게 잡기 때문에 DRAM-dominant access가 되도록 설계한다.

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

### 4.3 L1/L2 hit-rate와 eviction 가정

설계상 read 재사용 hit-rate는 거의 0에 가까워야 한다.

1. read는 `__ldcg`로 L1을 우회한다.
2. working set이 L2보다 훨씬 크다.
3. RTX 3090 sweep에서 1/2/4/8 GiB로 buffer를 키워도 slope 결과가 크게 바뀌지 않았다.

하지만 이것은 직접 counter 검증이 아니라 설계상 기대다. 특히 write는 read와 다르게 해석해야 한다.

1. NVIDIA GPU에서 일반 CUDA 옵션으로 global memory access가 L2를 완전히 bypass하도록 L2를 disable하는 방법은 없다.
2. PTX의 `ld/st` cache operator는 성능 hint다. `st.cg`, `st.cs`, `st.wt`, `L1::no_allocate`, eviction priority hint는 cache 동작을 유도할 수 있지만 L2 bypass를 보장하는 장치가 아니다.
3. `st.cs`는 streaming store에 적합한 evict-first 성격의 후보지만, 현재 `dram_pjbit_cupy.py`에는 별도 `--write-cache-op` 옵션을 구현하지 않았다.
4. `discard.global.L2`는 L2 line을 무효화할 수 있지만 data를 writeback하지 않고 해당 address range의 값이 undetermined가 될 수 있다. 따라서 pJ/bit write 실험의 정상 store traffic 검증 수단으로 쓰면 안 된다.
5. write의 L2 hit-rate는 read miss-rate와 같은 의미가 아니다. store는 L2에 hit/allocate된 뒤 dirty line이 나중에 DRAM으로 writeback될 수 있으므로, 최종 검증은 `dram write bytes` 또는 memory-controller write counter가 기대 byte와 맞는지를 봐야 한다.

따라서 최종 H100/A100 보고에서는 Nsight Compute로 L2/DRAM counter를 별도 validation run에서 확인하는 것이 좋다. NVML power 측정 run과 Nsight Compute profiling run은 분리한다.

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

### 6.1 주요 파라미터

`run_pjbit_cupy.sh`는 Python 환경을 고른 뒤 모든 인자를 그대로 `dram_pjbit_cupy.py`에 전달한다. 따라서 아래 옵션은 두 실행 방식에서 동일하게 쓴다.

| 파라미터 | 기본값 | 의미 | 권장/주의 |
|---|---:|---|---|
| `--device` | `0` | CUDA/NVML GPU index | 다중 GPU 시스템에서는 `nvidia-smi`의 index와 맞춰 명시한다. |
| `--modes` | `read write` | 측정할 traffic 종류 | read만 볼 때는 `--modes read`, write만 볼 때는 `--modes write`. |
| `--write-patterns` | `const` | write phase에서 쓸 data pattern 목록 | H100 write 해석은 `zero const address random toggle` sweep 권장. read mode에는 영향 없다. |
| `--targets` | `100` | calibration peak 대비 duty target percent | 보고용 sweep은 `0 25 50 75 100`. 단, 해석은 label보다 실제 `bandwidth_gbps` 기준으로 한다. |
| `--phase-seconds` | `5` | 각 mode/target phase를 유지하는 시간 | smoke test는 5초도 가능하지만 보고용은 최소 20초 권장. |
| `--idle-seconds` | `5` | active phase 전 pre-idle baseline 측정 시간 | 보고용은 15초 이상 권장. `summary.csv`의 dynamic pJ/bit baseline으로 쓰인다. 이 단계는 지정한 초 수 정도만 걸리는 것이 정상이다. |
| `--window-ms` | `20` | 25/50/75% duty-cycle 제어 window | 큰 buffer에서는 너무 작으면 pass quantization warning 발생. A100/H100 보고용은 `100` 또는 `200` 권장. |
| `--poll-hz` | `100` | NVML power/state polling 요청 주파수 | 기본 100 Hz는 10 ms마다 NVML 값을 요청한다는 뜻이다. 실제 sensor update/window는 더 느릴 수 있다. |
| `--gap-seconds` | `1.0` | phase 사이 idle gap | phase 경계의 sensor averaging/transition bleed-through를 줄이기 위한 구간이다. A100/H100에서 0% power가 의심스러우면 `2`로 늘려 비교한다. |
| `--phase-order` | `target-major` | phase 실행 순서 | `target-major`는 모든 0% phase를 active phase보다 먼저 실행한다. 예전 방식처럼 read sweep 전체 뒤 write sweep을 돌리려면 `workload-major`를 쓴다. |
| `--buf-bytes` | 자동 | mode별 GPU buffer 크기 | 기본값은 `max(1 GiB, 64 x L2)`. 8 GiB는 `8589934592`. read/write를 같이 돌리면 read buffer와 write buffer를 분리하므로 총 할당량은 약 2배다. |
| `--out-dir` | `reports` | 결과 파일 저장 디렉터리 | `reports/`는 git ignore 대상이다. |
| `--tag` | 빈 문자열 | 출력 파일명 suffix | 반복 run 구분을 위해 `baseline_w200_rep1`처럼 명시한다. |
| `--cal-passes` | `8` | calibration에서 repeat당 streaming pass 수 | calibration bandwidth가 흔들릴 때만 늘린다. 보통 기본값 유지. |
| `--cal-repeats` | `3` | calibration 반복 횟수 | peak 추정이 흔들리면 늘린다. 보통 기본값 유지. |

파라미터 선택 순서:

1. `--device`와 `--modes`로 측정 GPU와 read/write 범위를 정한다.
2. `--buf-bytes`는 기본값으로 시작하고, H100/A100 보고용이면 8 GiB sensitivity run을 추가한다.
3. H100 write 해석이면 `--write-patterns zero const address random toggle`로 pattern sensitivity를 같이 본다.
4. `--targets 0 25 50 75 100`으로 utilization sweep을 만들되, 최종 계산은 실제 `bandwidth_gbps`와 slope를 우선한다.
5. `--window-ms` warning이 나오면 A100/H100은 `100` 또는 `200`으로 올린다.
6. A100/H100에서 0% power가 이전 active phase의 영향을 받는 것처럼 보이면 `--gap-seconds 2`로 늘리고 `--phase-order target-major`를 유지한다.
7. 최종 보고는 같은 조건을 `rep1/rep2/rep3`으로 반복하고 `summarize_pjbit_repeats.py`로 평균/표준편차를 낸다.

write pattern 의미:

| pattern | 의미 | 목적 |
|---|---|---|
| `zero` | 모든 bit가 0 | compression/toggle-minimum 하한성 case |
| `const` | 기존 방식. pass별 constant `float4` | 과거 결과와 비교하는 legacy case |
| `address` | address-ramp bit pattern | constant-write 최적화 여부 확인 |
| `random` | xorshift 기반 deterministic pseudo-random bits | compression이 거의 안 되는 write stress. pattern 생성 ALU가 약간 섞일 수 있음 |
| `toggle` | checker/toggle bit pattern | 단순 0/1 전환과 bit-toggle sensitivity 확인 |

### 6.2 간편 3회 반복 run

보고용 기본 실행은 이 wrapper를 우선 사용한다. 각 반복 run은 `*_summary.csv`, `*_analysis.csv`, `*_quality_checks.csv`, `*_bandwidth.png`, `*_analysis.png`, `*_write_patterns.png`를 자동 생성하고, 마지막에 반복 평균/표준편차 요약 CSV와 PNG를 만든다.

RTX 3090:

```bash
cd util/dram_util_experiment
./run_pjbit_repeats.sh --profile rtx3090
```

A100 8 GiB:

```bash
cd util/dram_util_experiment
./run_pjbit_repeats.sh --profile a100-8gib --device 0
```

H100 8 GiB:

```bash
cd util/dram_util_experiment
./run_pjbit_repeats.sh --profile h100-8gib --device 0
```

자동 GPU profile 선택:

```bash
cd util/dram_util_experiment
./run_pjbit_repeats.sh --profile auto --device 0
```

출력 예:

```text
reports/<tag>_repeat_summary.csv
reports/<tag>_repeat_summary.png
reports/*<tag>_rep1*.png
reports/*<tag>_rep2*.png
reports/*<tag>_rep3*.png
```

자주 바꾸는 옵션:

```bash
./run_pjbit_repeats.sh \
  --profile h100-8gib \
  --device 0 \
  --tag h100_8gib_patterns \
  --repeats 3 \
  --phase-seconds 20 \
  --window-ms 200
```

### 6.3 빠른 smoke test

```bash
cd util/dram_util_experiment

./run_pjbit_cupy.sh \
  --modes read write \
  --targets 100 \
  --phase-seconds 5 \
  --idle-seconds 5 \
  --tag smoke
```

### 6.4 단일 보고용 run

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

### 6.5 수동 3회 반복 run

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
  --out reports/baseline_w200_repeat_summary.csv \
  --plot-out reports/baseline_w200_repeat_summary.png
```

### 6.6 A100 run

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

### 6.7 H100 run

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

H100 write-pattern sweep:

```bash
./run_pjbit_cupy.sh \
  --device 0 \
  --modes read write \
  --write-patterns zero const address random toggle \
  --targets 0 50 75 100 \
  --buf-bytes 8589934592 \
  --phase-seconds 20 \
  --idle-seconds 15 \
  --poll-hz 100 \
  --window-ms 200 \
  --tag h100_8gib_patterns_rep1
```

read/write order 영향을 확인하려면 같은 조건을 `--modes write read`로 한 번 더 돌린다.

## 7. 출력 파일

| 파일 | 내용 |
|---|---|
| `*_summary.csv` | phase별 workload/pattern, BW, power, energy, pJ/bit, clocks, temp, P-state |
| `*_trace.csv` | time-series power/util/clocks/temp/phase |
| `*_analysis.csv` | workload/pattern별 `100%-0%`, all-pair delta, slope pJ/bit, R2, residual |
| `*_quality_checks.csv` | target coverage, bandwidth separation, fit quality, H100 pattern coverage 자동 점검 |
| `*_metadata.json` | GPU, driver, CUDA, L2, buffer, calibration, power limit |
| `*.png` | power timeline, phase별 BW, active phase별 pJ/bit. power timeline x축은 poller 시작 이후 경과 시간(`t_s`)이며 단위는 second |
| `*_bandwidth.png` | target별 실제 measured bandwidth와 100% 실측값 대비 calibration peak 비교 |
| `*_analysis.png` | power-vs-bandwidth fit, residual, estimator 비교 |
| `*_write_patterns.png` | write pattern별 power/BW fit, pJ/bit, 100% power 비교. pattern이 2개 이상일 때 생성 |
| `<tag>_repeat_summary.csv` | `run_pjbit_repeats.sh`가 생성하는 반복 평균/표준편차 요약 |
| `<tag>_repeat_summary.png` | 반복 평균/표준편차 pJ/bit 요약 이미지 |

`trace.csv`는 NVML field API가 지원되면 아래 컬럼도 포함한다.

| 컬럼 | 의미 |
|---|---|
| `power_w` | 기존 `nvmlDeviceGetPowerUsage()` 값. Ampere 이상에서는 1초 평균 성격 |
| `power_instant_w` | `NVML_FI_DEV_POWER_INSTANT` 값 |
| `power_average_w` | `NVML_FI_DEV_POWER_AVERAGE` 값 |
| `power_instant_status` | instant field `nvmlReturn`; `0`이면 성공 |
| `power_average_status` | average field `nvmlReturn`; `0`이면 성공 |

## 8. 결과 해석

### 8.0 bandwidth normalization

`target_pct`는 요청한 duty target label이다. pJ/bit 계산에는 이 label이나 calibration peak를 분모로 쓰지 않는다. 항상 phase에서 실제 실행한 byte와 실제 걸린 시간을 사용한다.

```text
bytes_transferred = launches x passes_per_launch x buf_bytes
bandwidth_gbps    = bytes_transferred / wall_s / 1e9
```

따라서 `100%` phase도 calibration peak와 같을 필요가 없다. 예를 들어 calibration이 893 GB/s이고 `100%` phase 실측이 834 GB/s라면 pJ/bit 계산에는 834 GB/s가 들어간다. `*_bandwidth.png`는 이 차이를 매 run마다 시각화한다.

### 8.1 pre-idle baseline pJ/bit

`summary.csv`의 `dynamic_power_w`, `dynamic_energy_j`, `pj_per_bit`는 pre-idle baseline 기준이다.

`target=0` phase는 baseline 관찰용이다. byte 전송량이 0이므로 pJ/bit 분모가 없고, 따라서 `write_0` 또는 `read_0`의 pJ/bit는 해석 대상이 아니다. 만약 예전 이미지에서 `write_0` pJ/bit 막대가 크게 보인다면 그 막대는 유효한 DRAM energy 값이 아니므로 최신 스크립트로 다시 생성해야 한다.

A100/H100처럼 power API의 update/window가 길게 보이는 환경에서는 이전 active phase가 다음 0% phase의 `avg_power_w`에 일부 남을 수 있다. 기본 실행은 `--phase-order target-major`로 0% phase를 먼저 돌리고, phase 사이에는 `--gap-seconds 1.0`을 둔다. 그래도 `write_0` baseline이 높게 보이면 `--gap-seconds 2`로 재실행하고 `trace.csv`의 `power_w`, `power_instant_w`, `power_average_w`를 같이 확인한다.

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

### 8.5 H100 write pattern 해석

H100에서 read pJ/bit이 write보다 크게 보이면 곧바로 HBM device의 read/write energy 순서가 뒤집혔다고 해석하면 안 된다. 먼저 write pattern sensitivity를 본다.

1. `zero`/`const` write가 낮고 `random`/`toggle` write가 높으면 data pattern, compression, bit-toggle 영향이 섞인 것이다.
2. 모든 write pattern이 read보다 낮으면 read-return path, L2 fill, SM load path, read kernel accumulation 비용이 NVML board-level power에 더 크게 잡혔을 가능성이 있다.
3. `random`/`toggle` write가 read와 같거나 더 높아지면 기존 `const` write pattern이 H100 write energy를 낮게 보이게 만든 것이다.
4. 최종 보고는 `read`, `write:const`, `write:random`, `write:toggle`를 분리해서 적고, DRAM rail-only 값이 아니라 board/module marginal pJ/bit임을 명시한다.

### 8.6 write cache/eviction 해석

write phase가 DRAM traffic을 만들었는지 판단할 때는 "L2 hit-rate가 0인가?"만 보면 부족하다. store는 L2에 hit하거나 allocate된 뒤 나중에 dirty writeback으로 DRAM traffic을 만들 수 있기 때문이다.

권장 판단 순서:

1. `bandwidth_gbps`가 calibration peak의 85-95% 수준으로 나오고 target별 bandwidth가 잘 분리되는지 확인한다.
2. `--buf-bytes`가 L2보다 충분히 큰지 확인한다. RTX 3090 1 GiB는 약 `170 x L2`, H100 8 GiB는 약 `160 x L2` 수준이다.
3. `zero/const/address/random/toggle` pattern별 slope가 다르게 나오는지 확인한다. pattern 차이가 보이면 단순 cache-hit artifact보다 data pattern/toggle/compression sensitivity가 섞였을 가능성이 크다.
4. Nsight Compute가 있으면 별도 profiling run에서 DRAM write byte counter를 확인한다. 기대값은 대략 `launches x passes_per_launch x buf_bytes`와 같은 order여야 한다.
5. Nsight Compute가 없으면 현재 문서의 값은 "DRAM-dominant로 설계된 board-level marginal estimate"로 표현하고, cache eviction 직접 검증은 미완료라고 명시한다.

현재 구현은 L2를 끄거나 완전히 bypass하지 않는다. 따라서 read/write 비교에서 가장 안전한 문장은 "large working set과 pattern sweep으로 DRAM-dominant write를 유도했고, Nsight Compute counter가 있으면 이를 별도로 검증한다"이다.

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

### 12.3 Nsight Compute L2/DRAM counter validation

NVML power 실험과 Nsight Compute profiling은 목적이 다르므로 분리해서 실행한다. Nsight Compute는 replay/profiling overhead가 크고, power trace를 왜곡할 수 있다. 이 단계의 목적은 pJ/bit를 다시 재는 것이 아니라 cache/DRAM traffic이 의도대로 발생했는지 확인하는 것이다.

먼저 metric 이름을 현재 Nsight Compute 버전에서 확인한다.

```bash
ncu --query-metrics | grep -E "dram__.*write|dram__.*read|lts__.*write|lts__.*hit|l1tex__.*hit"
```

버전에 따라 metric 이름이 다르다. 아래 명령은 예시이며, 실제 환경에서는 `--query-metrics` 결과에 맞게 조정한다.

```bash
ncu \
  --target-processes all \
  --kernel-name regex:stream_write \
  --launch-skip 2 \
  --launch-count 1 \
  --metrics l2_tex_write_hit_rate,l2_tex_write_transactions,dram__bytes_write.sum,dram__bytes_read.sum \
  ./run_pjbit_cupy.sh \
    --modes write \
    --write-patterns random \
    --targets 100 \
    --buf-bytes 8589934592 \
    --phase-seconds 2 \
    --cal-passes 1 \
    --cal-repeats 1 \
    --tag ncu_write_random
```

확인 기준:

1. write phase에서 DRAM write byte counter가 기대 byte 수와 같은 order인지 본다.
2. write phase에서 예상 밖의 DRAM read byte가 큰지 확인한다. write allocate, compression, ECC, controller 동작 때문에 0이 아닐 수 있지만, read-dominant이면 해석에 주의한다.
3. L2 hit-rate는 보조 지표로만 본다. store hit가 곧 DRAM write 부재를 뜻하지 않는다.
4. read phase는 `stream_read`를 같은 방식으로 profiling해서 DRAM read byte와 L2 hit/miss 지표를 확인한다.
5. 이 validation 결과가 없으면 보고서에 "Nsight Compute L2/DRAM counter cross-check는 미수행"이라고 명시한다.

### 12.4 Native CUDA + Nsight Systems

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
4. NVIDIA PTX 문서 기준 cache operator는 memory consistency를 바꾸지 않는 performance hint다. `st.cs`/evict-first 계열은 streaming write에 유용한 후보지만 L2 bypass 보장은 아니다.
5. `discard.global.L2`는 L2 line을 invalidate할 수 있지만 writeback 없이 데이터를 버리므로 정상 write energy 측정에는 적합하지 않다.
6. Nsight Compute GPU Metrics가 있으면 DRAM read/write byte counter와 L2 counter로 NVML 기반 effective BW와 계단 모양을 cross-check한다.

참고 문서:

1. PTX ISA cache operators: <https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#cache-operators>
2. PTX ISA `discard.global.L2`: <https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard>
3. Nsight Compute metric aliases: <https://archive.docs.nvidia.com/nsight-compute/2022.1/NsightComputeCli/index.html>

## 15. 최종 보고 체크리스트

1. `nvidia-smi`로 다른 compute process가 없음을 확인했는가?
2. GPU name, driver, CUDA runtime, L2, buffer size가 metadata에 남았는가?
3. calibration peak GB/s가 GPU 기대 범위에 들어오는가?
4. `--window-ms` warning이 없거나, warning이 있는 target을 해석에서 제외했는가?
5. 50/75/100 실제 `bandwidth_gbps`가 잘 분리되는가?
6. `slope_avg_power_vs_bw`의 R2가 충분히 높고 residual이 작은가?
7. 3회 이상 반복 평균/표준편차를 냈는가?
8. H100이면 `power_instant_w`/`power_average_w` field를 cross-validation했는가?
9. H100 write 해석이면 `zero/const/address/random/toggle` pattern 결과를 분리해서 봤는가?
10. Nsight Compute가 있으면 DRAM read/write byte counter와 L2 counter를 별도 validation run으로 확인했는가?
11. Nsight Compute validation을 하지 못했다면 cache eviction 직접 검증이 미완료임을 보고서에 적었는가?
12. `*_quality_checks.csv`에 warning이 있으면 해석에서 반영했는가?
13. `100%-0%`와 slope 값을 구분해서 보고했는가?
14. DRAM rail-only가 아니라 GPU/board marginal dynamic pJ/bit임을 명시했는가?
