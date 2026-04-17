# DRAM Read Utilization 실험

GPU DRAM **read utilization** 을 **25% → 50% → 75% → 100%** 로 각각 10초씩
계단 형태로 강제 구동하고, Nsight Systems 타임라인에서 검증하기 위한
자동화 도구 모음.

AccelWattch GPU 전력 모델 검증(특히 DRAM 전력 항) 에서 **원하는 BW utilization
수준을 제어한 microbenchmark** 가 필요해서 만들었음.

## 원리

- **커널**: L2 보다 훨씬 큰 전역 float4 버퍼를 `__ldcg` 로 스트리밍 read →
  피크 DRAM read BW 포화.
- **Host-side duty cycling**: 20 ms 윈도우 안에서 `target%` 만큼만 커널을
  실행하고 나머지는 `sleep` → 10 s 평균으로 DRAM read utilization 이
  25/50/75/100% 로 수렴. 창이 작아서 Nsight Systems 의 throughput 샘플 그래프
  도 평탄하게 보임.
- **100% phase**: 단일 커널 런치로 10 초짜리 포화 구동 (큐가 비지 않음).
- **NVTX range**: 각 phase 를 `util_25 / util_50 / util_75 / util_100` 로 라벨,
  경계는 0.5 s `gap` range 로 분리 → nsys 타임라인에서 눈으로 바로 구분 가능.

## 파일

| 파일 | 설명 |
|---|---|
| `dram_util.cu` | 스트리밍 read 커널 + 호스트 duty cycling 드라이버 |
| `Makefile` | `SM` 변수로 arch 지정 (기본 `sm_86` = RTX 3090) |
| `run_nsys.sh` | 빌드 + `nsys profile` (GPU metrics 샘플링 포함) + sqlite export + 분석 |
| `run_nsys_a100.sh` | A100 80GB 프리셋 (`SM=80`, 버퍼 8 GiB) → `run_nsys.sh` 로 exec |
| `analyze.py` | nsys sqlite 에서 phase 별 DRAM read 지표 평균/표준편차 계산 |

## 요구사항

- CUDA Toolkit 12.x 이상 (`nvcc`)
- Nsight Systems 2024.x 이상 (`nsys`) — GPU metrics sampling 위해 최신 권장
- NVIDIA 드라이버
- `--gpu-metrics-device` 사용 시 관리자 프로파일링 권한:
  ```
  # /etc/modprobe.d/nvidia.conf
  options nvidia NVreg_RestrictProfilingToAdminUsers=0
  ```
  변경 후 재부팅.

## 사용

### RTX 3090 (기본)

```bash
cd util/dram_util_experiment
./run_nsys.sh
```

### A100 80GB

```bash
./run_nsys_a100.sh
```

### 옵션

- `--no-build` — 바이너리 재사용 (재컴파일 생략)
- `--no-analyze` — sqlite export/분석 스킵

### 수동 실행

```bash
make SM=86                          # 또는 SM=80 (A100), SM=90 (H100)
DRAM_BUF_BYTES=$((2*1024**3)) \
nsys profile -o report \
  --trace=cuda,nvtx --sample=none \
  --gpu-metrics-device=0 --gpu-metrics-frequency=10000 \
  ./dram_util
```

## 결과 확인

### 1. Nsight Systems GUI

```bash
nsys-ui reports/dram_util_*.nsys-rep
```

확인할 행:

- **GPU Metrics → DRAM Read Throughput** : 25/50/75/100% 계단
- **NVTX** : `util_25` / `util_50` / `util_75` / `util_100` (각 10 s) + `gap`
- **CUDA HW → Kernels** : `stream_read_kernel` 점유 패턴 (25% phase 는 20 ms 중 5 ms 만 점유 등)

### 2. CLI 자동 분석 (`analyze.py`)

`run_nsys.sh` 가 자동 실행. 예상 출력:

```
phase      metric                              target%     mean    stdev  samples
----------------------------------------------------------------------------------
util_25    DRAM Read Throughput                     25    25.3     2.1      1024
util_50    DRAM Read Throughput                     50    50.1     3.0      1024
util_75    DRAM Read Throughput                     75    74.8     2.4      1024
util_100   DRAM Read Throughput                    100    99.2     0.8      1024
```

목표치와 평균이 ±3% 이내면 실험 성공.

## 주의사항

- **WSL2**: `--gpu-metrics-device` 샘플링이 드라이버/nsys 버전에 따라 제한될 수
  있음. 실패 시 해당 플래그를 빼고 NVTX + 커널 타임라인만으로 점유율을 확인
  (kernel active time / phase time ≈ target util).
- **L2 cache bypass**: 커널은 `__ldcg` 를 쓰고 버퍼 크기는 `max(1 GiB, 64 × L2)`
  로 자동 결정되어 read 가 전부 DRAM 까지 내려가도록 함.
- **Read-only**: 저장(`sink`) 은 dead store (센티넬 조건부) 이므로 DRAM write
  utilization 은 0 에 수렴. Read 를 측정하고 싶을 때만 쓸 것.
- **피크 BW 기준**: 각 GPU 이론 피크 대비 utilization 이므로, 실제 측정된 peak
  GB/s 는 calibration 로그의 `[calib] ... GB/s peak DRAM read` 값을 참고.

## 확장

- **write utilization 실험**: 커널을 `sink[i] = v` 형태로 바꾸면 DRAM write 버전.
- **다른 계단 (예: 10/30/60/90)**: `dram_util.cu` 의 `targets[]` 수정.
- **phase 길이 변경**: `phase_ms` (기본 10000.0) 수정.
- **duty 윈도우 변경**: `window_ms` (기본 20.0) — 더 작게 하면 그래프가 더
  평탄해지지만 런치 오버헤드 비중이 커짐.
