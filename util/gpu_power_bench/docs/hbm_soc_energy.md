# 실행계획서: HBM→L2 Refill Path Energy 측정 및 분석

작성일: 2026-05-09  
대상 repo: `ByungwooBangKU/accelwattch`  
대상 디렉터리: `util/gpu_power_bench/`

---

## 0. 한 줄 요약

목표는 문헌 또는 별도 가정으로 둘 HBM pJ/bit 경계를 명시한 뒤,
**HBM에서 L2 cache fill까지 발생하는 추가 데이터 이동 비용**을 최대한 분리해
추정하는 것이다. 다만 NVML은 board-level power만 제공하므로, 이 값을 순수
component 직접 계측값으로 주장하면 안 된다. 가장 안전한 headline은 다음이다.

```text
Primary measured proxy:
  HBM-to-L2 refill path pJ/bit

Secondary derived proxy:
  SoC-side HBM PHY→L2 fill path pJ/bit
  = HBM-to-L2 refill path
    - assumed HBM interface pJ/bit
    - L2 fill/write proxy pJ/bit
```

따라서 본 계획의 핵심은 **cold prefetch와 hot prefetch의 차이**를 이용해
`HBM→L2 refill` 비용을 먼저 측정하고, 그 후 필요한 가정을 명시적으로 차감해
가정한 HBM boundary 이후→L2 proxy를 산출하는 것이다.

---

## 1. 배경과 측정 경계

사용자가 정의한 계층은 다음과 같다.

```text
① HBM DRAM cell / array
② HBM peripheral circuits
③ TSV
④ HBM base die / logic die
⑤ HBM-side PHY / I/O circuitry
⑥ microbump + silicon interposer
⑦ SoC-side HBM PHY
⑧ memory controller
⑨ L2 cache
⑩ SM / Tensor Core
```

문헌에서 흔히 말하는 `HBM2E 4.3 pJ/bit`, `HBM3 2.5 pJ/bit` 같은 값은
논문/자료마다 boundary가 다르다. 어떤 값은 stack 내부와 I/O를 포함하고, 어떤
값은 interface나 PHY 일부를 별도로 두기도 한다. 따라서 아래 범위는 단정이 아니라
본 분석에서 사용자가 명시할 수 있는 **가정 예시**다.

```text
①~⑦:
HBM stack 내부 접근
+ HBM-side PHY/I/O
+ microbump/interposer
+ SoC-side HBM PHY 근처
```

첨부 이미지의 의도는 빨간 박스 안의 HBM subsystem, 즉 ①~⑦을 기존 HBM
interface 계수로 둘 수 있는 영역으로 보고, 그 오른쪽의 SoC 내부 경로와 L2 fill을
분리해 보고 싶다는 것이다. 이 boundary를 사용하면 본 계획에서 추가로 알고 싶은
범위는 다음이다.

```text
가정한 ⑦ 이후 → ⑨ L2까지:
SoC-side HBM PHY 이후
→ memory controller / memory-ingress path
→ L2 fill / allocation 근처
```

만약 사용한 문헌값이 ①~⑦보다 좁거나 넓다면, derived residual도 그 문헌값
이후의 나머지 경로일 뿐이다. 그래서 CSV에는 `hbm_boundary_assumption`을 남기고,
residual은 항상 assumption-dependent proxy로 표기한다.

주의할 점은 이것을 `NoC energy`라고 부르지 않는 것이다. 내부 SoC fabric이나 crossbar가 실제로 어떤 구조인지 공개되지 않은 상태에서 `NoC`라고 naming하면 과장된 해석이 된다. 따라서 본 계획서에서는 보수적으로 다음 이름을 사용한다.

```text
MC→L2 fill path
SoC-side HBM PHY→L2 fill path
HBM-to-L2 refill path
```

---

## 2. 기존 코드 기반 확인

현재 repo에는 이미 관련 기반이 있다.

### 2.1 A.2 DRAM/STREAM probe

`TestCases.md`에는 A.2 DRAM bandwidth probe가 있으며, `stream_read`, `stream_write`, `stream_copy`, `stream_scale`, `stream_triad`를 통해 HBM/DRAM traffic pJ/bit를 추정하도록 되어 있다. 이 값은 HBM-streaming board-level memory path에 가깝고, L2 이후 SM까지 들어오는 비용 일부도 섞인다.

### 2.2 A.6 L2/SRAM resident traffic probe

`TestCases.md`에는 A.6 L2/SRAM resident traffic probe가 이미 정의되어 있다. 주요 op는 다음이다.

```text
reg_spin
l2_read_hit
l2_write_hit
l2_copy_hit
l2_sliding_delta
```

문서상 산출값은 이미 정확히 다음처럼 정의되어 있다.

```text
L2-hit traffic path energy,
not isolated SRAM bit-cell energy.
```

즉, 본 계획은 A.6을 확장해서 `HBM→L2 refill`에 가까운 probe를 추가하는 것이다.

### 2.3 BenchSpec 확장성

`benchmarks.py`의 `BenchSpec`에는 `extra: dict`가 존재한다. 따라서 L2/HBM refill 관련 metadata를 새 컬럼으로 넣기에 적합하다.

예상 metadata:

```text
estimated_prefetch_bits
estimated_hbm_refill_bits
estimated_l2_fill_bits
prefetch_line_bytes
repeat_inner
working_set_bytes
cold_pool_bytes
ptx_instruction
kernel_version
```

---

## 3. 목표와 비목표

## 3.1 목표

1. `HBM-to-L2 refill path pJ/bit`를 측정 가능한 proxy로 산출한다.
2. `SoC-side HBM PHY→L2 fill proxy`를 가정 기반 derived value로 산출한다.
3. 결과가 순수 component가 아님을 plot과 CSV에서 명확히 표시한다.
4. cold/hot prefetch 차분, reg_spin 차감, Nsight Compute counter validation을 통해 오류를 발라낸다.
5. 기존 A.2 HBM/STREAM 결과, A.6 L2-hit 결과와 일관성 검증을 수행한다.

## 3.2 비목표

다음은 본 실험으로 직접 측정하지 않는다.

```text
pure HBM DRAM cell energy
pure HBM-side PHY energy
pure SoC-side HBM PHY energy
pure memory controller energy
pure NoC/fabric energy
pure L2 SRAM bit-cell energy
```

특히 `SoC-side HBM PHY→L2` 단독값은 직접 측정값이 아니라 **derived proxy**다.

---

## 4. 핵심 측정 아이디어

## 4.1 세 가지 kernel

새 benchmark는 다음 세 kernel을 기본으로 한다.

```text
reg_spin
l2_prefetch_hot
l2_prefetch_cold
```

### reg_spin

memory traffic이 없는 baseline이다. 같은 block/grid/repeat 구조를 유지하되 global memory access를 하지 않는다.

목적:

```text
loop overhead
instruction overhead
control overhead
register-only overhead
```

를 차감하기 위함.

### l2_prefetch_hot

측정 전에 target window를 L2에 warm-up한다. 이후 같은 cache line을 다시 L2 prefetch한다.

목적:

```text
prefetch instruction overhead
hot L2 hit behavior
L2 tag/path overhead
```

를 얻기 위함.

### l2_prefetch_cold

cold pool에서 이전에 L2에 없던 cache line을 prefetch한다. 가능한 한 HBM에서 L2로 line이 채워지도록 만든다.

목적:

```text
HBM→L2 refill path energy
```

를 얻기 위함.

---

## 5. 권장 CUDA/PTX 구현

## 5.1 H100 1차 구현: PTX `cp.async.bulk.prefetch.L2.global`

Hopper H100, 즉 `sm_90+`에서는 PTX의 `cp.async.bulk.prefetch.L2.global`을 사용할 수 있다. 이 명령은 global memory 위치의 데이터를 L2 cache로 prefetch하도록 hint를 주는 non-blocking instruction이다. 단, 이것은 **cache hint**이며 항상 그대로 보장되는 강제 동작은 아니다. 그러므로 반드시 counter validation이 필요하다.

개념 예시:

```cuda
asm volatile(
  "cp.async.bulk.prefetch.L2.global [%0], %1;"
  :
  : "l"(ptr), "r"(size_bytes)
);
```

주의:

```text
현재 구현에서 H100 cp.async path의 size는 16/32/64/128/256B 중 하나여야 함
src address는 16B align 필요
sm_90 이상 필요
```

H100 profile의 기본값은 다음처럼 둔다.

```text
h100_sxm / h100_pcie:
  --l2-refill-ptx cp_async_bulk_prefetch_l2
  --l2-refill-window-mb 16 24 32 40
  --l2-refill-k-guess-pj-bit 3.0
```

## 5.2 RTX3090/Ampere fallback 구현: `prefetch.global.L2`

RTX3090은 Ampere `sm_86`이므로 `cp.async.bulk.prefetch.L2.global` headline
경로를 사용할 수 없다. RTX3090에서는 `prefetch.global.L2`를 기본 fallback으로
사용하고, H100 결과와 같은 headline으로 섞지 않는다.

```text
rtx3090:
  --l2-refill-ptx prefetch_global_l2
  --l2-refill-window-mb 1 2 3 4
  --l2-refill-k-guess-pj-bit 6.0
```

## 5.3 마지막 fallback: `__ldcg`

만약 `cp.async.bulk.prefetch.L2.global`이 빌드/실행 환경에서 불안정하면 fallback을 둔다.

```text
fallback 1: prefetch.global.L2
fallback 2: __ldcg load-based cold/hot probe
```

단, `__ldcg` fallback은 데이터를 SM load path까지 끌어오므로 `HBM→L2 refill`보다 넓은 경계를 측정한다. 이 경우 `path_semantics`를 다음처럼 기록해야 한다.

```text
path_semantics = l2_refill_proxy_load_based_fallback
```

강제로 `cp_async_bulk_prefetch_l2`를 RTX3090에서 사용하려 하면 benchmark는 실패해야
한다. 조용히 다른 경로로 바꾸면 H100과 RTX3090 비교가 섞이기 때문이다.

---

## 6. 수식

## 6.1 직접 측정되는 값

각 cell에서 NVML로 다음 dynamic energy를 얻는다.

```text
E_reg  = E_dyn(reg_spin)
E_hot  = E_dyn(l2_prefetch_hot)
E_cold = E_dyn(l2_prefetch_cold)
```

reg_spin 차감:

```text
E_hot_delta  = E_hot  - E_reg
E_cold_delta = E_cold - E_reg
```

cold-hot 차분:

```text
E_refill_delta = E_cold_delta - E_hot_delta
               = E_cold - E_hot
```

`reg_spin`은 hot/cold 양쪽에서 동일하면 대수적으로 상쇄된다. 그래도 별도 측정하는 이유는 품질 검증과 fallback 분석 때문이다.

## 6.2 primary measured proxy

```text
k_HBM_to_L2_refill_path
= slope(E_refill_delta vs cold_prefetched_bits)
```

이 값의 의미:

```text
HBM interface
+ SoC-side HBM PHY 이후 memory-ingress path
+ memory controller / scheduling 일부
+ L2 fill / allocation
+ prefetch instruction residual
```

따라서 이 값은 직접 측정 가능한 가장 안전한 headline이다.

## 6.3 secondary derived proxy

문헌 또는 기존 실험에서 다음을 둔다.

```text
k_HBM_interface_assumed
  예: HBM2E 4.3 pJ/bit, HBM3 2.5 pJ/bit 등
  단, boundary assumption 필수

k_L2_fill_proxy
  A.6 l2_write_hit 또는 l2_copy_hit에서 얻은 L2 write/copy path proxy.
  실제 refill allocation의 순수 L2 fill energy가 아니라 subtraction용 민감도 입력이다.
```

그 다음:

```text
k_SoC_HBM_PHY_to_L2_fill_proxy
= k_HBM_to_L2_refill_path
  - k_HBM_interface_assumed
  - k_L2_fill_proxy
```

주의:

```text
이 값은 직접 측정값이 아님.
가정 기반 derived residual임.
음수가 나오면 실제 물리값이 음수라는 뜻이 아니라 boundary 가정/측정 noise/over-subtraction 문제임.
따라서 논문/보고서 headline에는 primary measured proxy와 derived residual을 분리해서 제시해야 함.
```

---

## 7. 요구사양

## 7.1 기능 요구사항

### FR-1. 새 benchmark op 추가

A.6 L2 category 또는 새 A.7 category로 다음 op를 추가한다.

```text
l2_refill_reg_spin
l2_prefetch_hot
l2_prefetch_cold
l2_prefetch_cold_hot_pair
```

권장 category:

```text
category = l2
subcase = refill
```

새 category를 만들면 분석 코드가 복잡해지므로 기존 A.6 `category=l2`에 확장하는 것을 우선 권장한다.

### FR-2. CLI 옵션 추가

`gpu_power_bench.py`에는 benchmark 생성과 실행에 필요한 flag만 추가한다.

```text
--l2-refill-test
--l2-refill-window-mb
--l2-refill-cold-pool-gb
--l2-refill-repeat-inner
--l2-refill-target-energy-j
--l2-refill-k-guess-pj-bit
--l2-refill-line-bytes
--l2-refill-ptx
```

H100 권장 default:

```text
--gpu-profile                  h100_sxm 또는 h100_pcie
--l2-refill-window-mb       16 24 32 40
--l2-refill-cold-pool-gb    4
--l2-refill-repeat-inner    auto
--l2-refill-target-energy-j 10.0
--l2-refill-k-guess-pj-bit  3.0
--l2-refill-line-bytes      128
--l2-refill-ptx             cp_async_bulk_prefetch_l2
```

RTX3090 권장 default:

```text
--gpu-profile                  rtx3090
--l2-refill-window-mb       1 2 3 4
--l2-refill-cold-pool-gb    2
--l2-refill-repeat-inner    auto
--l2-refill-target-energy-j 10.0
--l2-refill-k-guess-pj-bit  6.0
--l2-refill-line-bytes      128
--l2-refill-ptx             prefetch_global_l2
```

`analyze.py`에는 분석 전용 flag를 둔다.

```text
--ncu-counter-csv
--hbm-interface-pj-bit
--hbm-boundary-assumption
```

### FR-3. CSV metadata 추가

메인 CSV row에 다음 컬럼을 추가한다.

```text
category
subcase
op
working_set_bytes
cold_pool_bytes
repeat_inner
prefetch_line_bytes
prefetched_lines
estimated_prefetch_bits
estimated_hbm_refill_bits
estimated_l2_fill_bits
ptx_instruction
kernel_version
path_semantics
pair_id
```

`pair_id`는 hot/cold/reg_spin row를 정확히 묶기 위해 필요하다.

예:

```text
pair_id = W32MB_R65536_line128_policydefault
```

`prefetch_line_bytes`와 `estimated_*_bits`는 요청한 logical bytes 기준이다. 실제 HBM
transaction 수는 GPU/driver/PTX lowering에 따라 달라질 수 있으므로 Nsight Compute
sector counter로 검증해야 한다.

### FR-4. 분석 sidecar 생성

`analyze.py`에서 다음 파일을 생성한다.

```text
*_02_l2_refill_summary.csv
*_02_l2_refill_fit_points.csv
*_02_l2_refill_validation_summary.csv
*_02_l2_refill_skip_reasons.csv
```

### FR-5. plotting 추가

다음 plot을 생성한다.

```text
*_02_l2_refill_cold_hot_fit.png
*_02_l2_refill_path_breakdown.png
*_02_l2_refill_counter_validation.png
*_02_l2_refill_quality_dashboard.png
```

---

## 7.2 비기능 요구사항

### NFR-1. P8 idle baseline 필수

이 실험은 차분 에너지가 작으므로 P0 idle baseline이면 결과가 깨진다. 다음 조건을 만족하지 않으면 hard fail을 권장한다.

```text
P-state = P8 또는 낮은 idle state
SM clock < 500 MHz
static baseline warning 없음
```

### NFR-2. energy signal 크기

NVML 분해능을 고려해 `E_refill_delta`는 최소 수 J 이상이어야 한다.

권장:

```text
E_refill_delta >= 5 J: low-confidence 가능
E_refill_delta >= 10 J: 권장
E_refill_delta >= 30 J: 매우 좋음
```

### NFR-3. counter run 분리

Nsight Compute profiling은 power 측정과 분리한다.

```text
Power run:
  NVML energy 측정

Counter run:
  같은 kernel/shape로 NCU sector/hit-rate 측정

Join:
  analyze.py에서 counter CSV와 power CSV join
```

### NFR-4. cache hint 불확실성 표시

`cp.async.bulk.prefetch`는 hint성 명령이므로 counter validation 없는 결과는 provisional로 표시한다.

```text
headline_source = logical_estimate_PROVISIONAL
```

---

## 8. 구현 단계

## Step 1. TestCases.md 업데이트

A.6에 refill subcase를 추가한다.

```text
A.6.1 L2 resident hit path
A.6.2 HBM-to-L2 refill path
```

새로 추가할 항목:

```text
l2_refill_reg_spin
l2_prefetch_hot
l2_prefetch_cold
```

명시해야 할 문장:

```text
This estimates HBM-to-L2 refill path energy, not pure memory-controller or NoC energy.
```

## Step 2. benchmarks.py 구현

### 2.1 custom CUDA extension builder 추가

파일 내부에 `_build_l2_refill_extension()`을 추가한다.

구성:

```text
- reg_spin_kernel
- prefetch_hot_kernel
- prefetch_cold_kernel
- optional load_based_fallback_kernel
```

### 2.2 BenchSpec 생성 함수 추가

```python
def build_l2_refill(
    op: str,
    working_set_bytes: int,
    cold_pool_bytes: int,
    repeat_inner: int,
    prefetch_line_bytes: int,
    device: str | torch.device = "cuda",
) -> BenchSpec:
    ...
```

`BenchSpec.extra`에는 metadata를 모두 넣는다.

### 2.3 auto repeat 계산

대략적인 반복 횟수:

```text
R_min = target_energy_j / (k_guess_pj_bit × bits_per_pass × 1e-12)
```

예:

```text
W = 32 MiB
bits_per_pass = 32 × 2^20 × 8
k_guess = 3 pJ/bit
target = 10 J

R_min ≈ 12,400
```

단, 실제 runtime이 너무 길면 `window-ms`와 `R`을 조정한다.

## Step 3. gpu_power_bench.py scheduling

`--cases l2` 또는 `--l2-refill-test`가 켜졌을 때 row를 생성한다.

권장 row 순서:

```text
for W in window_mb:
  for R in repeat_inner_candidates:
    reg_spin
    l2_prefetch_hot
    l2_prefetch_cold
```

hot/cold는 같은 pair_id를 가져야 한다.

주의:

```text
hot row는 측정 전에 warmup으로 window를 L2에 올림
cold row는 cold pool에서 offset을 바꿔 이전 L2 contents와 겹치지 않게 함
반복 측정에서는 cold pool offset을 pair/run별로 순환시켜 같은 L2 set/window 재사용을 피함
```

## Step 4. analyze.py 분석 함수 추가

함수:

```python
compute_l2_refill_energy(
  df,
  counter_df=None,
  hbm_interface_pj_bit=None,
  boundary_assumption="",
  l2_summary=None,
)
plot_l2_refill_cold_hot_fit(...)
plot_l2_refill_path_breakdown(...)
plot_l2_refill_counter_validation(...)
plot_l2_refill_quality_dashboard(...)
```

분석 로직:

```text
1. category=l2, subcase=refill row만 선택
2. pair_id별 reg/hot/cold 매칭
3. hot/cold의 iters가 다르면 outer-call당 dynamic energy로 정규화
4. common_iters = min(hot_iters, cold_iters)로 같은 호출 수만 비교
5. E_refill_delta = (E_cold/iters_cold - E_hot/iters_hot) × common_iters
6. x = cold_bits_per_outer × common_iters 또는 counter measured bits
7. WLS fit
8. slope를 pJ/bit로 변환
9. quality gate 적용
10. summary/plot 생성
```

## Step 5. component_validation_report.py 연동

이미 component validation report는 `_02_l2_summary.csv`를 읽는 구조가 있다. 새 sidecar도 반영한다.

추가 coefficient:

```text
k_hbm_to_l2_refill_path
k_soc_hbm_phy_to_l2_fill_proxy
```

주의:

```text
k_hbm_to_l2_refill_path = measured proxy
k_soc_hbm_phy_to_l2_fill_proxy = derived proxy
```

`coverage_matrix`에서는 다음 component를 추가할 수 있다.

```text
hbm_to_l2_refill_path
soc_hbm_phy_to_l2_proxy
```

---

## 9. 분석 및 시각화 계획

## 9.1 `_02_l2_refill_cold_hot_fit.png`

목적:

```text
cold-hot 차분이 prefetched bits에 선형인지 확인
```

구성:

```text
x-axis: cold_prefetched_bits 또는 NCU-measured HBM miss bits
y-axis: E_cold - E_hot [J]
points: W/R별 measured delta
line: WLS fit
annotation: slope pJ/bit, R², CI, n_points
```

해석:

```text
slope > 0
R² high
CI narrow
이면 refill path 추정이 안정적
```

## 9.2 `_02_l2_refill_path_breakdown.png`

목적:

```text
measured proxy와 derived proxy를 구분해서 보여줌
```

구성:

```text
bar 1:
  k_HBM_to_L2_refill_path  [measured proxy]

bar 2 stacked:
  HBM interface assumption
  L2 fill proxy
  residual = SoC-side PHY→L2 fill proxy
```

반드시 표시할 문구:

```text
Residual is assumption-dependent. Not a direct measurement.
```

음수 residual이면:

```text
Boundary mismatch or over-subtraction; do not interpret as negative physical energy.
```

## 9.3 `_02_l2_refill_counter_validation.png`

목적:

```text
logical prefetch bits와 실제 NCU sector가 맞는지 확인
```

구성:

```text
panel A:
  estimated_prefetch_bits vs NCU L2 sectors × 32B × 8

panel B:
  estimated_prefetch_bits vs NCU device/HBM sectors × 32B × 8

panel C:
  hot vs cold의 HBM sector 차이
```

필수 판정:

```text
cold: HBM/device sectors 선형 증가
hot: HBM/device sectors가 작아야 함
```

## 9.4 `_02_l2_refill_quality_dashboard.png`

목적:

```text
오류를 발라내는 최종 dashboard
```

항목:

```text
P8 idle baseline status
E_delta signal size
R²
CI width
n_points
counter attached 여부
cold/hot sector ratio
derived residual sign
```

상태:

```text
PASS
LOW_CONF
FAIL
PROVISIONAL
```

---

## 10. CSV schema

## 10.1 `_02_l2_refill_summary.csv`

```text
gpu
source_csv
status
headline_source
k_hbm_to_l2_refill_path_pj_bit
k_hbm_to_l2_refill_ci_lo
k_hbm_to_l2_refill_ci_hi
r2
n_points
mean_delta_energy_j
hbm_interface_pj_bit_assumption
l2_fill_proxy_pj_bit
k_soc_hbm_phy_to_l2_fill_proxy_pj_bit
derived_proxy_status
boundary_assumption
counter_attached
notes
```

## 10.2 `_02_l2_refill_fit_points.csv`

```text
pair_id
working_set_bytes
repeat_inner
prefetch_line_bytes
estimated_prefetch_bits
E_reg_j
E_hot_j
E_cold_j
E_cold_minus_hot_j
x_bits_used_for_fit
fit_weight
status
skip_reason
```

## 10.3 `_02_l2_refill_validation_summary.csv`

```text
pair_id
ncu_l2_sectors_hot
ncu_l2_sectors_cold
ncu_device_sectors_hot
ncu_device_sectors_cold
logical_to_l2_sector_error_pct
cold_device_sector_linearity_r2
hot_device_sector_leakage_pct
status
```

## 10.4 `_02_l2_refill_skip_reasons.csv`

```text
pair_id
op
reason
details
suggested_fix
```

---

## 11. Quality gate

## 11.1 Hard fail

아래 조건은 결과를 사용하면 안 된다.

```text
P8 idle baseline unavailable
E_cold - E_hot <= 0
k_HBM_to_L2_refill_path <= 0
n_points < 3
R² < 0.80
all points clipped or dyn_energy_j <= 0
hot prefetch가 cold와 비슷한 HBM/device sectors를 발생
```

## 11.2 Low confidence

아래 조건은 plot은 만들되 headline에서 제외하거나 low-confidence 표시한다.

```text
0.80 <= R² < 0.95
E_delta < 5 J
counter CSV 없음
CI width > 50%
derived residual < 0
logical bits와 NCU sector bits 오차 > 30%
```

## 11.3 PASS

권장 조건:

```text
P8 idle baseline clean
E_delta >= 10 J
R² >= 0.95
n_points >= 4
counter attached
hot HBM/device sectors << cold HBM/device sectors
CI width <= 30%
```

---

## 12. 실행 예시

## 12.1 H100 Power run

```bash
# 실험 전 clock reset 권장
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
sudo nvidia-smi -i 0 -rac

./run_bench.sh \
  --device 0 \
  --gpu-profile h100_sxm \
  --cases l2 \
  --l2-refill-test \
  --l2-refill-window-mb 16 24 32 40 \
  --l2-refill-target-energy-j 10 \
  --l2-refill-k-guess-pj-bit 3.0 \
  --l2-refill-cold-pool-gb 4 \
  --l2-refill-ptx cp_async_bulk_prefetch_l2 \
  --pstate-idle-wait 120 \
  --static-seconds 60 \
  --rebaseline-every 10 \
  --window-ms 6000 \
  --tag h100_l2_refill
```

H100 PCIe 80GB이면 `--gpu-profile h100_pcie`를 사용한다.

## 12.2 RTX3090 Power run

```bash
./run_bench.sh \
  --device 0 \
  --gpu-profile rtx3090 \
  --cases l2 \
  --l2-refill-test \
  --l2-refill-window-mb 1 2 3 4 \
  --l2-refill-target-energy-j 10 \
  --l2-refill-k-guess-pj-bit 6.0 \
  --l2-refill-cold-pool-gb 2 \
  --l2-refill-ptx prefetch_global_l2 \
  --pstate-idle-wait 120 \
  --static-seconds 60 \
  --rebaseline-every 10 \
  --window-ms 6000 \
  --tag rtx3090_l2_refill
```

RTX3090 결과는 H100 `cp_async_bulk_prefetch_l2` 결과와 같은 headline으로 합치지 않는다.
Ampere fallback 경로 검증/비교용으로 표시한다.

## 12.3 Counter run

Nsight Compute run은 별도 wrapper가 필요하다. power run과 동시에 하지 않는다.

개념:

```bash
ncu \
  --target-processes all \
  --metrics <L2 sector metrics>,<device/HBM sector metrics> \
  --csv \
  --log-file reports/h100_l2_refill_ncu.csv \
  python3 run_l2_refill_counter.py \
    --device 0 \
    --same-params-as-power-run
```

metric 이름은 Nsight Compute 버전과 GPU에 따라 달라질 수 있으므로 wrapper에서 probe해야 한다. 최소 검증 단위는 다음이다.

```text
L2 sectors
L2 hit rate
sector misses to device/HBM
bytes = sectors × 32B
```

## 12.4 Analysis run

```bash
python3 analyze.py \
  reports/gpu_power_bench_h100_*_h100_l2_refill.csv \
  --ncu-counter-csv reports/h100_l2_refill_ncu.csv \
  --hbm-interface-pj-bit 2.5 \
  --hbm-boundary-assumption "HBM stack + HBM-side PHY/I/O + interposer + SoC-side HBM PHY vicinity"
```

---

## 13. 예상 오류와 대응

## 13.1 cold-hot delta가 음수

가능 원인:

```text
hot prefetch가 실제로 L2 hit가 아님
cold prefetch가 compiler에 의해 제거됨
energy signal이 너무 작음
static baseline이 깨짐
```

대응:

```text
repeat_inner 증가
target_energy_j 증가
counter validation 확인
P8 baseline 확보
fallback load-based kernel과 비교
```

## 13.2 derived residual이 음수

가능 원인:

```text
HBM interface pJ/bit assumption이 너무 큼
L2 fill proxy를 과하게 차감
measured refill path가 low-confidence
boundary mismatch
```

해석:

```text
물리값이 음수라는 뜻 아님.
derived proxy invalid 또는 boundary assumption mismatch.
```

## 13.3 hot prefetch에서 HBM sector가 많음

가능 원인:

```text
window가 L2에 resident하지 않음
window가 너무 큼
cache hint가 무시됨
다른 kernel이 L2를 오염
```

대응:

```text
window size를 16/24MB로 줄임
persisting L2 policy 검토
counter run으로 hit-rate 확인
```

## 13.4 prefetch instruction이 build 실패

대응:

```text
sm_90+ 확인
CUDA toolkit 버전 확인
fallback to prefetch.global.L2
fallback to __ldcg load-based proxy
```

---

## 14. 문서와 plot에서 반드시 피해야 할 표현

금지 표현:

```text
pure NoC energy
pure memory controller energy
pure L2 SRAM bit-cell energy
exact SoC-side PHY energy
```

권장 표현:

```text
HBM-to-L2 refill path energy
SoC-side HBM PHY→L2 fill proxy
assumption-dependent residual
counter-validated proxy
```

---

## 15. 완료 기준

구현 완료는 다음 조건을 만족해야 한다.

```text
1. --l2-refill-test 실행 가능
2. reg_spin/hot/cold row가 pair_id로 매칭됨
3. _02_l2_refill_summary.csv 생성
4. _02_l2_refill_cold_hot_fit.png 생성
5. skip reason CSV 생성
6. counter CSV 없이도 PROVISIONAL 결과 생성
7. counter CSV가 있으면 validation plot 생성
8. derived proxy가 measured proxy와 구분되어 표시됨
9. plot/document 어디에도 NoC 단독 에너지라고 표현하지 않음
```

---

## 16. 자가점검

| 질문 | 답 |
|---|---|
| NVML만으로 7→L2를 직접 측정할 수 있는가? | 아니오 |
| 가장 안전한 직접 proxy는? | HBM-to-L2 refill path pJ/bit |
| 7→L2만 따로 말할 수 있는가? | 차감 기반 derived proxy로만 가능 |
| HBM pJ/bit assumption이 필요한가? | 예 |
| L2 fill proxy가 필요한가? | 예 |
| counter validation이 필요한가? | headline에는 사실상 필요 |
| NoC라고 불러도 되는가? | 아니오 |
| 기존 코드 기반으로 구현 가능한가? | 예. A.6 L2 probe, BenchSpec.extra, analyze sidecar 구조를 확장하면 됨 |

---

## 17. 최종 권장 구현 순서

```text
P0:
  1. TestCases.md에 A.6.2 refill subcase 추가
  2. benchmarks.py에 reg_spin/hot/cold prefetch kernels 추가
  3. gpu_power_bench.py에 CLI/scheduler 추가
  4. analyze.py에 cold-hot delta fit 추가
  5. skip reason CSV와 quality gate 구현

P1:
  6. Nsight Compute counter CSV join
  7. counter validation plot 추가
  8. component_validation_report.py에 새 coefficient 추가

P2:
  9. HBM interface assumption sensitivity plot 추가
  10. load-based fallback과 cp.async prefetch 결과 비교
  11. H100/HBM3, A100/HBM2E cross-GPU 비교 table 추가
```

---

## 18. 최종 해석 문구

보고서에는 다음 문구를 그대로 쓰는 것을 권장한다.

```text
We estimate the HBM-to-L2 refill path energy by subtracting an L2-hot
prefetch baseline from an HBM-cold prefetch measurement and regressing the
dynamic-energy delta against the prefetched bits. The resulting coefficient is
a measured proxy for the refill path into L2. A narrower SoC-side HBM
PHY-to-L2 fill coefficient is reported only as an assumption-dependent
residual after subtracting the assumed HBM interface energy and an L2 fill
proxy. It should not be interpreted as pure NoC, pure memory-controller, or
pure SRAM bit-cell energy.
```

한국어:

```text
HBM-cold prefetch 측정값에서 L2-hot prefetch baseline을 차감하고,
그 dynamic-energy delta를 prefetch된 bit 수에 대해 회귀하여
HBM-to-L2 refill path pJ/bit를 추정한다. SoC-side HBM PHY→L2 fill 값은
문헌 HBM interface energy와 L2 fill proxy를 추가로 차감한 가정 의존 residual이며,
순수 NoC, 순수 memory controller, 순수 SRAM bit-cell energy로 해석하지 않는다.
```
