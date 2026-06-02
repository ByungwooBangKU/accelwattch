# FP16 Energy Experiment Results

작성일: 2026-06-02
대상 GPU: NVIDIA GeForce RTX 3090, GA102, sm86
상세 보고서: `../experiment_progress_report_20260602.md`

이 디렉터리는 현재 `fp16_energy_impl` README와 분석 스크립트가 참조하는 RTX 3090 FP16 matmul energy 결과를 보관한다. 지금까지의 결과는 모두 diagnostic 또는 strict-like diagnostic이며, 아직 최종 publishable strict pJ/bit 값은 없다. RTX 3090 장비에서 Nsight Compute performance counter 접근이 `ERR_NVGPUCTRPERM`으로 막혀 no-L2/global-memory 조건과 Tensor Core HMMA activity를 hardware counter로 증명하지 못했기 때문이다.

## Current result summary

| 상태 | 결과 디렉터리 | 선택점 | Utilization | TFLOPS | `matmul_input_pj_per_bit` | 비고 |
|---|---|---:|---:|---:|---:|---|
| Strict-like calibrated diagnostic | `strict_fp16_launch_shape_rtx3090_20260602_115550/` | `threads=256`, `blocks/SM=1`, `threads/SM=256` | Tensor model 98.34%, SM 93.75% | 159.148 | `0.30846 +/- 0.02532` | NVML energy counter와 structural baseline은 통과했지만 NCU evidence가 없어 final strict 아님 |
| Latest no-NCU diagnostic | `diagnostic_fp16_launch_shape_rtx3090_20260602_125100/` | `threads=256`, `blocks/SM=1`, `threads/SM=256` | Tensor model 98.37%, SM 99.40% | 159.825 | `0.18327 +/- 0.10838` | NCU를 명시적으로 skip한 diagnostic run. `quality_gate_summary.json`의 selected target은 0 |
| Direct foreground diagnostic | `fp16_launch_shape_warpsync_rtx3090_20260602_direct/` | `threads=64`, `blocks/SM=2`, `threads/SM=128` | Tensor model 104.86%, SM telemetry 없음 | 149.237 | `0.35326` | foreground 1-repeat diagnostic. power/SM/NCU evidence 없음 |
| Initial Tensor Core matmul | `fp16_matmul_pjbit_gpu0_default_clock_cuda132_20260528_1331/` | 5 repeats | GPU util only | 158.08 | `0.22637 +/- 0.15864` | legacy `baseline_nop` 기반 diagnostic |
| Initial thread sweep | `fp16_matmul_thread_sweep_rtx3090_20260528/` | `threads=256` | GPU util 99.92% | 156.019 | `0.10961 +/- 0.01497` | NCU 없음, legacy baseline |
| Fine SM-util thread sweep | `fp16_matmul_thread_sweep_fine_m16n16_smutil_rtx3090_20260528/` | `threads=64`, `threads/SM=512` | SM util 100% | 159.111 | `0.25140 +/- 0.01425` | `nvidia-smi dmon` SM util 기반 legacy diagnostic |

해석 기준은 상세 보고서의 claim boundary를 따른다. 특히 `matmul_input_pj_per_bit`는 logical `m16n16k16` FP16 input bit당 baseline-subtracted board/NVML energy estimate이며, Tensor Core 내부 arithmetic unit만의 절대 에너지가 아니다.

## Sweep range used for the current launch-shape runs

```text
threads/block: 32, 64, 128, 256
blocks/SM:     1, 2, 4, 8
threads/SM:    32, 64, 128, 256, 512, 1024, 2048
logical shape: m16n16k16
input bits:    8192 bit/logical MMA
FLOPs:         8192 FLOP/logical MMA
```

Target selection은 최저 pJ/bit만 고르지 않는다. 현재 기준은 quality gate를 통과하고 Tensor model utilization sanity를 만족하는 행 중 첫 saturation point를 고르는 것이다. RTX 3090 launch-shape sweep에서는 `threads=256`, `blocks/SM=1`, `threads/SM=256`이 그 점으로 선택되었다.

## Architecture comparison readiness

`architecture_compare_rtx3090_readiness_20260602/`는 현재 RTX 3090 결과들을 architecture comparison tool로 묶은 readiness 산출물이다. `architecture_comparison_summary.json` 기준:

```text
publishable: false
required_strict_pass_count: 0/3
required_missing_architectures: ga100, gh100
required_diagnostic_only_architectures: ga102
```

즉 A100/H100/RTX3090 최종 비교에는 아직 사용할 수 없다. 세 GPU 모두에서 strict pipeline을 완료하고 `strict_result_audit.csv`의 `audit_pass=true` row가 있어야 architecture-level pJ/bit 비교가 publishable로 바뀐다.

## What changed from the initial design

최초 대비 주요 변화는 `../experiment_progress_report_20260602.md`에 정리되어 있다. 핵심은 다음과 같다.

| 항목 | 최초 상태 | 현재 상태 |
|---|---|---|
| Baseline | generic `baseline_nop` 중심 | Tensor Core용 `tensor_baseline_mov` structural no-memory baseline |
| Matrix size | denominator가 legacy analyzer fallback에 의존 가능 | benchmark JSON이 logical `m16n16k16`, 8192 input bits, 8192 FLOPs를 직접 기록 |
| Thread search | fixed/coarse thread sweep | `threads/block`와 `blocks/SM`을 함께 sweep해 `threads/SM` 기준으로 saturation point 선택 |
| L2 exclusion | software-side expected L2 flag 중심 | final strict에서는 NCU no-L2/global-memory validation과 Tensor activity evidence 필수 |
| Measurement source | power trace fallback 포함 | final strict는 NVML total energy counter와 `measurement_grade=strict_nvml_counter`만 허용 |
| Audit | selected row와 validation/resource evidence의 context mismatch 가능 | NCU/resource row의 `threads`, `blocks_per_sm`, `unroll`, `suppress_output_store` context를 selected measurement와 exact match하도록 강화 |
| Visualization | diagnostic과 strict point 혼동 가능 | strict report/dashboard에서 publishable strict, diagnostic, rejected marker를 분리 |
| Multi-GPU comparison | quality target만으로 coverage처럼 보일 수 있음 | architecture compare는 strict audit pass row만 publishable best로 사용 |

## Remaining blocker

RTX 3090 local strict run은 `ncu_permission_probe/ncu_permission_probe.json`에서 `permission_probe_pass=false`로 중단되었다. 이는 package 설치 문제가 아니라 NVIDIA performance counter 권한 문제다. 최종 pJ/bit claim을 만들려면 권한 있는 환경에서 strict mode를 실행해야 한다.
