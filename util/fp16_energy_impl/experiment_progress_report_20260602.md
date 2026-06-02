# FP16 energy experiment results and design delta report

작성일: 2026-06-02
대상 디렉터리: `util/fp16_energy_impl`
현재 측정 완료 GPU: NVIDIA GeForce RTX 3090, GA102, sm86
문서 목적: 지금까지 얻은 RTX 3090 FP16 matmul pJ/bit 결과와 최초 설계 대비 변경점을 기록한다.

## 1. 현재 결론

아직 최종 publishable strict pJ/bit 값은 없다. RTX 3090에서 여러 diagnostic sweep과 strict-like pipeline을 수행했지만, 현재 장비에서 Nsight Compute performance counter 접근이 `ERR_NVGPUCTRPERM`으로 막혀 no-L2/global-memory 조건과 Tensor Core HMMA activity를 hardware counter로 증명하지 못했다. 따라서 아래 pJ/bit 값은 모두 `baseline-subtracted incremental FP16 compute energy estimate`이며, 최종 A100/H100/RTX3090 비교값이 아니다.

현재까지 가장 의미 있는 RTX 3090 숫자는 다음 두 가지다.

| 상태 | 결과 디렉터리 | 선택점 | TFLOPS | `matmul_input_pj_per_bit` | 해석 |
|---|---|---:|---:|---:|---|
| Strict-like calibrated diagnostic | `results/strict_fp16_launch_shape_rtx3090_20260602_115550` | `threads=256`, `blocks/SM=1`, `threads/SM=256` | 159.148 | `0.30846 +/- 0.02532 pJ/bit` | Quality gate는 통과했지만 NCU no-L2/Tensor evidence가 없어 final strict claim은 아님 |
| Latest no-NCU diagnostic | `results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100` | `threads=256`, `blocks/SM=1`, `threads/SM=256` | 159.825 | `0.18327 +/- 0.10838 pJ/bit` | `--diagnostic-no-ncu` 성격의 결과이며 energy source/measurement resolution gate가 실패함 |

`strict_fp16_launch_shape_rtx3090_20260602_115550` 안에서는 더 낮은 pJ/bit 행도 있었다. 예를 들어 `threads=256`, `blocks/SM=8`, `threads/SM=2048`은 `0.07423 +/- 0.01586 pJ/bit`로 보였지만, target selection은 "첫 saturation point" 기준이므로 선택하지 않았다. 이 값은 더 많은 resident work가 fixed overhead를 희석한 diagnostic 후보로만 본다.

`diagnostic_fp16_launch_shape_rtx3090_20260602_125100`의 최저 관측 행은 `threads=128`, `blocks/SM=4`, `threads/SM=512`에서 `0.11432 +/- 0.14551 pJ/bit`였지만 `valid_no_l2_count=2/10`이라 target 후보가 아니다.

## 2. Claim boundary

이 실험의 pJ/bit는 Tensor Core 안의 물리적 FP16 arithmetic unit만의 절대 에너지가 아니다. 현재 정의는 다음과 같다.

```text
pJ/bit = (E_test - P_baseline * elapsed_test) / logical_FP16_input_bits
```

Tensor Core matmul 기준 denominator는 logical `m16n16k16`이다. 구현은 `mma.sync.aligned.m16n8k16` 두 번으로 N 방향 16 columns를 만들며, logical MMA 1회당 FP16 input bit는 다음처럼 계산한다.

```text
A bits + B bits = (16*16 + 16*16) * 16 = 8192 bit/logical MMA
FLOPs          = 2 * 16 * 16 * 16 = 8192 FLOP/logical MMA
```

따라서 문서와 figure의 `matmul_input_pj_per_bit`는 "logical `m16n16k16` FP16 input bit당 baseline-subtracted board/NVML energy estimate"로 읽어야 한다. L2/DRAM bit energy나 register-file bit energy를 직접 잰 값이 아니다.

## 3. 최초 대비 달라진 점

| 항목 | 최초 상태 | 현재 상태 |
|---|---|---|
| Baseline | `baseline_nop` 기반의 generic loop baseline | Tensor Core 전용 `tensor_baseline_mov` structural no-memory warp-sync baseline 사용 |
| Matrix shape | 초기 문서와 legacy 결과에서 denominator가 명확하지 않거나 analyzer fallback 가능 | logical `m16n16k16`을 benchmark JSON에 직접 기록, 8192 input bits와 8192 FLOPs를 quality gate에서 확인 |
| Thread 선택 | 고정 thread 후보 또는 coarse thread sweep | `threads/block = 32,64,128,256`과 `blocks/SM = 1,2,4,8` launch-shape sweep |
| X축 기준 | 단순 thread count | `threads_per_sm = threads/block * blocks/SM`을 주요 x축으로 사용 |
| Utilization 기준 | `nvidia-smi dmon`의 coarse SM utilization 중심 | Tensor Core matmul은 `tensor_model_utilization_pct_mean`을 target selection에 우선 사용, SM util은 보조 plot |
| `blocks_per_sm=8` 가정 | 고정값처럼 쓰일 수 있었음 | 별도 sweep으로 검증, 현재 target은 `blocks/SM=1`의 첫 saturation point |
| L2 조건 | 결과 metadata의 `expected_l2_touch=false`에 가까운 software-side 판정 | strict final에서는 NCU MemoryWorkloadAnalysis no-L2/global-memory validation과 Tensor activity evidence 요구 |
| Memory provenance | `suppress_output_store`와 `memory_bytes_estimate`에서 analyzer가 주로 간접 추론 | benchmark JSON이 `timed_kernel_global_input_loads`, `timed_kernel_global_output_stores`, `timed_kernel_has_intended_global_memory`를 직접 기록하고, analyzer/audit는 test와 baseline 양쪽 모두 no-memory일 때만 no-L2 후보로 인정 |
| NCU context matching | NCU row가 kernel/thread 후보와 느슨하게 매칭될 여지가 있었고, context 비교는 주로 `blocks_per_sm`, `unroll`, `suppress_output_store` 중심 | quality gate와 strict audit가 NCU validation row의 `threads`, `blocks_per_sm`, `unroll`, `suppress_output_store`를 selected measurement row와 모두 비교. 같은 kernel/thread/block에 여러 NCU row가 있으면 exact `unroll`/`suppress_output_store` row를 우선 선택. `threads`가 누락된 NCU report는 fallback matching되어도 strict target/audit에서 실패 |
| Energy source | `nvidia-smi` power trace fallback 중심 legacy 결과 포함 | 최종 비교는 `NVML total energy counter`, `measurement_grade=strict_nvml_counter` selected target만 허용 |
| Negative pJ/bit 처리 | 초기에는 음수 행이 result table에 섞일 수 있었음 | `all_runs_no_valid`, `not_valid_no_l2_candidate`, quality gate failure로 분리 |
| Architecture scope | 일반 CUDA 실행 설명 위주 | A100 sm80/GA100, RTX 3090 sm86/GA102, H100 sm90/GH100 metadata 검증 추가 |
| Architecture peak model | architecture별 단일 dense Tensor peak 기준 | `tensor_mma_f16acc`/`tensor_mma_f32acc` accumulator mode를 구분. RTX 3090/GA102는 f32acc peak를 f16acc의 절반 기준으로 normalize |
| H100 경로 | 명시 부족 | 현재는 공통 warp-level HMMA path 비교이며 H100 WGMMA/TMA 측정은 아님을 명시 |
| 설치/실행 | 수동 toolchain 의존 | `scripts/install_gpu_toolchain.sh`와 GPU별 env/run script 생성 |
| Pipeline | build/run/analyze가 분리 | `run_strict_fp16_pipeline.sh`가 preflight, NCU permission probe, calibration, sweep, analyze, resource audit, quality gate까지 orchestration |
| Register evidence | 질문 시 별도 확인 필요 | `summarize_kernel_resources.py`와 resource audit로 ptxas register/spill evidence 기록 |
| Diagnostic mode | strict 실패와 diagnostic 구분이 약함 | `--diagnostic-no-ncu`는 NCU를 명시적으로 skip하고 final claim에서 제외 |
| Tensor model sanity | model utilization fallback 값이 100%를 넘는 diagnostic row도 analyzer selected로 보일 수 있었음 | `tensor_model_utilization_pct_mean > 105%` row는 analyzer `selected_optimal` 후보와 quality gate target 후보에서 제외 |

최근 코드 검토에서 확인한 설계상 취약점은 NCU validation evidence가 selected launch context와 정확히 같은지 충분히 강제하지 않는 부분이었다. 현재는 `quality_gate.py`, `audit_strict_results.py`, `smoke_strict_pipeline.py`, README를 수정해 NCU row의 `threads`, `blocks_per_sm`, `unroll`, `suppress_output_store` context가 빠지거나 measurement row와 다르면 final strict target으로 채택되지 않도록 했다. 또한 중복 NCU row가 있을 때 행 순서가 아니라 exact context score로 row를 고르도록 바꿨다.

## 4. 결과 히스토리

| 날짜/디렉터리 | Sweep/조건 | 선택 또는 대표점 | pJ/bit | 비고 |
|---|---|---:|---:|---|
| `results/fp16_matmul_pjbit_gpu0_default_clock_cuda132_20260528_1331` | 초기 single matmul, `tensor_mma_f16acc` vs `baseline_nop` | 5 repeats | `0.22637 +/- 0.15864` | TFLOPS 158.08, valid 5/5, legacy diagnostic |
| `results/fp16_matmul_thread_sweep_rtx3090_20260528` | 초기 thread sweep | `threads=256` | `0.10961 +/- 0.01497` | valid no-L2 8/10, NCU 없음. 음수 pJ/bit 행은 invalid로 분리 필요 |
| `results/fp16_matmul_thread_sweep_fine_m16n16_smutil_rtx3090_20260528` | 낮은 thread 후보 포함, dmon SM util | `threads=64`, `threads/SM=512` | `0.25140 +/- 0.01425` | 평균 SM util 100%, `measurement_grade=power_trace_fallback`, quality pass 아님 |
| `results/fp16_launch_shape_warpsync_rtx3090_20260602_direct` | foreground direct diagnostic, 1 repeat | `threads=64`, `blocks/SM=2`, `threads/SM=128` | `0.35326` | power/sm util/NCU 없음, direct NVML delta만 사용. 105% 초과 Tensor model utilization 후보 9개 제외 후 analyzer diagnostic 선택점 |
| `results/fp16_work_slope_bar_repeat30_rtx3090_20260601` | work amount slope diagnostic | `threads=64`, `blocks/SM=8` | `0.20249` slope | work sweep에서 positive incremental-energy slope 확인 |
| `results/fp16_work_slope_bar_repeat30_rtx3090_20260601` | work amount slope diagnostic | `threads=128`, `blocks/SM=8` | `0.17906` slope | work sweep에서 positive incremental-energy slope 확인 |
| `results/strict_fp16_launch_shape_rtx3090_20260602_115550` | calibrated launch-shape sweep | `threads=256`, `blocks/SM=1`, `threads/SM=256` | `0.30846 +/- 0.02532` | quality gate target은 있었지만 NCU hardware validation이 없어 strict final 아님 |
| `results/strict_fp16_launch_shape_rtx3090_20260602_124900` | strict pipeline with NCU permission probe | 없음 | 없음 | `ERR_NVGPUCTRPERM`으로 calibration/sweep 전 중단 |
| `results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100` | latest no-NCU diagnostic launch-shape sweep | `threads=256`, `blocks/SM=1`, `threads/SM=256` | `0.18327 +/- 0.10838` | quality pass 0/176, target pass 0/16, diagnostic only |

## 5. Thread/SM sweep에서 배운 점

현재 launch-shape sweep range는 다음과 같다.

```text
threads/block: 32, 64, 128, 256
blocks/SM:     1, 2, 4, 8
threads/SM:    32, 64, 128, 256, 512, 1024, 2048
```

Tensor Core throughput은 RTX 3090에서 대략 `threads/SM=256` 근처부터 saturation에 도달한다. Strict-like calibrated sweep과 latest no-NCU diagnostic에서는 `threads=256`, `blocks/SM=1`, `threads/SM=256`이 첫 saturation point였다. Direct foreground diagnostic은 clock/SM telemetry가 없어 TFLOPS/reference peak fallback만 사용했으며, 이 fallback이 105%를 넘는 후보는 model/clock/accounting mismatch 가능성이 있으므로 selection에서 제외했다. 그 결과 direct diagnostic analyzer 선택점은 `threads=64`, `blocks/SM=2`, `threads/SM=128`로 바뀌었다.

따라서 target selection은 "가장 낮은 pJ/bit"가 아니라 "quality gate를 통과하고 Tensor model sanity를 만족한 행 중 Tensor model utilization이 충분히 포화되는 첫 point"를 우선한다. 이 기준을 쓰는 이유는 너무 많은 resident threads/blocks가 fixed overhead를 희석해 pJ/bit를 낮게 보이게 할 수 있지만, 동시에 baseline subtraction과 L2/global-memory validation이 더 취약해질 수 있기 때문이다.

SM utilization figure의 x축은 `threads_per_sm`이고, y축은 SM utilization 또는 Tensor model utilization이다. pJ/bit는 별도 figure 또는 marker/annotation으로 같이 봐야 한다. 현재 주요 figure는 다음 위치에 있다.

```text
results/strict_fp16_launch_shape_rtx3090_20260602_115550/figures/thread_sweep_tensor_mma_f16acc_vs_tensor_baseline_mov.png
results/strict_fp16_launch_shape_rtx3090_20260602_115550/figures/thread_sweep_pjbit_tensor_mma_f16acc_vs_tensor_baseline_mov.png
results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100/figures/thread_sweep_tensor_mma_f16acc_vs_tensor_baseline_mov.png
results/diagnostic_fp16_launch_shape_rtx3090_20260602_125100/figures/thread_sweep_pjbit_tensor_mma_f16acc_vs_tensor_baseline_mov.png
```

## 6. Register/spill evidence

현재 resource audit에서 selected Tensor Core test와 baseline은 모두 ptxas 기준 register pressure가 낮다.

| Kernel | Registers/thread | Stack | Spill |
|---|---:|---:|---:|
| `tensor_mma_f16acc` | 14 | 0 B | 0 B |
| `tensor_baseline_mov` | 14 | 0 B | 0 B |

즉 현재 pJ/bit 불확실성의 주 원인은 register spill이 아니라 NCU counter 권한, energy measurement resolution, baseline subtraction 안정성이다.

## 7. 아직 strict final이 아닌 이유

1. RTX 3090 장비에서 Nsight Compute performance counter 접근이 막혀 `ERR_NVGPUCTRPERM`이 발생했다.
2. 이 때문에 no-L2/global-memory traffic과 Tensor Core HMMA activity를 hardware counter로 확인하지 못했다.
3. latest diagnostic run은 `measurement_grade=mixed_or_unavailable`이고 baseline elapsed time이 quality threshold보다 짧아 measurement resolution gate를 통과하지 못했다.
4. A100과 H100 strict run은 아직 완료되지 않았다.
5. H100의 경우 현재 kernel은 common HMMA path이고 WGMMA path가 아니다.

`results/architecture_compare_rtx3090_readiness_20260602/`에는 현재 RTX 3090 결과 3개를 architecture comparison tool로 묶은 diagnostic readiness 산출물을 추가했다. 새 `architecture_comparison_summary.json`은 `publishable=false`, `required_strict_pass_count=0/3`, `required_missing_architectures=ga100,gh100`, `required_diagnostic_only_architectures=ga102`로 기록한다. 즉 RTX 3090 결과도 NCU evidence가 없으므로 A100/H100/RTX3090 최종 비교에서는 strict-pass로 세지 않는다.

## 8. 다음 실행 기준

최종 claim에 사용할 결과는 아래 조건을 모두 만족해야 한다.

1. `run_strict_fp16_pipeline.sh` strict mode로 완료되어야 한다.
2. `ncu_permission_probe/ncu_permission_probe.json`에서 permission probe가 pass여야 한다.
3. `quality_gate.py --require-ncu --require-ncu-tensor-activity` 결과 `selected_targets`가 있어야 한다.
4. selected target의 `measurement_grade`는 `strict_nvml_counter`여야 한다.
5. selected target의 baseline은 `tensor_baseline_mov`여야 한다.
6. selected target의 denominator는 logical `m16n16k16`, `8192` input bits/logical MMA여야 한다.
7. selected target은 `timed_kernel_memory_provenance_metadata_all=true`이고 test/baseline의 intended global-memory count가 모두 0이어야 한다.
8. resource audit에서 selected test/baseline 모두 stack/spill이 없어야 한다.
9. A100/H100/RTX3090 비교는 세 GPU 결과가 모두 audit을 통과한 뒤 `postprocess_strict_architectures.sh`로 묶어야 한다.

권한이 없는 local smoke나 pipeline sanity check는 계속 `--diagnostic-no-ncu`로 실행할 수 있다. 단, 이 결과는 README와 report에서 diagnostic-only로 표시하고 최종 pJ/bit 표에는 넣지 않는다.
