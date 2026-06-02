# GPU FP16 Incremental Compute Energy Benchmark

이 구현은 `GPU FP16 순수 연산 에너지 추출 실험 설계서`의 P0/P1 범위를 코드로 옮긴 것이다. 목표는 GPU 내부 FP16 unit만의 물리적 절대 에너지를 직접 측정하는 것이 아니라, register-resident 또는 fragment-resident FP16 compute workload에서 관찰되는 **baseline-subtracted incremental FP16 compute energy estimate**를 산출하는 것이다.

## 1. 구성

| 경로 | 역할 |
|---|---|
| `src/fp16_energy_bench.cu` | CUDA microbenchmark binary. FP16 half2, Tensor Core MMA, baseline, memory policy kernel 포함 |
| `configs/primary_matrix.json` | P0 primary FP16 실험 matrix |
| `configs/fp16_matmul_pjbit_matrix.json` | Tensor Core FP16 matmul logical pJ/bit 전용 matrix |
| `configs/fp16_matmul_thread_sweep.json` | L2/global output store를 억제한 Tensor Core thread-count sweep matrix |
| `configs/fp16_matmul_thread_sweep_fine.json` | 낮은 thread 후보까지 포함한 Tensor Core fine thread-count sweep matrix |
| `configs/fp16_matmul_thread_sweep_low_append.json` | 기존 fine sweep 결과 디렉터리에 32/64/96/128 후보를 append하는 matrix |
| `configs/fp16_matmul_launch_shape_sweep.json` | `threads/block`와 `blocks/SM`을 함께 훑어 `blocks_per_sm=8` 고정 가정을 검증하는 launch-shape sweep matrix |
| `configs/fp16_matmul_work_slope_mov.json` | 같은 thread 조건에서 unroll/work amount를 바꾸는 `tensor_baseline_mov` slope diagnostic matrix |
| `configs/p1_memory_policy_matrix.json` | P1 memory/cache policy 보정용 matrix |
| `scripts/run_experiment.py` | benchmark 실행 + `nvidia-smi` power/clock/temp 및 dmon SM utilization logging |
| `scripts/analyze_results.py` | NVML energy counter 우선 분석, power trace fallback, baseline subtraction, pJ/FLOP 계산, CSV/시각화 생성 |
| `scripts/architecture_models.py` | A100/H100/RTX3090 Tensor Core peak/occupancy normalization constants와 model sanity CSV/figure 생성 |
| `scripts/quality_gate.py` | 결과 채택 전 energy source, valid no-L2 반복 수, clock 안정성, SM utilization 포화 여부를 gate |
| `scripts/audit_strict_results.py` | A100/H100/RTX3090 strict 결과 디렉터리가 최종 비교 조건을 모두 만족하는지 일괄 audit |
| `scripts/report_strict_results.py` | strict audit/architecture compare 산출물을 최종 검토용 Markdown report와 dashboard figure로 요약 |
| `scripts/smoke_strict_pipeline.py` | GPU 없이 strict audit/report target-selection invariant를 synthetic fixture로 회귀 검증 |
| `scripts/generate_work_slope_matrix.py` | `quality_gate_summary.json`의 selected target과 같은 launch context로 work-slope matrix 생성 |
| `scripts/install_gpu_toolchain.sh` | GPU별 CUDA/Nsight Compute conda toolchain 설치와 strict pipeline용 env file 생성 |
| `scripts/postprocess_strict_architectures.sh` | A100/H100/RTX3090 결과 디렉터리를 받아 audit/compare/report/visualization을 한 번에 생성 |
| `scripts/summarize_kernel_resources.py` | ptxas register/spill evidence와 thread별 static occupancy model 산출 |
| `scripts/run_strict_fp16_pipeline.sh` | build/env/sweep/analyze/NCU/strict quality gate를 한 번에 실행하는 A100/H100/RTX3090용 pipeline |
| `scripts/run_rtx3090_strict_ncu_after_counter_enable.sh` | RTX 3090 WSL/native Linux에서 counter 권한 확인 후 strict NCU pipeline을 실행하는 helper |
| `scripts/run_strict_architecture_suite.sh` | 여러 GPU spec의 strict pipeline을 순차 실행하고 audit/compare/report까지 자동 생성 |
| `scripts/preflight_strict_architecture_suite.py` | suite 실행 전 tool, GPU target, active compute process 상태를 JSON/CSV로 검사 |
| `scripts/probe_ncu_permissions.py` | 긴 sweep 전에 Nsight Compute performance-counter 권한을 짧은 HMMA profile로 확인 |
| `scripts/write_strict_suite_summary.py` | suite-level publishability, energy policy, preflight/run/postprocess provenance를 JSON으로 고정 |
| `scripts/compare_architectures.py` | A100/H100/RTX3090 등 여러 결과 디렉터리의 FP16 energy/throughput/thread-sweep 비교 시각화. `--audit-dir`가 있으면 strict audit pass row만 publishable best로 사용 |
| `scripts/calibrate_matrix.py` | GPU별 timed duration을 맞추기 위해 matrix의 per-role `repeats`를 probe 또는 기존 `summary.csv` 기준으로 보정 |
| `scripts/verify_architecture.py` | runtime preflight JSON의 compute capability/chip/common-HMMA metadata가 요청한 CUDA architecture와 맞는지 검증 |
| `scripts/ncu_validate.sh` | Nsight Compute validation run 예시 |
| `scripts/ncu_validate_no_l2_thread_sweep.sh` | thread sweep 후보의 no-L2/global-memory validation run 예시 |
| `scripts/validate_ncu_reports.py` | Nsight Compute text report에서 HMMA/no-L2/local-spill/tensor-activity evidence를 자동 판정하고 memory counter를 normalized bytes로 요약 |
| `scripts/lock_clocks.sh` | GPU clock lock helper |
| `scripts/reset_clocks.sh` | GPU clock reset helper |
| `scripts/query_env.sh` | 실험 환경 metadata 수집 |
| `experiment_progress_report_20260602.md` | 지금까지 RTX 3090 결과, strict-final 미완료 사유, 최초 설계 대비 변경점 요약 |

## 2. 구현된 kernel

| 우선순위 | Kernel | 목적 | Memory policy |
|---|---|---|---|
| P0 | `fp16_half2` | CUDA core 기반 `half2` FMA 반복 | timed loop 내부 global/shared memory 접근 없음 |
| P0 | `baseline_nop` | loop/no-FP16 baseline | 동일 launch/loop 구조, FP16 연산 없음 |
| P0 | `baseline_regmove` | register/integer movement baseline | FP16 연산 없음, integer/register overhead 참고 |
| P0 | `tensor_mma_f16acc` | logical `m16n16k16` FP16 input + FP16 accumulate | timed loop 내부 register operand 기반 |
| P0 | `tensor_mma_f32acc` | logical `m16n16k16` FP16 input + FP32 accumulate | timed loop 내부 register operand 기반 |
| P0 | `tensor_baseline_mov`, `tensor_baseline_f32` | Tensor kernel용 strict baseline | MMA 없음. `tensor_baseline_mov`는 f16acc용 no-memory warp-sync baseline |
| P0 | `tensor_baseline_u32` | Tensor kernel용 ALU diagnostic baseline | MMA 없음. integer ALU가 커서 최종 f16acc subtraction에는 사용하지 않음 |
| P1 | `memory_default`, `memory_cg`, `memory_cs` | residual L1/L2/DRAM 보정 및 cache policy sanity check | default / `.cg` / `.cs`만 제한 반영 |

`fp16_half2`의 연산량 계산은 다음과 같다.

```text
N_FP16_ops = blocks × threads × repeats × iters × unroll × 4 half2-FMA × 4 FLOP/half2-FMA
```

Tensor Core kernel은 Ampere/Hopper에서 지원되는 `mma.sync.aligned.m16n8k16` 두 번으로 logical `m16n16k16` tile 하나를 구성한다. 따라서 logical MMA 1회당 `2 × 16 × 16 × 16 = 8192` FP16 FLOP으로 계산한다.

```text
N_FP16_ops = warps × repeats × iters × unroll × 8192
```

benchmark JSON은 `schema_version=fp16-energy-bench-v2`와 `schema_features`를 기록하고, 이 denominator를 `mma_logical_shape`, `mma_logical_count_estimate`, `mma_input_bits_per_logical_mma`, `mma_flops_per_logical_mma`로 직접 기록한다. CMake build에서는 `bench_build_git_commit`도 함께 남긴다. 또한 `timed_kernel_global_input_loads`, `timed_kernel_global_output_stores`, `timed_kernel_has_intended_global_memory`를 기록해 timed kernel이 global/L2 traffic을 의도하는지 test/baseline 양쪽에서 추적한다. 분석/quality gate는 이 값들이 logical `m16n16k16` 기준의 8192 input bits 및 8192 FLOP과 맞는지와 required `schema_features`에 `timed_kernel_memory_provenance`가 포함되는지 확인한다. 과거 JSON처럼 analyzer가 fallback formula로 denominator 또는 memory provenance를 재계산한 값은 diagnostic table에는 남기지만 strict target/audit는 통과시키지 않는다.

## 3. Build

```bash
cd fp16_energy_impl
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
cmake --build build -j
```

대상 GPU별 architecture는 일반적으로 다음과 같이 둘 수 있다.

| GPU | CUDA architecture |
|---|---:|
| A100 SXM4 | 80 |
| RTX 3090 | 86 |
| H100 SXM5 | 90 |

한 장비에서 하나의 GPU만 측정할 경우 빌드 시간을 줄이기 위해 해당 architecture만 지정해도 된다.

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="90"
```

### 3.1 GPU별 toolchain / Nsight Compute 설치

Strict pipeline은 `cmake`, `nvcc`, `ncu`, `python/matplotlib`, `nvidia-smi`가 필요하다. 서버의 기본 PATH가 불안정하거나 CUDA toolkit이 여러 개 섞여 있으면 아래 installer로 user-space conda/mamba 환경을 만든다. 이 스크립트는 NVIDIA driver를 설치하거나 업데이트하지 않는다. Driver와 NVML/libcuda는 host 또는 cluster image에 이미 있어야 한다.

기본 설치 CUDA toolkit은 12.1이다. A100/H100/RTX 3090 공통 HMMA benchmark에는 충분하고, 새 toolkit이 오래된 driver보다 앞서서 `CUDA driver version is insufficient for CUDA runtime version`가 나는 상황을 피하기 쉽다. 필요하면 `--cuda-version 12.4`처럼 override한다.

```bash
cd /home/bang001/gpupowermodeling/accelwattch/util/fp16_energy_impl

# RTX 3090 / GA102 / sm86
./scripts/install_gpu_toolchain.sh --gpu-kind rtx3090

# A100 / GA100 / sm80
./scripts/install_gpu_toolchain.sh --gpu-kind a100

# H100 / GH100 / sm90
./scripts/install_gpu_toolchain.sh --gpu-kind h100
```

설치되는 주요 package는 `cuda-nvcc`, `cuda-cudart-dev`, `cuda-cudart-static_linux-64`, `cuda-cccl`, `cuda-libraries-dev`, `nsight-compute`, `cmake`, `ninja`, `python`, `matplotlib`이다. 설치 후 스크립트는 다음 두 파일을 만든다.

| 파일 | 내용 |
|---|---|
| `env/toolchain_<gpu-kind>_sm<arch>_cuda<version>.sh` | `CMAKE_BIN`, `NVCC_BIN`, `NCU_BIN`, `PYTHON_BIN`, `NVIDIA_SMI_BIN`, `CMAKE_CUDA_FLAGS`, `CPATH`, `LIBRARY_PATH` export |
| `env/run_strict_<gpu-kind>_sm<arch>_cuda<version>.sh` | launch-shape strict pipeline 실행 예시 |

예를 들어 RTX 3090에서 설치 후 launch-shape strict pipeline을 실행하려면:

```bash
source env/toolchain_rtx3090_sm86_cuda121.sh

./scripts/run_strict_fp16_pipeline.sh \
  --gpu 0 \
  --cuda-arch 86 \
  --matrix configs/fp16_matmul_launch_shape_sweep.json \
  --threads 32,64,96,128,160,192,224,256,288,320,384 \
  --ncu-blocks-per-sm-csv 1,2,4,8 \
  --outdir results/strict_fp16_launch_shape_rtx3090
```

설치와 동시에 CMake build smoke까지 확인하려면 `--build-smoke`를 붙인다.

```bash
./scripts/install_gpu_toolchain.sh --gpu-kind h100 --build-smoke
```

공유 서버나 Docker/Slurm 환경에서 GPU auto-detect가 안 되면 `--gpu-kind` 또는 `--cuda-arch`를 명시한다. `CUDA_VISIBLE_DEVICES` 때문에 CUDA ordinal과 physical GPU id가 다르면 strict pipeline 실행 시 `--nvidia-smi-id GPU-...`를 함께 넘긴다. Nsight Compute가 `ERR_NVGPUCTRPERM`으로 실패하면 package 설치 문제가 아니라 NVIDIA performance counter 권한 문제다. 이 경우 cluster/admin policy로 profiling counter 접근을 허용하거나 권한 있는 job에서 NCU validation을 실행해야 한다. WSL2에서는 Linux `sudo`만으로 해결되지 않을 수 있으며, Windows host의 NVIDIA Control Panel에서 Developer settings와 GPU Performance Counters access를 허용한 뒤 WSL을 재시작해야 한다. Strict pipeline은 긴 sweep 전에 `ncu_permission_probe/` 아래 짧은 probe log/JSON/CSV를 남기고, counter 권한이 없으면 즉시 중단한다. 권한 없는 local 확인만 할 때는 `--diagnostic-no-ncu`를 명시해야 하며, 이 결과는 최종 pJ/bit claim에 사용하지 않는다.

RTX 3090 WSL2 host에서는 먼저 Windows host에서 NVIDIA Control Panel을 관리자 권한으로 열고 `Desktop > Enable Developer Settings`를 켠 뒤, `Developer > Manage GPU Performance Counters`에서 모든 사용자 접근을 허용한다. 그 다음 Windows PowerShell에서 `wsl --shutdown`을 실행하고 WSL을 다시 열어 아래 helper를 실행한다. 이 helper는 toolchain env를 재생성하고, `ncu` 경로를 확인한 뒤, `--probe-only`에서는 짧은 HMMA NCU permission probe만 수행한다. Full mode는 같은 thread/block sweep과 selected-target work-slope까지 실행한다.

```bash
cd /home/bang001/gpupowermodeling/accelwattch/util/fp16_energy_impl

# 권한 설정이 제대로 열렸는지 빠르게 확인
./scripts/run_rtx3090_strict_ncu_after_counter_enable.sh --probe-only

# probe가 통과하면 최종 strict RTX 3090 run 실행
./scripts/run_rtx3090_strict_ncu_after_counter_enable.sh
```

Native Linux에서 관리자가 profiling을 admin-only로 제한한 경우에만 `--use-sudo`를 붙인다. WSL2에서는 Windows host counter setting이 우선이며, 이 repository에서 확인한 RTX 3090 WSL2 run은 Linux `sudo`로도 `ERR_NVGPUCTRPERM`이 해결되지 않았다.

## 4. A100/H100 실행 범위와 자동화 범위

이 코드는 A100/H100에서도 실행 가능하다. GPU별 권장 CUDA architecture는 A100 `80`, RTX 3090 `86`, H100 `90`이다.

```bash
# A100
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j

# H100
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

실험 runner가 자동으로 처리하는 항목은 다음과 같다.

1. `--blocks 0`일 때 GPU SM 개수를 감지해 `blocks = SM_count * blocks_per_sm`로 설정한다.
2. matrix에 정의된 baseline/test 조건을 순서대로 실행한다.
3. `--repeat N`으로 전체 matrix를 N회 반복한다.
4. benchmark timed loop 직전/직후 `nvmlDeviceGetTotalEnergyConsumption()` 누적 에너지 카운터를 읽는다. 이 기능은 `libnvidia-ml.so.1`을 동적으로 load하므로 NVML header/link dependency 없이 빌드된다.
5. run별 `nvidia-smi` power/clock/temperature trace와 dmon SM utilization trace를 수집한다. `CUDA_VISIBLE_DEVICES`가 설정된 경우 runner는 CUDA ordinal에 대응하는 physical index/UUID를 telemetry id로 자동 해석한다.
6. `analyze_results.py`가 `summary.csv`, `condition_summary.csv`, thread sweep summary, figure를 생성한다.

수동으로 맞춰야 하는 항목은 다음과 같다.

1. CMake CUDA architecture 선택.
2. GPU clock 고정 및 실제 clock 안정성 확인.
3. 다른 GPU process 제거와 실험 환경 격리.
4. Nsight Compute performance counter 접근 권한 설정.
5. GPU별 `iters`/`repeats` 조정으로 power sample 수 확보.

H100에서도 현재 Tensor Core kernel은 `mma.sync.m16n8k16` 두 개로 logical `m16n16k16`을 만드는 warp-level 경로를 사용한다. 따라서 A100/H100/RTX 3090 간 같은 HMMA 계열 FP16 matmul path 비교에는 사용할 수 있지만, H100 고유 WGMMA/TMA 경로의 최대 matmul energy를 측정하는 실험은 아니다. H100 WGMMA 경로까지 측정하려면 별도 kernel과 validation matrix를 추가해야 한다.

Strict 재실험은 아래 helper 하나로 실행할 수 있다. GPU별로 `--cuda-arch`만 바꾼다.

```bash
# A100
./scripts/run_strict_fp16_pipeline.sh --gpu 0 --cuda-arch 80 --outdir results/strict_fp16_a100

# RTX 3090
./scripts/run_strict_fp16_pipeline.sh --gpu 0 --cuda-arch 86 --outdir results/strict_fp16_rtx3090

# H100
./scripts/run_strict_fp16_pipeline.sh --gpu 0 --cuda-arch 90 --outdir results/strict_fp16_h100
```

같은 서버에서 여러 대상 GPU를 볼 수 있으면 suite helper가 세 strict run과 postprocess를 한 번에 실행한다. 각 spec은 `label:CUDA_GPU:CUDA_ARCH[:NVIDIA_SMI_ID]` 형식이다. Suite helper는 긴 실험 전에 `strict_architecture_suite_preflight.json/csv`를 만들고, `cmake`/`nvcc`/`ncu`/`nvidia-smi`, GPU 이름과 arch 매칭, target GPU의 active compute process를 확인한다.

```bash
./scripts/run_strict_architecture_suite.sh \
  --spec a100:0:80 \
  --spec rtx3090:1:86 \
  --spec h100:2:90 \
  --require-work-slope \
  --outdir results/strict_fp16_suite
```

GPU가 서로 다른 서버에 있으면 각 서버에서 `run_strict_fp16_pipeline.sh`를 실행해 결과 디렉터리를 한 곳으로 복사한 뒤 `postprocess_strict_architectures.sh`를 실행한다. 단일 GPU만 재실험할 때는 suite helper에 `--require-architectures gh100`처럼 현재 architecture만 지정하거나 `--no-postprocess`를 사용한다.

Suite/postprocess helper와 `quality_gate.py`의 기본 required pair는 `tensor_mma_f16acc/tensor_baseline_mov`이다. `tensor_mma_f32acc`나 `fp16_half2`를 별도 diagnostic 비교로 묶을 때는 `--require-kernel/--require-baseline`을 명시한다. Suite helper는 이 값을 각 GPU strict pipeline의 quality gate, run manifest, postprocess audit, compare 단계에 동일하게 전달한다.

Preflight에서 toolchain이 빠졌거나, GPU metadata가 비어 있거나, 다른 compute process가 보이면 기본적으로 중단한다. Suite helper가 만든 preflight JSON/CSV는 postprocess report까지 전달되어 completion evidence matrix의 필수 항목으로 표시된다. 단일 GPU용 `run_strict_fp16_pipeline.sh`도 build 전에 같은 preflight를 실행해 `strict_pipeline_preflight.json/csv`를 남기고, compile은 되지만 runtime에서 실패할 CUDA driver/toolkit mismatch를 긴 sweep 전에 차단한다. CSV에는 target별 pass/fail뿐 아니라 `required_tools_pass`, `required_tool_fail_reasons`, `overall_preflight_pass`, `publishable_preflight_pass`도 기록해 GPU target은 맞지만 `cmake`/`nvcc`/`ncu`가 빠진 상황을 구분한다. 공유 장비에서 diagnostic만 하고 싶을 때는 `--allow-compute-apps`를 명시한다. `ncu`가 없거나 performance counter 권한이 없는 local smoke는 `--diagnostic-no-ncu`로 NCU preflight/probe/validation을 모두 건너뛰거나, 필요한 경우 `--skip-preflight`로만 수행한다. 최종 A100/H100/RTX3090 pJ/bit claim에는 이 예외 옵션을 쓰지 않는다.

Toolchain이 기본 PATH 밖에 있으면 `CMAKE_BIN`, `NVCC_BIN`, `NCU_BIN`, `NVIDIA_SMI_BIN`으로 명시한다. 추가 include/library flag가 필요한 split CUDA package 환경에서는 `CMAKE_CUDA_FLAGS`도 넘길 수 있다. Strict pipeline helper는 `CMAKE_CUDA_FLAGS`의 `-I`/`-L` 항목을 CMake compiler-id 단계에서도 보이도록 `CPATH`/`LIBRARY_PATH`에 자동으로 prepend한다. 예를 들어 conda CUDA toolchain을 쓰는 WSL 환경에서는 다음처럼 실행한다.

```bash
CMAKE_BIN=/path/to/cmake \
CMAKE_CUDA_FLAGS="-I/path/to/cuda/include -L/path/to/cuda/lib64" \
NVCC_BIN=/path/to/nvcc \
NCU_BIN=/path/to/ncu \
./scripts/run_strict_architecture_suite.sh \
  --spec rtx3090:0:86:GPU-... \
  --require-architectures ga102 \
  --outdir results/strict_fp16_rtx3090_suite
```

Preflight는 `nvidia-smi --version`의 `CUDA Version`과 `nvcc --version`의 release도 비교한다. `nvcc`가 드라이버가 지원하는 CUDA runtime보다 새 버전이면 compile은 성공해도 runtime preflight에서 `CUDA driver version is insufficient for CUDA runtime version`로 실패할 수 있으므로, 그 경우는 더 낮은 `NVCC_BIN`/toolkit을 쓰거나 드라이버를 업데이트해야 한다. 이 mismatch는 `strict_pipeline_preflight.json/csv`의 `toolchain_compatibility_pass=false`, `toolchain_nvcc_release`, `toolchain_driver_cuda_version`, `toolchain_recommended_cuda_toolkit`, `toolchain_recovery_commands`에 남아, 예를 들어 RTX 3090에서는 `install_gpu_toolchain.sh --gpu-kind rtx3090 --cuda-version 12.1`로 복구할 수 있음을 바로 확인할 수 있다.

Suite helper는 마지막에 `strict_architecture_suite_summary.json`도 쓴다. 이 파일의 `checks.publishable_pass=true`일 때만 suite 전체를 최종 보고 가능한 실행으로 본다. `--dry-run`, `--skip-preflight`, `--no-postprocess`, preflight 실패, run 실패, audit/report requirement 실패 중 하나라도 있으면 `diagnostic_only=true` 또는 `publishable_pass=false`로 남는다. `--require-work-slope`를 사용하면 summary의 `checks.work_slope_pass`와 `counts.work_slope_*`도 final readiness evidence가 된다. Summary에는 아래 energy policy도 기록되어, 최종 pJ/bit claim은 `nvml_total_energy_counter` / `strict_nvml_counter` selected target만 사용하고 `nvidia-smi` power trace는 fallback 또는 sanity check로만 취급한다.

Scheduler, Docker, Slurm 등에서 `CUDA_VISIBLE_DEVICES`가 GPU 순서를 바꾸는 경우 `--gpu`는 CUDA device ordinal이고, telemetry용 `nvidia-smi` 대상은 UUID로 명시할 수 있다.

```bash
GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
./scripts/run_strict_fp16_pipeline.sh \
  --gpu 0 \
  --nvidia-smi-id "${GPU_UUID}" \
  --cuda-arch 90 \
  --outdir results/strict_fp16_h100
```

이 pipeline은 clean build log를 `build_ptxas.log`로 저장하고, build 전에 `strict_pipeline_preflight.json/csv`로 toolchain/GPU/process 상태를 검사한 뒤, 환경 수집 후 짧은 CUDA runtime preflight를 실행한다. 또한 시작 시 `strict_pipeline_manifest_start.json`, 완료 시 `strict_pipeline_manifest.json`을 남겨 실행 인자, git head, tool version, binary hash, required kernel/baseline pair, NCU `blocks/SM` validation list, preflight/quality/NCU/resource 산출물 존재 여부를 고정한다. 이 단계가 실패하면 driver/runtime mismatch 또는 오래된 binary 문제이므로 긴 sweep을 시작하지 않는다. Runtime preflight JSON은 이어서 `verify_architecture.py --strict-chip --require-common-hmma`로 검증한다. 즉 `--cuda-arch 80`은 A100/GA100, `86`은 RTX 3090/GA102, `90`은 H100/GH100 metadata와 맞아야 하며, 공통 HMMA path가 아니라 WGMMA를 쓴 경우 strict pipeline은 중단된다. Strict mode에서는 그 다음 `probe_ncu_permissions.py`로 짧은 Tensor Core HMMA profile을 실행해 NCU performance counter 권한을 확인한다. 이 probe가 `ERR_NVGPUCTRPERM`을 기록하면 `ncu_permission_probe/ncu_permission_probe.json`에 원인이 남고, calibration/sweep을 시작하기 전에 중단한다. 이후 `calibrate_matrix.py`로 기본 `fp16_matmul_thread_sweep_fine.json`의 test/baseline timed duration이 기본 1초 이상이 되도록 per-role `repeats`를 GPU별로 보정하고, 보정된 `calibrated_matrix.json`으로 structural baseline sweep을 실행한다. 이 calibration은 기존 matrix의 repeats를 기본적으로 줄이지 않고, 필요한 경우만 늘린다. `--matrix`로 다른 matrix를 지정할 수 있고, `--no-calibrate-matrix`를 주면 원본 matrix를 그대로 사용한다. Sweep 뒤에는 `summarize_kernel_resources.py`로 register/spill/resource occupancy evidence를 남긴다. Strict mode에서는 이후 `ncu_validate_no_l2_thread_sweep.sh`로 같은 thread 후보의 NCU 검증을 수행하고, `quality_gate.py --require-ncu --require-ncu-tensor-activity --require-kernel tensor_mma_f16acc --require-baseline tensor_baseline_mov`까지 실행한다. Pipeline은 `--require-kernel/--require-baseline` 값을 `NCU_TEST_KERNEL/NCU_BASELINE_KERNEL`로 profiler validation helper에도 전달하므로 quality gate, NCU evidence, audit target이 같은 kernel pair를 본다. `--run-work-slope`를 명시하면 quality-gated selected target과 같은 `threads`/`blocks/SM`으로 `work_slope_matrix.json`을 생성하고, calibration 없이 raw work amount sweep을 실행해 `work_slope/work_slope_summary.csv`와 root-level `work_slope_summary.csv`를 남긴다. `--diagnostic-no-ncu`를 명시하면 NCU preflight/probe/validation을 모두 건너뛰고 `quality_gate.py`도 NCU summary 없이 실행하므로 결과는 diagnostic-only로 남는다. NCU가 performance counter 권한 문제 등으로 실패하면 `ncu_no_l2_thread_sweep/ncu_run_failures.csv`에 실패 kernel, exit code, log path, 재현 command가 남는다. 최종 pJ/bit 후보는 `quality_gate_summary.json`의 `selected_targets`가 비어 있지 않을 때만 채택한다. 기본 selected target은 `tensor_mma_f16acc/tensor_baseline_mov` pair로 제한되며, 다른 FP16 path는 `--require-kernel/--require-baseline`을 바꾸지 않는 한 diagnostic row로만 남는다. NCU metric 이름이 장비/버전에서 다르면 `NCU_METRICS="..." ./scripts/run_strict_fp16_pipeline.sh ...`처럼 override한다.

Calibration만 별도로 실행할 수도 있다. 새 장비에서는 binary probe를 사용하고, 이미 짧은 diagnostic run을 분석한 뒤에는 `summary.csv`를 이용해 반복 수를 재계산할 수 있다.

```bash
python3 scripts/calibrate_matrix.py \
  --matrix configs/fp16_matmul_thread_sweep_fine.json \
  --out-matrix results/strict_fp16_h100/calibrated_matrix.json \
  --outdir results/strict_fp16_h100 \
  --binary build/fp16_energy_bench \
  --gpu 0 \
  --target-test-s 1.0 \
  --target-baseline-s 1.0

python3 scripts/calibrate_matrix.py \
  --matrix configs/fp16_matmul_thread_sweep_fine.json \
  --from-summary results/diagnostic_h100/summary.csv \
  --out-matrix results/diagnostic_h100/calibrated_matrix.json \
  --outdir results/diagnostic_h100
```

baseline subtraction이 불안정하거나 `incremental_energy_j`가 음수로 흔들리면, 같은 thread 조건에서 work amount를 바꾸는 slope diagnostic을 실행한다. Strict pipeline에서는 `--run-work-slope`로 자동 실행할 수 있다. Suite 전체에서 final hard gate까지 같이 걸려면 `run_strict_architecture_suite.sh --require-work-slope`를 사용한다.

```bash
./scripts/run_strict_fp16_pipeline.sh \
  --gpu 0 \
  --cuda-arch 86 \
  --outdir results/strict_fp16_rtx3090 \
  --run-work-slope
```

수동으로 실행할 때도 matrix는 unroll별 work 차이를 유지해야 하므로 strict pipeline calibration 대신 raw runner를 사용한다. 최종 audit에 연결할 work-slope는 selected target과 같은 `threads` 및 `blocks_per_sm` context여야 하므로, 먼저 selected target에서 matrix를 자동 생성한다.

```bash
python3 scripts/generate_work_slope_matrix.py \
  --result-dir results/strict_fp16_rtx3090 \
  --out-matrix results/strict_fp16_rtx3090/work_slope_matrix.json

python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix results/strict_fp16_rtx3090/work_slope_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --outdir results/strict_fp16_rtx3090_work_slope

python3 scripts/analyze_results.py --input results/strict_fp16_rtx3090_work_slope
```

이 결과는 `work_slope_summary.csv`와 `figures/work_slope_*.png`를 만든다. `--run-work-slope`로 pipeline 내부에서 실행한 경우 root-level `work_slope_summary.csv`도 함께 복사되어 `audit_strict_results.py`가 자동으로 읽는다. 수동 work-slope 결과를 별도 디렉터리에 둔 경우에는 postprocess에 `--work-slope-dir results/strict_fp16_rtx3090_work_slope --require-work-slope`를 추가한다. 이 gate는 selected target과 같은 `test_kernel`, `baseline_kernel`, `threads`, `blocks_per_sm`, `fit_scope=valid_no_l2`, positive `slope_matmul_input_pj_per_bit`, 기본 `slope_r2 >= 0.80`을 요구한다. 또한 slope fit에 사용된 row들이 current benchmark schema, logical `m16n16k16` denominator provenance, timed-kernel no-memory provenance, no intended global-memory metadata, strict NVML energy counter를 모두 만족해야 한다. Work-slope는 단일 pJ/bit point를 대체하지 않고, work amount 증가에 따라 incremental energy가 선형으로 증가하는지 확인하는 baseline-subtraction 안정성 evidence다.

### Energy source policy

최종 energy 계산은 timed loop 내부의 NVML 누적 에너지 카운터 delta를 우선 사용한다. 즉 `bench.json`의 `nvml_energy_supported=true`이고 `nvml_energy_delta_j > 0`이면 `power_energy_j`와 `avg_power_w`는 `nvmlDeviceGetTotalEnergyConsumption()` 기반 값이다. 카운터가 지원되지 않는 GPU/driver 조합에서는 기존 방식대로 `nvidia-smi --query-gpu=power.draw` trace를 host timed interval에 적분한 값을 fallback으로 사용한다.

benchmark binary는 warmup 완료 후 CUDA event interval을 열기 직전에 energy counter를 읽고, `cudaEventSynchronize(stop)`으로 timed kernel 종료를 확인한 뒤 다시 읽는다. NVML device handle은 우선 CUDA device의 PCI bus id로 찾고, 실패 시 CUDA device index로 fallback한다. 따라서 `CUDA_VISIBLE_DEVICES`가 NVML index 순서를 바꾼 환경에서도 benchmark 내부 counter는 실행 중인 CUDA device와 같은 물리 GPU를 가리키도록 설계되어 있다.

`nvidia-smi` power trace는 제거하지 않는다. H100처럼 `power.draw`가 averaging/smoothing된 값을 줄 수 있는 환경에서는 NVML total energy counter가 더 직접적인 최종 에너지 값이고, power trace는 clock/temperature/throttling 및 counter-vs-trace sanity check 용도다. runner는 지원되는 경우 `power.draw.average`와 `power.draw.instant`를 별도 컬럼으로 남긴다. 분석 결과에는 `energy_source`, `power_trace_energy_j`, `power_trace_query_modes`, `nvml_energy_delta_j`, `energy_counter_vs_trace_delta_j`, `energy_counter_vs_trace_ratio`가 함께 기록된다.

`quality_gate.py`는 NVML counter 기반 energy와 power trace 적분값의 ratio도 확인한다. 기본값은 warning-only이며, H100/Ampere 이후 `power.draw`가 다른 시간 window의 평균값일 수 있기 때문이다. 이 sanity check까지 hard gate로 쓰려면 `--require-counter-trace-agreement`를 추가한다.

#### A100/H100 power API policy

최종 pJ/FLOP 또는 pJ/bit 산출용 API와 trace/debug용 API를 분리한다. 이 repo의 최종 energy path는 benchmark binary 내부의 `nvmlDeviceGetTotalEnergyConsumption()` delta이고, `nvidia-smi` trace는 fallback 또는 sanity check 용도다.

| API / field | A100/GA100 | H100/GH100 | 시간 의미와 제약 | 이 repo의 사용 |
|---|---|---|---|---|
| `nvmlDeviceGetTotalEnergyConsumption()` | 사용 | 사용 | driver reload 이후 device-level total energy 누적값, mJ. sampling API가 아니라 시작/끝 counter delta로 Joule을 계산한다. MIG/shared GPU에서는 같은 물리 GPU의 다른 workload energy가 섞일 수 있다. | 최종 `power_energy_j`, `avg_power_w`의 1순위 source |
| `nvidia-smi power.draw` | 사용 가능 | 사용 가능 | 장비/driver별 power telemetry 의미가 다를 수 있고, H100 계열에서는 smoothing/averaging window 영향이 커질 수 있다. | NVML counter 미지원 시 fallback 적분값과 counter-vs-trace sanity check |
| `power.draw.instant` | 사용 가능 | 사용 가능 | last measured power draw 계열 trace. polling interval을 낮춰도 실제 센서 갱신률이 그만큼 올라간다는 뜻은 아니다. | `power.csv`에 별도 컬럼으로 저장 |
| `power.draw.average` | A100/GA100에서는 `N/A`일 수 있음 | 사용 가능 | last-second average 계열 trace. 100 ms로 polling해도 독립적인 100 ms energy sample로 해석하면 안 된다. | H100 debug trace 컬럼 |
| `nvmlDeviceGetPowerUsage()` | instantaneous 계열로 해석 | 1초 평균 계열로 해석 | 같은 API라도 A100과 H100의 시간 의미가 다르므로 architecture 비교의 최종 energy source로 쓰지 않는다. | 직접 사용하지 않음 |
| `module.power.draw.*` | 일반적으로 해당 없음 | GH/Hopper module telemetry에서 보일 수 있음 | module scope는 GPU-only가 아니라 CPU/기타 module component를 포함할 수 있다. | FP16 pJ/bit 계산에 사용하지 않음 |

따라서 A100/H100/RTX 3090 비교에서 보고할 최종 값은 `energy_source=nvml_total_energy_counter`이고 `measurement_grade=strict_nvml_counter`인 selected target만 사용한다. `power_trace_integral` 결과는 legacy/diagnostic grade로 유지하되, strict 비교 표와 figure에는 섞지 않는다.

### Quality gate policy

`analyze_results.py`가 만든 수치는 바로 최종값으로 채택하지 않고, `quality_gate.py`로 다음 조건을 확인한다.

```bash
python3 scripts/quality_gate.py --input results/fp16_matmul_thread_sweep_fine_gpu0
```

Gate가 확인하는 핵심 조건은 다음과 같다.

| Gate | 의미 |
|---|---|
| positive increment | baseline subtraction 뒤 incremental power/energy가 양수 |
| no intended L2/global traffic | test/baseline benchmark JSON의 `timed_kernel_has_intended_global_memory=false`가 모두 성립해야 함. Legacy 결과는 `suppress_output_store`와 kernel type으로 fallback 추론되지만 strict target은 직접 metadata가 필요 |
| enough valid repeats | thread point별 `valid_no_l2_count >= max(3, ceil(run_count/2))` |
| stable clock | 기본값으로 `clock_span_mhz <= 60` |
| reliable energy source | `nvml_total_energy_counter` 우선. 미지원 시 power trace fallback은 최소 sample 수를 만족해도 diagnostic grade로만 통과하며, 기본 target selection에는 쓰지 않음 |
| counter-vs-trace cross-check | NVML counter energy와 `nvidia-smi` power trace 적분값의 ratio를 warning band로 확인. 기본은 warning-only |
| reliable energy signal | 기본값으로 `incremental_energy_fraction >= 0.01`이고 `baseline_energy_fraction <= 0.99`. 0.05 미만은 warning |
| measurement resolution | 기본값으로 test/baseline elapsed time >= 0.25 s, test energy >= 1 J, incremental energy >= 0.1 J |
| benchmark schema | test/baseline 모두 `schema_version=fp16-energy-bench-v2`이고 required `schema_features`를 포함해야 함 |
| matmul denominator | Tensor Core pJ/bit 분모가 benchmark JSON에서 직접 기록된 logical `m16n16k16` metadata인지 확인. `matmul_denominator_source=bench_json_metadata`, `matmul_input_bits_per_logical_mma=8192`, `matmul_flops_per_logical_mma=8192`가 아니면 최종 target에서 제외 |
| structural baseline | Tensor Core f16acc는 `tensor_baseline_mov`, f32acc는 `tensor_baseline_f32`, CUDA-core half2는 `baseline_regmove`를 strict baseline으로 사용 |
| common instruction path | A100/H100/RTX3090 비교에서는 WGMMA가 아니라 공통 HMMA `mma.sync.m16n8k16` pair path |
| NCU validation | 최종 claim에는 `validate_ncu_reports.py`가 만든 `ncu_validation_summary.csv`를 `--require-ncu`로 연결 |
| NCU validation context | NCU 검증 run의 `threads`, `blocks_per_sm`, `unroll`, `suppress_output_store`가 측정 row와 같은지 확인 |
| NCU tensor activity | explicit metric 또는 `ComputeWorkloadAnalysis` section에서 Tensor activity percentage를 추출해 selected Tensor Core point의 profiler-side utilization evidence로 기록 |
| ptxas resource audit | selected test/baseline kernel의 register/thread와 stack/spill bytes를 build log에서 추출하고 `threads`, `blocks_per_sm`, `unroll` context가 measurement row와 같은지 확인 |
| utilization target | strict quality gate를 통과한 후보군 안에서 Tensor Core matmul은 `tensor_model_utilization_pct_mean`, 그 외 kernel은 SM/GPU utilization 최대값 0.1 percentage point 이내로 포화된 가장 작은 `threads_per_sm` |
| Tensor model sanity | `tensor_model_utilization_pct_mean > 105%`는 architecture model, clock telemetry, FLOP accounting 중 하나가 어긋난 것으로 보고 quality gate에서 제외 |

출력은 `quality_gates.csv`, `quality_gate_summary.json`, `figures/quality_gate_thread_sweep_*.png`, `figures/quality_gate_target_metric_thread_sweep_*.png`이다. `target_pass=true`인 row가 최종 thread-count 추천점이며 `quality_gate_summary.json`의 `selected_targets`에 들어간다. Target 판정은 기본적으로 `quality_pass=true`이고 `measurement_grade=strict_nvml_counter`인 row만 utilization reference pool로 사용하므로, L2/global traffic, NCU, energy source, denominator, 측정 해상도 gate를 통과하지 못한 high-utilization row가 최종 target을 밀어내지 못한다. 또한 `selected_targets`는 기본적으로 `--require-kernel tensor_mma_f16acc --require-baseline tensor_baseline_mov` pair 안에서만 생성된다. `fp16_half2`나 `tensor_mma_f32acc`가 같은 matrix에서 quality gate를 통과해도 required pair를 바꾸지 않으면 `target_selection_note=not_required_kernel_baseline_target_scope`로 남고 final `selected_targets`에는 들어가지 않는다. Tensor Core matmul kernel은 NVML SM utilization이 낮은 occupancy에서도 100%로 포화될 수 있으므로, target selection에는 `tensor_model_utilization_pct_mean`을 우선 사용한다. dmon SM/GPU utilization이 없더라도 Tensor model utilization이 유효하면 target metric fallback으로 기록하고, `sm_util_available=false`, `target_util_available=true`, fallback warning을 남긴다. 이 fallback은 target metric 선택에만 쓰이며 clock/energy/NCU 같은 다른 strict gate를 대체하지 않는다. Analyzer와 quality gate는 기본적으로 `tensor_model_utilization_pct_mean > 105%` 후보를 utilization sanity 실패로 보고 selected/target 후보에서 제외한다. `quality_gates.csv`에는 `quality_gate_selected_target`, `util_reference_scope`, `util_reference_max_pct`, `util_metric_source`, `target_selection_note`와 함께 selected NCU row에서 온 `common_hmma_seen`, `wgmma_token_seen`, `tensor_inst_seen`, `memory_counter_classes_complete`, `no_l2`, `no_dram`, `no_local_spill` evidence를 test/baseline 양쪽 prefix로 기록한다. 기존 sweep logic이 고른 point라도 strict gate를 통과하지 못하면 `selected_diagnostics`로만 남긴다. `measurement_grade=power_trace_fallback`은 기존 RTX 3090 결과처럼 NVML energy counter가 없는 legacy run을 의미하므로, A100/H100 최종 비교에서는 같은 matrix를 다시 실행해 `strict_nvml_counter` 결과를 우선 사용한다. power-trace 결과를 임시 target으로 보고 싶을 때만 `quality_gate.py --allow-power-trace-target`을 명시한다. `baseline_match_grade=generic_nop_baseline`인 결과는 utilization diagnostic으로만 보고, 최종 FP16 pJ/bit에는 쓰지 않는다.

최종 보고용 gate는 Nsight Compute 검증 결과까지 묶어서 실행한다.

```bash
python3 scripts/quality_gate.py \
  --input results/fp16_matmul_thread_sweep_fine_gpu0 \
  --ncu-summary results/ncu_no_l2_thread_sweep_gpu0/ncu_validation_summary.csv \
  --require-ncu
```

### Architecture comparison policy

A100, H100, RTX 3090 비교는 같은 logical workload와 같은 instruction family를 비교하는 방식으로 해석한다. 현재 Tensor Core kernel은 세 GPU 모두에서 warp-level `mma.sync.m16n8k16` 두 개를 묶어 logical `m16n16k16`을 만들며, H100에서 지원되는 WGMMA 경로를 사용하지 않는다. 따라서 이 결과는 "H100의 최대 WGMMA matmul efficiency"가 아니라 "A100/H100/RTX3090에서 공통 HMMA FP16 path를 같은 baseline subtraction으로 측정한 값"이다.

benchmark JSON과 분석 CSV에는 `architecture_generation`, `architecture_chip`, `recommended_cuda_arch`, `fp16_tensor_instruction_path`, `wgmma_supported`, `benchmark_uses_wgmma`가 기록된다. 오래된 결과 JSON도 `analyze_results.py`가 `device_name`과 `compute_capability`로 fallback 분류한다. Summary에는 `baseline_energy_fraction`, `incremental_energy_fraction`, `baseline_power_fraction`, `valid_no_l2`, `pure_fp16_candidate`, `separation_quality`가 추가되어, baseline이 너무 크거나 L2/global traffic이 예상되는 run을 pJ/bit 최종 후보에서 분리할 수 있다.

분석 CSV는 architecture별 dense Tensor Core peak model도 함께 기록한다. `tensor_peak_tflops_model`은 측정 run의 `sm_count`와 평균 SM clock에서 계산한 dense FP16 Tensor Core peak이고, `achieved_flops_per_sm_cycle`와 `tensor_model_utilization_pct`는 실제 measured TFLOPS가 그 common HMMA model 대비 어느 정도인지 보여준다. 이 값은 thread sweep target을 해석하기 위한 normalization metric이며, pJ/bit의 energy source를 대체하지 않는다. H100에서는 WGMMA 최대 경로가 아니라 이 benchmark가 실제 사용하는 warp-level HMMA `m16n8k16` pair path 기준으로 해석한다. `architecture_models.py`에는 reference dense/sparse TFLOPS, product/reference source URL, Tensor Core architecture source URL을 같이 둔다. H100 public product table의 FP16 Tensor Core 값은 sparsity footnote가 붙어 있고, Hopper architecture table은 dense/sparse throughput을 분리해 제시하므로 dense reference는 common HMMA dense model 기준으로 둔다.

Peak normalization은 accumulator mode도 구분한다. 최종 strict target인 `tensor_mma_f16acc`는 FP16 input + FP16 accumulate 기준이고, diagnostic `tensor_mma_f32acc`는 FP16 input + FP32 accumulate 기준이다. A100/GA100과 H100/GH100은 현재 common HMMA dense model에서 두 mode의 reference를 같은 값으로 둔다. RTX 3090/GA102는 NVIDIA GA102 whitepaper의 `Peak FP16 Tensor TFLOPS with FP16 Accumulate` 142/284와 `with FP32 Accumulate` 71/142 구분을 반영해, f32acc utilization을 f16acc peak의 절반 기준으로 normalize한다. Analyzer output의 `tensor_model_accumulator_mode`, `tensor_model_flop_per_sm_cycle`, `tensor_model_reference_dense_tflops`를 보면 각 row가 어떤 기준으로 normalized 되었는지 확인할 수 있다.

`compare_architectures.py`는 launch-shape sweep 비교에서 `threads`만이 아니라 `threads_per_sm`과 `blocks_per_sm_requested`까지 보존한다. 따라서 `threads=64, blocks/SM=4`와 `threads=64, blocks/SM=8`처럼 thread/block은 같지만 SM당 resident thread 수가 다른 후보가 quality gate/target marker와 pJ/bit annotation에서 서로 덮이지 않는다.

Architecture model 자체의 내부 일관성은 GPU 없이도 확인할 수 있다. 아래 명령은 `dense_tensor_fp16_flop_per_sm_cycle * reference_sm_count * reference_boost_clock_mhz`로 reference dense TFLOPS를 재계산하고, reference table 값과의 오차를 CSV/figure로 남긴다. 또한 per-SM Tensor Core capacity와 thread/register/block resource limit figure를 생성한다.

```bash
python3 scripts/architecture_models.py \
  --outdir results/architecture_models \
  --fail-on-model-error-pct 1.0 \
  --fail-on-missing-metadata
```

## 5. 실험 전 환경 수집

```bash
./scripts/query_env.sh 0 results/env_gpu0.txt
```

세 번째 인자로 binary path를 넘기면 CUDA runtime probe와 resource-usage dump도 같이 시도한다.

```bash
./scripts/query_env.sh 0 results/env_gpu0.txt build/fp16_energy_bench
```

첫 번째 인자는 `nvidia-smi` telemetry id이며 UUID도 가능하다. telemetry id와 CUDA device ordinal이 다르면 네 번째 인자로 CUDA device index를 넘긴다.

```bash
./scripts/query_env.sh "${GPU_UUID}" results/env_gpu0.txt build/fp16_energy_bench 0
```

`cuobjdump`가 설치되어 있으면 `query_env.sh` 출력에 kernel별 resource usage가 포함된다. 설치되어 있지 않은 환경에서는 CMake가 이미 `-Xptxas=-v`로 build output에 register 수와 spill 정보를 출력하므로, 다음처럼 build log를 남겨 확인한다.

```bash
cmake --build build -j 2 2>&1 | tee results/build_ptxas.log
rg "Used .* registers|spill" results/build_ptxas.log
```

가능하면 GPU clock을 고정한다.

```bash
sudo ./scripts/lock_clocks.sh 0 <SM_CLOCK_MHZ> <MEM_CLOCK_MHZ>
```

실험 후 원복한다.

```bash
sudo ./scripts/reset_clocks.sh 0
```

`nvidia-smi -q -d CLOCK`으로 가능한 clock 값을 확인한 뒤 실제 장비에 맞는 값을 넣는다. `-lgc`/`-lmc`가 제한되는 datacenter 환경에서는 application clocks 또는 cluster policy에 맞춰 동일한 목적의 clock 고정 절차를 사용한다.

## 6. P0 primary 실험 실행

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/primary_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/p0_gpu0
```

분석과 시각화는 다음 명령으로 생성한다.

```bash
python3 scripts/analyze_results.py --input results/p0_gpu0
```

`run_experiment.py`는 기존 `runs.jsonl`이 있는 outdir에는 기본적으로 쓰지 않는다. 같은 outdir에 의도적으로 run을 누적할 때만 `--append`를 추가한다. `--nvidia-smi-id`를 생략하면 `CUDA_VISIBLE_DEVICES` mapping을 먼저 보고, 없으면 `--gpu` 값을 telemetry id로 사용한다.

For FP16 Tensor Core matmul logical pJ/bit estimates:

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_pjbit_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/fp16_matmul_pjbit_gpu0

python3 scripts/analyze_results.py --input results/fp16_matmul_pjbit_gpu0
```

For FP16 Tensor Core thread-count sweep under no intended L2/global traffic:

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_thread_sweep.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/fp16_matmul_thread_sweep_gpu0

python3 scripts/analyze_results.py --input results/fp16_matmul_thread_sweep_gpu0
python3 scripts/quality_gate.py --input results/fp16_matmul_thread_sweep_gpu0
```

이 sweep은 `threads = 32, 64, 128, 256, 512, 1024`를 훑는다. Matrix default의 `suppress_output_store=true`가 test와 `tensor_baseline_mov` timed kernel의 final global store를 끄므로, 의도된 timed-loop L2/global memory traffic 없이 Tensor Core utilization만 비교한다. `tensor_baseline_mov`는 no-memory warp-sync baseline이라 ptxas가 empty/register-only loop를 제거하는 문제를 피한다. 분석기는 `thread_sweep_summary.csv`를 만들고, 충분한 valid no-L2 후보 중 Tensor Core matmul은 `tensor_model_utilization_pct_mean`이 포화되는 가장 작은 `threads_per_sm` point를 `selected_optimal=True`로 표시한다. 다만 model utilization이 105%를 넘는 후보는 clock telemetry, FLOP accounting, architecture peak model 중 하나가 어긋난 diagnostic으로 보고 analyzer selection에서도 제외한다. non-Tensor Core kernel은 dmon `avg_sm_util_pct_mean`, `avg_gpu_util_pct_mean` 순서로 selection metric을 사용하고, measured utilization이 없을 때만 diagnostic plot/selection에 한해 dense Tensor Core peak model 대비 `tensor_model_utilization_pct_mean`으로 fallback한다.

Coarse sweep에서 유효 후보가 좁혀지면 fine sweep을 추가로 실행한다. Fine matrix는 `threads = 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384`를 훑는다. 160 이상 후보는 test `repeats=2`, baseline `repeats=20`을 사용한다. 32/64/96/128 후보는 duration과 sample 수를 맞추기 위해 각각 더 큰 repeats를 사용한다.

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/fp16_matmul_thread_sweep_fine.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/fp16_matmul_thread_sweep_fine_gpu0

python3 scripts/analyze_results.py --input results/fp16_matmul_thread_sweep_fine_gpu0
python3 scripts/quality_gate.py --input results/fp16_matmul_thread_sweep_fine_gpu0
```

여러 반복이 있는 thread sweep에서는 `valid_no_l2_count >= max(3, ceil(run_count/2))`인 후보를 우선 선택한다. 이 조건을 만족하는 후보가 없을 때만 더 약한 후보군으로 fallback한다. 최종 `target_pass` 선택 기준은 Tensor Core matmul의 경우 strict `quality_pass=true` 후보 중 max `tensor_model_utilization_pct_mean`에서 0.1 percentage point 이내로 포화된 가장 작은 `threads_per_sm`이며, 같은 `threads_per_sm`에서는 TFLOPS와 clock stability로 tie-break한다. non-Tensor Core kernel은 measured SM/GPU utilization을 selection metric으로 사용한다.

`blocks_per_sm=8`이 최적이라는 가정도 별도 확인할 수 있다. `configs/fp16_matmul_launch_shape_sweep.json`은 `threads/block = 32,64,96,128,160,192,224,256,288,320,384`와 `blocks/SM = 1,2,4,8`을 함께 훑어 같은 launched `threads_per_sm`이라도 다른 launch shape가 SM utilization, pJ/bit, baseline subtraction 안정성에 어떤 영향을 주는지 확인한다. `analyze_results.py`는 `thread_sweep_summary.csv`를 `threads`와 `blocks_per_sm_requested` 둘 다로 집계하고, `quality_gate.py`와 NCU validation context도 blocks/SM을 같이 매칭한다. 또한 target selection에 사용한 utilization metric, saturation margin, valid no-L2 후보 내 pJ/bit rank, 선택점보다 낮은 pJ/bit 후보 목록을 기록해 "first saturation point" target과 "lowest observed pJ/bit" 후보를 구분한다.

```bash
./scripts/run_strict_fp16_pipeline.sh \
  --gpu 0 \
  --cuda-arch 86 \
  --matrix configs/fp16_matmul_launch_shape_sweep.json \
  --threads 32,64,96,128,160,192,224,256,288,320,384 \
  --ncu-blocks-per-sm-csv 1,2,4,8 \
  --outdir results/strict_fp16_launch_shape_rtx3090
```

Nsight Compute validation도 여러 blocks/SM을 함께 확인할 수 있다.

```bash
NCU_BLOCKS_PER_SM_CSV=1,2,4,8 \
  ./scripts/ncu_validate_no_l2_thread_sweep.sh \
  build/fp16_energy_bench \
  results/ncu_launch_shape_gpu0 \
  0 \
  32,64,96,128,160,192,224,256,288,320,384
```

2026-06-02 RTX 3090 direct launch-shape diagnostic 결과는 `results/fp16_launch_shape_warpsync_rtx3090_20260602_direct/`에 포함했다. 이 환경에서는 WSL이 background GPU telemetry와 Python/nested-shell child CUDA launch를 막아 `run_experiment.py`, dmon, NCU를 동시에 사용할 수 없었다. 따라서 benchmark binary를 foreground command로 직접 32회 실행하고, 에너지는 benchmark 내부 `nvmlDeviceGetTotalEnergyConsumption()` delta만 사용했다. `power.csv`, `sm_util.csv`, NCU validation이 없으므로 이 결과는 strict final target이 아니라 diagnostic이다. Analyzer는 measured SM/GPU utilization 대신 TFLOPS/reference dense peak 기반 `tensor_model_utilization_pct_mean`으로 saturation point를 표시했다.

| Result dir | Utilization source | Selected diagnostic launch shape | Selected threads/SM | valid no-L2 | TFLOPS | `matmul_input_pj_per_bit` | Quality status |
|---|---|---:|---:|---:|---:|---:|---|
| `results/fp16_launch_shape_warpsync_rtx3090_20260602_direct` | TFLOPS/reference dense peak fallback | `threads=64`, `blocks/SM=2` | 128 | 1/1 | 149.24 | 0.3533 pJ/bit | `selected_optimal=true`, `target_pass=false` |

이 diagnostic sweep에서는 measured SM util이 없어서 TFLOPS/reference dense peak 기반 model-utilization을 사용했다. 재분석 기준에서는 `tensor_model_utilization_pct_mean > 105%` 후보 9개를 model/clock/accounting sanity 실패로 제외했고, analyzer selection이 `t64/b2`를 표시한다. 더 큰 `threads_per_sm`에서도 pJ/bit가 더 낮게 보이는 지점이 있으나, 현재 run에는 measured SM util, clock trace, power trace cross-check, NCU no-L2/Tensor activity evidence가 없다. 특히 `threads=256, blocks/SM=2`는 baseline subtraction 뒤 incremental energy가 음수라 `stats_scope=all_runs_no_valid`로 남는다. 최종 보고용 RTX 3090/A100/H100 값은 strict pipeline에서 dmon/clock/NCU까지 통과한 `quality_gate_summary.json:selected_targets`만 사용한다.

관련 figure:

```text
results/fp16_launch_shape_warpsync_rtx3090_20260602_direct/figures/thread_sweep_tensor_mma_f16acc_vs_tensor_baseline_mov.png
results/fp16_launch_shape_warpsync_rtx3090_20260602_direct/figures/thread_sweep_pjbit_tensor_mma_f16acc_vs_tensor_baseline_mov.png
results/fp16_launch_shape_warpsync_rtx3090_20260602_direct/figures/quality_gate_thread_sweep_tensor_mma_f16acc_vs_tensor_baseline_mov.png
```

2026-05-28 RTX 3090 local run 결과는 다음과 같다. 기준 matmul은 logical `m16n16k16`이며, 구현은 `mma.sync.m16n8k16` instruction 두 개로 N 방향 16 columns를 채운다. pJ/bit denominator는 A/B logical FP16 input bit인 `(16*16 + 16*16) * 16 = 8192 bit/logical MMA`다. 이 run은 이전 matrix의 `baseline_nop` 기반 legacy diagnostic 결과이며, Nsight Compute counter validation은 이 환경에서 `ncu`가 PATH에 없어 수행하지 못했다. 현재 strict matrix는 `tensor_baseline_mov` no-memory warp-sync baseline을 사용하므로 최종 pJ/bit는 재실행 후 `quality_gate.py`의 `target_pass=true` 결과를 사용한다.

| Sweep | 후보 threads/block | Selected threads/SM | Selected threads/block | valid no-L2 | Avg SM util (%) | TFLOPS | `matmul_input_pj_per_bit` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine + dmon SM util | 32,64,96,128,160,192,224,256,288,320,384 | 512 | 64 | 9/10 | 100.000 +/- 0.000 | 159.1 | 0.2514 +/- 0.0142 pJ/bit |

Fine + dmon legacy run에서는 `threads_per_sm=512`에서 평균 SM utilization이 100%에 도달했다. 따라서 SM utilization 첫 포화점 기준 후보 thread count는 512 threads/SM, 즉 64 threads/block이다. pJ/bit 자체는 224 threads/block 후보에서 `0.0819 +/- 0.0121 pJ/bit`로 더 낮게 관찰되었지만, valid no-L2 count가 5/10이고 incremental power가 3.23 W 수준이라 target saturation point로 채택하지 않았다. 256 threads/block 이상 후보는 baseline subtraction 후 incremental power가 음수로 나와 `all_runs_no_valid`로 집계되었다. 이 값들은 board-level `nvidia-smi` power trace와 `baseline_nop` subtraction 기반 estimate이므로, 최종 수치 채택 전에는 현재 matrix로 재실행하고 Nsight Compute로 no-L2/global-memory 조건과 HMMA instruction path를 별도 확인해야 한다.

여러 GPU에서 같은 matrix를 실행한 뒤 architecture-level 비교 figure를 생성하려면 각 결과 디렉터리에 대해 strict pipeline을 완료한 다음 아래 wrapper를 실행한다. 이 wrapper는 strict audit이 실패해도 compare/report 산출물을 먼저 남기고, 기본적으로 마지막에 nonzero로 종료한다. `--require-architectures`는 audit, compare, architecture model, report 단계에 동일하게 전달되고, `--require-kernel/--require-baseline`은 audit과 compare 단계에 동일하게 전달된다. 실패 상태까지 report로 남기고 싶을 때는 `--no-fail`을 추가한다.

```bash
./scripts/postprocess_strict_architectures.sh \
  --outdir results/strict_fp16_postprocess \
  --require-kernel tensor_mma_f16acc \
  --require-baseline tensor_baseline_mov \
  results/strict_fp16_a100 \
  results/strict_fp16_h100 \
  results/strict_fp16_rtx3090
```

수동으로 단계를 분리해서 실행할 수도 있다.

```bash
python3 scripts/audit_strict_results.py \
  --input results/strict_fp16_a100 \
          results/strict_fp16_h100 \
          results/strict_fp16_rtx3090 \
  --outdir results/strict_fp16_audit

python3 scripts/compare_architectures.py \
  --input results/strict_fp16_a100 \
          results/strict_fp16_h100 \
          results/strict_fp16_rtx3090 \
  --outdir results/architecture_compare_fp16 \
  --audit-dir results/strict_fp16_audit \
  --require-kernel tensor_mma_f16acc \
  --require-baseline tensor_baseline_mov

python3 scripts/report_strict_results.py \
  --audit-dir results/strict_fp16_audit \
  --compare-dir results/architecture_compare_fp16 \
  --suite-preflight-json results/strict_fp16_suite/strict_architecture_suite_preflight.json \
  --suite-preflight-csv results/strict_fp16_suite/strict_architecture_suite_preflight.csv \
  --outdir results/strict_fp16_report \
  --fail-on-missing-requirements
```

`audit_strict_results.py`는 각 결과가 `quality_gate.py --require-ncu --require-ncu-tensor-activity`를 통과했고, `measurement_grade=strict_nvml_counter`, `baseline_match_grade=structural_baseline`, `ncu_validation_pass=true`인 selected target을 갖는지 확인한다. `quality_gate_summary.json`에 여러 `selected_targets`가 있으면 `--require-kernel/--require-baseline`과 일치하는 target을 골라 audit하며, 요구 kernel/baseline target이 없으면 실패한다. 또한 selected target이 current quality gate의 `quality_gate_selected_target=true`, `util_reference_scope=quality_pass`, `util_saturated=true`, `target_selection_note=quality_gate_first_saturation_point` evidence를 갖는지도 확인해, 오래된 selection logic으로 만든 결과가 strict report에 섞이지 않게 한다. 또한 `strict_pipeline_manifest.json`이 `fp16-strict-pipeline-manifest-v1`, `status=completed`이고 git head, binary SHA256, required kernel/baseline pair, preflight/quality/NCU/resource 산출물 evidence를 담고 있는지 확인한다. `strict_pipeline_preflight.json`도 `overall_pass=true`, `required_tools_pass=true`, `dry_run=false`, CUDA toolchain compatibility pass, manifest target row match를 만족해야 하며 `skip_preflight=1` 또는 `allow_compute_apps=1`인 diagnostic run은 final audit에서 실패한다. Strict audit은 `ncu_permission_probe/ncu_permission_probe.json`이 `permission_probe_pass=true`이고 `permission_denied=false`였는지도 별도로 요구한다. 이 probe가 없거나 `ERR_NVGPUCTRPERM`을 기록한 run은 긴 sweep 결과가 일부 남아 있어도 final pJ/bit claim에 쓰지 않는다. 최종 report는 여기에 suite preflight가 `overall_pass=true`이고 `dry_run=false`였는지, 그리고 `architecture_models/architecture_model_summary.csv`가 required architecture의 peak model을 담고 reference 재계산 오차가 1% 이내인지도 추가로 요구한다. `postprocess_strict_architectures.sh`의 최종 exit status도 audit pass와 report requirement pass가 모두 true일 때만 success다. Strict audit/report는 NCU `validation_pass=true`만 복사하지 않고 selected test row의 `common_hmma_seen=true`, `wgmma_token_seen=false`, selected baseline row의 no Tensor evidence, 그리고 test/baseline 모두의 `memory_counter_classes_complete=true`, `no_l2=true`, `no_dram=true`, `no_local_spill=true`를 별도 requirement로 노출한다. `resource_audit/thread_resource_occupancy.csv`에서는 selected test/baseline kernel의 ptxas stack/spill usage가 없는지와 resource row의 `threads`, `blocks_per_sm_requested`, `unroll` context가 selected measurement row와 같은지 확인한다. `tensor_model_utilization_pct_mean`도 유한/양수이며 기본적으로 105%를 넘지 않아야 한다. 이 sanity check가 실패하면 architecture model, clock telemetry, FLOP estimate 중 하나가 어긋났을 가능성이 크다. baseline subtraction 품질도 gate에 포함되어, 기본값으로 `incremental_energy_fraction_mean >= 0.01`이고 `baseline_energy_fraction_mean <= 0.99`인 selected target만 strict audit을 통과한다. 측정 해상도도 gate에 포함되어 test/baseline duration과 Joule 단위 신호가 너무 작으면 fail된다. NVML counter와 power trace 적분값의 cross-check는 기본적으로 warning이며, trace agreement까지 필수 조건으로 보려면 `--require-counter-trace-agreement`를 사용한다. Work-slope scaling evidence까지 final gate로 요구하려면 `audit_strict_results.py` 또는 `postprocess_strict_architectures.sh`에 `--work-slope-dir`와 `--require-work-slope`를 함께 전달한다. 이때 audit row와 report는 `work_slope_valid`, `work_slope_pj_per_bit`, `work_slope_r2`뿐 아니라 work-slope fit row의 denominator/schema/no-memory/NVML provenance fields를 기록하고, selected target과 같은 launch context의 valid work-slope가 없거나 provenance가 부족하면 실패한다. 기본 required architecture는 `ga100,gh100,ga102`이며, 하나라도 빠지거나 legacy power-trace 결과가 섞이면 nonzero로 종료한다. 최종 A100/H100/RTX3090 comparison figure는 이 audit이 통과한 결과만 해석한다.

`compare_architectures.py`도 기본적으로 `--require-kernel tensor_mma_f16acc --require-baseline tensor_baseline_mov` pair만 best/coverage 후보로 본다. 이 required pair 안에서 `target_pass=true`, `measurement_grade=strict_nvml_counter`, `ncu_required=true`, `ncu_validation_pass=true`, `ncu_validation_context_match=true`, `test_ncu_tensor_activity_observed=true`, `test_ncu_tensor_activity_pct > 0`, test common-HMMA evidence, baseline no-Tensor evidence, test/baseline no-L2/no-DRAM/no-local-spill evidence가 있는 thread point만 strict quality-gate target으로 본다. `--audit-dir`를 함께 주면 `architecture_best_fp16.csv`와 coverage publishability는 `strict_result_audit.csv`의 `audit_pass=true` row를 기준으로 하되, audit row가 required kernel/baseline과 다르면 `strict_audit_required_kernel_baseline_mismatch`로 reject된다. `postprocess_strict_architectures.sh`는 audit 뒤 compare를 실행하므로 이 audit evidence를 자동으로 넘긴다. `target_pass=true`라도 audit이 실패하면 `strict_audit_failed`로 reject되어 coverage에서 diagnostic/rejected로 남는다. `target_pass=true`라도 NCU 없이 만든 diagnostic target은 `quality_gate_target_pass_without_required_ncu_evidence`로 reject되어 coverage에서 diagnostic/rejected로 남고, NCU context는 있지만 Tensor activity/common-HMMA/no-memory evidence가 없으면 각각 `quality_gate_target_pass_without_required_tensor_activity_evidence`, `quality_gate_target_pass_without_required_hmma_evidence`, `quality_gate_target_pass_without_required_no_memory_evidence`로 reject된다. required pair가 없으면 `missing_required_kernel_baseline`으로 reject된다. `quality_pass=true`지만 utilization saturation target이 아니거나 power-trace fallback인 row도 diagnostic으로 남기되, 최종 best pJ/bit 그림에는 올리지 않는다. Architecture thread-sweep figure는 utilization plot, pJ/bit plot, Tensor model utilization plot 외에도 `architecture_thread_sweep_util_pjbit_*.png`를 생성해 x축 `threads_per_sm`, y축 utilization, color scale pJ/bit를 한 화면에서 보여준다. Diagnostic 후보까지 강제로 best table에 넣어야 할 때만 `--allow-diagnostic-best`를 사용한다. `tensor_mma_f32acc`나 `fp16_half2`를 별도 diagnostic 비교 대상으로 삼을 때만 `--require-kernel/--require-baseline`을 명시적으로 바꾼다. quality gate metadata가 없는 legacy result directory는 기본적으로 `missing_quality_gate_no_best`로 reject되며, old result를 그림에 넣어야 할 때만 `--allow-legacy-best`를 명시한다.

Strict audit/report의 기본 invariant는 GPU 없이도 smoke test로 확인할 수 있다.

```bash
python3 scripts/smoke_strict_pipeline.py
```

이 smoke는 synthetic GH100 결과 디렉터리를 만들고, `selected_targets`의 첫 항목이 요구 kernel이 아닌 경우에도 `tensor_mma_f16acc/tensor_baseline_mov` target을 정확히 선택하는지 확인한다. 또한 요구 target이 없으면 audit이 실패하는지, report의 `required kernel target selected` requirement가 pass로 전달되는지도 검증한다. 실제 A100/H100/RTX3090 pJ/bit claim을 대체하지 않고, strict postprocess code path의 회귀 검사용이다.

주요 산출물은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `architecture_best_fp16.csv` | 결과 디렉터리별 required kernel/baseline pair의 `target_pass=true` 및 `measurement_grade=strict_nvml_counter` pure FP16 후보 중 best row. strict target이 없으면 `quality_rejected=True`와 `selection_note=quality_gate_no_target_pass`, `quality_gate_target_pass_without_strict_nvml_counter`, `missing_required_kernel_baseline`, `missing_quality_gate_no_best` 등으로 표시 |
| `architecture_strict_coverage.csv`, `architecture_strict_coverage.png` | required architecture(`ga100,gh100,ga102` 기본값)별 strict NVML target coverage. 누락/diagnostic-only/strict-pass 상태를 명시해 일부 GPU 결과가 전체 비교처럼 보이지 않게 함 |
| `architecture_comparison_summary.json/csv`, `architecture_comparison_readiness.png` | required architecture coverage가 모두 strict publishable인지 JSON/CSV/figure로 고정. 일부 architecture가 missing 또는 diagnostic-only이면 `publishable=false`와 missing chip list를 기록 |
| `architecture_quality_gates.csv` | 여러 결과 디렉터리의 `quality_gates.csv`를 architecture tag와 함께 병합한 파일 |
| `architecture_best_matmul_input_pj_per_bit.png` | GPU architecture별 best logical matmul input pJ/bit 비교 |
| `architecture_best_tflops.png` | GPU architecture별 best FP16 throughput 비교 |
| `architecture_best_tensor_model_utilization.png` | GPU architecture별 best 후보의 dense Tensor Core model utilization 비교 |
| `architecture_best_incremental_energy_fraction.png` | GPU architecture별 best 후보의 incremental energy signal fraction 비교 |
| `architecture_thread_sweep_util_pjbit_*.png` | x축 `threads_per_sm`, y축 SM/GPU utilization, color scale pJ/bit로 thread sweep energy/utilization tradeoff를 한 화면에 표시 |
| `architecture_models/architecture_model_summary.csv` | A100/H100/RTX3090 dense Tensor Core peak model의 SM/clock/FLOP-per-cycle 구성과 reference 재계산 오차 |
| `architecture_models/architecture_model_dense_peak.png` | architecture별 reference dense peak와 derived dense peak 비교 |
| `architecture_models/architecture_model_per_sm_capacity.png` | architecture별 f16acc dense FP16 Tensor Core FLOP/SM/cycle 비교 |
| `architecture_models/architecture_model_accumulator_modes.png` | architecture별 f16acc/f32acc dense FP16 Tensor Core reference peak 비교 |
| `architecture_models/architecture_model_resource_limits.png` | architecture별 max thread/warp/block/register resource limit 비교 |
| `architecture_thread_sweep_util_*.png` | x축 launched threads/SM, y축 SM utilization의 multi-GPU 비교. marker는 publishable strict target, diagnostic target, diagnostic `quality_pass`, fail, legacy/no-gate 상태를 구분하고 selected/target point에는 threads/block와 pJ/bit를 표시 |
| `architecture_thread_sweep_model_util_*.png` | x축 launched threads/SM, y축 dense Tensor Core model utilization의 multi-GPU 비교. marker는 publishable strict target과 diagnostic 상태를 구분 |
| `architecture_thread_sweep_pjbit_*.png` | x축 launched threads/SM, y축 logical pJ/bit의 multi-GPU 비교. publishable strict target point에는 threads/block와 pJ/bit 값을 직접 표시하고, NCU/audit 없는 target과 non-strict legacy target은 diagnostic marker로 분리 |
| `architecture_thread_sweep_energy_fraction_*.png` | x축 launched threads/SM, y축 test energy 대비 FP16 incremental fraction과 baseline-scaled fraction의 multi-GPU 비교. target에는 threads/block, pJ/bit, incremental/base 비율을 표시 |
| `architecture_resource_occupancy.csv`, `architecture_resource_occupancy_*.png` | ptxas register 기반 static occupancy model의 architecture 비교 |
| `fp16_strict_report.md` | strict audit와 architecture compare를 묶은 publishability 중심 Markdown report |
| `fp16_strict_report_requirements.csv` | strict audit, suite preflight, denominator, NCU, resource context/spill, architecture model sanity requirement matrix |
| `fp16_strict_report_dashboard.png` | selected TFLOPS 대비 logical pJ/bit dashboard |

For memory/cache-policy calibration and DRAM pJ/bit estimates:

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/p1_memory_policy_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/p1_gpu0

python3 scripts/analyze_results.py --input results/p1_gpu0
```

생성물은 다음과 같다.

| 파일/디렉터리 | 내용 |
|---|---|
| `results/p0_gpu0/runs.jsonl` | run별 benchmark metadata |
| `results/p0_gpu0/raw/*/power.csv` | run별 power/clock/temp trace |
| `results/p0_gpu0/raw/*/sm_util.csv` | run별 dmon SM utilization trace |
| `results/p0_gpu0/raw/*/bench.json` | timed loop의 CUDA event timing과 optional NVML total-energy counter delta |
| `results/p0_gpu0/strict_pipeline_manifest_start.json` | strict pipeline 시작 시점의 invocation, git/tool/env, binary/build/matrix provenance snapshot |
| `results/p0_gpu0/strict_pipeline_manifest.json` | strict pipeline 완료 또는 실패 시점의 provenance snapshot. `status=completed/failed`, binary hash와 quality/NCU/resource 산출물 존재 여부를 포함 |
| `results/p0_gpu0/ncu_permission_probe/ncu_permission_probe.json` | 긴 sweep 전에 수행한 Nsight Compute performance-counter 권한 probe 결과 |
| `results/p0_gpu0/ncu_permission_probe/ncu_permission_probe.ncu.txt` | 권한 probe의 raw Nsight Compute log. `ERR_NVGPUCTRPERM`이면 strict run은 최종 결과가 될 수 없음 |
| `results/p0_gpu0/strict_pipeline_preflight.json` | 단일 strict pipeline의 build 전 toolchain/GPU/process preflight 결과 |
| `results/p0_gpu0/strict_pipeline_preflight.csv` | 단일 strict pipeline preflight의 CSV 요약 |
| `results/strict_fp16_suite/strict_architecture_suite_preflight.json` | suite 실행 전 tool/GPU/process preflight 결과 |
| `results/strict_fp16_suite/strict_architecture_suite_preflight.csv` | suite target별 preflight pass/fail과 required toolchain/global blocker 요약 |
| `results/strict_fp16_suite/strict_architecture_suite_runs.csv` | suite target별 strict run 결과 디렉터리와 exit code |
| `results/strict_fp16_suite/strict_architecture_suite_summary.json` | suite-level `publishable_pass`, diagnostic flag, required energy policy, preflight/run/postprocess provenance |
| `results/p0_gpu0/runtime_preflight.json` | strict pipeline 시작 전 CUDA runtime/GPU metadata probe |
| `results/p0_gpu0/architecture_preflight.json` | requested CUDA arch, detected chip, common-HMMA path 검증 결과 |
| `results/p0_gpu0/calibrated_matrix.json` | strict pipeline에서 GPU별 duration target에 맞춰 생성한 matrix |
| `results/p0_gpu0/matrix_calibration_summary.csv` | calibration에 사용한 observed elapsed, old/new repeats, action 요약 |
| `results/p0_gpu0/summary.csv` | baseline/test pair별 baseline subtraction 결과 |
| `results/p0_gpu0/condition_summary.csv` | condition별 반복 측정 통계(mean/std/min/max/95% CI) |
| `results/p0_gpu0/thread_sweep_summary.csv` | thread-count sweep일 때 thread별 utilization/TFLOPS 집계와 `selected_optimal` 표시 |
| `results/p0_gpu0/work_slope_summary.csv` | 같은 thread 조건에서 work amount가 3점 이상 바뀐 경우 incremental energy slope와 pJ/bit 회귀 진단 |
| `results/p0_gpu0/quality_gates.csv` | pair/thread point별 quality gate 통과 여부와 실패 이유 |
| `results/p0_gpu0/quality_gate_summary.json` | 선택된 target point와 gate threshold 요약 |
| `results/p0_gpu0/figures/quality_gate_target_metric_thread_sweep_*.png` | target selection에 실제 사용한 utilization metric과 selected/lowest pJ 후보 시각화 |
| `results/ncu_*/ncu_validation_summary.csv` | Nsight Compute report별 HMMA/no-L2/local-spill 자동 검증 |
| `results/ncu_*/figures/ncu_validation_summary.png` | NCU validation pass/fail 시각화 |
| `results/ncu_*/figures/ncu_memory_counter_bytes.png` | NCU DRAM/L2/local memory counter의 normalized bytes 시각화 |
| `results/ncu_*/figures/ncu_activity_pct.png` | NCU Tensor/SM activity percentage 시각화 |
| `results/p0_gpu0/resource_audit/kernel_resource_summary.csv` | ptxas kernel별 registers/thread, stack/spill bytes |
| `results/p0_gpu0/resource_audit/thread_resource_occupancy.csv` | thread sweep 후보별 static resource occupancy model. strict report는 selected row의 thread/block/unroll context 일치 여부를 별도 requirement로 표시 |
| `results/p0_gpu0/resource_audit/figures/thread_sweep_resource_occupancy.png` | launched threads/SM 대비 static occupancy와 measured SM utilization 비교 |
| `results/strict_fp16_audit/strict_result_audit.csv` | 여러 strict 결과 디렉터리의 최종 채택 가능 여부 audit |
| `results/strict_fp16_audit/strict_result_failure_summary.csv` | audit fail reason을 NCU permission/context, no-memory evidence, resource context, energy source 등 blocker category로 집계 |
| `results/strict_fp16_audit/figures/strict_result_audit.png` | architecture별 strict audit pass/fail 시각화 |
| `results/strict_fp16_audit/figures/strict_result_failure_categories.png` | strict audit blocker category별 fail count 시각화 |
| `results/strict_fp16_audit/figures/strict_result_matmul_input_pj_per_bit.png` | strict selected target의 logical FP16 input pJ/bit 비교 |
| `results/strict_fp16_audit/figures/strict_result_tflops.png` | strict selected target의 TFLOPS 비교 |
| `results/strict_fp16_audit/figures/strict_result_elapsed_s.png` | strict selected target의 test duration 비교 |
| `results/strict_fp16_audit/figures/strict_result_sm_utilization.png` | strict selected target의 SM utilization 비교 |
| `results/strict_fp16_audit/figures/strict_result_tensor_model_utilization.png` | strict selected target의 dense Tensor Core model utilization 비교 |
| `results/strict_fp16_audit/figures/strict_result_ncu_tensor_activity.png` | strict selected target의 NCU Tensor activity 비교 |
| `results/strict_fp16_audit/figures/strict_result_incremental_energy_fraction.png` | strict selected target의 incremental energy signal fraction 비교 |
| `results/strict_fp16_audit/figures/strict_result_incremental_energy_j.png` | strict selected target의 incremental energy magnitude 비교 |
| `results/strict_fp16_audit/figures/strict_result_counter_trace_ratio.png` | strict selected target의 NVML energy counter / power trace energy sanity ratio |
| `results/strict_fp16_audit/figures/strict_result_baseline_energy_fraction.png` | strict selected target의 baseline-scaled energy fraction 비교 |
| `results/strict_fp16_postprocess/architecture_models/architecture_model_summary.csv` | postprocess와 함께 기록되는 A100/H100/RTX3090 architecture model sanity table |
| `results/strict_fp16_postprocess/architecture_models/architecture_model_dense_peak.png` | reference dense TFLOPS와 model-derived dense TFLOPS 비교 |
| `results/strict_fp16_postprocess/architecture_models/architecture_model_per_sm_capacity.png` | f16acc dense FP16 Tensor Core FLOP/SM/cycle 비교 |
| `results/strict_fp16_postprocess/architecture_models/architecture_model_accumulator_modes.png` | f16acc/f32acc dense FP16 Tensor Core reference peak 비교 |
| `results/strict_fp16_postprocess/architecture_models/architecture_model_resource_limits.png` | thread/warp/block/register resource limit 비교 |
| `results/strict_fp16_report/fp16_strict_report.md` | strict audit/compare 결과를 사람이 검토하기 위한 최종 Markdown report |
| `results/strict_fp16_report/fp16_strict_report_dashboard.png` | selected TFLOPS와 logical pJ/bit를 pass/fail 색상으로 표시 |
| `results/p0_gpu0/run_level_summary.csv` | run 단위 selected energy, NVML counter delta, power trace integration 결과 |
| `results/p0_gpu0/figures/pj_per_flop_bar.png` | pJ/FLOP bar chart |
| `results/p0_gpu0/figures/tflops_vs_pj_per_flop.png` | TFLOPS vs pJ/FLOP scatter |
| `results/p0_gpu0/figures/fp16_energy_separation_stack.png` | test interval energy를 baseline-scaled energy와 FP16 incremental energy로 분리한 stack plot |
| `results/p0_gpu0/figures/thread_sweep_*.png` | launched threads/SM별 SM utilization/TFLOPS plot |
| `results/p0_gpu0/figures/thread_sweep_pjbit_*.png` | launched threads/SM별 matmul logical pJ/bit plot. label은 threads/block와 pJ/bit 값 |
| `results/p0_gpu0/figures/work_slope_*.png` | unroll/work amount 변화에 대한 incremental energy slope fit |
| `results/p0_gpu0/figures/quality_gate_thread_sweep_*.png` | quality gate pass/fail과 pJ/bit label이 포함된 thread sweep plot |
| `results/p0_gpu0/figures/power_trace_*.png` | baseline/test power trace 비교 |
| `results/p0_gpu0/figures/clock_*.png`, `temperature_*.png` | clock/temperature timeline |

## 7. P1 memory/cache policy 보정 실험

P0 결과에서 L1/L2/DRAM residual traffic이 관찰될 때만 아래 실험을 사용한다.

```bash
python3 scripts/run_experiment.py \
  --binary build/fp16_energy_bench \
  --matrix configs/p1_memory_policy_matrix.json \
  --gpu 0 \
  --sample-ms 100 \
  --repeat 10 \
  --outdir results/p1_memory_gpu0

python3 scripts/analyze_results.py --input results/p1_memory_gpu0
```

이 구현은 cache operator sweep을 과도하게 늘리지 않는다. 보정용으로만 `default`, `.cg`, `.cs`를 제공한다. `.ca`는 기본 정책과 중복되기 쉬워 primary matrix에 넣지 않았고, `.cv`는 일반 device DRAM access 강제 수단으로 오해될 수 있어 제외했다.

## 8. Nsight Compute 검증

Power 실험과 별도로 짧은 validation run을 수행한다.

```bash
./scripts/ncu_validate.sh ./build/fp16_energy_bench results/ncu_gpu0 0
```

Thread sweep에서 선택된 후보가 L2/global memory를 의도적으로 touch하지 않는지 확인하려면 no-L2 validation을 별도로 수행한다.

```bash
./scripts/ncu_validate_no_l2_thread_sweep.sh \
  ./build/fp16_energy_bench \
  results/ncu_no_l2_thread_sweep_gpu0 \
  0 \
  32,64,128,256,512,1024
```

이 validation은 `--suppress-output-store`로 required test/baseline pair의 final global store를 제거한 상태에서 Nsight Compute `MemoryWorkloadAnalysis`를 남긴다. Helper의 기본 pair는 `NCU_TEST_KERNEL=tensor_mma_f16acc`와 `NCU_BASELINE_KERNEL=tensor_baseline_mov`이며, `NCU_TEST_KERNEL=tensor_mma_f32acc`를 지정하면 baseline 기본값은 `tensor_baseline_f32`로 바뀐다. Helper는 기본적으로 strict matrix와 같은 `threads`, `blocks_per_sm=8`, `unroll=8`, `suppress_output_store=true` context를 `ncu_validation_summary.csv`에 기록하고, `validation_test_kernel`, `validation_baseline_kernel`, `validation_pair_role`과 JSON `validation_context`에 어떤 required pair를 검증했는지도 남긴다. `quality_gate.py --require-ncu --require-ncu-tensor-activity`는 이 context가 측정 row와 맞는지와 test kernel의 Tensor pipe activity evidence가 있는지도 확인한다. Launch-shape sweep처럼 여러 `blocks/SM`을 검증할 때는 `NCU_BLOCKS_PER_SM_CSV=1,2,4,8` 또는 pipeline의 `--ncu-blocks-per-sm-csv 1,2,4,8`을 사용하며, validation report 이름과 quality/audit 매칭은 `threads + blocks_per_sm`를 함께 사용한다. 같은 kernel/thread/blocks_per_sm에 대해 여러 NCU row가 있으면 quality gate와 audit는 `unroll`과 `suppress_output_store`까지 가장 정확히 맞는 row를 우선 선택한다. `threads`가 누락된 NCU report는 fallback으로 matching될 수 있어도 strict quality/audit context check에서 실패한다. `tensor_baseline_mov`는 ptxas가 empty/register-only loop를 제거하지 못하도록 no-memory warp-sync step을 사용한다. 필요한 경우 `NCU_TEST_KERNEL`, `NCU_BASELINE_KERNEL`, `NCU_BLOCKS_PER_SM`, `NCU_BLOCKS_PER_SM_CSV`, `NCU_UNROLL`, `NCU_SUPPRESS_OUTPUT_STORE`, `NCU_ITERS`, `NCU_REPEATS`, `NCU_WARMUP`, `NCU_MIN_TENSOR_ACTIVITY_PCT` 환경변수로 profiler validation run의 launch context와 Tensor activity threshold를 override한다. Diagnostic run에서만 `NCU_REQUIRE_TENSOR_ACTIVITY=0`으로 Tensor activity hard gate를 끌 수 있다. GeForce/WSL 환경에서는 NVIDIA performance counter 권한 때문에 `ERR_NVGPUCTRPERM`으로 막힐 수 있다.

두 validation helper는 실행 후 `validate_ncu_reports.py`를 호출해 다음 산출물을 만든다.

```text
results/ncu_*/ncu_validation_summary.csv
results/ncu_*/ncu_validation_summary.json
results/ncu_*/figures/ncu_validation_summary.png
```

strict mode에서는 explicit DRAM/L2/local counter class가 모두 있어야 pass된다. Nsight Compute 버전별 metric 이름 차이 때문에 counter가 빠진 경우, `ncu_validation_summary.csv`의 `fail_reasons`, `validation_*`, `*_counter_sources`를 확인해 metric set과 validation context를 조정한 뒤 다시 실행한다. `--allow-missing-counters`는 SASS token fallback을 쓰는 diagnostic 모드일 뿐, 최종 pJ/bit claim에는 사용하지 않는다. metric 이름에 `sectors`가 들어간 counter는 기본 32 bytes/sector로 normalized bytes로 변환한 뒤 threshold와 비교한다. 필요한 경우 `validate_ncu_reports.py --l2-sector-bytes`, `--dram-sector-bytes`, `--local-sector-bytes`로 조정한다.

NCU helper는 memory counter 외에 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`와 `sm__throughput.avg.pct_of_peak_sustained_elapsed`도 기본 metric set에 포함한다. `validate_ncu_reports.py`는 이 explicit metric이 없으면 `ComputeWorkloadAnalysis` section label에서 Tensor/SM activity percentage를 best-effort로 추출한다. 이 값은 `nvidia-smi dmon` SM utilization을 대체하지 않고, HMMA 경로가 실제 Tensor pipe를 사용했다는 profiler-side evidence로 사용한다. Strict pipeline과 suite postprocess는 기본적으로 이 evidence를 hard gate로 사용하며, diagnostic postprocess에서만 `--no-require-ncu-tensor-activity`로 끌 수 있다.

Tensor instruction evidence는 strict 비교에서 더 좁게 본다. `tensor_mma_*` kernel은 공통 warp-level HMMA `mma.sync.m16n8k16` path 증거가 필요하고, `smsp__inst_executed_pipe_tensor.sum` 같은 generic tensor instruction counter만으로는 pass하지 않는다. H100에서 WGMMA token이 보이면 strict A100/H100/RTX3090 비교용 결과에서는 실패 처리한다. 이 benchmark의 목적은 Hopper WGMMA 최대 성능이 아니라 세 GPU가 공통으로 실행할 수 있는 HMMA path의 pJ/bit 비교이기 때문이다. `ncu_validation_summary.csv`에는 이를 확인할 수 있도록 `common_hmma_seen`, `hmma_metric_seen`, `tensor_inst_seen`, `wgmma_token_seen` 필드가 함께 기록된다.

NCU helper는 parser가 요구하는 counter를 `--metrics`로 명시 수집한다. 기본 metric set은 `NCU_METRICS` 환경 변수로 override할 수 있다.

P0 결과 채택 기준은 최소한 다음을 확인해야 한다.

| 기준 | 확인 방법 |
|---|---|
| 의도한 FP16/HMMA instruction path | Nsight Compute `ComputeWorkloadAnalysis`, SASS 확인 |
| local memory spill 없음 | Nsight Compute memory/local 관련 항목, `ptxas -v` |
| timed loop 내부 L1/L2/DRAM traffic 최소 | Nsight Compute `MemoryWorkloadAnalysis` |
| clock 안정 | `power.csv`, `clock_*.png`, `summary.csv`의 `clock_span_mhz` |
| thermal/power throttling 없음 | `nvidia-smi -q`, power/temperature trace |
| 다른 process 없음 | `nvidia-smi pmon`, 실험 환경 격리 |

## 9. 결과 해석

`summary.csv`의 핵심 컬럼은 다음이다.

| 컬럼 | 의미 |
|---|---|
| `tflops` | CUDA event elapsed time 기준 FP16 throughput |
| `elapsed_s`, `baseline_elapsed_s` | test/baseline CUDA event elapsed time. 너무 짧으면 measurement-resolution gate에서 reject |
| `pair_index`, `repeat_index` | baseline/test pair 번호와 runner 반복 번호 |
| `test_avg_power_w` | test run의 평균 power |
| `baseline_avg_power_w` | baseline run의 평균 power |
| `test_energy_j` | test run의 selected energy. NVML total-energy counter가 있으면 그 값을 쓰고, 없으면 power trace 적분값으로 fallback |
| `baseline_energy_j` | baseline run 자체 selected energy. duration이 다를 수 있으므로 pJ/FLOP 계산에는 직접 빼지 않음 |
| `test_energy_counter_vs_trace_ratio` | NVML total-energy delta / power trace 적분 energy. 최종 energy source가 NVML일 때 trace sanity check 용도 |
| `baseline_energy_counter_vs_trace_ratio` | baseline run의 NVML total-energy delta / power trace 적분 energy |
| `baseline_scaled_energy_j` | baseline 평균 power × test elapsed time |
| `incremental_power_w` | test 평균 power - baseline 평균 power |
| `incremental_energy_j` | test_energy_j - baseline_scaled_energy_j |
| `incremental_energy_fraction` | `incremental_energy_j / test_energy_j`. 너무 작으면 baseline subtraction noise에 민감 |
| `baseline_energy_fraction` | `baseline_scaled_energy_j / test_energy_j`. 너무 크면 test energy 대부분이 baseline으로 설명됨 |
| `pj_per_flop` | incremental energy / FP16 ops × 1e12 |
| `matmul_input_pj_per_bit` | Tensor Core matmul의 A/B FP16 input bits 기준 incremental pJ/bit |
| `matmul_input_bits_per_logical_mma` | logical `m16n16k16` 한 번당 A/B FP16 input bit 수. 정상값은 `(16*16 + 16*16) * 16 = 8192` |
| `matmul_flops_per_logical_mma` | logical `m16n16k16` 한 번당 FLOP 수. 정상값은 `2 * 16 * 16 * 16 = 8192` |
| `matmul_logical_mma_count` | `fp16_ops / matmul_flops_per_logical_mma`로 해석되는 logical MMA count. 새 benchmark JSON에서는 `mma_logical_count_estimate`로 직접 기록 |
| `matmul_denominator_valid` | Tensor Core pJ/bit 분모 metadata가 logical `m16n16k16` input-bit denominator와 일치하는지 여부 |
| `matmul_denominator_source` | `bench_json_metadata`이면 benchmark binary가 직접 기록한 denominator, `derived_legacy_formula`이면 analyzer fallback으로 재계산한 legacy/diagnostic 값 |
| `matmul_denominator_metadata_complete` | strict gate에 필요한 logical MMA metadata가 benchmark JSON에 모두 있는지 여부 |
| `matmul_arithmetic_read_pj_per_bit` | A/B input bits + accumulator read bits 기준 incremental pJ/bit |
| `matmul_register_read_write_pj_per_bit` | A/B input bits + accumulator read bits + output bits 기준 incremental pJ/bit |
| `test_benchmark_schema_version`, `baseline_benchmark_schema_version` | test/baseline benchmark JSON schema. strict 결과는 둘 다 `fp16-energy-bench-v2`여야 함 |
| `test_benchmark_schema_features`, `baseline_benchmark_schema_features` | timed NVML energy counter, explicit denominator, strict denominator provenance, timed-kernel memory provenance feature marker |
| `test_bench_build_git_commit`, `baseline_bench_build_git_commit` | benchmark binary build 시 CMake가 기록한 source commit |
| `w_per_tflops` | incremental power / achieved TFLOPS |
| `avg_gpu_util_pct`, `max_gpu_util_pct` | test interval 안의 `nvidia-smi utilization.gpu` 평균/최대값 |
| `avg_sm_util_pct`, `max_sm_util_pct` | test interval 안의 `nvidia-smi dmon -s u` `sm` 컬럼 평균/최대값 |
| `test_ncu_tensor_activity_pct`, `baseline_ncu_tensor_activity_pct` | NCU validation report에서 추출한 Tensor pipe activity percentage |
| `ncu_validation_context_match` | NCU validation run의 launch context가 측정 row의 `threads`, `blocks_per_sm`, `unroll`, `suppress_output_store`와 일치하는지 여부 |
| `tensor_peak_tflops_model` | run의 SM count와 평균 SM clock으로 계산한 dense FP16 Tensor Core peak model |
| `tensor_model_accumulator_mode` | peak normalization에 사용한 accumulator mode. `tensor_mma_f16acc`는 `f16acc`, `tensor_mma_f32acc`는 `f32acc` |
| `tensor_model_flop_per_sm_cycle` | 해당 accumulator mode의 dense FP16 Tensor Core FLOP/SM/cycle |
| `achieved_flops_per_sm_cycle` | measured TFLOPS를 SM count와 평균 SM clock으로 나눈 FLOP/SM/cycle |
| `tensor_model_utilization_pct` | measured TFLOPS / `tensor_peak_tflops_model` × 100 |
| `tensor_model_reference_url` | architecture peak model의 NVIDIA reference URL |
| `suppress_output_store` | compute kernel의 final global output store를 제거했는지 여부 |
| `test_timed_kernel_global_input_loads`, `baseline_timed_kernel_global_input_loads` | timed kernel 안에서 global input load를 의도하는지 여부 |
| `test_timed_kernel_global_output_stores`, `baseline_timed_kernel_global_output_stores` | timed kernel 안에서 global output store를 의도하는지 여부 |
| `test_timed_kernel_has_intended_global_memory`, `baseline_timed_kernel_has_intended_global_memory` | test/baseline 각각이 timed kernel 안에서 의도된 global memory op를 가지는지 여부 |
| `test_timed_kernel_memory_provenance_source`, `baseline_timed_kernel_memory_provenance_source` | `bench_json_metadata`이면 benchmark JSON 직접 기록, `derived_legacy_formula`이면 analyzer fallback 추론 |
| `expected_l2_touch` | test 또는 baseline timed kernel이 의도적으로 global/L2 traffic을 만들 것으로 예상되는지 여부. 새 strict 결과에서는 두 쪽 모두 `timed_kernel_has_intended_global_memory=false`여야 함 |
| `valid_basic` | power sample, work estimate, positive incremental power/energy에 대한 최소 sanity flag. Nsight 검증을 대체하지 않음 |
| `valid_no_l2` | `valid_basic=True`이고 `expected_l2_touch=False`인 pair. 의도된 L2/global traffic이 없다는 metadata gate이며, 실제 L2 traffic 0을 증명하지는 않음 |
| `pure_fp16_candidate` | `valid_no_l2=True`이고 kernel이 FP16 half2 또는 Tensor Core FP16 compute 계열인 후보 |
| `separation_quality` | `pure_fp16_candidate_no_l2`, `valid_but_expected_l2_touch`, `invalid_or_nonpositive_increment` 등 baseline subtraction 품질 분류 |
| `separation_quality_counts` | condition/thread point 안에서 각 `separation_quality` 값이 몇 번 나왔는지 요약 |
| `stats_scope`, `stats_scope_note` | mean/std 계산에 어떤 row 집합을 썼는지와 그 이유. `all_runs_no_valid*`는 최종 후보가 아니라 diagnostic 통계 |

`thread_sweep_summary.csv`의 핵심 컬럼은 다음이다.

| 컬럼 | 의미 |
|---|---|
| `threads` | threads per block |
| `threads_per_sm` | launched threads per SM. 기본 matrix에서는 `threads * blocks_per_sm`와 같음 |
| `required_valid_no_l2_count` | 해당 thread point가 최종 후보군에 들어가기 위해 필요한 최소 valid no-L2 반복 수. 기본은 `max(3, ceil(run_count/2))` |
| `valid_no_l2_count` | `valid_basic=True`이고 `expected_l2_touch=False`인 반복 수 |
| `valid_no_l2_requirement_met` | `valid_no_l2_count >= required_valid_no_l2_count` 여부 |
| `expected_l2_touch_count` | metadata상 timed kernel이 global/L2 traffic을 의도한다고 분류된 반복 수 |
| `timed_kernel_memory_provenance_metadata_count` | test/baseline 모두 benchmark JSON에서 timed-kernel memory provenance metadata를 제공한 반복 수 |
| `timed_kernel_memory_provenance_metadata_all` | 해당 thread point의 모든 반복이 직접 memory provenance metadata를 제공했는지 여부 |
| `test_timed_kernel_has_intended_global_memory_count`, `baseline_timed_kernel_has_intended_global_memory_count` | test/baseline 중 의도된 timed-kernel global memory op가 있다고 분류된 반복 수 |
| `valid_basic_expected_l2_touch_count` | energy/power sanity는 통과했지만 no-L2 조건은 만족하지 못한 반복 수 |
| `invalid_or_nonpositive_increment_count` | baseline subtraction 뒤 incremental power/energy가 양수가 아니거나 reliable energy가 없어 `valid_basic`에 실패한 반복 수 |
| `separation_quality_counts` | thread point별 `pure_fp16_candidate_no_l2`, `valid_but_expected_l2_touch`, `invalid_or_nonpositive_increment` 분포 |
| `elapsed_s_mean`, `baseline_elapsed_s_mean` | thread point별 test/baseline 평균 CUDA event duration |
| `avg_sm_util_pct_mean` | thread point별 평균 SM utilization |
| `avg_gpu_util_pct_mean` | dmon SM utilization이 없을 때 fallback으로 쓰는 평균 GPU utilization |
| `tflops_mean` | thread point별 평균 Tensor Core throughput |
| `tensor_model_utilization_pct_mean` | thread point별 dense Tensor Core peak model 대비 평균 utilization. Tensor Core matmul target selection의 primary saturation metric이며, non-Tensor Core kernel에서는 measured SM/GPU utilization이 없을 때 diagnostic plot/selection fallback으로만 사용 |
| `incremental_energy_fraction_mean` | thread point별 평균 incremental energy signal fraction |
| `incremental_energy_j_mean` | thread point별 평균 incremental energy magnitude |
| `test_energy_counter_vs_trace_ratio_mean` | thread point별 평균 test NVML-counter/power-trace ratio |
| `baseline_energy_counter_vs_trace_ratio_mean` | thread point별 평균 baseline NVML-counter/power-trace ratio |
| `baseline_energy_fraction_mean` | thread point별 평균 baseline-scaled energy fraction |
| `matmul_input_pj_per_bit_mean` | thread point별 logical input bit 기준 pJ/bit |
| `matmul_input_bits_per_logical_mma_mean` | thread point별 logical MMA input-bit denominator. strict 결과는 8192여야 함 |
| `matmul_denominator_valid_count` | 해당 thread point에서 pJ/bit denominator metadata가 통과한 반복 수 |
| `matmul_denominator_metadata_complete_count` | 해당 thread point에서 benchmark JSON denominator metadata가 complete한 반복 수 |
| `benchmark_schema_v2_count`, `benchmark_schema_v2_all` | 해당 thread point의 test/baseline이 현재 schema에서 나온 반복 수와 all-pass 여부 |
| `benchmark_schema_features_required_all` | 해당 thread point의 모든 반복이 required `schema_features`를 포함하는지 여부 |
| `selected_optimal` | 충분한 반복 수의 valid no-L2 후보 중 Tensor Core matmul은 Tensor Core model utilization 첫 포화점, 그 외 kernel은 SM/GPU utilization 첫 포화점으로 선택한 추천 point. 후보가 없으면 어떤 row에도 표시하지 않음 |
| `selection_status`, `selection_note` | `selected_optimal`이 설정되었는지 또는 왜 설정되지 않았는지에 대한 analyzer-side selection evidence |
| `selection_util_sanity_pass`, `selection_util_sanity_note` | Analyzer selection용 utilization sanity evidence. Tensor model utilization fallback이 기본 105%를 넘으면 `selected_optimal` 후보에서 제외하고 이유를 기록 |

`stats_scope=all_runs_no_valid` 또는 `all_runs_no_valid_basic`은 해당 thread point/condition에서 `valid_basic=True`인 반복이 없었다는 뜻이다. 이 경우 mean/std는 plot과 원인 분석을 위한 전체 run 통계일 뿐, 최종 pJ/bit 후보로 쓰면 안 된다. 원인은 `separation_quality_counts`, `invalid_or_nonpositive_increment_count`, `expected_l2_touch_count`로 분리해서 본다. `valid_no_l2` 역시 “코드가 의도적으로 L2/global memory를 touch하지 않는다”는 조건이지, hardware counter 기반 증명은 아니므로 최종 보고 전에는 Nsight Compute로 `MemoryWorkloadAnalysis`를 확인한다.

Analyzer는 `all_runs_no_valid*`, 충분한 `valid_no_l2_count`가 없는 thread point, 또는 Tensor model utilization sanity를 넘는 thread point를 utilization이 높다는 이유만으로 `selected_optimal=true`로 표시하지 않는다. 예를 들어 structural baseline subtraction 뒤 `incremental_energy_j_mean < 0`이면 pJ/bit 값과 utilization은 diagnostic plot에 남기지만 target thread count로 채택하지 않는다. 이 경우 `selection_status=no_valid_no_l2_candidate` 또는 `not_selected_tensor_model_util_overmax`가 기록되고, strict target은 `quality_gate.py --require-ncu --require-ncu-tensor-activity`에서 비어 있어야 정상이다.

`work_slope_summary.csv`는 같은 `threads`/`blocks_per_sm`/kernel pair에서 `matmul_input_bits`가 3개 이상 달라질 때만 생성된다. x축은 logical matmul input bits, y축은 `incremental_energy_j`이며, slope를 `slope_matmul_input_pj_per_bit`로 기록한다. 이것은 launch/static energy와 단일 point noise를 줄이기 위한 diagnostic evidence다. 기본 strict target은 여전히 quality gate를 통과한 thread sweep row이지만, RTX 3090처럼 baseline subtraction이 음수로 흔들리는 경우 `configs/fp16_matmul_work_slope_mov.json`을 raw runner로 별도 실행해 baseline 선택과 work scaling을 점검한다.

2026-06-01 RTX 3090 work-slope diagnostic 결과는 `results/fp16_work_slope_bar_repeat30_rtx3090_20260601/`에 요약만 포함했다. 이 run은 A100/H100 strict claim이 아니라 baseline 안정성 smoke이며, baseline repeats를 30으로 늘려 NVML counter 평균 power를 안정화했다.

| threads/block | threads/SM | slope pJ/input-bit | R2 | slope valid | valid no-L2 points |
|---:|---:|---:|---:|---|---:|
| 64 | 512 | 0.202493 | 0.881501 | true | 5 |
| 128 | 1024 | 0.179064 | 0.810314 | true | 3 |

코드와 표에서 baseline/control이라는 표현은 GPU의 control unit 에너지를 의미하지 않는다. 여기서는 같은 launch/loop/register 구조에서 FP16/HMMA instruction만 제거한 기준 루프 비용을 뜻한다. 최종 Tensor Core f16acc pJ/bit에는 `tensor_baseline_mov` no-memory structural baseline을 사용한다. `tensor_baseline_u32`는 integer ALU가 커서 baseline이 test보다 비싸지는지 확인하는 diagnostic baseline이고, legacy `baseline_nop` 결과도 diagnostic으로만 본다.

`resource_audit/thread_resource_occupancy.csv`의 occupancy 값은 ptxas register count와 architecture별 thread/block/register limit을 사용한 static model이다. 이것은 measured SM utilization을 대체하지 않는다. 목적은 selected FP16 후보가 local spill 없이 실행 가능한지, 그리고 thread sweep에서 occupancy/resource limit이 utilization 포화점과 어떻게 맞물리는지 확인하는 것이다. 최종 report dashboard는 publishable strict point와 diagnostic/rejected point를 다른 marker로 표시하고, requirement matrix는 ptxas resource row의 `threads`, `blocks_per_sm_requested`, `unroll`이 selected measurement와 일치하는지도 별도 gate로 표시한다.

최종 보고서에는 `p0_cuda_core_half2_vs_nop`과 `p0_cuda_core_half2_vs_regmove`를 모두 제시하는 것이 좋다. 두 baseline 간 차이는 baseline sensitivity로 취급한다. Tensor Core는 `f16acc`와 `f32acc`를 분리 보고한다.

## 10. 단일 kernel 수동 실행 예시

```bash
./build/fp16_energy_bench \
  --device 0 \
  --kernel fp16_half2 \
  --blocks 0 \
  --blocks-per-sm 8 \
  --threads 256 \
  --iters 2000000 \
  --unroll 8 \
  --warmup 2
```

Tensor Core 예시:

```bash
./build/fp16_energy_bench --device 0 --kernel tensor_mma_f16acc --iters 1000000 --unroll 8
./build/fp16_energy_bench --device 0 --kernel tensor_mma_f32acc --iters 1000000 --unroll 8
```

Memory policy sanity 예시:

```bash
./build/fp16_energy_bench --device 0 --kernel memory_cg --iters 2048 --unroll 4 --mem-mib 1024
```

## 11. 주의사항

이 코드는 실험 자동화의 출발점이다. 실제 논문/보고서 수준의 결과로 사용하려면 다음을 반드시 수행해야 한다.

1. 각 GPU에서 `iters`를 조정해 run duration이 power sampling interval보다 충분히 길도록 만든다.
2. clock 고정 후 실제 clock trace가 안정적인지 확인한다.
3. Nsight Compute로 P0 kernel의 memory traffic과 spill을 검증한다.
4. `summary.csv`의 `valid_basic`만으로 결과를 채택하지 않는다.
5. H100/A100/RTX 3090은 power telemetry 범위와 정확도가 다르므로 GPU 간 비교 시 같은 clock 조건, default 조건, max stable 조건을 분리한다.
6. NVML total-energy counter는 device-level 누적값이므로 MIG/공유 GPU/다른 tenant가 같은 물리 GPU에서 동시에 실행되면 job energy로 분리되지 않는다. 최종 pJ/bit는 exclusive GPU 환경에서 측정한다.

## Metric definitions

P0 FP16 compute energy:

```text
baseline_scaled_energy  = avg_power_baseline * elapsed_test_s
incremental_energy      = energy_test - baseline_scaled_energy
incremental_pJ_per_FLOP = incremental_energy / fp16_ops * 1e12
```

Tensor Core FP16 matmul logical pJ/bit:

```text
mma_count                    = fp16_ops / 8192
matmul_input_bits            = mma_count * (16*16 + 16*16) * 16
matmul_input_pJ_per_bit      = incremental_energy / matmul_input_bits * 1e12
matmul_arithmetic_read_bits  = matmul_input_bits + mma_count * 16*16 * accumulator_bits
matmul_register_rw_bits      = matmul_arithmetic_read_bits + mma_count * 16*16 * accumulator_bits
```

`matmul_input_pJ_per_bit`는 DRAM bit energy가 아니라 logical `m16n16k16`의 A/B FP16 operand bit 기준 compute energy estimate다. 구현은 `mma.sync.m16n8k16` instruction 두 개로 N 방향 16 columns를 채운다. `accumulator_bits`는 `tensor_mma_f16acc`에서 16, `tensor_mma_f32acc`에서 32다. 최종 target은 `benchmark_schema_current=true`, `matmul_denominator_valid=true`, `matmul_denominator_source=bench_json_metadata`여야 하며, 이 gate가 실패하면 결과가 stale binary에서 나왔거나 pJ/bit denominator가 잘못되었거나 legacy fallback인 결과로 보고하지 않는다.

P1 memory/cache-policy energy:

```text
baseline_scaled_energy = avg_power_baseline * elapsed_test_s
incremental_pJ_per_bit = (energy_memory_test - baseline_scaled_energy) / (memory_bytes * 8) * 1e12
total_pJ_per_bit       = total_energy_memory_test / (memory_bytes * 8) * 1e12
```

Use `incremental_pJ_per_bit` for residual memory-traffic calibration. `total_pJ_per_bit` includes idle/leakage/static platform power and is usually much larger or more workload-duration dependent.
