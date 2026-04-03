# 참고문헌 (References)

> 01~08 문서 전체에서 인용된 논문, 기술 문서, 도구, 데이터 소스를 일괄 정리  
> AccelWattch 논문(MICRO 2021)의 참고문헌 번호 [1]~[51]을 기준으로, 문서에서 추가로 참조한 자료를 포함  

---

## A. 핵심 논문 (Power Modeling)

| # | 논문 | 저자 | 학회/저널 | 연도 | 문서 참조 |
|---|------|------|----------|------|----------|
| **[본문]** | **AccelWattch: A Power Modeling Framework for Modern GPUs** | V. Kandiah, S. Peverelle, M. Khairy, J. Pan, A. Manjunath, T. Rogers, T. Aamodt, N. Hardavellas | MICRO | 2021 | 01~08 전체 |
| [21] | **GPUWattch: Enabling Energy Optimizations in GPGPUs** | J. Leng, T. Hetherington, A. ElTantawy, S. Gilani, N.S. Kim, T. Aamodt, V.J. Reddi | ISCA | 2013 | 01, 03 |
| [24] | **GPUSimPow: A GPGPU Power Simulator** | J. Lucas, S. Lal, M. Andersch, M. Alvarez-Mesa, B. Juurlink | ISPASS | 2013 | 01 |
| [16] | **An Integrated GPU Power and Performance Model (IPP)** | S. Hong, H. Kim | ISCA | 2010 | 01, 02 |
| [13] | **GPGPU Power Modeling for Multi-Domain Voltage-Frequency Scaling** | J. Guerreiro, A. Ilic, N. Roma, P. Tomas | HPCA | 2018 | 01, 02 |
| [2] | **Understanding the Future of Energy Efficiency in Multi-Module GPUs (GPUJoule)** | A. Arunkumar, E. Bolotin, D. Nellans, C. Wu | HPCA | 2019 | 02 |
| [47] | **GPGPU Performance and Power Estimation Using Machine Learning** | G. Wu, J. Greathouse, A. Lyashevsky, N. Jayasena, D. Chiou | HPCA | 2015 | 02 |
| [23] | **Power Modeling for GPU Architectures Using McPAT** | J. Lim, N. Lakshminarayana, H. Kim, W. Song, S. Yalamanchili, W. Sung | ACM TADS | 2014 | 01 |

## B. GPU 아키텍처 분석

| # | 논문 | 저자 | 학회/저널 | 연도 | 문서 참조 |
|---|------|------|----------|------|----------|
| [19] | **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking** | Z. Jia, M. Maggioni, B. Staiger, D.P. Scarpazza | arXiv:1804.06826 | 2018 | 01, 02, 04 |
| [14] | **Lost in Abstraction: Pitfalls of Analyzing GPUs at the Intermediate Language Level** | A. Gutierrez et al. | HPCA | 2018 | 01, 08 |

## C. 시뮬레이터 및 프레임워크

| # | 논문 | 저자 | 학회/저널 | 연도 | 문서 참조 |
|---|------|------|----------|------|----------|
| [20] | **Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling** | M. Khairy, Z. Shen, T. Aamodt, T. Rogers | ISCA | 2020 | 01, 06, 07, 08 |
| [3] | **Analyzing CUDA Workloads Using a Detailed GPU Simulator (GPGPU-Sim)** | A. Bakhoda, G. Yuan, W. Fung, H. Wong, T. Aamodt | ISPASS | 2009 | 01, 06 |
| [45] | **NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs** | O. Villa, M. Stephenson, D. Nellans, S. Keckler | MICRO | 2019 | 01, 08 |
| [22] | **McPAT: An Integrated Power, Area, and Timing Modeling Framework** | S. Li, J.H. Ahn, R. Strong, J. Brockman, D. Tullsen, N. Jouppi | MICRO | 2009 | 01, 03, 05 |
| [5] | **Wattch: A Framework for Architectural-Level Power Analysis** | D. Brooks, V. Tiwari, M. Martonosi | ISCA | 2000 | 01 |

## D. 벤치마크 및 워크로드

| # | 이름 | 저자/출처 | 용도 | 문서 참조 |
|---|------|----------|------|----------|
| [7] | **Rodinia: A Benchmark Suite for Heterogeneous Computing** | S. Che et al. | 검증 워크로드 (backprop, hotspot, kmeans 등) | 01, 03 |
| [41] | **Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing** | J. Stratton et al. | 검증 워크로드 (mri-q, sad, sgemm) | 01, 03 |
| [32] | **CUTLASS: CUDA Templates for Linear Algebra** | NVIDIA | Tensor Core 검증 (cutlass-wmma) | 01, 03 |
| [25] | **DeepBench** | Baidu SVAIL | DL 워크로드 (CONV, LSTM, GEMM) | 01, 07 |
| [35] | **NVIDIA CUDA Samples** | NVIDIA | SDK 검증 커널 (26개) | 01 |
| [8] | **cuDNN: Efficient Primitives for Deep Learning** | S. Chetlur et al. | DL 라이브러리 | 01 |
| [38] | **cuBLAS** | NVIDIA | 선형대수 라이브러리 | 01 |
| - | **MLPerf Power** | MLCommons | ML 워크로드 전력 측정 표준 | 02 |

## E. NVIDIA 공식 기술 문서

| # | 문서 | 대상 GPU | URL | 문서 참조 |
|---|------|---------|-----|----------|
| [30] | **NVIDIA Volta Architecture Whitepaper** | V100 (GV100) | [PDF](http://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) | 01, 04 |
| - | **NVIDIA Ampere Architecture Whitepaper** | A100 (GA100) | [PDF](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) | 04, 06 |
| [31] | **NVIDIA Turing Architecture Whitepaper** | RTX 2060S | [PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) | 01 |
| [29] | **NVIDIA Pascal Architecture Whitepaper** | Titan X (P100) | [PDF](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf) | 01 |
| [27] | **NVIDIA Fermi Architecture Whitepaper** | GTX 480 | [PDF](https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf) | 01 |
| [33] | **NVML API Reference** | 전체 | [Link](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html) | 01, 03, 04 |
| [28] | **nvidia-smi Documentation** | 전체 | [PDF](http://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi367.38.pdf) | 01, 04 |
| [39] | **Nsight Compute** | 전체 | [Link](https://docs.nvidia.com/nsight-compute/) | 01, 04 |
| [37] | **PTX ISA Reference (v7.0)** | 전체 | [Link](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | 08 |
| [36] | **CUDA Instruction Set Reference** | 전체 | [Link](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref) | 06, 08 |
| [34] | **NVCC Compiler Documentation (v11.0)** | 전체 | [Link](https://docs.nvidia.com/cuda/archive/11.0/cuda-compiler-driver-nvcc/index.html) | 08 |

## F. 기술 표준 및 로드맵

| # | 표준/로드맵 | 설명 | 문서 참조 |
|---|-----------|------|----------|
| [17] | **IRDS (International Roadmap for Devices and Systems)** | 공정 기술 파라미터 (7nm, 5nm, 3nm, 2nm). CACTI의 technology.cc에 22nm까지 내장 | 01, 02, 04, 08 |
| - | **ITRS (International Technology Roadmap for Semiconductors)** | IRDS의 전신. CACTI 코드에 90nm~22nm 데이터 내장 | 05 |

## G. 이론 서적

| # | 서적 | 저자 | 출판사 | 연도 | 문서 참조 |
|---|------|------|--------|------|----------|
| [4] | **Convex Optimization** | S. Boyd, L. Vandenberghe | Cambridge University Press | 2004 | 01, 02 (QP 이론) |

## H. 도구 및 소프트웨어

| 도구 | 용도 | 라이선스 | 문서 참조 |
|------|------|---------|----------|
| **AccelWattch** | GPU cycle-level power modeling | BSD 2.0 | 전체 |
| **Accel-Sim** | GPU 성능 시뮬레이터 (SASS/PTX trace-driven) | BSD | 전체 |
| **GPGPU-Sim v4.0** | GPU functional 시뮬레이터 (AccelWattch 기반) | BSD | 01, 06, 08 |
| **McPAT** | CPU/GPU power, area, timing 모델링 | BSD | 01, 03, 05 |
| **CACTI** | 캐시/SRAM/DRAM 타이밍 및 전력 모델 | BSD | 05 (technology.cc) |
| **NVBit** | NVIDIA GPU 바이너리 계측 도구 | NVIDIA 라이선스 | 01, 07, 08 |
| **NVML** | NVIDIA GPU 관리 라이브러리 (전력/온도 모니터링) | NVIDIA | 01, 03, 04 |
| **cvxpy** | Python 볼록 최적화 라이브러리 (QP solver 대체) | Apache 2.0 | 02, 07, 08 |
| **MATLAB quadprog** | Quadratic Programming solver (현재 구현) | 상용 | 01, 03, 05 |
| **matplotlib** | Python 시각화 (Figure 생성) | PSF | 07 |

## I. GitHub 저장소

| 저장소 | URL | 용도 |
|--------|-----|------|
| Accel-Sim Framework | https://github.com/accel-sim/accel-sim-framework | 본 프로젝트 기반 |
| GPU Application Collection | https://github.com/accel-sim/gpu-app-collection | 벤치마크 워크로드 |
| CUTLASS | https://github.com/NVIDIA/cutlass | Tensor Core 벤치마크 |
| 본 연구 저장소 | https://github.com/ByungwooBangKU/accelwattch | 문서 및 분석 결과 |

## J. 기타 참조

| # | 출처 | 내용 | 문서 참조 |
|---|------|------|----------|
| [43] | TOP500 List (2021/06) | HPC 시스템 GPU 사용 통계 | 01 |
| [11] | Forbes (2019) | NVIDIA GPU AI 가속기 시장 점유 | 01 |
| [40] | Intersect360 Research | HPC 애플리케이션의 GPU 지원 현황 | 01 |
| - | arXiv:2502.20075 | A100 SM clock 210~1410MHz, 81단계 확인 | 04 |
| [9] | Neurocomputing (2016) | MAPE 정의 및 회귀 모델 오차 분석 | 01 |
| [10] | ISCA (2001) | 시뮬레이터 실험 오차 측정 방법론 | 01 |

---

## 문서별 주요 참조 매핑

| 문서 | 핵심 참조 |
|------|----------|
| **01 Whitepaper** | AccelWattch[본문], GPUWattch[21], McPAT[22], GPGPU-Sim[3], NVBit[45], Rodinia[7], Parboil[41] |
| **02 Improvement** | Guerreiro[13], GPUJoule[2], Wu-ML[47], IRDS[17], Ampere Whitepaper |
| **03 Equation Examples** | AccelWattch[본문], McPAT[22] (energy 계산 근거) |
| **04 A100 Changes** | Ampere Whitepaper, IRDS[17], arXiv:2502.20075, Jia-Volta[19] |
| **05 Errata** | quadprog_solver.m, gpgpu_sim_wrapper.cc, gen_sim_power_csv.py (소스코드 직접 참조) |
| **06 Feature Support** | ampere_opcode.h, trace_driven.cc, XML_Parse.h (소스코드 직접 참조) |
| **07 Roadmap** | AccelWattch[본문] Section 7.1, DeepBench[25], Boyd-Convex[4] |
| **08 PTX Guide** | PTX ISA[37], trace_parser.cc, accel-sim.cc (소스코드 직접 참조) |
