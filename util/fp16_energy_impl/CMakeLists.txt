cmake_minimum_required(VERSION 3.20)
project(fp16_energy_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # H100: 90, A100: 80, RTX 3090: 86
  set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
endif()

add_executable(fp16_energy_bench
  src/fp16_energy_bench.cu
)

target_include_directories(fp16_energy_bench PRIVATE include)

target_compile_options(fp16_energy_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

set_target_properties(fp16_energy_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
