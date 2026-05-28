#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t _err = (call);                                                       \
    if (_err != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(_err));                                       \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                               \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                          \
  do {                                                                              \
    cudaError_t _err = cudaGetLastError();                                           \
    if (_err != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA kernel launch error %s:%d: %s\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(_err));                             \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                               \
  } while (0)
