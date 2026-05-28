#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "cuda_check.cuh"
#include "json_escape.hpp"

namespace {

struct Args {
  std::string kernel = "fp16_half2";
  int device = 0;
  int blocks = 0;               // 0 means auto: SM count * blocks_per_sm
  int blocks_per_sm = 8;
  int threads = 256;
  int iters = 1000000;
  int unroll = 8;
  int warmup = 2;
  int repeats = 1;
  int mem_mib = 256;
  int mem_stride = 1;
  bool suppress_output_store = false;
  std::string json_out;
  bool help = false;
};

struct TimingResult {
  float elapsed_ms = 0.0f;
  uint64_t host_start_ns = 0;
  uint64_t host_end_ns = 0;
};

uint64_t now_unix_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n\n"
      << "Kernels:\n"
      << "  fp16_half2                P0 CUDA-core half2 FMA test\n"
      << "  baseline_nop             P0 control-flow/no-op baseline\n"
      << "  baseline_regmove         P0 integer/register-move baseline\n"
      << "  tensor_mma_f16acc        P0 Tensor Core MMA, FP16 input + FP16 accumulate\n"
      << "  tensor_mma_f32acc        P0 Tensor Core MMA, FP16 input + FP32 accumulate\n"
      << "  tensor_baseline_u32      P0 Tensor baseline for f16acc output shape\n"
      << "  tensor_baseline_f32      P0 Tensor baseline for f32acc output shape\n"
      << "  memory_default           P1 memory baseline, default load/store policy\n"
      << "  memory_cg                P1 memory baseline, ld/st.global.cg\n"
      << "  memory_cs                P1 memory baseline, ld/st.global.cs\n\n"
      << "Options:\n"
      << "  --device N               CUDA device index [0]\n"
      << "  --kernel NAME            kernel name [fp16_half2]\n"
      << "  --blocks N               grid blocks; 0 = SM count * --blocks-per-sm [0]\n"
      << "  --blocks-per-sm N        auto block multiplier [8]\n"
      << "  --threads N              threads per block [256]\n"
      << "  --iters N                loop iterations per kernel [1000000]\n"
      << "  --unroll N               compile-time unroll selector: 1,2,4,8,16,32,64 [8]\n"
      << "  --warmup N               warmup launches before measurement [2]\n"
      << "  --repeats N              timed launches enclosed by one CUDA event interval [1]\n"
      << "  --mem-mib N              memory working set size for memory_* kernels [256]\n"
      << "  --mem-stride N           stride for memory_* kernels [1]\n"
      << "  --suppress-output-store  skip final compute-kernel global store for no-L2 validation\n"
      << "  --json-out PATH          write JSON result to file instead of stdout\n"
      << "  --help                   show this message\n";
}

int parse_int(const char* s, const char* name) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0') {
    std::cerr << "Invalid integer for " << name << ": " << s << "\n";
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(v);
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k(argv[i]);
    auto need_value = [&](const char* opt) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << opt << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (k == "--help" || k == "-h") {
      a.help = true;
    } else if (k == "--device") {
      a.device = parse_int(need_value("--device"), "--device");
    } else if (k == "--kernel") {
      a.kernel = need_value("--kernel");
    } else if (k == "--blocks") {
      a.blocks = parse_int(need_value("--blocks"), "--blocks");
    } else if (k == "--blocks-per-sm") {
      a.blocks_per_sm = parse_int(need_value("--blocks-per-sm"), "--blocks-per-sm");
    } else if (k == "--threads") {
      a.threads = parse_int(need_value("--threads"), "--threads");
    } else if (k == "--iters") {
      a.iters = parse_int(need_value("--iters"), "--iters");
    } else if (k == "--unroll") {
      a.unroll = parse_int(need_value("--unroll"), "--unroll");
    } else if (k == "--warmup") {
      a.warmup = parse_int(need_value("--warmup"), "--warmup");
    } else if (k == "--repeats") {
      a.repeats = parse_int(need_value("--repeats"), "--repeats");
    } else if (k == "--mem-mib") {
      a.mem_mib = parse_int(need_value("--mem-mib"), "--mem-mib");
    } else if (k == "--mem-stride") {
      a.mem_stride = parse_int(need_value("--mem-stride"), "--mem-stride");
    } else if (k == "--suppress-output-store") {
      a.suppress_output_store = true;
    } else if (k == "--json-out") {
      a.json_out = need_value("--json-out");
    } else {
      std::cerr << "Unknown argument: " << k << "\n";
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  return a;
}

__global__ void init_half2_kernel(half2* p, size_t n, float base) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    float v = base + static_cast<float>((i & 1023) + 1) * 0.0001f;
    p[i] = __float2half2_rn(v);
  }
}

__global__ void init_u32_kernel(uint32_t* p, size_t n, uint32_t seed) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    p[i] = seed ^ static_cast<uint32_t>(i * 2654435761u);
  }
}

__global__ void init_float_kernel(float* p, size_t n, float base) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    p[i] = base + static_cast<float>(i & 255) * 0.001f;
  }
}

__device__ __forceinline__ uint32_t control_step(uint32_t x, int i, int u) {
  (void)i;
  (void)u;
  asm volatile("add.u32 %0, %0, 1;" : "+r"(x));
  return x;
}

__device__ __forceinline__ void consume_u32(uint32_t x) {
  asm volatile("" :: "r"(x));
}

__device__ __forceinline__ void consume_float(float x) {
  asm volatile("" :: "f"(x));
}

constexpr uint64_t kTensorLogicalM = 16;
constexpr uint64_t kTensorLogicalN = 16;
constexpr uint64_t kTensorLogicalK = 16;
constexpr uint64_t kTensorFlopsPerLogicalMma =
    2ull * kTensorLogicalM * kTensorLogicalN * kTensorLogicalK;

template <int UNROLL>
__global__ void fp16_half2_kernel(const half2* __restrict__ in, half2* __restrict__ out,
                                  int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  half2 a = in[tid];
  half2 b = __float2half2_rn(1.0009765625f);
  half2 c = __float2half2_rn(0.99951171875f);
  half2 x0 = a;
  half2 x1 = __hadd2(a, b);
  half2 x2 = __hadd2(a, c);
  half2 x3 = __hmul2(a, b);

#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      x0 = __hfma2(x0, b, c);
      x1 = __hfma2(x1, c, x0);
      x2 = __hfma2(x2, b, x1);
      x3 = __hfma2(x3, c, x2);
    }
  }

  out[tid] = __hadd2(__hadd2(x0, x1), __hadd2(x2, x3));
}

template <int UNROLL>
__global__ void baseline_nop_kernel(const half2* __restrict__ in, half2* __restrict__ out,
                                    int iters, bool suppress_output_store) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t* in_u32 = reinterpret_cast<const uint32_t*>(in);
  uint32_t seed = suppress_output_store ? static_cast<uint32_t>(tid) * 0x9e3779b9u : in_u32[tid];
  uint32_t marker = static_cast<uint32_t>(tid);
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      marker = control_step(marker, i, u);
    }
  }
  if (suppress_output_store) {
    consume_u32(seed ^ marker);
  } else {
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(out);
    out_u32[tid] = seed ^ marker;
  }
}

template <int UNROLL>
__global__ void baseline_regmove_kernel(const half2* __restrict__ in, half2* __restrict__ out,
                                        int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t* in_u32 = reinterpret_cast<const uint32_t*>(in);
  uint32_t s0 = in_u32[tid];
  uint32_t s1 = s0 ^ 0x9e3779b9u;
  uint32_t s2 = s0 + 0x7f4a7c15u;
  uint32_t s3 = s0 ^ 0x85ebca6bu;

#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      uint32_t t0 = s1;
      uint32_t t1 = s2;
      uint32_t t2 = s3;
      uint32_t t3 = s0;
      s0 = t0 ^ static_cast<uint32_t>(i + u);
      s1 = t1 + 0x01000193u;
      s2 = t2 ^ (t1 >> 3);
      s3 = t3 + (t0 << 1);
    }
  }

  uint32_t* out_u32 = reinterpret_cast<uint32_t*>(out);
  out_u32[tid] = s0 ^ s1 ^ s2 ^ s3;
}

template <int UNROLL>
__global__ void tensor_mma_f16acc_kernel(uint32_t* __restrict__ out, int iters,
                                         bool suppress_output_store) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 800
  uint32_t a0 = 0x3c003c00u;  // half2(1.0, 1.0)
  uint32_t a1 = 0x3c003c00u;
  uint32_t a2 = 0x3c003c00u;
  uint32_t a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u;
  uint32_t b1 = 0x3c003c00u;
  uint32_t b2 = 0x3c003c00u;
  uint32_t b3 = 0x3c003c00u;
  uint32_t c0 = static_cast<uint32_t>(tid & 1) ? 0x00010001u : 0x00000000u;
  uint32_t c1 = static_cast<uint32_t>(tid & 2) ? 0x00010001u : 0x00000000u;
  uint32_t c2 = static_cast<uint32_t>(tid & 4) ? 0x00010001u : 0x00000000u;
  uint32_t c3 = static_cast<uint32_t>(tid & 8) ? 0x00010001u : 0x00000000u;

#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      uint32_t d0, d1, d2, d3;
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
          : "=r"(d0), "=r"(d1)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0),
            "r"(c1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
          : "=r"(d2), "=r"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "r"(c2),
            "r"(c3));
      c0 = d0;
      c1 = d1;
      c2 = d2;
      c3 = d3;
    }
  }
  if (suppress_output_store) {
    consume_u32(c0);
    consume_u32(c1);
    consume_u32(c2);
    consume_u32(c3);
  } else {
    const size_t base = static_cast<size_t>(tid) * 4;
    out[base + 0] = c0;
    out[base + 1] = c1;
    out[base + 2] = c2;
    out[base + 3] = c3;
  }
#else
  if (tid == 0) out[0] = 0u;
#endif
}

template <int UNROLL>
__global__ void tensor_mma_f32acc_kernel(float* __restrict__ out, int iters,
                                         bool suppress_output_store) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 800
  uint32_t a0 = 0x3c003c00u;  // half2(1.0, 1.0)
  uint32_t a1 = 0x3c003c00u;
  uint32_t a2 = 0x3c003c00u;
  uint32_t a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u;
  uint32_t b1 = 0x3c003c00u;
  uint32_t b2 = 0x3c003c00u;
  uint32_t b3 = 0x3c003c00u;
  float c0 = static_cast<float>(tid & 31) * 0.001f;
  float c1 = c0 + 0.01f;
  float c2 = c0 + 0.02f;
  float c3 = c0 + 0.03f;
  float c4 = c0 + 0.04f;
  float c5 = c0 + 0.05f;
  float c6 = c0 + 0.06f;
  float c7 = c0 + 0.07f;

#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      float d0, d1, d2, d3, d4, d5, d6, d7;
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
            "f"(c1), "f"(c2), "f"(c3));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          : "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "f"(c4),
            "f"(c5), "f"(c6), "f"(c7));
      c0 = d0;
      c1 = d1;
      c2 = d2;
      c3 = d3;
      c4 = d4;
      c5 = d5;
      c6 = d6;
      c7 = d7;
    }
  }
  if (suppress_output_store) {
    consume_float(c0);
    consume_float(c1);
    consume_float(c2);
    consume_float(c3);
    consume_float(c4);
    consume_float(c5);
    consume_float(c6);
    consume_float(c7);
  } else {
    const size_t base = static_cast<size_t>(tid) * 8;
    out[base + 0] = c0;
    out[base + 1] = c1;
    out[base + 2] = c2;
    out[base + 3] = c3;
    out[base + 4] = c4;
    out[base + 5] = c5;
    out[base + 6] = c6;
    out[base + 7] = c7;
  }
#else
  if (tid == 0) out[0] = 0.0f;
#endif
}

template <int UNROLL>
__global__ void tensor_baseline_u32_kernel(uint32_t* __restrict__ out, int iters,
                                           bool suppress_output_store) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t c0 = 0x00010001u ^ static_cast<uint32_t>(tid);
  uint32_t c1 = 0x00020002u ^ static_cast<uint32_t>(tid << 1);
  uint32_t c2 = 0x00030003u ^ static_cast<uint32_t>(tid << 2);
  uint32_t c3 = 0x00040004u ^ static_cast<uint32_t>(tid << 3);
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      c0 = control_step(c0, i, u);
      c1 ^= c0;
      c2 = control_step(c2, i, u + 1);
      c3 ^= c2;
    }
  }
  if (suppress_output_store) {
    consume_u32(c0);
    consume_u32(c1);
    consume_u32(c2);
    consume_u32(c3);
  } else {
    const size_t base = static_cast<size_t>(tid) * 4;
    out[base + 0] = c0;
    out[base + 1] = c1;
    out[base + 2] = c2;
    out[base + 3] = c3;
  }
}

template <int UNROLL>
__global__ void tensor_baseline_f32_kernel(float* __restrict__ out, int iters,
                                           bool suppress_output_store) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float c0 = static_cast<float>(tid & 31) * 0.001f;
  float c1 = c0 + 0.01f;
  float c2 = c0 + 0.02f;
  float c3 = c0 + 0.03f;
  float c4 = c0 + 0.04f;
  float c5 = c0 + 0.05f;
  float c6 = c0 + 0.06f;
  float c7 = c0 + 0.07f;
  uint32_t marker = static_cast<uint32_t>(tid);
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      marker = control_step(marker, i, u);
    }
  }
  if (suppress_output_store) {
    consume_u32(marker);
    consume_float(c1);
    consume_float(c2);
    consume_float(c3);
    consume_float(c4);
    consume_float(c5);
    consume_float(c6);
    consume_float(c7);
  } else {
    const size_t base = static_cast<size_t>(tid) * 8;
    out[base + 0] = static_cast<float>(marker);
    out[base + 1] = c1;
    out[base + 2] = c2;
    out[base + 3] = c3;
    out[base + 4] = c4;
    out[base + 5] = c5;
    out[base + 6] = c6;
    out[base + 7] = c7;
  }
}

template <int POLICY>
__device__ __forceinline__ uint32_t ld_policy(const uint32_t* p) {
  uint32_t v;
  if constexpr (POLICY == 1) {
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  } else if constexpr (POLICY == 2) {
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
  } else {
    v = *p;
  }
  return v;
}

template <int POLICY>
__device__ __forceinline__ void st_policy(uint32_t* p, uint32_t v) {
  if constexpr (POLICY == 1) {
    asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(v));
  } else if constexpr (POLICY == 2) {
    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(v));
  } else {
    *p = v;
  }
}

template <int UNROLL, int POLICY>
__global__ void memory_policy_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                     size_t n_words, int iters, int stride_words) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t n_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t acc = static_cast<uint32_t>(tid);

#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      size_t offset = (tid + (static_cast<size_t>(i) * UNROLL + u) *
                               static_cast<size_t>(stride_words) * n_threads) %
                      n_words;
      uint32_t v = ld_policy<POLICY>(in + offset);
      acc ^= v + static_cast<uint32_t>(u);
      st_policy<POLICY>(out + offset, acc);
    }
  }
}

bool is_half2_kernel(const std::string& k) {
  return k == "fp16_half2" || k == "baseline_nop" || k == "baseline_regmove";
}

bool is_tensor_u32_kernel(const std::string& k) {
  return k == "tensor_mma_f16acc" || k == "tensor_baseline_u32";
}

bool is_tensor_f32_kernel(const std::string& k) {
  return k == "tensor_mma_f32acc" || k == "tensor_baseline_f32";
}

bool is_memory_kernel(const std::string& k) {
  return k == "memory_default" || k == "memory_cg" || k == "memory_cs";
}

bool supports_suppress_output_store(const std::string& k) {
  return k == "baseline_nop" || is_tensor_u32_kernel(k) || is_tensor_f32_kernel(k);
}

template <int UNROLL>
TimingResult launch_timed(const Args& args, int blocks, int threads, size_t total_threads, size_t mem_words) {
  float ms = 0.0f;
  uint64_t timed_host_start_ns = 0;
  uint64_t timed_host_end_ns = 0;

  if (is_half2_kernel(args.kernel)) {
    half2* d_in = nullptr;
    half2* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_threads * sizeof(half2)));
    CUDA_CHECK(cudaMalloc(&d_out, total_threads * sizeof(half2)));
    init_half2_kernel<<<blocks, threads>>>(d_in, total_threads, 1.0f);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&]() {
      if (args.kernel == "fp16_half2") {
        fp16_half2_kernel<UNROLL><<<blocks, threads>>>(d_in, d_out, args.iters);
      } else if (args.kernel == "baseline_nop") {
        baseline_nop_kernel<UNROLL><<<blocks, threads>>>(d_in, d_out, args.iters,
                                                         args.suppress_output_store);
      } else if (args.kernel == "baseline_regmove") {
        baseline_regmove_kernel<UNROLL><<<blocks, threads>>>(d_in, d_out, args.iters);
      }
      CUDA_KERNEL_CHECK();
    };

    for (int i = 0; i < args.warmup; ++i) {
      run();
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    timed_host_start_ns = now_unix_ns();
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < args.repeats; ++r) run();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    timed_host_end_ns = now_unix_ns();
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
  } else if (is_tensor_u32_kernel(args.kernel)) {
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total_threads * 4 * sizeof(uint32_t)));
    init_u32_kernel<<<blocks, threads>>>(d_out, total_threads * 4, 0x12345678u);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&]() {
      if (args.kernel == "tensor_mma_f16acc") {
        tensor_mma_f16acc_kernel<UNROLL><<<blocks, threads>>>(d_out, args.iters,
                                                             args.suppress_output_store);
      } else {
        tensor_baseline_u32_kernel<UNROLL><<<blocks, threads>>>(d_out, args.iters,
                                                                args.suppress_output_store);
      }
      CUDA_KERNEL_CHECK();
    };

    for (int i = 0; i < args.warmup; ++i) {
      run();
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    timed_host_start_ns = now_unix_ns();
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < args.repeats; ++r) run();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    timed_host_end_ns = now_unix_ns();
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
  } else if (is_tensor_f32_kernel(args.kernel)) {
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total_threads * 8 * sizeof(float)));
    init_float_kernel<<<blocks, threads>>>(d_out, total_threads * 8, 0.0f);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&]() {
      if (args.kernel == "tensor_mma_f32acc") {
        tensor_mma_f32acc_kernel<UNROLL><<<blocks, threads>>>(d_out, args.iters,
                                                             args.suppress_output_store);
      } else {
        tensor_baseline_f32_kernel<UNROLL><<<blocks, threads>>>(d_out, args.iters,
                                                                args.suppress_output_store);
      }
      CUDA_KERNEL_CHECK();
    };

    for (int i = 0; i < args.warmup; ++i) {
      run();
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    timed_host_start_ns = now_unix_ns();
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < args.repeats; ++r) run();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    timed_host_end_ns = now_unix_ns();
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
  } else if (is_memory_kernel(args.kernel)) {
    uint32_t* d_in = nullptr;
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, mem_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, mem_words * sizeof(uint32_t)));
    init_u32_kernel<<<blocks, threads>>>(d_in, mem_words, 0xdecafbadu);
    init_u32_kernel<<<blocks, threads>>>(d_out, mem_words, 0x0badf00du);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&]() {
      if (args.kernel == "memory_cg") {
        memory_policy_kernel<UNROLL, 1><<<blocks, threads>>>(d_in, d_out, mem_words, args.iters,
                                                             args.mem_stride);
      } else if (args.kernel == "memory_cs") {
        memory_policy_kernel<UNROLL, 2><<<blocks, threads>>>(d_in, d_out, mem_words, args.iters,
                                                             args.mem_stride);
      } else {
        memory_policy_kernel<UNROLL, 0><<<blocks, threads>>>(d_in, d_out, mem_words, args.iters,
                                                             args.mem_stride);
      }
      CUDA_KERNEL_CHECK();
    };

    for (int i = 0; i < args.warmup; ++i) {
      run();
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    timed_host_start_ns = now_unix_ns();
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < args.repeats; ++r) run();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    timed_host_end_ns = now_unix_ns();
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
  } else {
    std::cerr << "Unsupported kernel: " << args.kernel << "\n";
    std::exit(EXIT_FAILURE);
  }
  return TimingResult{ms, timed_host_start_ns, timed_host_end_ns};
}

TimingResult launch_by_unroll(const Args& args, int blocks, int threads, size_t total_threads, size_t mem_words) {
  switch (args.unroll) {
    case 1: return launch_timed<1>(args, blocks, threads, total_threads, mem_words);
    case 2: return launch_timed<2>(args, blocks, threads, total_threads, mem_words);
    case 4: return launch_timed<4>(args, blocks, threads, total_threads, mem_words);
    case 8: return launch_timed<8>(args, blocks, threads, total_threads, mem_words);
    case 16: return launch_timed<16>(args, blocks, threads, total_threads, mem_words);
    case 32: return launch_timed<32>(args, blocks, threads, total_threads, mem_words);
    case 64: return launch_timed<64>(args, blocks, threads, total_threads, mem_words);
    default:
      std::cerr << "Unsupported --unroll. Use one of 1,2,4,8,16,32,64.\n";
      std::exit(EXIT_FAILURE);
  }
}

uint64_t fp16_ops_estimate(const Args& args, int blocks, int threads) {
  const uint64_t total_threads = static_cast<uint64_t>(blocks) * threads;
  const uint64_t repeats = static_cast<uint64_t>(args.repeats);
  const uint64_t iters = static_cast<uint64_t>(args.iters);
  const uint64_t unroll = static_cast<uint64_t>(args.unroll);
  if (args.kernel == "fp16_half2") {
    // Per inner unroll: 4 half2 FMA instructions. One half2 FMA = 2 lanes * 2 FLOP = 4 FLOP.
    return total_threads * repeats * iters * unroll * 4ull * 4ull;
  }
  if (args.kernel == "tensor_mma_f16acc" || args.kernel == "tensor_mma_f32acc") {
    // One logical m16n16k16 tile is implemented as two m16n8k16 warp-level MMA instructions.
    const uint64_t warps = total_threads / 32ull;
    return warps * repeats * iters * unroll * kTensorFlopsPerLogicalMma;
  }
  return 0ull;
}

uint64_t memory_bytes_estimate(const Args& args, int blocks, int threads) {
  if (!is_memory_kernel(args.kernel)) return 0ull;
  const uint64_t total_threads = static_cast<uint64_t>(blocks) * threads;
  const uint64_t accesses = total_threads * static_cast<uint64_t>(args.repeats) *
                            static_cast<uint64_t>(args.iters) *
                            static_cast<uint64_t>(args.unroll);
  return accesses * 8ull;  // one 32-bit load + one 32-bit store
}

std::string cache_policy_label(const std::string& kernel) {
  if (kernel == "memory_cg") return "cg";
  if (kernel == "memory_cs") return "cs";
  if (kernel == "memory_default") return "default";
  return "none";
}

std::string fp16_path_label(const std::string& kernel) {
  if (kernel == "fp16_half2") return "cuda_core_half2_fma";
  if (kernel == "tensor_mma_f16acc") return "tensor_core_mma_m16n16k16_f16acc";
  if (kernel == "tensor_mma_f32acc") return "tensor_core_mma_m16n16k16_f32acc";
  return "baseline_or_memory";
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (args.help) {
    usage(argv[0]);
    return 0;
  }

  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

  if (args.blocks_per_sm <= 0 || args.threads <= 0 || args.iters <= 0 || args.repeats <= 0 ||
      args.warmup < 0 || args.mem_mib <= 0 || args.mem_stride <= 0) {
    std::cerr << "blocks-per-sm, threads, iters, repeats, mem-mib, and mem-stride "
              << "must be positive; warmup must be non-negative.\n";
    return EXIT_FAILURE;
  }
  if (args.blocks <= 0) {
    args.blocks = prop.multiProcessorCount * args.blocks_per_sm;
  }
  if (args.blocks <= 0) {
    std::cerr << "blocks must be positive after auto-selection.\n";
    return EXIT_FAILURE;
  }
  if ((is_tensor_u32_kernel(args.kernel) || is_tensor_f32_kernel(args.kernel)) &&
      (args.threads % 32 != 0)) {
    std::cerr << "Tensor kernels require --threads to be a multiple of 32.\n";
    return EXIT_FAILURE;
  }
  if ((is_tensor_u32_kernel(args.kernel) || is_tensor_f32_kernel(args.kernel)) &&
      prop.major < 8) {
    std::cerr << "Tensor MMA inline PTX kernels require compute capability >= 8.0.\n";
    return EXIT_FAILURE;
  }
  if (args.suppress_output_store && !supports_suppress_output_store(args.kernel)) {
    std::cerr << "--suppress-output-store is supported only for baseline_nop and Tensor kernels.\n";
    return EXIT_FAILURE;
  }

  const size_t total_threads = static_cast<size_t>(args.blocks) * args.threads;
  const size_t mem_words = (static_cast<size_t>(args.mem_mib) * 1024ull * 1024ull) / sizeof(uint32_t);

  // Warm start runtime before host timestamp interval.
  CUDA_CHECK(cudaFree(nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  const TimingResult timing = launch_by_unroll(args, args.blocks, args.threads, total_threads, mem_words);
  CUDA_CHECK(cudaDeviceSynchronize());
  const uint64_t host_start_ns = timing.host_start_ns;
  const uint64_t host_end_ns = timing.host_end_ns;
  const float elapsed_ms = timing.elapsed_ms;

  const uint64_t ops = fp16_ops_estimate(args, args.blocks, args.threads);
  const uint64_t mem_bytes = memory_bytes_estimate(args, args.blocks, args.threads);
  const double elapsed_s = static_cast<double>(elapsed_ms) / 1000.0;
  const double tflops = ops > 0 && elapsed_s > 0.0 ? static_cast<double>(ops) / elapsed_s / 1.0e12 : 0.0;
  const double gbps = mem_bytes > 0 && elapsed_s > 0.0 ? static_cast<double>(mem_bytes) / elapsed_s / 1.0e9 : 0.0;

  std::ostringstream os;
  os << std::setprecision(12);
  os << "{\n";
  os << "  \"schema_version\": \"fp16-energy-bench-v1\",\n";
  os << "  \"kernel\": \"" << json_escape(args.kernel) << "\",\n";
  os << "  \"fp16_path\": \"" << fp16_path_label(args.kernel) << "\",\n";
  os << "  \"cache_policy\": \"" << cache_policy_label(args.kernel) << "\",\n";
  os << "  \"device_index\": " << args.device << ",\n";
  os << "  \"device_name\": \"" << json_escape(prop.name) << "\",\n";
  os << "  \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
  os << "  \"sm_count\": " << prop.multiProcessorCount << ",\n";
  os << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n";
  os << "  \"shared_mem_per_block\": " << prop.sharedMemPerBlock << ",\n";
  os << "  \"regs_per_block\": " << prop.regsPerBlock << ",\n";
  os << "  \"blocks\": " << args.blocks << ",\n";
  os << "  \"threads\": " << args.threads << ",\n";
  os << "  \"blocks_per_sm_requested\": " << args.blocks_per_sm << ",\n";
  os << "  \"iters\": " << args.iters << ",\n";
  os << "  \"unroll\": " << args.unroll << ",\n";
  os << "  \"warmup\": " << args.warmup << ",\n";
  os << "  \"repeats\": " << args.repeats << ",\n";
  os << "  \"mem_mib\": " << args.mem_mib << ",\n";
  os << "  \"mem_stride\": " << args.mem_stride << ",\n";
  os << "  \"suppress_output_store\": " << (args.suppress_output_store ? "true" : "false") << ",\n";
  if (args.kernel == "tensor_mma_f16acc" || args.kernel == "tensor_mma_f32acc") {
    os << "  \"mma_m\": " << kTensorLogicalM << ",\n";
    os << "  \"mma_n\": " << kTensorLogicalN << ",\n";
    os << "  \"mma_k\": " << kTensorLogicalK << ",\n";
    os << "  \"mma_instructions_per_logical_mma\": 2,\n";
    os << "  \"mma_flops_per_logical_mma\": " << kTensorFlopsPerLogicalMma << ",\n";
  }
  os << "  \"host_start_unix_ns\": " << host_start_ns << ",\n";
  os << "  \"host_end_unix_ns\": " << host_end_ns << ",\n";
  os << "  \"cuda_elapsed_ms\": " << elapsed_ms << ",\n";
  os << "  \"fp16_ops_estimate\": " << ops << ",\n";
  os << "  \"memory_bytes_estimate\": " << mem_bytes << ",\n";
  os << "  \"estimated_tflops\": " << tflops << ",\n";
  os << "  \"estimated_memory_gbps\": " << gbps << "\n";
  os << "}\n";

  if (!args.json_out.empty()) {
    std::ofstream f(args.json_out);
    if (!f) {
      std::cerr << "Failed to open --json-out path: " << args.json_out << "\n";
      return EXIT_FAILURE;
    }
    f << os.str();
  } else {
    std::cout << os.str();
  }
  return 0;
}
