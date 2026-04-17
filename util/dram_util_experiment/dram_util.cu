// DRAM read utilization experiment: drive 25/50/75/100% read BW for 10 s each.
//
// Strategy: memory-bound streaming-read kernel on a buffer far larger than L2,
// gated by host-side duty cycling inside a 20 ms window. NVTX ranges mark each
// phase so the timeline in Nsight Systems shows labeled 10 s segments.

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CK(x)                                                                 \
    do {                                                                      \
        cudaError_t _e = (x);                                                 \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(_e));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// Streaming read: __ldcg bypasses L1 so pressure lands on L2 -> DRAM.
// `sink` write is dead unless acc hits a sentinel, which keeps the compiler
// from eliminating the loads.
__global__ void stream_read_kernel(const float4* __restrict__ in,
                                   float4* __restrict__ sink, size_t n,
                                   int passes) {
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc    = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < passes; ++p) {
        for (size_t i = tid; i < n; i += stride) {
            float4 v = __ldcg(in + i);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
    }
    if (acc.x == 1.2345e-30f) sink[tid % 1024] = acc;
}

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));

    // Buffer size: must be >> L2 so reads miss to DRAM.
    // RTX 3090 L2 =  6 MiB, A100 L2 = 40 MiB, H100 L2 = 50 MiB.
    // Default = max(1 GiB, 64 * L2). Override with DRAM_BUF_BYTES.
    size_t l2_bytes    = (size_t)prop.l2CacheSize;
    size_t auto_bytes  = std::max<size_t>(1ULL << 30, l2_bytes * 64ULL);
    size_t bytes       = auto_bytes;
    if (const char* e = std::getenv("DRAM_BUF_BYTES"))
        bytes = std::strtoull(e, nullptr, 10);
    size_t n = bytes / sizeof(float4);

    float4 *d_in = nullptr, *d_sink = nullptr;
    CK(cudaMalloc(&d_in, n * sizeof(float4)));
    CK(cudaMalloc(&d_sink, 1024 * sizeof(float4)));
    CK(cudaMemset(d_in, 1, n * sizeof(float4)));

    int threads = 256;
    int blocks  = prop.multiProcessorCount * 32;
    std::printf("[info] GPU=%s  SMs=%d  L2=%.1f MiB  buf=%.2f GiB  grid=%d x %d\n",
                prop.name, prop.multiProcessorCount, l2_bytes / (double)(1 << 20),
                bytes / (double)(1 << 30), blocks, threads);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    // ---- calibrate ms / pass ----
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    stream_read_kernel<<<blocks, threads, 0, s>>>(d_in, d_sink, n, 1);  // warmup
    CK(cudaStreamSynchronize(s));
    const int CAL = 4;
    cudaEventRecord(a, s);
    stream_read_kernel<<<blocks, threads, 0, s>>>(d_in, d_sink, n, CAL);
    cudaEventRecord(b, s);
    CK(cudaStreamSynchronize(s));
    float cal_ms = 0.f;
    cudaEventElapsedTime(&cal_ms, a, b);
    double ms_per_pass = cal_ms / CAL;
    double peak_gbps   = (double)(n * sizeof(float4)) /
                       (ms_per_pass * 1e-3) / 1e9;
    std::printf("[calib] %.3f ms/pass  ~%.1f GB/s peak DRAM read\n",
                ms_per_pass, peak_gbps);

    // ---- phases ----
    const int targets[]     = {25, 50, 75, 100};
    const double phase_ms   = 10000.0;   // 10 s each
    const double window_ms  = 20.0;      // duty cycle window

    using clock = std::chrono::steady_clock;

    for (int target : targets) {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "util_%d", target);
        nvtxRangeId_t rid = nvtxRangeStartA(tag);
        std::printf("[phase] %s start\n", tag);
        std::fflush(stdout);

        auto phase_t0 = clock::now();

        if (target >= 100) {
            // Saturate: one launch sized for the full 10 s.
            int passes = std::max(1, (int)std::round(phase_ms / ms_per_pass));
            stream_read_kernel<<<blocks, threads, 0, s>>>(d_in, d_sink, n,
                                                          passes);
            CK(cudaStreamSynchronize(s));
        } else {
            double active_ms = window_ms * target / 100.0;
            int passes = std::max(1, (int)std::round(active_ms / ms_per_pass));
            while (true) {
                auto now = clock::now();
                double elapsed =
                    std::chrono::duration<double, std::milli>(now - phase_t0)
                        .count();
                if (elapsed >= phase_ms) break;

                auto w0 = clock::now();
                stream_read_kernel<<<blocks, threads, 0, s>>>(d_in, d_sink, n,
                                                              passes);
                CK(cudaStreamSynchronize(s));
                auto w1    = clock::now();
                double w_ms =
                    std::chrono::duration<double, std::milli>(w1 - w0).count();
                double rest_ms = window_ms - w_ms;
                if (rest_ms > 0.2) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds((long)(rest_ms * 1000)));
                }
            }
        }

        nvtxRangeEnd(rid);

        // 500 ms idle gap so phase boundaries are visible in the timeline.
        nvtxRangeId_t gid = nvtxRangeStartA("gap");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        nvtxRangeEnd(gid);
    }

    CK(cudaStreamDestroy(s));
    CK(cudaFree(d_in));
    CK(cudaFree(d_sink));
    std::printf("[done]\n");
    return 0;
}
