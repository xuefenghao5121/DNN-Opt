/// @file bench_threading.cpp
/// Multi-threading performance benchmark for GEMM.
///
/// Tests scaling across different thread counts and problem sizes.
/// Evaluates:
///   - OpenMP parallel scaling
///   - 2D thread decomposition (M × N threading)
///   - big.LITTLE awareness
///   - False sharing prevention

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_threading.h"
#include "dnnopt/timer.h"
#include "dnnopt/arm_hwcaps.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kRepeats = 10;
constexpr double kMinTimeSeconds = 0.5;

// Test problem sizes
struct ProblemSize {
    int M, N, K;
    const char* name;
};

constexpr ProblemSize kProblemSizes[] = {
    // Small problems (should stay single-threaded)
    {64, 64, 64, "64x64x64"},
    {128, 128, 128, "128x128x128"},

    // Medium problems
    {256, 256, 256, "256x256x256"},
    {512, 512, 256, "512x512x256"},

    // Large problems
    {1024, 1024, 512, "1024x1024x512"},
    {2048, 2048, 512, "2048x2048x512"},

    // Tall-skinny (2D threading helps)
    {2048, 128, 256, "2048x128x256_ts"},
    {4096, 64, 256, "4096x64x256_ts"},

    // Short-wide
    {128, 2048, 256, "128x2048x256_sw"},
    {64, 4096, 256, "64x4096x256_sw"},
};

/// Allocate aligned matrix.
std::vector<float> alloc_matrix(int rows, int cols, int lda) {
    std::vector<float> mat(lda * cols, 0.0f);
    return mat;
}

/// Initialize matrix with random values.
void init_matrix(int rows, int cols, int lda, float* mat, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            mat[i + j * lda] = dist(rng);
        }
    }
}

/// Benchmark result structure.
struct ThreadBenchResult {
    int num_threads;
    double time_ms;
    double gflops;
    double efficiency;
};

/// Benchmark GEMM with specific thread count.
ThreadBenchResult bench_gemm_threads(int M, int N, int K, int num_threads,
                                      const float* A, int lda,
                                      const float* B, int ldb,
                                      float* C, int ldc) {
    // Set thread count
    dnnopt::gemm_set_num_threads(num_threads);

    // Warm-up
    for (int i = 0; i < 3; ++i) {
        dnnopt::gemm_fp32(M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
    }

    // Timing loop
    int repeats = kRepeats;
    double total_time = 0.0;

    dnnopt::Timer timer;
    do {
        for (int i = 0; i < repeats; ++i) {
            timer.start();
            dnnopt::gemm_fp32(M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
            timer.stop();
            total_time += timer.elapsed_sec();
        }

        if (total_time < kMinTimeSeconds) {
            repeats *= 2;
            total_time = 0.0;
        }
    } while (total_time < kMinTimeSeconds);

    double avg_time_ms = (total_time / repeats) * 1000.0;
    double gflops = (2.0 * M * N * K) / (total_time / repeats) / 1e9;

    ThreadBenchResult result;
    result.num_threads = num_threads;
    result.time_ms = avg_time_ms;
    result.gflops = gflops;
    result.efficiency = 0.0;  // Will be computed relative to baseline
    return result;
}

}  // namespace

// ============================================================
// Main benchmark
// ============================================================

int main(int argc, char** argv) {
    printf("=================================================================\n");
    printf("GEMM Multi-Threading Performance Benchmark\n");
    printf("=================================================================\n\n");

    // Print hardware info
    auto& profile = dnnopt::detect_arm_hwcaps();
    printf("Hardware: %s\n", profile.cpu_name.c_str());
    printf("Cores: %u @ %u MHz\n", profile.num_cores, profile.freq_mhz);
    printf("Topologies: %s\n",
           profile.topology.is_heterogeneous ? "big.LITTLE" : "Homogeneous");
    printf("  Big cores: %u\n", profile.topology.big_cores);
    printf("  Little cores: %u\n", profile.topology.little_cores);
    printf("\n");

    // RNG for initialization
    std::mt19937 rng(42);

    // Determine max threads to test
    int max_threads = std::min(8, static_cast<int>(profile.num_cores));

    // Header
    printf("%-16s %8s %10s %10s %10s %10s %8s\n",
           "Size", "Threads", "Time (ms)", "GFLOPS", "Speedup", "Efficiency", "Shape");
    printf("%-16s %8s %10s %10s %10s %10s %8s\n",
           "----------------", "--------", "----------", "----------", "----------",
           "----------", "--------");

    for (const auto& size : kProblemSizes) {
        int M = size.M, N = size.N, K = size.K;
        int64_t flops = 2LL * M * N * K;

        // Allocate matrices
        int lda = K, ldb = N, ldc = N;
        auto A = alloc_matrix(M, K, lda);
        auto B = alloc_matrix(K, N, ldb);
        auto C = alloc_matrix(M, N, ldc);

        // Initialize
        init_matrix(M, K, lda, A.data(), rng);
        init_matrix(K, N, ldb, B.data(), rng);

        // Baseline (1 thread)
        auto baseline = bench_gemm_threads(M, N, K, 1,
                                           A.data(), lda, B.data(), ldb, C.data(), ldc);

        printf("%-16s %8d %10.4f %10.2f %10.2fx %10.1f%% %8s\n",
               size.name, 1, baseline.time_ms, baseline.gflops,
               1.0, 100.0, "");

        // Test different thread counts
        for (int t = 2; t <= max_threads; ++t) {
            auto result = bench_gemm_threads(M, N, K, t,
                                             A.data(), lda, B.data(), ldb, C.data(), ldc);

            double speedup = baseline.time_ms / result.time_ms;
            double efficiency = (speedup / t) * 100.0;

            printf("%-16s %8d %10.4f %10.2f %10.2fx %10.1f%% %8s\n",
                   "", t, result.time_ms, result.gflops,
                   speedup, efficiency, "");

            // Check for poor scaling (indicates potential issues)
            if (efficiency < 50.0 && t <= 4) {
                printf("  ⚠ Warning: Poor parallel efficiency at %d threads\n", t);
            }
        }

        printf("\n");
    }

    // Thread affinity test
    printf("=================================================================\n");
    printf("Thread Affinity Test (big.LITTLE awareness)\n");
    printf("=================================================================\n\n");

    // Test effective thread selection for different problem sizes
    auto& topo = profile.topology;

    struct AffinityTest {
        int64_t flops;
        int expected_threads;
        const char* description;
    };

    AffinityTest affinity_tests[] = {
        {100000, 1, "Tiny problem (< 100K FLOPs)"},
        {1000000, static_cast<int>(std::min(4u, topo.big_cores)), "Small problem (1M FLOPs)"},
        {10000000, static_cast<int>(topo.big_cores), "Medium problem (10M FLOPs)"},
        {100000000, static_cast<int>(topo.big_cores + topo.little_cores), "Large problem (100M FLOPs)"},
    };

    printf("Effective thread selection based on problem size:\n");
    printf("(threshold = 50000 FLOPs for threading)\n\n");

    for (const auto& test : affinity_tests) {
        int threads = dnnopt::get_effective_threads(topo, test.flops, 50000);
        printf("%-30s -> %d threads (expected: %d)\n",
               test.description, threads, test.expected_threads);
    }

    printf("\n");
    printf("Benchmark complete.\n");

    return 0;
}
