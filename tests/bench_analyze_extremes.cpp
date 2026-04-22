/// @file bench_analyze_extremes.cpp
/// Analyze extreme cases from irregular shape test.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace dnnopt;

static double measure_gemm_gflops(int M, int N, int K, int warmup = 10, int iterations = 15) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);

    for (int w = 0; w < warmup; ++w) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
    }

    Timer timer;
    double times[iterations];
    for (int t = 0; t < iterations; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        times[t] = timer.elapsed_us();
    }

    std::sort(times, times + iterations);

    // Use median, also report std dev
    double median = times[iterations/2];
    double mean = 0;
    for (int i = 0; i < iterations; ++i) mean += times[i];
    mean /= iterations;

    double variance = 0;
    for (int i = 0; i < iterations; ++i) {
        variance += (times[i] - mean) * (times[i] - mean);
    }
    double stddev = std::sqrt(variance / iterations);

    printf("  Times: median=%.1fus, mean=%.1fus, std=%.1fus (%.1f%%)\n",
           median, mean, stddev, stddev/mean*100);

    return 2.0 * M * N * K / (median * 1000.0);
}

int main() {
    printf("=== Extreme Case Analysis ===\n\n");

    // Shapes with extreme ratios from previous test
    int shapes[][3] = {
        // Best improvements
        {257, 257, 257},   // +11.69%
        {257, 511, 1023},  // +8.06%
        {15, 1024, 1024},  // +5.58%
        {10, 1024, 1024},  // +3.83%
        {9, 1024, 1024},   // +3.11%

        // Worst degradations
        {1021, 1021, 1021}, // -6.98%
        {3, 1024, 1024},    // -1.52%
        {127, 127, 1024},   // -1.93%

        // Stable cases
        {512, 512, 512},    // baseline
        {1024, 1024, 1024}, // baseline
    };

    printf("Detailed analysis with high iterations (15 iterations, 10 warmup):\n\n");

    for (int i = 0; i < 10; ++i) {
        int M = shapes[i][0], N = shapes[i][1], K = shapes[i][2];

        printf("\n=== M=%d, N=%d, K=%d ===\n", M, N, K);

        // M % Mr analysis
        printf("M mod tile: M%%4=%d, M%%6=%d, M%%8=%d\n", M%4, M%6, M%8);
        printf("N mod tile: N%%12=%d, N%%16=%d\n", N%12, N%16);

        // Heuristic with detailed timing
        get_tile_cache().clear();
        get_blocking_cache().clear();
        get_gemm_shape_cache().clear();
        unsetenv("DNNOPT_AUTOTUNE");

        printf("\nHeuristic:\n");
        double gf_h = measure_gemm_gflops(M, N, K, 10, 15);

        // Autotune with detailed timing
        get_tile_cache().clear();
        get_blocking_cache().clear();
        setenv("DNNOPT_AUTOTUNE", "1", 1);
        auto block = select_blocking_params(M, N, K);
        auto tile = select_tile_params(M, N, K);

        printf("\nAutotune (blocking=%d, tile=%dx%d):\n",
               (int)block.preset, tile.Mr, tile.Nr);
        double gf_a = measure_gemm_gflops(M, N, K, 10, 15);

        double ratio = gf_a / gf_h;
        printf("\nResult: %.2f -> %.2f GFLOPS (ratio=%.4f, %.2f%%)\n",
               gf_h, gf_a, ratio, (ratio-1)*100);

        if (ratio < 0.98) {
            printf("⚠️  DEGRADATION detected!\n");
        } else if (ratio > 1.02) {
            printf("✅ IMPROVEMENT detected!\n");
        } else {
            printf("➡️  Within noise margin.\n");
        }

        // Additional: try different blocking presets
        printf("\nBlocking presets:\n");
        const char* preset_names[] = {"Conserv", "Standard", "Moderate", "Aggress", "Maximum"};
        for (int p = 0; p < 5; ++p) {
            get_blocking_cache().clear();
            setenv("DNNOPT_AUTOTUNE", "1", 1);

            GemmShapeKey key;
            key.M = M; key.N = N; key.K = K; key.dtype = 0; key.algo = 1;
            BlockingSelection sel;
            sel.preset = (BlockingPreset)p;
            sel.valid = true;
            get_blocking_cache().insert(key.hash(), sel);

            // Quick measure (fewer iterations for speed)
            auto A = aligned_array<float>(M * K);
            auto B = aligned_array<float>(K * N);
            auto C = aligned_array<float>(M * N);
            for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
            for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);

            Timer timer;
            double times[5];
            for (int t = 0; t < 5; ++t) {
                std::memset(C.get(), 0, M * N * sizeof(float));
                timer.start();
                gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
                timer.stop();
                times[t] = timer.elapsed_us();
            }
            std::sort(times, times + 5);
            double gf_p = 2.0 * M * N * K / (times[2] * 1000.0);
            printf("  %s: %.2f GFLOPS\n", preset_names[p], gf_p);
        }
    }

    return 0;
}