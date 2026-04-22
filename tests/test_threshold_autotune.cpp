/// @file test_threshold_autotune.cpp
/// Test threshold autotune functionality.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace dnnopt;

static double measure_gemm_gflops(int M, int N, int K, int warmup = 5, int iterations = 7) {
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
    return 2.0 * M * N * K / (times[iterations/2] * 1000.0);
}

int main() {
    printf("=== Threshold Autotune Test ===\n\n");

    // Clear all caches
    get_tile_cache().clear();
    get_blocking_cache().clear();
    get_gemm_shape_cache().clear();
    get_threshold_cache().clear();

    // Run threshold autotune
    printf("Running threshold autotune...\n");
    setenv("DNNOPT_AUTOTUNE", "1", 1);
    ThresholdSelection thresh = autotune_thresholds();

    printf("\nAutotuned thresholds:\n");
    printf("  small_m_bound = %d (default: 8)\n", thresh.small_m_bound);
    printf("  wide_n_bound = %d (default: 48)\n", thresh.wide_n_bound);
    printf("  unpacked_thresh = %d (%.1fM, default: 4M)\n",
           thresh.unpacked_thresh,
           thresh.unpacked_thresh / (1024.0 * 1024.0));
    printf("  benchmark_gflops = %.2f\n", thresh.benchmark_gflops);

    if (!thresh.valid) {
        printf("ERROR: Threshold selection not valid!\n");
        return 1;
    }

    // Test boundary shapes with autotuned thresholds
    printf("\n=== Testing Boundary Shapes ===\n\n");

    int shapes[][3] = {
        {6, 1024, 1024},   // M=6 (small-M path if small_m_bound > 6)
        {7, 1024, 1024},   // M=7
        {8, 1024, 1024},   // M=8 (boundary)
        {9, 1024, 1024},   // M=9 (packed if small_m_bound <= 9)
        {4, 48, 1024},     // N=48 (wide driver if wide_n_bound <= 48)
        {4, 32, 1024},     // N=32 (wide driver if wide_n_bound <= 32)
        {6, 40, 1024},     // M=6, N=40
    };

    printf("Shape      Threshold  Path           GFLOPS\n");
    printf("================================================\n");

    for (int i = 0; i < 7; ++i) {
        int M = shapes[i][0], N = shapes[i][1], K = shapes[i][2];

        // Determine expected path based on thresholds
        const char* path = "unknown";
        if (M < thresh.small_m_bound) {
            if (M >= 2 && (M < 4 || N >= thresh.wide_n_bound)) {
                path = "smallm_wide";
            } else {
                path = "smallm";
            }
        } else {
            if ((int64_t)M * N * K < thresh.unpacked_thresh) {
                path = "adaptive_tile";
            } else {
                path = "packed";
            }
        }

        get_tile_cache().clear();
        get_blocking_cache().clear();
        double gf = measure_gemm_gflops(M, N, K, 5, 7);

        printf("M=%2d N=%3d K=%4d  %-10s %-12s %7.2f\n",
               M, N, K,
               (M < thresh.small_m_bound) ? "small-M" : "packed",
               path, gf);
    }

    printf("\n=== Comparison: Default vs Autotuned ===\n\n");

    // Test with default thresholds
    printf("Testing with default thresholds (small_m_bound=8, wide_n_bound=48):\n");
    get_threshold_cache().clear();
    ThresholdSelection default_thresh;
    default_thresh.small_m_bound = 8;
    default_thresh.wide_n_bound = 48;
    default_thresh.unpacked_thresh = 4 * 1024 * 1024;
    default_thresh.valid = true;
    get_threshold_cache().set(default_thresh);

    double gf_default[5];
    for (int i = 0; i < 5; ++i) {
        int M = shapes[i][0], N = shapes[i][1], K = shapes[i][2];
        get_tile_cache().clear();
        get_blocking_cache().clear();
        gf_default[i] = measure_gemm_gflops(M, N, K, 5, 7);
    }

    // Test with autotuned thresholds
    printf("Testing with autotuned thresholds:\n");
    get_threshold_cache().set(thresh);

    double gf_autotuned[5];
    for (int i = 0; i < 5; ++i) {
        int M = shapes[i][0], N = shapes[i][1], K = shapes[i][2];
        get_tile_cache().clear();
        get_blocking_cache().clear();
        gf_autotuned[i] = measure_gemm_gflops(M, N, K, 5, 7);
    }

    printf("\nShape      Default    Autotuned   Ratio\n");
    printf("=========================================\n");
    double total_default = 0, total_autotuned = 0;
    for (int i = 0; i < 5; ++i) {
        double ratio = gf_autotuned[i] / gf_default[i];
        printf("M=%2d N=%3d K=%4d  %7.2f    %7.2f    %.4f\n",
               shapes[i][0], shapes[i][1], shapes[i][2],
               gf_default[i], gf_autotuned[i], ratio);
        total_default += gf_default[i];
        total_autotuned += gf_autotuned[i];
    }

    printf("=========================================\n");
    printf("Average:   %7.2f    %7.2f    %.4f\n",
           total_default/5, total_autotuned/5, total_autotuned/total_default);

    double improvement = (total_autotuned/total_default - 1) * 100;
    printf("\nOverall improvement: %.2f%%\n", improvement);

    if (improvement > -2.0) {
        printf("\n✅ Threshold autotune PASSED (no significant degradation)\n");
        return 0;
    } else {
        printf("\n⚠️  Threshold autotune WARNING: degradation detected\n");
        return 1;
    }
}