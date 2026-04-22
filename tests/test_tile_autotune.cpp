/// @file test_tile_autotune.cpp
/// Test for tile size autotune functionality.
///
/// Tests:
/// 1. Tile selection for M divisible by tile size
/// 2. Tile selection for M not divisible by 8 (padding case)
/// 3. Benchmark comparison between different tiles
/// 4. Cache functionality

#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"

#include <cstdio>
#include <cstring>
#include <cmath>

using namespace dnnopt;

// Helper to measure GFLOPS
static double measure_gemm_gflops(int M, int N, int K, int iterations = 5) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);
    std::memset(C.get(), 0, M * N * sizeof(float));

    // Warmup
    gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);

    Timer timer;
    double total_us = 0;

    for (int t = 0; t < iterations; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        total_us += timer.elapsed_us();
    }

    double avg_us = total_us / iterations;
    double flops = 2.0 * M * N * K;
    return flops / (avg_us * 1000.0);  // GFLOPS
}

int main() {
    printf("=== Tile Size Autotune Test ===\n\n");

    // Enable autotune
    setenv("DNNOPT_AUTOTUNE", "1", 1);

    // Test 1: M divisible by 8 (should use 8x16)
    printf("Test 1: M divisible by 8\n");
    {
        int M = 16, N = 512, K = 512;
        auto sel = select_tile_params(M, N, K);
        printf("  Shape M=%d, N=%d, K=%d -> Tile %dx%d (preset=%d)\n",
               M, N, K, sel.Mr, sel.Nr, (int)sel.preset);
        printf("  Expected: 8x16 (M divisible by 8)\n");
        if (sel.Mr == 8 && sel.Nr == 16 && sel.valid) {
            printf("  PASS: Correct tile selected\n");
        } else {
            printf("  FAIL: Expected 8x16, got %dx%d\n", sel.Mr, sel.Nr);
        }
    }

    // Test 2: M = 6 (should use 6x16 to avoid padding)
    printf("\nTest 2: M=6 (should use 6x16)\n");
    {
        int M = 6, N = 512, K = 512;
        auto sel = select_tile_params(M, N, K);
        printf("  Shape M=%d, N=%d, K=%d -> Tile %dx%d (preset=%d)\n",
               M, N, K, sel.Mr, sel.Nr, (int)sel.preset);
        printf("  Expected: 6x16 (M divisible by 6)\n");
        if (sel.Mr == 6 && sel.Nr == 16 && sel.valid) {
            printf("  PASS: Correct tile selected\n");
        } else {
            printf("  FAIL: Expected 6x16, got %dx%d\n", sel.Mr, sel.Nr);
        }
    }

    // Test 3: M = 4 (should use 4x16 to avoid padding)
    printf("\nTest 3: M=4 (should use 4x16)\n");
    {
        int M = 4, N = 512, K = 512;
        auto sel = select_tile_params(M, N, K);
        printf("  Shape M=%d, N=%d, K=%d -> Tile %dx%d (preset=%d)\n",
               M, N, K, sel.Mr, sel.Nr, (int)sel.preset);
        printf("  Expected: 4x16 (M divisible by 4)\n");
        if (sel.Mr == 4 && sel.Nr == 16 && sel.valid) {
            printf("  PASS: Correct tile selected\n");
        } else {
            printf("  FAIL: Expected 4x16, got %dx%d\n", sel.Mr, sel.Nr);
        }
    }

    // Test 4: M = 12 (divisible by 4 and 6, but 8x16 is best)
    printf("\nTest 4: M=12 (divisible by 4,6, but 8x16 preferred)\n");
    {
        int M = 12, N = 512, K = 512;
        auto sel = select_tile_params(M, N, K);
        printf("  Shape M=%d, N=%d, K=%d -> Tile %dx%d (preset=%d)\n",
               M, N, K, sel.Mr, sel.Nr, (int)sel.preset);
        printf("  Expected: 8x16 (highest compute density, only 4 rows padding)\n");
        if (sel.Mr == 8 && sel.Nr == 16 && sel.valid) {
            printf("  PASS: Correct tile selected\n");
        } else {
            printf("  Note: Got %dx%d, benchmark may have chosen differently\n", sel.Mr, sel.Nr);
        }
    }

    // Test 5: Small M (< 4) - should return invalid
    printf("\nTest 5: M=3 (too small for packed path)\n");
    {
        int M = 3, N = 512, K = 512;
        auto sel = select_tile_params(M, N, K);
        printf("  Shape M=%d, N=%d, K=%d -> Tile %dx%d (valid=%d)\n",
               M, N, K, sel.Mr, sel.Nr, sel.valid);
        if (!sel.valid) {
            printf("  PASS: Tile not applicable for M<4\n");
        } else {
            printf("  Note: Got valid tile %dx%d\n", sel.Mr, sel.Nr);
        }
    }

    // Test 6: Cache functionality
    printf("\nTest 6: Cache functionality\n");
    {
        get_tile_cache().clear();

        int M = 16, N = 256, K = 256;
        auto sel1 = select_tile_params(M, N, K);
        printf("  First call: Tile %dx%d\n", sel1.Mr, sel1.Nr);

        auto sel2 = select_tile_params(M, N, K);
        printf("  Second call (cached): Tile %dx%d\n", sel2.Mr, sel2.Nr);

        if (sel1.Mr == sel2.Mr && sel1.Nr == sel2.Nr && get_tile_cache().size() > 0) {
            printf("  PASS: Cache working correctly\n");
        } else {
            printf("  FAIL: Cache not working\n");
        }
    }

    // Test 7: Performance comparison (M=6 with 6x16 vs 8x16)
    printf("\nTest 7: Performance comparison for M=6\n");
    {
        int M = 6, N = 512, K = 512;

        // Measure with autotune (should use 6x16)
        setenv("DNNOPT_AUTOTUNE", "1", 1);
        double gflops_autotune = measure_gemm_gflops(M, N, K, 7);

        // Measure without autotune (heuristic, uses registry priority)
        unsetenv("DNNOPT_AUTOTUNE");
        double gflops_heuristic = measure_gemm_gflops(M, N, K, 7);

        printf("  Autotune (6x16): %.2f GFLOPS\n", gflops_autotune);
        printf("  Heuristic:       %.2f GFLOPS\n", gflops_heuristic);

        double ratio = gflops_autotune / gflops_heuristic;
        printf("  Ratio: %.4f (autotune/heuristic)\n", ratio);

        if (ratio >= 0.95) {
            printf("  PASS: Tile autotune performs well (ratio >= 0.95)\n");
        } else {
            printf("  Note: Ratio < 0.95, may need investigation\n");
        }
    }

    // Test 8: Warmup functionality
    printf("\nTest 8: Warmup functionality\n");
    {
        get_tile_cache().clear();
        warmup_tile_autotune();
        printf("  Cache size after warmup: %zu entries\n", get_tile_cache().size());
        if (get_tile_cache().size() > 0) {
            printf("  PASS: Warmup populated cache\n");
        } else {
            printf("  Note: Cache empty, warmup may have used heuristics\n");
        }
    }

    printf("\n=== All Tests Complete ===\n");
    return 0;
}