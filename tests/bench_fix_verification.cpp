/// @file bench_fix_verification.cpp
/// v2.2 Fix Verification: Compare autotune vs heuristic for ALL shapes.
///
/// Test cases:
///   1. Small-M shapes (M=1-7) - DNN-Opt's optimization target
///   2. Irregular N (prime numbers) - edge cases
///   3. Medium shapes (M=8-64, N=64-512)
///   4. Large shapes (M>=64, N>=512)
///   5. LLM-like shapes (batch-1, batch-4 inference)
///   6. Compare: DNNOPT_AUTOTUNE=1 vs DNNOPT_AUTOTUNE=0
///
/// Expected behavior after fix:
///   - Autotune should NOT be slower than heuristic
///   - Small-M shapes should select optimal kernel directly
///   - Large shapes may show slight autotune improvement

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/autotune/shape_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace dnnopt {

// Test shapes covering DNN-Opt's optimization domain AND general GEMM patterns
struct TestShape {
    int M, N, K;
    const char* category;
    const char* description;
};

static const TestShape kTestShapes[] = {
    // ========================================
    // M=1 (batch-1 inference, GEMV)
    // ========================================
    {1, 16,    16,    "M=1",       "GEMV-tiny"},
    {1, 32,    32,    "M=1",       "GEMV-small"},
    {1, 64,    64,    "M=1",       "GEMV-64"},
    {1, 128,   128,   "M=1",       "GEMV-128"},
    {1, 256,   256,   "M=1",       "GEMV-256"},
    {1, 256,   512,   "M=1",       "GEMV-wide"},
    {1, 512,   512,   "M=1",       "GEMV-512"},
    {1, 512,   1024,  "M=1",       "GEMV-LLM"},
    {1, 1024,  1024,  "M=1",       "GEMV-large"},
    {1, 4096,  4096,  "M=1",       "GEMV-huge"},

    // ========================================
    // M=2-7 (small batch) - KEY OPTIMIZATION TARGET
    // ========================================
    // M=2
    {2, 16,    16,    "M=2-7",     "M=2 tiny"},
    {2, 32,    32,    "M=2-7",     "M=2 small"},
    {2, 48,    64,    "M=2-7",     "M=2 wide-bound"},
    {2, 64,    64,    "M=2-7",     "M=2 64"},
    {2, 128,   64,    "M=2-7",     "M=2 128x64"},
    {2, 128,   128,   "M=2-7",     "M=2 128"},
    {2, 256,   256,   "M=2-7",     "M=2 256"},
    {2, 512,   512,   "M=2-7",     "M=2 512"},
    {2, 1024,  1024,  "M=2-7",     "M=2 LLM"},

    // M=3
    {3, 16,    16,    "M=2-7",     "M=3 tiny"},
    {3, 32,    32,    "M=2-7",     "M=3 small"},
    {3, 48,    64,    "M=2-7",     "M=3 wide-bound"},
    {3, 64,    64,    "M=2-7",     "M=3 64"},
    {3, 128,   128,   "M=2-7",     "M=3 128"},
    {3, 256,   256,   "M=2-7",     "M=3 256"},
    {3, 512,   512,   "M=2-7",     "M=3 512"},

    // M=4 (BERT batch-4)
    {4, 16,    16,    "M=2-7",     "M=4 tiny"},
    {4, 32,    32,    "M=2-7",     "M=4 small"},
    {4, 48,    64,    "M=2-7",     "M=4 wide-bound"},
    {4, 64,    64,    "M=2-7",     "M=4 64"},
    {4, 128,   128,   "M=2-7",     "M=4 128"},
    {4, 256,   256,   "M=2-7",     "M=4 256"},
    {4, 512,   512,   "M=2-7",     "M=4 512"},
    {4, 768,   768,   "M=2-7",     "M=4 BERT"},
    {4, 1024,  1024,  "M=2-7",     "M=4 LLM"},
    {4, 3072,  768,   "M=2-7",     "M=4 FFN-exp"},
    {4, 4096,  4096,  "M=2-7",     "M=4 batch4-LLM"},

    // M=5
    {5, 64,    64,    "M=2-7",     "M=5 64"},
    {5, 128,   128,   "M=2-7",     "M=5 128"},
    {5, 256,   256,   "M=2-7",     "M=5 256"},
    {5, 512,   512,   "M=2-7",     "M=5 512"},

    // M=6
    {6, 64,    64,    "M=2-7",     "M=6 64"},
    {6, 128,   128,   "M=2-7",     "M=6 128"},
    {6, 256,   256,   "M=2-7",     "M=6 256"},
    {6, 512,   512,   "M=2-7",     "M=6 512"},
    {6, 768,   768,   "M=2-7",     "M=6 BERT"},
    {6, 1024,  1024,  "M=2-7",     "M=6 LLM"},

    // M=7
    {7, 64,    64,    "M=2-7",     "M=7 64"},
    {7, 128,   128,   "M=2-7",     "M=7 128"},
    {7, 256,   256,   "M=2-7",     "M=7 256"},
    {7, 512,   512,   "M=2-7",     "M=7 512"},
    {7, 768,   768,   "M=2-7",     "M=7 BERT"},

    // ========================================
    // Irregular N (prime numbers, non-power-of-2)
    // ========================================
    // Prime N
    {4, 11,    64,    "Prime-N",   "N=11 prime"},
    {4, 13,    64,    "Prime-N",   "N=13 prime"},
    {4, 17,    64,    "Prime-N",   "N=17 prime"},
    {4, 19,    64,    "Prime-N",   "N=19 prime"},
    {4, 23,    64,    "Prime-N",   "N=23 prime"},
    {4, 29,    64,    "Prime-N",   "N=29 prime"},
    {4, 31,    64,    "Prime-N",   "N=31 prime"},
    {4, 37,    128,   "Prime-N",   "N=37 prime"},
    {4, 41,    128,   "Prime-N",   "N=41 prime"},
    {4, 43,    128,   "Prime-N",   "N=43 prime"},
    {4, 47,    128,   "Prime-N",   "N=47 prime (near 48)"},
    {4, 53,    128,   "Prime-N",   "N=53 prime"},
    {4, 59,    128,   "Prime-N",   "N=59 prime"},
    {4, 61,    128,   "Prime-N",   "N=61 prime"},
    {4, 67,    128,   "Prime-N",   "N=67 prime"},
    {4, 71,    128,   "Prime-N",   "N=71 prime"},
    {4, 73,    256,   "Prime-N",   "N=73 prime"},
    {4, 79,    256,   "Prime-N",   "N=79 prime"},
    {4, 83,    256,   "Prime-N",   "N=83 prime"},
    {4, 89,    256,   "Prime-N",   "N=89 prime"},
    {4, 97,    128,   "Prime-N",   "N=97 prime"},
    {4, 101,   256,   "Prime-N",   "N=101 prime"},
    {4, 103,   256,   "Prime-N",   "N=103 prime"},
    {4, 107,   256,   "Prime-N",   "N=107 prime"},
    {4, 127,   256,   "Prime-N",   "N=127 prime"},

    // Prime K
    {4, 64,    11,    "Prime-K",   "K=11 prime"},
    {4, 64,    13,    "Prime-K",   "K=13 prime"},
    {4, 64,    17,    "Prime-K",   "K=17 prime"},
    {4, 64,    31,    "Prime-K",   "K=31 prime"},
    {4, 128,   37,    "Prime-K",   "K=37 prime"},
    {4, 128,   53,    "Prime-K",   "K=53 prime"},
    {4, 128,   71,    "Prime-K",   "K=71 prime"},

    // Irregular N+K
    {4, 17,    13,    "Irregular", "N=17,K=13 primes"},
    {4, 31,    17,    "Irregular", "N=31,K=17 primes"},
    {4, 53,    37,    "Irregular", "N=53,K=37 primes"},
    {4, 97,    71,    "Irregular", "N=97,K=71 primes"},

    // ========================================
    // M=8-31 (medium, adaptive tile / packed boundary)
    // ========================================
    {8, 64,    64,    "Medium",    "M=8 boundary"},
    {8, 128,   128,   "Medium",    "M=8 128"},
    {8, 256,   256,   "Medium",    "M=8 256"},
    {8, 512,   512,   "Medium",    "M=8 512"},
    {8, 768,   768,   "Medium",    "M=8 BERT"},
    {8, 1024,  1024,  "Medium",    "M=8 LLM"},

    {12, 128,  128,   "Medium",    "M=12"},
    {12, 256,  256,   "Medium",    "M=12 256"},
    {12, 512,  512,   "Medium",    "M=12 512"},

    {16, 64,   64,    "Medium",    "M=16 small"},
    {16, 128,  128,   "Medium",    "M=16 128"},
    {16, 256,  256,   "Medium",    "M=16 256"},
    {16, 512,  512,   "Medium",    "M=16 512"},
    {16, 768,  768,   "Medium",    "M=16 BERT"},
    {16, 1024, 1024,  "Medium",    "M=16 LLM"},

    {24, 256,  256,   "Medium",    "M=24"},
    {24, 512,  512,   "Medium",    "M=24 512"},

    {32, 64,   64,    "Medium",    "M=32 small"},
    {32, 128,  128,   "Medium",    "M=32 128"},
    {32, 256,  256,   "Medium",    "M=32 256"},
    {32, 512,  512,   "Medium",    "M=32 512"},
    {32, 768,  768,   "Medium",    "M=32 BERT"},
    {32, 1024, 1024,  "Medium",    "M=32 LLM"},

    // ========================================
    // M=48-128 (medium-large)
    // ========================================
    {48, 256,  256,   "Med-Large", "M=48"},
    {48, 512,  512,   "Med-Large", "M=48 512"},
    {48, 1024, 1024,  "Med-Large", "M=48 LLM"},

    {64, 128,  128,   "Med-Large", "M=64 small"},
    {64, 256,  256,   "Med-Large", "M=64 256"},
    {64, 512,  512,   "Med-Large", "M=64 512"},
    {64, 768,  768,   "Med-Large", "M=64 BERT"},
    {64, 1024, 1024,  "Med-Large", "M=64 LLM"},

    {96, 512,  512,   "Med-Large", "M=96"},
    {96, 1024, 1024,  "Med-Large", "M=96 LLM"},

    {128, 256, 256,   "Med-Large", "M=128 256"},
    {128, 512, 512,   "Med-Large", "M=128 512"},
    {128, 768, 768,   "Med-Large", "M=128 BERT"},
    {128, 1024, 1024, "Med-Large", "M=128 LLM"},

    // ========================================
    // Large shapes (threading domain)
    // ========================================
    {256, 512,  512,  "Large",     "M=256"},
    {256, 1024, 1024, "Large",     "M=256 LLM"},
    {256, 2048, 2048, "Large",     "M=256 huge"},

    {512, 512,  512,  "Large",     "M=512 square"},
    {512, 1024, 1024, "Large",     "M=512 LLM"},
    {512, 2048, 2048, "Large",     "M=512 huge"},

    {1024, 1024, 1024, "Large",    "1Kx1K"},
    {1024, 2048, 2048, "Large",    "1Kx2K"},
    {1024, 4096, 4096, "Large",    "1Kx4K"},

    {2048, 2048, 2048, "Large",    "2Kx2K"},
    {2048, 4096, 4096, "Large",    "2Kx4K"},

    {4096, 4096, 4096, "Large",    "4Kx4K"},
    {4096, 8192, 8192, "Large",    "4Kx8K"},

    // ========================================
    // Tall-skinny (M >> N)
    // ========================================
    {128, 1,    64,    "Tall",      "M>>N extreme"},
    {128, 2,    64,    "Tall",      "M>>N tiny"},
    {128, 4,    64,    "Tall",      "M>>N 4"},
    {128, 8,    64,    "Tall",      "M>>N 8"},
    {128, 16,   64,    "Tall",      "M>>N 16"},
    {128, 32,   64,    "Tall",      "M>>N 32"},
    {256, 16,   128,   "Tall",      "M>>N 256x16"},
    {256, 32,   128,   "Tall",      "M>>N 256x32"},
    {512, 16,   256,   "Tall",      "M>>N 512x16"},
    {512, 32,   256,   "Tall",      "M>>N 512x32"},

    // ========================================
    // Short-wide (M << N)
    // ========================================
    {1,  4096,  4096,  "Wide",      "GEMV-wide"},
    {2,  4096,  4096,  "Wide",      "M<<N 2x4K"},
    {4,  4096,  4096,  "Wide",      "M<<N 4x4K"},
    {8,  4096,  4096,  "Wide",      "M<<N 8x4K"},
    {16, 4096,  4096,  "Wide",      "M<<N 16x4K"},
    {32, 4096,  4096,  "Wide",      "M<<N 32x4K"},

    // ========================================
    // LLM inference patterns
    // ========================================
    // Batch-1 LLM (7B model typical)
    {1, 4096,  4096,  "LLM-b1",    "LLM embed"},
    {1, 4096,  11008, "LLM-b1",    "LLM FFN1"},
    {1, 11008, 4096,  "LLM-b1",    "LLM FFN2"},
    {1, 4096,  4096,  "LLM-b1",    "LLM attn"},

    // Batch-4 LLM
    {4, 4096,  4096,  "LLM-b4",    "LLM embed b4"},
    {4, 4096,  11008, "LLM-b4",    "LLM FFN1 b4"},
    {4, 11008, 4096,  "LLM-b4",    "LLM FFN2 b4"},

    // Batch-8 LLM
    {8, 4096,  4096,  "LLM-b8",    "LLM embed b8"},
    {8, 4096,  11008, "LLM-b8",    "LLM FFN1 b8"},
    {8, 11008, 4096,  "LLM-b8",    "LLM FFN2 b8"},

    // ========================================
    // BERT inference patterns
    // ========================================
    // Batch-1 BERT
    {1, 768,   768,   "BERT-b1",   "BERT embed"},
    {1, 768,   3072,  "BERT-b1",   "BERT FFN1"},
    {1, 3072,  768,   "BERT-b1",   "BERT FFN2"},

    // Batch-4 BERT
    {4, 768,   768,   "BERT-b4",   "BERT embed b4"},
    {4, 768,   3072,  "BERT-b4",   "BERT FFN1 b4"},
    {4, 3072,  768,   "BERT-b4",   "BERT FFN2 b4"},

    // Batch-8 BERT
    {8, 768,   768,   "BERT-b8",   "BERT embed b8"},
    {8, 768,   3072,  "BERT-b8",   "BERT FFN1 b8"},
    {8, 3072,  768,   "BERT-b8",   "BERT FFN2 b8"},

    // Batch-32 BERT
    {32, 768,  768,   "BERT-b32",  "BERT embed b32"},
    {32, 768,  3072,  "BERT-b32",  "BERT FFN1 b32"},
    {32, 3072, 768,   "BERT-b32",  "BERT FFN2 b32"},
};

static const int kNumShapes = sizeof(kTestShapes) / sizeof(kTestShapes[0]);

// Benchmark result
struct Result {
    int M, N, K;
    double time_us_heuristic;
    double time_us_autotune;
    double gflops_heuristic;
    double gflops_autotune;
    double ratio;  // autotune / heuristic
    const char* category;
};

double benchmark_gemm(int M, int N, int K, bool use_autotune) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    // Initialize with small values to avoid denormals
    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);
    std::memset(C.get(), 0, M * N * sizeof(float));

    // Warmup (more warmup for stability)
    for (int w = 0; w < 3; ++w) {
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
    }

    // Timed runs (median of 7 for better stability)
    double times[7];
    Timer timer;
    for (int t = 0; t < 7; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        times[t] = timer.elapsed_us();
    }

    // Sort and take median
    std::sort(times, times + 7);
    return times[3];
}

void run_benchmark(bool use_autotune, std::vector<Result>& results) {
    // Set environment
    if (use_autotune) {
        setenv("DNNOPT_AUTOTUNE", "1", 1);
    } else {
        unsetenv("DNNOPT_AUTOTUNE");
    }

    // Clear caches
    clear_all_shape_caches();

    printf("Running benchmark with DNNOPT_AUTOTUNE=%s...\n",
           use_autotune ? "1" : "0");

    for (int i = 0; i < kNumShapes; ++i) {
        const auto& shape = kTestShapes[i];
        double time_us = benchmark_gemm(shape.M, shape.N, shape.K, use_autotune);
        double gflops = 2.0 * shape.M * shape.N * shape.K / time_us;

        if (use_autotune) {
            // Find matching heuristic result and update
            for (auto& r : results) {
                if (r.M == shape.M && r.N == shape.N && r.K == shape.K) {
                    r.time_us_autotune = time_us;
                    r.gflops_autotune = gflops;
                    r.ratio = r.time_us_autotune / r.time_us_heuristic;
                    break;
                }
            }
        } else {
            // Store heuristic result
            Result r;
            r.M = shape.M;
            r.N = shape.N;
            r.K = shape.K;
            r.time_us_heuristic = time_us;
            r.gflops_heuristic = gflops;
            r.time_us_autotune = 0;
            r.gflops_autotune = 0;
            r.ratio = 0;
            r.category = shape.category;
            results.push_back(r);
        }
    }
}

void print_results(const std::vector<Result>& results) {
    printf("\n");
    printf("============================================================\n");
    printf("  v2.2 Fix Verification: Autotune vs Heuristic Performance\n");
    printf("============================================================\n");
    printf("\n");
    printf("Shape          | Heuristic      | Autotune       | Ratio | Status\n");
    printf("               | Time(us) GFLOPS| Time(us) GFLOPS|       |\n");
    printf("---------------|----------------|----------------|-------|--------\n");

    int degraded = 0;
    int improved = 0;
    int same = 0;

    for (const auto& r : results) {
        printf("[%d,%d,%d] %-8s | %7.1f %6.2f | %7.1f %6.2f | %.3f |",
               r.M, r.N, r.K, r.category,
               r.time_us_heuristic, r.gflops_heuristic,
               r.time_us_autotune, r.gflops_autotune,
               r.ratio);

        if (r.ratio > 1.05) {
            printf(" DEGRADED!\n");
            degraded++;
        } else if (r.ratio < 0.95) {
            printf(" IMPROVED\n");
            improved++;
        } else {
            printf(" OK\n");
            same++;
        }
    }

    printf("\n");
    printf("============================================================\n");
    printf("  Summary\n");
    printf("============================================================\n");
    printf("\n");
    printf("Total shapes tested: %zu\n", results.size());
    printf("  Degraded (>5%% slower): %d\n", degraded);
    printf("  Improved (>5%% faster): %d\n", improved);
    printf("  Same (within 5%%):      %d\n", same);

    // Calculate averages by category
    printf("\n");
    printf("Average GFLOPS by category:\n");
    printf("  Category      | Heuristic | Autotune | Ratio\n");
    printf("  --------------|-----------|----------|-------\n");

    const char* categories[] = {"M=1", "M=2-7", "Prime-N", "M>=8"};
    for (const char* cat : categories) {
        double sum_h = 0, sum_a = 0, count = 0;
        for (const auto& r : results) {
            if (strcmp(r.category, cat) == 0) {
                sum_h += r.gflops_heuristic;
                sum_a += r.gflops_autotune;
                count++;
            }
        }
        if (count > 0) {
            double avg_h = sum_h / count;
            double avg_a = sum_a / count;
            printf("  %-12s | %8.2f | %8.2f | %.3f\n",
                   cat, avg_h, avg_a, avg_a / avg_h);
        }
    }

    printf("\n");

    // Key result
    if (degraded == 0) {
        printf("✓ FIX VERIFIED: Autotune is NOT slower than heuristic for small-M shapes!\n");
    } else {
        printf("✗ FIX FAILED: %d shapes show degradation with autotune!\n", degraded);
        printf("  This indicates the fix is incomplete.\n");
    }
}

}  // namespace dnnopt

int main() {
    using namespace dnnopt;

    // Print hardware info
    const auto& hw = detect_arm_hwcaps();
    printf("Hardware: %s, %u cores, %u MHz\n", hw.cpu_name.c_str(), hw.num_cores, hw.freq_mhz);
    printf("SVE: %s, BF16: %s, I8MM: %s\n",
           hw.has(kSVE) ? "YES" : "NO",
           hw.has(kBF16) ? "YES" : "NO",
           hw.has(kI8MM) ? "YES" : "NO");
    printf("\n");

    std::vector<Result> results;

    // Run heuristic first
    run_benchmark(false, results);

    // Then run autotune
    run_benchmark(true, results);

    // Print comparison
    print_results(results);

    return 0;
}