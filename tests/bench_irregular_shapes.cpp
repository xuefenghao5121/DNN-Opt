/// @file bench_irregular_shapes.cpp
/// Irregular/non-power-of-2 shapes that stress blocking params.
/// Focus on shapes that don't follow 2^n patterns.

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

struct IrregularShape {
    int M, N, K;
    const char* category;
    const char* note;
};

int main() {
    printf("=== Irregular/Non-Power-of-2 Shape Performance ===\n\n");

    // Irregular shapes - deliberately avoiding 2^n patterns
    // Categories:
    // 1. Prime numbers (worst for tiling)
    // 2. Odd numbers (padding overhead)
    // 3. Prime multiples (slightly better but still irregular)
    // 4. Strange ratios (non-standard proportions)
    // 5. Real-world but non-standard neural net sizes
    // 6. Boundary cases (just above/below tile sizes)
    // 7. Anti-power-of-2 (3^n, 5^n patterns)
    // 8. Mixed irregular (all three dims irregular)

    IrregularShape shapes[] = {
        // === Prime numbers ===
        {127,   127,   127,   "Prime",      "127^3 (prime)"},
        {257,   257,   257,   "Prime",      "257^3 (prime)"},
        {511,   511,   511,   "Prime",      "511 (prime-ish)"},
        {1021,  1021,  1021,  "Prime",      "1021^3 (prime)"},
        {127,   256,   512,   "Prime-M",    "Prime M × regular N,K"},
        {512,   127,   1024,  "Prime-N",    "Regular M,K × prime N"},
        {1024,  512,   127,   "Prime-K",    "Regular M,N × prime K"},
        {127,   127,   1024,  "Prime-MNK",  "Prime M,N × regular K"},
        {257,   511,   1023,  "Prime-mix",  "Multiple primes mixed"},

        // === Odd numbers (not divisible by tile Mr=4,6,8) ===
        {13,    1024,  1024,  "Odd-M",      "M=13 (not divisible by 4,6,8)"},
        {15,    1024,  1024,  "Odd-M",      "M=15 (not divisible by 4,6,8)"},
        {17,    1024,  1024,  "Odd-M",      "M=17 (not divisible by 4,6,8)"},
        {19,    1024,  1024,  "Odd-M",      "M=19 (not divisible by 4,6,8)"},
        {21,    1024,  1024,  "Odd-M",      "M=21 (not divisible by 4,6,8)"},
        {23,    1024,  1024,  "Odd-M",      "M=23 (not divisible by 4,6,8)"},
        {25,    1024,  1024,  "Odd-M",      "M=25 (divisible by 5 only)"},
        {27,    1024,  1024,  "Odd-M",      "M=27 (divisible by 3,9)"},
        {29,    1024,  1024,  "Odd-M",      "M=29 (prime)"},
        {31,    1024,  1024,  "Odd-M",      "M=31 (prime)"},
        {33,    1024,  1024,  "Odd-M",      "M=33 (divisible by 3,11)"},
        {35,    1024,  1024,  "Odd-M",      "M=35 (divisible by 5,7)"},
        {37,    1024,  1024,  "Odd-M",      "M=37 (prime)"},
        {39,    1024,  1024,  "Odd-M",      "M=39 (divisible by 3,13)"},
        {41,    1024,  1024,  "Odd-M",      "M=41 (prime)"},
        {43,    1024,  1024,  "Odd-M",      "M=43 (prime)"},
        {45,    1024,  1024,  "Odd-M",      "M=45 (divisible by 5,9)"},
        {47,    1024,  1024,  "Odd-M",      "M=47 (prime)"},
        {49,    1024,  1024,  "Odd-M",      "M=49 (7×7)"},

        // === Strange N (odd, prime, anti-tile) ===
        {512,   17,    512,   "Odd-N",      "N=17 (anti-tile)"},
        {512,   19,    512,   "Odd-N",      "N=19 (anti-tile)"},
        {512,   23,    512,   "Odd-N",      "N=23 (anti-tile)"},
        {512,   31,    512,   "Odd-N",      "N=31 (anti-tile)"},
        {512,   33,    512,   "Odd-N",      "N=33"},
        {512,   47,    512,   "Odd-N",      "N=47"},
        {512,   63,    512,   "Odd-N",      "N=63 (close to 64)"},
        {512,   65,    512,   "Odd-N",      "N=65 (just above 64)"},
        {512,   127,   512,   "Odd-N",      "N=127"},
        {512,   255,   512,   "Odd-N",      "N=255 (just below 256)"},
        {512,   257,   512,   "Odd-N",      "N=257 (just above 256)"},

        // === Anti-power-of-2 patterns (3^n, 5^n, 7^n) ===
        {9,     1024,  1024,  "3^n",        "M=9 (3×3)"},
        {27,    1024,  1024,  "3^n",        "M=27 (3×9)"},
        {81,    1024,  1024,  "3^n",        "M=81 (3×27)"},
        {243,   1024,  1024,  "3^n",        "M=243 (3×81)"},
        {729,   1024,  1024,  "3^n",        "M=729 (3×243)"},
        {2187,  1024,  1024,  "3^n",        "M=2187 (3×729)"},
        {25,    1024,  1024,  "5^n",        "M=25 (5×5)"},
        {125,   1024,  1024,  "5^n",        "M=125 (5×25)"},
        {625,   1024,  1024,  "5^n",        "M=625 (5×125)"},
        {3125,  1024,  1024,  "5^n",        "M=3125 (5×625)"},
        {49,    1024,  1024,  "7^n",        "M=49 (7×7)"},
        {343,   1024,  1024,  "7^n",        "M=343 (7×49)"},
        {2401,  1024,  1024,  "7^n",        "M=2401 (7×343)"},

        // === Real-world non-standard neural net sizes ===
        {768,   768,   1024,  "NN-irreg",   "BERT small hidden × intermediate"},
        {1024,  768,   1024,  "NN-irreg",   "Intermediate × hidden"},
        {768,   3072,  768,   "NN-irreg",   "BERT hidden × FFN intermediate"},
        {3072,  768,   768,   "NN-irreg",   "FFN intermediate × hidden"},
        {5120,  5120,  5120,  "NN-irreg",   "5K hidden (non-standard LLM)"},
        {4096,  4096,  5120,  "NN-irreg",   "4K × 4K × 5K"},
        {5120,  4096,  4096,  "NN-irreg",   "5K × 4K × 4K"},
        {2816,  2816,  2816,  "NN-irreg",   "2.8K hidden"},
        {6400,  6400,  6400,  "NN-irreg",   "6.4K hidden"},
        {8960,  8960,  8960,  "NN-irreg",   "8.96K hidden (LLaMA-65B style)"},
        {11008, 4096,  4096,  "NN-irreg",   "LLaMA FFN intermediate"},
        {4096,  11008, 4096,  "NN-irreg",   "LLaMA FFN transpose"},
        {14336, 5120,  5120,  "NN-irreg",   "LLaMA-2 70B FFN intermediate"},
        {5120,  14336, 5120,  "NN-irreg",   "LLaMA-2 FFN transpose"},
        {12288, 12288, 12288, "NN-irreg",   "12K hidden (large model)"},

        // === Boundary cases (just above/below tile sizes) ===
        {3,     1024,  1024,  "Boundary",   "M=3 (below smallest tile Mr=4)"},
        {5,     1024,  1024,  "Boundary",   "M=5 (between Mr=4 and 6)"},
        {7,     1024,  1024,  "Boundary",   "M=7 (between Mr=6 and 8)"},
        {9,     1024,  1024,  "Boundary",   "M=9 (just above Mr=8)"},
        {10,    1024,  1024,  "Boundary",   "M=10 (padding for Mr=8)"},
        {11,    1024,  1024,  "Boundary",   "M=11 (padding for Mr=8)"},
        {12,    1024,  1024,  "Boundary",   "M=12 (divisible by Mr=4)"},
        {14,    1024,  1024,  "Boundary",   "M=14 (padding for Mr=8)"},
        {15,    1024,  1024,  "Boundary",   "M=15 (no good tile)"},
        {16,    1024,  1024,  "Boundary",   "M=16 (perfect for Mr=8)"},

        // === Mixed irregular (all three dims irregular) ===
        {123,   456,   789,   "Mixed",      "All irregular small"},
        {234,   567,   890,   "Mixed",      "All irregular medium"},
        {345,   678,   901,   "Mixed",      "All irregular medium"},
        {456,   789,   1023,  "Mixed",      "All irregular medium-large"},
        {567,   890,   1234,  "Mixed",      "All irregular large"},
        {678,   901,   1456,  "Mixed",      "All irregular large"},
        {789,   1023,  1678,  "Mixed",      "All irregular large"},
        {890,   1234,  1890,  "Mixed",      "All irregular large"},
        {901,   1456,  2123,  "Mixed",      "All irregular very large"},
        {1023,  1678,  2345,  "Mixed",      "All irregular very large"},
        {1234,  1890,  2567,  "Mixed",      "All irregular very large"},
        {1456,  2123,  2789,  "Mixed",      "All irregular huge"},
        {1678,  2345,  3012,  "Mixed",      "All irregular huge"},
        {1890,  2567,  3345,  "Mixed",      "All irregular huge"},
        {2123,  2789,  3567,  "Mixed",      "All irregular huge"},
        {2345,  3012,  3890,  "Mixed",      "All irregular huge"},
        {2567,  3345,  4012,  "Mixed",      "All irregular huge"},
        {2789,  3567,  4234,  "Mixed",      "All irregular huge"},
        {3012,  3890,  4567,  "Mixed",      "All irregular huge"},
        {3345,  4012,  4789,  "Mixed",      "All irregular very huge"},
        {3567,  4234,  5012,  "Mixed",      "All irregular very huge"},
        {3890,  4567,  5234,  "Mixed",      "All irregular very huge"},
        {4012,  4789,  5456,  "Mixed",      "All irregular massive"},
        {4234,  5012,  5678,  "Mixed",      "All irregular massive"},
        {4567,  5234,  5890,  "Mixed",      "All irregular massive"},
    };

    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("Shape              Category    Heuristic    Autotune     Ratio   Note\n");
    printf("==============================================================================\n");

    double total_heuristic = 0, total_autotune = 0;
    double min_ratio = 1e18, max_ratio = 0;
    int n_tested = 0;

    for (int i = 0; i < n_shapes; ++i) {
        int M = shapes[i].M, N = shapes[i].N, K = shapes[i].K;

        // Skip very large shapes (will timeout)
        int64_t vol = (int64_t)M * N * K;
        if (vol > 10LL * 1024 * 1024 * 1024) continue;  // > 10B elements

        // Clear caches
        get_tile_cache().clear();
        get_blocking_cache().clear();
        get_gemm_shape_cache().clear();

        // Heuristic
        unsetenv("DNNOPT_AUTOTUNE");
        double gf_h = measure_gemm_gflops(M, N, K, 4, 5);

        // Autotune
        setenv("DNNOPT_AUTOTUNE", "1", 1);
        auto block = select_blocking_params(M, N, K);
        auto tile = select_tile_params(M, N, K);
        double gf_a = measure_gemm_gflops(M, N, K, 4, 5);

        double ratio = gf_a / gf_h;

        // Print (compact format)
        if (ratio < 0.98 || ratio > 1.02) {
            // Highlight significant changes
            printf("M=%4d N=%4d K=%4d %-10s %7.2f    %7.2f    %.4f ★ %s\n",
                   M, N, K, shapes[i].category, gf_h, gf_a, ratio, shapes[i].note);
        } else {
            printf("M=%4d N=%4d K=%4d %-10s %7.2f    %7.2f    %.4f   %s\n",
                   M, N, K, shapes[i].category, gf_h, gf_a, ratio, shapes[i].note);
        }

        total_heuristic += gf_h;
        total_autotune += gf_a;
        min_ratio = std::min(min_ratio, ratio);
        max_ratio = std::max(max_ratio, ratio);
        n_tested++;
    }

    printf("==============================================================================\n");
    printf("Average:           %7.2f    %7.2f    %.4f\n",
           total_heuristic/n_tested, total_autotune/n_tested,
           total_autotune/total_heuristic);
    printf("\nMin ratio: %.4f, Max ratio: %.4f\n", min_ratio, max_ratio);
    printf("\nOverall improvement: %.2f%%\n",
           (total_autotune/total_heuristic - 1) * 100);

    // Check for degradation
    if (min_ratio < 0.95) {
        printf("\n⚠️  WARNING: Some shapes show degradation (ratio < 0.95)\n");
    } else {
        printf("\n✅ No significant degradation (all ratios >= 0.95)\n");
    }

    // Count significant improvements
    int n_improved = 0, n_degraded = 0;
    printf("\nShapes with ratio > 1.02 (significant improvement): ");
    // Count in summary loop
    // We already printed, so just summarize

    return 0;
}