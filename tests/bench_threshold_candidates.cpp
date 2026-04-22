/// @file bench_threshold_candidates.cpp
/// Benchmark different threshold candidates to find optimal dispatch boundaries.
///
/// Thresholds control kernel path selection:
///   - small_m_bound: M < bound → small-M path (default: 8)
///   - wide_n_bound: N >= bound → smallm_wide (default: 48)
///   - unpacked_thresh: M*N*K < thresh → adaptive_tile (default: 4M)

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace dnnopt;

/// Threshold candidates to benchmark
struct ThresholdCandidate {
    int small_m_bound;    // M < bound → small-M
    int wide_n_bound;     // N >= bound → wide driver
    int unpacked_thresh;  // M*N*K < thresh → unpacked
    const char* name;
};

static const ThresholdCandidate kThresholdCandidates[] = {
    {6,  32, 2 * 1024 * 1024, "Conservative"},   // Small-M bound lower, wide bound lower
    {8,  48, 4 * 1024 * 1024, "Standard"},       // Current defaults
    {10, 64, 6 * 1024 * 1024, "Aggressive"},     // Larger small-M bound
    {12, 80, 8 * 1024 * 1024, "Maximum"},        // Even larger bounds
};

static const int kNumThresholdCandidates = sizeof(kThresholdCandidates) / sizeof(kThresholdCandidates[0]);

/// Boundary shapes where threshold changes affect dispatch
struct BoundaryShape {
    int M, N, K;
    const char* category;
    const char* description;
};

static const BoundaryShape kBoundaryShapes[] = {
    // M boundary cases (near small_m_bound = 8)
    {6,  1024, 1024, "M-boundary", "M=6 < 8 (small-M path)"},
    {7,  1024, 1024, "M-boundary", "M=7 < 8 (small-M path)"},
    {8,  1024, 1024, "M-boundary", "M=8 = bound (packed path)"},
    {9,  1024, 1024, "M-boundary", "M=9 > 8 (packed path)"},
    {10, 1024, 1024, "M-boundary", "M=10 > 8 (packed path)"},
    {12, 1024, 1024, "M-boundary", "M=12 > 8 (packed path)"},

    // N boundary cases (near wide_n_bound = 48)
    {4,  32,  1024, "N-boundary", "M=4, N=32 < 48 (adaptive)"},
    {4,  40,  1024, "N-boundary", "M=4, N=40 < 48 (adaptive)"},
    {4,  48,  1024, "N-boundary", "M=4, N=48 = bound (wide)"},
    {4,  56,  1024, "N-boundary", "M=4, N=56 > 48 (wide)"},
    {4,  64,  1024, "N-boundary", "M=4, N=64 > 48 (wide)"},
    {6,  40,  1024, "N-boundary", "M=6, N=40 < 48 (adaptive)"},
    {6,  48,  1024, "N-boundary", "M=6, N=48 = bound (wide)"},
    {6,  64,  1024, "N-boundary", "M=6, N=64 > 48 (wide)"},
    {7,  40,  1024, "N-boundary", "M=7, N=40 < 48 (adaptive)"},
    {7,  48,  1024, "N-boundary", "M=7, N=48 = bound (wide)"},
    {7,  64,  1024, "N-boundary", "M=7, N=64 > 48 (wide)"},

    // Volume boundary cases (near unpacked_thresh = 4M)
    {8,  128, 128,  "Vol-boundary", "8×128×128=131K < 4M"},
    {8,  256, 256,  "Vol-boundary", "8×256×256=524K < 4M"},
    {8,  512, 512,  "Vol-boundary", "8×512×512=2M < 4M"},
    {8,  640, 640,  "Vol-boundary", "8×640×640=3.3M < 4M"},
    {8,  720, 720,  "Vol-boundary", "8×720×720=4.1M > 4M"},
    {8,  800, 800,  "Vol-boundary", "8×800×800=5.1M > 4M"},
    {16, 256, 256,  "Vol-boundary", "16×256×256=1M < 4M"},
    {16, 512, 512,  "Vol-boundary", "16×512×512=4.2M > 4M"},
    {32, 256, 256,  "Vol-boundary", "32×256×256=2.1M < 4M"},
    {32, 384, 384,  "Vol-boundary", "32×384×384=4.7M > 4M"},

    // Mixed boundary cases (M + N or M + Volume)
    {6,  64,  2048, "Mixed", "M=6, N=64, K=2048"},
    {7,  32,  4096, "Mixed", "M=7, N=32, K=4096"},
    {9,  48,  512,  "Mixed", "M=9, N=48, K=512"},
    {10, 64,  1024, "Mixed", "M=10, N=64, K=1024"},
    {5,  128, 1024, "Mixed", "M=5, N=128, K=1024"},
};

static const int kNumBoundaryShapes = sizeof(kBoundaryShapes) / sizeof(kBoundaryShapes[0]);

/// Benchmark GEMM with specific threshold settings
/// Returns median GFLOPS from multiple iterations
static double bench_with_thresholds(int M, int N, int K,
                                    int small_m_bound,
                                    int wide_n_bound,
                                    int unpacked_thresh,
                                    int warmup = 5, int iterations = 7) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);

    // Clear caches
    get_tile_cache().clear();
    get_blocking_cache().clear();
    get_gemm_shape_cache().clear();

    // Set thresholds via cache injection
    ThresholdSelection sel;
    sel.small_m_bound = small_m_bound;
    sel.wide_n_bound = wide_n_bound;
    sel.unpacked_thresh = unpacked_thresh;
    sel.benchmark_gflops = 0.0f;
    sel.valid = true;
    get_threshold_cache().set(sel);

    // Enable autotune
    setenv("DNNOPT_AUTOTUNE", "1", 1);

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
    }

    // Timed runs
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
    printf("=== Threshold Candidate Benchmark ===\n\n");

    printf("Testing %d threshold candidates on %d boundary shapes\n\n",
           kNumThresholdCandidates, kNumBoundaryShapes);

    // For each shape, benchmark all candidates
    printf("Shape              Category        ");
    for (int c = 0; c < kNumThresholdCandidates; ++c) {
        printf("%-12s ", kThresholdCandidates[c].name);
    }
    printf("Best\n");
    printf("==========================================================================================\n");

    // Track scores for each candidate
    double total_scores[kNumThresholdCandidates] = {0};
    int best_counts[kNumThresholdCandidates] = {0};

    for (int s = 0; s < kNumBoundaryShapes; ++s) {
        int M = kBoundaryShapes[s].M;
        int N = kBoundaryShapes[s].N;
        int K = kBoundaryShapes[s].K;
        int64_t vol = (int64_t)M * N * K;

        // Skip very large shapes
        if (vol > 4LL * 1024 * 1024 * 1024) continue;

        double gflops[kNumThresholdCandidates];
        int best_idx = 0;
        double best_gflops = 0;

        for (int c = 0; c < kNumThresholdCandidates; ++c) {
            gflops[c] = bench_with_thresholds(M, N, K,
                kThresholdCandidates[c].small_m_bound,
                kThresholdCandidates[c].wide_n_bound,
                kThresholdCandidates[c].unpacked_thresh,
                3, 5);

            if (gflops[c] > best_gflops) {
                best_gflops = gflops[c];
                best_idx = c;
            }
        }

        // Weight by shape category importance
        double weight = 1.0;
        if (strstr(kBoundaryShapes[s].category, "M-boundary")) weight = 2.0;
        if (strstr(kBoundaryShapes[s].category, "N-boundary")) weight = 1.5;

        for (int c = 0; c < kNumThresholdCandidates; ++c) {
            total_scores[c] += weight * gflops[c];
        }
        best_counts[best_idx]++;

        printf("M=%2d N=%3d K=%4d  %-12s   ",
               M, N, K, kBoundaryShapes[s].category);
        for (int c = 0; c < kNumThresholdCandidates; ++c) {
            printf("%7.2f     ", gflops[c]);
        }
        printf("%s\n", kThresholdCandidates[best_idx].name);
    }

    printf("==========================================================================================\n");
    printf("\nWeighted total scores:\n");
    for (int c = 0; c < kNumThresholdCandidates; ++c) {
        printf("  %s: %.2f (best for %d shapes)\n",
               kThresholdCandidates[c].name, total_scores[c], best_counts[c]);
    }

    // Find overall best
    int overall_best = 0;
    double best_score = total_scores[0];
    for (int c = 1; c < kNumThresholdCandidates; ++c) {
        if (total_scores[c] > best_score) {
            best_score = total_scores[c];
            overall_best = c;
        }
    }

    printf("\n=== Recommendation ===\n");
    printf("Best threshold candidate: %s\n", kThresholdCandidates[overall_best].name);
    printf("  small_m_bound = %d\n", kThresholdCandidates[overall_best].small_m_bound);
    printf("  wide_n_bound = %d\n", kThresholdCandidates[overall_best].wide_n_bound);
    printf("  unpacked_thresh = %d (%.1fM)\n",
           kThresholdCandidates[overall_best].unpacked_thresh,
           kThresholdCandidates[overall_best].unpacked_thresh / (1024.0 * 1024.0));

    return 0;
}