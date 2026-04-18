/// @file gemm_autotune.cpp
/// Enhanced runtime auto-tuning for GEMM cache blocking parameters.
///
/// v0.9.18 improvements:
///   - Expanded search grid (5 candidates, shape-aware)
///   - Multi-shape testing (large, small, tall-skinny)
///   - Shape-specific tuning profiles
///   - Total cost: ~10-15ms on a modern ARM core

#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/timer.h"

#include <atomic>
#include <cstring>
#include <cmath>
#include <mutex>

namespace dnnopt {

// ============================================================
// Auto-tune search grid (v0.9.18: expanded)
// ============================================================

struct TuneCandidate {
    float l1d_util;
    float l2_util;
    float l3_util;
    const char* name;
};

// 5 candidates: Conservative → Aggressive
static const TuneCandidate kCandidates[] = {
    {0.25f, 0.25f, 0.15f, "Conservative"},   // Safe on all cores, low cache pressure
    {0.30f, 0.30f, 0.20f, "Standard"},       // Default oneDNN-style
    {0.35f, 0.35f, 0.25f, "Moderate"},       // Good general-purpose
    {0.40f, 0.40f, 0.30f, "Aggressive"},     // Best when HW prefetch is strong
    {0.45f, 0.45f, 0.35f, "Maximum"},        // For high-bandwidth CPUs (V-series)
};

static const int kNumCandidates = sizeof(kCandidates) / sizeof(kCandidates[0]);

// ============================================================
// Shape-specific tuning (v0.9.18)
// ============================================================

enum class TuneShapeClass {
    kLarge,      // M,N,K >= 512, square-ish
    kSmall,      // M*N*K < 4M flops, memory-bound
    kTallSkinny, // M >> N, bandwidth-bound
    kSquare,     // Balanced M,N,K
};

struct TuneShape {
    int M, N, K;
    TuneShapeClass shape_class;
    const char* name;
};

// Test shapes for different workloads
static const TuneShape kTuneShapes[] = {
    {512,  512,  512,  TuneShapeClass::kLarge,      "Large-512"},
    {256,  256,  256,  TuneShapeClass::kSquare,     "Square-256"},
    {128,  128,  128,  TuneShapeClass::kSmall,      "Small-128"},
    {128,  64,   768,  TuneShapeClass::kTallSkinny, "TallSkinny-128"},
    {4,    1024, 1024, TuneShapeClass::kSmall,      "Batch1-4"},
};

static const int kNumTuneShapes = sizeof(kTuneShapes) / sizeof(kTuneShapes[0]);

// ============================================================
// Micro-benchmark
// ============================================================

/// Run a small GEMM and return elapsed microseconds.
static double bench_gemm_shape(const ArmHwProfile& hw,
                                const CpuTuningProfile& profile,
                                int M, int N, int K) {
    // Allocate matrices
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    // Initialize with small values to avoid denormals
    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);
    std::memset(C.get(), 0, M * N * sizeof(float));

    // Warmup (1 iteration)
    gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);

    // Timed run (3 iterations, take median)
    Timer timer;
    double times[3];
    for (int t = 0; t < 3; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        times[t] = timer.elapsed_us();
    }

    // Return median
    if (times[0] > times[1]) std::swap(times[0], times[1]);
    if (times[1] > times[2]) std::swap(times[1], times[2]);
    if (times[0] > times[1]) std::swap(times[0], times[1]);
    return times[1];
}

// ============================================================
// Cached profile (v0.9.18: shape-aware)
// ============================================================

static std::mutex g_autotune_mutex;
static std::atomic<bool> g_autotuned{false};
static CpuTuningProfile g_tuned_profile;

// Shape-specific tuned configurations
static TuneCandidate g_best_for_shape[kNumTuneShapes];

const CpuTuningProfile& get_autotuned_profile() {
    // Fast path: already tuned or has built-in profile
    if (g_autotuned.load(std::memory_order_acquire))
        return g_tuned_profile;

    std::lock_guard<std::mutex> lock(g_autotune_mutex);

    // Double-check after acquiring lock
    if (g_autotuned.load(std::memory_order_relaxed))
        return g_tuned_profile;

    const auto& hw = detect_arm_hwcaps();
    const auto& builtin = lookup_tuning_profile(hw);

    // If we have an exact match (not the generic default), use it directly
    if (builtin.part_number != 0) {
        g_tuned_profile = builtin;
        g_autotuned.store(true, std::memory_order_release);
        return g_tuned_profile;
    }

    // Generic default → run enhanced auto-tune
    g_tuned_profile = builtin;

    // Test each candidate on each shape
    double total_scores[kNumCandidates] = {0};
    int best_overall_idx = 2;  // default to moderate

    for (int s = 0; s < kNumTuneShapes; ++s) {
        const auto& shape = kTuneShapes[s];
        double best_time = 1e18;
        int best_idx = 2;

        for (int c = 0; c < kNumCandidates; ++c) {
            CpuTuningProfile trial = builtin;
            trial.l1d_util = kCandidates[c].l1d_util;
            trial.l2_util  = kCandidates[c].l2_util;
            trial.l3_util  = kCandidates[c].l3_util;

            double t = bench_gemm_shape(hw, trial, shape.M, shape.N, shape.K);

            // Weight by shape class importance
            double weight = 1.0;
            switch (shape.shape_class) {
                case TuneShapeClass::kLarge:      weight = 3.0; break;  // Large matrices dominate inference
                case TuneShapeClass::kSquare:     weight = 2.0; break;
                case TuneShapeClass::kSmall:      weight = 1.5; break;  // Small-M common in batch-1
                case TuneShapeClass::kTallSkinny: weight = 1.0; break;
            }

            total_scores[c] += weight * t;

            if (t < best_time) {
                best_time = t;
                best_idx = c;
            }
        }

        g_best_for_shape[s] = kCandidates[best_idx];
    }

    // Pick overall best (lowest weighted score)
    double best_score = 1e18;
    for (int c = 0; c < kNumCandidates; ++c) {
        if (total_scores[c] < best_score) {
            best_score = total_scores[c];
            best_overall_idx = c;
        }
    }

    // Apply best overall candidate
    g_tuned_profile.l1d_util = kCandidates[best_overall_idx].l1d_util;
    g_tuned_profile.l2_util  = kCandidates[best_overall_idx].l2_util;
    g_tuned_profile.l3_util  = kCandidates[best_overall_idx].l3_util;
    g_tuned_profile.name     = "Auto-tuned (runtime)";

    g_autotuned.store(true, std::memory_order_release);
    return g_tuned_profile;
}

/// Get shape-specific tuning candidate (for advanced usage).
const TuneCandidate* get_best_for_shape_class(TuneShapeClass shape_class) {
    if (!g_autotuned.load(std::memory_order_acquire))
        return nullptr;

    // Find matching shape in test shapes
    for (int s = 0; s < kNumTuneShapes; ++s) {
        if (kTuneShapes[s].shape_class == shape_class)
            return &g_best_for_shape[s];
    }
    return nullptr;
}

void reset_autotune_cache() {
    std::lock_guard<std::mutex> lock(g_autotune_mutex);
    g_autotuned.store(false, std::memory_order_release);
}

}  // namespace dnnopt
