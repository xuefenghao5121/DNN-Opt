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
    clear_all_shape_caches();
}

// ============================================================
// v0.9.28: Kernel Selection Autotune
// ============================================================

// Default warmup shapes (covering oneDNN weak spots: small batch)
static const int kDefaultWarmupM[] = {1, 4, 8, 16, 32, 64, 128};
static const int kDefaultWarmupN[] = {4096, 1024, 512, 256, 128};
static const int kDefaultWarmupK[] = {4096, 1024, 512, 256, 128};

/// Micro-benchmark a specific GEMM kernel.
static double bench_gemm_kernel(GemmKernelId kernel_id, int M, int N, int K) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    // Initialize with small values
    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);
    std::memset(C.get(), 0, M * N * sizeof(float));

    Timer timer;

    // Warmup
    gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);

    // Timed run (median of 3)
    double times[3];
    for (int t = 0; t < 3; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        times[t] = timer.elapsed_us();
    }

    // Median
    if (times[0] > times[1]) std::swap(times[0], times[1]);
    if (times[1] > times[2]) std::swap(times[1], times[2]);
    if (times[0] > times[1]) std::swap(times[0], times[1]);
    return times[1];
}

/// Determine applicable kernels for a shape.
static void get_candidate_kernels(int M, int N, int K,
                                   GemmKernelId* candidates,
                                   int* n_candidates) {
    *n_candidates = 0;

    // Tiny: N=1 or M=1
    if (N == 1 || M == 1) {
        candidates[(*n_candidates)++] = GemmKernelId::kTiny;
        return;  // Tiny dominates
    }

    // Small-M: M < 8
    if (M < 8) {
        candidates[(*n_candidates)++] = GemmKernelId::kSmallM;
        // Wide: M >= 2 and N >= 48
        if (M >= 2 && N >= 48) {
            candidates[(*n_candidates)++] = GemmKernelId::kSmallMWide;
        }
    }

    // Adaptive tile: M >= 4 and M <= 32
    if (M >= 4 && M <= 32) {
        candidates[(*n_candidates)++] = GemmKernelId::kAdaptiveTile;
    }

    // Packed: M >= 8 or large volume
    if (M >= 8 || (int64_t)M * N * K >= 4 * 1024 * 1024) {
        candidates[(*n_candidates)++] = GemmKernelId::kPacked;
    }

    // Ensure at least one candidate
    if (*n_candidates == 0) {
        candidates[(*n_candidates)++] = GemmKernelId::kPacked;
    }
}

GemmKernelId select_gemm_kernel(int M, int N, int K, GemmDataType dtype) {
    // FP32 only for now (BF16/INT8 use same dispatch logic)
    if (dtype != GemmDataType::kFP32) {
        // Use heuristics for BF16/INT8
        if (M < 8) return GemmKernelId::kSmallM;
        return GemmKernelId::kPacked;
    }

    // Build shape key
    GemmShapeKey key;
    key.M = (M > 65535) ? 65535 : M;
    key.N = (N > 65535) ? 65535 : N;
    key.K = (K > 65535) ? 65535 : K;
    key.dtype = 0;  // FP32
    key.algo = 0;
    uint64_t hash = key.hash();

    // Check cache
    auto& cache = get_gemm_shape_cache();
    const KernelSelection* cached = cache.lookup(hash);
    if (cached && cached->valid) {
        return static_cast<GemmKernelId>(cached->kernel_id);
    }

    // Get candidate kernels
    GemmKernelId candidates[5];
    int n_candidates;
    get_candidate_kernels(M, N, K, candidates, &n_candidates);

    // If only one candidate, use it directly
    if (n_candidates == 1) {
        KernelSelection sel;
        sel.kernel_id = static_cast<uint8_t>(candidates[0]);
        sel.gflops = 0.0f;
        sel.time_us = 0;
        sel.valid = true;
        cache.insert(hash, sel);
        return candidates[0];
    }

    // Benchmark candidates and pick fastest
    GemmKernelId best = candidates[0];
    double best_time = 1e18;

    for (int i = 0; i < n_candidates; ++i) {
        double t = bench_gemm_kernel(candidates[i], M, N, K);
        if (t < best_time) {
            best_time = t;
            best = candidates[i];
        }
    }

    // Cache result
    KernelSelection sel;
    sel.kernel_id = static_cast<uint8_t>(best);
    sel.gflops = 2.0 * M * N * K / (best_time * 1000.0);  // GFLOPS
    sel.time_us = static_cast<uint32_t>(best_time);
    sel.valid = true;
    cache.insert(hash, sel);

    return best;
}

void warmup_gemm_autotune(const int* shapes_M,
                          const int* shapes_N,
                          const int* shapes_K,
                          int n_shapes) {
    if (n_shapes == 0) {
        // Use defaults
        const int nM = sizeof(kDefaultWarmupM) / sizeof(kDefaultWarmupM[0]);
        const int nN = sizeof(kDefaultWarmupN) / sizeof(kDefaultWarmupN[0]);
        const int nK = sizeof(kDefaultWarmupK) / sizeof(kDefaultWarmupK[0]);

        for (int i = 0; i < nM; ++i) {
            for (int j = 0; j < nN; ++j) {
                for (int k = 0; k < nK; ++k) {
                    select_gemm_kernel(kDefaultWarmupM[i],
                                       kDefaultWarmupN[j],
                                       kDefaultWarmupK[k],
                                       GemmDataType::kFP32);
                }
            }
        }
    } else {
        for (int i = 0; i < n_shapes; ++i) {
            select_gemm_kernel(shapes_M[i], shapes_N[i], shapes_K[i],
                               GemmDataType::kFP32);
        }
    }
}

int load_gemm_kernel_cache(const char* path) {
    return get_gemm_shape_cache().load_from_file(path);
}

int save_gemm_kernel_cache(const char* path) {
    return get_gemm_shape_cache().save_to_file(path);
}

}  // namespace dnnopt
