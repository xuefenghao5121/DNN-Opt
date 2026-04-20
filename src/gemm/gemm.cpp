/// @file gemm.cpp
/// Top-level GEMM dispatch with registry-based adaptive kernel selection.
/// Falls back to legacy drivers when no registry kernel matches.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"
#include "dnnopt/gemm/gemm_driver_generic.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/cpu_tuning_profile.h"

#include <cstdlib>  // for getenv

namespace dnnopt {

// Forward declarations
void gemm_driver_fp32(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);
void gemm_smallm_wide_driver_fp32(int M, int N, int K,
                                   float alpha, const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta, float* C, int ldc);
void gemm_driver_bf16(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);
void gemm_driver_int8(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc);

// Small-M BF16/INT8 kernels (defined in gemm_smallm_bf16.cpp, gemm_smallm_int8.cpp)
void gemm_smallm_bf16(int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc);
void gemm_smallm_int8(int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc);

// Tiny GEMM kernels (defined in gemm_tiny_fp32.cpp)
bool gemm_tiny_dispatch_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);

// Small-M SVE kernel (defined in gemm_smallm_fp32_sve.cpp)
#ifdef __ARM_FEATURE_SVE
void gemm_smallm_fp32_sve(int M, int N, int K,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float* C, int ldc,
                          float alpha, float beta);
#endif

namespace {

/// Check if autotune mode is enabled.
/// Controlled by environment variable DNNOPT_AUTOTUNE=1.
static bool is_autotune_enabled() {
    static bool enabled = []() {
        const char* env = std::getenv("DNNOPT_AUTOTUNE");
        return env != nullptr && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
    }();
    return enabled;
}

/// Naive FP32 GEMM: C = alpha * A * B + beta * C
void gemm_naive_fp32(int M, int N, int K,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
}

/// Dispatch via registry + generic driver.
/// Returns true if handled, false if caller should use legacy path.
/// When allow_m_pad=true, accepts kernels where M < Mr (zero-pads via edge_bufs).
bool dispatch_via_registry(GemmDataType dtype,
                           int M, int N, int K,
                           float alpha, const float* A, int lda,
                           const float* B, int ldb,
                           float beta, float* C, int ldc,
                           bool allow_m_pad = false) {
    const auto& hw = detect_arm_hwcaps();
    const auto& profile = get_autotuned_profile();

    // Try all matching kernels in priority order until one fits M.
    auto candidates = GemmUkernelRegistry::instance().select_all(dtype, hw);
    for (const auto* desc : candidates) {
        int Nr = desc->nr_is_vla ? desc->compute_nr(hw.sve_vector_bits) : desc->Nr;
        int Mr = desc->Mr;

        // Skip if M < Mr unless M-padding is explicitly allowed.
        // M-padding is wasteful for very small M but reasonable when M is close to Mr
        // (e.g., M=6 with Mr=8 wastes only 25% compute but gains packed B-panel reuse).
        if (M < Mr && !allow_m_pad) continue;
        // When M-padding, only accept kernels where M >= Mr/2 to avoid excessive waste.
        if (M < Mr && M * 2 < Mr) continue;

        // v2 Autotune: Use autotuned blocking params if enabled
        auto bp = get_autotuned_blocking_params(M, N, K, Mr, Nr, desc->Kgroup,
                                                 desc->packed_a_elem_bytes,
                                                 desc->packed_b_elem_bytes);

        GemmDriverConfig cfg;
        cfg.Mr = Mr;
        cfg.Nr = Nr;
        cfg.Kgroup = desc->Kgroup;
        cfg.Mc = bp.Mc;
        cfg.Nc = bp.Nc;
        cfg.Kc = bp.Kc;
        cfg.packed_a_elem_bytes = desc->packed_a_elem_bytes;
        cfg.packed_b_elem_bytes = desc->packed_b_elem_bytes;
        cfg.dtype = dtype;
        cfg.ukernel = desc->ukernel;
        cfg.pack_a = desc->pack_a;
        cfg.pack_b = desc->pack_b;

        // Threading config from tuning profile
        cfg.threading_min_flops = profile.threading_min_flops;
        cfg.prefer_2d_threading = profile.prefer_2d_threading;
        cfg.shape = classify_shape(M, N, K);

        gemm_driver_generic(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
        return true;
    }
    return false;
}

}  // namespace

// ============================================================
// Public API
// ============================================================

void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, GemmAlgo::kAuto);
}

void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc,
               GemmAlgo algo) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (algo == GemmAlgo::kNaive) {
        gemm_naive_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Auto: try registry dispatch first
    if (algo == GemmAlgo::kAuto) {
        // Tiny shapes: N=1, M=1, or M,N ≤ 4 get specialized kernels
#ifdef __ARM_NEON
        if (gemm_tiny_dispatch_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)) {
            return;
        }
#endif

        // Autotune-guided dispatch (if enabled via DNNOPT_AUTOTUNE=1)
        if (is_autotune_enabled()) {
            GemmKernelId kid = select_gemm_kernel(M, N, K, GemmDataType::kFP32);
#ifdef __ARM_NEON
            switch (kid) {
            case GemmKernelId::kTiny:
                // Already handled above, but just in case
                if (gemm_tiny_dispatch_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc))
                    return;
                break;
            case GemmKernelId::kSmallM:
                gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            case GemmKernelId::kSmallMWide:
                gemm_smallm_wide_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            case GemmKernelId::kAdaptiveTile:
                gemm_adaptive_tile_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            case GemmKernelId::kPacked:
                if (dispatch_via_registry(GemmDataType::kFP32, M, N, K,
                                          alpha, A, lda, B, ldb, beta, C, ldc))
                    return;
                // Fallback to adaptive tile if registry failed
                gemm_adaptive_tile_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
#endif
        }

        // Small-M uses dedicated fast path (no packing)
        // Phase B: M=2-7 with large N use wide driver (48-col macro-tiling + Kc blocking).
        // M=1 uses dedicated GEMV path. M=4-7 with small N falls through to adaptive tile.
        // Phase 13: When N*K is very large, prefer packed path with threading instead
        // of small-M path — packing overhead is amortized and threading is beneficial.
        // Phase 14: SVE kernel for small-M (predicate-based edge handling for irregular N).
        if (M < 8) {
#ifdef __ARM_FEATURE_SVE
            // SVE kernel: use predicate for N edge handling (irregular N, prime N)
            // This is cleaner than NEON scalar tail handling.
            gemm_smallm_fp32_sve(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
            return;
#endif
#ifdef __ARM_NEON
            if (M == 1) {
                gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
            // Phase 13: For M=4 with very large N*K, use packed registry path.
            // This enables 2D threading + huge pages, which outperforms small-M
            // wide driver for shapes like batch4-LLM (4×4096×4096).
            constexpr int64_t kLargeNKThreshold = 4 * 1024 * 1024;
            if (M == 4 && (int64_t)N * K > kLargeNKThreshold) {
                // Fall through to registry dispatch (packed + threaded)
                goto registry_dispatch;
            }
            // Phase 13I: M=6 with large N*K — use packed registry path.
            // 8x16 kernel with M-padding (6→8 rows) outperforms the adaptive tile
            // path for large shapes because packed B gives sequential access while
            // adaptive tile reads B with stride N (cache thrash for large K).
            // The 25% compute waste from 2 extra rows is compensated by better
            // cache utilization and the 8x16 kernel's higher compute density.
            if (M == 6 && (int64_t)N * K > kLargeNKThreshold) {
                if (dispatch_via_registry(GemmDataType::kFP32, M, N, K,
                                          alpha, A, lda, B, ldb, beta, C, ldc,
                                          /*allow_m_pad=*/true))
                    return;
                // If registry failed, fall through to adaptive tile
            }
            // M=5,7 with large N*K: use adaptive tile (now OpenMP-threaded)
            // rather than wide driver (single-threaded).
            if ((M == 5 || M == 7) && (int64_t)N * K > kLargeNKThreshold) {
                // Fall through to adaptive tile path below
            } else
            // M=2-7: use wide driver for N >= 48 (macro-tiling benefit).
            // M=2-3: always use wide driver (was original routing).
            // M=4-7: only for N >= 48 where 48-col panels amortize B loads.
            if (M >= 2 && (M < 4 || N >= 48)) {
                gemm_smallm_wide_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
#endif
        }

#ifdef __ARM_NEON
        // Phase 7D: small-K fast path (K ≤ 16): preloads B, shares across rows
        if (K <= 16) {
            gemm_smallK_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }

        // Phase 8/10/11: adaptive tile GEMM (autoGEMM-style, unpacked + Kc blocking)
        // Use for: (a) small vol shapes (packing overhead > compute), OR
        // (b) M=4-7 — asm kernels outperform packed 8x12 which zero-pads M to 8.
        if (M >= 4 && (
            (int64_t)M * N * K < kUnpackedFlopsThreshold ||
            M < 8
        )) {
            gemm_adaptive_tile_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
#endif

        registry_dispatch:
        if (dispatch_via_registry(GemmDataType::kFP32, M, N, K,
                                  alpha, A, lda, B, ldb, beta, C, ldc))
            return;
    }

    // Explicit NEON or fallback from registry
#ifdef __ARM_NEON
    if (algo == GemmAlgo::kNeonFp32 || algo == GemmAlgo::kAuto) {
        if (M == 1) {
            gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (M >= 2 && M < 8 && (M < 4 || N >= 48)) {
            gemm_smallm_wide_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (K <= 16) {
            gemm_smallK_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if (M >= 4 && (
                   (int64_t)M * N * K < kUnpackedFlopsThreshold ||
                   M < 8)) {
            gemm_adaptive_tile_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            gemm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        return;
    }
#endif

    gemm_naive_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_bf16(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

#ifdef __ARM_NEON
    const auto& hw = detect_arm_hwcaps();
    bool has_bf16 = (hw.hwcaps & static_cast<uint64_t>(HwCap::kBF16)) != 0;

    // Small-M BF16 kernel: use when BF16 hardware available and M <= 8.
    // For very small M (M <= 4), FP32 may be faster due to lower conversion overhead,
    // but for M=5-8, BF16 compute density wins.
    if (has_bf16 && M <= kGemmMrBf16) {
        gemm_smallm_bf16(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // M <= kGemmMrBf16/2 without BF16: FP32 is better (memory-bound, conversion overhead)
    if (M <= kGemmMrBf16 / 2) {
        gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Try registry dispatch
    if (dispatch_via_registry(GemmDataType::kBF16, M, N, K,
                              alpha, A, lda, B, ldb, beta, C, ldc))
        return;

    // Legacy fallback
    if (has_bf16) {
        gemm_driver_bf16(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_int8(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

#ifdef __ARM_NEON
    const auto& hw = detect_arm_hwcaps();
    bool has_i8mm = (hw.hwcaps & static_cast<uint64_t>(HwCap::kI8MM)) != 0;

    // Small-M INT8 kernel: use when I8MM hardware available and M <= 8.
    // For very small M (M <= 4), FP32 may be faster due to quantization overhead,
    // but for M=5-8, INT8 SMMLA compute density wins.
    if (has_i8mm && M <= kGemmMrInt8) {
        gemm_smallm_int8(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // M <= kGemmMrInt8/2 without I8MM: FP32 is better (memory-bound, quantization overhead)
    if (M <= kGemmMrInt8 / 2) {
        gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Try registry dispatch
    if (dispatch_via_registry(GemmDataType::kINT8, M, N, K,
                              alpha, A, lda, B, ldb, beta, C, ldc))
        return;

    // Legacy fallback
    if (has_i8mm) {
        gemm_driver_int8(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace dnnopt
