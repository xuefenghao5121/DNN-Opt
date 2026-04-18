/// @file gemm_fp8.cpp
/// FP8 GEMM framework: API declarations and stub implementations.
///
/// FP8 (Floating Point 8-bit) is introduced in ARMv9-A for AI acceleration.
/// Two formats:
///   - E4M3FN: sign(1) + exp(4) + mantissa(3), precision-optimized
///   - E5M2: sign(1) + exp(5) + mantissa(2), range-optimized
///
/// Hardware requirements:
///   - ARMv9-A CPU (Neoverse V3, Cortex-X4, etc.)
///   - __ARM_FEATURE_FP8 compiler support (GCC 13+, Clang 16+)
///
/// Current implementation: API + type definitions + stub fallback.
/// Full kernel implementation pending hardware availability.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_types.h"
#include "dnnopt/arm_hwcaps.h"

#include <cstring>

namespace dnnopt {

// ============================================================
// FP8 Hardware Support Detection
// ============================================================

/// Check if FP8 hardware support is available.
/// Returns true only if:
///   1. __ARM_FEATURE_FP8 is defined (compiler support)
///   2. Runtime CPU supports FP8 (ARMv9-A + FP8 extension)
bool has_fp8_support() {
#if defined(__ARM_FEATURE_FP8)
    // Compiler supports FP8 intrinsics
    // Runtime check: ARMv9-A CPUs with FP8 (V3, X4, etc.)
    auto& hw = detect_arm_hwcaps();
    return hw.has(kFP8);  // kFP8 defined in arm_hwcaps.h
#else
    // No compiler support, definitely no FP8
    return false;
#endif
}

// ============================================================
// FP8 GEMM API Implementation
// ============================================================

/// FP8 E4M3 GEMM: precision-optimized format.
/// A is FP8 E4M3, B is FP8 E5M2 (mixed precision).
/// C is FP32 output.
void gemm_fp8_e4m3(int M, int N, int K,
                   float alpha, const fp8_e4m3_t* A, int lda,
                   const fp8_e5m2_t* B, int ldb,
                   float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_FP8)
    if (has_fp8_support()) {
        // FP8 kernel implementation (requires ARM intrinsics)
        // Currently not implemented - hardware unavailable
        // When hardware available, use FDOT or F8MatMul intrinsics:
        //   - vfdot_f32: FP8 dot product
        //   - svmopa_fp8: SVE FP8 matrix multiply accumulate

        // Placeholder: fall through to software fallback
    }
#endif

    // Software fallback: convert FP8 to FP32, compute with FP32 GEMM
    std::vector<float> a_f32(M * K);
    std::vector<float> b_f32(K * N);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            a_f32[i * K + k] = (float)A[i * lda + k];
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            b_f32[k * N + j] = (float)B[k * ldb + j];
        }
    }

    gemm_fp32(M, N, K, alpha, a_f32.data(), K, b_f32.data(), N, beta, C, ldc);
}

/// FP8 E5M2 GEMM: range-optimized format.
/// Both A and B are FP8 E5M2.
void gemm_fp8_e5m2(int M, int N, int K,
                   float alpha, const fp8_e5m2_t* A, int lda,
                   const fp8_e5m2_t* B, int ldb,
                   float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_FP8)
    if (has_fp8_support()) {
        // FP8 kernel pending implementation
    }
#endif

    // Software fallback
    std::vector<float> a_f32(M * K);
    std::vector<float> b_f32(K * N);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            a_f32[i * K + k] = (float)A[i * lda + k];
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            b_f32[k * N + j] = (float)B[k * ldb + j];
        }
    }

    gemm_fp32(M, N, K, alpha, a_f32.data(), K, b_f32.data(), N, beta, C, ldc);
}

/// FP8 GEMM with FP32 input (auto-quantize).
/// Converts FP32 to FP8 E4M3/E5M2 internally.
void gemm_fp8(int M, int N, int K,
              float alpha, const float* A, int lda,
              const float* B, int ldb,
              float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_FP8)
    if (has_fp8_support()) {
        // Quantize FP32 to FP8, then compute
        // E4M3 for A (precision), E5M2 for B (range)
        std::vector<fp8_e4m3_t> a_fp8(M * K);
        std::vector<fp8_e5m2_t> b_fp8(K * N);

        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                a_fp8[i * K + k] = fp8_e4m3_t(A[i * lda + k]);
            }
        }
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                b_fp8[k * N + j] = fp8_e5m2_t(B[k * ldb + j]);
            }
        }

        gemm_fp8_e4m3(M, N, K, alpha, a_fp8.data(), K, b_fp8.data(), N, beta, C, ldc);
        return;
    }
#endif

    // Fallback to FP32 GEMM (no quantization overhead)
    gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace dnnopt