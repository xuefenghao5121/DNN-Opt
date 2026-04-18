/// @file gemm_smallm_bf16.cpp
/// Small-M BF16 GEMM kernels (M=1-7) without packing.
///
/// For small M, packing overhead dominates. These kernels compute
/// directly from FP32 input with inline BF16 conversion, avoiding
/// the packing step while leveraging BFMMLA for compute.
///
/// Strategies:
///   - M=1: GEMV with BF16 B vector, vectorized load
///   - M=2-7: Row-by-row compute, each row converts A slice to BF16
///   - Uses NEON-SVE bridge for edge handling on SVE hardware

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/arm_neon_sve_bridge.h"
#include "dnnopt/gemm/gemm_types.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_bf16.h>  // For __bf16 type if available
#endif

#include <algorithm>
#include <cstring>
#include <vector>

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================
// FP32 -> BF16 conversion helpers
// ============================================================

/// Convert 4 FP32 values to 4 BF16 values.
static inline bfloat16x4_t fp32_to_bf16_4(const float* ptr) {
    float32x4_t f32 = vld1q_f32(ptr);
    return vcvt_bf16_f32(f32);
}

/// Convert 8 FP32 values to 8 BF16 values (stored as two 4-element vectors).
static inline void fp32_to_bf16_8_store(const float* ptr, bfloat16_t* out) {
    float32x4_t f32_lo = vld1q_f32(ptr);
    float32x4_t f32_hi = vld1q_f32(ptr + 4);
    bfloat16x4_t bf16_lo = vcvt_bf16_f32(f32_lo);
    bfloat16x4_t bf16_hi = vcvt_bf16_f32(f32_hi);
    vst1_bf16(reinterpret_cast<__bf16*>(out), bf16_lo);
    vst1_bf16(reinterpret_cast<__bf16*>(out + 4), bf16_hi);
}

/// Load 8 BF16 values as bfloat16x8_t.
static inline bfloat16x8_t load_bf16_8(const bfloat16_t* ptr) {
    bfloat16x4_t lo = vld1_bf16(reinterpret_cast<const __bf16*>(ptr));
    bfloat16x4_t hi = vld1_bf16(reinterpret_cast<const __bf16*>(ptr + 4));
    return vcombine_bf16(lo, hi);
}

/// Create zero BF16 vector.
static inline bfloat16x4_t vdup_zero_bf16() {
    // Use reinterpret cast from zero float
    float32x4_t zero_f32 = vdupq_n_f32(0.0f);
    return vcvt_bf16_f32(zero_f32);
}

// ============================================================
// M=1 BF16 GEMV
// ============================================================

/// BF16 GEMV: C[1xN] = alpha * A[1xK] * B[KxN] + beta * C[1xN]
/// Uses BFMMLA in row-pair mode: each instruction computes 2 output elements.
void gemm_mx1_bf16(int K, int N,
                    float alpha, const float* A, int /*lda*/,
                    const float* B, int ldb,
                    float beta, float* C, int /*ldc*/) {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    // Convert A row to BF16 (K elements, pad to multiple of 4)
    int k_padded = ((K + 3) / 4) * 4;
    std::vector<bfloat16_t> a_bf16(k_padded);
    for (int k = 0; k < K; ++k) {
        a_bf16[k] = bfloat16_t(A[k]);
    }
    for (int k = K; k < k_padded; ++k) {
        a_bf16[k] = bfloat16_t(0.0f);
    }

    // Process N columns in pairs (each BFMMLA computes 2 cols)
    for (int j = 0; j < N; j += 2) {
        float32x4_t acc = vdupq_n_f32(0);

        // K-loop: each iteration processes 4 K values
        int k4 = k_padded / 4;

        for (int ki = 0; ki < k4; ++ki) {
            // Load 4 BF16 A values (row format)
            bfloat16x4_t a_bf = vld1_bf16(reinterpret_cast<const __bf16*>(a_bf16.data() + ki * 4));

            // Convert B columns to BF16
            float b0[4] = {0}, b1[4] = {0};
            for (int kk = 0; kk < 4; ++kk) {
                int k_idx = ki * 4 + kk;
                if (k_idx < K) {
                    b0[kk] = B[k_idx * ldb + j];
                    if (j + 1 < N) b1[kk] = B[k_idx * ldb + j + 1];
                }
            }
            bfloat16x4_t b0_bf = fp32_to_bf16_4(b0);
            bfloat16x4_t b1_bf = fp32_to_bf16_4(b1);
            bfloat16x8_t b_pair = vcombine_bf16(b0_bf, b1_bf);

            // BFMMLA: create row-pair from A (duplicate with zeros)
            bfloat16x8_t a_dup = vcombine_bf16(a_bf, vdup_zero_bf16());
            acc = vbfmmlaq_f32(acc, a_dup, b_pair);
        }

        // Extract results: acc = [c0, c1, _, _]
        float c0 = alpha * vgetq_lane_f32(acc, 0);
        float c1 = alpha * vgetq_lane_f32(acc, 1);

        if (beta == 0.0f) {
            C[j] = c0;
            if (j + 1 < N) C[j + 1] = c1;
        } else {
            C[j] = c0 + beta * C[j];
            if (j + 1 < N) C[j + 1] = c1 + beta * C[j + 1];
        }
    }

    // Handle remaining odd N
    if (N % 2 != 0) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k] * B[k * ldb + N - 1];
        }
        if (beta == 0.0f) C[N - 1] = alpha * sum;
        else C[N - 1] = alpha * sum + beta * C[N - 1];
    }
#else
    // Fallback to FP32 scalar
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k] * B[k * ldb + j];
        }
        C[j] = alpha * sum + beta * C[j];
    }
#endif
}

// ============================================================
// M=2-7 BF16 Small-M kernels
// ============================================================

/// BF16 small-M kernel for M=2-8 using BFMMLA.
void gemm_smallm_bf16_mops(int M, int N, int K,
                            float alpha, const float* A, int lda,
                            const float* B, int ldb,
                            float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    // Pad K to multiple of 4 (BFMMLA processes 4 BF16 per K-step)
    int k_padded = ((K + 3) / 4) * 4;

    // Convert A to BF16 (M rows)
    std::vector<bfloat16_t> a_bf16(M * k_padded);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            a_bf16[i * k_padded + k] = bfloat16_t(A[i * lda + k]);
        }
        for (int k = K; k < k_padded; ++k) {
            a_bf16[i * k_padded + k] = bfloat16_t(0.0f);
        }
    }

    // Initialize C with beta
    if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (beta == 0.0f) C[i * ldc + j] = 0.0f;
                else C[i * ldc + j] *= beta;
            }
        }
    }

    // Process in BFMMLA tiles (row-pairs × col-pairs)
    // Each BFMMLA: 2 rows × 2 cols × 4 K = 16 ops
    for (int j = 0; j < N; j += 2) {
        for (int i = 0; i < M; i += 2) {
            float32x4_t acc = vdupq_n_f32(0);

            // K-loop
            int k4 = k_padded / 4;
            for (int ki = 0; ki < k4; ++ki) {
                // Load A row-pair (2 rows × 4 K values = 8 BF16)
                const bfloat16_t* a_row0 = &a_bf16[i * k_padded + ki * 4];
                const bfloat16_t* a_row1 = (i + 1 < M) ? &a_bf16[(i + 1) * k_padded + ki * 4] : a_row0;

                bfloat16x4_t a0 = vld1_bf16(reinterpret_cast<const __bf16*>(a_row0));
                bfloat16x4_t a1 = vld1_bf16(reinterpret_cast<const __bf16*>(a_row1));
                bfloat16x8_t a_pair = vcombine_bf16(a0, a1);

                // Convert B col-pair to BF16
                float b0[4] = {0}, b1[4] = {0};
                for (int kk = 0; kk < 4; ++kk) {
                    int k_idx = ki * 4 + kk;
                    if (k_idx < K) {
                        b0[kk] = B[k_idx * ldb + j];
                        if (j + 1 < N) b1[kk] = B[k_idx * ldb + j + 1];
                    }
                }
                bfloat16x4_t b0_bf = fp32_to_bf16_4(b0);
                bfloat16x4_t b1_bf = fp32_to_bf16_4(b1);
                bfloat16x8_t b_pair = vcombine_bf16(b0_bf, b1_bf);

                // BFMMLA: acc[i:i+1, j:j+1] += a_pair × b_pair
                acc = vbfmmlaq_f32(acc, a_pair, b_pair);
            }

            // Store 2×2 block
            float scale = alpha;
            float r0c0 = scale * vgetq_lane_f32(acc, 0);
            float r0c1 = scale * vgetq_lane_f32(acc, 1);
            float r1c0 = scale * vgetq_lane_f32(acc, 2);
            float r1c1 = scale * vgetq_lane_f32(acc, 3);

            C[i * ldc + j] += r0c0;
            if (j + 1 < N) C[i * ldc + j + 1] += r0c1;
            if (i + 1 < M) {
                C[(i + 1) * ldc + j] += r1c0;
                if (j + 1 < N) C[(i + 1) * ldc + j + 1] += r1c1;
            }
        }
    }
#else
    // Fallback to FP32
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (beta == 0.0f) C[i * ldc + j] = alpha * sum;
            else C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
#endif
}

// ============================================================
// Dispatch wrapper
// ============================================================

/// BF16 Small-M dispatch: picks the right kernel for M=1-7.
void gemm_smallm_bf16(int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc) {
    if (M == 1) {
        gemm_mx1_bf16(K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (M <= 8) {
        gemm_smallm_bf16_mops(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // Fallback to FP32 kernel
        extern void gemm_smallm_driver_fp32(int M, int N, int K,
                                             float alpha, const float* A, int lda,
                                             const float* B, int ldb,
                                             float beta, float* C, int ldc);
        gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

#else  // !__ARM_NEON

void gemm_mx1_bf16(int K, int N, float alpha, const float* A, int lda,
                   const float* B, int ldb, float beta, float* C, int ldc) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[k] * B[k * ldb + j];
        C[j] = alpha * sum + beta * C[j];
    }
}

void gemm_smallm_bf16(int M, int N, int K, float alpha, const float* A, int lda,
                      const float* B, int ldb, float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt