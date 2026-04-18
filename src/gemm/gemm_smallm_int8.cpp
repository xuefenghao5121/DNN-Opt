/// @file gemm_smallm_int8.cpp
/// Small-M INT8 GEMM kernels (M=1-7) without packing.
///
/// For small M, packing overhead dominates. These kernels compute
/// directly from FP32 input with dynamic quantization, avoiding
/// the packing step while leveraging SMMLA for compute.
///
/// Quantization strategy:
///   - Per-tensor symmetric quantization (scale = max_abs / 127)
///   - Quantize A and B on-the-fly during compute
///   - Dequantize result before writing to C

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/arm_neon_sve_bridge.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>  // SMMLA intrinsics
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================
// Dynamic quantization helpers
// ============================================================

/// Compute quantization scale from max absolute value.
static inline float compute_scale_from_max(float max_abs) {
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 127.0f;
}

/// Quantize FP32 vector to INT8 with given scale.
/// Clamps to [-128, 127] range.
static inline int8x16_t quantize_fp32_to_int8(const float* ptr, float scale, int count) {
    float inv_scale = 1.0f / scale;

    // Load and quantize
    float32x4_t f0 = vld1q_f32(ptr);
    float32x4_t f1 = (count > 4) ? vld1q_f32(ptr + 4) : vdupq_n_f32(0);
    float32x4_t f2 = (count > 8) ? vld1q_f32(ptr + 8) : vdupq_n_f32(0);
    float32x4_t f3 = (count > 12) ? vld1q_f32(ptr + 12) : vdupq_n_f32(0);

    // Scale and convert to int32
    int32x4_t i0 = vcvtq_s32_f32(vmulq_n_f32(f0, inv_scale));
    int32x4_t i1 = vcvtq_s32_f32(vmulq_n_f32(f1, inv_scale));
    int32x4_t i2 = vcvtq_s32_f32(vmulq_n_f32(f2, inv_scale));
    int32x4_t i3 = vcvtq_s32_f32(vmulq_n_f32(f3, inv_scale));

    // Narrow to int16, then to int8 with saturation
    int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));

    return vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
}

/// Compute max absolute value of a row/column.
static inline float compute_max_abs(const float* ptr, int count, int stride = 1) {
    float32x4_t max_v = vdupq_n_f32(0);

    int i = 0;
    if (stride == 1) {
        // Contiguous memory: use vectorized load
        for (; i + 4 <= count; i += 4) {
            float32x4_t v = vld1q_f32(ptr + i);
            max_v = vmaxq_f32(max_v, vabsq_f32(v));
        }
    } else {
        // Strided memory: scalar load
        for (; i < count; ++i) {
            float val = fabsf(ptr[i * stride]);
            max_v = vmaxq_f32(max_v, vdupq_n_f32(val));
        }
    }

    // Horizontal max
    float max_val = vmaxvq_f32(max_v);

    // Tail
    if (stride == 1) {
        for (; i < count; ++i) {
            max_val = std::max(max_val, fabsf(ptr[i]));
        }
    }

    return max_val;
}

// ============================================================
// M=1 INT8 GEMV
// ============================================================

/// INT8 GEMV: C[1xN] = alpha * dequant * (A_q8[1xK] * B_q8[KxN]) + beta * C[1xN]
/// Dynamic quantization: quantize A row once, then quantize B columns per output.
void gemm_mx1_int8(int K, int N,
                    float alpha, const float* A, int lda,
                    const float* B, int ldb,
                    float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    // Compute quantization scale for A
    float scale_A = compute_scale_from_max(compute_max_abs(A, K));
    float inv_scale_A = 1.0f / scale_A;

    // Quantize A row to INT8 (K elements, pad to multiple of 8)
    int k_padded = ((K + 7) / 8) * 8;
    std::vector<int8_t> a_i8(k_padded, 0);

    for (int k = 0; k < K; ++k) {
        int32_t q = (int32_t)lrintf(A[k] * inv_scale_A);
        a_i8[k] = std::max(-128, std::min(127, q));
    }

    // Process N columns in pairs (each SMMLA computes 2 outputs)
    // SMMLA layout: row-pair × col-pair = 2×2 INT32 output
    // For M=1, we need special handling...

    // For M=1 GEMV, SMMLA isn't ideal (it's optimized for M≥2).
    // We use dot product instead: SDOT for single row.
    // Or fallback to scalar if SMMLA overhead is too high.

    // Simple approach: quantize B on-the-fly, use scalar accumulate
    // This is still faster than FP32 GEMV due to 4x INT8 throughput

    for (int j = 0; j < N; ++j) {
        // Compute scale for this column of B
        float scale_B = compute_scale_from_max(compute_max_abs(B + j, K, ldb));

        int32_t sum = 0;
        const int8_t* a_ptr = a_i8.data();
        const float* b_col = B + j;

        int k8 = K / 8;
        for (int ki = 0; ki < k8; ++ki) {
            // Load 8 INT8 A values
            int8x16_t a_vec = vld1q_s8(a_ptr + ki * 8);
            // Replicate as row-pair: [a0..a7, 0..0] for SMMLA
            int8x16_t a_pair = vcombine_s8(vld1_s8(a_ptr + ki * 8), vdup_n_s8(0));

            // Quantize B column slice
            float b_vals[8];
            for (int kk = 0; kk < 8; ++kk) {
                b_vals[kk] = b_col[(ki * 8 + kk) * ldb];
            }
            float scale_B_slice = compute_scale_from_max(
                std::max(fabsf(b_vals[0]), std::max(fabsf(b_vals[1]),
                std::max(fabsf(b_vals[2]), std::max(fabsf(b_vals[3]),
                std::max(fabsf(b_vals[4]), std::max(fabsf(b_vals[5]),
                std::max(fabsf(b_vals[6]), fabsf(b_vals[7])))))))));

            int8x16_t b_vec = quantize_fp32_to_int8(b_vals, scale_B_slice, 8);

            // Use SDOT if available, or manual dot product
            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), a_pair, b_vec);
            sum += vgetq_lane_s32(dot, 0) + vgetq_lane_s32(dot, 1) +
                   vgetq_lane_s32(dot, 2) + vgetq_lane_s32(dot, 3);
        }

        // Tail K
        for (int k = k8 * 8; k < K; ++k) {
            float b_val = b_col[k * ldb];
            int8_t b_q = (int8_t)lrintf(b_val / scale_B);
            sum += a_i8[k] * b_q;
        }

        // Dequantize and store
        float dequant = scale_A * scale_B;
        float result = alpha * dequant * sum;
        if (beta == 0.0f) {
            C[j] = result;
        } else {
            C[j] = result + beta * C[j];
        }
    }

#else  // !__ARM_FEATURE_MATMUL_INT8
    // Fallback to FP32
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k] * B[k * ldb + j];
        }
        if (beta == 0.0f) C[j] = alpha * sum;
        else C[j] = alpha * sum + beta * C[j];
    }
#endif
}

// ============================================================
// M=2-7 INT8 Small-M kernels
// ============================================================

/// INT8 small-M kernel for M=2-8 using SMMLA.
/// Requires ARMv8.6-A I8MM support.
void gemm_smallm_int8_smmla(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    // Compute quantization scales
    float scale_A = compute_scale_from_max(compute_max_abs(A, M * K, 1));
    float scale_B = compute_scale_from_max(compute_max_abs(B, K * N, ldb));

    float dequant_scale = scale_A * scale_B;

    // Pad K to multiple of 8 (SMMLA processes 8 INT8 per K-step)
    int k_padded = ((K + 7) / 8) * 8;

    // Quantize A (M rows)
    std::vector<int8_t> a_i8(M * k_padded, 0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            int32_t q = (int32_t)lrintf(A[i * lda + k] / scale_A);
            a_i8[i * k_padded + k] = std::max(-128, std::min(127, q));
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

    // Process in 8x8 SMMLA tiles
    // SMMLA computes: 8×8 INT32 from 8×8 INT8 inputs (per 8 K values)
    // Tile layout: row-pairs × col-pairs

    for (int j = 0; j < N; j += 8) {
        int n_tile = std::min(8, N - j);

        for (int i = 0; i < M; i += 8) {
            int m_tile = std::min(8, M - i);

            // Quantize B tile (K × n_tile)
            std::vector<int8_t> b_tile(k_padded * n_tile, 0);
            for (int k = 0; k < K; ++k) {
                for (int jj = 0; jj < n_tile; ++jj) {
                    int32_t q = (int32_t)lrintf(B[k * ldb + j + jj] / scale_B);
                    b_tile[k * n_tile + jj] = std::max(-128, std::min(127, q));
                }
            }

            // Accumulators: 16 INT32 for 8×8 tile (each 2×2 block)
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
            int32x4_t c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
            int32x4_t c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
            int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
            int32x4_t c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
            int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
            int32x4_t c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

            // K-loop
            int k8 = k_padded / 8;
            const int8_t* a_ptr = &a_i8[i * k_padded];
            const int8_t* b_ptr = b_tile.data();

            for (int ki = 0; ki < k8; ++ki) {
                // Load A row-pairs (4 vectors for 8 rows)
                int8x16_t a0 = vld1q_s8(a_ptr + ki * 8);
                int8x16_t a1 = (m_tile >= 3) ? vld1q_s8(a_ptr + k_padded + ki * 8) : vdupq_n_s8(0);
                int8x16_t a2 = (m_tile >= 5) ? vld1q_s8(a_ptr + 2*k_padded + ki * 8) : vdupq_n_s8(0);
                int8x16_t a3 = (m_tile >= 7) ? vld1q_s8(a_ptr + 3*k_padded + ki * 8) : vdupq_n_s8(0);

                // Load B col-pairs
                int8x16_t b0 = vld1q_s8(b_ptr + ki * n_tile * 8);
                int8x16_t b1 = (n_tile >= 3) ? vld1q_s8(b_ptr + ki * n_tile * 8 + 16) : vdupq_n_s8(0);
                int8x16_t b2 = (n_tile >= 5) ? vld1q_s8(b_ptr + ki * n_tile * 8 + 32) : vdupq_n_s8(0);
                int8x16_t b3 = (n_tile >= 7) ? vld1q_s8(b_ptr + ki * n_tile * 8 + 48) : vdupq_n_s8(0);

                // 16 SMMLA
                c00 = vmmlaq_s32(c00, a0, b0);
                c01 = vmmlaq_s32(c01, a0, b1);
                c02 = vmmlaq_s32(c02, a0, b2);
                c03 = vmmlaq_s32(c03, a0, b3);
                c10 = vmmlaq_s32(c10, a1, b0);
                c11 = vmmlaq_s32(c11, a1, b1);
                c12 = vmmlaq_s32(c12, a1, b2);
                c13 = vmmlaq_s32(c13, a1, b3);
                c20 = vmmlaq_s32(c20, a2, b0);
                c21 = vmmlaq_s32(c21, a2, b1);
                c22 = vmmlaq_s32(c22, a2, b2);
                c23 = vmmlaq_s32(c23, a2, b3);
                c30 = vmmlaq_s32(c30, a3, b0);
                c31 = vmmlaq_s32(c31, a3, b1);
                c32 = vmmlaq_s32(c32, a3, b2);
                c33 = vmmlaq_s32(c33, a3, b3);

                b_ptr += n_tile * 8;
            }

            // Dequantize and store
            float scale = alpha * dequant_scale;
            float32x4_t scale_v = vdupq_n_f32(scale);
            float32x4_t beta_v = vdupq_n_f32(beta);

#define STORE_INT8_ROW_PAIR(row, a0, a1, a2, a3) do { \
    float32x4_t f0 = vcvtq_f32_s32(a0); \
    float32x4_t f1 = vcvtq_f32_s32(a1); \
    float32x4_t f2 = vcvtq_f32_s32(a2); \
    float32x4_t f3 = vcvtq_f32_s32(a3); \
    float32x2_t lo0 = vget_low_f32(f0), lo1 = vget_low_f32(f1); \
    float32x2_t lo2 = vget_low_f32(f2), lo3 = vget_low_f32(f3); \
    float32x2_t hi0 = vget_high_f32(f0), hi1 = vget_high_f32(f1); \
    float32x2_t hi2 = vget_high_f32(f2), hi3 = vget_high_f32(f3); \
    float32x4_t row0_lo = vmulq_f32(scale_v, vcombine_f32(lo0, lo1)); \
    float32x4_t row0_hi = vmulq_f32(scale_v, vcombine_f32(lo2, lo3)); \
    float32x4_t row1_lo = vmulq_f32(scale_v, vcombine_f32(hi0, hi1)); \
    float32x4_t row1_hi = vmulq_f32(scale_v, vcombine_f32(hi2, hi3)); \
    float* Cr0 = C + (row) * ldc + j; \
    float* Cr1 = C + ((row)+1) * ldc + j; \
    if (beta == 0.0f) { \
        vst1q_f32(Cr0, row0_lo); vst1q_f32(Cr0+4, row0_hi); \
        vst1q_f32(Cr1, row1_lo); vst1q_f32(Cr1+4, row1_hi); \
    } else { \
        vst1q_f32(Cr0, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr0)), scale_v, vcombine_f32(lo0, lo1))); \
    } \
} while(0)

            if (m_tile >= 2) STORE_INT8_ROW_PAIR(0, c00, c01, c02, c03);
            if (m_tile >= 4) STORE_INT8_ROW_PAIR(2, c10, c11, c12, c13);
            if (m_tile >= 6) STORE_INT8_ROW_PAIR(4, c20, c21, c22, c23);
            if (m_tile >= 8) STORE_INT8_ROW_PAIR(6, c30, c31, c32, c33);

#undef STORE_INT8_ROW_PAIR
        }
    }

#else  // !__ARM_FEATURE_MATMUL_INT8
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

/// INT8 Small-M dispatch: picks the right kernel for M=1-7.
void gemm_smallm_int8(int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc) {
    if (M == 1) {
        gemm_mx1_int8(K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (M <= 8) {
        gemm_smallm_int8_smmla(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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

void gemm_mx1_int8(int K, int N, float alpha, const float* A, int lda,
                   const float* B, int ldb, float beta, float* C, int ldc) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[k] * B[k * ldb + j];
        C[j] = alpha * sum + beta * C[j];
    }
}

void gemm_smallm_int8(int M, int N, int K, float alpha, const float* A, int lda,
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