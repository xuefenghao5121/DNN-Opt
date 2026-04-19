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
// Edge tile store helper
// ============================================================

/// Store partial tile results with n_tile boundary check.
/// Beta is already applied to C, so we just add vals.
static inline void store_edge_tile_row(
    float* C_row, int n_tile,
    float32x4_t row_lo, float32x4_t row_hi) {
    // Extract scalar values from vectors (already scaled)
    float vals[8];
    vst1q_f32(vals, row_lo);
    vst1q_f32(vals + 4, row_hi);

    // Store only valid elements (n_tile <= 8)
    // Beta is already applied, so just add vals[c] to C_row[c]
    for (int c = 0; c < n_tile && c < 8; ++c) {
        C_row[c] += vals[c];
    }
}

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
/// NOTE: count must be >= 4, otherwise use scalar quantization.
static inline int8x16_t quantize_fp32_to_int8(const float* ptr, float scale, int count) {
    float inv_scale = 1.0f / scale;

    // Load and quantize - only load valid elements
    float32x4_t f0 = (count >= 4) ? vld1q_f32(ptr) : vdupq_n_f32(0);
    float32x4_t f1 = (count >= 8) ? vld1q_f32(ptr + 4) : vdupq_n_f32(0);
    float32x4_t f2 = (count >= 12) ? vld1q_f32(ptr + 8) : vdupq_n_f32(0);
    float32x4_t f3 = (count >= 16) ? vld1q_f32(ptr + 12) : vdupq_n_f32(0);

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
    // Compute global quantization scales
    float max_A = 0.0f, max_B = 0.0f;
    for (int k = 0; k < K; ++k) {
        float av = std::fabs(A[k]);
        if (av > max_A) max_A = av;
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            float bv = std::fabs(B[k * ldb + j]);
            if (bv > max_B) max_B = bv;
        }
    }

    float scale_A = (max_A == 0.0f) ? 1.0f : max_A / 127.0f;
    float scale_B = (max_B == 0.0f) ? 1.0f : max_B / 127.0f;
    float dequant_scale = scale_A * scale_B;

    // Quantize A row to INT8
    std::vector<int8_t> a_i8(K, 0);
    for (int k = 0; k < K; ++k) {
        int32_t q = (int32_t)lrintf(A[k] / scale_A);
        a_i8[k] = std::max(-128, std::min(127, q));
    }

    // Quantize B to INT8 (column-major for GEMV)
    std::vector<int8_t> b_i8(K * N, 0);
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            int32_t q = (int32_t)lrintf(B[k * ldb + j] / scale_B);
            b_i8[k * N + j] = std::max(-128, std::min(127, q));
        }
    }

    // Scalar accumulate (safe for small K)
    for (int j = 0; j < N; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += a_i8[k] * b_i8[k * N + j];
        }
        float result = alpha * dequant_scale * sum;
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
/// Handles edge tiles for both M and N dimensions.
void gemm_smallm_int8_smmla(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    // Compute quantization scales: max(|A|) / 127, max(|B|) / 127
    // A is M×K row-major: iterate M rows, each row has K elements
    // B is K×N row-major: iterate K rows, each row has N elements
    float max_A = 0.0f, max_B = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float av = std::fabs(A[i * lda + k]);
            if (av > max_A) max_A = av;
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            float bv = std::fabs(B[k * ldb + j]);
            if (bv > max_B) max_B = bv;
        }
    }

    float scale_A = compute_scale_from_max(max_A);
    float scale_B = compute_scale_from_max(max_B);

    float dequant_scale = scale_A * scale_B;

    // Pad K to multiple of 8 (SMMLA processes 8 INT8 per K-step)
    int k_padded = ((K + 7) / 8) * 8;

    // Quantize A (M rows) - use fixed buffer
    int a_buf_size = M * k_padded;
    std::vector<int8_t> a_i8(a_buf_size, 0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            int32_t q = (int32_t)lrintf(A[i * lda + k] / scale_A);
            a_i8[i * k_padded + k] = std::max(-128, std::min(127, q));
        }
    }

    // Fixed B buffer (max 64 bytes per K-group)
    int b_buf_size = k_padded / 8 * 64;  // One tile's worth
    std::vector<int8_t> b_i8(b_buf_size, 0);

    // Initialize C with beta (same as original)
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

            // Quantize B tile: SMMLA format [K-group][col-pair]
            // Each 16 bytes = [col0_k0..k7, col1_k0..k7] for one col-pair
            // 4 col-pairs per K-group = 64 bytes per K-group
            int tile_k8 = k_padded / 8;  // K-groups in this tile
            // Use pre-allocated buffer
            int8_t* b_tile = b_i8.data();
            std::fill(b_i8.begin(), b_i8.end(), 0);  // Clear buffer

            int b_idx = 0;
            for (int ki = 0; ki < tile_k8; ++ki) {
                // Pack 4 col-pairs: (j+0,j+1), (j+2,j+3), (j+4,j+5), (j+6,j+7)
                for (int cp = 0; cp < 8; cp += 2) {
                    int c0 = j + cp, c1 = j + cp + 1;

                    for (int kk = 0; kk < 8; ++kk) {
                        int k = ki * 8 + kk;
                        float v0 = (k < K && c0 < N) ? B[k * ldb + c0] : 0;
                        float v1 = (k < K && c1 < N) ? B[k * ldb + c1] : 0;
                        int32_t q0 = (int32_t)lrintf(v0 / scale_B);
                        int32_t q1 = (int32_t)lrintf(v1 / scale_B);
                        b_tile[b_idx + kk] = (int8_t)std::max(-128, std::min(127, q0));
                        b_tile[b_idx + 8 + kk] = (int8_t)std::max(-128, std::min(127, q1));
                    }
                    b_idx += 16;
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
            const int8_t* a_tile_ptr = &a_i8[i * k_padded];  // Base pointer for this M tile
            const int8_t* b_tile_ptr = b_tile;               // Base pointer for this B tile

            for (int ki = 0; ki < tile_k8; ++ki) {
                // Load A row-pairs (4 vectors for 8 rows)
                // vmmlaq_s32 expects: [row0_k0..k7, row1_k0..k7]
                // Need to combine two rows' K values for each row-pair

                // Row-pair 0 (rows 0,1): combine row0 and row1 K values
                int8x8_t a_r0_k = vld1_s8(a_tile_ptr + ki * 8);                           // row0 K[ki*8..ki*8+7]
                int8x8_t a_r1_k = vld1_s8(a_tile_ptr + k_padded + ki * 8);                 // row1 K[ki*8..ki*8+7]
                int8x16_t a0 = vcombine_s8(a_r0_k, a_r1_k);                               // [row0_k, row1_k]

                // Row-pair 1 (rows 2,3)
                int8x8_t a_r2_k = (m_tile >= 3) ? vld1_s8(a_tile_ptr + 2*k_padded + ki * 8) : vdup_n_s8(0);
                int8x8_t a_r3_k = (m_tile >= 4) ? vld1_s8(a_tile_ptr + 3*k_padded + ki * 8) : vdup_n_s8(0);
                int8x16_t a1 = vcombine_s8(a_r2_k, a_r3_k);

                // Row-pair 2 (rows 4,5)
                int8x8_t a_r4_k = (m_tile >= 5) ? vld1_s8(a_tile_ptr + 4*k_padded + ki * 8) : vdup_n_s8(0);
                int8x8_t a_r5_k = (m_tile >= 6) ? vld1_s8(a_tile_ptr + 5*k_padded + ki * 8) : vdup_n_s8(0);
                int8x16_t a2 = vcombine_s8(a_r4_k, a_r5_k);

                // Row-pair 3 (rows 6,7)
                int8x8_t a_r6_k = (m_tile >= 7) ? vld1_s8(a_tile_ptr + 6*k_padded + ki * 8) : vdup_n_s8(0);
                int8x8_t a_r7_k = (m_tile >= 8) ? vld1_s8(a_tile_ptr + 7*k_padded + ki * 8) : vdup_n_s8(0);
                int8x16_t a3 = vcombine_s8(a_r6_k, a_r7_k);

                // Load B col-pairs (4 vectors for 8 cols)
                // Each vector = [col0_k0..k7, col1_k0..k7] for K-group ki
                // b_tile_ptr + ki * 64 + cp * 16
                int8x16_t b0 = vld1q_s8(b_tile_ptr + ki * 64);           // cols 0-1 at K-group ki
                int8x16_t b1 = (n_tile >= 3) ? vld1q_s8(b_tile_ptr + ki * 64 + 16) : vdupq_n_s8(0);  // cols 2-3
                int8x16_t b2 = (n_tile >= 5) ? vld1q_s8(b_tile_ptr + ki * 64 + 32) : vdupq_n_s8(0);  // cols 4-5
                int8x16_t b3 = (n_tile >= 7) ? vld1q_s8(b_tile_ptr + ki * 64 + 48) : vdupq_n_s8(0);  // cols 6-7

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
            }

            // Dequantize and store
            float scale = alpha * dequant_scale;
            float32x4_t scale_v = vdupq_n_f32(scale);

            // Store row-pair 0 (rows i+0, i+1)
            if (m_tile >= 2) {
                float32x4_t f0 = vcvtq_f32_s32(c00);
                float32x4_t f1 = vcvtq_f32_s32(c01);
                float32x4_t f2 = vcvtq_f32_s32(c02);
                float32x4_t f3 = vcvtq_f32_s32(c03);
                float32x2_t lo0 = vget_low_f32(f0), lo1 = vget_low_f32(f1);
                float32x2_t lo2 = vget_low_f32(f2), lo3 = vget_low_f32(f3);
                float32x2_t hi0 = vget_high_f32(f0), hi1 = vget_high_f32(f1);
                float32x2_t hi2 = vget_high_f32(f2), hi3 = vget_high_f32(f3);
                float32x4_t row0_lo = vmulq_f32(scale_v, vcombine_f32(lo0, lo1));
                float32x4_t row0_hi = vmulq_f32(scale_v, vcombine_f32(lo2, lo3));
                float32x4_t row1_lo = vmulq_f32(scale_v, vcombine_f32(hi0, hi1));
                float32x4_t row1_hi = vmulq_f32(scale_v, vcombine_f32(hi2, hi3));

                float* Cr0 = C + (i + 0) * ldc + j;
                float* Cr1 = C + (i + 1) * ldc + j;

                // Use vector store for full tiles, edge helper for partial
                if (n_tile == 8) {
                    // Beta already applied, just add vals
                    vst1q_f32(Cr0, vfmaq_f32(vld1q_f32(Cr0), vdupq_n_f32(1.0f), row0_lo));
                    vst1q_f32(Cr0 + 4, vfmaq_f32(vld1q_f32(Cr0 + 4), vdupq_n_f32(1.0f), row0_hi));
                    vst1q_f32(Cr1, vfmaq_f32(vld1q_f32(Cr1), vdupq_n_f32(1.0f), row1_lo));
                    vst1q_f32(Cr1 + 4, vfmaq_f32(vld1q_f32(Cr1 + 4), vdupq_n_f32(1.0f), row1_hi));
                } else {
                    store_edge_tile_row(Cr0, n_tile, row0_lo, row0_hi);
                    store_edge_tile_row(Cr1, n_tile, row1_lo, row1_hi);
                }
            }

            // Store row-pair 1 (rows i+2, i+3)
            if (m_tile >= 4) {
                float32x4_t f0 = vcvtq_f32_s32(c10);
                float32x4_t f1 = vcvtq_f32_s32(c11);
                float32x4_t f2 = vcvtq_f32_s32(c12);
                float32x4_t f3 = vcvtq_f32_s32(c13);
                float32x2_t lo0 = vget_low_f32(f0), lo1 = vget_low_f32(f1);
                float32x2_t lo2 = vget_low_f32(f2), lo3 = vget_low_f32(f3);
                float32x2_t hi0 = vget_high_f32(f0), hi1 = vget_high_f32(f1);
                float32x2_t hi2 = vget_high_f32(f2), hi3 = vget_high_f32(f3);
                float32x4_t row0_lo = vmulq_f32(scale_v, vcombine_f32(lo0, lo1));
                float32x4_t row0_hi = vmulq_f32(scale_v, vcombine_f32(lo2, lo3));
                float32x4_t row1_lo = vmulq_f32(scale_v, vcombine_f32(hi0, hi1));
                float32x4_t row1_hi = vmulq_f32(scale_v, vcombine_f32(hi2, hi3));

                float* Cr0 = C + (i + 2) * ldc + j;
                float* Cr1 = C + (i + 3) * ldc + j;

                if (n_tile == 8) {
                    vst1q_f32(Cr0, vfmaq_f32(vld1q_f32(Cr0), vdupq_n_f32(1.0f), row0_lo));
                    vst1q_f32(Cr0 + 4, vfmaq_f32(vld1q_f32(Cr0 + 4), vdupq_n_f32(1.0f), row0_hi));
                    vst1q_f32(Cr1, vfmaq_f32(vld1q_f32(Cr1), vdupq_n_f32(1.0f), row1_lo));
                    vst1q_f32(Cr1 + 4, vfmaq_f32(vld1q_f32(Cr1 + 4), vdupq_n_f32(1.0f), row1_hi));
                } else {
                    store_edge_tile_row(Cr0, n_tile, row0_lo, row0_hi);
                    store_edge_tile_row(Cr1, n_tile, row1_lo, row1_hi);
                }
            }

            // Store row-pair 2 (rows i+4, i+5)
            if (m_tile >= 5) {
                float32x4_t f0 = vcvtq_f32_s32(c20);
                float32x4_t f1 = vcvtq_f32_s32(c21);
                float32x4_t f2 = vcvtq_f32_s32(c22);
                float32x4_t f3 = vcvtq_f32_s32(c23);
                float32x2_t lo0 = vget_low_f32(f0), lo1 = vget_low_f32(f1);
                float32x2_t lo2 = vget_low_f32(f2), lo3 = vget_low_f32(f3);
                float32x2_t hi0 = vget_high_f32(f0), hi1 = vget_high_f32(f1);
                float32x2_t hi2 = vget_high_f32(f2), hi3 = vget_high_f32(f3);
                float32x4_t row0_lo = vmulq_f32(scale_v, vcombine_f32(lo0, lo1));
                float32x4_t row0_hi = vmulq_f32(scale_v, vcombine_f32(lo2, lo3));
                float32x4_t row1_lo = vmulq_f32(scale_v, vcombine_f32(hi0, hi1));
                float32x4_t row1_hi = vmulq_f32(scale_v, vcombine_f32(hi2, hi3));

                float* Cr0 = C + (i + 4) * ldc + j;
                float* Cr1 = C + (i + 5) * ldc + j;

                // Store row 4 always if m_tile >= 5, row 5 only if m_tile >= 6
                store_edge_tile_row(Cr0, n_tile, row0_lo, row0_hi);
                if (m_tile >= 6) {
                    store_edge_tile_row(Cr1, n_tile, row1_lo, row1_hi);
                }
            }

            // Store row-pair 3 (rows i+6, i+7) - only if m_tile >= 7
            if (m_tile >= 7) {
                float32x4_t f0 = vcvtq_f32_s32(c30);
                float32x4_t f1 = vcvtq_f32_s32(c31);
                float32x4_t f2 = vcvtq_f32_s32(c32);
                float32x4_t f3 = vcvtq_f32_s32(c33);
                float32x2_t lo0 = vget_low_f32(f0), lo1 = vget_low_f32(f1);
                float32x2_t lo2 = vget_low_f32(f2), lo3 = vget_low_f32(f3);
                float32x2_t hi0 = vget_high_f32(f0), hi1 = vget_high_f32(f1);
                float32x2_t hi2 = vget_high_f32(f2), hi3 = vget_high_f32(f3);
                float32x4_t row0_lo = vmulq_f32(scale_v, vcombine_f32(lo0, lo1));
                float32x4_t row0_hi = vmulq_f32(scale_v, vcombine_f32(lo2, lo3));
                float32x4_t row1_lo = vmulq_f32(scale_v, vcombine_f32(hi0, hi1));
                float32x4_t row1_hi = vmulq_f32(scale_v, vcombine_f32(hi2, hi3));

                float* Cr0 = C + (i + 6) * ldc + j;
                float* Cr1 = C + (i + 7) * ldc + j;

                if (n_tile == 8) {
                    vst1q_f32(Cr0, vfmaq_f32(vld1q_f32(Cr0), vdupq_n_f32(1.0f), row0_lo));
                    vst1q_f32(Cr0 + 4, vfmaq_f32(vld1q_f32(Cr0 + 4), vdupq_n_f32(1.0f), row0_hi));
                    vst1q_f32(Cr1, vfmaq_f32(vld1q_f32(Cr1), vdupq_n_f32(1.0f), row1_lo));
                    vst1q_f32(Cr1 + 4, vfmaq_f32(vld1q_f32(Cr1 + 4), vdupq_n_f32(1.0f), row1_hi));
                } else {
                    store_edge_tile_row(Cr0, n_tile, row0_lo, row0_hi);
                    store_edge_tile_row(Cr1, n_tile, row1_lo, row1_hi);
                }
            }
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