/// @file gemm_int8_native.cpp
/// Native INT8 GEMM: INT8 input → INT32 output.
///
/// This version takes pre-quantized INT8 input and produces INT32 output.
/// No quantization/dequantization overhead - pure INT8×INT8→INT32 compute.
///
/// Use case: Conv3D/Conv2D INT8 where input/filter are already quantized.
///
/// Implementation:
///   - SMMLA 8×8 tile for optimal throughput (4x compute density vs FP32)
///   - K-packing for 8-element groups
///   - Scalar fallback for edge cases

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#if defined(__ARM_FEATURE_MATMUL_INT8)
// SMMLA available
#endif
#endif

namespace dnnopt {

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_MATMUL_INT8)

/// SMMLA 8×8 microkernel for native INT8 GEMM.
/// Process 8 rows × 8 columns tile using SMMLA instructions.
/// A is row-major [M, K], B is column-major [N, K] (ldb = K).
/// Each vmmlaq_s32 computes 2×2 block from [row0,row1] × [col0,col1] with 8 K values each.
static inline void gemm_int8_tile_8x8_smmla(
    int K,
    const int8_t* A, int lda,
    const int8_t* B, int ldb,
    int32_t* C, int ldc) {

    // 16 INT32 accumulators for 8×8 tile (each 2×2 block)
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

    int k_tail = K % 8;
    int k_main = K - k_tail;

    for (int k = 0; k < k_main; k += 8) {
        // Load 8 K values for each of 8 rows of A
        // Each int8x8_t = 8 K values for one row
        int8x8_t a_r0 = vld1_s8(A + 0*lda + k);
        int8x8_t a_r1 = vld1_s8(A + 1*lda + k);
        int8x8_t a_r2 = vld1_s8(A + 2*lda + k);
        int8x8_t a_r3 = vld1_s8(A + 3*lda + k);
        int8x8_t a_r4 = vld1_s8(A + 4*lda + k);
        int8x8_t a_r5 = vld1_s8(A + 5*lda + k);
        int8x8_t a_r6 = vld1_s8(A + 6*lda + k);
        int8x8_t a_r7 = vld1_s8(A + 7*lda + k);

        // Combine into row pairs for vmmlaq_s32
        // Each int8x16_t = [row_even, row_odd] for 2 rows × 8 K
        int8x16_t a_pair0 = vcombine_s8(a_r0, a_r1);  // rows 0-1
        int8x16_t a_pair1 = vcombine_s8(a_r2, a_r3);  // rows 2-3
        int8x16_t a_pair2 = vcombine_s8(a_r4, a_r5);  // rows 4-5
        int8x16_t a_pair3 = vcombine_s8(a_r6, a_r7);  // rows 6-7

        // Load 8 K values for each of 8 columns of B (column-major [N, K])
        // B[j*ldb + k] gives column j at K position k
        int8x8_t b_c0 = vld1_s8(B + 0*ldb + k);
        int8x8_t b_c1 = vld1_s8(B + 1*ldb + k);
        int8x8_t b_c2 = vld1_s8(B + 2*ldb + k);
        int8x8_t b_c3 = vld1_s8(B + 3*ldb + k);
        int8x8_t b_c4 = vld1_s8(B + 4*ldb + k);
        int8x8_t b_c5 = vld1_s8(B + 5*ldb + k);
        int8x8_t b_c6 = vld1_s8(B + 6*ldb + k);
        int8x8_t b_c7 = vld1_s8(B + 7*ldb + k);

        // Combine into column pairs for vmmlaq_s32
        int8x16_t b_pair0 = vcombine_s8(b_c0, b_c1);  // cols 0-1
        int8x16_t b_pair1 = vcombine_s8(b_c2, b_c3);  // cols 2-3
        int8x16_t b_pair2 = vcombine_s8(b_c4, b_c5);  // cols 4-5
        int8x16_t b_pair3 = vcombine_s8(b_c6, b_c7);  // cols 6-7

        // 16 SMMLA instructions for 8×8 tile
        // Row pair 0 (rows 0-1) × all column pairs
        c00 = vmmlaq_s32(c00, a_pair0, b_pair0);
        c01 = vmmlaq_s32(c01, a_pair0, b_pair1);
        c02 = vmmlaq_s32(c02, a_pair0, b_pair2);
        c03 = vmmlaq_s32(c03, a_pair0, b_pair3);

        // Row pair 1 (rows 2-3)
        c10 = vmmlaq_s32(c10, a_pair1, b_pair0);
        c11 = vmmlaq_s32(c11, a_pair1, b_pair1);
        c12 = vmmlaq_s32(c12, a_pair1, b_pair2);
        c13 = vmmlaq_s32(c13, a_pair1, b_pair3);

        // Row pair 2 (rows 4-5)
        c20 = vmmlaq_s32(c20, a_pair2, b_pair0);
        c21 = vmmlaq_s32(c21, a_pair2, b_pair1);
        c22 = vmmlaq_s32(c22, a_pair2, b_pair2);
        c23 = vmmlaq_s32(c23, a_pair2, b_pair3);

        // Row pair 3 (rows 6-7)
        c30 = vmmlaq_s32(c30, a_pair3, b_pair0);
        c31 = vmmlaq_s32(c31, a_pair3, b_pair1);
        c32 = vmmlaq_s32(c32, a_pair3, b_pair2);
        c33 = vmmlaq_s32(c33, a_pair3, b_pair3);
    }

    // Store results: extract from 2×2 blocks
    // vmmlaq_s32 result layout: [r0c0, r0c1, r1c0, r1c1]
    // We need to extract rows 0-7 and columns 0-7

    // Extract row 0: c00[0], c00[1], c01[0], c01[1], c02[0], c02[1], c03[0], c03[1]
    // Row 0: low half of c00, c01, c02, c03

    // Store each row by extracting values from the 2×2 blocks
    for (int r = 0; r < 8; ++r) {
        int32_t row_vals[8];
        int pair = r / 2;  // 0-3
        int half = r % 2;  // 0 = low (row even), 1 = high (row odd)

        // Get the 4 accumulator vectors for this row pair
        int32x4_t acc0, acc1, acc2, acc3;
        switch (pair) {
            case 0: acc0 = c00; acc1 = c01; acc2 = c02; acc3 = c03; break;
            case 1: acc0 = c10; acc1 = c11; acc2 = c12; acc3 = c13; break;
            case 2: acc0 = c20; acc1 = c21; acc2 = c22; acc3 = c23; break;
            case 3: acc0 = c30; acc1 = c31; acc2 = c32; acc3 = c33; break;
        }

        // Extract 2 values from each accumulator
        if (half == 0) {
            // Even row: extract [0] and [1] from each
            row_vals[0] = vgetq_lane_s32(acc0, 0);
            row_vals[1] = vgetq_lane_s32(acc0, 1);
            row_vals[2] = vgetq_lane_s32(acc1, 0);
            row_vals[3] = vgetq_lane_s32(acc1, 1);
            row_vals[4] = vgetq_lane_s32(acc2, 0);
            row_vals[5] = vgetq_lane_s32(acc2, 1);
            row_vals[6] = vgetq_lane_s32(acc3, 0);
            row_vals[7] = vgetq_lane_s32(acc3, 1);
        } else {
            // Odd row: extract [2] and [3] from each
            row_vals[0] = vgetq_lane_s32(acc0, 2);
            row_vals[1] = vgetq_lane_s32(acc0, 3);
            row_vals[2] = vgetq_lane_s32(acc1, 2);
            row_vals[3] = vgetq_lane_s32(acc1, 3);
            row_vals[4] = vgetq_lane_s32(acc2, 2);
            row_vals[5] = vgetq_lane_s32(acc2, 3);
            row_vals[6] = vgetq_lane_s32(acc3, 2);
            row_vals[7] = vgetq_lane_s32(acc3, 3);
        }

        // Store row
        for (int c = 0; c < 8; ++c) {
            C[r * ldc + c] = row_vals[c];
        }
    }

    // Handle K tail (scalar)
    if (k_tail > 0) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                int32_t sum = C[r * ldc + c];
                for (int k = k_main; k < K; ++k) {
                    sum += (int32_t)A[r * lda + k] * (int32_t)B[c * ldb + k];
                }
                C[r * ldc + c] = sum;
            }
        }
    }
}

#endif  // __ARM_FEATURE_MATMUL_INT8
#endif  // __ARM_NEON

/// Native INT8 GEMM: INT8 input → INT32 output.
/// Computes C = A × B where A and B are INT8, C is INT32.
/// A is M×K row-major, B is K×N row-major (transposed to column-major for SMMLA).
void gemm_int8_int8int8int32(int M, int N, int K,
                              const int8_t* A, int lda,
                              const int8_t* B, int ldb,
                              int32_t* C, int ldc) {
#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_MATMUL_INT8)  // SMMLA enabled
    // Use SMMLA 8×8 tiles when dimensions align
    const int M_tiles = M / 8;
    const int N_tiles = N / 8;
    const int M_tail = M % 8;
    const int N_tail = N % 8;

    // Process 8×8 tiles
    for (int mi = 0; mi < M_tiles; ++mi) {
        for (int ni = 0; ni < N_tiles; ++ni) {
            gemm_int8_tile_8x8_smmla(K,
                A + mi * 8 * lda, lda,
                B + ni * 8 * ldb, ldb,  // B must be column-major: [N, K]
                C + mi * 8 * ldc + ni * 8, ldc);
        }
    }

    // Handle M/N tails with scalar
    // M tail (last M_tail rows)
    for (int i = M_tiles * 8; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;  // Initialize to 0
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)A[i * lda + k] * (int32_t)B[j * ldb + k];
            }
            C[i * ldc + j] = sum;
        }
    }

    // N tail (last N_tail columns)
    for (int i = 0; i < M_tiles * 8; ++i) {
        for (int j = N_tiles * 8; j < N; ++j) {
            int32_t sum = 0;  // Initialize to 0
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)A[i * lda + k] * (int32_t)B[j * ldb + k];
            }
            C[i * ldc + j] = sum;
        }
    }

    return;
#endif
#endif

    // Scalar fallback: C = A × B^T where B is [N, K]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)A[i * lda + k] * (int32_t)B[j * ldb + k];
            }
            C[i * ldc + j] = sum;
        }
    }
}

}  // namespace dnnopt