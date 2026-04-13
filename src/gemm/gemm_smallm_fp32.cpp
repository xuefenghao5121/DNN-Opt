/// @file gemm_smallm_fp32.cpp
/// Small-M specialized FP32 GEMM driver and microkernel.
///
/// For M < Mr (8), the standard 8×12 BLIS path wastes compute on zero-padded
/// rows and incurs unnecessary A-packing overhead. This module provides:
///   - A 1×48 NEON microkernel (12 accumulators, 4x K-unroll)
///   - M=1 driver: no packing at all (GEMV-like, direct B access)
///   - M=2-7 driver: pack B only, iterate rows with 1×48 kernel

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// Nr for small-M path: 48 columns (12 NEON registers)
static constexpr int kSmallMNr = 48;

/// 1×48 microkernel: computes one row of C from unpacked A row and B block.
///
/// C[0, 0:48] = alpha * sum_k(A[k] * B[k, 0:48]) + beta * C[0, 0:48]
///
/// @param K      number of K iterations
/// @param A      pointer to A row (contiguous, stride irrelevant for 1 row)
/// @param B      pointer to B block, layout depends on packed flag
/// @param ldb    stride of B (used when B is not packed; ignored when packed)
/// @param C      output row pointer
/// @param alpha  scaling factor
/// @param beta   scaling factor for existing C
/// @param packed if true, B is packed as kSmallMNr-wide panels (contiguous per K)
static void gemm_ukernel_fp32_1x48(int K,
                                    const float* A,
                                    const float* B, int ldb,
                                    float* C,
                                    float alpha, float beta,
                                    bool packed) {
    // 12 accumulator registers for 48 columns
    float32x4_t c0  = vdupq_n_f32(0), c1  = vdupq_n_f32(0);
    float32x4_t c2  = vdupq_n_f32(0), c3  = vdupq_n_f32(0);
    float32x4_t c4  = vdupq_n_f32(0), c5  = vdupq_n_f32(0);
    float32x4_t c6  = vdupq_n_f32(0), c7  = vdupq_n_f32(0);
    float32x4_t c8  = vdupq_n_f32(0), c9  = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);

    int k = 0;

    if (packed) {
        // Packed B: 48 contiguous floats per K iteration
        for (; k + 3 < K; k += 4) {
            // Iteration 0
            {
                float32x4_t a = vdupq_n_f32(A[k]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
            // Iteration 1
            {
                float32x4_t a = vdupq_n_f32(A[k + 1]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
            // Iteration 2
            {
                float32x4_t a = vdupq_n_f32(A[k + 2]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
            // Iteration 3
            {
                float32x4_t a = vdupq_n_f32(A[k + 3]);
                c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
                c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
                c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
                c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
                c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
                c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
                c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
                c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
                c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
                c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
                c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
                c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
                B += kSmallMNr;
            }
        }
        for (; k < K; ++k) {
            float32x4_t a = vdupq_n_f32(A[k]);
            c0  = vfmaq_f32(c0,  a, vld1q_f32(B));
            c1  = vfmaq_f32(c1,  a, vld1q_f32(B + 4));
            c2  = vfmaq_f32(c2,  a, vld1q_f32(B + 8));
            c3  = vfmaq_f32(c3,  a, vld1q_f32(B + 12));
            c4  = vfmaq_f32(c4,  a, vld1q_f32(B + 16));
            c5  = vfmaq_f32(c5,  a, vld1q_f32(B + 20));
            c6  = vfmaq_f32(c6,  a, vld1q_f32(B + 24));
            c7  = vfmaq_f32(c7,  a, vld1q_f32(B + 28));
            c8  = vfmaq_f32(c8,  a, vld1q_f32(B + 32));
            c9  = vfmaq_f32(c9,  a, vld1q_f32(B + 36));
            c10 = vfmaq_f32(c10, a, vld1q_f32(B + 40));
            c11 = vfmaq_f32(c11, a, vld1q_f32(B + 44));
            B += kSmallMNr;
        }
    } else {
        // Unpacked B: stride = ldb between rows
        for (; k + 3 < K; k += 4) {
            const float* b0 = B + (k)     * ldb;
            const float* b1 = B + (k + 1) * ldb;
            const float* b2 = B + (k + 2) * ldb;
            const float* b3 = B + (k + 3) * ldb;

            float32x4_t a0 = vdupq_n_f32(A[k]);
            c0  = vfmaq_f32(c0,  a0, vld1q_f32(b0));
            c1  = vfmaq_f32(c1,  a0, vld1q_f32(b0 + 4));
            c2  = vfmaq_f32(c2,  a0, vld1q_f32(b0 + 8));
            c3  = vfmaq_f32(c3,  a0, vld1q_f32(b0 + 12));
            c4  = vfmaq_f32(c4,  a0, vld1q_f32(b0 + 16));
            c5  = vfmaq_f32(c5,  a0, vld1q_f32(b0 + 20));
            c6  = vfmaq_f32(c6,  a0, vld1q_f32(b0 + 24));
            c7  = vfmaq_f32(c7,  a0, vld1q_f32(b0 + 28));
            c8  = vfmaq_f32(c8,  a0, vld1q_f32(b0 + 32));
            c9  = vfmaq_f32(c9,  a0, vld1q_f32(b0 + 36));
            c10 = vfmaq_f32(c10, a0, vld1q_f32(b0 + 40));
            c11 = vfmaq_f32(c11, a0, vld1q_f32(b0 + 44));

            float32x4_t a1 = vdupq_n_f32(A[k + 1]);
            c0  = vfmaq_f32(c0,  a1, vld1q_f32(b1));
            c1  = vfmaq_f32(c1,  a1, vld1q_f32(b1 + 4));
            c2  = vfmaq_f32(c2,  a1, vld1q_f32(b1 + 8));
            c3  = vfmaq_f32(c3,  a1, vld1q_f32(b1 + 12));
            c4  = vfmaq_f32(c4,  a1, vld1q_f32(b1 + 16));
            c5  = vfmaq_f32(c5,  a1, vld1q_f32(b1 + 20));
            c6  = vfmaq_f32(c6,  a1, vld1q_f32(b1 + 24));
            c7  = vfmaq_f32(c7,  a1, vld1q_f32(b1 + 28));
            c8  = vfmaq_f32(c8,  a1, vld1q_f32(b1 + 32));
            c9  = vfmaq_f32(c9,  a1, vld1q_f32(b1 + 36));
            c10 = vfmaq_f32(c10, a1, vld1q_f32(b1 + 40));
            c11 = vfmaq_f32(c11, a1, vld1q_f32(b1 + 44));

            float32x4_t a2 = vdupq_n_f32(A[k + 2]);
            c0  = vfmaq_f32(c0,  a2, vld1q_f32(b2));
            c1  = vfmaq_f32(c1,  a2, vld1q_f32(b2 + 4));
            c2  = vfmaq_f32(c2,  a2, vld1q_f32(b2 + 8));
            c3  = vfmaq_f32(c3,  a2, vld1q_f32(b2 + 12));
            c4  = vfmaq_f32(c4,  a2, vld1q_f32(b2 + 16));
            c5  = vfmaq_f32(c5,  a2, vld1q_f32(b2 + 20));
            c6  = vfmaq_f32(c6,  a2, vld1q_f32(b2 + 24));
            c7  = vfmaq_f32(c7,  a2, vld1q_f32(b2 + 28));
            c8  = vfmaq_f32(c8,  a2, vld1q_f32(b2 + 32));
            c9  = vfmaq_f32(c9,  a2, vld1q_f32(b2 + 36));
            c10 = vfmaq_f32(c10, a2, vld1q_f32(b2 + 40));
            c11 = vfmaq_f32(c11, a2, vld1q_f32(b2 + 44));

            float32x4_t a3 = vdupq_n_f32(A[k + 3]);
            c0  = vfmaq_f32(c0,  a3, vld1q_f32(b3));
            c1  = vfmaq_f32(c1,  a3, vld1q_f32(b3 + 4));
            c2  = vfmaq_f32(c2,  a3, vld1q_f32(b3 + 8));
            c3  = vfmaq_f32(c3,  a3, vld1q_f32(b3 + 12));
            c4  = vfmaq_f32(c4,  a3, vld1q_f32(b3 + 16));
            c5  = vfmaq_f32(c5,  a3, vld1q_f32(b3 + 20));
            c6  = vfmaq_f32(c6,  a3, vld1q_f32(b3 + 24));
            c7  = vfmaq_f32(c7,  a3, vld1q_f32(b3 + 28));
            c8  = vfmaq_f32(c8,  a3, vld1q_f32(b3 + 32));
            c9  = vfmaq_f32(c9,  a3, vld1q_f32(b3 + 36));
            c10 = vfmaq_f32(c10, a3, vld1q_f32(b3 + 40));
            c11 = vfmaq_f32(c11, a3, vld1q_f32(b3 + 44));
        }
        for (; k < K; ++k) {
            const float* bk = B + k * ldb;
            float32x4_t a = vdupq_n_f32(A[k]);
            c0  = vfmaq_f32(c0,  a, vld1q_f32(bk));
            c1  = vfmaq_f32(c1,  a, vld1q_f32(bk + 4));
            c2  = vfmaq_f32(c2,  a, vld1q_f32(bk + 8));
            c3  = vfmaq_f32(c3,  a, vld1q_f32(bk + 12));
            c4  = vfmaq_f32(c4,  a, vld1q_f32(bk + 16));
            c5  = vfmaq_f32(c5,  a, vld1q_f32(bk + 20));
            c6  = vfmaq_f32(c6,  a, vld1q_f32(bk + 24));
            c7  = vfmaq_f32(c7,  a, vld1q_f32(bk + 28));
            c8  = vfmaq_f32(c8,  a, vld1q_f32(bk + 32));
            c9  = vfmaq_f32(c9,  a, vld1q_f32(bk + 36));
            c10 = vfmaq_f32(c10, a, vld1q_f32(bk + 40));
            c11 = vfmaq_f32(c11, a, vld1q_f32(bk + 44));
        }
    }

    // Epilogue: C = alpha * acc + beta * C
    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

#define STORE_48(off, acc) do {                                        \
    if (beta == 0.0f)                                                  \
        vst1q_f32(C + (off), vmulq_f32(av, acc));                     \
    else                                                               \
        vst1q_f32(C + (off), vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + (off))), av, acc)); \
} while(0)

    STORE_48(0,  c0);  STORE_48(4,  c1);  STORE_48(8,  c2);
    STORE_48(12, c3);  STORE_48(16, c4);  STORE_48(20, c5);
    STORE_48(24, c6);  STORE_48(28, c7);  STORE_48(32, c8);
    STORE_48(36, c9);  STORE_48(40, c10); STORE_48(44, c11);

#undef STORE_48
}

/// Scalar tail for N-remainder < 4 (when N is not divisible by 4).
static void gemm_scalar_1xn(int K, int n_rem,
                             const float* A,
                             const float* B, int ldb,
                             float* C,
                             float alpha, float beta) {
    for (int j = 0; j < n_rem; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[k] * B[k * ldb + j];
        if (beta == 0.0f)
            C[j] = alpha * acc;
        else
            C[j] = alpha * acc + beta * C[j];
    }
}

/// NEON 1×(4n) helper for N-remainder between 4 and 47.
static void gemm_neon_1xn(int K, int n_cols,
                           const float* A,
                           const float* B, int ldb,
                           float* C,
                           float alpha, float beta) {
    // Process 4 columns at a time
    int j = 0;
    for (; j + 3 < n_cols; j += 4) {
        float32x4_t acc = vdupq_n_f32(0);
        for (int k = 0; k < K; ++k) {
            float32x4_t a = vdupq_n_f32(A[k]);
            acc = vfmaq_f32(acc, a, vld1q_f32(&B[k * ldb + j]));
        }
        float32x4_t av = vdupq_n_f32(alpha);
        if (beta == 0.0f)
            vst1q_f32(&C[j], vmulq_f32(av, acc));
        else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(&C[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C[j])), av, acc));
        }
    }
    // Scalar tail
    if (j < n_cols)
        gemm_scalar_1xn(K, n_cols - j, A, &B[j], ldb, &C[j], alpha, beta);
}

// ============================================================
// Multi-row small-M kernels (M=2,4 parallel processing)
// ============================================================

/// 2×N microkernel: process 2 rows together for better SIMD utilization.
/// Each row shares the same B access pattern, reducing memory traffic.
/// K must be small enough to fit accumulators in registers.
static void gemm_ukernel_fp32_2xN_kblock(int k_len,
                                          const float* A0, const float* A1,
                                          const float* B, int ldb,
                                          float32x4_t c0, float32x4_t c1,
                                          int j_start, int j_len) {
    for (int k = 0; k < k_len; ++k) {
        float32x4_t bk = vld1q_f32(&B[k * ldb + j_start]);
        c0 = vfmaq_n_f32(c0, bk, A0[k]);
        c1 = vfmaq_n_f32(c1, bk, A1[k]);
    }
}

/// 4×N microkernel: process 4 rows together with K blocking.
static void gemm_ukernel_fp32_4xN_kblock(int k_len,
                                          const float* A, int lda,
                                          const float* B, int ldb,
                                          float32x4_t c0, float32x4_t c1,
                                          float32x4_t c2, float32x4_t c3,
                                          int j_start) {
    const float* A0 = A;
    const float* A1 = A + lda;
    const float* A2 = A + 2 * lda;
    const float* A3 = A + 3 * lda;

    for (int k = 0; k < k_len; ++k) {
        float32x4_t bk = vld1q_f32(&B[k * ldb + j_start]);
        c0 = vfmaq_n_f32(c0, bk, A0[k]);
        c1 = vfmaq_n_f32(c1, bk, A1[k]);
        c2 = vfmaq_n_f32(c2, bk, A2[k]);
        c3 = vfmaq_n_f32(c3, bk, A3[k]);
    }
}

/// 2×N with K-blocking for large K matrices.
static void gemm_ukernel_fp32_2xN(int N, int K,
                                   const float* A0, const float* A1, int lda,
                                   const float* B, int ldb,
                                   float* C0, float* C1, int ldc,
                                   float alpha, float beta) {
    auto bp = get_gemm_blocking_params();
    int Kc = bp.Kc;  // K-block size

    // Process N in chunks of 4
    int j = 0;
    for (; j + 3 < N; j += 4) {
        float32x4_t c0 = vdupq_n_f32(0);
        float32x4_t c1 = vdupq_n_f32(0);

        // K blocking for cache efficiency
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            for (int k = 0; k < kc; ++k) {
                float32x4_t bk = vld1q_f32(&B[(pc + k) * ldb + j]);
                c0 = vfmaq_n_f32(c0, bk, A0[pc + k]);
                c1 = vfmaq_n_f32(c1, bk, A1[pc + k]);
            }
        }

        float32x4_t av = vdupq_n_f32(alpha);
        if (beta == 0.0f) {
            vst1q_f32(&C0[j], vmulq_f32(av, c0));
            vst1q_f32(&C1[j], vmulq_f32(av, c1));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(&C0[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C0[j])), av, c0));
            vst1q_f32(&C1[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C1[j])), av, c1));
        }
    }

    // Scalar tail
    for (; j < N; ++j) {
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int k = 0; k < K; ++k) {
            float bkj = B[k * ldb + j];
            sum0 += A0[k] * bkj;
            sum1 += A1[k] * bkj;
        }
        C0[j] = (beta == 0.0f) ? alpha * sum0 : alpha * sum0 + beta * C0[j];
        C1[j] = (beta == 0.0f) ? alpha * sum1 : alpha * sum1 + beta * C1[j];
    }
}

/// 4×N with K-blocking for large K matrices.
static void gemm_ukernel_fp32_4xN(int N, int K,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float* C, int ldc,
                                   float alpha, float beta) {
    auto bp = get_gemm_blocking_params();
    int Kc = bp.Kc;  // K-block size

    const float* A0 = A;
    const float* A1 = A + lda;
    const float* A2 = A + 2 * lda;
    const float* A3 = A + 3 * lda;
    float* C0 = C;
    float* C1 = C + ldc;
    float* C2 = C + 2 * ldc;
    float* C3 = C + 3 * ldc;

    // Process N in chunks of 4
    int j = 0;
    for (; j + 3 < N; j += 4) {
        float32x4_t c0 = vdupq_n_f32(0);
        float32x4_t c1 = vdupq_n_f32(0);
        float32x4_t c2 = vdupq_n_f32(0);
        float32x4_t c3 = vdupq_n_f32(0);

        // K blocking for cache efficiency
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            for (int k = 0; k < kc; ++k) {
                float32x4_t bk = vld1q_f32(&B[(pc + k) * ldb + j]);
                c0 = vfmaq_n_f32(c0, bk, A0[pc + k]);
                c1 = vfmaq_n_f32(c1, bk, A1[pc + k]);
                c2 = vfmaq_n_f32(c2, bk, A2[pc + k]);
                c3 = vfmaq_n_f32(c3, bk, A3[pc + k]);
            }
        }

        float32x4_t av = vdupq_n_f32(alpha);
        if (beta == 0.0f) {
            vst1q_f32(&C0[j], vmulq_f32(av, c0));
            vst1q_f32(&C1[j], vmulq_f32(av, c1));
            vst1q_f32(&C2[j], vmulq_f32(av, c2));
            vst1q_f32(&C3[j], vmulq_f32(av, c3));
        } else {
            float32x4_t bv = vdupq_n_f32(beta);
            vst1q_f32(&C0[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C0[j])), av, c0));
            vst1q_f32(&C1[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C1[j])), av, c1));
            vst1q_f32(&C2[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C2[j])), av, c2));
            vst1q_f32(&C3[j], vfmaq_f32(vmulq_f32(bv, vld1q_f32(&C3[j])), av, c3));
        }
    }

    // Scalar tail
    for (; j < N; ++j) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (int k = 0; k < K; ++k) {
            float bkj = B[k * ldb + j];
            sum0 += A0[k] * bkj;
            sum1 += A1[k] * bkj;
            sum2 += A2[k] * bkj;
            sum3 += A3[k] * bkj;
        }
        C0[j] = (beta == 0.0f) ? alpha * sum0 : alpha * sum0 + beta * C0[j];
        C1[j] = (beta == 0.0f) ? alpha * sum1 : alpha * sum1 + beta * C1[j];
        C2[j] = (beta == 0.0f) ? alpha * sum2 : alpha * sum2 + beta * C2[j];
        C3[j] = (beta == 0.0f) ? alpha * sum3 : alpha * sum3 + beta * C3[j];
    }
}

// ============================================================
// K-major GEMV: sequential B row access for bandwidth efficiency
// ============================================================

/// K-major GEMV for large K with strided B access.
///
/// Instead of processing N in small panels (strided B reads with 44KB jumps),
/// sweeps all N columns per K iteration. Each B[k*ldb + 0:N-1] row is read
/// sequentially, allowing the HW stream prefetcher to sustain full DRAM bandwidth.
///
/// Uses a temp buffer (N floats) for accumulators, which stays in cache
/// while B rows stream through L1D.
///
/// Single-threaded version. Call with N-column range for parallel case.
static void gemm_gemv_kmajor_st(int N, int K,
                                  const float* A,
                                  const float* B, int ldb, int j_offset,
                                  float alpha, float beta,
                                  float* C) {
    // Stack buffer: up to 8192 floats = 32KB (fits L1D)
    constexpr int kMaxStackN = 8192;
    float temp_stack[kMaxStackN];
    float* temp = (N <= kMaxStackN) ? temp_stack : new float[N];

    // Zero-init temp
    std::memset(temp, 0, N * sizeof(float));

    // 4x K-unrolled K-major sweep.
    // Key optimization: accumulate 4 B rows into temp before write-back,
    // reducing temp memory traffic by 4x (1 load+store per 4 FMAs vs 1 per 1).
    // Also enables better pipelining: 4 independent FMLA chains overlap with loads.
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float a0 = A[k], a1 = A[k + 1], a2 = A[k + 2], a3 = A[k + 3];
        float32x4_t a0v = vdupq_n_f32(a0);
        float32x4_t a1v = vdupq_n_f32(a1);
        float32x4_t a2v = vdupq_n_f32(a2);
        float32x4_t a3v = vdupq_n_f32(a3);

        const float* bk0 = B + k * ldb + j_offset;
        const float* bk1 = B + (k + 1) * ldb + j_offset;
        const float* bk2 = B + (k + 2) * ldb + j_offset;
        const float* bk3 = B + (k + 3) * ldb + j_offset;

        int j = 0;
        for (; j + 3 < N; j += 4) {
            float32x4_t t = vld1q_f32(temp + j);
            t = vfmaq_f32(t, a0v, vld1q_f32(bk0 + j));
            t = vfmaq_f32(t, a1v, vld1q_f32(bk1 + j));
            t = vfmaq_f32(t, a2v, vld1q_f32(bk2 + j));
            t = vfmaq_f32(t, a3v, vld1q_f32(bk3 + j));
            vst1q_f32(temp + j, t);
        }
        // Scalar tail
        for (; j < N; ++j)
            temp[j] += a0 * bk0[j] + a1 * bk1[j] + a2 * bk2[j] + a3 * bk3[j];
    }
    // K tail: single iteration at a time
    for (; k < K; ++k) {
        float ak = A[k];
        float32x4_t av = vdupq_n_f32(ak);
        const float* bk = B + k * ldb + j_offset;

        int j = 0;
        for (; j + 3 < N; j += 4) {
            float32x4_t t = vld1q_f32(temp + j);
            t = vfmaq_f32(t, av, vld1q_f32(bk + j));
            vst1q_f32(temp + j, t);
        }
        for (; j < N; ++j)
            temp[j] += ak * bk[j];
    }

    // Epilogue: C = alpha * temp + beta * C
    float32x4_t alv = vdupq_n_f32(alpha);
    int j = 0;
    if (beta == 0.0f) {
        for (; j + 3 < N; j += 4)
            vst1q_f32(C + j, vmulq_f32(alv, vld1q_f32(temp + j)));
    } else {
        float32x4_t bv = vdupq_n_f32(beta);
        for (; j + 3 < N; j += 4)
            vst1q_f32(C + j, vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + j)),
                                         alv, vld1q_f32(temp + j)));
    }
    for (; j < N; ++j)
        C[j] = (beta == 0.0f) ? alpha * temp[j]
                               : alpha * temp[j] + beta * C[j];

    if (N > kMaxStackN) delete[] temp;
}

/// K-major GEMV with OpenMP parallelization across N columns.
static void gemm_gemv_kmajor_fp32(int N, int K,
                                    const float* A,
                                    const float* B, int ldb,
                                    float alpha, float beta,
                                    float* C) {
#ifdef _OPENMP
    int n_threads = 1;
    if ((int64_t)N * K > 500000) {
        #pragma omp parallel
        {
            #pragma omp single
            n_threads = omp_get_num_threads();
        }
    }
    if (n_threads > 1 && N >= 256 * n_threads) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            int j_start = tid * (N / nth);
            int j_end = (tid == nth - 1) ? N : (tid + 1) * (N / nth);
            int my_N = j_end - j_start;
            if (my_N > 0)
                gemm_gemv_kmajor_st(my_N, K, A, B, ldb, j_start,
                                     alpha, beta, C + j_start);
        }
        return;
    }
#endif
    // Single-threaded
    gemm_gemv_kmajor_st(N, K, A, B, ldb, 0, alpha, beta, C);
}

// ============================================================
// Small-M drivers
// ============================================================

/// M=1 driver: optimized GEMV with panel-based processing + Kc blocking.
///
/// Phase 11: Added Kc blocking for cache-friendly B access.
/// For K=4096 with kPanelN=128: B per panel = K*128*4 = 2MB without blocking.
/// With Kc=128: B per block = 128*128*4 = 64KB, fits L1D exactly.
/// Accumulators persist across Kc blocks (no alpha/beta complication).
static void gemm_smallm1_fp32(int N, int K,
                               float alpha, const float* A, int lda,
                               const float* B, int ldb,
                               float beta, float* C, int ldc) {
    (void)lda; (void)ldc;  // M=1, stride not needed

    // NOTE: K-major sweep (K-outer/N-inner for sequential B reads) was evaluated
    // but found not beneficial on Neoverse N2. The temp buffer read-modify-write
    // per K iteration offsets the sequential B access benefit. The panel approach
    // keeps accumulators in SIMD registers (zero temp traffic), which is more
    // efficient despite the strided B access pattern.

    // Process N in panels of 48 columns using the optimized 1x48 kernel
    // (12 named accumulators, 4x K-unrolling, no register spills).
    // The 1x48 kernel always processes exactly 48 columns — only use for full panels.

    // For large shapes, parallelize N-panels across threads (embarrassingly parallel:
    // each panel writes to independent C[j0..j0+47] with no overlap).
    int64_t flops = (int64_t)2 * N * K;
#ifdef _OPENMP
    int n_threads = 1;
    if (flops > 200000) {  // 200K FLOPS threshold for threading
        #pragma omp parallel
        {
            #pragma omp single
            n_threads = omp_get_num_threads();
        }
    }
    if (n_threads > 1 && N >= kSmallMNr * n_threads) {
        // Parallel path: each thread gets a range of N-panels
        #pragma omp parallel for schedule(static)
        for (int j0 = 0; j0 + kSmallMNr <= N; j0 += kSmallMNr) {
            gemm_ukernel_fp32_1x48(K, A, B + j0, ldb, C + j0,
                                    alpha, beta, /*packed=*/false);
        }
        // Handle remaining columns in single thread
        int j0 = (N / kSmallMNr) * kSmallMNr;
        if (j0 < N) {
            gemm_neon_1xn(K, N - j0, A, B + j0, ldb, C + j0, alpha, beta);
        }
        return;
    }
#endif

    // Single-threaded path
    int j0 = 0;
    for (; j0 + kSmallMNr <= N; j0 += kSmallMNr) {
        gemm_ukernel_fp32_1x48(K, A, B + j0, ldb, C + j0,
                                alpha, beta, /*packed=*/false);
    }
    if (j0 < N) {
        gemm_neon_1xn(K, N - j0, A, B + j0, ldb, C + j0, alpha, beta);
    }
}

/// M=2-7 driver: process multiple rows together for better B cache reuse.
/// Uses 4-row and 2-row microkernels when possible.
static void gemm_smallm_multi_fp32(int M, int N, int K,
                                    float alpha, const float* A, int lda,
                                    const float* B, int ldb,
                                    float beta, float* C, int ldc) {
    int i = 0;

    // Process M=4 blocks
    for (; i + 3 < M; i += 4) {
        gemm_ukernel_fp32_4xN(N, K, A + i * lda, lda, B, ldb,
                               C + i * ldc, ldc, alpha, beta);
    }

    // Process M=2 blocks
    for (; i + 1 < M; i += 2) {
        gemm_ukernel_fp32_2xN(N, K,
                               A + i * lda, A + (i + 1) * lda, lda,
                               B, ldb,
                               C + i * ldc, C + (i + 1) * ldc, ldc,
                               alpha, beta);
    }

    // Process remaining single row
    if (i < M) {
        gemm_smallm1_fp32(N, K, alpha, A + i * lda, lda, B, ldb, beta, C + i * ldc, ldc);
    }
}

/// Public small-M driver entry point.
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
    if (M == 1) {
        gemm_smallm1_fp32(N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        gemm_smallm_multi_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
