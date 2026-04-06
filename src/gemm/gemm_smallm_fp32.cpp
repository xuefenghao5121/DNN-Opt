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
// Small-M drivers
// ============================================================

/// M=1 driver: no packing, direct B access with 1×48 microkernel.
static void gemm_smallm1_fp32(int N, int K,
                               float alpha, const float* A, int lda,
                               const float* B, int ldb,
                               float beta, float* C, int ldc) {
    auto bp = get_gemm_blocking_params();
    int Kc = bp.Kc;

    // Outer N loop, step 48
    int j = 0;
    for (; j + kSmallMNr - 1 < N; j += kSmallMNr) {
        // K blocking
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            float beta_eff = (pc == 0) ? beta : 1.0f;
            float alpha_eff = (pc + kc >= K) ? alpha : 1.0f;
            gemm_ukernel_fp32_1x48(kc, &A[pc], &B[pc * ldb + j], ldb,
                                   &C[j], alpha_eff, beta_eff, /*packed=*/false);
        }
    }
    // N tail (< 48 columns)
    if (j < N) {
        int n_rem = N - j;
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            float beta_eff = (pc == 0) ? beta : 1.0f;
            float alpha_eff = (pc + kc >= K) ? alpha : 1.0f;
            gemm_neon_1xn(kc, n_rem, &A[pc], &B[pc * ldb + j], ldb,
                          &C[j], alpha_eff, beta_eff);
        }
    }
}

/// M=2-7 driver: no A packing, no B packing, iterate rows with direct B access.
/// B reuse across M rows is too small to justify packing overhead.
static void gemm_smallm_multi_fp32(int M, int N, int K,
                                    float alpha, const float* A, int lda,
                                    const float* B, int ldb,
                                    float beta, float* C, int ldc) {
    auto bp = get_gemm_blocking_params();
    int Kc = bp.Kc;

    for (int i = 0; i < M; ++i) {
        const float* a_row = &A[i * lda];
        float* c_row = &C[i * ldc];

        // Full 48-col panels
        int j = 0;
        for (; j + kSmallMNr - 1 < N; j += kSmallMNr) {
            for (int pc = 0; pc < K; pc += Kc) {
                int kc = std::min(Kc, K - pc);
                float beta_eff = (pc == 0) ? beta : 1.0f;
                float alpha_eff = (pc + kc >= K) ? alpha : 1.0f;
                gemm_ukernel_fp32_1x48(kc, &a_row[pc], &B[pc * ldb + j], ldb,
                                       &c_row[j], alpha_eff, beta_eff,
                                       /*packed=*/false);
            }
        }
        // N tail (< 48 columns)
        if (j < N) {
            int n_rem = N - j;
            for (int pc = 0; pc < K; pc += Kc) {
                int kc = std::min(Kc, K - pc);
                float beta_eff = (pc == 0) ? beta : 1.0f;
                float alpha_eff = (pc + kc >= K) ? alpha : 1.0f;
                gemm_neon_1xn(kc, n_rem, &a_row[pc], &B[pc * ldb + j], ldb,
                              &c_row[j], alpha_eff, beta_eff);
            }
        }
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
