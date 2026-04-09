/// @file gemm_driver_unpacked_fp32.cpp
/// Unpacked fast paths for small-to-medium GEMM.
///
/// 1) Small-K kernel (K ≤ 16): Preloads B rows into registers,
///    shares across M rows. Eliminates K-loop overhead for tiny K.
/// 2) 8×12 unpacked microkernel: For M=8-32 with larger K.

#include "dnnopt/gemm/gemm_config.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================
// Small-K GEMM kernel (K ≤ 16)
// Key optimization: preload A outside j-loop (A read once per row-group)
// B is preloaded per j-block and shared across rows.
// For K ≤ 4: M_GROUP=4 (16 A scalars + 4 B SIMD + 4 C = 24 regs)
// For K ≤ 8: M_GROUP=4 (32 A scalars + 8 B SIMD + 4 C → some spill, OK)
// ============================================================

void gemm_smallK_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc) {
    int n_full4 = (N / 4) * 4;
    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bvv = vdupq_n_f32(beta);
    bool beta_zero = (beta == 0.0f);

    // Process M in groups of 4
    int i = 0;
    for (; i + 3 < M; i += 4) {
        const float* Ar[4] = {A + i*lda, A + (i+1)*lda, A + (i+2)*lda, A + (i+3)*lda};
        float* Cr[4] = {C + i*ldc, C + (i+1)*ldc, C + (i+2)*ldc, C + (i+3)*ldc};

        // Preload A values for all 4 rows (read once, reuse across all j-blocks)
        float ap[4][16];
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 4; ++r)
                ap[r][k] = Ar[r][k];

        // Full N-blocks (j_len == 4): use direct SIMD stores
        for (int j = 0; j < n_full4; j += 4) {
            // Preload B[k, j..j+3]
            float32x4_t bk[16];
            for (int k = 0; k < K; ++k)
                bk[k] = vld1q_f32(B + k * ldb + j);

            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);

            for (int k = 0; k < K; ++k) {
                c0 = vfmaq_n_f32(c0, bk[k], ap[0][k]);
                c1 = vfmaq_n_f32(c1, bk[k], ap[1][k]);
                c2 = vfmaq_n_f32(c2, bk[k], ap[2][k]);
                c3 = vfmaq_n_f32(c3, bk[k], ap[3][k]);
            }

            // Direct SIMD store (j_len == 4 guaranteed)
            float32x4_t s0 = vmulq_f32(av, c0);
            float32x4_t s1 = vmulq_f32(av, c1);
            float32x4_t s2 = vmulq_f32(av, c2);
            float32x4_t s3 = vmulq_f32(av, c3);
            if (!beta_zero) {
                s0 = vfmaq_f32(s0, bvv, vld1q_f32(Cr[0] + j));
                s1 = vfmaq_f32(s1, bvv, vld1q_f32(Cr[1] + j));
                s2 = vfmaq_f32(s2, bvv, vld1q_f32(Cr[2] + j));
                s3 = vfmaq_f32(s3, bvv, vld1q_f32(Cr[3] + j));
            }
            vst1q_f32(Cr[0] + j, s0);
            vst1q_f32(Cr[1] + j, s1);
            vst1q_f32(Cr[2] + j, s2);
            vst1q_f32(Cr[3] + j, s3);
        }

        // N tail (< 4 cols)
        for (int j = n_full4; j < N; ) {
            int j_len = std::min(4, N - j);
            float32x4_t bk[16];
            for (int k = 0; k < K; ++k)
                bk[k] = vld1q_f32(B + k * ldb + j);

            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);
            for (int k = 0; k < K; ++k) {
                c0 = vfmaq_n_f32(c0, bk[k], ap[0][k]);
                c1 = vfmaq_n_f32(c1, bk[k], ap[1][k]);
                c2 = vfmaq_n_f32(c2, bk[k], ap[2][k]);
                c3 = vfmaq_n_f32(c3, bk[k], ap[3][k]);
            }
            float32x4_t s0 = vmulq_f32(av, c0), s1 = vmulq_f32(av, c1);
            float32x4_t s2 = vmulq_f32(av, c2), s3 = vmulq_f32(av, c3);
            float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
            vst1q_f32(tmp0, s0); vst1q_f32(tmp1, s1);
            vst1q_f32(tmp2, s2); vst1q_f32(tmp3, s3);
            for (int jj = 0; jj < j_len; ++jj) {
                Cr[0][j+jj] = beta_zero ? tmp0[jj] : tmp0[jj] + beta * Cr[0][j+jj];
                Cr[1][j+jj] = beta_zero ? tmp1[jj] : tmp1[jj] + beta * Cr[1][j+jj];
                Cr[2][j+jj] = beta_zero ? tmp2[jj] : tmp2[jj] + beta * Cr[2][j+jj];
                Cr[3][j+jj] = beta_zero ? tmp3[jj] : tmp3[jj] + beta * Cr[3][j+jj];
            }
            j += j_len;
        }
    }

    // Handle remaining rows (< 4)
    for (; i < M; ++i) {
        const float* Ai = A + i * lda;
        float* Ci = C + i * ldc;

        // Preload A for this row
        float ap1[16];
        for (int k = 0; k < K; ++k) ap1[k] = Ai[k];

        for (int j = 0; j < n_full4; j += 4) {
            float32x4_t bk[16];
            for (int k = 0; k < K; ++k)
                bk[k] = vld1q_f32(B + k * ldb + j);

            float32x4_t c = vdupq_n_f32(0);
            for (int k = 0; k < K; ++k)
                c = vfmaq_n_f32(c, bk[k], ap1[k]);

            float32x4_t s = vmulq_f32(av, c);
            if (!beta_zero) s = vfmaq_f32(s, bvv, vld1q_f32(Ci + j));
            vst1q_f32(Ci + j, s);
        }
        for (int j = n_full4; j < N; ) {
            int j_len = std::min(4, N - j);
            float32x4_t bk[16];
            for (int k = 0; k < K; ++k)
                bk[k] = vld1q_f32(B + k * ldb + j);

            float32x4_t c = vdupq_n_f32(0);
            for (int k = 0; k < K; ++k)
                c = vfmaq_n_f32(c, bk[k], ap1[k]);

            float tmp[4];
            vst1q_f32(tmp, vmulq_f32(av, c));
            for (int jj = 0; jj < j_len; ++jj)
                Ci[j+jj] = beta_zero ? tmp[jj] : tmp[jj] + beta * Ci[j+jj];
            j += j_len;
        }
    }
}

// ============================================================
// 8×12 unpacked microkernel
// Reads A[i*lda+k] (gather) and B[k*ldb+j] (contiguous)
// 24 acc registers + 2 A gather + 3 B load = 29 regs
// ============================================================

static void ukernel_8x12_unpacked(int K,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float* C, int ldc,
                                   int j0, int n_len,
                                   float alpha, float beta) {
    // 8 rows × 3 acc each = 24 accumulators (c_i_j where i=row, j=N-segment)
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0), c42 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0), c52 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0), c62 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0), c72 = vdupq_n_f32(0);

    const float* A0 = A;
    const float* A1 = A + lda;
    const float* A2 = A + 2 * lda;
    const float* A3 = A + 3 * lda;
    const float* A4 = A + 4 * lda;
    const float* A5 = A + 5 * lda;
    const float* A6 = A + 6 * lda;
    const float* A7 = A + 7 * lda;

    int k = 0;
    // 4x K-unrolled main loop
    for (; k + 3 < K; k += 4) {
        for (int ki = 0; ki < 4; ++ki) {
            int kk = k + ki;
            // Gather A column (8 scalar loads)
            float a0 = A0[kk], a1 = A1[kk], a2 = A2[kk], a3 = A3[kk];
            float a4 = A4[kk], a5 = A5[kk], a6 = A6[kk], a7 = A7[kk];

            // Load B row (up to 3 SIMD loads)
            const float* bk = B + kk * ldb + j0;

            if (n_len > 0) {
                float32x4_t b0 = vld1q_f32(bk);
                c00 = vfmaq_n_f32(c00, b0, a0); c10 = vfmaq_n_f32(c10, b0, a1);
                c20 = vfmaq_n_f32(c20, b0, a2); c30 = vfmaq_n_f32(c30, b0, a3);
                c40 = vfmaq_n_f32(c40, b0, a4); c50 = vfmaq_n_f32(c50, b0, a5);
                c60 = vfmaq_n_f32(c60, b0, a6); c70 = vfmaq_n_f32(c70, b0, a7);
            }
            if (n_len > 4) {
                float32x4_t b1 = vld1q_f32(bk + 4);
                c01 = vfmaq_n_f32(c01, b1, a0); c11 = vfmaq_n_f32(c11, b1, a1);
                c21 = vfmaq_n_f32(c21, b1, a2); c31 = vfmaq_n_f32(c31, b1, a3);
                c41 = vfmaq_n_f32(c41, b1, a4); c51 = vfmaq_n_f32(c51, b1, a5);
                c61 = vfmaq_n_f32(c61, b1, a6); c71 = vfmaq_n_f32(c71, b1, a7);
            }
            if (n_len > 8) {
                float32x4_t b2 = vld1q_f32(bk + 8);
                c02 = vfmaq_n_f32(c02, b2, a0); c12 = vfmaq_n_f32(c12, b2, a1);
                c22 = vfmaq_n_f32(c22, b2, a2); c32 = vfmaq_n_f32(c32, b2, a3);
                c42 = vfmaq_n_f32(c42, b2, a4); c52 = vfmaq_n_f32(c52, b2, a5);
                c62 = vfmaq_n_f32(c62, b2, a6); c72 = vfmaq_n_f32(c72, b2, a7);
            }
        }
    }
    // Scalar K tail
    for (; k < K; ++k) {
        float a0 = A0[k], a1 = A1[k], a2 = A2[k], a3 = A3[k];
        float a4 = A4[k], a5 = A5[k], a6 = A6[k], a7 = A7[k];
        const float* bk = B + k * ldb + j0;

        if (n_len > 0) {
            float32x4_t b0 = vld1q_f32(bk);
            c00 = vfmaq_n_f32(c00, b0, a0); c10 = vfmaq_n_f32(c10, b0, a1);
            c20 = vfmaq_n_f32(c20, b0, a2); c30 = vfmaq_n_f32(c30, b0, a3);
            c40 = vfmaq_n_f32(c40, b0, a4); c50 = vfmaq_n_f32(c50, b0, a5);
            c60 = vfmaq_n_f32(c60, b0, a6); c70 = vfmaq_n_f32(c70, b0, a7);
        }
        if (n_len > 4) {
            float32x4_t b1 = vld1q_f32(bk + 4);
            c01 = vfmaq_n_f32(c01, b1, a0); c11 = vfmaq_n_f32(c11, b1, a1);
            c21 = vfmaq_n_f32(c21, b1, a2); c31 = vfmaq_n_f32(c31, b1, a3);
            c41 = vfmaq_n_f32(c41, b1, a4); c51 = vfmaq_n_f32(c51, b1, a5);
            c61 = vfmaq_n_f32(c61, b1, a6); c71 = vfmaq_n_f32(c71, b1, a7);
        }
        if (n_len > 8) {
            float32x4_t b2 = vld1q_f32(bk + 8);
            c02 = vfmaq_n_f32(c02, b2, a0); c12 = vfmaq_n_f32(c12, b2, a1);
            c22 = vfmaq_n_f32(c22, b2, a2); c32 = vfmaq_n_f32(c32, b2, a3);
            c42 = vfmaq_n_f32(c42, b2, a4); c52 = vfmaq_n_f32(c52, b2, a5);
            c62 = vfmaq_n_f32(c62, b2, a6); c72 = vfmaq_n_f32(c72, b2, a7);
        }
    }

    // Epilogue: store C
    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

#define STORE_ROW_8x12(row, cc0, cc1, cc2) do {                          \
    float* crow = C + (row) * ldc + j0;                                   \
    if (n_len > 0) {                                                       \
        int n0 = (n_len > 4) ? 4 : n_len;                                 \
        float tmp0[4];                                                     \
        if (beta == 0.0f) vst1q_f32(tmp0, vmulq_f32(av, cc0));           \
        else vst1q_f32(tmp0, vfmaq_f32(vmulq_f32(bv, vld1q_f32(crow)), av, cc0)); \
        memcpy(crow, tmp0, n0 * sizeof(float));                            \
    }                                                                      \
    if (n_len > 4) {                                                       \
        int n1 = (n_len > 8) ? 4 : n_len - 4;                             \
        float tmp1[4];                                                     \
        if (beta == 0.0f) vst1q_f32(tmp1, vmulq_f32(av, cc1));           \
        else vst1q_f32(tmp1, vfmaq_f32(vmulq_f32(bv, vld1q_f32(crow+4)), av, cc1)); \
        memcpy(crow+4, tmp1, n1 * sizeof(float));                          \
    }                                                                      \
    if (n_len > 8) {                                                       \
        int n2 = n_len - 8;                                                \
        float tmp2[4];                                                     \
        if (beta == 0.0f) vst1q_f32(tmp2, vmulq_f32(av, cc2));           \
        else vst1q_f32(tmp2, vfmaq_f32(vmulq_f32(bv, vld1q_f32(crow+8)), av, cc2)); \
        memcpy(crow+8, tmp2, n2 * sizeof(float));                          \
    }                                                                      \
} while(0)

    STORE_ROW_8x12(0, c00, c01, c02);
    STORE_ROW_8x12(1, c10, c11, c12);
    STORE_ROW_8x12(2, c20, c21, c22);
    STORE_ROW_8x12(3, c30, c31, c32);
    STORE_ROW_8x12(4, c40, c41, c42);
    STORE_ROW_8x12(5, c50, c51, c52);
    STORE_ROW_8x12(6, c60, c61, c62);
    STORE_ROW_8x12(7, c70, c71, c72);

#undef STORE_ROW_8x12
}

// ============================================================
// Public driver
// ============================================================

void gemm_driver_unpacked_fp32(int M, int N, int K,
                                float alpha, const float* A, int lda,
                                const float* B, int ldb,
                                float beta, float* C, int ldc) {
    // Process N in panels of Nr=12
    for (int j0 = 0; j0 < N; j0 += kGemmNrFp32) {
        int n_len = std::min(kGemmNrFp32, N - j0);

        // Process M in blocks of 8
        int i = 0;
        for (; i + 7 < M; i += 8) {
            ukernel_8x12_unpacked(K, A + i * lda, lda, B, ldb,
                                   C + i * ldc, ldc, j0, n_len, alpha, beta);
        }

        // Handle M remainder (< 8 rows) with wide smallm kernel
        if (i < M) {
            int m_rem = M - i;
            const float* A_rem[7];
            float* C_rem[7];
            for (int r = 0; r < m_rem; ++r) {
                A_rem[r] = A + (i + r) * lda;
                C_rem[r] = C + (i + r) * ldc;
            }

            // Inline small-M for the N-panel [j0, j0+n_len)
            int sub_full4 = (n_len / 4) * 4;
            int n_acc = (n_len + 3) / 4;
            float32x4_t* acc = reinterpret_cast<float32x4_t*>(
                __builtin_alloca(m_rem * n_acc * sizeof(float32x4_t)));
            for (int x = 0; x < m_rem * n_acc; ++x) acc[x] = vdupq_n_f32(0);

            for (int k = 0; k < K; ++k) {
                const float* bk = B + k * ldb + j0;
                for (int j4 = 0; j4 < sub_full4; j4 += 4) {
                    float32x4_t bv = vld1q_f32(bk + j4);
                    int aidx = j4 / 4;
                    for (int r = 0; r < m_rem; ++r) {
                        float aik = A_rem[r][k];
                        acc[r * n_acc + aidx] = vfmaq_n_f32(acc[r * n_acc + aidx], bv, aik);
                    }
                }
            }

            float32x4_t av = vdupq_n_f32(alpha);
            for (int r = 0; r < m_rem; ++r) {
                float* cr = C_rem[r] + j0;
                for (int j4 = 0; j4 < sub_full4; j4 += 4) {
                    int aidx = j4 / 4;
                    if (beta == 0.0f)
                        vst1q_f32(cr + j4, vmulq_f32(av, acc[r * n_acc + aidx]));
                    else {
                        float32x4_t bvv = vdupq_n_f32(beta);
                        vst1q_f32(cr + j4,
                            vfmaq_f32(vmulq_f32(bvv, vld1q_f32(cr + j4)), av, acc[r * n_acc + aidx]));
                    }
                }
                for (int j = sub_full4; j < n_len; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                        sum += A_rem[r][k] * B[k * ldb + j0 + j];
                    cr[j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * cr[j];
                }
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
