/// @file gemm_ukernel_fp32_6x16.cpp
/// Packed 6x16 FP32 NEON micro-kernel for GEMM registry.
///
/// Purpose: enables packed+threaded path for M=6 shapes (e.g., 6×4096×4096)
/// where the 8x16 kernel would waste 25% compute on zero-padded rows 6-7.
///
/// Clang-optimized: uses vfmaq_laneq_f32 for rows 0-3 (fused broadcast+FMLA),
/// vfmaq_n_f32 for rows 4-5.
///
/// Register budget: 24 acc (6 rows × 4 quads) + 4 B = 28 SIMD regs.
/// 2x K-unrolling: 24 acc + 4 B = 28 (B regs reused between iterations).
///
/// Packed memory layout:
///   packed_A: for each k, 6 contiguous floats (Mr=6 rows)
///   packed_B: for each k, 16 contiguous floats (Nr=16 cols)

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __aarch64__

namespace dnnopt {

// ============================================================
// Pack A: 6 contiguous floats per K
// ============================================================

static void pack_a_fp32_6x16(int m_len, int k_len,
                              const float* A, int lda,
                              void* packed_A_v, int /*Mr*/, float* /*scale_out*/) {
    float* packed_A = static_cast<float*>(packed_A_v);
    constexpr int Mr = 6;
    for (int i = 0; i < m_len; i += Mr) {
        int m_rem = std::min(Mr, m_len - i);
        if (m_rem == Mr) {
            for (int k = 0; k < k_len; ++k) {
                packed_A[0] = A[(i + 0) * lda + k];
                packed_A[1] = A[(i + 1) * lda + k];
                packed_A[2] = A[(i + 2) * lda + k];
                packed_A[3] = A[(i + 3) * lda + k];
                packed_A[4] = A[(i + 4) * lda + k];
                packed_A[5] = A[(i + 5) * lda + k];
                packed_A += 6;
            }
        } else {
            // Tail: zero-pad to Mr=6
            for (int k = 0; k < k_len; ++k) {
                int r = 0;
                for (; r < m_rem; ++r)
                    packed_A[r] = A[(i + r) * lda + k];
                for (; r < Mr; ++r)
                    packed_A[r] = 0.0f;
                packed_A += 6;
            }
        }
    }
}

// ============================================================
// Pack B: 16 contiguous floats per K (identical to 8x16)
// ============================================================

static void pack_b_fp32_6x16(int k_len, int n_len,
                              const float* B, int ldb,
                              void* packed_B_v, int /*Nr*/, float* /*scale_out*/) {
    float* packed_B = static_cast<float*>(packed_B_v);
    constexpr int Nr = 16;
    for (int j = 0; j < n_len; j += Nr) {
        int n_rem = std::min(Nr, n_len - j);
        if (n_rem == Nr) {
            for (int k = 0; k < k_len; ++k) {
                const float* bk = &B[k * ldb + j];
                packed_B[0]  = bk[0];  packed_B[1]  = bk[1];
                packed_B[2]  = bk[2];  packed_B[3]  = bk[3];
                packed_B[4]  = bk[4];  packed_B[5]  = bk[5];
                packed_B[6]  = bk[6];  packed_B[7]  = bk[7];
                packed_B[8]  = bk[8];  packed_B[9]  = bk[9];
                packed_B[10] = bk[10]; packed_B[11] = bk[11];
                packed_B[12] = bk[12]; packed_B[13] = bk[13];
                packed_B[14] = bk[14]; packed_B[15] = bk[15];
                packed_B += 16;
            }
        } else {
            for (int k = 0; k < k_len; ++k) {
                const float* bk = &B[k * ldb + j];
                int c = 0;
                for (; c < n_rem; ++c)
                    packed_B[c] = bk[c];
                for (; c < Nr; ++c)
                    packed_B[c] = 0.0f;
                packed_B += 16;
            }
        }
    }
}

// ============================================================
// 6x16 packed microkernel
// ============================================================
// 24 acc (6 rows × 4 quads) + 4 B = 28 SIMD regs.
// Uses vfmaq_laneq_f32 for rows 0-3, vfmaq_n_f32 for rows 4-5.
// 2x K-unrolling: reuse B registers between iterations.

static void ukernel_fp32_6x16(int K,
                               const void* packed_A,
                               const void* packed_B,
                               float* C, int ldc,
                               float alpha, float beta,
                               float /*extra*/) {
    const float* __restrict__ pA = static_cast<const float*>(packed_A);
    const float* __restrict__ pB = static_cast<const float*>(packed_B);

    // 24 accumulators: rows 0-5 × quads 0-3
    float32x4_t a00=vdupq_n_f32(0),a01=vdupq_n_f32(0),a02=vdupq_n_f32(0),a03=vdupq_n_f32(0);
    float32x4_t a10=vdupq_n_f32(0),a11=vdupq_n_f32(0),a12=vdupq_n_f32(0),a13=vdupq_n_f32(0);
    float32x4_t a20=vdupq_n_f32(0),a21=vdupq_n_f32(0),a22=vdupq_n_f32(0),a23=vdupq_n_f32(0);
    float32x4_t a30=vdupq_n_f32(0),a31=vdupq_n_f32(0),a32=vdupq_n_f32(0),a33=vdupq_n_f32(0);
    float32x4_t a40=vdupq_n_f32(0),a41=vdupq_n_f32(0),a42=vdupq_n_f32(0),a43=vdupq_n_f32(0);
    float32x4_t a50=vdupq_n_f32(0),a51=vdupq_n_f32(0),a52=vdupq_n_f32(0),a53=vdupq_n_f32(0);

    // Prefetch distance
    const float* __restrict__ pA_pf = pA;
    const float* __restrict__ pB_pf = pB;

    int k = 0;
    for (; k + 1 < K; k += 2) {
        // --- Iteration k ---
        // Prefetch ahead
        __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(pB_pf + 32) : "memory");
        __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(pA_pf + 24) : "memory");
        pA_pf += 12;  // 2 K iterations × 6 floats
        pB_pf += 32;  // 2 K iterations × 16 floats

        float32x4_t b0 = vld1q_f32(pB);
        float32x4_t b1 = vld1q_f32(pB + 4);
        float32x4_t b2 = vld1q_f32(pB + 8);
        float32x4_t b3 = vld1q_f32(pB + 12);
        pB += 16;

        // Load A: rows 0-3 as quad, rows 4-5 as scalars
        float32x4_t ar04 = vld1q_f32(pA);
        float a4 = pA[4];
        float a5 = pA[5];
        pA += 6;

        // Rows 0-3: fused broadcast + FMLA via .s[N]
        a00=vfmaq_laneq_f32(a00,b0,ar04,0); a01=vfmaq_laneq_f32(a01,b1,ar04,0);
        a02=vfmaq_laneq_f32(a02,b2,ar04,0); a03=vfmaq_laneq_f32(a03,b3,ar04,0);
        a10=vfmaq_laneq_f32(a10,b0,ar04,1); a11=vfmaq_laneq_f32(a11,b1,ar04,1);
        a12=vfmaq_laneq_f32(a12,b2,ar04,1); a13=vfmaq_laneq_f32(a13,b3,ar04,1);
        a20=vfmaq_laneq_f32(a20,b0,ar04,2); a21=vfmaq_laneq_f32(a21,b1,ar04,2);
        a22=vfmaq_laneq_f32(a22,b2,ar04,2); a23=vfmaq_laneq_f32(a23,b3,ar04,2);
        a30=vfmaq_laneq_f32(a30,b0,ar04,3); a31=vfmaq_laneq_f32(a31,b1,ar04,3);
        a32=vfmaq_laneq_f32(a32,b2,ar04,3); a33=vfmaq_laneq_f32(a33,b3,ar04,3);

        // Rows 4-5: scalar broadcast + FMLA
        a40=vfmaq_n_f32(a40,b0,a4); a41=vfmaq_n_f32(a41,b1,a4);
        a42=vfmaq_n_f32(a42,b2,a4); a43=vfmaq_n_f32(a43,b3,a4);
        a50=vfmaq_n_f32(a50,b0,a5); a51=vfmaq_n_f32(a51,b1,a5);
        a52=vfmaq_n_f32(a52,b2,a5); a53=vfmaq_n_f32(a53,b3,a5);

        // --- Iteration k+1 ---
        b0 = vld1q_f32(pB);
        b1 = vld1q_f32(pB + 4);
        b2 = vld1q_f32(pB + 8);
        b3 = vld1q_f32(pB + 12);
        pB += 16;

        ar04 = vld1q_f32(pA);
        a4 = pA[4];
        a5 = pA[5];
        pA += 6;

        a00=vfmaq_laneq_f32(a00,b0,ar04,0); a01=vfmaq_laneq_f32(a01,b1,ar04,0);
        a02=vfmaq_laneq_f32(a02,b2,ar04,0); a03=vfmaq_laneq_f32(a03,b3,ar04,0);
        a10=vfmaq_laneq_f32(a10,b0,ar04,1); a11=vfmaq_laneq_f32(a11,b1,ar04,1);
        a12=vfmaq_laneq_f32(a12,b2,ar04,1); a13=vfmaq_laneq_f32(a13,b3,ar04,1);
        a20=vfmaq_laneq_f32(a20,b0,ar04,2); a21=vfmaq_laneq_f32(a21,b1,ar04,2);
        a22=vfmaq_laneq_f32(a22,b2,ar04,2); a23=vfmaq_laneq_f32(a23,b3,ar04,2);
        a30=vfmaq_laneq_f32(a30,b0,ar04,3); a31=vfmaq_laneq_f32(a31,b1,ar04,3);
        a32=vfmaq_laneq_f32(a32,b2,ar04,3); a33=vfmaq_laneq_f32(a33,b3,ar04,3);

        a40=vfmaq_n_f32(a40,b0,a4); a41=vfmaq_n_f32(a41,b1,a4);
        a42=vfmaq_n_f32(a42,b2,a4); a43=vfmaq_n_f32(a43,b3,a4);
        a50=vfmaq_n_f32(a50,b0,a5); a51=vfmaq_n_f32(a51,b1,a5);
        a52=vfmaq_n_f32(a52,b2,a5); a53=vfmaq_n_f32(a53,b3,a5);
    }

    // K tail (odd K)
    if (k < K) {
        float32x4_t b0 = vld1q_f32(pB);
        float32x4_t b1 = vld1q_f32(pB + 4);
        float32x4_t b2 = vld1q_f32(pB + 8);
        float32x4_t b3 = vld1q_f32(pB + 12);

        float32x4_t ar04 = vld1q_f32(pA);
        float a4 = pA[4];
        float a5 = pA[5];

        a00=vfmaq_laneq_f32(a00,b0,ar04,0); a01=vfmaq_laneq_f32(a01,b1,ar04,0);
        a02=vfmaq_laneq_f32(a02,b2,ar04,0); a03=vfmaq_laneq_f32(a03,b3,ar04,0);
        a10=vfmaq_laneq_f32(a10,b0,ar04,1); a11=vfmaq_laneq_f32(a11,b1,ar04,1);
        a12=vfmaq_laneq_f32(a12,b2,ar04,1); a13=vfmaq_laneq_f32(a13,b3,ar04,1);
        a20=vfmaq_laneq_f32(a20,b0,ar04,2); a21=vfmaq_laneq_f32(a21,b1,ar04,2);
        a22=vfmaq_laneq_f32(a22,b2,ar04,2); a23=vfmaq_laneq_f32(a23,b3,ar04,2);
        a30=vfmaq_laneq_f32(a30,b0,ar04,3); a31=vfmaq_laneq_f32(a31,b1,ar04,3);
        a32=vfmaq_laneq_f32(a32,b2,ar04,3); a33=vfmaq_laneq_f32(a33,b3,ar04,3);

        a40=vfmaq_n_f32(a40,b0,a4); a41=vfmaq_n_f32(a41,b1,a4);
        a42=vfmaq_n_f32(a42,b2,a4); a43=vfmaq_n_f32(a43,b3,a4);
        a50=vfmaq_n_f32(a50,b0,a5); a51=vfmaq_n_f32(a51,b1,a5);
        a52=vfmaq_n_f32(a52,b2,a5); a53=vfmaq_n_f32(a53,b3,a5);
    }

    // Store results
    float32x4_t av = vdupq_n_f32(alpha);
    #define STORE_ROW6X(r, a0n,a1n,a2n,a3n) do { \
        float* cr = C + (r)*ldc; \
        float32x4_t s0=vmulq_f32(av,a0n), s1=vmulq_f32(av,a1n), \
                    s2=vmulq_f32(av,a2n), s3=vmulq_f32(av,a3n); \
        if (beta != 0.0f) { \
            float32x4_t bv=vdupq_n_f32(beta); \
            s0=vfmaq_f32(s0,bv,vld1q_f32(cr)); s1=vfmaq_f32(s1,bv,vld1q_f32(cr+4)); \
            s2=vfmaq_f32(s2,bv,vld1q_f32(cr+8)); s3=vfmaq_f32(s3,bv,vld1q_f32(cr+12)); \
        } \
        vst1q_f32(cr,s0); vst1q_f32(cr+4,s1); \
        vst1q_f32(cr+8,s2); vst1q_f32(cr+12,s3); \
    } while(0)
    STORE_ROW6X(0, a00,a01,a02,a03);
    STORE_ROW6X(1, a10,a11,a12,a13);
    STORE_ROW6X(2, a20,a21,a22,a23);
    STORE_ROW6X(3, a30,a31,a32,a33);
    STORE_ROW6X(4, a40,a41,a42,a43);
    STORE_ROW6X(5, a50,a51,a52,a53);
    #undef STORE_ROW6X
}

// Wrapper for registry (matches ukernel_fn signature)
static void ukernel_fp32_6x16_wrap(int K,
                                    const void* A, const void* B,
                                    float* C, int ldc,
                                    float alpha, float beta,
                                    float extra) {
    ukernel_fp32_6x16(K, A, B, C, ldc, alpha, beta, extra);
}

// Register in the kernel registry
const GemmMicrokernelDesc neon_fp32_6x16_desc = {
    "neon_fp32_6x16",
    GemmDataType::kFP32,
    kNEON,                // required_hwcaps
    6,                    // Mr = 6
    16,                   // Nr = 16
    1,                    // Kgroup
    false,                // nr_is_vla
    115,                  // priority (between 8x16=120 and 8x12=100, prefer 6-row for M<=6)
    sizeof(float),        // packed_a_elem_bytes
    sizeof(float),        // packed_b_elem_bytes
    0,                    // min_sve_bits
    ukernel_fp32_6x16_wrap,
    pack_a_fp32_6x16,
    pack_b_fp32_6x16,
};

static RegisterKernel reg_neon_fp32_6x16(neon_fp32_6x16_desc);

#endif  // __aarch64__

}  // namespace dnnopt
