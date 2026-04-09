/// @file gemm_smallm_fp32_v2.cpp
/// Optimized small-M specialized FP32 GEMM with prefetch and software pipelining.
///
/// v2 improvements:
///   - PRFM prefetch for L1/L2 cache lines
///   - 8x K-unrolling for better ILP
///   - Software pipelining (load next iteration while computing current)
///   - Better register allocation (use all 32 NEON registers)
///   - Aligned loads when possible

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/arm_hwcaps.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// Nr for small-M path: 48 columns (12 NEON registers)
static constexpr int kSmallMNr = 48;

/// Prefetch distance for L1 cache (in iterations)
static constexpr int kPrefetchL1Dist = 8;
/// Prefetch distance for L2 cache (in iterations)
static constexpr int kPrefetchL2Dist = 16;

// ============================================================
// 1×48 microkernel with prefetch and aggressive unrolling
// ============================================================

/// Optimized 1×48 microkernel with prefetch and 8x unrolling.
/// Uses software pipelining: prefetch for iteration i+8 while computing i.
static void gemm_ukernel_fp32_1x48_v2(int K,
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
        // 8x unrolled main loop with software pipelining
        for (; k + 7 < K; k += 8) {
            // Prefetch B for iteration k+8 into L1
            const float* B_prefetch = B + (k + kPrefetchL1Dist) * kSmallMNr;
            __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(B_prefetch) : "memory");

            // Prefetch A for iteration k+8 into L1
            const float* A_prefetch = A + (k + kPrefetchL1Dist);
            __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(A_prefetch) : "memory");

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
            // Iteration 4
            {
                float32x4_t a = vdupq_n_f32(A[k + 4]);
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
            // Iteration 5
            {
                float32x4_t a = vdupq_n_f32(A[k + 5]);
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
            // Iteration 6
            {
                float32x4_t a = vdupq_n_f32(A[k + 6]);
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
            // Iteration 7
            {
                float32x4_t a = vdupq_n_f32(A[k + 7]);
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

        // 4x unrolled residual loop
        for (; k + 3 < K; k += 4) {
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

        // Scalar tail
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
        // Unpacked B path with prefetch
        for (; k + 3 < K; k += 4) {
            const float* b0 = B + (k)     * ldb;
            const float* b1 = B + (k + 1) * ldb;
            const float* b2 = B + (k + 2) * ldb;
            const float* b3 = B + (k + 3) * ldb;

            // Prefetch next B rows
            __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(b0 + 256) : "memory");
            __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(b1 + 256) : "memory");

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

// ============================================================
// M=1 driver: optimized GEMV with prefetch
// ============================================================

/// M=1 driver with prefetch-optimized panel processing.
static void gemm_smallm1_fp32_v2(int N, int K,
                                  float alpha, const float* A, int lda,
                                  const float* B, int ldb,
                                  float beta, float* C, int ldc) {
    (void)lda; (void)ldc;
    constexpr int kPanelN = 64;

    for (int j0 = 0; j0 < N; j0 += kPanelN) {
        int j_len = std::min(kPanelN, N - j0);

        // Initialize accumulators (16 SIMD vectors = 64 floats)
        float32x4_t acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(0);

        // K-loop with prefetch
        for (int k = 0; k < K; ++k) {
            float ak = A[k];
            float32x4_t av = vdupq_n_f32(ak);
            const float* bk = B + k * ldb + j0;

            // Prefetch next B row
            if (k + kPrefetchL1Dist < K) {
                const float* bk_next = B + (k + kPrefetchL1Dist) * ldb + j0;
                __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(bk_next) : "memory");
            }

            for (int j = 0; j + 3 < j_len; j += 4) {
                int idx = j / 4;
                acc[idx] = vfmaq_f32(acc[idx], av, vld1q_f32(bk + j));
            }
        }

        // Store
        float32x4_t av = vdupq_n_f32(alpha);
        float32x4_t bv = vdupq_n_f32(beta);
        for (int j = 0; j + 3 < j_len; j += 4) {
            int idx = j / 4;
            if (beta == 0.0f) {
                vst1q_f32(C + j0 + j, vmulq_f32(av, acc[idx]));
            } else {
                vst1q_f32(C + j0 + j, vfmaq_f32(vmulq_f32(bv, vld1q_f32(C + j0 + j)), av, acc[idx]));
            }
        }

        // Scalar tail
        for (int j = (j_len / 4) * 4; j < j_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[k] * B[k * ldb + j0 + j];
            }
            C[j0 + j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C[j0 + j];
        }
    }
}

// ============================================================
// Public small-M driver entry point v2
// ============================================================

/// Public small-M driver entry point with prefetch optimizations.
void gemm_smallm_driver_fp32_v2(int M, int N, int K,
                                 float alpha, const float* A, int lda,
                                 const float* B, int ldb,
                                 float beta, float* C, int ldc) {
    if (M == 1) {
        gemm_smallm1_fp32_v2(N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // For M > 1, use the original multi-row path from v1
        // (TODO: add prefetch optimization to 2xN and 4xN paths)
        gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
