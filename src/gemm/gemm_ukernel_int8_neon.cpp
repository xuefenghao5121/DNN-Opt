/// @file gemm_ukernel_int8_neon.cpp
/// 8×8 INT8 SMMLA microkernel using NEON intrinsics.
///
/// Computes an 8×8 tile of C from packed INT8 A and B:
///   C[8×8] = alpha * dequant * (packed_A_i8[8×K] * packed_B_i8[K×8]) + beta * C[8×8]
///
/// packed_A layout: per K-group of 8, 4 row-pairs × 16 INT8 = 64 bytes
/// packed_B layout: per K-group of 8, 4 col-pairs × 16 INT8 = 64 bytes
/// K must be a multiple of 8 (guaranteed by packing).
///
/// Accumulator layout: 16 regs, each holds a 2×2 INT32 sub-block.
/// Same spatial arrangement as BF16 BFMMLA.
/// Each 2×2 block: [r0c0, r0c1, r1c0, r1c1].

#include "dnnopt/gemm/gemm_config.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

void gemm_ukernel_int8_8x8(int K,
                            const int8_t* packed_A,
                            const int8_t* packed_B,
                            float* C, int ldc,
                            float alpha, float beta,
                            float dequant_scale) {
    // 16 INT32 accumulators for 8×8 tile (each 2×2 INT32)
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

    // K-loop: process 8 K-elements per iteration
    // Each iteration: 4 A loads (64 INT8) + 4 B loads (64 INT8) + 16 SMMLA
    int k8 = K / 8;
    for (int ki = 0; ki < k8; ++ki) {
        // Load A row-pairs: 4 × int8x16_t
        int8x16_t a0 = vld1q_s8(packed_A);
        int8x16_t a1 = vld1q_s8(packed_A + 16);
        int8x16_t a2 = vld1q_s8(packed_A + 32);
        int8x16_t a3 = vld1q_s8(packed_A + 48);

        // Load B col-pairs: 4 × int8x16_t
        int8x16_t b0 = vld1q_s8(packed_B);
        int8x16_t b1 = vld1q_s8(packed_B + 16);
        int8x16_t b2 = vld1q_s8(packed_B + 32);
        int8x16_t b3 = vld1q_s8(packed_B + 48);

        // 16 SMMLA instructions
        // Row pair 0 (rows 0-1)
        c00 = vmmlaq_s32(c00, a0, b0);
        c01 = vmmlaq_s32(c01, a0, b1);
        c02 = vmmlaq_s32(c02, a0, b2);
        c03 = vmmlaq_s32(c03, a0, b3);

        // Row pair 1 (rows 2-3)
        c10 = vmmlaq_s32(c10, a1, b0);
        c11 = vmmlaq_s32(c11, a1, b1);
        c12 = vmmlaq_s32(c12, a1, b2);
        c13 = vmmlaq_s32(c13, a1, b3);

        // Row pair 2 (rows 4-5)
        c20 = vmmlaq_s32(c20, a2, b0);
        c21 = vmmlaq_s32(c21, a2, b1);
        c22 = vmmlaq_s32(c22, a2, b2);
        c23 = vmmlaq_s32(c23, a2, b3);

        // Row pair 3 (rows 6-7)
        c30 = vmmlaq_s32(c30, a3, b0);
        c31 = vmmlaq_s32(c31, a3, b1);
        c32 = vmmlaq_s32(c32, a3, b2);
        c33 = vmmlaq_s32(c33, a3, b3);

        packed_A += 64;  // 4 row-pairs × 16 INT8
        packed_B += 64;  // 4 col-pairs × 16 INT8
    }

    // Epilogue: convert INT32 accumulators → FP32, apply dequant_scale * alpha,
    // add beta * C, store to row-major C.
    // Same 2×2 block layout as BFMMLA: [r0c0, r0c1, r1c0, r1c1]
    // low 64-bit  = [r0c0, r0c1]
    // high 64-bit = [r1c0, r1c1]

    float32x4_t scale_v = vdupq_n_f32(dequant_scale * alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

#define STORE_ROW_PAIR_INT8(row, a0, a1, a2, a3) do {                   \
    /* Convert INT32 → FP32 */                                          \
    float32x4_t f0 = vcvtq_f32_s32(a0);                                \
    float32x4_t f1 = vcvtq_f32_s32(a1);                                \
    float32x4_t f2 = vcvtq_f32_s32(a2);                                \
    float32x4_t f3 = vcvtq_f32_s32(a3);                                \
                                                                         \
    /* Extract row 0: low halves */                                      \
    float32x2_t lo0 = vget_low_f32(f0);                                \
    float32x2_t lo1 = vget_low_f32(f1);                                \
    float32x2_t lo2 = vget_low_f32(f2);                                \
    float32x2_t lo3 = vget_low_f32(f3);                                \
    float32x4_t row0_lo = vcombine_f32(lo0, lo1);                      \
    float32x4_t row0_hi = vcombine_f32(lo2, lo3);                      \
                                                                         \
    /* Extract row 1: high halves */                                     \
    float32x2_t hi0 = vget_high_f32(f0);                               \
    float32x2_t hi1 = vget_high_f32(f1);                               \
    float32x2_t hi2 = vget_high_f32(f2);                               \
    float32x2_t hi3 = vget_high_f32(f3);                               \
    float32x4_t row1_lo = vcombine_f32(hi0, hi1);                      \
    float32x4_t row1_hi = vcombine_f32(hi2, hi3);                      \
                                                                         \
    float* Cr0 = C + (row) * ldc;                                      \
    float* Cr1 = C + ((row) + 1) * ldc;                                \
                                                                         \
    if (beta == 0.0f) {                                                 \
        vst1q_f32(Cr0,     vmulq_f32(scale_v, row0_lo));              \
        vst1q_f32(Cr0 + 4, vmulq_f32(scale_v, row0_hi));              \
        vst1q_f32(Cr1,     vmulq_f32(scale_v, row1_lo));              \
        vst1q_f32(Cr1 + 4, vmulq_f32(scale_v, row1_hi));              \
    } else {                                                            \
        vst1q_f32(Cr0,     vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr0)),     scale_v, row0_lo)); \
        vst1q_f32(Cr0 + 4, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr0 + 4)), scale_v, row0_hi)); \
        vst1q_f32(Cr1,     vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr1)),     scale_v, row1_lo)); \
        vst1q_f32(Cr1 + 4, vfmaq_f32(vmulq_f32(beta_v, vld1q_f32(Cr1 + 4)), scale_v, row1_hi)); \
    }                                                                   \
} while(0)

    STORE_ROW_PAIR_INT8(0, c00, c01, c02, c03);
    STORE_ROW_PAIR_INT8(2, c10, c11, c12, c13);
    STORE_ROW_PAIR_INT8(4, c20, c21, c22, c23);
    STORE_ROW_PAIR_INT8(6, c30, c31, c32, c33);

#undef STORE_ROW_PAIR_INT8
}

#endif  // __ARM_NEON

}  // namespace dnnopt
