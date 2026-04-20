#pragma once
/// @file arm_neon_sve_bridge.h
/// NEON-SVE Bridge: utilities for mixing NEON and SVE operations.
///
/// ARM ACLE provides arm_neon_sve_bridge.h with functions like
/// svset_neonq/svget_neonq to convert between NEON 128-bit vectors
/// and SVE variable-length vectors.
///
/// These intrinsics are available in GCC 10+ and Clang 13+.
/// For older compilers, we use inline assembly fallback.

#include <arm_neon.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

namespace dnnopt {

// ============================================================
// Core Bridge Functions: NEON <-> SVE conversion
// ============================================================

// Check if svset_neonq intrinsics are available
// Note: These are NOT available in Clang 15, despite __clang_major__ >= 13
// They require Clang 16+ or GCC 10+
#if __clang_major__ >= 16 || __GNUC__ >= 10
#define DNNOPT_HAS_NEON_SVE_BRIDGE 1
#else
#define DNNOPT_HAS_NEON_SVE_BRIDGE 0
#endif

/// Embed a NEON vector into the first 128-bit of an SVE vector.
inline svfloat32_t neon_to_sve_f32(float32x4_t neon) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_f32(svdup_n_f32(0.0f), neon);
#else
    // Fallback: store NEON to memory, load as SVE
    float tmp[4];
    vst1q_f32(tmp, neon);
    return svld1_f32(svptrue_b32(), tmp);
#endif
}

/// Extract the first 128-bit of an SVE vector as a NEON vector.
inline float32x4_t sve_to_neon_f32(svfloat32_t sve) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svget_neonq_f32(sve);
#else
    // Fallback: store SVE to memory, load as NEON
    float tmp[svcntw()];
    svst1_f32(svptrue_b32(), tmp, sve);
    return vld1q_f32(tmp);
#endif
}

// INT32: svint32_t <-> int32x4_t
inline svint32_t neon_to_sve_s32(int32x4_t neon) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_s32(svdup_n_s32(0), neon);
#else
    int32_t tmp[4];
    vst1q_s32(tmp, neon);
    return svld1_s32(svptrue_b32(), tmp);
#endif
}

inline int32x4_t sve_to_neon_s32(svint32_t sve) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svget_neonq_s32(sve);
#else
    int32_t tmp[svcntw()];
    svst1_s32(svptrue_b32(), tmp, sve);
    return vld1q_s32(tmp);
#endif
}

// INT8: svint8_t <-> int8x16_t
inline svint8_t neon_to_sve_s8(int8x16_t neon) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_s8(svdup_n_s8(0), neon);
#else
    int8_t tmp[16];
    vst1q_s8(tmp, neon);
    return svld1_s8(svptrue_b8(), tmp);
#endif
}

inline int8x16_t sve_to_neon_s8(svint8_t sve) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svget_neonq_s8(sve);
#else
    int8_t tmp[svcntb()];
    svst1_s8(svptrue_b8(), tmp, sve);
    return vld1q_s8(tmp);
#endif
}

// BF16: svbfloat16_t <-> bfloat16x8_t (requires BF16 support)
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
inline svbfloat16_t neon_to_sve_bf16(bfloat16x8_t neon) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_bf16(svdup_n_bf16(0), neon);
#else
    __bf16 tmp[8];
    vst1q_bf16(tmp, neon);
    return svld1_bf16(svptrue_b16(), tmp);
#endif
}

inline bfloat16x8_t sve_to_neon_bf16(svbfloat16_t sve) {
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svget_neonq_bf16(sve);
#else
    __bf16 tmp[svcntb() / 2];
    svst1_bf16(svptrue_b16(), tmp, sve);
    return vld1q_bf16(tmp);
#endif
}
#endif  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

// ============================================================
// Hybrid NEON-in-SVE operations
// ============================================================

/// Perform NEON FMLA on first 128-bit of SVE vector.
inline svfloat32_t sve_fmla_neon_prefix(svfloat32_t sve_acc,
                                         svfloat32_t sve_a,
                                         svfloat32_t sve_b) {
    float32x4_t acc_neon = sve_to_neon_f32(sve_acc);
    float32x4_t a_neon = sve_to_neon_f32(sve_a);
    float32x4_t b_neon = sve_to_neon_f32(sve_b);

    acc_neon = vfmaq_f32(acc_neon, a_neon, b_neon);

    // Re-embed NEON result into SVE (upper bits unchanged)
#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_f32(sve_acc, acc_neon);
#else
    // Fallback: zero upper bits and set lower 128-bit
    svfloat32_t result = neon_to_sve_f32(acc_neon);
    return result;
#endif
}

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
inline svfloat32_t sve_bfmmla_neon_prefix(svfloat32_t sve_acc,
                                           svbfloat16_t sve_a,
                                           svbfloat16_t sve_b) {
    bfloat16x8_t a_neon = sve_to_neon_bf16(sve_a);
    bfloat16x8_t b_neon = sve_to_neon_bf16(sve_b);
    float32x4_t acc_neon = sve_to_neon_f32(sve_acc);

    acc_neon = vbfmmlaq_f32(acc_neon, a_neon, b_neon);

#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_f32(sve_acc, acc_neon);
#else
    return neon_to_sve_f32(acc_neon);
#endif
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
inline svint32_t sve_smmla_neon_prefix(svint32_t sve_acc,
                                        svint8_t sve_a,
                                        svint8_t sve_b) {
    int8x16_t a_neon = sve_to_neon_s8(sve_a);
    int8x16_t b_neon = sve_to_neon_s8(sve_b);
    int32x4_t acc_neon = sve_to_neon_s32(sve_acc);

    acc_neon = vmmlaq_s32(acc_neon, a_neon, b_neon);

#if DNNOPT_HAS_NEON_SVE_BRIDGE
    return svset_neonq_s32(sve_acc, acc_neon);
#else
    return neon_to_sve_s32(acc_neon);
#endif
}
#endif

// ============================================================
// SVE Vector Length utilities
// ============================================================

inline bool is_sve_128bit() { return svcntb() == 16; }
inline bool is_sve_wide() { return svcntb() >= 32; }
inline int get_sve_bits() { return (int)svcntb() * 8; }
inline int get_sve_f32_count() { return (int)svcntw(); }

inline bool prefer_neon_over_sve(int M, int N, int K) {
    if (is_sve_128bit()) return true;
    if (M <= 8 && N <= 16 && K <= 64) return true;
    return false;
}

}  // namespace dnnopt

#else  // !__ARM_FEATURE_SVE

namespace dnnopt {
inline bool prefer_neon_over_sve(int, int, int) { return true; }
}

#endif  // __ARM_FEATURE_SVE