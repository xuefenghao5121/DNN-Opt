#pragma once
/// @file arm_neon_sve_bridge.h
/// NEON-SVE Bridge: utilities for mixing NEON and SVE operations.
///
/// ARM ACLE provides arm_neon_sve_bridge.h (since GCC 10+ / Clang 11+)
/// with functions like svset_neonq/svget_neonq to convert between
/// NEON 128-bit vectors and SVE variable-length vectors.
///
/// Use cases:
///   1. Edge handling: NEON for tail elements (avoid predicate overhead)
///   2. Small tiles: NEON faster for M≤8, N≤12 on SVE-128 hardware
///   3. Hybrid compute: embed NEON ops in SVE loops
///   4. Register reuse: first 128-bit of SVE reg = NEON view
///
/// Reference: ARM ACLE Section 4.6 (NEON-SVE Bridge)

#include <arm_neon.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

namespace dnnopt {

// ============================================================
// Core Bridge Functions: NEON <-> SVE conversion
// ============================================================

/// Embed a NEON vector into the first 128-bit of an SVE vector.
/// The remaining bits (for VL > 128) are zeroed.
///
/// Example: Convert NEON float32x4_t to SVE svfloat32_t
///   float32x4_t neon_vec = vld1q_f32(ptr);
///   svfloat32_t sve_vec = neon_to_sve_f32(neon_vec);
inline svfloat32_t neon_to_sve_f32(float32x4_t neon) {
    // Use ACLE intrinsic if available, otherwise fallback
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svset_neonq_f32(svdup_n_f32(0), neon);
#else
    // Fallback: reinterpret as SVE (works when VL=128)
    svfloat32_t sve;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(sve) : "w"(neon));
    return sve;
#endif
}

/// Extract the first 128-bit of an SVE vector as a NEON vector.
/// Useful for processing tail elements with NEON after SVE loop.
///
/// Example: Get NEON view of SVE accumulator for final store
///   svfloat32_t sve_acc = ...;  // SVE accumulator
///   float32x4_t neon_acc = sve_to_neon_f32(sve_acc);
///   vst1q_f32(C + col, neon_acc);  // Store with NEON
inline float32x4_t sve_to_neon_f32(svfloat32_t sve) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svget_neonq_f32(sve);
#else
    // Fallback: extract low 128-bit
    float32x4_t neon;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(neon) : "w"(sve));
    return neon;
#endif
}

// ============================================================
// Type-specific bridge variants
// ============================================================

// INT32: svint32_t <-> int32x4_t
inline svint32_t neon_to_sve_s32(int32x4_t neon) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svset_neonq_s32(svdup_n_s32(0), neon);
#else
    svint32_t sve;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(sve) : "w"(neon));
    return sve;
#endif
}

inline int32x4_t sve_to_neon_s32(svint32_t sve) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svget_neonq_s32(sve);
#else
    int32x4_t neon;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(neon) : "w"(sve));
    return neon;
#endif
}

// INT8: svint8_t <-> int8x16_t
inline svint8_t neon_to_sve_s8(int8x16_t neon) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svset_neonq_s8(svdup_n_s8(0), neon);
#else
    svint8_t sve;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(sve) : "w"(neon));
    return sve;
#endif
}

inline int8x16_t sve_to_neon_s8(svint8_t sve) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svget_neonq_s8(sve);
#else
    int8x16_t neon;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(neon) : "w"(sve));
    return neon;
#endif
}

// BF16: svbfloat16_t <-> bfloat16x8_t
// Note: BF16 SVE intrinsics require __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
inline svbfloat16_t neon_to_sve_bf16(bfloat16x8_t neon) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svset_neonq_bf16(svdup_n_bf16(0), neon);
#else
    svbfloat16_t sve;
    // BF16 uses the same register representation as INT16
    __asm__ volatile("mov %0.v, %1.v" : "=w"(sve) : "w"(neon));
    return sve;
#endif
}

inline bfloat16x8_t sve_to_neon_bf16(svbfloat16_t sve) {
#if __GNUC__ >= 10 || __clang_major__ >= 11
    return svget_neonq_bf16(sve);
#else
    bfloat16x8_t neon;
    __asm__ volatile("mov %0.v, %1.v" : "=w"(neon) : "w"(sve));
    return neon;
#endif
}
#endif  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

// ============================================================
// Hybrid NEON-in-SVE operations
// ============================================================

/// Perform NEON FMLA on first 128-bit of SVE vector.
/// Useful for small tiles where NEON FMLA has better latency.
///
/// sve_acc[0:127] += sve_a[0:127] * sve_b[0:127]
/// The remaining bits of sve_acc are unchanged.
inline svfloat32_t sve_fmla_neon_prefix(svfloat32_t sve_acc,
                                         svfloat32_t sve_a,
                                         svfloat32_t sve_b) {
    // Extract NEON views
    float32x4_t acc_neon = sve_to_neon_f32(sve_acc);
    float32x4_t a_neon = sve_to_neon_f32(sve_a);
    float32x4_t b_neon = sve_to_neon_f32(sve_b);

    // NEON FMLA (single instruction, better latency for small ops)
    acc_neon = vfmaq_f32(acc_neon, a_neon, b_neon);

    // Re-embed into SVE (preserves upper bits of sve_acc if VL > 128)
    return svset_neonq_f32(sve_acc, acc_neon);
}

/// Perform NEON BFMMLA on first 128-bit of SVE BF16 vectors.
/// This is the optimal path for SVE-128 BF16 GEMM.
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
inline svfloat32_t sve_bfmmla_neon_prefix(svfloat32_t sve_acc,
                                           svbfloat16_t sve_a,
                                           svbfloat16_t sve_b) {
    // Convert to NEON BF16 views
    bfloat16x8_t a_neon = sve_to_neon_bf16(sve_a);
    bfloat16x8_t b_neon = sve_to_neon_bf16(sve_b);
    float32x4_t acc_neon = sve_to_neon_f32(sve_acc);

    // NEON BFMMLA: 8x8 tile in one instruction
    acc_neon = vbfmmlaq_f32(acc_neon, a_neon, b_neon);

    return svset_neonq_f32(sve_acc, acc_neon);
}
#endif

/// Perform NEON SMMLA (INT8 matmul) on first 128-bit of SVE vectors.
#if defined(__ARM_FEATURE_MATMUL_INT8)
inline svint32_t sve_smmla_neon_prefix(svint32_t sve_acc,
                                        svint8_t sve_a,
                                        svint8_t sve_b) {
    int8x16_t a_neon = sve_to_neon_s8(sve_a);
    int8x16_t b_neon = sve_to_neon_s8(sve_b);
    int32x4_t acc_neon = sve_to_neon_s32(sve_acc);

    // NEON SMMLA: 8x8 INT8 tile in one instruction
    acc_neon = vmmlaq_s32(acc_neon, a_neon, b_neon);

    return svset_neonq_s32(sve_acc, acc_neon);
}
#endif

// ============================================================
// SVE Vector Length utilities
// ============================================================

/// Check if SVE vector length is 128-bit (minimum SVE).
/// On SVE-128, NEON and SVE have identical register views,
/// making NEON operations optimal.
inline bool is_sve_128bit() {
    return svcntb() == 16;  // 128 bits = 16 bytes
}

/// Check if SVE vector length is >= 256-bit.
/// On wide SVE, VLA kernels have advantage over NEON.
inline bool is_sve_wide() {
    return svcntb() >= 32;  // >= 256 bits
}

/// Get SVE vector length in bits.
inline int get_sve_bits() {
    return (int)svcntb() * 8;
}

/// Get SVE vector length in FP32 elements.
inline int get_sve_f32_count() {
    return (int)svcntw();
}

// ============================================================
// Hybrid dispatch helper
// ============================================================

/// Decide whether to use NEON or SVE for a given tile size.
///
/// Returns true if NEON is recommended (better for this case).
/// Factors considered:
///   - SVE vector length (NEON optimal on SVE-128)
///   - Tile size (NEON optimal for small fixed tiles)
///   - Edge handling (NEON avoids predicate overhead)
inline bool prefer_neon_over_sve(int M, int N, int K) {
    // SVE-128: NEON always optimal (same register view, better latency)
    if (is_sve_128bit()) return true;

    // Wide SVE: use VLA for large N, NEON for small tiles
    if (M <= 8 && N <= 16 && K <= 64) return true;  // Small tile: NEON

    // Large regular shapes: prefer SVE VLA
    return false;
}

}  // namespace dnnopt

#else  // !__ARM_FEATURE_SVE

// No SVE support: provide empty stubs for NEON-only builds
namespace dnnopt {

// Stubs that just return NEON vectors unchanged
inline float32x4_t neon_to_sve_stub_f32(float32x4_t neon) { return neon; }
inline float32x4_t sve_to_neon_stub_f32(float32x4_t sve) { return sve; }

// Always prefer NEON when SVE unavailable
inline bool prefer_neon_over_sve(int, int, int) { return true; }

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE