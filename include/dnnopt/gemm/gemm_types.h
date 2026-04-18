#pragma once
/// @file gemm_types.h
/// Shared types for GEMM operations.

#include <cstdint>

namespace dnnopt {

/// bfloat16 storage type: 16-bit with 7-bit mantissa.
/// Binary-compatible with oneDNN's bfloat16_t and arm_neon.h's internal type.
struct bfloat16_t {
    uint16_t raw_bits;

    bfloat16_t() = default;
    constexpr explicit bfloat16_t(uint16_t raw, bool) : raw_bits(raw) {}

    /// Construct from float (round-to-nearest-even).
    explicit bfloat16_t(float f) {
        union { float f32; uint32_t u32; } u;
        u.f32 = f;
        // Round-to-nearest-even: add 0x7FFF + (bit 15 of mantissa)
        uint32_t round = 0x7FFF + ((u.u32 >> 16) & 1);
        raw_bits = static_cast<uint16_t>((u.u32 + round) >> 16);
    }

    /// Convert to float.
    operator float() const {
        union { float f32; uint32_t u32; } u;
        u.u32 = static_cast<uint32_t>(raw_bits) << 16;
        return u.f32;
    }

    bfloat16_t& operator=(float f) {
        *this = bfloat16_t(f);
        return *this;
    }
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

// ============================================================
// FP8 (Floating Point 8-bit) Types
// ============================================================
// ARMv9-A introduces FP8 for AI inference acceleration.
// Two formats: E4M3FN (precision) and E5M2 (range).
// Requires __ARM_FEATURE_FP8 (GCC 13+, Clang 16+, ARMv9 hardware).

/// FP8 E4M3FN format: sign(1) + exponent(4) + mantissa(3)
/// Range: [-448, 448], no inf/nan (special encoding for max values)
/// Use: Activations, weights where precision matters
struct fp8_e4m3_t {
    uint8_t raw_bits;

    fp8_e4m3_t() = default;
    constexpr explicit fp8_e4m3_t(uint8_t raw, bool) : raw_bits(raw) {}

    /// Construct from float (round-to-nearest-even).
    /// Note: No inf/nan support; saturates to max finite value.
    explicit fp8_e4m3_t(float f) {
        // FP8 E4M3: bias=8, max_exp=7 (normal), min_exp=-6 (subnormal)
        // Max finite: 448 (0x7E), Min positive: 2^-6 ≈ 0.015625
        union { float f32; uint32_t u32; } u;
        u.f32 = f;
        uint32_t sign = (u.u32 >> 31) & 1;
        int32_t exp = ((u.u32 >> 23) & 0xFF) - 127 + 8;  // FP32 bias 127, FP8 bias 8
        uint32_t mant = (u.u32 >> 20) & 0x7;  // Take top 3 bits of mantissa

        // Clamp exponent to valid range [-6, 7]
        if (exp > 7) { exp = 7; mant = 0x7; }  // Max finite (saturate)
        else if (exp < -6) { exp = 0; mant = 0; }  // Zero (flush to zero)

        // Encode: sign(1) + exp(4) + mant(3)
        if (exp > 0) {
            raw_bits = (sign << 7) | ((exp & 0xF) << 3) | mant;
        } else {
            // Subnormal: exp field = 0, mantissa has leading zeros
            int shift = -exp;
            if (shift >= 8) {
                raw_bits = 0;  // Flush to zero
            } else {
                mant = (mant >> shift) | (0x8 >> shift);  // Add implicit leading bit
                raw_bits = (sign << 7) | mant;
            }
        }
    }

    /// Convert to float.
    operator float() const {
        uint32_t sign = (raw_bits >> 7) & 1;
        uint32_t exp_field = (raw_bits >> 3) & 0xF;
        uint32_t mant = raw_bits & 0x7;

        union { float f32; uint32_t u32; } u;
        if (exp_field == 0) {
            // Subnormal or zero
            if (mant == 0) {
                u.u32 = sign << 31;  // +/- zero
            } else {
                // Subnormal: exp = 2^-6 * mant/8
                int actual_exp = -6;
                while (!(mant & 0x8)) { mant <<= 1; actual_exp--; }
                mant &= 0x7;
                u.u32 = (sign << 31) | ((actual_exp + 127) << 23) | (mant << 20);
            }
        } else if (exp_field == 0xF && mant == 0x7) {
            // Max finite: 448 * 2^sign
            u.f32 = sign ? -448.0f : 448.0f;
        } else {
            // Normal: exp_field - bias (8)
            int actual_exp = exp_field - 8 + 127;
            u.u32 = (sign << 31) | (actual_exp << 23) | (mant << 20);
        }
        return u.f32;
    }

    fp8_e4m3_t& operator=(float f) {
        *this = fp8_e4m3_t(f);
        return *this;
    }
};

static_assert(sizeof(fp8_e4m3_t) == 1, "fp8_e4m3_t must be 1 byte");

/// FP8 E5M2 format: sign(1) + exponent(5) + mantissa(2)
/// Range: [-57344, 57344], supports inf/nan
/// Use: Gradients, intermediate results where range matters
struct fp8_e5m2_t {
    uint8_t raw_bits;

    fp8_e5m2_t() = default;
    constexpr explicit fp8_e5m2_t(uint8_t raw, bool) : raw_bits(raw) {}

    /// Construct from float (round-to-nearest-even).
    explicit fp8_e5m2_t(float f) {
        union { float f32; uint32_t u32; } u;
        u.f32 = f;
        uint32_t sign = (u.u32 >> 31) & 1;
        int32_t exp = ((u.u32 >> 23) & 0xFF) - 127 + 16;  // FP32 bias 127, FP8 bias 16
        uint32_t mant = (u.u32 >> 21) & 0x3;  // Take top 2 bits of mantissa

        // Clamp: max_exp=30 (normal), min_exp=-14 (subnormal)
        // Inf: exp=31, mant=0; Nan: exp=31, mant!=0
        if (exp > 30) {
            // Overflow: inf (no saturation, IEEE style)
            raw_bits = (sign << 7) | (0x1F << 2) | 0;
        } else if (exp < -14) {
            // Underflow: flush to zero
            raw_bits = sign << 7;
        } else if (exp > 0) {
            raw_bits = (sign << 7) | ((exp & 0x1F) << 2) | mant;
        } else {
            // Subnormal handling
            int shift = -exp;
            if (shift >= 4) {
                raw_bits = sign << 7;  // Flush to zero
            } else {
                mant = (mant >> shift) | (0x4 >> shift);
                raw_bits = (sign << 7) | mant;
            }
        }
    }

    /// Convert to float.
    operator float() const {
        uint32_t sign = (raw_bits >> 7) & 1;
        uint32_t exp_field = (raw_bits >> 2) & 0x1F;
        uint32_t mant = raw_bits & 0x3;

        union { float f32; uint32_t u32; } u;
        if (exp_field == 0) {
            // Subnormal or zero
            if (mant == 0) {
                u.u32 = sign << 31;
            } else {
                int actual_exp = -14;
                while (!(mant & 0x4)) { mant <<= 1; actual_exp--; }
                mant &= 0x3;
                u.u32 = (sign << 31) | ((actual_exp + 127) << 23) | (mant << 21);
            }
        } else if (exp_field == 0x1F) {
            // Inf or Nan
            if (mant == 0) {
                u.u32 = (sign << 31) | (0xFF << 23);  // +/- Inf
            } else {
                u.u32 = (sign << 31) | (0xFF << 23) | (mant << 21);  // Nan
            }
        } else {
            int actual_exp = exp_field - 16 + 127;
            u.u32 = (sign << 31) | (actual_exp << 23) | (mant << 21);
        }
        return u.f32;
    }

    fp8_e5m2_t& operator=(float f) {
        *this = fp8_e5m2_t(f);
        return *this;
    }
};

static_assert(sizeof(fp8_e5m2_t) == 1, "fp8_e5m2_t must be 1 byte");

/// Algorithm selection for GEMM dispatch.
enum class GemmAlgo {
    kAuto,        // Automatic: pick best for current hardware + shape
    kNaive,       // Scalar reference (for testing only)
    kNeonFp32,    // NEON 8x12 FP32 microkernel + BLIS blocking
    kBf16Bfmmla,  // BF16 BFMMLA microkernel
    kInt8Smmla,   // INT8 SMMLA microkernel
    kInt8Sdot,    // INT8 SDOT microkernel (future)
    kSveFp32,     // SVE FP32 microkernel
    kSveBf16,     // SVE BF16 microkernel
    kSveInt8,     // SVE INT8 microkernel
    kSmeFp32,     // SME FP32 microkernel (future)
    // FP8 algorithms (ARMv9-A, requires __ARM_FEATURE_FP8)
    kFp8E4m3Fdot, // FP8 E4M3 FDOT microkernel
    kFp8E5m2Fdot, // FP8 E5M2 FDOT microkernel
    kSveFp8E4m3,  // SVE FP8 E4M3 microkernel
    kSveFp8E5m2,  // SVE FP8 E5M2 microkernel
};

/// Cache blocking parameters for BLIS-style GEMM.
struct GemmBlockingParams {
    int Mr;   // Microkernel row tile
    int Nr;   // Microkernel column tile
    int Mc;   // L2 blocking on M
    int Nc;   // L3 blocking on N
    int Kc;   // L2 blocking on K
};

}  // namespace dnnopt
