/// @file gemm_ukernel_int8_sme.cpp
/// SME INT8 GEMM microkernel using SMOPA (signed INT8 outer product accumulate).
///
/// COMPILE-ONLY: SME is not available on current hardware (Neoverse N2).
/// This kernel auto-activates on SME-capable CPUs (Neoverse V3+).
///
/// SME INT8 programming model (SMOPA):
///   1. SMSTART SM: enter streaming mode, enable ZA tile
///   2. ZERO {za}: clear ZA accumulator (INT32 tile)
///   3. SMOPA za0.s, p0/m, p0/m, z0.b, z1.b: outer product accumulate
///      - z0.b: SVL_bytes INT8 values (row vector from A)
///      - z1.b: SVL_bytes INT8 values (col vector from B)
///      - za0.s: SVL_words × SVL_words INT32 accumulators
///      - Each SMOPA: outer product of 2 SVL_bytes vectors → SVL_words × SVL_words INT32
///   4. MOVA: extract INT32 rows from ZA tile
///   5. Convert INT32 → FP32, apply dequant_scale * alpha + beta * C
///   6. SMSTOP SM: exit streaming mode
///
/// SMOPA advantage vs FMOPA:
///   - INT8 input: 4x K throughput (32 INT8 vs 8 FP32 per vector)
///   - INT32 accumulate: no overflow concerns
///   - Tile size: SVL_words × SVL_words (e.g., 8×8 for SVL-256, 16×16 for SVL-512)

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

#if defined(DNNOPT_HAS_SME)

#include <arm_neon.h>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace dnnopt {

// ============================================================
// SME SMOPA INT8 microkernel (true INT8 path)
// ============================================================

/// Query streaming SVE vector length in bytes.
/// For INT8: SVL_bytes = number of INT8 elements per vector.
/// For INT32: SVL_words = SVL_bytes / 4 = number of INT32 accumulators per row/col.
static inline uint64_t sme_svl_bytes() {
    uint64_t svl;
    asm volatile(
        ".arch_extension sme\n\t"
        "rdsvl %0, #1\n\t"
        : "=r"(svl)
    );
    return svl;
}

/// SME INT8 GEMM microkernel using SMOPA (true INT8 path).
///
/// packed_A: INT8 panel, SVL_bytes × K (row-major)
/// packed_B: INT8 panel, SVL_bytes × K (row-major)
/// C: FP32 output matrix
///
/// dequant_scale: scale_A * scale_B from INT8 quantization
///                applied to convert INT32 accumulate → FP32 output
static void gemm_ukernel_int8_sme_smopa(int K,
                                         const int8_t* packed_A,
                                         const int8_t* packed_B,
                                         float* C, int ldc,
                                         float alpha, float beta,
                                         float dequant_scale) {
    const uint64_t svl_bytes = sme_svl_bytes();
    const int svl_int8 = (int)svl_bytes;          // INT8 elements per vector
    const int svl_int32 = (int)(svl_bytes / 4);  // INT32 elements per vector (tile dimension)

    // Temporary buffer for ZA tile extraction (INT32 accumulators)
    // Max size: 16×16 INT32 for SVL-512
    int32_t za_tile[16 * 16];

    // Effective scaling: alpha * dequant_scale converts INT32 → FP32
    const float effective_scale = alpha * dequant_scale;

    // Enter streaming mode and zero ZA tile
    asm volatile(
        ".arch_extension sme\n\t"
        "smstart sm\n\t"
        "zero {za}\n\t"
        ::: "memory"
    );

    // K-loop: each iteration does one outer product update
    // SMOPA za0.s, p0/m, p0/m, z0.b, z1.b
    //   za0 += z0 (column) ⊗ z1 (row)  (INT8 outer product → INT32 accumulate)
    for (int k = 0; k < K; ++k) {
        const int8_t* a_ptr = packed_A + k * svl_int8;
        const int8_t* b_ptr = packed_B + k * svl_int8;

        asm volatile(
            ".arch_extension sme\n\t"
            "ptrue p0.b\n\t"              // All INT8 lanes active
            "ld1b {z0.b}, p0/z, [%[a]]\n\t"  // Load SVL_bytes INT8 from A
            "ld1b {z1.b}, p0/z, [%[b]]\n\t"  // Load SVL_bytes INT8 from B
            "smopa za0.s, p0/m, p0/m, z0.b, z1.b\n\t"  // INT8 outer product → INT32 ZA
            :
            : [a] "r"(a_ptr), [b] "r"(b_ptr)
            : "z0", "z1", "p0", "memory"
        );
    }

    // Extract ZA tile rows into za_tile buffer
    // MOVA z0.s, p0/m, za0h.s[w12, #0] extracts row w12 from za0
    for (int i = 0; i < svl_int32; ++i) {
        int32_t* dst = za_tile + i * svl_int32;
        asm volatile(
            ".arch_extension sme\n\t"
            "ptrue p0.s\n\t"
            "mov w12, %w[idx]\n\t"
            "mova z0.s, p0/m, za0h.s[w12, #0]\n\t"
            "st1w {z0.s}, p0, [%[dst]]\n\t"
            :
            : [idx] "r"(i), [dst] "r"(dst)
            : "z0", "p0", "w12", "memory"
        );
    }

    // Exit streaming mode
    asm volatile(
        ".arch_extension sme\n\t"
        "smstop sm\n\t"
        ::: "memory"
    );

    // Epilogue: convert INT32 → FP32, apply scaling, store to C
    // C[i,j] = effective_scale * za_tile[i,j] + beta * C[i,j]
    for (int i = 0; i < svl_int32; ++i) {
        float* Cr = C + i * ldc;
        const int32_t* src = za_tile + i * svl_int32;

        int j = 0;
        // NEON vector path (process 4 INT32 at once)
        float32x4_t scale_v = vdupq_n_f32(effective_scale);
        float32x4_t beta_v = vdupq_n_f32(beta);

        if (beta == 0.0f) {
            for (; j + 3 < svl_int32; j += 4) {
                // Convert INT32 → FP32, apply scale
                int32x4_t acc_i32 = vld1q_s32(src + j);
                float32x4_t acc_f32 = vcvtq_f32_s32(acc_i32);
                vst1q_f32(Cr + j, vmulq_f32(scale_v, acc_f32));
            }
            for (; j < svl_int32; ++j) {
                Cr[j] = effective_scale * (float)src[j];
            }
        } else {
            for (; j + 3 < svl_int32; j += 4) {
                int32x4_t acc_i32 = vld1q_s32(src + j);
                float32x4_t acc_f32 = vcvtq_f32_s32(acc_i32);
                float32x4_t c_old = vld1q_f32(Cr + j);
                // C = scale * acc + beta * C_old
                float32x4_t result = vfmaq_f32(vmulq_f32(beta_v, c_old), scale_v, acc_f32);
                vst1q_f32(Cr + j, result);
            }
            for (; j < svl_int32; ++j) {
                Cr[j] = effective_scale * (float)src[j] + beta * Cr[j];
            }
        }
    }
}

// ============================================================
// Fallback: FMOPA path with FP32-packed data (compatibility)
// ============================================================

/// SME INT8 GEMM fallback using FMOPA with FP32 inputs.
/// Used when INT8 packing is not available or for compatibility.
static void gemm_ukernel_int8_sme_fmopa(int K,
                                         const float* packed_A,
                                         const float* packed_B,
                                         float* C, int ldc,
                                         float alpha, float beta,
                                         float dequant_scale) {
    const uint64_t svl_bytes = sme_svl_bytes();
    const int svl_words = (int)(svl_bytes / 4);

    float za_rows[16 * 16];
    float effective_alpha = alpha * dequant_scale;

    asm volatile(
        ".arch_extension sme\n\t"
        "smstart sm\n\t"
        "zero {za}\n\t"
        ::: "memory"
    );

    for (int k = 0; k < K; ++k) {
        const float* a_ptr = packed_A + k * svl_words;
        const float* b_ptr = packed_B + k * svl_words;

        asm volatile(
            ".arch_extension sme\n\t"
            "ptrue p0.s\n\t"
            "ld1w {z0.s}, p0/z, [%[a]]\n\t"
            "ld1w {z1.s}, p0/z, [%[b]]\n\t"
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
            :
            : [a] "r"(a_ptr), [b] "r"(b_ptr)
            : "z0", "z1", "p0", "memory"
        );
    }

    for (int i = 0; i < svl_words; ++i) {
        float* dst = za_rows + i * svl_words;
        asm volatile(
            ".arch_extension sme\n\t"
            "ptrue p0.s\n\t"
            "mov w12, %w[idx]\n\t"
            "mova z0.s, p0/m, za0h.s[w12, #0]\n\t"
            "st1w {z0.s}, p0, [%[dst]]\n\t"
            :
            : [idx] "r"(i), [dst] "r"(dst)
            : "z0", "p0", "w12", "memory"
        );
    }

    asm volatile(
        ".arch_extension sme\n\t"
        "smstop sm\n\t"
        ::: "memory"
    );

    float32x4_t alpha_v = vdupq_n_f32(effective_alpha);
    float32x4_t beta_v = vdupq_n_f32(beta);

    for (int i = 0; i < svl_words; ++i) {
        float* Cr = C + i * ldc;
        const float* src = za_rows + i * svl_words;
        int j = 0;

        if (beta == 0.0f) {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                vst1q_f32(Cr + j, vmulq_f32(alpha_v, acc));
            }
            for (; j < svl_words; ++j) Cr[j] = effective_alpha * src[j];
        } else {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                float32x4_t c_old = vld1q_f32(Cr + j);
                vst1q_f32(Cr + j, vfmaq_f32(vmulq_f32(beta_v, c_old), alpha_v, acc));
            }
            for (; j < svl_words; ++j)
                Cr[j] = effective_alpha * src[j] + beta * Cr[j];
        }
    }
}

// ============================================================
// INT8 packing for SMOPA
// ============================================================

/// Pack A matrix for SMOPA: convert FP32 → INT8 with scaling.
/// Layout: SVL_int8 × K (row-major, contiguous K)
/// Each row is SVL_int8 INT8 values, packed for SMOPA outer product.
static void pack_a_int8_for_smopa(int m_len, int k_len,
                                   const float* A, int lda,
                                   int8_t* packed_A,
                                   float* scale_out) {
    // Compute scale from max absolute value
    float max_val = 0.0f;
    for (int i = 0; i < m_len; ++i) {
        for (int k = 0; k < k_len; ++k) {
            float v = std::fabs(A[i * lda + k]);
            if (v > max_val) max_val = v;
        }
    }
    float scale = (max_val > 0.0f) ? 127.0f / max_val : 1.0f;
    *scale_out = scale;

    // Pack: FP32 → INT8 with scaling
    for (int i = 0; i < m_len; ++i) {
        int8_t* row = packed_A + i * k_len;
        for (int k = 0; k < k_len; ++k) {
            float v = A[i * lda + k] * scale;
            // Clamp to INT8 range
            int32_t iv = (int32_t)std::round(v);
            iv = std::max(-128, std::min(127, iv));
            row[k] = (int8_t)iv;
        }
    }
}

/// Pack B matrix for SMOPA: convert FP32 → INT8 with scaling.
/// Layout: SVL_int8 × K (row-major, contiguous K)
static void pack_b_int8_for_smopa(int k_len, int n_len,
                                   const float* B, int ldb,
                                   int8_t* packed_B,
                                   float* scale_out) {
    float max_val = 0.0f;
    for (int k = 0; k < k_len; ++k) {
        for (int j = 0; j < n_len; ++j) {
            float v = std::fabs(B[k * ldb + j]);
            if (v > max_val) max_val = v;
        }
    }
    float scale = (max_val > 0.0f) ? 127.0f / max_val : 1.0f;
    *scale_out = scale;

    // Pack: transpose B so each row is contiguous K elements
    // SMOPA needs B rows (K × N → packed as N rows of K elements)
    for (int j = 0; j < n_len; ++j) {
        int8_t* row = packed_B + j * k_len;
        for (int k = 0; k < k_len; ++k) {
            float v = B[k * ldb + j] * scale;
            int32_t iv = (int32_t)std::round(v);
            iv = std::max(-128, std::min(127, iv));
            row[k] = (int8_t)iv;
        }
    }
}

// ============================================================
// Registry wrappers
// ============================================================

// SMOPA path wrapper (true INT8)
static void ukernel_int8_smopa_wrap(int K, const void* packed_A,
                                     const void* packed_B,
                                     float* C, int ldc, float alpha,
                                     float beta, float dequant_scale) {
    gemm_ukernel_int8_sme_smopa(K,
                                static_cast<const int8_t*>(packed_A),
                                static_cast<const int8_t*>(packed_B),
                                C, ldc, alpha, beta, dequant_scale);
}

// FMOPA fallback wrapper (FP32 packed)
static void ukernel_int8_fmopa_wrap(int K, const void* packed_A,
                                     const void* packed_B,
                                     float* C, int ldc, float alpha,
                                     float beta, float dequant_scale) {
    gemm_ukernel_int8_sme_fmopa(K,
                                static_cast<const float*>(packed_A),
                                static_cast<const float*>(packed_B),
                                C, ldc, alpha, beta, dequant_scale);
}

// INT8 packing wrapper for SMOPA
static void pack_a_int8_smopa_wrap(int m_len, int k_len, const float* A, int lda,
                                    void* packed_A, int /*Mr*/, float* scale_out) {
    pack_a_int8_for_smopa(m_len, k_len, A, lda,
                          static_cast<int8_t*>(packed_A), scale_out);
}

static void pack_b_int8_smopa_wrap(int k_len, int n_len, const float* B, int ldb,
                                    void* packed_B, int /*Nr*/, float* scale_out) {
    pack_b_int8_for_smopa(k_len, n_len, B, ldb,
                          static_cast<int8_t*>(packed_B), scale_out);
}

// FP32 packing wrapper for FMOPA fallback
void pack_a_fp32(int m_len, int k_len, const float* A, int lda, float* packed_A);
void pack_b_fp32(int k_len, int n_len, const float* B, int ldb, float* packed_B);

static void pack_a_int8_fmopa_wrap(int m_len, int k_len, const float* A, int lda,
                                    void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

static void pack_b_int8_fmopa_wrap(int k_len, int n_len, const float* B, int ldb,
                                    void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

// ============================================================
// Kernel registration
// ============================================================

// SME INT8 SMOPA kernel (true INT8 path, 4x K throughput)
// Priority 310: higher than FMOPA fallback
static const GemmMicrokernelDesc sme_int8_smopa_desc = {
    "sme_int8_smopa_VLxVL",
    GemmDataType::kINT8,
    kSME | kI8MM,           // requires SME + I8MM
    0,                      // Mr (VLA = SVL_int32)
    0,                      // Nr (VLA = SVL_int32)
    1,                      // Kgroup
    true,                   // nr_is_vla
    310,                    // priority: highest for true INT8
    sizeof(int8_t),         // packed as INT8
    sizeof(int8_t),
    0,                      // min_sve_bits
    ukernel_int8_smopa_wrap,
    pack_a_int8_smopa_wrap,
    pack_b_int8_smopa_wrap,
};

static RegisterKernel reg_sme_int8_smopa(sme_int8_smopa_desc);

// SME INT8 FMOPA fallback (FP32 packed, compatibility)
// Priority 300: used when INT8 packing not available
static const GemmMicrokernelDesc sme_int8_fmopa_desc = {
    "sme_int8_fmopa_VLxVL",
    GemmDataType::kINT8,
    kSME,                   // SME only (no I8MM needed for FMOPA)
    0,                      // Mr (VLA)
    0,                      // Nr (VLA)
    1,                      // Kgroup
    true,                   // nr_is_vla
    300,                    // priority
    sizeof(float),          // packed as FP32
    sizeof(float),
    0,
    ukernel_int8_fmopa_wrap,
    pack_a_int8_fmopa_wrap,
    pack_b_int8_fmopa_wrap,
};

static RegisterKernel reg_sme_int8_fmopa(sme_int8_fmopa_desc);

}  // namespace dnnopt

#endif  // DNNOPT_HAS_SME