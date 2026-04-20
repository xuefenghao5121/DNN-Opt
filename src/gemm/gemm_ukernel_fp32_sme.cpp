/// @file gemm_ukernel_fp32_sme.cpp
/// SME FP32 GEMM microkernel using FMOPA (outer product accumulate).
///
/// COMPILE-ONLY: SME is not available on current hardware (Neoverse N2).
/// This kernel auto-activates on SME-capable CPUs (Neoverse V3+).
///
/// SME programming model:
///   1. SMSTART SM: enter streaming mode, enable ZA tile storage
///   2. ZERO {za}: clear ZA accumulator tile
///   3. FMOPA za0.s, p0/m, p0/m, z0.s, z1.s: outer product accumulate
///      - For SVL=512: 16x16 FP32 outer product in ONE instruction
///      - For SVL=256: 8x8 FP32 outer product
///   4. MOVA: extract rows from ZA to vector registers
///   5. Scale with alpha, add beta*C, store
///   6. SMSTOP SM: exit streaming mode
///
/// GCC 10.2 does not support -march=...+sme, so we use inline assembly
/// with .arch_extension sme assembler directive.
///
/// NOTE: This file compiles but the kernel cannot run without SME hardware.

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"

// SME requires both compile-time opt-in AND runtime hwcap detection.
// We use DNNOPT_ENABLE_SME cmake option to gate compilation.
#if defined(DNNOPT_HAS_SME)

#include <arm_neon.h>
#include <cstring>
#include <algorithm>

namespace dnnopt {

// ============================================================
// SME FP32 microkernel via inline assembly
// ============================================================

/// Query streaming SVE vector length (SVL) in bytes.
/// This must be called from streaming mode or use rdsvl instruction.
/// Returns SVL in bytes (32 for SVL-256, 64 for SVL-512).
static inline uint64_t sme_svl_bytes() {
    uint64_t svl;
    // RDSVL reads streaming vector length regardless of mode
    asm volatile(
        ".arch_extension sme\n\t"
        "rdsvl %0, #1\n\t"
        : "=r"(svl)
    );
    return svl;
}

/// SME FMOPA-based FP32 GEMM microkernel.
/// Tile size = SVL_words x SVL_words where SVL_words = SVL_bytes / 4.
/// On SVL=256: 8x8 tile; SVL=512: 16x16 tile.
///
/// packed_A: Mr-wide column panels (Mr floats per K iteration, contiguous)
/// packed_B: Nr-wide row panels (Nr floats per K iteration, contiguous)
///
/// Both Mr and Nr equal SVL_words for SME.
static void gemm_ukernel_fp32_sme(int K,
                                    const float* packed_A,
                                    const float* packed_B,
                                    float* C, int ldc,
                                    float alpha, float beta) {
    const uint64_t svl_bytes = sme_svl_bytes();
    const int svl_words = (int)(svl_bytes / 4);  // FP32 elements per SVE vector
    // Mr = Nr = svl_words

    // Temporary buffer to extract ZA rows into (max SVL-512 = 16 floats per row)
    // We allocate for 16 rows × 16 floats maximum
    float za_rows[16 * 16];

    // Enter streaming mode and zero ZA tile
    asm volatile(
        ".arch_extension sme\n\t"
        "smstart sm\n\t"
        "zero {za}\n\t"
        ::: "memory"
    );

    // K-loop: each iteration does one rank-1 outer product update
    // FMOPA za0.s, p0/m, p0/m, z0.s, z1.s
    //   za0 += z0 (column) * z1 (row)^T
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

    // Extract ZA rows into za_rows buffer.
    // MOVA extracts row w12 from za0h.s into a vector register.
    // Since the number of rows is runtime-dependent (svl_words),
    // we iterate using w12 as the tile slice index.
    //
    // For each row i in [0, svl_words):
    //   mov w12, #i
    //   mova z0.s, p0/m, za0h.s[w12, #0]
    //   st1w {z0.s}, p0, [za_rows + i * svl_words * 4]
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

    // Exit streaming mode
    asm volatile(
        ".arch_extension sme\n\t"
        "smstop sm\n\t"
        ::: "memory"
    );

    // Epilogue: C = alpha * za_rows + beta * C (now in normal mode, use NEON)
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);

    for (int i = 0; i < svl_words; ++i) {
        float* Cr = C + i * ldc;
        const float* src = za_rows + i * svl_words;
        int j = 0;

        if (beta == 0.0f) {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                vst1q_f32(Cr + j, vmulq_f32(alpha_v, acc));
            }
            for (; j < svl_words; ++j) {
                Cr[j] = alpha * src[j];
            }
        } else {
            for (; j + 3 < svl_words; j += 4) {
                float32x4_t acc = vld1q_f32(src + j);
                float32x4_t c_old = vld1q_f32(Cr + j);
                vst1q_f32(Cr + j, vfmaq_f32(vmulq_f32(beta_v, c_old), alpha_v, acc));
            }
            for (; j < svl_words; ++j) {
                Cr[j] = alpha * src[j] + beta * Cr[j];
            }
        }
    }
}

// ============================================================
// Registry wrappers
// ============================================================

static void ukernel_fp32_sme_wrap(int K, const void* packed_A,
                                    const void* packed_B,
                                    float* C, int ldc, float alpha,
                                    float beta, float /*extra*/) {
    gemm_ukernel_fp32_sme(K,
                            static_cast<const float*>(packed_A),
                            static_cast<const float*>(packed_B),
                            C, ldc, alpha, beta);
}

// Reuse FP32 packing (same layout works for SME outer product)
void pack_a_fp32(int m_len, int k_len, const float* A, int lda, float* packed_A);
void pack_b_fp32(int k_len, int n_len, const float* B, int ldb, float* packed_B);

static void pack_a_fp32_sme_wrap(int m_len, int k_len, const float* A, int lda,
                                   void* packed_A, int /*Mr*/, float* /*scale_out*/) {
    pack_a_fp32(m_len, k_len, A, lda, static_cast<float*>(packed_A));
}

static void pack_b_fp32_sme_wrap(int k_len, int n_len, const float* B, int ldb,
                                   void* packed_B, int /*Nr*/, float* /*scale_out*/) {
    pack_b_fp32(k_len, n_len, B, ldb, static_cast<float*>(packed_B));
}

// SME FP32: tile size is VLA (SVL_words × SVL_words)
// On SVL=512: 16×16 tile; SVL=256: 8×8 tile
// Priority 300: highest, always preferred when SME is available
static const GemmMicrokernelDesc sme_fp32_desc = {
    "sme_fp32_VLxVL",
    GemmDataType::kFP32,
    kSME,                 // required_hwcaps
    0,                    // Mr (VLA, computed at dispatch)
    0,                    // Nr (VLA, computed at dispatch)
    1,                    // Kgroup
    true,                 // nr_is_vla (both Mr and Nr scale)
    300,                  // priority: highest
    sizeof(float),
    sizeof(float),
    0,                    // min_sve_bits (SME has its own SVL)
    ukernel_fp32_sme_wrap,
    pack_a_fp32_sme_wrap,
    pack_b_fp32_sme_wrap,
};

static RegisterKernel reg_sme_fp32(sme_fp32_desc);

}  // namespace dnnopt

#endif  // DNNOPT_HAS_SME
