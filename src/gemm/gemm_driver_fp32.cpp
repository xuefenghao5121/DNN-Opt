/// @file gemm_driver_fp32.cpp
/// BLIS-style FP32 GEMM driver with cache blocking and packing.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>

namespace dnnopt {

// Packing routines (defined in gemm_pack_fp32.cpp)
void pack_a_fp32(int m_len, int k_len, const float* A, int lda, float* packed_A);
void pack_b_fp32(int k_len, int n_len, const float* B, int ldb, float* packed_B);

// Microkernel (defined in gemm_ukernel_fp32_neon.cpp)
#ifdef __ARM_NEON
void gemm_ukernel_fp32_8x12(int K, const float* packed_A, const float* packed_B,
                             float* C, int ldc, float alpha, float beta);
#endif

/// BLIS 5-loop FP32 GEMM driver.
void gemm_driver_fp32(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc) {
#ifdef __ARM_NEON
    auto bp = get_gemm_blocking_params();
    const int Mr = bp.Mr;
    const int Nr = bp.Nr;
    const int Mc = bp.Mc;
    const int Nc = bp.Nc;
    const int Kc = bp.Kc;

    // Allocate packing buffers (once for the entire GEMM)
    // Sizes account for zero-padding to full Mr/Nr panels
    int m_panels_max = (Mc + Mr - 1) / Mr;
    int n_panels_max = (Nc + Nr - 1) / Nr;
    size_t packed_a_size = (size_t)m_panels_max * Mr * Kc;
    size_t packed_b_size = (size_t)n_panels_max * Nr * Kc;
    auto packed_A = aligned_array<float>(packed_a_size);
    auto packed_B = aligned_array<float>(packed_b_size);

    // Edge buffer for partial tiles
    float edge_buf[kGemmMrFp32 * kGemmNrFp32];

    // Loop 1: N dimension (L3 blocking)
    for (int jc = 0; jc < N; jc += Nc) {
        int nc = std::min(Nc, N - jc);

        // Loop 2: K dimension (L2 blocking)
        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);

            // beta handling: only apply original beta on the first K-pass;
            // subsequent passes accumulate (beta=1.0)
            float beta_eff = (pc == 0) ? beta : 1.0f;
            // alpha: apply on every K-pass so each partial sum is scaled
            // uniformly. Combined with beta_eff this gives:
            //   First pass:  C = alpha * partial_0 + beta * C_old
            //   Later passes: C = alpha * partial_i + 1.0 * C_accumulated
            float alpha_eff = alpha;

            // Pack B panel: B[pc:pc+kc, jc:jc+nc]
            pack_b_fp32(kc, nc, &B[pc * ldb + jc], ldb, packed_B.get());

            // Loop 3: M dimension (L2 blocking)
            for (int ic = 0; ic < M; ic += Mc) {
                int mc = std::min(Mc, M - ic);

                // Pack A block: A[ic:ic+mc, pc:pc+kc]
                pack_a_fp32(mc, kc, &A[ic * lda + pc], lda, packed_A.get());

                // Number of Mr-panels and Nr-panels
                int m_panels = (mc + Mr - 1) / Mr;
                int n_panels = (nc + Nr - 1) / Nr;

                // Loop 4+5: micro-tiles
                for (int jr = 0; jr < n_panels; jr++) {
                    int n_start = jr * Nr;
                    int n_rem = std::min(Nr, nc - n_start);
                    const float* B_panel = packed_B.get() + jr * kc * Nr;

                    for (int ir = 0; ir < m_panels; ir++) {
                        int m_start = ir * Mr;
                        int m_rem = std::min(Mr, mc - m_start);
                        const float* A_panel = packed_A.get() + ir * kc * Mr;

                        float* C_ptr = &C[(ic + m_start) * ldc + jc + n_start];

                        if (m_rem == Mr && n_rem == Nr) {
                            // Full tile: direct write to C
                            gemm_ukernel_fp32_8x12(kc, A_panel, B_panel,
                                                   C_ptr, ldc, alpha_eff, beta_eff);
                        } else {
                            // Edge tile: compute into buffer, copy valid portion
                            memset(edge_buf, 0, sizeof(edge_buf));
                            // Load existing C values
                            if (beta_eff != 0.0f) {
                                for (int i = 0; i < m_rem; i++)
                                    memcpy(&edge_buf[i * Nr], &C_ptr[i * ldc],
                                           n_rem * sizeof(float));
                            }
                            gemm_ukernel_fp32_8x12(kc, A_panel, B_panel,
                                                   edge_buf, Nr, alpha_eff, beta_eff);
                            // Copy back valid portion
                            for (int i = 0; i < m_rem; i++)
                                memcpy(&C_ptr[i * ldc], &edge_buf[i * Nr],
                                       n_rem * sizeof(float));
                        }
                    }
                }
            }
        }
    }
#else
    // No NEON: should not reach here (caller checks)
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
#endif
}

}  // namespace dnnopt
