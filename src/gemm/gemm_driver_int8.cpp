/// @file gemm_driver_int8.cpp
/// BLIS-style INT8 GEMM driver with cache blocking, packing, and quantization.
/// Input/output are FP32; internal computation uses INT8 SMMLA.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// INT8 packing (defined in gemm_pack_int8.cpp)
void pack_a_int8(int m_len, int k_len, const float* A, int lda,
                 int8_t* packed_A, float* scale_A);
void pack_b_int8(int k_len, int n_len, const float* B, int ldb,
                 int8_t* packed_B, float* scale_B);

// INT8 microkernel (defined in gemm_ukernel_int8_neon.cpp)
void gemm_ukernel_int8_8x8(int K, const int8_t* packed_A, const int8_t* packed_B,
                            float* C, int ldc, float alpha, float beta,
                            float dequant_scale);

void gemm_driver_int8(int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc) {
    constexpr int Mr = kGemmMrInt8;  // 8
    constexpr int Nr = kGemmNrInt8;  // 8
    constexpr int Kgroup = 8;

    auto bp = get_gemm_blocking_params();
    int Mc = bp.Mc;
    int Nc = bp.Nc;
    // INT8 is 4x denser than FP32, allow larger Kc
    int Kc = std::min(bp.Kc * 4, K);
    Kc = (Kc + Kgroup - 1) / Kgroup * Kgroup;

    // Packing buffers (INT8)
    // Per Mr-panel per K-group: 4 row-pairs × 16 INT8 = 64 bytes
    int m_panels_max = (Mc + Mr - 1) / Mr;
    int k_groups_max = (Kc + Kgroup - 1) / Kgroup;
    size_t packed_a_size = (size_t)m_panels_max * k_groups_max * 64;

    int n_panels_max = (Nc + Nr - 1) / Nr;
    size_t packed_b_size = (size_t)n_panels_max * k_groups_max * 64;

    auto packed_A = aligned_array<int8_t>(packed_a_size);
    auto packed_B = aligned_array<int8_t>(packed_b_size);

    // Edge buffer for partial tiles
    float edge_buf[Mr * Nr];

    // BLIS 5-loop
    for (int jc = 0; jc < N; jc += Nc) {
        int nc = std::min(Nc, N - jc);

        for (int pc = 0; pc < K; pc += Kc) {
            int kc = std::min(Kc, K - pc);
            int kc_padded = (kc + Kgroup - 1) / Kgroup * Kgroup;

            float beta_eff = (pc == 0) ? beta : 1.0f;
            float alpha_eff = alpha;

            // Pack B panel with quantization
            float scale_B;
            pack_b_int8(kc, nc, &B[pc * ldb + jc], ldb, packed_B.get(), &scale_B);

            for (int ic = 0; ic < M; ic += Mc) {
                int mc = std::min(Mc, M - ic);

                // Pack A block with quantization
                float scale_A;
                pack_a_int8(mc, kc, &A[ic * lda + pc], lda, packed_A.get(), &scale_A);

                float dequant_scale = scale_A * scale_B;

                int m_panels = (mc + Mr - 1) / Mr;
                int n_panels = (nc + Nr - 1) / Nr;

                // Packed panel sizes in INT8 elements
                // Per Mr-panel: (kc_padded / 8) * 64 INT8
                size_t a_panel_stride = (size_t)(kc_padded / Kgroup) * 64;
                size_t b_panel_stride = (size_t)(kc_padded / Kgroup) * 64;

                for (int jr = 0; jr < n_panels; jr++) {
                    int n_start = jr * Nr;
                    int n_rem = std::min(Nr, nc - n_start);
                    const int8_t* B_panel = packed_B.get() + jr * b_panel_stride;

                    for (int ir = 0; ir < m_panels; ir++) {
                        int m_start = ir * Mr;
                        int m_rem = std::min(Mr, mc - m_start);
                        const int8_t* A_panel = packed_A.get() + ir * a_panel_stride;

                        float* C_ptr = &C[(ic + m_start) * ldc + jc + n_start];

                        if (m_rem == Mr && n_rem == Nr) {
                            gemm_ukernel_int8_8x8(kc_padded, A_panel, B_panel,
                                                  C_ptr, ldc, alpha_eff, beta_eff,
                                                  dequant_scale);
                        } else {
                            // Edge tile
                            memset(edge_buf, 0, sizeof(edge_buf));
                            if (beta_eff != 0.0f) {
                                for (int i = 0; i < m_rem; i++)
                                    memcpy(&edge_buf[i * Nr], &C_ptr[i * ldc],
                                           n_rem * sizeof(float));
                            }
                            gemm_ukernel_int8_8x8(kc_padded, A_panel, B_panel,
                                                  edge_buf, Nr, alpha_eff, beta_eff,
                                                  dequant_scale);
                            for (int i = 0; i < m_rem; i++)
                                memcpy(&C_ptr[i * ldc], &edge_buf[i * Nr],
                                       n_rem * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt
