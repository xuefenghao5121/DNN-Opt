/// @file gemm_driver_generic.cpp
/// Generic parameterized BLIS-style GEMM driver with 2D M×N parallelization.
///
/// Threading model:
///   - Decomposes thread team into mt × nt (M-threads × N-threads)
///   - Each thread handles a subset of M-blocks and N-blocks
///   - Shape-aware: tall-skinny → more mt, short-wide → more nt
///   - Falls back to M-only (mt=nthreads, nt=1) when 2D is not beneficial

#include "dnnopt/gemm/gemm_driver_generic.h"
#include "dnnopt/gemm/gemm_threading.h"
#include "dnnopt/gemm/gemm_thread_decomp.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dnnopt {

void gemm_driver_generic(int M, int N, int K,
                         float alpha, const float* A, int lda,
                         const float* B, int ldb,
                         float beta, float* C, int ldc,
                         const GemmDriverConfig& cfg) {
    const int Mr = cfg.Mr;
    const int Nr = cfg.Nr;
    const int Kgroup = cfg.Kgroup;
    const int Mc = cfg.Mc;
    const int Nc = cfg.Nc;
    int Kc = cfg.Kc;
    if (Kgroup > 1) Kc = (Kc / Kgroup) * Kgroup;
    if (Kc < Kgroup) Kc = Kgroup;

    const int a_elem = cfg.packed_a_elem_bytes;
    const int b_elem = cfg.packed_b_elem_bytes;

    // Panel stride lambdas
    auto a_panel_stride = [&](int kc_padded) -> size_t {
        if (cfg.dtype == GemmDataType::kFP32)
            return (size_t)Mr * kc_padded * a_elem;
        return (size_t)(kc_padded / Kgroup) * Mr * Kgroup * a_elem;
    };
    auto b_panel_stride = [&](int kc_padded) -> size_t {
        if (cfg.dtype == GemmDataType::kFP32)
            return (size_t)Nr * kc_padded * b_elem;
        return (size_t)(kc_padded / Kgroup) * Nr * Kgroup * b_elem;
    };

    // Determine thread count
    int num_threads = gemm_get_num_threads();
    int64_t flops = (int64_t)2 * M * N * K;
    if (flops < cfg.threading_min_flops) num_threads = 1;

    // Compute 2D thread decomposition
    ThreadDecomp td;
    if (num_threads > 1 && cfg.prefer_2d_threading) {
        td = compute_thread_decomp(M, N, Mc, Nc, num_threads, cfg.shape);
    } else {
        td = {num_threads, 1, num_threads};
    }
    const int mt = td.mt;
    const int nt = td.nt;
    num_threads = mt * nt;

    // Block counts
    int n_mc_blocks = (M + Mc - 1) / Mc;
    int n_nc_blocks = (N + Nc - 1) / Nc;

    // Allocate per-thread buffers
    int n_panels_max = (Nc + Nr - 1) / Nr;
    size_t packed_b_size = (size_t)n_panels_max * b_panel_stride(Kc);

    int m_panels_max = (Mc + Mr - 1) / Mr;
    size_t packed_a_size = (size_t)m_panels_max * a_panel_stride(Kc);

    // packed_A: one per mt-thread (threads sharing same tid_m share A panel)
    // packed_B: one per nt-thread (threads sharing same tid_n share B panel)
    std::vector<AlignedPtr<uint8_t>> packed_A_bufs(mt);
    for (int t = 0; t < mt; ++t)
        packed_A_bufs[t] = aligned_array<uint8_t>(packed_a_size);

    // packed_B: one per nt-thread, use huge pages for large panels
    std::vector<HugePagePtr<uint8_t>> packed_B_bufs(nt);
    for (int t = 0; t < nt; ++t)
        packed_B_bufs[t] = aligned_array_huge<uint8_t>(packed_b_size);

    // Per-thread edge buffers (one per thread)
    std::vector<std::vector<float>> edge_bufs(num_threads);
    for (int t = 0; t < num_threads; ++t)
        edge_bufs[t].resize(Mr * Nr, 0.0f);

    // Assign M-blocks and N-blocks to thread coordinates
    // tid_m in [0, mt): handles M-blocks [mc_start, mc_end)
    // tid_n in [0, nt): handles N-blocks [nc_start, nc_end)
    auto block_range = [](int n_blocks, int n_threads, int tid,
                          int* start, int* end) {
        int per_thread = (n_blocks + n_threads - 1) / n_threads;
        *start = tid * per_thread;
        *end = std::min(*start + per_thread, n_blocks);
    };

    // Main computation with 2D parallelism
    // The N-blocking (jc loop) and K-blocking (pc loop) are serial outer loops.
    // The M-blocking is parallelized across mt, and within each mt's M-block,
    // the N-panels are parallelized across nt.
    //
    // For 2D decomposition, we distribute Nc-blocks across nt threads and
    // Mc-blocks across mt threads within a single parallel region.

    // K-loop is always serial (accumulation dependency)
    for (int pc = 0; pc < K; pc += Kc) {
        int kc = std::min(Kc, K - pc);
        int kc_padded = (Kgroup > 1) ? ((kc + Kgroup - 1) / Kgroup) * Kgroup : kc;

        float beta_eff  = (pc == 0) ? beta : 1.0f;
        // Apply alpha on every K-pass so each partial sum is scaled
        // uniformly. Combined with beta_eff this gives:
        //   First pass:  C = alpha * partial_0 + beta * C_old
        //   Later passes: C = alpha * partial_i + 1.0 * C_accumulated
        float alpha_eff = alpha;

        size_t a_stride = a_panel_stride(kc_padded);
        size_t b_stride = b_panel_stride(kc_padded);

#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads) if(num_threads > 1)
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            int tid_m = tid % mt;
            int tid_n = tid / mt;

            // This thread's M-block range
            int mc_start, mc_end;
            block_range(n_mc_blocks, mt, tid_m, &mc_start, &mc_end);

            // This thread's N-block range
            int nc_start, nc_end;
            block_range(n_nc_blocks, nt, tid_n, &nc_start, &nc_end);

            uint8_t* my_packed_A = packed_A_bufs[tid_m].get();
            uint8_t* my_packed_B = packed_B_bufs[tid_n].get();
            auto& my_edge_buf = edge_bufs[tid];

            // Iterate over N-blocks assigned to this thread
            for (int nc_idx = nc_start; nc_idx < nc_end; ++nc_idx) {
                int jc = nc_idx * Nc;
                int nc = std::min(Nc, N - jc);
                int n_panels = (nc + Nr - 1) / Nr;

                // Pack B for this N-block
                float scale_B = 1.0f;
                cfg.pack_b(kc, nc, &B[pc * ldb + jc], ldb,
                           my_packed_B, Nr, &scale_B);

                // Iterate over M-blocks assigned to this thread
                for (int mc_idx = mc_start; mc_idx < mc_end; ++mc_idx) {
                    int ic = mc_idx * Mc;
                    int mc = std::min(Mc, M - ic);

                    // Pack A for this M-block
                    float scale_A = 1.0f;
                    cfg.pack_a(mc, kc, &A[ic * lda + pc], lda,
                               my_packed_A, Mr, &scale_A);

                    float extra = (cfg.dtype == GemmDataType::kINT8)
                        ? (scale_A * scale_B) : 0.0f;

                    int m_panels = (mc + Mr - 1) / Mr;

                    // Loop 4+5: micro-tiles
                    for (int jr = 0; jr < n_panels; ++jr) {
                        int n_start = jr * Nr;
                        int n_rem = std::min(Nr, nc - n_start);
                        const void* B_panel = my_packed_B + jr * b_stride;

                        for (int ir = 0; ir < m_panels; ++ir) {
                            int m_start = ir * Mr;
                            int m_rem = std::min(Mr, mc - m_start);
                            const void* A_panel = my_packed_A + ir * a_stride;

                            float* C_ptr = &C[(ic + m_start) * ldc + jc + n_start];

                            if (m_rem == Mr && n_rem == Nr) {
                                cfg.ukernel(kc_padded, A_panel, B_panel,
                                           C_ptr, ldc, alpha_eff, beta_eff, extra);
                            } else {
                                std::fill(my_edge_buf.begin(), my_edge_buf.end(), 0.0f);
                                if (beta_eff != 0.0f) {
                                    for (int i = 0; i < m_rem; ++i)
                                        memcpy(&my_edge_buf[i * Nr], &C_ptr[i * ldc],
                                               n_rem * sizeof(float));
                                }
                                cfg.ukernel(kc_padded, A_panel, B_panel,
                                           my_edge_buf.data(), Nr, alpha_eff, beta_eff,
                                           extra);
                                for (int i = 0; i < m_rem; ++i)
                                    memcpy(&C_ptr[i * ldc], &my_edge_buf[i * Nr],
                                           n_rem * sizeof(float));
                            }
                        }
                    }
                }
            }
        }  // end omp parallel
    }
}

}  // namespace dnnopt
