#pragma once
/// @file gemm_thread_decomp.h
/// 2D thread decomposition for M×N parallelism in GEMM.
///
/// Instead of parallelizing only the M dimension, this decomposes
/// the thread team into mt × nt, where mt threads cover M-blocks
/// and nt threads cover N-blocks. This improves load balance for
/// shapes where one dimension is much larger (e.g., BERT inference).

#include "dnnopt/cpu_tuning_profile.h"
#include <algorithm>
#include <cmath>

namespace dnnopt {

/// 2D thread decomposition result.
struct ThreadDecomp {
    int mt;           // Threads on M dimension
    int nt;           // Threads on N dimension
    int num_threads;  // Total = mt * nt (may be <= original if rounding)
};

/// Compute optimal M×N thread decomposition.
///
/// Algorithm: enumerate factor pairs (mt, nt) where mt*nt <= num_threads.
/// Score each by work imbalance + shape bias. Return lowest score.
///
/// @param M  Total rows
/// @param N  Total columns
/// @param Mc M-block size
/// @param Nc N-block size
/// @param num_threads  Available threads
/// @param shape  Matrix shape classification
/// @return Optimal (mt, nt) decomposition
inline ThreadDecomp compute_thread_decomp(int M, int N, int Mc, int Nc,
                                           int num_threads, ShapeClass shape) {
    if (num_threads <= 1) return {1, 1, 1};

    int m_blocks = (M + Mc - 1) / Mc;
    int n_blocks = (N + Nc - 1) / Nc;

    // If only 1 block in one dimension, force all threads to the other
    if (m_blocks <= 1) return {1, num_threads, num_threads};
    if (n_blocks <= 1) return {num_threads, 1, num_threads};

    ThreadDecomp best = {num_threads, 1, num_threads};
    float best_score = 1e9f;

    // Shape bias: prefer more threads on the larger dimension
    float m_bias = 0.0f;
    switch (shape) {
        case ShapeClass::kTallSkinny: m_bias = -2.0f; break;  // Prefer mt > nt
        case ShapeClass::kShortWide:  m_bias =  2.0f; break;  // Prefer nt > mt
        case ShapeClass::kBertLike:   m_bias =  1.5f; break;  // N is large, slight nt preference
        default: break;
    }

    // Enumerate all factor pairs
    for (int mt = 1; mt <= num_threads; ++mt) {
        if (num_threads % mt != 0) continue;
        int nt = num_threads / mt;

        // Work per thread in each dimension
        int m_per_thread = (m_blocks + mt - 1) / mt;
        int n_per_thread = (n_blocks + nt - 1) / nt;

        // Wasted work (idle threads due to remainder)
        int m_waste = mt * m_per_thread - m_blocks;
        int n_waste = nt * n_per_thread - n_blocks;

        // Imbalance score: minimize wasted work
        float imbalance = (float)(m_waste + n_waste);

        // Balance score: prefer similar work per thread on each dimension
        float ratio_diff = std::fabs((float)m_per_thread - (float)n_per_thread);

        // Shape bias adjusts preference for mt vs nt
        float shape_penalty = m_bias * (float)(mt - nt);

        float score = imbalance * 10.0f + ratio_diff + shape_penalty;

        if (score < best_score) {
            best_score = score;
            best.mt = mt;
            best.nt = nt;
            best.num_threads = mt * nt;
        }
    }

    return best;
}

}  // namespace dnnopt
