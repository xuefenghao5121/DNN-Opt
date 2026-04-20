/// @file gemm_smallm_fp32_sve.cpp
/// Small-M specialized FP32 GEMM using SVE2 predicates.
///
/// SVE2 advantages for small-M:
///   - Predicate-based edge handling (N%4!=0, irregular N)
///   - Single instruction for variable N width
///   - Cleaner code without NEON edge handling
///
/// Kernels:
///   - gemm_ukernel_fp32_1xVL_sve: M=1, N=VL-wide
///   - gemm_ukernel_fp32_MxVL_sve: M=2-7, N=VL-wide
///   - gemm_smallm_fp32_sve: Full driver with predicate dispatch

#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/arm_hwcaps.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#include <cstring>
#include <algorithm>

namespace dnnopt {

// ============================================================
// SVE Small-M Microkernels
// ============================================================

/// 1×VL microkernel: compute one row of C with SVE predicate.
/// Uses VL-wide (128-bit = 4 floats on N2) processing.
///
/// C[0, 0:N] = alpha * sum_k(A[k] * B[k, 0:N]) + beta * C[0, 0:N]
static void gemm_ukernel_fp32_1xVL_sve(int K, int N,
                                        const float* A,
                                        const float* B, int ldb,
                                        float* C, int ldc,
                                        float alpha, float beta) {
    // Predicate for N columns
    svbool_t pg_n = svwhilelt_b32(0, N);

    // Single accumulator for VL columns
    svfloat32_t acc = svdup_f32(0.0f);

    // K-loop
    for (int k = 0; k < K; ++k) {
        // Broadcast A[k] to all lanes
        svfloat32_t a_vec = svdup_f32(A[k]);

        // Load B[k, 0:N] with predicate
        svfloat32_t b_vec = svld1_f32(pg_n, &B[k * ldb]);

        // FMLA: acc += a * b
        acc = svmla_f32_m(pg_n, acc, a_vec, b_vec);
    }

    // Scale by alpha
    acc = svmul_f32_z(svptrue_b32(), acc, svdup_f32(alpha));

    // Apply beta to existing C
    if (beta != 0.0f) {
        svfloat32_t c_old = svld1_f32(pg_n, C);
        acc = svmla_f32_m(pg_n, acc, c_old, svdup_f32(beta));
    }

    // Store result
    svst1_f32(pg_n, C, acc);
}

/// M×VL microkernel: compute M rows (M=2-7) of C with SVE predicate.
/// Each row uses its own accumulator, A broadcast, and B load.
static void gemm_ukernel_fp32_MxVL_sve(int M, int K, int N,
                                        const float* A, int lda,
                                        const float* B, int ldb,
                                        float* C, int ldc,
                                        float alpha, float beta) {
    // Predicate for N columns
    svbool_t pg_n = svwhilelt_b32(0, N);

    // Accumulators for each row (max 7 rows)
    svfloat32_t acc[7];
    for (int m = 0; m < M; ++m) {
        acc[m] = svdup_f32(0.0f);
    }

    // K-loop
    for (int k = 0; k < K; ++k) {
        // Load B[k, 0:N] once (shared by all rows)
        svfloat32_t b_vec = svld1_f32(pg_n, &B[k * ldb]);

        // Process each row
        for (int m = 0; m < M; ++m) {
            // Broadcast A[m, k] to all lanes
            svfloat32_t a_vec = svdup_f32(A[m * lda + k]);

            // FMLA: acc[m] += a * b
            acc[m] = svmla_f32_m(pg_n, acc[m], a_vec, b_vec);
        }
    }

    // Store results for each row
    for (int m = 0; m < M; ++m) {
        // Scale by alpha
        svfloat32_t result = svmul_f32_z(svptrue_b32(), acc[m], svdup_f32(alpha));

        // Apply beta to existing C
        if (beta != 0.0f) {
            svfloat32_t c_old = svld1_f32(pg_n, &C[m * ldc]);
            result = svmla_f32_m(pg_n, result, c_old, svdup_f32(beta));
        }

        // Store
        svst1_f32(pg_n, &C[m * ldc], result);
    }
}

// ============================================================
// Driver: gemm_smallm_fp32_sve
// ============================================================

/// Small-M FP32 GEMM driver using SVE2 predicates.
/// Dispatches to appropriate microkernel based on M.
///
/// @param M, N, K  Matrix dimensions
/// @param A        Input A matrix (M×K)
/// @param lda      A stride
/// @param B        Input B matrix (K×N)
/// @param ldb      B stride
/// @param C        Output C matrix (M×N)
/// @param ldc      C stride
/// @param alpha, beta  Scaling factors
void gemm_smallm_fp32_sve(int M, int N, int K,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float* C, int ldc,
                          float alpha, float beta) {
    // Process N in VL-wide panels (4 floats on N2)
    const int VL = (int)svcntw();  // FP32 elements per SVE register

    for (int j = 0; j < N; j += VL) {
        int n_panel = std::min(VL, N - j);

        if (M == 1) {
            // Single row: 1×VL kernel
            gemm_ukernel_fp32_1xVL_sve(K, n_panel,
                                       A,
                                       B + j, ldb,
                                       C + j, ldc,
                                       alpha, beta);
        } else {
            // Multiple rows: M×VL kernel
            gemm_ukernel_fp32_MxVL_sve(M, K, n_panel,
                                       A, lda,
                                       B + j, ldb,
                                       C + j, ldc,
                                       alpha, beta);
        }
    }
}

}  // namespace dnnopt

#else  // !__ARM_FEATURE_SVE

// Fallback to NEON when SVE not available
namespace dnnopt {

// Forward declaration
void gemm_smallm_driver_fp32(int M, int N, int K,
                              float alpha, const float* A, int lda,
                              const float* B, int ldb,
                              float beta, float* C, int ldc);

void gemm_smallm_fp32_sve(int M, int N, int K,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float* C, int ldc,
                          float alpha, float beta) {
    // Fallback: call NEON small-M driver
    gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace dnnopt

#endif  // __ARM_FEATURE_SVE