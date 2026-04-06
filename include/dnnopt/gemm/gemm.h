#pragma once
/// @file gemm.h
/// Public GEMM API for DNN-Opt.
///
/// BLAS-like interface: C = alpha * A * B + beta * C
/// All matrices are row-major.

#include "dnnopt/gemm/gemm_types.h"

namespace dnnopt {

/// FP32 GEMM with automatic algorithm selection.
/// A: M×K, B: K×N, C: M×N (row-major).
void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

/// FP32 GEMM with explicit algorithm choice (for benchmarking/debugging).
void gemm_fp32(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc,
               GemmAlgo algo);

}  // namespace dnnopt
