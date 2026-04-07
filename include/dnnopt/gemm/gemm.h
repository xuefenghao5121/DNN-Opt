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

/// BF16 GEMM: input/output FP32, internal compute BF16 via BFMMLA.
/// Requires ARMv8.6+ BF16 support. Falls back to FP32 if unavailable.
void gemm_bf16(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

}  // namespace dnnopt
