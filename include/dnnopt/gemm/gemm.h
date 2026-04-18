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

/// BF16 GEMM with native bfloat16 input (for oneDNN integration).
/// A: M×K bfloat16 row-major, B: K×N bfloat16 row-major, C: M×N float row-major.
/// Computes: C = alpha * A * B + beta * C
void gemm_bf16_bf16bf16f32(int M, int N, int K,
                           float alpha, const bfloat16_t* A, int lda,
                           const bfloat16_t* B, int ldb,
                           float beta, float* C, int ldc);

/// INT8 GEMM: input/output FP32, internal compute INT8 via SMMLA.
/// Uses symmetric per-tensor quantization. Requires ARMv8.6+ I8MM support.
/// Falls back to FP32 if unavailable.
void gemm_int8(int M, int N, int K,
               float alpha, const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

// ============================================================
// FP8 GEMM (ARMv9-A, requires hardware support)
// ============================================================

/// FP8 E4M3 GEMM: precision-optimized format.
/// A is FP8 E4M3, B is FP8 E5M2 (mixed precision for better range).
/// C is FP32 output. Requires ARMv9-A FP8 support.
///
/// Note: Returns immediately if hardware doesn't support FP8.
///       Check with `has_fp8_support()` before calling.
void gemm_fp8_e4m3(int M, int N, int K,
                   float alpha, const fp8_e4m3_t* A, int lda,
                   const fp8_e5m2_t* B, int ldb,
                   float beta, float* C, int ldc);

/// FP8 E5M2 GEMM: range-optimized format.
/// Both A and B are FP8 E5M2. C is FP32 output.
/// Requires ARMv9-A FP8 support.
void gemm_fp8_e5m2(int M, int N, int K,
                   float alpha, const fp8_e5m2_t* A, int lda,
                   const fp8_e5m2_t* B, int ldb,
                   float beta, float* C, int ldc);

/// FP8 GEMM with FP32 input (auto-quantize).
/// Input A/B are FP32, converted to FP8 E4M3/E5M2 internally.
/// Requires ARMv9-A FP8 support.
void gemm_fp8(int M, int N, int K,
              float alpha, const float* A, int lda,
              const float* B, int ldb,
              float beta, float* C, int ldc);

/// Check if FP8 hardware support is available.
/// Returns true if __ARM_FEATURE_FP8 is defined and runtime check passes.
bool has_fp8_support();

/// Set the number of threads for GEMM operations.
/// n=0: auto (use all cores), n=1: single-threaded.
void gemm_set_num_threads(int n);

/// Get the current thread count for GEMM operations.
int gemm_get_num_threads();

}  // namespace dnnopt
