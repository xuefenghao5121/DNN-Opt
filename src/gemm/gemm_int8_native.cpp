/// @file gemm_int8_native.cpp
/// Native INT8 GEMM: INT8 input → INT32 output.
///
/// This version takes pre-quantized INT8 input and produces INT32 output.
/// No quantization/dequantization overhead - pure INT8×INT8→INT32 compute.
///
/// Use case: Conv3D/Conv2D INT8 where input/filter are already quantized.
///
/// TODO: Optimize with SMMLA instruction for 8×8 tile processing.
///       Current implementation is scalar for correctness.
///       SMMLA requires specific packed layout for optimal performance.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/aligned_alloc.h"

#include <algorithm>
#include <cstring>

namespace dnnopt {

/// Native INT8 GEMM: INT8 input → INT32 output.
/// Computes C = A × B where A and B are INT8, C is INT32.
/// A is M×K row-major, B is K×N row-major, C is M×N row-major.
void gemm_int8_int8int8int32(int M, int N, int K,
                              const int8_t* A, int lda,
                              const int8_t* B, int ldb,
                              int32_t* C, int ldc) {
    // Simple scalar implementation for correctness
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)A[i * lda + k] * (int32_t)B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

}  // namespace dnnopt