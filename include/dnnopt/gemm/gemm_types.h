#pragma once
/// @file gemm_types.h
/// Shared types for GEMM operations.

#include <cstdint>

namespace dnnopt {

/// Algorithm selection for GEMM dispatch.
enum class GemmAlgo {
    kAuto,        // Automatic: pick best for current hardware + shape
    kNaive,       // Scalar reference (for testing only)
    kNeonFp32,    // NEON 8x12 FP32 microkernel + BLIS blocking
    kBf16Bfmmla,  // BF16 BFMMLA microkernel (future)
    kInt8Smmla,   // INT8 SMMLA microkernel (future)
    kInt8Sdot,    // INT8 SDOT microkernel (future)
};

/// Cache blocking parameters for BLIS-style GEMM.
struct GemmBlockingParams {
    int Mr;   // Microkernel row tile
    int Nr;   // Microkernel column tile
    int Mc;   // L2 blocking on M
    int Nc;   // L3 blocking on N
    int Kc;   // L2 blocking on K
};

}  // namespace dnnopt
