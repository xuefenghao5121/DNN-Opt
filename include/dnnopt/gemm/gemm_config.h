#pragma once
/// @file gemm_config.h
/// Platform-specific cache blocking parameters for GEMM.

#include "dnnopt/gemm/gemm_types.h"
#include "dnnopt/arm_hwcaps.h"

namespace dnnopt {

// FP32 NEON microkernel tile dimensions.
constexpr int kGemmMrFp32 = 8;
constexpr int kGemmNrFp32 = 12;

/// Select cache blocking parameters based on detected CPU.
inline GemmBlockingParams get_gemm_blocking_params() {
    const auto& hw = detect_arm_hwcaps();
    GemmBlockingParams p;
    p.Mr = kGemmMrFp32;
    p.Nr = kGemmNrFp32;

    switch (hw.part_number) {
    case 0xd40:  // Neoverse V1
    case 0xd4f:  // Neoverse V2
        p.Mc = 128;  p.Nc = 2048;  p.Kc = 384;
        break;
    case 0xd48:  // Cortex-X2
        p.Mc = 64;   p.Nc = 1024;  p.Kc = 256;
        break;
    case 0xd0c:  // Neoverse N1
    case 0xd49:  // Neoverse N2
    default:
        p.Mc = 128;  p.Nc = 2048;  p.Kc = 512;
        break;
    }
    return p;
}

}  // namespace dnnopt
