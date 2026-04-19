#pragma once
/// @file gemm_autotune.h
/// Enhanced runtime auto-tuning for GEMM on unknown ARM CPUs.
///
/// v0.9.18 improvements:
///   - Expanded search grid (5 candidates, shape-aware)
///   - Multi-shape testing (large, small, tall-skinny)
///   - Shape-specific tuning profiles
///   - Total cost: ~10-15ms for full autotune
///
/// v0.9.28 improvements:
///   - Kernel selection autotune (small-M vs packed vs adaptive)
///   - ShapeCache integration for persistent kernel choices
///   - Built-in profiles + file cache + online fallback
///
/// Inspired by autoGEMM's TVM-based auto-tuning, but dramatically
/// lighter: 5 shapes × 5 parameter sets = 25 trials, <15ms total.

#include "dnnopt/cpu_tuning_profile.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/gemm/gemm_ukernel_registry.h"  // for GemmDataType

namespace dnnopt {

/// Get the best tuning profile, with auto-tuning for unknown CPUs.
///
/// Flow:
///   1. Try built-in database (lookup_tuning_profile)
///   2. If exact match found → return immediately
///   3. If only generic default matched → run micro-benchmark
///   4. Cache result → return tuned profile
///
/// Thread-safe. First call may take ~10-15ms for auto-tuning;
/// subsequent calls return cached result instantly.
const CpuTuningProfile& get_autotuned_profile();

/// Force re-running auto-tune (e.g., after CPU migration in VM).
void reset_autotune_cache();

// ============================================================
// v0.9.28: Kernel Selection Autotune
// ============================================================

/// Select optimal GEMM kernel for given shape.
/// Uses ShapeCache + micro-benchmark fallback.
///
/// @param M, N, K   Matrix dimensions
/// @param dtype     Data type (FP32, BF16, INT8)
/// @return          Optimal kernel ID
GemmKernelId select_gemm_kernel(int M, int N, int K, GemmDataType dtype);

/// Warmup GEMM autotune for common shapes.
/// Runs micro-benchmarks and populates cache.
///
/// @param shapes    Array of shapes to warmup (nullptr = use defaults)
/// @param n_shapes  Number of shapes
void warmup_gemm_autotune(const int* shapes_M = nullptr,
                          const int* shapes_N = nullptr,
                          const int* shapes_K = nullptr,
                          int n_shapes = 0);

/// Load GEMM kernel cache from file.
/// Returns number of entries loaded, -1 on error.
int load_gemm_kernel_cache(const char* path);

/// Save GEMM kernel cache to file.
/// Returns 0 on success, -1 on error.
int save_gemm_kernel_cache(const char* path);

}  // namespace dnnopt
