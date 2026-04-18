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
/// Inspired by autoGEMM's TVM-based auto-tuning, but dramatically
/// lighter: 5 shapes × 5 parameter sets = 25 trials, <15ms total.

#include "dnnopt/cpu_tuning_profile.h"
#include "dnnopt/arm_hwcaps.h"

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

}  // namespace dnnopt
