#pragma once
/// @file gemm_autotune.h
/// Lightweight runtime auto-tuning for GEMM on unknown ARM CPUs.
///
/// When the CPU is not in the built-in tuning profile database,
/// this module runs a quick micro-benchmark to find near-optimal
/// cache blocking parameters. The tuned profile is cached for the
/// session lifetime.
///
/// Inspired by autoGEMM's TVM-based auto-tuning, but dramatically
/// lighter: 3 shapes × 3 parameter sets = 9 trials, <5ms total.

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
/// Thread-safe. First call may take ~5ms for auto-tuning;
/// subsequent calls return cached result instantly.
const CpuTuningProfile& get_autotuned_profile();

/// Force re-running auto-tune (e.g., after CPU migration in VM).
void reset_autotune_cache();

}  // namespace dnnopt
