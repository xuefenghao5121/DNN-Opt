#pragma once
/// @file gemm_threading.h
/// Thread control API for GEMM operations.

#include "dnnopt/arm_hwcaps.h"
#include <cstdint>

namespace dnnopt {

/// Set the number of threads for GEMM operations.
/// n=0 means auto (use all available cores).
/// n=1 disables threading.
void gemm_set_num_threads(int n);

/// Get the current thread count for GEMM operations.
int gemm_get_num_threads();

/// Get effective thread count considering big.LITTLE topology.
/// For large problems, use all big cores (+ little if needed).
/// For small problems, limit to big cores only.
int get_effective_threads(const CoreTopology& topo, int64_t flops,
                          int threading_min_flops);

/// Apply thread affinity for big.LITTLE scheduling.
/// Pins OpenMP threads to big cores first, then little cores.
/// No-op on homogeneous systems.
void apply_thread_affinity(const CoreTopology& topo, int num_threads);

}  // namespace dnnopt
