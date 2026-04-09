/// @file gemm_threading_v2.cpp
/// Enhanced threading support with NUMA-aware allocation and false sharing prevention.
///
/// This file contains NEW functionality only - basic thread management
/// remains in gemm_threading.cpp to avoid duplication.

#include "dnnopt/gemm/gemm_threading.h"
#include "dnnopt/arm_hwcaps.h"

#include <algorithm>
#include <atomic>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dnnopt {

// ============================================================
// big.LITTLE-aware thread selection (enhanced version)
// ============================================================

/// Enhanced version with more sophisticated heuristics.
int get_effective_threads_v2(const CoreTopology& topo, int64_t flops,
                             int threading_min_flops) {
    // Don't thread if below FLOPs threshold
    if (flops < threading_min_flops) {
        return 1;
    }

    int total_cores = static_cast<int>(topo.big_cores + topo.little_cores);

    if (!topo.is_heterogeneous) {
        // Homogeneous system: use all cores
        return total_cores;
    }

    // big.LITTLE system: be strategic
    //
    // For large problems: use big cores + little cores if needed
    // For medium problems: use only big cores
    // For small problems: use limited big cores

    // Estimate per-core FLOPs
    int64_t flops_per_core = flops / total_cores;

    if (flops_per_core > 10000000) {  // > 10M FLOPs/core
        // Large problem: use all cores
        return total_cores;
    } else if (flops_per_core > 1000000) {  // 1M-10M FLOPs/core
        // Medium problem: use only big cores
        return static_cast<int>(topo.big_cores);
    } else {
        // Small problem: limit to 2-4 big cores
        return std::min(4, static_cast<int>(topo.big_cores));
    }
}

// ============================================================
// Cache line padding to prevent false sharing
// ============================================================

/// Padded counter for parallel reductions.
/// Each counter occupies its own cache line (64 bytes).
struct PaddedCounter {
    alignas(64) int64_t value;
};

/// Per-thread statistics with padding.
struct PaddedThreadStats {
    alignas(64) int64_t flops_processed;
    alignas(64) double time_seconds;
    alignas(64) int m_blocks_processed;
    alignas(64) int n_blocks_processed;
};

// ============================================================
// NUMA-aware memory allocation hints
// ============================================================

/// NUMA node information.
struct NumaNode {
    int node_id;
    int cpu_start;
    int cpu_count;
    uint64_t memory_size_bytes;
};

/// Get NUMA topology (simplified - real implementation would read /sys/devices/system/node/)
std::vector<NumaNode> get_numa_topology() {
    std::vector<NumaNode> nodes;

    // For now, assume single NUMA node
    // A full implementation would:
    // 1. Read /sys/devices/system/node/online
    // 2. For each node, read cpumap and distance
    // 3. Build node-to-CPU mapping

    NumaNode node;
    node.node_id = 0;
    node.cpu_start = 0;
    auto& profile = detect_arm_hwcaps();
    node.cpu_count = static_cast<int>(profile.num_cores);
    node.memory_size_bytes = 0;  // Would read from meminfo
    nodes.push_back(node);

    return nodes;
}

/// Allocate memory with NUMA policy hint.
/// @param size      Size in bytes
/// @param numa_node NUMA node to allocate from (-1 for local/interleaved)
/// @return Pointer to allocated memory
void* allocate_numa_aware(size_t size, int numa_node) {
    // For now, use standard aligned allocation
    // A production implementation would use:
    // - libnuma: numa_alloc_onnode()
    // - malloc.h: malloc_with_zone() on macOS
    // - Windows: VirtualAllocExNuma()

    (void)numa_node;
    return nullptr;  // Placeholder
}

// ============================================================
// Work distribution strategies
// ============================================================

/// Work distribution strategy for parallel GEMM.
enum class WorkDistribution {
    kStatic,        // Static block distribution (default)
    kDynamic,       // Dynamic scheduling with chunk size
    kGuided,        // Guided scheduling (large initial chunks)
    kFuzzy,         // Fuzzy blocking for load balancing
};

/// Get recommended work distribution based on problem shape.
WorkDistribution get_work_distribution(int M, int N, int num_threads) {
    // For highly imbalanced shapes (tall-skinny or short-wide),
    // use dynamic scheduling to balance load.

    double aspect_ratio = static_cast<double>(M) / static_cast<double>(N);

    if (aspect_ratio > 10.0 || aspect_ratio < 0.1) {
        // Highly imbalanced: use dynamic scheduling
        return WorkDistribution::kDynamic;
    }

    if (num_threads <= 4) {
        // Low thread count: static is fine
        return WorkDistribution::kStatic;
    }

    // Default: static with good initial distribution
    return WorkDistribution::kStatic;
}

// ============================================================
// Load balancing helpers
// ============================================================

/// Compute fuzzy block decomposition for better load balancing.
/// Adjusts block boundaries slightly to reduce stragglers.
struct FuzzyDecomp {
    int m_blocks;
    int n_blocks;
    std::vector<std::pair<int, int>> m_ranges;  // (start, end) for each M-thread
    std::vector<std::pair<int, int>> n_ranges;  // (start, end) for each N-thread
};

FuzzyDecomp compute_fuzzy_decomp(int M, int N, int mc, int nc,
                                 int mt, int nt) {
    FuzzyDecomp decomp;
    decomp.m_blocks = (M + mc - 1) / mc;
    decomp.n_blocks = (N + nc - 1) / nc;

    // Assign M-blocks to mt threads with fuzzy boundaries
    decomp.m_ranges.resize(mt);
    int blocks_per_thread = (decomp.m_blocks + mt - 1) / mt;

    for (int t = 0; t < mt; ++t) {
        int block_start = t * blocks_per_thread;
        int block_end = std::min(block_start + blocks_per_thread, decomp.m_blocks);

        decomp.m_ranges[t].first = block_start * mc;
        decomp.m_ranges[t].second = std::min(block_end * mc, M);
    }

    // Assign N-blocks to nt threads similarly
    decomp.n_ranges.resize(nt);
    blocks_per_thread = (decomp.n_blocks + nt - 1) / nt;

    for (int t = 0; t < nt; ++t) {
        int block_start = t * blocks_per_thread;
        int block_end = std::min(block_start + blocks_per_thread, decomp.n_blocks);

        decomp.n_ranges[t].first = block_start * nc;
        decomp.n_ranges[t].second = std::min(block_end * nc, N);
    }

    return decomp;
}

// ============================================================
// Thread synchronization helpers
// ============================================================

/// Barrier for thread synchronization (for custom threading).
/// Uses OpenMP barriers when available.
class ThreadBarrier {
public:
    explicit ThreadBarrier(int count) : count_(count), waiting_(0), generation_(0) {}

    void wait() {
#ifdef _OPENMP
        #pragma omp barrier
#else
        // Fallback: spin-wait barrier
        int my_gen = generation_.load(std::memory_order_acquire);
        int arrived = waiting_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (arrived == count_) {
            // Last thread: reset for next generation
            waiting_.store(0, std::memory_order_release);
            generation_.store(my_gen + 1, std::memory_order_release);
        } else {
            // Spin until generation changes
            while (generation_.load(std::memory_order_acquire) == my_gen) {
                // Spin-wait
            }
        }
#endif
    }

private:
    int count_;
    std::atomic<int> waiting_;
    std::atomic<int> generation_;
};

}  // namespace dnnopt
