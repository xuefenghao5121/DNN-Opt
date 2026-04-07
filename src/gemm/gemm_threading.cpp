/// @file gemm_threading.cpp
/// Thread control for GEMM operations with big.LITTLE awareness.

#include "dnnopt/gemm/gemm_threading.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#endif

#include <algorithm>

namespace dnnopt {

static int g_gemm_num_threads = 0;  // 0 = auto

void gemm_set_num_threads(int n) {
    g_gemm_num_threads = (n < 0) ? 0 : n;
}

int gemm_get_num_threads() {
    if (g_gemm_num_threads > 0) return g_gemm_num_threads;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int get_effective_threads(const CoreTopology& topo, int64_t flops,
                          int threading_min_flops) {
    int max_threads = gemm_get_num_threads();

    // Below minimum FLOPS threshold: single thread
    if (flops < threading_min_flops) return 1;

    if (!topo.is_heterogeneous) return max_threads;

    // Heterogeneous: prefer big cores for compute-intensive work
    int big = (int)topo.big_cores;
    if (big <= 0) return max_threads;

    // For medium workloads, use only big cores
    // For large workloads (>10M FLOPS), use all cores
    if (flops < 10000000LL) {
        return std::min(max_threads, big);
    }
    return max_threads;
}

void apply_thread_affinity(const CoreTopology& topo, int num_threads) {
#if defined(__linux__) && defined(_OPENMP)
    if (!topo.is_heterogeneous) return;
    if (topo.clusters.empty()) return;

    // Build ordered CPU list: big cores first, then little cores
    std::vector<int> cpu_order;
    for (const auto& cl : topo.clusters) {
        if (cl.is_big) {
            for (uint32_t i = 0; i < cl.count; ++i)
                cpu_order.push_back((int)(cl.first_cpu + i));
        }
    }
    for (const auto& cl : topo.clusters) {
        if (!cl.is_big) {
            for (uint32_t i = 0; i < cl.count; ++i)
                cpu_order.push_back((int)(cl.first_cpu + i));
        }
    }

    // Apply affinity inside parallel region
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        if (tid < (int)cpu_order.size()) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_order[tid], &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }
    }
#else
    (void)topo;
    (void)num_threads;
#endif
}

}  // namespace dnnopt
