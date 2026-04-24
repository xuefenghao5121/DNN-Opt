/// @file bench_shape_autotune.cpp
/// Analyze special shapes from shape_rec using DNN-Opt autotune capabilities.
///
/// Shapes:
/// - M=35, N=1, K=800  (irregular M, GEMV-like)
/// - M=35, N=400, K=400 (irregular M, medium)
/// - M=39, N=1, K=5 (tiny, irregular M)
/// - M=39, N=400, K=400 (irregular M, medium)
/// - M=46, N=1, K=5/492 (irregular M, tiny/GEMV)
///
/// All shapes have irregular M dimensions (non-power-of-2).

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/timer.h"

using namespace dnnopt;

struct ShapeInfo {
    int count;
    int M, N, K;
    std::string category;
};

// Check if n is power of 2
bool is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Classify shape based on dimensions (local version)
std::string classify_shape_local(int M, int N, int K) {
    if (N == 1) return "GEMV";
    if (M == 1) return "GEMV_T";
    if (M <= 4) return "Tiny";
    if (M <= 16) return "SmallM";
    if (M <= 32) return "MediumM";
    if (!is_pow2(M)) return "IrregularM";
    return "Large";
}

// Calculate theoretical ops
int64_t calc_ops(int M, int N, int K) {
    return 2LL * M * N * K;
}

void print_shape_analysis(const ShapeInfo& shape) {
    std::cout << "\n========================================\n";
    std::cout << "Shape: M=" << shape.M << ", N=" << shape.N << ", K=" << shape.K;
    std::cout << " (Count=" << shape.count << ", " << shape.category << ")\n";
    std::cout << "========================================\n";

    int64_t ops = calc_ops(shape.M, shape.N, shape.K);
    std::cout << "Total ops: " << ops << " (" << ops / 1e6 << " Mops)\n";

    // Kernel selection analysis
    std::cout << "\n--- Kernel Selection ---\n";
    GemmKernelId kernel = select_gemm_kernel(shape.M, shape.N, shape.K, GemmDataType::kFP32);

    const char* kernel_names[] = {
        "SmallM", "SmallMWide", "AdaptiveTile", "Packed", "Tiny", "SME"
    };
    std::cout << "Selected kernel: " << kernel_names[(int)kernel] << "\n";

    // Kernel selection reasoning
    std::cout << "Reasoning:\n";
    if (shape.N == 1 || shape.M == 1) {
        std::cout << "  - N=1 or M=1 → kTiny (GEMV dispatch)\n";
    } else if (shape.M < 8) {
        std::cout << "  - M < 8 → kSmallM or kSmallMWide\n";
    } else if (shape.M <= 32) {
        if (shape.M * shape.N * shape.K < 4000000) {
            std::cout << "  - M*N*K < 4M → kAdaptiveTile (no packing overhead)\n";
        } else {
            std::cout << "  - M <= 32, total ops moderate → kAdaptiveTile or kPacked\n";
        }
    } else if (!is_pow2(shape.M)) {
        std::cout << "  - M=" << shape.M << " is NOT power-of-2 → irregular dimension\n";
        std::cout << "  - DNN-Opt handles padding better than oneDNN\n";
    } else {
        std::cout << "  - Large M → kPacked (full packing + threading)\n";
    }

    // Blocking parameter analysis
    std::cout << "\n--- Blocking Parameters ---\n";
    BlockingSelection blocking = select_blocking_params(shape.M, shape.N, shape.K);

    const char* blocking_names[] = {
        "Conservative", "Standard", "Moderate", "Aggressive", "Maximum"
    };
    std::cout << "Selected preset: " << blocking_names[(int)blocking.preset] << "\n";

    BlockingParams bp = get_blocking_params_from_preset(blocking.preset);
    std::cout << "  L1D util: " << bp.l1d_util << ", L2 util: " << bp.l2_util
              << ", L3 util: " << bp.l3_util << "\n";

    // Blocking reasoning for irregular shapes
    if (!is_pow2(shape.M)) {
        std::cout << "Special handling for irregular M=" << shape.M << ":\n";
        // Calculate optimal Mc considering M not divisible by common tile sizes
        int tile_sizes[] = {4, 6, 8, 12, 16};
        std::cout << "  - Tile size candidates: ";
        for (int t : tile_sizes) {
            int remainder = shape.M % t;
            std::cout << t << "(r=" << remainder << ") ";
        }
        std::cout << "\n";

        // Find best tile with minimum remainder
        int best_tile = 8;
        int min_remainder = shape.M % best_tile;
        for (int t : tile_sizes) {
            int r = shape.M % t;
            if (r < min_remainder) {
                min_remainder = r;
                best_tile = t;
            }
        }
        std::cout << "  - Best tile: " << best_tile << " (remainder=" << min_remainder << ")\n";
    }

    // Tile size analysis (for packed kernels)
    if (kernel == GemmKernelId::kPacked || shape.M >= 8) {
        std::cout << "\n--- Tile Size Selection ---\n";
        TileSelection tile = select_tile_params(shape.M, shape.N, shape.K);

        const char* tile_names[] = {"4x16", "6x16", "8x12", "8x16"};
        std::cout << "Selected tile: " << tile_names[(int)tile.preset]
                  << " (Mr=" << (int)tile.Mr << ", Nr=" << (int)tile.Nr << ")\n";

        // Tile reasoning
        if (shape.M % 8 == 0) {
            std::cout << "  - M divisible by 8 → 8x16 (maximum compute density)\n";
        } else if (shape.M % 6 == 0) {
            std::cout << "  - M divisible by 6 → 6x16 (no padding)\n";
        } else if (shape.M % 4 == 0) {
            std::cout << "  - M divisible by 4 → 4x16 (no padding)\n";
        } else {
            std::cout << "  - M=" << shape.M << " not divisible by common tiles\n";
            std::cout << "  - Autotune selects best via micro-benchmark\n";
        }
    }

    // Threshold analysis
    std::cout << "\n--- Threshold Analysis ---\n";
    ThresholdSelection thresholds = get_current_thresholds();
    std::cout << "Current thresholds:\n";
    std::cout << "  small_m_bound: " << (int)thresholds.small_m_bound << "\n";
    std::cout << "  wide_n_bound: " << (int)thresholds.wide_n_bound << "\n";
    std::cout << "  unpacked_thresh: " << (int)thresholds.unpacked_thresh << "\n";

    // Check if shape triggers threshold boundaries
    bool at_boundary = false;
    if (shape.M == thresholds.small_m_bound ||
        shape.M == thresholds.small_m_bound + 1 ||
        shape.M == thresholds.small_m_bound - 1) {
        std::cout << "  ⚠ M=" << shape.M << " near small_m_bound=" << thresholds.small_m_bound << "\n";
        at_boundary = true;
    }
    if (shape.N == thresholds.wide_n_bound ||
        shape.N == thresholds.wide_n_bound - 1) {
        std::cout << "  ⚠ N=" << shape.N << " near wide_n_bound=" << thresholds.wide_n_bound << "\n";
        at_boundary = true;
    }
    if (shape.M * shape.N * shape.K == thresholds.unpacked_thresh ||
        shape.M * shape.N * shape.K == thresholds.unpacked_thresh - 1) {
        std::cout << "  ⚠ Total ops near unpacked_thresh\n";
        at_boundary = true;
    }
    if (!at_boundary) {
        std::cout << "  ✓ Shape not at threshold boundary\n";
    }

    // Actual benchmark
    std::cout << "\n--- Performance Benchmark ---\n";

    // Allocate aligned memory
    size_t A_size = shape.M * shape.K;
    size_t B_size = shape.K * shape.N;
    size_t C_size = shape.M * shape.N;

    float* A = new float[A_size];
    float* B = new float[B_size];
    float* C = new float[C_size];

    // Initialize with random-ish values
    for (size_t i = 0; i < A_size; i++) A[i] = (i % 100) * 0.01f;
    for (size_t i = 0; i < B_size; i++) B[i] = (i % 100) * 0.01f;
    for (size_t i = 0; i < C_size; i++) C[i] = 0.0f;

    // Warmup
    gemm_fp32(shape.M, shape.N, shape.K, 1.0f, A, shape.K, B, shape.N, 0.0f, C, shape.N);

    // Benchmark
    Timer timer;
    int n_iters = (ops < 100000) ? 1000 : (ops < 1000000) ? 100 : 10;

    timer.start();
    for (int i = 0; i < n_iters; i++) {
        gemm_fp32(shape.M, shape.N, shape.K, 1.0f, A, shape.K, B, shape.N, 0.0f, C, shape.N);
    }
    timer.stop();

    double total_time_us = timer.elapsed_us();
    double time_per_iter_us = total_time_us / n_iters;
    double gflops = (ops / 1e9) / (time_per_iter_us / 1e6);

    std::cout << "Iterations: " << n_iters << "\n";
    std::cout << "Total time: " << total_time_us << " us\n";
    std::cout << "Time per iter: " << time_per_iter_us << " us\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";

    delete[] A;
    delete[] B;
    delete[] C;
}

int main() {
    std::cout << "DNN-Opt Autotune Analysis for Special Shapes\n";
    std::cout << "=============================================\n\n";

    // Initialize autotune
    std::cout << "Initializing DNN-Opt autotune...\n";
    warmup_all_autotune();
    std::cout << "Autotune warmup complete.\n\n";

    // Shapes from shape_rec
    std::vector<ShapeInfo> shapes = {
        {1, 35, 1, 800, classify_shape_local(35, 1, 800)},
        {4, 35, 400, 400, classify_shape_local(35, 400, 400)},
        {1, 39, 1, 5, classify_shape_local(39, 1, 5)},
        {3, 35, 1, 492, classify_shape_local(35, 1, 492)},
        {1, 39, 1, 800, classify_shape_local(39, 1, 800)},
        {1, 35, 1, 800, classify_shape_local(35, 1, 800)},
        {4, 39, 1, 800, classify_shape_local(39, 1, 800)},
        {4, 39, 400, 400, classify_shape_local(39, 400, 400)},
        {1, 46, 1, 5, classify_shape_local(46, 1, 5)},
        {3, 46, 1, 492, classify_shape_local(46, 1, 492)},
    };

    // Summary table
    std::cout << "\n=== Shape Summary ===\n";
    std::cout << std::setw(6) << "M" << std::setw(6) << "N" << std::setw(6) << "K"
              << std::setw(12) << "TotalOps" << std::setw(14) << "Category"
              << std::setw(8) << "Pow2(M)" << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (const auto& s : shapes) {
        int64_t ops = calc_ops(s.M, s.N, s.K);
        std::cout << std::setw(6) << s.M << std::setw(6) << s.N << std::setw(6) << s.K
                  << std::setw(12) << ops << std::setw(14) << s.category
                  << std::setw(8) << (is_pow2(s.M) ? "Yes" : "No") << "\n";
    }

    // Key observation
    std::cout << "\n=== Key Observations ===\n";
    std::cout << "1. ALL shapes have IRREGULAR M dimensions (35, 39, 46)\n";
    std::cout << "   - Not power-of-2 → causes padding overhead in standard kernels\n";
    std::cout << "   - DNN-Opt's adaptive tile kernel handles this better\n\n";

    std::cout << "2. GEMV-like shapes (N=1):\n";
    std::cout << "   - M=35,N=1,K=800: vector-matrix multiply\n";
    std::cout << "   - DNN-Opt's tiny kernel optimized for N=1\n\n";

    std::cout << "3. Medium shapes with irregular M:\n";
    std::cout << "   - M=35/39,N=400,K=400: inference embedding layers\n";
    std::cout << "   - Autotune can select optimal blocking for cache efficiency\n\n";

    // Detailed analysis for each shape
    std::cout << "\n=== Detailed Autotune Analysis ===\n";

    // Only analyze unique shapes (skip duplicates)
    std::vector<ShapeInfo> unique_shapes;
    for (const auto& s : shapes) {
        bool found = false;
        for (const auto& u : unique_shapes) {
            if (u.M == s.M && u.N == s.N && u.K == s.K) {
                found = true;
                break;
            }
        }
        if (!found) unique_shapes.push_back(s);
    }

    for (const auto& s : unique_shapes) {
        print_shape_analysis(s);
    }

    // Optimization recommendations
    std::cout << "\n========================================\n";
    std::cout << "=== Optimization Recommendations ===\n";
    std::cout << "========================================\n\n";

    std::cout << "For irregular M dimensions (35, 39, 46):\n";
    std::cout << "  1. Use adaptive_tile kernel (no packing overhead)\n";
    std::cout << "  2. Autotune blocking for minimum remainder\n";
    std::cout << "  3. Consider tile sizes 4x16 or 6x16 if M divisible\n\n";

    std::cout << "For GEMV shapes (N=1):\n";
    std::cout << "  1. Use tiny kernel (vectorized GEMV)\n";
    std::cout << "  2. Block K dimension for cache efficiency\n";
    std::cout << "  3. Use SVE vector length optimization\n\n";

    std::cout << "For medium irregular shapes:\n";
    std::cout << "  1. Autotune blocking preset per shape\n";
    std::cout << "  2. Consider Moderate or Aggressive preset\n";
    std::cout << "  3. Benchmark different tile combinations\n\n";

    std::cout << "Autotune cache statistics:\n";
    std::cout << "  GEMM kernel cache: " << get_gemm_shape_cache().size() << " entries\n";
    std::cout << "  Blocking cache: " << get_blocking_cache().size() << " entries\n";
    std::cout << "  Tile cache: " << get_tile_cache().size() << " entries\n";

    return 0;
}