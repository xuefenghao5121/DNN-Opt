#pragma once
/// @file cpu_tuning_profile.h
/// Per-CPU tuning profiles for hardware-adaptive GEMM optimization.
///
/// Each ARM CPU family has different optimal cache blocking ratios,
/// prefetch distances, and threading strategies. This system provides
/// a built-in database of tuned profiles with auto-tuning fallback
/// for unknown CPUs.

#include "dnnopt/arm_hwcaps.h"
#include <cstdint>

namespace dnnopt {

/// Matrix shape classification for adaptive blocking.
/// Inspired by autoGEMM (SC'24) dynamic tiling: different shapes
/// need fundamentally different blocking strategies.
enum class ShapeClass {
    kSquare,      // M ~= N ~= K: balanced Mc/Nc/Kc
    kTallSkinny,  // M >> N: maximize Mc, small Nc, thread on M
    kShortWide,   // N >> M: large Nc, small Mc, thread on N
    kSmallGemm,   // M*N*K < threshold: minimal blocking, no threading
    kBertLike,    // M small (1-32), N/K large: small-M optimized path
};

/// Per-datatype tuning hints.
struct DtypeHints {
    int preferred_k_unroll;   // K-loop unroll factor
    bool use_nontemporal_c;   // Use NT stores for large C writeback
};

/// Complete tuning profile for a CPU family.
struct CpuTuningProfile {
    uint32_t part_number;     // ARM part number (0 = generic default)
    const char* name;         // Human-readable name

    // --- Cache utilization ratios ---
    // These replace the hardcoded 40%/40%/30% in compute_blocking_params().
    // Different cores have different optimal ratios due to:
    //   - Cache associativity and replacement policy
    //   - Prefetch hardware aggressiveness
    //   - Number of outstanding loads supported
    float l1d_util;           // Fraction of L1D for GEMM working set (0.30-0.50)
    float l2_util;            // Fraction of L2 for packed A panel (0.30-0.45)
    float l3_util;            // Fraction of L3 for packed B panel (0.20-0.35)

    // --- Blocking bounds ---
    // Upper limits on Mc/Nc to prevent excessive packing overhead.
    // Larger caches allow larger blocks, but diminishing returns above
    // certain sizes due to TLB pressure and packing cost.
    int mc_max;
    int nc_max;

    // --- Prefetch tuning ---
    // Number of K-iterations ahead to prefetch A/B panels.
    // Depends on memory latency and pipeline depth of the core.
    int prefetch_distance;

    // --- Threading ---
    int threading_min_flops;  // Min FLOPS to enable OpenMP (fork/join overhead)
    bool prefer_2d_threading; // Use M×N 2D decomposition instead of M-only

    // --- Per-dtype hints ---
    DtypeHints fp32;
    DtypeHints bf16;
    DtypeHints int8;

    // --- Shape-class adjustments ---
    // Multipliers applied to base utilization ratios per shape class.
    // e.g., tall-skinny needs more L2 for A (bigger Mc), less L3 for B.
    struct ShapeAdjust {
        float l1d_mult;  // Multiplier for l1d_util
        float l2_mult;   // Multiplier for l2_util
        float l3_mult;   // Multiplier for l3_util
        float mc_mult;   // Multiplier for mc_max
        float nc_mult;   // Multiplier for nc_max
    };
    ShapeAdjust shape_adj_tall_skinny;
    ShapeAdjust shape_adj_short_wide;
    ShapeAdjust shape_adj_small;
    ShapeAdjust shape_adj_bert;
};

/// Classify matrix shape for adaptive blocking.
ShapeClass classify_shape(int M, int N, int K);

/// Look up the best tuning profile for the detected hardware.
/// Priority: part_number exact match → implementer family → generic default.
/// Thread-safe, returns a reference to a static profile.
const CpuTuningProfile& lookup_tuning_profile(const ArmHwProfile& hw);

/// Get the tuning profile for the current hardware.
/// Convenience wrapper: calls detect_arm_hwcaps() + lookup_tuning_profile().
const CpuTuningProfile& get_tuning_profile();

}  // namespace dnnopt
