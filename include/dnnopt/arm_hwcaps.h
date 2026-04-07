#pragma once
/// @file arm_hwcaps.h
/// ARM hardware capability detection for AArch64.
/// Detects NEON, SVE/SVE2, BF16, I8MM, SME, DotProd and cache hierarchy.

#include <cstdint>
#include <string>
#include <vector>

namespace dnnopt {

/// Bit flags for ARM hardware capabilities.
enum HwCap : uint64_t {
    kNone       = 0,
    kNEON       = 1ULL << 0,   // ASIMD (always on AArch64)
    kFP16       = 1ULL << 1,   // Half-precision float
    kDotProd    = 1ULL << 2,   // SDOT/UDOT (ARMv8.2 DotProd)
    kSVE        = 1ULL << 3,   // Scalable Vector Extension
    kSVE2       = 1ULL << 4,   // SVE2 (ARMv9)
    kBF16       = 1ULL << 5,   // BFloat16 (BFMMLA, BFDOT)
    kI8MM       = 1ULL << 6,   // Int8 Matrix Multiply (SMMLA/UMMLA)
    kSME        = 1ULL << 7,   // Scalable Matrix Extension
    kSME2       = 1ULL << 8,   // SME2
    kSVEBF16    = 1ULL << 9,   // SVE BF16 instructions
    kSVEI8MM    = 1ULL << 10,  // SVE I8MM instructions
    kFRINT      = 1ULL << 11,  // FRINT (float round) instructions
    kAES        = 1ULL << 12,  // AES crypto extension
    kSHA256     = 1ULL << 13,  // SHA-256 extension
    kAtomics    = 1ULL << 14,  // LSE atomics (ARMv8.1)
};

/// Cache level descriptor.
struct CacheInfo {
    uint32_t size_bytes = 0;   // Total size in bytes
    uint32_t line_size  = 0;   // Cache line size in bytes
    uint32_t ways       = 0;   // Associativity
    uint32_t sets       = 0;   // Number of sets
};

/// Core cluster descriptor for big.LITTLE topology.
struct CoreCluster {
    uint32_t first_cpu = 0;       // First CPU index in cluster
    uint32_t count = 0;           // Number of CPUs in cluster
    uint32_t max_freq_khz = 0;    // Max frequency from cpuinfo_max_freq
    bool is_big = true;           // Performance core (true) vs efficiency (false)
};

/// CPU core topology (big.LITTLE awareness).
struct CoreTopology {
    std::vector<CoreCluster> clusters;
    uint32_t big_cores = 0;       // Total performance cores
    uint32_t little_cores = 0;    // Total efficiency cores
    bool is_heterogeneous = false; // True if big.LITTLE detected
};

/// Complete hardware profile for the current ARM CPU.
struct ArmHwProfile {
    // CPU identification
    uint32_t    implementer = 0;   // CPU implementer (0x41 = ARM)
    uint32_t    part_number = 0;   // CPU part number
    uint32_t    variant     = 0;   // CPU variant
    uint32_t    revision    = 0;   // CPU revision
    std::string cpu_name;          // Human-readable name (e.g. "Neoverse N2")
    uint32_t    num_cores   = 0;   // Number of online cores
    uint32_t    freq_mhz    = 0;   // CPU frequency in MHz (0 if unknown)

    // Capability flags
    uint64_t    hwcaps = kNone;

    // SVE vector length (bits), 0 if SVE not supported
    uint32_t    sve_vector_bits = 0;

    // Cache hierarchy
    CacheInfo   l1d;   // L1 Data
    CacheInfo   l1i;   // L1 Instruction
    CacheInfo   l2;    // L2 Unified
    CacheInfo   l3;    // L3 Unified (may be 0 if not present)

    // Core topology (big.LITTLE awareness)
    CoreTopology topology;

    // Theoretical peak performance (single core)
    double      fp32_gflops_per_core = 0.0;
    double      bf16_gflops_per_core = 0.0;
    double      int8_gops_per_core   = 0.0;

    // Helpers
    bool has(HwCap cap) const { return (hwcaps & cap) != 0; }
};

/// Detect hardware capabilities of the current ARM CPU.
/// Results are cached after first call.
const ArmHwProfile& detect_arm_hwcaps();

/// Print a formatted summary of hardware capabilities to stdout.
void print_hwcaps_summary(const ArmHwProfile& profile);

/// Get a short string tag for the current platform (e.g. "neoverse_n2_sve2_256")
std::string platform_tag(const ArmHwProfile& profile);

}  // namespace dnnopt
