/// @file cpu_tuning_profiles.cpp
/// Built-in tuning profile database for ARM CPU families.
///
/// Profiles are derived from:
///   - Cache hierarchy specifications (ARM TRMs)
///   - Empirical GEMM tuning on representative hardware
///   - autoGEMM (SC'24) insights on shape-dependent blocking
///
/// For unknown CPUs, returns a conservative generic profile that
/// works reasonably across all ARM cores.

#include "dnnopt/cpu_tuning_profile.h"
#include <algorithm>
#include <cstdint>

namespace dnnopt {

// ============================================================
// Shape classification
// ============================================================

ShapeClass classify_shape(int M, int N, int K) {
    int64_t vol = (int64_t)M * N * K;

    // Small GEMM: total work < 64K multiply-adds
    if (vol < 65536) return ShapeClass::kSmallGemm;

    // BERT-like: M is very small, N/K are large (batch-1 inference)
    if (M <= 32 && N >= 256 && K >= 256) return ShapeClass::kBertLike;

    // Aspect ratio thresholds
    float mn_ratio = (float)M / std::max(N, 1);
    float nm_ratio = (float)N / std::max(M, 1);

    // Tall-skinny: M >> N (e.g., M=4096, N=64)
    if (mn_ratio > 4.0f) return ShapeClass::kTallSkinny;

    // Short-wide: N >> M (e.g., M=64, N=4096)
    if (nm_ratio > 4.0f) return ShapeClass::kShortWide;

    return ShapeClass::kSquare;
}

// ============================================================
// Default shape adjustments (used by most profiles)
// ============================================================

static constexpr CpuTuningProfile::ShapeAdjust kDefaultTallSkinny = {
    1.0f,   // l1d_mult: keep L1 ratio
    1.2f,   // l2_mult: more L2 for bigger A panels
    0.7f,   // l3_mult: less L3 needed (small N)
    1.5f,   // mc_mult: allow larger Mc
    0.5f,   // nc_mult: reduce Nc
};

static constexpr CpuTuningProfile::ShapeAdjust kDefaultShortWide = {
    1.0f,   // l1d_mult
    0.7f,   // l2_mult: less L2 (small M, small A)
    1.2f,   // l3_mult: more L3 for bigger B panels
    0.5f,   // mc_mult: reduce Mc
    1.5f,   // nc_mult: allow larger Nc
};

static constexpr CpuTuningProfile::ShapeAdjust kDefaultSmall = {
    0.6f,   // l1d_mult: smaller blocks reduce packing overhead
    0.5f,   // l2_mult
    0.5f,   // l3_mult
    0.25f,  // mc_mult
    0.25f,  // nc_mult
};

static constexpr CpuTuningProfile::ShapeAdjust kDefaultBert = {
    1.0f,   // l1d_mult
    0.5f,   // l2_mult: small M → small A panels
    1.0f,   // l3_mult: keep B panel large
    0.25f,  // mc_mult: small Mc (M is tiny)
    1.2f,   // nc_mult: maximize Nc
};

// ============================================================
// Built-in profiles
// ============================================================

// Neoverse N1 (Graviton2, Ampere Altra)
// L1D=64KB 4-way, L2=1MB 8-way, L3 varies (shared)
// 2x FMLA pipes, 128-bit NEON only (no SVE on most N1 implementations)
static const CpuTuningProfile kProfileN1 = {
    0xd0c, "Neoverse N1",
    0.35f, 0.40f, 0.30f,       // cache util
    256, 4096,                  // mc_max, nc_max
    8,                          // prefetch distance
    200000, false,              // threading
    {4, false}, {4, false}, {4, false},  // dtype hints
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Neoverse N2 (Graviton3E, our dev environment)
// L1D=64KB 4-way, L2=1MB 8-way, L3=64MB 16-way
// 2x FMLA, SVE2 128-bit, BF16, I8MM
static const CpuTuningProfile kProfileN2 = {
    0xd49, "Neoverse N2",
    0.40f, 0.40f, 0.30f,       // cache util
    512, 8192,                  // mc_max, nc_max
    8,                          // prefetch distance
    200000, false,              // threading
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Neoverse V1 (Graviton3)
// L1D=64KB 4-way, L2=1MB 8-way, L3 shared (per-cluster)
// 4x 128-bit SVE pipes (effectively 256-bit SVE throughput)
// Aggressive prefetcher → lower cache util ratios to avoid thrashing
static const CpuTuningProfile kProfileV1 = {
    0xd40, "Neoverse V1",
    0.35f, 0.35f, 0.25f,       // cache util (aggressive HW prefetch)
    512, 4096,                  // mc_max, nc_max
    12,                         // prefetch distance (deeper pipeline)
    150000, true,               // threading: prefer 2D for wide SVE
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Neoverse V2
// L1D=64KB 4-way, L2=1MB 8-way, L3 shared
// SVE2 128-bit, BF16, I8MM, SME (optional)
// Very aggressive prefetcher
static const CpuTuningProfile kProfileV2 = {
    0xd4f, "Neoverse V2",
    0.35f, 0.35f, 0.25f,
    512, 4096,
    12,
    150000, true,
    {4, true}, {4, true}, {4, true},    // NT stores beneficial
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Cortex-A78 (mobile/laptop)
// L1D=64KB 4-way, L2=256KB-512KB 8-way, L3 varies
// Smaller caches → tighter blocking
static const CpuTuningProfile kProfileA78 = {
    0xd41, "Cortex-A78",
    0.45f, 0.40f, 0.30f,       // higher L1 util (simpler prefetch)
    128, 2048,                  // smaller bounds for smaller caches
    6,                          // shorter prefetch
    300000, false,              // higher threading threshold (mobile power)
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Cortex-X2 (high-perf mobile)
// L1D=64KB, L2=512KB-1MB, L3 shared
static const CpuTuningProfile kProfileX2 = {
    0xd48, "Cortex-X2",
    0.40f, 0.35f, 0.25f,
    256, 4096,
    10,
    200000, false,
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Cortex-X3
static const CpuTuningProfile kProfileX3 = {
    0xd4e, "Cortex-X3",
    0.40f, 0.35f, 0.25f,
    256, 4096,
    10,
    200000, true,
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Cortex-A55 (efficiency core)
// L1D=32KB, L2=128-256KB, no L3 typically
// Very constrained → tight blocking
static const CpuTuningProfile kProfileA55 = {
    0xd05, "Cortex-A55",
    0.50f, 0.45f, 0.30f,       // high util (no HW prefetch competition)
    64, 1024,                   // tiny bounds
    4,                          // short prefetch
    500000, false,              // high threshold (slow core)
    {2, false}, {2, false}, {2, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Cortex-A510 (ARMv9 efficiency core)
static const CpuTuningProfile kProfileA510 = {
    0xd46, "Cortex-A510",
    0.45f, 0.40f, 0.30f,
    96, 1536,
    4,
    400000, false,
    {2, false}, {2, false}, {2, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Fujitsu A64FX (SVE 512-bit, HBM2)
// L1D=64KB, L2=8MB (per CMG), no L3
// Very wide SVE → 2D threading critical
static const CpuTuningProfile kProfileA64FX = {
    0x001, "A64FX",
    0.35f, 0.30f, 0.25f,       // conservative (HBM bandwidth-limited)
    512, 8192,
    16,                         // long prefetch (HBM latency)
    100000, true,               // prefer 2D threading
    {8, true}, {8, true}, {8, true},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// HiSilicon Kunpeng 920 (TSV110, Neoverse N1 derivative)
static const CpuTuningProfile kProfileKunpeng920 = {
    0xd01, "Kunpeng 920 (TSV110)",
    0.35f, 0.40f, 0.30f,
    256, 4096,
    8,
    200000, false,
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// Generic default profile: conservative parameters that work
// reasonably on any ARM core. Used as fallback when CPU is unknown.
static const CpuTuningProfile kProfileGenericDefault = {
    0, "Generic ARM (default)",
    0.35f, 0.35f, 0.25f,
    256, 4096,
    8,
    200000, false,
    {4, false}, {4, false}, {4, false},
    kDefaultTallSkinny, kDefaultShortWide, kDefaultSmall, kDefaultBert,
};

// ============================================================
// Profile lookup table
// ============================================================

static const CpuTuningProfile* kProfileTable[] = {
    &kProfileN1,
    &kProfileN2,
    &kProfileV1,
    &kProfileV2,
    &kProfileA78,
    &kProfileX2,
    &kProfileX3,
    &kProfileA55,
    &kProfileA510,
    &kProfileA64FX,
    &kProfileKunpeng920,
};

static const int kProfileTableSize =
    sizeof(kProfileTable) / sizeof(kProfileTable[0]);

const CpuTuningProfile& lookup_tuning_profile(const ArmHwProfile& hw) {
    // 1. Exact part_number match
    for (int i = 0; i < kProfileTableSize; ++i) {
        if (kProfileTable[i]->part_number == hw.part_number)
            return *kProfileTable[i];
    }

    // 2. Implementer-family fallback
    // ARM Ltd (0x41): use generic default with small adjustments
    // HiSilicon (0x48): use Kunpeng profile
    // Fujitsu (0x46): use A64FX profile
    if (hw.implementer == 0x48) return kProfileKunpeng920;
    if (hw.implementer == 0x46) return kProfileA64FX;

    // 3. Generic default
    return kProfileGenericDefault;
}

const CpuTuningProfile& get_tuning_profile() {
    const auto& hw = detect_arm_hwcaps();
    return lookup_tuning_profile(hw);
}

}  // namespace dnnopt
