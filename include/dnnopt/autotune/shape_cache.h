#pragma once
/// @file shape_cache.h
/// Shape-based kernel selection cache for runtime autotuning.
///
/// Provides:
///   - ShapeKey: 64-bit hash of matrix/convolution shape
///   - KernelSelection: optimal kernel + measured performance
///   - ShapeCache: LRU cache with 256 entries + file persistence
///
/// This is the core component for avoiding repeated micro-benchmarks
/// on known shapes during inference.

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <list>

namespace dnnopt {

// ============================================================
// Shape Keys
// ============================================================

/// GEMM shape key (48 bits packed into 64-bit)
/// Supports M/N/K up to 65535, dtype + algo hint
struct GemmShapeKey {
    uint64_t hash() const;

    uint16_t M;
    uint16_t N;
    uint16_t K;
    uint8_t  dtype;    // 0=FP32, 1=BF16, 2=INT8
    uint8_t  algo;     // 0=auto, reserved for future
};

/// Conv2D shape key (56 bits packed into 64-bit)
struct ConvShapeKey {
    uint64_t hash() const;

    uint16_t batch;    // N (up to 65535)
    uint16_t IH;       // Input height
    uint16_t IW;       // Input width
    uint8_t  IC;       // Input channels (≤255)
    uint8_t  OC;       // Output channels (≤255)
    uint8_t  KH;       // Kernel height
    uint8_t  KW;       // Kernel width
    uint8_t  stride;   // stride (combined h+w into single byte)
    uint8_t  pad;      // padding (combined h+w into single byte)
    uint8_t  groups;   // 0=ungrouped, IC=depthwise
};

// ============================================================
// Kernel Selection Result
// ============================================================

/// Kernel ID for GEMM
enum class GemmKernelId : uint8_t {
    kSmallM       = 0,   // M<8, no packing (gemm_smallm_driver)
    kSmallMWide   = 1,   // M<8, N>=48 (gemm_smallm_wide_driver)
    kAdaptiveTile = 2,   // M=4-32, unpacked (gemm_adaptive_tile)
    kPacked       = 3,   // M>=32, packing + threading (registry)
    kTiny         = 4,   // N=1 or M=1 (gemm_tiny_dispatch)
};

/// Kernel ID for Conv2D
enum class ConvKernelId : uint8_t {
    kIm2colGemm   = 0,   // General im2col + GEMM
    kWinograd     = 1,   // F(2x2, 3x3) for 3x3 stride=1
    kDepthwise    = 2,   // groups=IC (MobileNet)
    kGrouped      = 3,   // groups>1 (ResNeXt)
    k1x1Direct    = 4,   // 1x1 stride=1 pad=0 (direct GEMM)
};

/// Kernel selection result with performance data
struct KernelSelection {
    uint8_t  kernel_id;     // GemmKernelId or ConvKernelId
    float    gflops;        // Measured performance (GFLOPS)
    uint32_t time_us;       // Benchmark time (microseconds)
    bool     valid;         // True if benchmark was run
};

// ============================================================
// Shape Cache (LRU, 256 entries)
// ============================================================

/// LRU cache for shape-based kernel selection.
/// Thread-safe for lookup, insert requires external synchronization.
class ShapeCache {
public:
    static constexpr int kMaxEntries = 256;

    ShapeCache();

    /// Lookup cached kernel selection.
    /// Returns nullptr if not found.
    const KernelSelection* lookup(uint64_t key) const;

    /// Insert new kernel selection.
    /// Evicts oldest entry if cache is full.
    void insert(uint64_t key, const KernelSelection& sel);

    /// Load cache from binary file.
    /// Returns number of entries loaded, -1 on error.
    int load_from_file(const char* path);

    /// Save cache to binary file.
    /// Returns 0 on success, -1 on error.
    int save_to_file(const char* path) const;

    /// Clear all entries.
    void clear();

    /// Get number of cached entries.
    size_t size() const;

private:
    std::unordered_map<uint64_t, KernelSelection> cache_;
    std::list<uint64_t> lru_order_;  // Front = newest, Back = oldest

    void evict_oldest();
};

// ============================================================
// Global Cache Instance
// ============================================================

/// Global GEMM shape cache singleton.
ShapeCache& get_gemm_shape_cache();

/// Global Conv2D shape cache singleton.
ShapeCache& get_conv_shape_cache();

/// Clear all caches.
void clear_all_shape_caches();

}  // namespace dnnopt