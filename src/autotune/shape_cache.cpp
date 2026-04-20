/// @file shape_cache.cpp
/// Implementation of ShapeKey hash and ShapeCache LRU.

#include "dnnopt/autotune/shape_cache.h"

#include <fstream>
#include <cstring>

namespace dnnopt {

// ============================================================
// ShapeKey Hash Functions
// ============================================================

uint64_t GemmShapeKey::hash() const {
    // FNV-1a hash variant for packed struct
    uint64_t h = 14695981039346656037ULL;  // FNV offset basis
    const uint8_t* data = reinterpret_cast<const uint8_t*>(this);
    for (size_t i = 0; i < sizeof(GemmShapeKey); ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;  // FNV prime
    }
    return h;
}

uint64_t ConvShapeKey::hash() const {
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(this);
    for (size_t i = 0; i < sizeof(ConvShapeKey); ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ============================================================
// ShapeCache Implementation
// ============================================================

ShapeCache::ShapeCache() {}

const KernelSelection* ShapeCache::lookup(uint64_t key) const {
    auto it = cache_.find(key);
    if (it == cache_.end()) return nullptr;
    return &it->second;
}

void ShapeCache::insert(uint64_t key, const KernelSelection& sel) {
    // If already exists, update and move to front
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second = sel;
        // Move to front of LRU
        lru_order_.remove(key);
        lru_order_.push_front(key);
        return;
    }

    // Evict if full
    while (cache_.size() >= kMaxEntries) {
        evict_oldest();
    }

    // Insert new entry
    cache_[key] = sel;
    lru_order_.push_front(key);
}

void ShapeCache::evict_oldest() {
    if (lru_order_.empty()) return;
    uint64_t oldest_key = lru_order_.back();
    lru_order_.pop_back();
    cache_.erase(oldest_key);
}

void ShapeCache::clear() {
    cache_.clear();
    lru_order_.clear();
}

size_t ShapeCache::size() const {
    return cache_.size();
}

// ============================================================
// File Persistence
// ============================================================

// File format:
//   Header: "DNNAUTO" (8 bytes) + version (4 bytes) + num_entries (4 bytes)
//   Entries: shape_key (8 bytes) + kernel_id (1 byte) + gflops (4 bytes) + time_us (4 bytes)

static const char kMagic[] = "DNNAUTO";
static const uint32_t kVersion = 1;

int ShapeCache::load_from_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return -1;

    // Read header
    char magic[8];
    uint32_t version, num_entries;
    file.read(magic, 8);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&num_entries), 4);

    if (std::memcmp(magic, kMagic, 8) != 0 || version != kVersion) {
        return -1;
    }

    // Read entries
    clear();
    for (uint32_t i = 0; i < num_entries && i < kMaxEntries; ++i) {
        uint64_t key;
        uint8_t kernel_id;
        float gflops;
        uint32_t time_us;

        file.read(reinterpret_cast<char*>(&key), 8);
        file.read(reinterpret_cast<char*>(&kernel_id), 1);
        file.read(reinterpret_cast<char*>(&gflops), 4);
        file.read(reinterpret_cast<char*>(&time_us), 4);

        if (!file) break;

        KernelSelection sel;
        sel.kernel_id = kernel_id;
        sel.gflops = gflops;
        sel.time_us = time_us;
        sel.valid = true;
        insert(key, sel);
    }

    return static_cast<int>(cache_.size());
}

int ShapeCache::save_to_file(const char* path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return -1;

    // Write header
    uint32_t num_entries = static_cast<uint32_t>(cache_.size());
    file.write(kMagic, 8);
    file.write(reinterpret_cast<const char*>(&kVersion), 4);
    file.write(reinterpret_cast<const char*>(&num_entries), 4);

    // Write entries (in LRU order, newest first)
    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        uint64_t key = *it;
        const auto& sel = cache_.at(key);
        file.write(reinterpret_cast<const char*>(&key), 8);
        file.write(reinterpret_cast<const char*>(&sel.kernel_id), 1);
        file.write(reinterpret_cast<const char*>(&sel.gflops), 4);
        file.write(reinterpret_cast<const char*>(&sel.time_us), 4);
    }

    return file ? 0 : -1;
}

// ============================================================
// Global Cache Instances
// ============================================================

static ShapeCache g_gemm_cache;
static ShapeCache g_conv_cache;
static BlockingCache g_blocking_cache;
static TileCache g_tile_cache;
static ThresholdCache g_threshold_cache;

ShapeCache& get_gemm_shape_cache() {
    return g_gemm_cache;
}

ShapeCache& get_conv_shape_cache() {
    return g_conv_cache;
}

BlockingCache& get_blocking_cache() {
    return g_blocking_cache;
}

TileCache& get_tile_cache() {
    return g_tile_cache;
}

ThresholdCache& get_threshold_cache() {
    return g_threshold_cache;
}

void clear_all_shape_caches() {
    g_gemm_cache.clear();
    g_conv_cache.clear();
    g_blocking_cache.clear();
    g_tile_cache.clear();
    g_threshold_cache.clear();
}

// ============================================================
// Blocking/Tile Helper Functions (v2 Autotune)
// ============================================================

BlockingParams get_blocking_params_from_preset(BlockingPreset preset) {
    BlockingParams p;
    switch (preset) {
    case BlockingPreset::kConservative:
        p.l1d_util = 0.30f; p.l2_util = 0.30f; p.l3_util = 0.20f;
        break;
    case BlockingPreset::kStandard:
        p.l1d_util = 0.35f; p.l2_util = 0.35f; p.l3_util = 0.25f;
        break;
    case BlockingPreset::kModerate:
        p.l1d_util = 0.40f; p.l2_util = 0.40f; p.l3_util = 0.30f;
        break;
    case BlockingPreset::kAggressive:
        p.l1d_util = 0.45f; p.l2_util = 0.45f; p.l3_util = 0.35f;
        break;
    case BlockingPreset::kMaximum:
        p.l1d_util = 0.50f; p.l2_util = 0.50f; p.l3_util = 0.40f;
        break;
    default:
        p.l1d_util = 0.40f; p.l2_util = 0.40f; p.l3_util = 0.30f;  // Moderate default
        break;
    }
    return p;
}

void get_tile_params_from_preset(TilePreset preset, int& Mr, int& Nr) {
    switch (preset) {
    case TilePreset::k4x16:  Mr = 4;  Nr = 16; break;
    case TilePreset::k6x16:  Mr = 6;  Nr = 16; break;
    case TilePreset::k8x12:  Mr = 8;  Nr = 12; break;
    case TilePreset::k8x16:  Mr = 8;  Nr = 16; break;
    default:                 Mr = 8;  Nr = 12; break;  // Standard FP32
    }
}

// ============================================================
// BlockingCache Implementation (v2 Autotune)
// ============================================================

BlockingCache::BlockingCache() {}

const BlockingSelection* BlockingCache::lookup(uint64_t key) const {
    auto it = cache_.find(key);
    if (it == cache_.end()) return nullptr;
    return &it->second;
}

void BlockingCache::insert(uint64_t key, const BlockingSelection& sel) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second = sel;
        lru_order_.remove(key);
        lru_order_.push_front(key);
        return;
    }

    while (cache_.size() >= kMaxEntries) {
        evict_oldest();
    }

    cache_[key] = sel;
    lru_order_.push_front(key);
}

void BlockingCache::evict_oldest() {
    if (lru_order_.empty()) return;
    uint64_t oldest_key = lru_order_.back();
    lru_order_.pop_back();
    cache_.erase(oldest_key);
}

void BlockingCache::clear() {
    cache_.clear();
    lru_order_.clear();
}

size_t BlockingCache::size() const {
    return cache_.size();
}

// ============================================================
// TileCache Implementation (v2 Autotune)
// ============================================================

TileCache::TileCache() {}

const TileSelection* TileCache::lookup(uint64_t key) const {
    auto it = cache_.find(key);
    if (it == cache_.end()) return nullptr;
    return &it->second;
}

void TileCache::insert(uint64_t key, const TileSelection& sel) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second = sel;
        lru_order_.remove(key);
        lru_order_.push_front(key);
        return;
    }

    while (cache_.size() >= kMaxEntries) {
        evict_oldest();
    }

    cache_[key] = sel;
    lru_order_.push_front(key);
}

void TileCache::evict_oldest() {
    if (lru_order_.empty()) return;
    uint64_t oldest_key = lru_order_.back();
    lru_order_.pop_back();
    cache_.erase(oldest_key);
}

void TileCache::clear() {
    cache_.clear();
    lru_order_.clear();
}

size_t TileCache::size() const {
    return cache_.size();
}

// ============================================================
// ThresholdCache Implementation (v2 Autotune)
// ============================================================

ThresholdCache::ThresholdCache() : valid_(false) {
    sel_.small_m_bound = 8;
    sel_.wide_n_bound = 48;
    sel_.unpacked_thresh = 4 * 1024 * 1024;
}

const ThresholdSelection* ThresholdCache::get() const {
    if (valid_) return &sel_;
    return nullptr;
}

void ThresholdCache::set(const ThresholdSelection& sel) {
    sel_ = sel;
    valid_ = true;
}

void ThresholdCache::clear() {
    valid_ = false;
}

bool ThresholdCache::valid() const {
    return valid_;
}

}  // namespace dnnopt