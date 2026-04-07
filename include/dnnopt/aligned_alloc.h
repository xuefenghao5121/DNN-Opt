#pragma once
/// @file aligned_alloc.h
/// Cache-line-aligned memory allocation utilities with huge page support.

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace dnnopt {

/// Default alignment: 64 bytes (cache line on most ARM cores).
constexpr size_t kCacheLineSize = 64;

/// Threshold for huge page allocation (2MB).
constexpr size_t kHugePageThreshold = 2 * 1024 * 1024;

/// Allocate `size` bytes aligned to `alignment`.
/// Returns nullptr on failure. Must be freed with aligned_free().
void* aligned_malloc(size_t size, size_t alignment = kCacheLineSize);

/// Allocate with huge pages for large buffers.
/// Falls back to aligned_malloc if huge pages unavailable.
/// Must be freed with aligned_free_huge().
void* aligned_malloc_huge(size_t size, size_t alignment = kCacheLineSize);

/// Free memory allocated by aligned_malloc().
void aligned_free(void* ptr);

/// Free memory allocated by aligned_malloc_huge().
/// Handles both mmap'd and posix_memalign'd memory.
void aligned_free_huge(void* ptr, size_t size);

/// RAII wrapper for aligned memory.
template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const { aligned_free(ptr); }
};

template<typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

/// Allocate an aligned array of `count` elements of type T.
template<typename T>
AlignedPtr<T> aligned_array(size_t count, size_t alignment = kCacheLineSize) {
    void* p = aligned_malloc(count * sizeof(T), alignment);
    return AlignedPtr<T>(static_cast<T*>(p));
}

/// RAII wrapper for huge-page memory (carries size for munmap).
template<typename T>
struct HugePageDeleter {
    size_t size_bytes = 0;
    void operator()(T* ptr) const { aligned_free_huge(ptr, size_bytes); }
};

template<typename T>
using HugePagePtr = std::unique_ptr<T[], HugePageDeleter<T>>;

/// Allocate a huge-page array of `count` elements of type T.
template<typename T>
HugePagePtr<T> aligned_array_huge(size_t count, size_t alignment = kCacheLineSize) {
    size_t sz = count * sizeof(T);
    void* p = aligned_malloc_huge(sz, alignment);
    HugePageDeleter<T> d;
    d.size_bytes = sz;
    return HugePagePtr<T>(static_cast<T*>(p), d);
}

}  // namespace dnnopt
